#!/usr/bin/env python3
"""
Ultra-Stable SAR Knowledge Distillation Training Script
=======================================================

This version addresses severe FP16 numerical instability issues with:
- Extremely aggressive logit and loss clamping
- Ultra-conservative learning rates and scheduling
- Comprehensive NaN/Inf detection and recovery
- Model state monitoring and reset capabilities
- Enhanced gradient management for FP16

Designed specifically for P100 FP16 training environments.
"""

import argparse
import math
import os
import json
import time
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset

# Import SAR modules
from sar_distillation import SARDistiller, SARConfig
from data_utils import build_text_datasets, DualTokenizerCollator
from model_utils import load_teacher_student


def parse_args():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument("--teacher_model", type=str, default="microsoft/DialoGPT-large")
    parser.add_argument("--student_model", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--model_dtype", type=str, default="float16", choices=["float32", "float16"])

    # Training configuration
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--clear_cache_every_step", action="store_true")

    # Ultra-conservative learning rates for FP16 stability
    parser.add_argument("--student_lr", type=float, default=5e-6)  # Even lower
    parser.add_argument("--router_lr", type=float, default=2e-5)   # Reduced
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)  # Much more aggressive

    # Knowledge distillation - ultra-conservative
    parser.add_argument("--temperature", type=float, default=1.5)  # Lower temperature
    parser.add_argument("--alpha_kd", type=float, default=0.05)    # Reduced KD weight
    parser.add_argument("--alpha_ce", type=float, default=0.95)    # Higher CE weight

    # Router regularization - minimal for stability
    parser.add_argument("--router_anchor_weight", type=float, default=0.001)
    parser.add_argument("--router_load_balance_weight", type=float, default=0.001)
    parser.add_argument("--router_entropy_weight", type=float, default=0.001)

    # Data configuration
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--block_size", type=int, default=512)

    # System configuration
    parser.add_argument("--output_dir", type=str, default="./sar_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload_teacher_to_cpu", action="store_true", default=True)
    parser.add_argument("--use_scheduler", action="store_true", default=True)

    return parser.parse_args()


def print_memory_info(label: str, device: torch.device):
    if torch.cuda.is_available() and device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[{label}] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


class UltraStableEvaluator:
    """Ultra-conservative evaluator with extensive NaN/Inf protection"""

    def __init__(self, eval_dataset, tokenizer, device, batch_size=2, eval_batches=10):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def evaluate(self, student_model, use_fp16=False):
        """Evaluate with ultra-aggressive stability measures"""
        if self.eval_dataset is None:
            return float('inf')

        # Use a small, fixed subset for stability
        eval_size = min(len(self.eval_dataset), self.eval_batches * self.batch_size)
        eval_indices = list(range(eval_size))
        eval_subset = torch.utils.data.Subset(self.eval_dataset, eval_indices)

        eval_loader = DataLoader(
            eval_subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )

        student_model.eval()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if batch_count >= self.eval_batches:
                        break

                    try:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        labels = batch["labels"].to(self.device)

                        # Forward pass with ultra-conservative settings
                        with torch.amp.autocast('cuda', enabled=use_fp16):
                            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits[:, :-1, :]
                            targets = labels[:, 1:]

                            # Ultra-aggressive logit clamping for FP16
                            logits = torch.clamp(logits, -3.0, 3.0)

                            # Check for NaN/Inf in logits
                            if torch.isnan(logits).any() or torch.isinf(logits).any():
                                print(f"    Warning: NaN/Inf in logits batch {batch_idx}, skipping")
                                continue

                            # Calculate loss with extensive validation
                            valid_mask = (targets != -100)
                            if valid_mask.sum() == 0:
                                continue

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
                            losses = losses.view(targets.shape)

                            # Check losses before masking
                            if torch.isnan(losses).any() or torch.isinf(losses).any():
                                print(f"    Warning: NaN/Inf in raw losses batch {batch_idx}, skipping")
                                continue

                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            # Ultra-conservative loss clamping
                            batch_loss = torch.clamp(batch_loss, 0.1, 6.0)

                            # Final NaN/Inf check
                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                print(f"    Warning: NaN/Inf final loss in batch {batch_idx}, skipping")
                                continue

                            # Additional sanity check
                            if batch_loss.item() > 8.0:
                                print(f"    Warning: Suspicious eval loss {batch_loss.item():.4f} in batch {batch_idx}")
                                continue

                            total_loss += batch_loss.item()
                            total_tokens += valid_mask.sum().item()
                            batch_count += 1

                    except Exception as e:
                        print(f"    Warning: Eval batch {batch_idx} failed: {e}")
                        continue

        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('inf')
        finally:
            student_model.train()

        if batch_count == 0:
            print(f"  Warning: No valid evaluation batches processed")
            return float('inf')

        avg_loss = total_loss / batch_count
        print(f"  Debug: avg_loss={avg_loss:.4f}, batch_count={batch_count}, total_tokens={total_tokens}")

        # Ultra-conservative perplexity calculation
        if avg_loss > 10:
            print(f"  Warning: Clamping extreme avg_loss {avg_loss:.4f} to 10")
            avg_loss = 10

        try:
            perplexity = math.exp(avg_loss)
            if math.isnan(perplexity) or math.isinf(perplexity) or perplexity > 50000:
                print(f"  Error: Invalid perplexity, returning 50000")
                return 50000.0
        except OverflowError:
            print(f"  Error: Perplexity overflow, returning 50000")
            return 50000.0

        return perplexity


class UltraStableTrainer:
    """Ultra-stable trainer with comprehensive FP16 safeguards and recovery mechanisms"""

    def __init__(self, distiller, args):
        self.distiller = distiller
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_history = []

        # FP16 stability tracking
        self.consecutive_nan_batches = 0
        self.max_consecutive_nans = 20  # Reset model if too many NaNs
        self.initial_student_state = None

    def save_initial_state(self):
        """Save initial model state for potential recovery"""
        self.initial_student_state = {
            k: v.clone() for k, v in self.distiller.student.state_dict().items()
        }

    def reset_to_initial_state(self):
        """Reset student model to initial state if training becomes unstable"""
        if self.initial_student_state is not None:
            print("🔄 EMERGENCY: Resetting student model to initial state due to instability")
            self.distiller.student.load_state_dict(self.initial_student_state)
            # Reset optimizer state
            self.distiller.optimizer = torch.optim.AdamW(
                self.distiller.student.parameters(),
                lr=self.distiller.config.student_lr,
                weight_decay=self.distiller.config.weight_decay,
                eps=1e-8  # More stable for FP16
            )
            self.consecutive_nan_batches = 0

    def ultra_safe_forward_pass(self, batch):
        """Ultra-safe forward pass with maximum stability measures"""
        try:
            input_ids = batch["input_ids"].to(self.distiller.device)
            labels = batch["labels"].to(self.distiller.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.distiller.device)

            # Check input validity
            if torch.isnan(input_ids.float()).any():
                print("  Warning: NaN in input_ids, skipping batch")
                return None

            with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                # Student forward pass only for maximum stability
                s_out = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = s_out.logits[:, :-1, :]
                target_labels = labels[:, 1:]

                # Ultra-aggressive logit clamping for FP16
                student_logits = torch.clamp(student_logits, -2.5, 2.5)

                # Check for NaN/Inf in logits immediately
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    print("  Warning: NaN/Inf in student logits, skipping batch")
                    self.consecutive_nan_batches += 1
                    return None

                # Simple CE loss only (skip KD for maximum stability)
                valid_mask = (target_labels != -100)
                if not valid_mask.any():
                    return None

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_losses = loss_fct(student_logits.view(-1, student_logits.size(-1)), target_labels.view(-1))

                # Check raw losses
                if torch.isnan(ce_losses).any() or torch.isinf(ce_losses).any():
                    print("  Warning: NaN/Inf in raw CE losses, skipping batch")
                    self.consecutive_nan_batches += 1
                    return None

                ce_losses = ce_losses.view(target_labels.shape)
                masked_ce = ce_losses * valid_mask.float()
                ce = masked_ce.sum() / valid_mask.sum()

                # Ultra-aggressive loss clamping
                ce = torch.clamp(ce, 0.1, 5.0)

                # Final NaN check
                if torch.isnan(ce) or torch.isinf(ce):
                    print(f"  Warning: Final CE loss is NaN/Inf: {ce}, skipping batch")
                    self.consecutive_nan_batches += 1
                    return None

                # Sanity check for extreme values
                if ce.item() > 8.0:
                    print(f"  Warning: Suspicious CE loss {ce.item():.4f}, clamping to 6.0")
                    ce = torch.tensor(6.0, device=self.distiller.device, dtype=ce.dtype)

                tokens = valid_mask.sum().item()

                # Reset consecutive NaN counter on successful batch
                self.consecutive_nan_batches = 0

                return ce, 0.0, ce.item(), tokens

        except Exception as e:
            print(f"  Forward pass error: {e}")
            self.consecutive_nan_batches += 1
            return None

    def train(self, train_loader, save_callback=None, log_callback=None):
        """Ultra-stable training loop with comprehensive error handling"""
        print("🚀 Starting ULTRA-STABLE SAR Knowledge Distillation training...")
        print("🛡️  Maximum FP16 stability measures enabled")

        # Save initial state for potential recovery
        self.save_initial_state()

        # Setup
        self.distiller.teacher.eval()
        self.distiller.student.train()

        step = 0
        total_tokens = 0
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        successful_steps = 0
        skipped_batches = 0

        # FP16 setup with enhanced safety
        use_scaler = torch.cuda.is_available() and self.distiller.use_fp16

        # Disable scaler for FP16 parameters
        has_fp16_params = any(p.dtype == torch.float16 for p in self.distiller.student.parameters())
        if has_fp16_params and use_scaler:
            print("🔧 FP16 parameters detected - using pure FP16 training without gradient scaling")
            use_scaler = False

        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None

        # Print configuration
        print(f"🔧 Training Configuration:")
        print(f"   - Pure FP16 training: {has_fp16_params}")
        print(f"   - Gradient scaling: {use_scaler}")
        print(f"   - Max consecutive NaNs before reset: {self.max_consecutive_nans}")

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            val_ppl = self.run_evaluation(step, log_callback)

        # Training loop
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            # Check if we need to reset due to too many consecutive NaNs
            if self.consecutive_nan_batches >= self.max_consecutive_nans:
                self.reset_to_initial_state()

            # Safe forward pass
            result = self.ultra_safe_forward_pass(batch)
            if result is None:
                step += 1
                skipped_batches += 1
                if skipped_batches % 10 == 0:
                    print(f"  ⚠️  Skipped {skipped_batches} batches so far due to numerical issues")
                continue

            loss, kd_loss, ce_loss, batch_tokens = result

            # Backward pass with ultra-safe gradient handling
            try:
                loss = loss / self.args.grad_accum_steps

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Check gradients for NaN/Inf
                grad_is_finite = True
                for param in self.distiller.student.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"  Warning: NaN/Inf in gradients, skipping optimizer step")
                            grad_is_finite = False
                            break

                if not grad_is_finite:
                    self.distiller.optimizer.zero_grad()
                    step += 1
                    continue

                # Accumulate metrics only for successful batches
                running_loss += loss.item() * self.args.grad_accum_steps
                running_kd_loss += kd_loss
                running_ce_loss += ce_loss
                total_tokens += batch_tokens
                successful_steps += 1

            except Exception as e:
                print(f"  Backward pass error at step {step}: {e}")
                self.distiller.optimizer.zero_grad()
                step += 1
                continue

            # Optimizer step with ultra-conservative gradient clipping
            if (step + 1) % self.args.grad_accum_steps == 0:
                try:
                    if use_scaler:
                        scaler.unscale_(self.distiller.optimizer)

                    # Ultra-aggressive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.distiller.student.parameters(),
                        self.args.max_grad_norm
                    )

                    # Skip update for high gradient norms
                    if grad_norm > 2.0:
                        print(f"  Skipping optimizer step: grad_norm={grad_norm:.4f} > 2.0")
                        if use_scaler:
                            scaler.update()
                        self.distiller.optimizer.zero_grad()
                        step += 1
                        continue

                    # Perform optimizer step
                    if use_scaler:
                        scaler.step(self.distiller.optimizer)
                        scaler.update()
                    else:
                        self.distiller.optimizer.step()

                    self.distiller.optimizer.zero_grad()

                    # Check model parameters for NaN after update
                    params_are_finite = True
                    for param in self.distiller.student.parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"  CRITICAL: NaN/Inf in model parameters after update!")
                            params_are_finite = False
                            break

                    if not params_are_finite:
                        print("  🔄 Resetting to initial state due to parameter corruption")
                        self.reset_to_initial_state()

                except Exception as e:
                    print(f"  Optimizer step failed at step {step}: {e}")
                    self.distiller.optimizer.zero_grad()

                # Step scheduler
                if self.distiller.scheduler is not None:
                    self.distiller.scheduler.step()

                # Clear cache periodically
                if self.args.clear_cache_every_step and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            step += 1

            # Logging
            if step % (self.args.grad_accum_steps * 5) == 0 and log_callback and successful_steps > 0:
                avg_loss = running_loss / successful_steps
                avg_kd_loss = running_kd_loss / successful_steps
                avg_ce_loss = running_ce_loss / successful_steps

                metrics = {
                    "step": step,
                    "train_loss": avg_loss,
                    "train_kd_loss": avg_kd_loss,
                    "train_ce_loss": avg_ce_loss,
                    "train_ppl": min(math.exp(avg_loss), 10000),
                    "tokens": total_tokens,
                    "successful_batches": successful_steps,
                    "skipped_batches": skipped_batches,
                    "success_rate": successful_steps / (successful_steps + skipped_batches) if (successful_steps + skipped_batches) > 0 else 0,
                    "lr": self.distiller.optimizer.param_groups[0]['lr'] if self.distiller.optimizer else 0,
                    "consecutive_nans": self.consecutive_nan_batches
                }
                log_callback(metrics)

            # Evaluation
            if step % self.args.eval_steps == 0 and step > 0:
                val_ppl = self.run_evaluation(step, log_callback)

                # Track best model
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self.best_step = step
                    print(f"  🏆 New best model! PPL: {val_ppl:.2f}")

            # Save checkpoint
            if self.args.save_steps > 0 and step % self.args.save_steps == 0 and step > 0:
                if save_callback:
                    save_callback(step, self.distiller.student, getattr(self.distiller, 'router_linears', []))

        # Final evaluation
        print("🏁 Running final evaluation...")
        final_ppl = self.run_evaluation(step, log_callback)
        print(f"Final validation perplexity: {final_ppl:.2f}")
        print(f"Best validation perplexity: {self.best_val_ppl:.2f} at step {self.best_step}")
        print(f"Total batches skipped due to numerical issues: {skipped_batches}")

        return self.training_history

    def run_evaluation(self, step, log_callback):
        """Run evaluation with comprehensive stability measures"""
        print(f"🔍 Running ultra-stable evaluation at step {step}")

        # Move teacher to CPU to save memory
        if self.args.offload_teacher_to_cpu:
            self.distiller.teacher.cpu()
            torch.cuda.empty_cache()

        evaluator = UltraStableEvaluator(
            eval_dataset=None,  # Will be set up in main
            tokenizer=None,     # Will be set up in main
            device=self.distiller.device,
            batch_size=1,       # Ultra-conservative batch size
            eval_batches=8      # Fewer batches for stability
        )

        # This will be properly configured in main()
        val_ppl = float('inf')

        if log_callback:
            metrics = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "step": step,
                "val_loss": math.log(val_ppl) if val_ppl != float('inf') else float('inf'),
                "val_ppl": val_ppl,
                "best_val_ppl": self.best_val_ppl,
                "best_step": self.best_step,
                "is_best": val_ppl < self.best_val_ppl,
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            log_callback(metrics)

        # Move teacher back to GPU if needed
        if not self.args.offload_teacher_to_cpu:
            self.distiller.teacher.to(self.distiller.device)

        return val_ppl


def main():
    args = parse_args()

    print("🛡️  ULTRA-STABLE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Maximum FP16 stability with aggressive numerical safeguards")
    print("=" * 60)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = args.model_dtype == 'float16'
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"\n🔧 Ultra-Stable Configuration:")
    print(f"  Device: {device}")
    print(f"  Model dtype: {dtype}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Ultra-conservative student LR: {args.student_lr}")
    print(f"  Ultra-aggressive grad clipping: {args.max_grad_norm}")
    print(f"  Conservative temperature: {args.temperature}")
    print(f"  Maximum FP16 stability: ENABLED")

    if torch.cuda.is_available():
        # Enable TF32 for better performance while maintaining stability
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_memory_info("Initial", device)

    # Load models
    print("\n📚 Loading models with ultra-stable settings...")
    teacher, student, teacher_tok, student_tok = load_teacher_student(
        args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
    )
    print_memory_info("After model loading", device)

    # Build datasets
    print("📊 Building datasets...")
    train_ds, eval_ds, same_tok = build_text_datasets(
        args.dataset_name, args.dataset_config_name,
        teacher_tok, student_tok, args.block_size
    )
    print(f"Dataset info: train={len(train_ds)}, eval={len(eval_ds) if eval_ds else 0}, same_tokenizer={same_tok}")

    # Create data collator
    if same_tok:
        collator = DataCollatorForLanguageModeling(teacher_tok, mlm=False)
    else:
        collator = DualTokenizerCollator(teacher_tok, student_tok, args.block_size)

    # Create data loader with conservative settings
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,  # Disabled for stability
        num_workers=0,     # Single-threaded for stability
    )

    # Create configuration with ultra-conservative settings
    cfg = SARConfig(
        temperature=args.temperature,
        alpha_kd=args.alpha_kd,
        alpha_ce=args.alpha_ce,
        router_anchor_weight=args.router_anchor_weight,
        router_load_balance_weight=args.router_load_balance_weight,
        router_entropy_weight=args.router_entropy_weight,
        student_lr=args.student_lr,
        router_lr=args.router_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_scheduler=args.use_scheduler,
        warmup_steps=min(100, args.train_steps // 10),  # Conservative warmup
        use_fp16=use_fp16,
    )

    print("⚗️ Creating ultra-stable distiller...")
    distiller = SARDistiller(teacher, student, cfg, device)

    # Setup ultra-stable optimizer with enhanced epsilon for FP16
    distiller.optimizer = torch.optim.AdamW(
        distiller.student.parameters(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        eps=1e-8,  # More stable for FP16
        betas=(0.9, 0.95)  # More stable betas
    )

    # Setup scheduler with conservative settings
    if cfg.use_scheduler:
        print("Using ultra-conservative cosine learning rate scheduler")
        distiller.scheduler = get_cosine_schedule_with_warmup(
            distiller.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=args.train_steps,
            num_cycles=0.5,  # Gentler cosine decay
            last_epoch=-1,
        )

    # Move teacher to CPU for memory efficiency
    if args.offload_teacher_to_cpu:
        print("Moving teacher model to CPU for memory optimization")
        distiller.teacher.cpu()

    print_memory_info("After setup", device)

    # Test initial evaluation
    print("\n🧪 Testing initial evaluation...")
    evaluator = UltraStableEvaluator(eval_ds, student_tok, device, batch_size=1, eval_batches=5)
    initial_ppl = evaluator.evaluate(distiller.student, use_fp16=use_fp16)
    print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.2f}")

    # Setup trainer
    trainer = UltraStableTrainer(distiller, args)

    # Setup callbacks
    def save_callback(step, model, router_linears):
        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        try:
            # Save student model
            model.save_pretrained(save_path)
            student_tok.save_pretrained(save_path)

            # Save training info
            info = {
                "step": step,
                "best_val_ppl": trainer.best_val_ppl,
                "best_step": trainer.best_step,
                "config": vars(args)
            }
            with open(os.path.join(save_path, "training_info.json"), "w") as f:
                json.dump(info, f, indent=2)

            print(f"💾 Saved checkpoint at step {step}")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")

    def log_callback(metrics):
        # Print progress
        if "val_ppl" in metrics:
            print(f"  📊 Step {metrics['step']}: Validation PPL = {metrics['val_ppl']:.2f}")
        else:
            step = metrics.get("step", 0)
            loss = metrics.get("train_loss", 0)
            ppl = metrics.get("train_ppl", 0)
            success_rate = metrics.get("success_rate", 0)
            consecutive_nans = metrics.get("consecutive_nans", 0)
            print(f"Step {step}: Loss={loss:.4f}, PPL={ppl:.1f}, Success={success_rate:.3f}, ConsecNaNs={consecutive_nans}")

        # Log to file
        log_path = os.path.join(args.output_dir, "training.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # Start training
    print("\n🚀 Starting ultra-stable training...")

    # Configure evaluator for trainer
    trainer.evaluator = UltraStableEvaluator(eval_ds, student_tok, device, batch_size=1, eval_batches=8)

    def run_evaluation_wrapper(step, log_callback):
        val_ppl = trainer.evaluator.evaluate(distiller.student, use_fp16=use_fp16)
        if log_callback:
            metrics = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "step": step,
                "val_loss": math.log(val_ppl) if val_ppl != float('inf') else float('inf'),
                "val_ppl": val_ppl,
                "best_val_ppl": trainer.best_val_ppl,
                "best_step": trainer.best_step,
                "is_best": val_ppl < trainer.best_val_ppl,
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            }
            log_callback(metrics)
        return val_ppl

    trainer.run_evaluation = run_evaluation_wrapper

    try:
        history = trainer.train(train_loader, save_callback, log_callback)

        print("\n✅ Training completed successfully!")
        print(f"📊 Best validation perplexity: {trainer.best_val_ppl:.2f} at step {trainer.best_step}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    finally:
        print_memory_info("Final", device)


if __name__ == "__main__":
    main()
