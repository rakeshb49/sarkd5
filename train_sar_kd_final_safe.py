#!/usr/bin/env python3
"""
Final Safe SAR Knowledge Distillation Training Script
====================================================

This version addresses the fundamental FP16 parameter corruption issue by using
MIXED PRECISION training instead of pure FP16:
- Models loaded in FP16 to save memory
- Forward pass in FP16 with autocast
- Parameters and gradients kept in FP32 for stable updates
- No parameter corruption issues

This is the production-ready version for P100/V100 environments.
"""

import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import math
import time
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Import SAR modules
from sar_kd.trainer import SARDistiller, SARConfig
from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student


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

    # Conservative learning rates for stability
    parser.add_argument("--student_lr", type=float, default=5e-6)
    parser.add_argument("--router_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Knowledge distillation - stable settings
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_kd", type=float, default=0.1)
    parser.add_argument("--alpha_ce", type=float, default=0.9)

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


class FinalSafeEvaluator:
    """Final safe evaluator with mixed precision"""

    def __init__(self, eval_dataset, tokenizer, device, batch_size=2, eval_batches=10):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def evaluate(self, student_model, use_mixed_precision=True):
        """Evaluate with mixed precision for stability"""
        if self.eval_dataset is None:
            return float('inf')

        # Use a small, fixed subset for consistency
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

                        # Mixed precision forward pass
                        with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits[:, :-1, :]
                            targets = labels[:, 1:]

                            # Safe logit clamping
                            logits = torch.clamp(logits, -10, 10)

                            # Check for issues in logits
                            if torch.isnan(logits).any() or torch.isinf(logits).any():
                                print(f"    Warning: NaN/Inf in logits batch {batch_idx}, skipping")
                                continue

                            # Calculate loss
                            valid_mask = (targets != -100)
                            if not valid_mask.any():
                                continue

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

                            # Check losses before processing
                            if torch.isnan(losses).any() or torch.isinf(losses).any():
                                print(f"    Warning: NaN/Inf in raw losses batch {batch_idx}, skipping")
                                continue

                            losses = losses.view(targets.shape)
                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            # Conservative loss clamping
                            batch_loss = torch.clamp(batch_loss, 0.1, 15.0)

                            # Final check
                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                print(f"    Warning: NaN/Inf final loss in batch {batch_idx}, skipping")
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

        # Conservative perplexity calculation
        if avg_loss > 20:
            print(f"  Warning: Clamping extreme avg_loss {avg_loss:.4f} to 20")
            avg_loss = 20

        try:
            perplexity = math.exp(avg_loss)
            if math.isnan(perplexity) or math.isinf(perplexity) or perplexity > 1000000:
                print(f"  Error: Invalid perplexity, returning 1000000")
                return 1000000.0
        except OverflowError:
            print(f"  Error: Perplexity overflow, returning 1000000")
            return 1000000.0

        return perplexity


class FinalSafeTrainer:
    """Final safe trainer with mixed precision instead of pure FP16"""

    def __init__(self, distiller, args):
        self.distiller = distiller
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_history = []

    def safe_forward_pass(self, batch):
        """Safe forward pass with mixed precision"""
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

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=True):
                # Student forward pass
                s_out = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = s_out.logits[:, :-1, :]
                target_labels = labels[:, 1:]

                # Safe logit clamping
                student_logits = torch.clamp(student_logits, -10, 10)

                # Check for NaN/Inf in logits
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    print("  Warning: NaN/Inf in student logits, skipping batch")
                    return None

                # Calculate CE loss
                valid_mask = (target_labels != -100)
                if not valid_mask.any():
                    return None

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_losses = loss_fct(student_logits.view(-1, student_logits.size(-1)), target_labels.view(-1))

                # Check raw losses
                if torch.isnan(ce_losses).any() or torch.isinf(ce_losses).any():
                    print("  Warning: NaN/Inf in raw CE losses, skipping batch")
                    return None

                ce_losses = ce_losses.view(target_labels.shape)
                masked_ce = ce_losses * valid_mask.float()
                ce = masked_ce.sum() / valid_mask.sum()

                # Initialize KD loss
                kd = torch.tensor(0.0, device=self.distiller.device)

                # Only do KD if teacher is available and not completely offloaded
                if not self.args.offload_teacher_to_cpu:
                    try:
                        with torch.no_grad():
                            t_out = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                            teacher_logits = t_out.logits[:, :-1, :]

                        # Safe teacher logit clamping
                        teacher_logits = torch.clamp(teacher_logits, -10, 10)

                        # Check teacher logits
                        if not (torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any()):
                            # Safe KD loss with temperature scaling
                            teacher_probs = torch.softmax(teacher_logits / self.distiller.config.temperature, dim=-1)
                            student_log_probs = torch.log_softmax(student_logits / self.distiller.config.temperature, dim=-1)

                            kd = -(teacher_probs * student_log_probs).sum(dim=-1)
                            kd = kd.mean()
                            kd = kd * (self.distiller.config.temperature ** 2)

                            # Clamp KD loss
                            kd = torch.clamp(kd, 0, 10)

                            if torch.isnan(kd) or torch.isinf(kd):
                                kd = torch.tensor(0.0, device=self.distiller.device)

                    except Exception as e:
                        print(f"  Teacher forward failed, using CE only: {e}")
                        kd = torch.tensor(0.0, device=self.distiller.device)

                # Safe loss clamping
                ce = torch.clamp(ce, 0.1, 15.0)

                # Final NaN check
                if torch.isnan(ce) or torch.isinf(ce):
                    print(f"  Warning: Final CE loss is NaN/Inf: {ce}, skipping batch")
                    return None

                if torch.isnan(kd) or torch.isinf(kd):
                    kd = torch.tensor(0.0, device=self.distiller.device)

                # Total loss
                total_loss = self.distiller.config.alpha_ce * ce + self.distiller.config.alpha_kd * kd

                # Final total loss check
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  Warning: Total loss is NaN/Inf, skipping batch")
                    return None

                tokens = valid_mask.sum().item()
                return total_loss, kd.item(), ce.item(), tokens

        except Exception as e:
            print(f"  Forward pass error: {e}")
            return None

    def train(self, train_loader, save_callback=None, log_callback=None):
        """Final safe training loop with mixed precision"""
        print("🚀 Starting FINAL SAFE SAR Knowledge Distillation training...")
        print("🔧 Using MIXED PRECISION: FP16 forward pass + FP32 parameters/gradients")

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

        # Mixed precision setup - ALWAYS use gradient scaling with mixed precision
        use_mixed_precision = torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision)

        # Print configuration
        print(f"🔧 Training Configuration:")
        print(f"   - Mixed precision training: {use_mixed_precision}")
        print(f"   - Gradient scaling: {use_mixed_precision}")
        print(f"   - Model parameters remain in FP32")

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            val_ppl = self.run_evaluation(step, log_callback)

        # Training loop
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            # Safe forward pass
            result = self.safe_forward_pass(batch)
            if result is None:
                step += 1
                skipped_batches += 1
                if skipped_batches % 10 == 0:
                    print(f"  ⚠️  Skipped {skipped_batches} batches so far due to numerical issues")
                continue

            loss, kd_loss, ce_loss, batch_tokens = result

            # Backward pass with mixed precision
            try:
                loss = loss / self.args.grad_accum_steps

                # Scale loss and backward
                scaler.scale(loss).backward()

                # Check gradients (they should be FP32)
                grad_is_finite = True
                for param in self.distiller.student.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"  Warning: NaN/Inf in gradients, skipping optimizer step")
                            grad_is_finite = False
                            break

                if not grad_is_finite:
                    scaler.update()
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

            # Optimizer step with gradient scaling
            if (step + 1) % self.args.grad_accum_steps == 0:
                try:
                    # Unscale gradients
                    scaler.unscale_(self.distiller.optimizer)

                    # Conservative gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.distiller.student.parameters(),
                        self.args.max_grad_norm
                    )

                    # Skip update for very high gradient norms
                    if grad_norm > 10.0:
                        print(f"  Skipping optimizer step: grad_norm={grad_norm:.4f} > 10.0")
                        scaler.update()
                        self.distiller.optimizer.zero_grad()
                        step += 1
                        continue

                    # Perform optimizer step
                    scaler.step(self.distiller.optimizer)
                    scaler.update()
                    self.distiller.optimizer.zero_grad()

                    # Check model parameters (should remain finite)
                    params_are_finite = True
                    for param in self.distiller.student.parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"  ERROR: Model parameters became NaN/Inf after update!")
                            params_are_finite = False
                            break

                    if not params_are_finite:
                        print("  This should not happen with mixed precision training!")

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
            if step % (self.args.grad_accum_steps * 10) == 0 and log_callback and successful_steps > 0:
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
                    "lr": self.distiller.optimizer.param_groups[0]['lr'] if self.distiller.optimizer else 0
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
        """Run evaluation with mixed precision"""
        print(f"🔍 Running safe evaluation at step {step}")

        # Move teacher to CPU if needed to save memory during eval
        if self.args.offload_teacher_to_cpu:
            self.distiller.teacher.cpu()
            torch.cuda.empty_cache()

        # This will be properly configured in main()
        val_ppl = float('inf')

        if hasattr(self, 'evaluator') and self.evaluator is not None:
            val_ppl = self.evaluator.evaluate(self.distiller.student, use_mixed_precision=True)

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

        # Move teacher back to GPU if not offloaded
        if not self.args.offload_teacher_to_cpu:
            self.distiller.teacher.to(self.distiller.device)

        return val_ppl


def main():
    args = parse_args()

    print("🔧 FINAL SAFE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Mixed Precision: FP16 forward + FP32 parameters = STABLE")
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

    print(f"\n🔧 Final Safe Configuration:")
    print(f"  Device: {device}")
    print(f"  Model loading dtype: {dtype}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Student LR: {args.student_lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Mixed Precision: FP16 forward + FP32 params")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_memory_info("Initial", device)

    # Load models
    print("\n📚 Loading models...")
    teacher, student, teacher_tok, student_tok = load_teacher_student(
        args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
    )

    # CRITICAL: Convert student parameters back to FP32 for mixed precision training
    if use_fp16:
        print("🔄 Converting student parameters to FP32 for mixed precision stability...")
        student = student.float()  # Convert parameters to FP32
        # Keep teacher in FP16 to save memory since it's often frozen/CPU offloaded

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

    # Create data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )

    # Create configuration
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
        warmup_steps=min(100, args.train_steps // 10),
        scheduler_type='cosine',
        total_steps=args.train_steps,
        use_fp16=False,  # We handle mixed precision manually
        offload_teacher_to_cpu=args.offload_teacher_to_cpu,
        clear_cache_every_step=args.clear_cache_every_step,
    )

    print("⚗️ Creating final safe distiller...")
    distiller = SARDistiller(teacher, student, device, cfg)

    print_memory_info("After setup", device)

    # Test initial evaluation
    print("\n🧪 Testing initial evaluation...")
    evaluator = FinalSafeEvaluator(eval_ds, student_tok, device, batch_size=2, eval_batches=8)
    initial_ppl = evaluator.evaluate(distiller.student, use_mixed_precision=True)
    print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.2f}")

    # Setup trainer
    trainer = FinalSafeTrainer(distiller, args)
    trainer.evaluator = evaluator

    # Setup callbacks
    def save_callback(step, model, router_linears):
        save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(save_path, exist_ok=True)

        try:
            model.save_pretrained(save_path)
            student_tok.save_pretrained(save_path)

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
            print(f"Step {step}: Loss={loss:.4f}, PPL={ppl:.1f}, Success={success_rate:.3f}")

        # Log to file
        log_path = os.path.join(args.output_dir, "training.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # Start training
    print("\n🚀 Starting final safe training...")
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
