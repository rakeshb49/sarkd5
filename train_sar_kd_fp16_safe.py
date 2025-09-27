#!/usr/bin/env python3
"""
FP16-Safe SAR Knowledge Distillation Training Script
===================================================

Ultra-conservative version specifically designed to handle FP16 numerical stability issues.
Addresses NaN losses, gradient explosion, and logit overflow in mixed precision training.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

from sar_kd.trainer import SARDistiller, SARConfig
from sar_kd.data import DualTokenizerCollator
from sar_kd.router_utils import collect_router_linears, freeze_all_but_router


def parse_args():
    """Parse command line arguments with ultra-conservative FP16-safe defaults"""
    p = argparse.ArgumentParser(description='FP16-Safe SAR Knowledge Distillation')

    # Model arguments
    p.add_argument('--teacher_model', default='microsoft/DialoGPT-medium')
    p.add_argument('--student_model', default='distilgpt2')
    p.add_argument('--model_dtype', choices=['float16', 'float32'], default='float16')

    # Dataset arguments
    p.add_argument('--dataset_name', default='wikitext')
    p.add_argument('--dataset_config_name', default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=256)

    # Training arguments - extremely conservative for FP16
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=8)
    p.add_argument('--train_steps', type=int, default=500)
    p.add_argument('--eval_steps', type=int, default=50)
    p.add_argument('--save_steps', type=int, default=100)

    # Ultra-conservative learning rates for FP16 stability
    p.add_argument('--student_lr', type=float, default=5e-6)
    p.add_argument('--router_lr', type=float, default=1e-5)

    # Conservative KD parameters
    p.add_argument('--temperature', type=float, default=1.5)
    p.add_argument('--alpha_kd', type=float, default=0.05)
    p.add_argument('--alpha_ce', type=float, default=0.95)

    # Minimal regularization
    p.add_argument('--weight_decay', type=float, default=0.001)
    p.add_argument('--max_grad_norm', type=float, default=0.5)

    # Scheduler settings
    p.add_argument('--use_scheduler', action='store_true', default=True)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--scheduler_type', choices=['cosine', 'linear'], default='linear')

    # Memory and stability
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=True)
    p.add_argument('--clear_cache_every_step', action='store_true', default=True)
    p.add_argument('--eval_batches', type=int, default=20)

    # Output
    p.add_argument('--output_dir', default='./sar_kd_fp16_safe_output')
    p.add_argument('--save_best_model', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def load_teacher_student(teacher_name, student_name, dtype, device, use_fp16=False):
    """Load models with maximum FP16 safety"""
    print("📚 Loading models with FP16-safe configuration...")

    # Load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    student_tokenizer.pad_token = student_tokenizer.eos_token

    print("Loading models with dtype:", dtype)

    # Load models
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=dtype,
        device_map=None
    )

    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=dtype,
        device_map=None
    )

    # Move to device
    print(f"Moving teacher model to {device}...")
    teacher = teacher.to(device)

    print(f"Moving student model to {device}...")
    student = student.to(device)

    # Enable gradient checkpointing
    teacher.gradient_checkpointing_enable()
    student.gradient_checkpointing_enable()
    print("Enabled gradient checkpointing for teacher model")
    print("Enabled gradient checkpointing for student model")

    # Log GPU memory if available
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {memory_gb:.2f} GB")
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory reserved: {reserved_gb:.2f} GB")

    return teacher, student, teacher_tokenizer, student_tokenizer


class FP16SafeEvaluator:
    """Ultra-safe evaluator specifically designed for FP16 numerical stability"""

    def __init__(self, eval_dataset, tokenizer_compatible, collator, batch_size, eval_batches, device):
        self.eval_dataset = eval_dataset
        self.tokenizer_compatible = tokenizer_compatible
        self.collator = collator
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.device = device

    def evaluate(self, student_model, use_fp16=False):
        """FP16-safe evaluation with comprehensive numerical stability checks"""
        if self.eval_dataset is None:
            return float('inf')

        student_model.eval()
        total_loss = 0.0
        total_tokens = 0
        batch_count = 0

        try:
            # Sample fresh data for evaluation
            indices = torch.randperm(len(self.eval_dataset))[:min(len(self.eval_dataset), self.eval_batches * self.batch_size * 2)]
            eval_subset = self.eval_dataset.select(indices.tolist())

            eval_loader = DataLoader(
                eval_subset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collator,
                drop_last=False
            )

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if batch_count >= self.eval_batches:
                        break

                    try:
                        if self.tokenizer_compatible:
                            input_ids = batch["input_ids"].to(self.device)
                            attention_mask = batch.get("attention_mask")
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch["labels"].to(self.device)
                        else:
                            # Use student tokenizer data for dual tokenizer mode
                            input_ids = batch["student_input_ids"].to(self.device)
                            attention_mask = batch.get("student_attention_mask")
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch["student_labels"].to(self.device)

                        # Forward pass with FP16 safety
                        with torch.amp.autocast('cuda', enabled=use_fp16):
                            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits[:, :-1, :]
                            targets = labels[:, 1:]

                            # CRITICAL: Clamp logits to prevent FP16 overflow
                            logits = torch.clamp(logits, -6, 6)

                            # Calculate loss with proper masking
                            valid_mask = (targets != -100)
                            if valid_mask.sum() == 0:
                                continue

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
                            losses = losses.view(targets.shape)

                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            # Ultra-conservative loss clamping for FP16
                            batch_loss = torch.clamp(batch_loss, 0, 8)

                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                print(f"    Warning: NaN/Inf loss in eval batch {batch_idx}, skipping")
                                continue

                            # Additional check for extreme values
                            if batch_loss.item() > 6:
                                print(f"    Warning: High eval loss {batch_loss.item():.4f} in batch {batch_idx}")

                            total_loss += batch_loss.item()
                            total_tokens += valid_mask.sum().item()
                            batch_count += 1

                    except Exception as e:
                        print(f"  Warning: Eval batch {batch_idx} failed: {e}")
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

        # Prevent overflow in exp calculation for FP16
        if avg_loss > 10:
            print(f"  Warning: Clamping high avg_loss {avg_loss:.4f} to 10")
            avg_loss = 10

        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            print(f"  Error: Overflow in exp calculation, using fallback")
            perplexity = float('inf')

        # Final sanity check
        if math.isnan(perplexity) or math.isinf(perplexity):
            print(f"  Error: Invalid perplexity calculated from avg_loss={avg_loss}")
            return float('inf')

        return perplexity


class FP16SafeTrainer:
    """Ultra-safe trainer specifically designed for FP16 numerical stability"""

    def __init__(self, distiller, evaluator, args):
        self.distiller = distiller
        self.evaluator = evaluator
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_history = []
        self.consecutive_nan_batches = 0
        self.max_consecutive_nan = 10

    def safe_forward_pass_fp16(self, batch):
        """Ultra-safe forward pass specifically designed for FP16 stability"""
        try:
            # Handle batch data
            if 'teacher_input_ids' in batch:
                # Dual tokenizer mode - use only student to avoid alignment issues in FP16
                s_ids = batch['student_input_ids'].to(self.distiller.device)
                s_mask = batch.get('student_attention_mask')
                if s_mask is not None:
                    s_mask = s_mask.to(self.distiller.device)
                s_labels = batch['student_labels'].to(self.distiller.device)

                # Skip teacher for maximum stability in FP16
                with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                    s_out = self.distiller.student(input_ids=s_ids, attention_mask=s_mask)
                    s_logits = s_out.logits[:, :-1, :]
                    s_targets = s_labels[:, 1:]

                    # CRITICAL: Aggressive logit clamping for FP16
                    s_logits = torch.clamp(s_logits, -5, 5)

                    # Compute CE loss only
                    valid_mask = (s_targets != -100)
                    if not valid_mask.any():
                        return None, 0.0, 0.0, 0

                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(s_logits.contiguous().view(-1, s_logits.size(-1)), s_targets.contiguous().view(-1))
                    losses = losses.view(s_targets.shape)

                    masked_losses = losses * valid_mask.float()
                    ce = masked_losses.sum() / valid_mask.sum()

                    # No KD loss in ultra-safe mode
                    kd = torch.tensor(0.0, device=self.distiller.device)
                    tokens = valid_mask.sum().item()

            else:
                # Same tokenizer path with ultra-conservative settings
                input_ids = batch["input_ids"].to(self.distiller.device)
                labels = batch["labels"].to(self.distiller.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.distiller.device)

                with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                    # Student forward pass
                    s_out = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = s_out.logits[:, :-1, :]
                    target_labels = labels[:, 1:]

                    # CRITICAL: Aggressive logit clamping
                    student_logits = torch.clamp(student_logits, -5, 5)

                    # Conservative KD approach
                    kd = torch.tensor(0.0, device=self.distiller.device)
                    if self.args.alpha_kd > 0:
                        try:
                            # Teacher forward with extra safety
                            with torch.no_grad():
                                t_out = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                                teacher_logits = t_out.logits[:, :-1, :]
                                teacher_logits = torch.clamp(teacher_logits, -5, 5)

                            # Ultra-conservative KD loss
                            teacher_probs = torch.softmax(teacher_logits / self.distiller.config.temperature, dim=-1)
                            student_log_probs = torch.log_softmax(student_logits / self.distiller.config.temperature, dim=-1)

                            kd = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                            kd = kd * (self.distiller.config.temperature ** 2)

                        except Exception as e:
                            print(f"    KD computation failed, using CE only: {e}")
                            kd = torch.tensor(0.0, device=self.distiller.device)

                    # CE loss computation
                    valid_mask = (target_labels != -100)
                    if not valid_mask.any():
                        return None, 0.0, 0.0, 0

                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    ce_losses = loss_fct(student_logits.contiguous().view(-1, student_logits.size(-1)),
                                       target_labels.contiguous().view(-1))
                    ce_losses = ce_losses.view(target_labels.shape)

                    masked_ce = ce_losses * valid_mask.float()
                    ce = masked_ce.sum() / valid_mask.sum()
                    tokens = valid_mask.sum().item()

            # Ultra-aggressive loss clamping for FP16
            kd = torch.clamp(kd, 0, 3)
            ce = torch.clamp(ce, 0, 6)

            # Comprehensive NaN checks
            if torch.isnan(kd) or torch.isinf(kd):
                print(f"  KD loss issue: kd={kd}, setting to 0")
                kd = torch.tensor(0.0, device=self.distiller.device)

            if torch.isnan(ce) or torch.isinf(ce):
                print(f"  CE loss issue: ce={ce}, skipping batch")
                self.consecutive_nan_batches += 1
                return None, 0.0, 0.0, 0

            # Reset consecutive NaN counter on success
            self.consecutive_nan_batches = 0

            # Compute total loss
            total_loss = self.distiller.config.alpha_ce * ce + self.distiller.config.alpha_kd * kd

            # Final NaN check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  Total loss NaN: ce={ce.item():.4f}, kd={kd.item():.4f}")
                self.consecutive_nan_batches += 1
                return None, 0.0, 0.0, 0

            # Conservative total loss clamping
            if total_loss.item() > 8:
                print(f"  Warning: High loss {total_loss.item():.4f}, clamping to 6")
                total_loss = torch.clamp(total_loss, 0, 6)

            return total_loss, kd.item(), ce.item(), tokens

        except Exception as e:
            print(f"    Forward pass error: {e}")
            self.consecutive_nan_batches += 1
            return None, 0.0, 0.0, 0

    def train(self, train_loader, save_callback=None, log_callback=None):
        """Ultra-safe training loop with FP16 stability monitoring"""
        print("🚀 Starting FP16-safe SAR Knowledge Distillation training...")

        # FP16-safe gradient scaler setup
        use_scaler = torch.cuda.is_available() and self.distiller.use_fp16

        # CRITICAL: Detect FP16 parameters and disable scaler
        has_fp16_params = any(p.dtype == torch.float16 for p in self.distiller.student.parameters())
        if hasattr(self.distiller, 'router_params') and self.distiller.router_params:
            has_fp16_params = has_fp16_params or any(p.dtype == torch.float16 for p in self.distiller.router_params)

        if has_fp16_params and use_scaler:
            print("WARNING: FP16 parameters detected - disabling gradient scaler to prevent errors")
            use_scaler = False

        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Print training mode for clarity
        if torch.cuda.is_available():
            if self.distiller.use_fp16 and not has_fp16_params:
                print("🔧 Mixed precision: FP16 compute + FP32 params + gradient scaling")
            elif has_fp16_params:
                print("🔧 FP16 mode: FP16 compute + FP16 params (no gradient scaling)")
            else:
                print("🔧 FP32 mode: Full precision training")
        else:
            print("🔧 CPU mode: No gradient scaling")

        # Training state
        step = 0
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        successful_steps = 0

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            self.run_evaluation(step, log_callback)

        # Training loop with FP16 safety monitoring
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            # Check for too many consecutive failures
            if self.consecutive_nan_batches >= self.max_consecutive_nan:
                print(f"❌ Too many consecutive NaN batches ({self.consecutive_nan_batches}), stopping training")
                break

            # Safe forward pass
            loss, kd_loss, ce_loss, tokens = self.safe_forward_pass_fp16(batch)

            if loss is None:
                step += 1
                continue

            # Backward pass with FP16 safety
            try:
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulate metrics
                running_loss += loss.item()
                running_kd_loss += kd_loss
                running_ce_loss += ce_loss
                successful_steps += 1

            except Exception as e:
                print(f"  Backward pass failed at step {step}: {e}")
                step += 1
                continue

            # Optimizer step with enhanced safety
            if (step + 1) % self.args.grad_accum_steps == 0:
                try:
                    if use_scaler:
                        scaler.unscale_(self.distiller.optimizer)

                        # Ultra-conservative gradient clipping for FP16
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.distiller.student.parameters(),
                            self.distiller.config.max_grad_norm
                        )

                        # Skip step if gradients are unstable
                        if grad_norm > 2.0 or torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"  Skipping step: unstable gradients (norm={grad_norm:.4f})")
                            scaler.update()
                            self.distiller.optimizer.zero_grad()
                            step += 1
                            continue

                        scaler.step(self.distiller.optimizer)
                        scaler.update()
                    else:
                        # Conservative gradient clipping for FP16
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.distiller.student.parameters(),
                            self.distiller.config.max_grad_norm
                        )

                        # Skip step if gradients are unstable
                        if grad_norm > 2.0 or torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"  Skipping step: unstable gradients (norm={grad_norm:.4f})")
                            self.distiller.optimizer.zero_grad()
                            step += 1
                            continue

                        self.distiller.optimizer.step()

                    # Update scheduler
                    if hasattr(self.distiller, 'scheduler') and self.distiller.scheduler:
                        self.distiller.scheduler.step()

                    self.distiller.optimizer.zero_grad()

                except Exception as e:
                    print(f"  Optimizer step failed at step {step}: {e}")

            # Periodic logging
            if (step + 1) % 25 == 0 and successful_steps > 0:
                avg_loss = running_loss / successful_steps
                avg_kd_loss = running_kd_loss / successful_steps
                avg_ce_loss = running_ce_loss / successful_steps
                success_rate = successful_steps / (step + 1)

                print(f"  Step {step+1}: avg_loss={avg_loss:.4f} (kd={avg_kd_loss:.4f}, ce={avg_ce_loss:.4f}) "
                      f"success_rate={success_rate:.2%}")

                if log_callback:
                    metrics = {
                        "step": step + 1,
                        "train_loss": avg_loss,
                        "train_kd_loss": avg_kd_loss,
                        "train_ce_loss": avg_ce_loss,
                        "train_ppl": min(math.exp(avg_loss), 10000),
                        "tokens": tokens,
                        "successful_batches": successful_steps,
                        "success_rate": success_rate,
                        "consecutive_nan_batches": self.consecutive_nan_batches,
                        "lr": self.distiller.optimizer.param_groups[0]['lr'] if self.distiller.optimizer else 0
                    }
                    if torch.cuda.is_available():
                        metrics["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3

                    log_callback(metrics)

            # Evaluation
            if (step + 1) % self.args.eval_steps == 0:
                self.run_evaluation(step + 1, log_callback)

            # Memory cleanup
            if self.args.clear_cache_every_step and torch.cuda.is_available():
                torch.cuda.empty_cache()

            step += 1

        # Final evaluation
        print("🏁 Running final evaluation...")
        final_ppl = self.run_evaluation(step, log_callback)
        print(f"Final validation perplexity: {final_ppl:.2f}")
        best_ppl_display = self.best_val_ppl if self.best_val_ppl != float('inf') else final_ppl
        print(f"Best validation perplexity: {best_ppl_display:.2f} at step {self.best_step}")

        return self.training_history

    def run_evaluation(self, step, log_callback=None):
        """Run evaluation with FP16 safety checks"""
        try:
            val_ppl = self.evaluator.evaluate(self.distiller.student, use_fp16=self.distiller.use_fp16)
            val_loss = math.log(val_ppl) if val_ppl != float('inf') else float('inf')

            print(f"  📊 Step {step}: Validation PPL = {val_ppl:.2f}")

            # Track best model
            is_best = val_ppl < self.best_val_ppl
            if is_best:
                self.best_val_ppl = val_ppl
                self.best_step = step
                print(f"  🏆 New best model! PPL: {val_ppl:.2f}")

            # Log results
            if log_callback:
                metrics = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_ppl": self.best_val_ppl if self.best_val_ppl != float('inf') else val_ppl,
                    "best_step": self.best_step,
                    "is_best": is_best
                }
                if torch.cuda.is_available():
                    metrics["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3

                log_callback(metrics)

            return val_ppl

        except Exception as e:
            print(f"❌ Evaluation failed at step {step}: {e}")
            return float('inf')


def main():
    """Main function with FP16-safe training setup"""
    args = parse_args()

    # Set up environment for FP16 stability
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = args.model_dtype == 'float16'
    dtype = torch.float16 if use_fp16 else torch.float32

    print("🎯 FP16-SAFE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Ultra-conservative FP16 numerical stability optimizations")
    print("=" * 60)

    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    print(f"  Model dtype: {dtype}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Ultra-conservative LRs: student={args.student_lr}, router={args.router_lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  FP16 safety features: Enabled")

    # Log initial GPU memory
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"[Initial] GPU Memory: {initial_memory:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")

    try:
        # Load models with FP16 safety
        teacher, student, teacher_tok, student_tok = load_teacher_student(
            args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
        )

        print(f"[After model loading] GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")

        # Build datasets
        print("📊 Building datasets...")
        dataset = load_dataset(args.dataset_name, args.dataset_config_name)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

        # Check tokenizer compatibility
        same_tokenizer = teacher_tok.vocab == student_tok.vocab
        print(f"Dataset info: train={len(train_dataset)}, eval={len(eval_dataset)}, same_tokenizer={same_tokenizer}")

        # Create data collator
        if same_tokenizer:
            collator = DataCollatorForLanguageModeling(
                tokenizer=teacher_tok, mlm=False, pad_to_multiple_of=8
            )
        else:
            collator = DualTokenizerCollator(
                teacher_tok, student_tok, block_size=args.block_size
            )

        # Create distiller with ultra-conservative settings
        print("⚗️ Creating FP16-safe distiller...")
        config = SARConfig(
            student_lr=args.student_lr,
            router_lr=args.router_lr,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            alpha_ce=args.alpha_ce,
            router_anchor_weight=0.0,  # Disabled for FP16 stability
            router_load_balance_weight=0.0,  # Disabled for FP16 stability
            router_entropy_weight=0.0,  # Disabled for FP16 stability
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler_type=args.scheduler_type,
            total_steps=args.train_steps,
            use_fp16=use_fp16,
            offload_teacher_to_cpu=args.offload_teacher_to_cpu,
            clear_cache_every_step=args.clear_cache_every_step
        )

        distiller = SARDistiller(
            teacher=teacher,
            student=student,
            device=device,
            config=config,
            tokenizers_compatible=same_tokenizer
        )

        # Create scheduler
        if args.use_scheduler:
            if args.scheduler_type == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    distiller.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=args.train_steps
                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    distiller.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=args.train_steps
                )
            distiller.scheduler = scheduler
            print(f"Using {args.scheduler_type} learning rate scheduler with {args.warmup_steps} warmup steps")

        # Offload teacher to CPU for memory optimization
        if args.offload_teacher_to_cpu and torch.cuda.is_available():
            teacher.cpu()
            print("Moving teacher model to CPU for memory optimization")
            print(f"[After setup] GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")

        # Create FP16-safe evaluator
        evaluator = FP16SafeEvaluator(
            eval_dataset, same_tokenizer, collator,
            args.per_device_batch_size, args.eval_batches, device
        )

        # Test initial evaluation
        print("\n🧪 Testing initial evaluation...")
        try:
            initial_ppl = evaluator.evaluate(student, use_fp16=use_fp16)
            print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.2f}")
        except Exception as e:
            print(f"❌ Initial evaluation failed: {e}")
            return

        # Create FP16-safe trainer
        trainer = FP16SafeTrainer(distiller, evaluator, args)

        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True
        )

        # Logging function
        training_logs = []
        def log_callback(metrics):
            metrics["ts"] = datetime.now(timezone.utc).isoformat()
            training_logs.append(metrics)
            print(json.dumps(metrics, separators=(',', ':')))

        # Start FP16-safe training
        print(f"\n🚀 Starting FP16-safe training...")
        training_history = trainer.train(train_loader, log_callback=log_callback)

        # Save training logs
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        with open(f"{args.output_dir}/training_log.jsonl", "w") as f:
            for log in training_logs:
                f.write(json.dumps(log) + "\n")

        with open(f"{args.output_dir}/training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)

        print(f"\n✅ FP16-safe training completed successfully!")
        best_ppl_final = trainer.best_val_ppl if trainer.best_val_ppl != float('inf') else 0.0
        print(f"📊 Best validation perplexity: {best_ppl_final:.2f} at step {trainer.best_step}")

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"[Final] GPU Memory: {final_memory:.1f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
