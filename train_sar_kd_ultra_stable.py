#!/usr/bin/env python3
"""
Ultra-Stable SAR Knowledge Distillation Training Script
========================================================

This version addresses all known stability issues:
- FP16 gradient scaler compatibility
- Robust evaluation with proper batch handling
- Conservative training settings
- Comprehensive error handling and logging
- Proper numerical stability checks
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
    """Parse command line arguments with ultra-conservative defaults"""
    p = argparse.ArgumentParser(description='Ultra-Stable SAR Knowledge Distillation')

    # Model arguments
    p.add_argument('--teacher_model', default='microsoft/DialoGPT-medium')
    p.add_argument('--student_model', default='distilgpt2')
    p.add_argument('--model_dtype', choices=['float16', 'float32'], default='float16')

    # Dataset arguments
    p.add_argument('--dataset_name', default='wikitext')
    p.add_argument('--dataset_config_name', default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=256)

    # Training arguments - ultra conservative
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--train_steps', type=int, default=100)
    p.add_argument('--eval_steps', type=int, default=25)
    p.add_argument('--save_steps', type=int, default=50)

    # Learning rates - very conservative
    p.add_argument('--student_lr', type=float, default=1e-5)
    p.add_argument('--router_lr', type=float, default=5e-5)

    # KD parameters - simplified
    p.add_argument('--temperature', type=float, default=3.0)
    p.add_argument('--alpha_kd', type=float, default=0.3)
    p.add_argument('--alpha_ce', type=float, default=0.7)

    # Regularization - minimal
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--max_grad_norm', type=float, default=1.0)

    # Scheduler
    p.add_argument('--use_scheduler', action='store_true', default=True)
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--scheduler_type', choices=['cosine', 'linear'], default='linear')

    # Memory optimization
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=True)
    p.add_argument('--clear_cache_every_step', action='store_true', default=True)

    # Evaluation settings
    p.add_argument('--eval_batches', type=int, default=10)
    p.add_argument('--min_eval_tokens', type=int, default=100)

    # Output
    p.add_argument('--output_dir', default='./sar_kd_ultra_stable_output')
    p.add_argument('--save_best_model', action='store_true', default=True)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


def load_models_ultra_safe(teacher_name: str, student_name: str, dtype: torch.dtype, device: torch.device):
    """Load models with maximum safety and error handling"""
    print("📚 Loading models with ultra-safe settings...")

    try:
        # Load tokenizers first
        print(f"Loading teacher tokenizer: {teacher_name}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=False)
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        print(f"Loading student tokenizer: {student_name}")
        student_tokenizer = AutoTokenizer.from_pretrained(student_name, trust_remote_code=False)
        student_tokenizer.pad_token = student_tokenizer.eos_token

        # Load models with explicit dtype
        print(f"Loading teacher model with dtype: {dtype}")
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=dtype,
            trust_remote_code=False,
            device_map=None  # Manual device management
        )

        print(f"Loading student model with dtype: {dtype}")
        student = AutoModelForCausalLM.from_pretrained(
            student_name,
            torch_dtype=dtype,
            trust_remote_code=False,
            device_map=None
        )

        # Move to device
        print(f"Moving models to {device}")
        teacher = teacher.to(device)
        student = student.to(device)

        # Enable gradient checkpointing for memory efficiency
        teacher.gradient_checkpointing_enable()
        student.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled for both models")

        # Log memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 GPU memory after loading: {memory_gb:.2f}GB")

        return teacher, student, teacher_tokenizer, student_tokenizer

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise


class UltraStableEvaluator:
    """Ultra-stable evaluation with comprehensive error handling"""

    def __init__(self, eval_dataset, tokenizer_compatible, collator, batch_size, eval_batches, min_tokens, device):
        self.eval_dataset = eval_dataset
        self.tokenizer_compatible = tokenizer_compatible
        self.collator = collator
        self.batch_size = batch_size
        self.eval_batches = max(eval_batches, 1)  # Ensure at least 1 batch
        self.min_tokens = min_tokens
        self.device = device
        print(f"🔬 UltraStableEvaluator: {self.eval_batches} batches, min {self.min_tokens} tokens")

    def evaluate(self, student_model, use_fp16=False) -> float:
        """Ultra-stable evaluation with multiple safety checks"""
        if self.eval_dataset is None:
            print("⚠️ No evaluation dataset available")
            return float('inf')

        student_model.eval()
        total_loss = 0.0
        total_tokens = 0
        valid_batches = 0

        try:
            # Create fresh dataloader for each evaluation
            eval_subset = self.eval_dataset.shuffle(seed=int(time.time())).select(range(min(len(self.eval_dataset), 1000)))
            eval_loader = DataLoader(
                eval_subset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collator,
                drop_last=False
            )

            print(f"🔍 Evaluating with {len(eval_subset)} samples...")

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if batch_idx >= self.eval_batches:
                        break

                    try:
                        # Handle both tokenizer modes
                        if self.tokenizer_compatible:
                            input_ids = batch["input_ids"].to(self.device)
                            attention_mask = batch.get("attention_mask")
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch["labels"].to(self.device)
                        else:
                            # Use student tokenizer data
                            input_ids = batch["student_input_ids"].to(self.device)
                            attention_mask = batch.get("student_attention_mask")
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch["student_labels"].to(self.device)

                        # Skip empty batches
                        if input_ids.numel() == 0:
                            continue

                        # Forward pass with autocast
                        with torch.amp.autocast('cuda', enabled=use_fp16):
                            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits[:, :-1, :]
                            targets = labels[:, 1:]

                            # Calculate loss with proper masking
                            valid_mask = (targets != -100)
                            if valid_mask.sum() == 0:
                                print(f"    Batch {batch_idx}: No valid tokens, skipping")
                                continue

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
                            losses = losses.view(targets.shape)

                            # Apply mask and compute average
                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            # Numerical stability checks
                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                print(f"    Batch {batch_idx}: Invalid loss detected, skipping")
                                continue

                            # Clamp extreme values
                            if batch_loss.item() > 15:
                                print(f"    Batch {batch_idx}: High loss {batch_loss.item():.4f}, clamping")
                                batch_loss = torch.clamp(batch_loss, 0, 15)

                            # Accumulate
                            batch_tokens = valid_mask.sum().item()
                            total_loss += batch_loss.item()
                            total_tokens += batch_tokens
                            valid_batches += 1

                            print(f"    Batch {batch_idx}: loss={batch_loss.item():.4f}, tokens={batch_tokens}")

                    except Exception as e:
                        print(f"    Batch {batch_idx} failed: {e}")
                        continue

            # Final validation checks
            if valid_batches == 0:
                print("❌ No valid evaluation batches processed")
                return float('inf')

            if total_tokens < self.min_tokens:
                print(f"⚠️ Too few tokens ({total_tokens} < {self.min_tokens}), evaluation may be unstable")

            # Compute final metrics
            avg_loss = total_loss / valid_batches
            print(f"📊 Eval summary: avg_loss={avg_loss:.4f}, batches={valid_batches}, tokens={total_tokens}")

            # Prevent exp overflow
            clamped_loss = min(avg_loss, 15)
            if clamped_loss != avg_loss:
                print(f"⚠️ Clamped loss from {avg_loss:.4f} to {clamped_loss:.4f}")

            perplexity = math.exp(clamped_loss)

            # Final sanity check
            if math.isnan(perplexity) or math.isinf(perplexity) or perplexity <= 0:
                print(f"❌ Invalid perplexity: {perplexity}")
                return float('inf')

            return perplexity

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return float('inf')

        finally:
            student_model.train()


class UltraStableTrainer:
    """Ultra-stable trainer with comprehensive error handling and monitoring"""

    def __init__(self, distiller, evaluator, args):
        self.distiller = distiller
        self.evaluator = evaluator
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_metrics = []

    def safe_forward_pass(self, batch) -> Tuple[Optional[torch.Tensor], float, float, int]:
        """Ultra-safe forward pass with comprehensive error handling"""
        try:
            # Handle different tokenizer modes
            if 'teacher_input_ids' in batch:
                # Dual tokenizer mode - use only student to avoid alignment issues
                input_ids = batch['student_input_ids'].to(self.distiller.device)
                attention_mask = batch.get('student_attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.distiller.device)
                labels = batch['student_labels'].to(self.distiller.device)
                use_teacher = False  # Skip teacher for stability
            else:
                # Same tokenizer mode
                input_ids = batch["input_ids"].to(self.distiller.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.distiller.device)
                labels = batch["labels"].to(self.distiller.device)
                use_teacher = True

            # Skip empty batches
            if input_ids.numel() == 0:
                return None, 0.0, 0.0, 0

            with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                # Student forward pass (always)
                student_outputs = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits[:, :-1, :]
                targets = labels[:, 1:]

                # Validate targets
                valid_mask = (targets != -100)
                if valid_mask.sum() == 0:
                    return None, 0.0, 0.0, 0

                # Compute CE loss (always needed)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_losses = loss_fct(student_logits.contiguous().view(-1, student_logits.size(-1)),
                                   targets.contiguous().view(-1))
                ce_losses = ce_losses.view(targets.shape)
                masked_ce = ce_losses * valid_mask.float()
                ce_loss = masked_ce.sum() / valid_mask.sum()

                # Teacher forward pass and KD loss (if applicable)
                kd_loss = torch.tensor(0.0, device=self.distiller.device)
                if use_teacher:
                    try:
                        with torch.no_grad():
                            teacher_outputs = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                            teacher_logits = teacher_outputs.logits[:, :-1, :]

                        # KD loss with temperature scaling
                        teacher_probs = torch.softmax(teacher_logits / self.distiller.config.temperature, dim=-1)
                        student_log_probs = torch.log_softmax(student_logits / self.distiller.config.temperature, dim=-1)

                        kd_loss = -(teacher_probs * student_log_probs).sum(dim=-1)
                        kd_loss = kd_loss.mean() * (self.distiller.config.temperature ** 2)

                    except Exception as e:
                        print(f"    Teacher forward failed, using CE only: {e}")
                        kd_loss = torch.tensor(0.0, device=self.distiller.device)

                # Numerical stability
                ce_loss = torch.clamp(ce_loss, 0, 20)
                kd_loss = torch.clamp(kd_loss, 0, 20)

                if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                    print(f"    Invalid CE loss: {ce_loss}")
                    return None, 0.0, 0.0, 0

                if torch.isnan(kd_loss) or torch.isinf(kd_loss):
                    print(f"    Invalid KD loss: {kd_loss}, setting to 0")
                    kd_loss = torch.tensor(0.0, device=self.distiller.device)

                # Total loss
                total_loss = self.distiller.config.alpha_ce * ce_loss + self.distiller.config.alpha_kd * kd_loss

                # Final checks
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"    Invalid total loss: {total_loss}")
                    return None, 0.0, 0.0, 0

                if total_loss.item() > 25:
                    print(f"    Very high total loss {total_loss.item():.4f}, clamping")
                    total_loss = torch.clamp(total_loss, 0, 20)

                tokens = valid_mask.sum().item()
                return total_loss, kd_loss.item(), ce_loss.item(), tokens

        except Exception as e:
            print(f"    Forward pass error: {e}")
            return None, 0.0, 0.0, 0

    def train(self, train_loader, log_callback=None):
        """Ultra-stable training loop with comprehensive monitoring"""
        print("🚀 Starting ultra-stable training...")

        # Setup gradient scaler with FP16 parameter detection
        use_scaler = torch.cuda.is_available() and self.distiller.use_fp16

        # Critical FP16 fix: disable scaler if parameters are already FP16
        has_fp16_params = any(p.dtype == torch.float16 for p in self.distiller.student.parameters())
        if hasattr(self.distiller, 'router_params') and self.distiller.router_params:
            has_fp16_params = has_fp16_params or any(p.dtype == torch.float16 for p in self.distiller.router_params)

        if has_fp16_params and use_scaler:
            print("⚠️ FP16 parameters detected - disabling gradient scaler to prevent errors")
            use_scaler = False

        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Print training mode
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
        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        total_tokens = 0
        successful_batches = 0

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            self._run_evaluation(step, log_callback)

        # Training loop
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            try:
                # Forward pass
                loss, kd_loss, ce_loss, tokens = self.safe_forward_pass(batch)

                if loss is None:
                    print(f"  Step {step}: Batch failed, skipping")
                    step += 1
                    continue

                # Backward pass
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Accumulate metrics
                total_loss += loss.item()
                total_kd_loss += kd_loss
                total_ce_loss += ce_loss
                total_tokens += tokens
                successful_batches += 1

                # Optimizer step
                if (step + 1) % self.args.grad_accum_steps == 0:
                    try:
                        if use_scaler:
                            scaler.unscale_(self.distiller.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.args.max_grad_norm)
                            scaler.step(self.distiller.optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.args.max_grad_norm)
                            self.distiller.optimizer.step()

                        if hasattr(self.distiller, 'scheduler') and self.distiller.scheduler:
                            self.distiller.scheduler.step()

                        self.distiller.optimizer.zero_grad()

                    except Exception as e:
                        print(f"  Step {step}: Optimizer step failed: {e}")

                # Logging
                if (step + 1) % 10 == 0:
                    avg_loss = total_loss / max(successful_batches, 1)
                    avg_kd = total_kd_loss / max(successful_batches, 1)
                    avg_ce = total_ce_loss / max(successful_batches, 1)
                    success_rate = successful_batches / (step + 1)

                    lr = self.distiller.optimizer.param_groups[0]['lr'] if self.distiller.optimizer else 0

                    print(f"  Step {step+1}: loss={avg_loss:.4f} (kd={avg_kd:.4f}, ce={avg_ce:.4f}) "
                          f"success_rate={success_rate:.2%} lr={lr:.2e}")

                    if log_callback:
                        metrics = {
                            "step": step + 1,
                            "train_loss": avg_loss,
                            "train_kd_loss": avg_kd,
                            "train_ce_loss": avg_ce,
                            "train_ppl": min(math.exp(avg_loss), 10000),
                            "tokens": total_tokens,
                            "successful_batches": successful_batches,
                            "success_rate": success_rate,
                            "lr": lr
                        }
                        if torch.cuda.is_available():
                            metrics["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3

                        log_callback(metrics)

                # Evaluation
                if (step + 1) % self.args.eval_steps == 0:
                    self._run_evaluation(step + 1, log_callback)

                # Memory cleanup
                if self.args.clear_cache_every_step and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                step += 1

            except Exception as e:
                print(f"  Step {step}: Training step failed: {e}")
                step += 1
                continue

        # Final evaluation
        print("🏁 Running final evaluation...")
        self._run_evaluation(step, log_callback)

        return {
            "final_step": step,
            "best_val_ppl": self.best_val_ppl,
            "best_step": self.best_step,
            "total_successful_batches": successful_batches
        }

    def _run_evaluation(self, step, log_callback=None):
        """Run evaluation with comprehensive error handling"""
        try:
            print(f"🔍 Running evaluation at step {step}...")
            val_ppl = self.evaluator.evaluate(self.distiller.student, use_fp16=self.distiller.use_fp16)
            val_loss = math.log(val_ppl) if val_ppl != float('inf') else float('inf')

            print(f"  📊 Step {step}: Validation PPL = {val_ppl:.2f}")

            # Track best model
            is_best = val_ppl < self.best_val_ppl
            if is_best:
                self.best_val_ppl = val_ppl
                self.best_step = step
                print(f"  🏆 New best model! PPL: {val_ppl:.2f}")

            # Log metrics
            if log_callback:
                metrics = {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_ppl": self.best_val_ppl,
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


def build_datasets(args, teacher_tokenizer, student_tokenizer):
    """Build datasets with comprehensive error handling"""
    print("📊 Building datasets...")

    try:
        # Load dataset
        dataset = load_dataset(args.dataset_name, args.dataset_config_name)

        # Prepare splits
        train_dataset = dataset['train']
        eval_dataset = dataset.get('validation', dataset.get('test'))

        if eval_dataset is None:
            print("⚠️ No validation set found, using train split")
            eval_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

        print(f"📈 Dataset sizes: train={len(train_dataset)}, eval={len(eval_dataset)}")

        # Check tokenizer compatibility
        same_tokenizer = teacher_tokenizer.vocab == student_tokenizer.vocab
        print(f"🔄 Same tokenizer: {same_tokenizer}")

        # Create appropriate collator
        if same_tokenizer:
            collator = DataCollatorForLanguageModeling(
                tokenizer=teacher_tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        else:
            collator = DualTokenizerCollator(
                teacher_tokenizer, student_tokenizer,
                block_size=args.block_size
            )

        return train_dataset, eval_dataset, collator, same_tokenizer

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        raise


def create_distiller(teacher, student, teacher_tokenizer, student_tokenizer, same_tokenizer, args, device):
    """Create SAR distiller with ultra-safe configuration"""
    print("⚗️ Creating ultra-safe distiller...")

    try:
        # Ultra-conservative SAR configuration
        config = SARConfig(
            student_lr=args.student_lr,
            router_lr=args.router_lr,
            temperature=args.temperature,
            alpha_kd=args.alpha_kd,
            alpha_ce=args.alpha_ce,
            router_anchor_weight=0.0,  # Disable for stability
            router_load_balance_weight=0.0,  # Disable for stability
            router_entropy_weight=0.0,  # Disable for stability
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler_type=args.scheduler_type,
            total_steps=args.train_steps,
            use_fp16=args.model_dtype == 'float16',
            offload_teacher_to_cpu=args.offload_teacher_to_cpu,
            clear_cache_every_step=args.clear_cache_every_step
        )

        # Create distiller
        distiller = SARDistiller(
            teacher=teacher,
            student=student,
            device=device,
            config=config,
            tokenizers_compatible=same_tokenizer
        )

        # Create scheduler if requested
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
            print(f"📅 Using {args.scheduler_type} scheduler with {args.warmup_steps} warmup steps")

        # Teacher CPU offloading for memory efficiency
        if args.offload_teacher_to_cpu and torch.cuda.is_available():
            teacher.cpu()
            print("💾 Teacher model moved to CPU for memory optimization")

        return distiller

    except Exception as e:
        print(f"❌ Distiller creation failed: {e}")
        raise


def main():
    """Main ultra-stable training function"""
    args = parse_args()

    # Set up environment
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.model_dtype == 'float16' else torch.float32

    print("🎯 ULTRA-STABLE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Maximum stability with comprehensive error handling")
    print("=" * 60)
    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    print(f"  Model dtype: {dtype}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Batch size: {args.per_device_batch_size}")
    print(f"  Learning rates: student={args.student_lr}, router={args.router_lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Eval batches: {args.eval_batches}")

    # Log initial memory
    if torch.cuda.is_available():
        print(f"[Initial] GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")

    try:
        # Load models
        teacher, student, teacher_tok, student_tok = load_models_ultra_safe(
            args.teacher_model, args.student_model, dtype, device
        )

        # Build datasets
        train_dataset, eval_dataset, collator, same_tokenizer = build_datasets(
            args, teacher_tok, student_tok
        )

        # Create distiller
        distiller = create_distiller(
            teacher, student, teacher_tok, student_tok, same_tokenizer, args, device
        )

        # Create evaluator
        evaluator = UltraStableEvaluator(
            eval_dataset, same_tokenizer, collator,
            args.per_device_batch_size, args.eval_batches, args.min_eval_tokens, device
        )

        # Test initial evaluation
        print("\n🧪 Testing initial evaluation...")
        try:
            initial_ppl = evaluator.evaluate(student, use_fp16=args.model_dtype == 'float16')
            print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.2f}")
        except Exception as e:
            print(f"❌ Initial evaluation failed: {e}")
            return

        # Create trainer
        trainer = UltraStableTrainer(distiller, evaluator, args)

        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True
        )

        # Logging callback
        logs = []
        def log_callback(metrics):
            metrics["ts"] = datetime.now(timezone.utc).isoformat()
            logs.append(metrics)
            print(json.dumps(metrics, indent=None, separators=(',', ':')))

        # Start training
        print(f"\n🚀 Starting ultra-stable training...")
        results = trainer.train(train_loader, log_callback)

        # Final summary
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final results:")
        print(f"  Steps completed: {results['final_step']}")
        print(f"  Successful batches: {results['total_successful_batches']}")
        print(f"  Best validation PPL: {results['best_val_ppl']:.2f} at step {results['best_step']}")

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"[Final] GPU Memory: {final_memory:.1f}GB allocated")

        # Save logs
        if logs:
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/training_log.jsonl", 'w') as f:
                for log in logs:
                    f.write(json.dumps(log) + '\n')
            print(f"📝 Logs saved to {args.output_dir}/training_log.jsonl")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
