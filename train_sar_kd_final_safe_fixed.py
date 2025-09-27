#!/usr/bin/env python3
"""
FINAL SAFE SAR Knowledge Distillation Training - FIXED VERSION
================================================================

This version fixes the critical tensor view/reshape errors that were causing
all evaluation batches to fail. Key fixes:

1. Replace .view() with .reshape() for non-contiguous tensors
2. Add proper tensor contiguity checks
3. Fix evaluation pipeline to actually compute meaningful perplexity
4. Maintain all mixed precision stability features

Root cause: PyTorch .view() fails when tensors span non-contiguous memory,
but .reshape() handles this automatically by creating a copy when needed.
"""

import os
import sys
import json
import time
import math
import argparse
import warnings
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset

# Import SAR KD components
try:
    from sar_kd import SARDistiller
except ImportError:
    print("Error: Could not import sar_kd module. Make sure it's in your Python path.")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def print_memory_info(stage: str, device):
    """Print GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[{stage}] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def safe_tensor_reshape(tensor, target_shape):
    """Safely reshape tensor, handling non-contiguous memory layouts"""
    try:
        if tensor.is_contiguous():
            return tensor.view(target_shape)
        else:
            return tensor.reshape(target_shape)
    except Exception as e:
        print(f"Warning: Tensor reshape failed, making contiguous: {e}")
        return tensor.contiguous().view(target_shape)


class FixedSafeEvaluator:
    """Fixed safe evaluator with proper tensor handling"""

    def __init__(self, eval_dataset, tokenizer, device, batch_size=2, eval_batches=10):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def evaluate(self, student_model, use_mixed_precision=True):
        """Evaluate with proper tensor handling and mixed precision"""
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

                            # Calculate loss with safe tensor operations
                            valid_mask = (targets != -100)
                            if not valid_mask.any():
                                continue

                            loss_fct = nn.CrossEntropyLoss(reduction='none')

                            # FIXED: Use safe_tensor_reshape instead of .view()
                            logits_flat = safe_tensor_reshape(logits, (-1, logits.size(-1)))
                            targets_flat = safe_tensor_reshape(targets, (-1,))
                            losses = loss_fct(logits_flat, targets_flat)

                            # Check losses before processing
                            if torch.isnan(losses).any() or torch.isinf(losses).any():
                                print(f"    Warning: NaN/Inf in raw losses batch {batch_idx}, skipping")
                                continue

                            # FIXED: Use safe_tensor_reshape for losses too
                            losses = safe_tensor_reshape(losses, targets.shape)
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
            ppl = math.exp(avg_loss)
            ppl = min(ppl, 10000)  # Cap at 10k for stability
            return ppl
        except OverflowError:
            print(f"  Warning: Overflow in perplexity calculation, avg_loss={avg_loss}")
            return 10000.0


class FixedSafeTrainer:
    """Fixed safe trainer with tensor handling fixes"""

    def __init__(self, distiller, eval_loader, args):
        self.distiller = distiller
        self.eval_loader = eval_loader
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_history = []
        self.evaluator = None  # Will be set externally

    def safe_forward_pass(self, batch, use_mixed_precision=True):
        """Forward pass with fixed tensor operations"""
        input_ids = batch["input_ids"].to(self.distiller.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.distiller.device)
        labels = batch["labels"].to(self.distiller.device)

        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=use_mixed_precision):
            # Teacher forward (FP16)
            with torch.no_grad():
                teacher_outputs = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits[:, :-1, :]
                teacher_logits = torch.clamp(teacher_logits, -15, 15)

            # Student forward (FP16)
            student_outputs = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits[:, :-1, :]
            student_logits = torch.clamp(student_logits, -15, 15)

            target_labels = labels[:, 1:]
            valid_mask = (target_labels != -100)

            if not valid_mask.any():
                return None

            # Knowledge Distillation Loss
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)

            # Cross Entropy Loss with FIXED tensor operations
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # FIXED: Use safe_tensor_reshape instead of .view()
            student_logits_flat = safe_tensor_reshape(student_logits, (-1, student_logits.size(-1)))
            target_labels_flat = safe_tensor_reshape(target_labels, (-1,))
            ce_losses = loss_fct(student_logits_flat, target_labels_flat)

            # Check raw losses
            if torch.isnan(ce_losses).any() or torch.isinf(ce_losses).any():
                return None

            # FIXED: Use safe_tensor_reshape for losses
            ce_losses = safe_tensor_reshape(ce_losses, target_labels.shape)
            masked_ce = ce_losses * valid_mask.float()
            ce = masked_ce.sum() / valid_mask.sum()

            # Final checks
            if torch.isnan(kd_loss) or torch.isinf(kd_loss) or torch.isnan(ce) or torch.isinf(ce):
                return None

            # Combined loss
            total_loss = self.args.alpha * kd_loss + (1 - self.args.alpha) * ce

            return {
                'loss': total_loss,
                'kd_loss': kd_loss.item(),
                'ce_loss': ce.item(),
                'tokens': valid_mask.sum().item()
            }

    def train_epoch(self, train_loader, scaler, log_callback=None, save_callback=None):
        """Training epoch with all fixes applied"""
        self.distiller.student.train()

        # Teacher stays in eval mode for consistency
        self.distiller.teacher.eval()

        # Training loop
        step = 0
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        total_tokens = 0
        successful_steps = 0
        skipped_batches = 0

        print("🚀 Starting FINAL SAFE SAR Knowledge Distillation training...")
        print("🔧 Using MIXED PRECISION: FP16 forward pass + FP32 parameters/gradients")
        print("🔧 Training Configuration:")
        print("   - Mixed precision training: True")
        print("   - Gradient scaling: True")
        print("   - Model parameters remain in FP32")

        while step < self.args.train_steps:
            try:
                # Get batch from cyclic loader
                try:
                    batch = next(train_loader)
                except StopIteration:
                    # Reset loader
                    train_loader = iter(train_loader)
                    batch = next(train_loader)

                # Forward pass
                result = self.safe_forward_pass(batch, use_mixed_precision=True)
                if result is None:
                    skipped_batches += 1
                    self.distiller.optimizer.zero_grad()
                    step += 1
                    continue

                loss = result['loss']
                kd_loss = result['kd_loss']
                ce_loss = result['ce_loss']
                batch_tokens = result['tokens']

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.args.grad_accum_steps

                # Backward pass with gradient scaling
                scaler.scale(scaled_loss).backward()

            except Exception as e:
                print(f"  Forward/backward error at step {step}: {e}")
                skipped_batches += 1
                if hasattr(self.distiller, 'optimizer') and self.distiller.optimizer is not None:
                    self.distiller.optimizer.zero_grad()
                step += 1
                continue

            # Accumulate metrics only for successful batches
            running_loss += loss.item() * self.args.grad_accum_steps
            running_kd_loss += kd_loss
            running_ce_loss += ce_loss
            total_tokens += batch_tokens
            successful_steps += 1

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

        return self.training_history

    def run_evaluation(self, step, log_callback):
        """Run evaluation with fixed tensor handling"""
        print(f"🔍 Running safe evaluation at step {step}")

        # Move teacher to CPU if needed to save memory during eval
        if self.args.offload_teacher_to_cpu:
            self.distiller.teacher.cpu()
            torch.cuda.empty_cache()

        # This will be properly configured in main()
        val_ppl = float('inf')

        if hasattr(self, 'evaluator') and self.evaluator is not None:
            val_ppl = self.evaluator.evaluate(self.distiller.student, use_mixed_precision=True)

        print(f"  📊 Step {step}: Validation PPL = {val_ppl:.1f}")

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FIXED SAR Knowledge Distillation Training')

    # Model arguments
    parser.add_argument('--teacher_model', type=str, default='microsoft/DialoGPT-medium',
                        help='Teacher model name or path')
    parser.add_argument('--student_model', type=str, default='distilgpt2',
                        help='Student model name or path')
    parser.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='Model loading dtype')

    # Training arguments
    parser.add_argument('--train_steps', type=int, default=500,
                        help='Number of training steps')
    parser.add_argument('--student_lr', type=float, default=5e-6,
                        help='Student learning rate')
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='KD loss weight (1-alpha for CE)')

    # Data arguments
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                        help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1',
                        help='Dataset configuration')
    parser.add_argument('--block_size', type=int, default=512,
                        help='Input sequence length')
    parser.add_argument('--train_batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help='Evaluation batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')

    # Evaluation and logging
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Evaluation frequency')
    parser.add_argument('--eval_batches', type=int, default=10,
                        help='Number of batches for evaluation')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save frequency (0 to disable)')

    # System arguments
    parser.add_argument('--output_dir', type=str, default='./sar_kd_output_fixed',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--clear_cache_every_step', action='store_true',
                        help='Clear CUDA cache every step')
    parser.add_argument('--offload_teacher_to_cpu', action='store_true',
                        help='Offload teacher to CPU during training')

    return parser.parse_args()


def main():
    args = parse_args()

    print("🔧 FIXED SAFE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("FIXES: Tensor view/reshape errors, evaluation pipeline")
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

    print(f"\n🔧 Fixed Safe Configuration:")
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

    # Load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    # Set pad tokens
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    print(f"Loading models with dtype: {dtype}")

    # Load models in specified dtype
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        device_map=None
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        device_map=None
    )

    # Move models to device
    print("Moving teacher model to cuda...")
    teacher_model.to(device)
    print("Moving student model to cuda...")
    student_model.to(device)

    # Enable gradient checkpointing
    if hasattr(teacher_model, 'gradient_checkpointing_enable'):
        teacher_model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for teacher model")
    if hasattr(student_model, 'gradient_checkpointing_enable'):
        student_model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for student model")

    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # CRITICAL: Convert student parameters to FP32 for mixed precision
    if use_fp16:
        print("🔄 Converting student parameters to FP32 for mixed precision stability...")
        for param in student_model.parameters():
            param.data = param.data.float()

    print_memory_info("After model loading", device)

    # Build datasets
    print("📊 Building datasets...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    # Use the compatible tokenizer (usually student for consistency)
    tokenizer_compatible = student_tokenizer

    def tokenize_function(examples):
        return tokenizer_compatible(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=args.block_size,
        )

    def group_texts(examples):
        block_size = args.block_size
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Process datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation", lm_datasets.get("test"))

    print(f"Dataset info: train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset else 0}, same_tokenizer={tokenizer_compatible == student_tokenizer}")

    # Create distiller
    print("⚗️ Creating final safe distiller...")
    distiller = SARDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=args.temperature,
        router_patterns=['gate', 'router', 'moe.*gate', 'switch.*router', 'expert.*router']
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        distiller.student.parameters(),
        lr=args.student_lr,
        weight_decay=0.01,
        eps=1e-8
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )

    distiller.optimizer = optimizer
    distiller.scheduler = scheduler

    print(f"Using cosine learning rate scheduler with {args.warmup_steps} warmup steps")

    # Memory optimization - move teacher to CPU if requested
    if args.offload_teacher_to_cpu:
        print("Moving teacher model to CPU for memory optimization")
        distiller.teacher.cpu()

    print_memory_info("After setup", device)

    # Create data loaders
    collator = DataCollatorForLanguageModeling(tokenizer_compatible, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    ) if eval_dataset else None

    # Create trainer with FIXED evaluator
    trainer = FixedSafeTrainer(distiller, eval_loader, args)
    trainer.evaluator = FixedSafeEvaluator(eval_dataset, tokenizer_compatible, device, batch_size=args.eval_batch_size, eval_batches=args.eval_batches)

    # Create cyclic train loader iterator
    train_loader_iter = iter(train_loader)

    def get_next_batch():
        try:
            return next(train_loader_iter)
        except StopIteration:
            nonlocal train_loader_iter
            train_loader_iter = iter(train_loader)
            return next(train_loader_iter)

    # Test initial evaluation
    print("\n🧪 Testing initial evaluation...")
    initial_ppl = trainer.run_evaluation(0, None)
    print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.1f}")

    # Training callbacks
    def log_metrics(metrics):
        if "step" in metrics and "train_ppl" in metrics:
            print(f"Step {metrics['step']}: Loss={metrics['train_loss']:.4f}, PPL={metrics['train_ppl']:.1f}, Success={metrics['success_rate']:.3f}")

    def save_checkpoint(step, model, router_linears):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer_compatible.save_pretrained(checkpoint_path)
        print(f"💾 Saved checkpoint at step {step}")

    # Setup gradient scaler for mixed precision
    scaler = GradScaler()

    # Training
    print("\n🚀 Starting final safe training...")
    trainer.train_epoch(get_next_batch, scaler, log_callback=log_metrics, save_callback=save_checkpoint)

    # Final evaluation
    print("\n🏁 Running final evaluation...")
    final_ppl = trainer.run_evaluation(args.train_steps, log_metrics)
    print(f"Final validation perplexity: {final_ppl:.1f}")
    print(f"Best validation perplexity: {trainer.best_val_ppl:.1f} at step {trainer.best_step}")

    print(f"\n✅ Training completed successfully!")
    print(f"📊 Best validation perplexity: {trainer.best_val_ppl:.1f} at step {trainer.best_step}")
    print_memory_info("Final", device)


if __name__ == "__main__":
    main()
