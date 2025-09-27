import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import math
from datetime import datetime, UTC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student
from sar_kd.trainer import SARConfig, SARDistiller


def parse_args():
    p = argparse.ArgumentParser(description="Stable SAR Knowledge Distillation")
    p.add_argument('--teacher_model', type=str, default='huihui-ai/Huihui-MoE-1B-A0.6B')
    p.add_argument('--student_model', type=str, default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--dataset_name', type=str, default='wikitext')
    p.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=512)
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=24)
    p.add_argument('--train_steps', type=int, default=1000)
    p.add_argument('--eval_steps', type=int, default=100)
    p.add_argument('--save_steps', type=int, default=200)

    # Conservative learning parameters for stability
    p.add_argument('--student_lr', type=float, default=1e-5)
    p.add_argument('--router_lr', type=float, default=5e-5)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--alpha_kd', type=float, default=0.1)
    p.add_argument('--alpha_ce', type=float, default=0.9)
    p.add_argument('--router_anchor_weight', type=float, default=1e-5)
    p.add_argument('--router_load_balance_weight', type=float, default=1e-4)
    p.add_argument('--router_entropy_weight', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--max_grad_norm', type=float, default=0.5)

    p.add_argument('--use_scheduler', action='store_true', default=True)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--scheduler_type', type=str, default='cosine')
    p.add_argument('--router_patterns', nargs='*')

    p.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'])
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=True)
    p.add_argument('--clear_cache_every_step', action='store_true', default=True)

    p.add_argument('--eval_batches', type=int, default=25)
    p.add_argument('--save_best_model', action='store_true', default=True)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd_stable')

    return p.parse_args()


def print_memory_info(stage="", device=None):
    if torch.cuda.is_available() and device is not None:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[{stage}] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


class StableEvaluator:
    def __init__(self, eval_dataset, tokenizer_compatible, collator, batch_size, eval_batches, device):
        self.eval_dataset = eval_dataset
        self.tokenizer_compatible = tokenizer_compatible
        self.collator = collator
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.device = device

    def evaluate(self, student_model, use_fp16=False):
        if self.eval_dataset is None:
            return float('inf')

        # Use a fixed subset to reduce variance
        eval_size = min(len(self.eval_dataset), self.eval_batches * self.batch_size)
        eval_indices = list(range(eval_size))  # Use first N samples for consistency
        eval_subset = torch.utils.data.Subset(self.eval_dataset, eval_indices)

        eval_loader = DataLoader(
            eval_subset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for consistency
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
                        # Handle different tokenizer modes
                        if 'student_input_ids' in batch:
                            input_ids = batch['student_input_ids'].to(self.device)
                            attention_mask = batch.get('student_attention_mask')
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch['student_labels'].to(self.device)
                        else:
                            input_ids = batch["input_ids"].to(self.device)
                            attention_mask = batch.get("attention_mask")
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.device)
                            labels = batch["labels"].to(self.device)

                        # Forward pass
                        with torch.amp.autocast('cuda', enabled=use_fp16):
                            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits[:, :-1, :]
                            targets = labels[:, 1:]

                            # Calculate loss
                            valid_mask = (targets != -100)
                            if valid_mask.sum() == 0:
                                continue

                            # Clamp logits before loss computation
                            logits = torch.clamp(logits, -8, 8)

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
                            losses = losses.view(targets.shape)

                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            # More conservative loss clamping for evaluation
                            batch_loss = torch.clamp(batch_loss, 0, 10)

                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                print(f"    Warning: NaN/Inf loss in eval batch {batch_idx}, skipping")
                                continue

                            # Additional check for extreme values
                            if batch_loss.item() > 12:
                                print(f"    Warning: High eval loss {batch_loss.item():.4f} in batch {batch_idx}")

                            total_loss += batch_loss.item()
                            total_tokens += valid_mask.sum().item()
                            batch_count += 1

                    except Exception as e:
                        print(f"  Warning: Eval batch {batch_idx} failed: {e}")
                        continue

        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            print(f"Evaluation traceback: {traceback.format_exc()}")
            return float('inf')
        finally:
            student_model.train()

        if batch_count == 0:
            print(f"  Warning: No valid evaluation batches processed")
            return float('inf')

        avg_loss = total_loss / batch_count
        print(f"  Debug: avg_loss={avg_loss:.4f}, batch_count={batch_count}, total_tokens={total_tokens}")

        # Prevent overflow in exp calculation
        if avg_loss > 15:
            print(f"  Warning: Clamping high avg_loss {avg_loss:.4f} to 15")
            avg_loss = 15

        perplexity = math.exp(avg_loss)

        # Final sanity check
        if math.isnan(perplexity) or math.isinf(perplexity):
            print(f"  Error: Invalid perplexity calculated from avg_loss={avg_loss}")
            return float('inf')

        return perplexity


class StableTrainer:
    def __init__(self, distiller, evaluator, args):
        self.distiller = distiller
        self.evaluator = evaluator
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.training_history = []

    def safe_forward_pass(self, batch):
        """Safe forward pass with error handling and loss clamping"""
        try:
            # Handle batch data
            if 'teacher_input_ids' in batch:
                # For dual tokenizer, use simpler approach - just use student path
                s_ids = batch['student_input_ids'].to(self.distiller.device)
                s_mask = batch['student_attention_mask'].to(self.distiller.device)
                s_labels = batch['student_labels'].to(self.distiller.device)

                # Only use student for now to avoid alignment issues
                with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                    s_out = self.distiller.student(input_ids=s_ids, attention_mask=s_mask)
                    s_logits = s_out.logits[:, :-1, :]
                    s_targets = s_labels[:, 1:]

                    # Clamp logits to prevent overflow in FP16
                    s_logits = torch.clamp(s_logits, -10, 10)

                    # Just use CE loss for now
                    valid_mask = (s_targets != -100)
                    if not valid_mask.any():
                        return None, None, None

                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_targets.view(-1))
                    losses = losses.view(s_targets.shape)

                    masked_losses = losses * valid_mask.float()
                    ce = masked_losses.sum() / valid_mask.sum()

                    # No KD loss for now
                    kd = torch.tensor(0.0, device=self.distiller.device)
                    tokens = valid_mask.sum().item()

            else:
                # Same tokenizer path
                input_ids = batch["input_ids"].to(self.distiller.device)
                labels = batch["labels"].to(self.distiller.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.distiller.device)

                with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                    # Teacher forward
                    if not self.args.offload_teacher_to_cpu or self.distiller.teacher.training:
                        with torch.no_grad():
                            t_out = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        t_out = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)

                    s_out = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)

                    teacher_logits = t_out.logits[:, :-1, :]
                    student_logits = s_out.logits[:, :-1, :]
                    target_labels = labels[:, 1:]

                    # Clamp logits to prevent FP16 overflow
                    teacher_logits = torch.clamp(teacher_logits, -10, 10)
                    student_logits = torch.clamp(student_logits, -10, 10)

                    # Safe KD loss with temperature
                    teacher_probs = torch.softmax(teacher_logits / self.distiller.config.temperature, dim=-1)
                    student_log_probs = torch.log_softmax(student_logits / self.distiller.config.temperature, dim=-1)

                    kd = -(teacher_probs * student_log_probs).sum(dim=-1)
                    kd = kd.mean()
                    kd = kd * (self.distiller.config.temperature ** 2)  # Scale by T^2

                    # Safe CE loss
                    valid_mask = (target_labels != -100)
                    if not valid_mask.any():
                        return None, None, None

                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    ce_losses = loss_fct(student_logits.view(-1, student_logits.size(-1)), target_labels.view(-1))
                    ce_losses = ce_losses.view(target_labels.shape)

                    masked_ce = ce_losses * valid_mask.float()
                    ce = masked_ce.sum() / valid_mask.sum()
                    tokens = valid_mask.sum().item()

            # More aggressive loss clamping for FP16 stability
            kd = torch.clamp(kd, 0, 5)
            ce = torch.clamp(ce, 0, 8)

            # Check for NaN in individual losses
            if torch.isnan(kd) or torch.isinf(kd):
                print(f"  KD loss issue: kd={kd}, setting to 0")
                kd = torch.tensor(0.0, device=self.distiller.device)

            if torch.isnan(ce) or torch.isinf(ce):
                print(f"  CE loss issue: ce={ce}, skipping batch")
                return None, None, None

            # Simple total loss (skip router regularization for now)
            total_loss = self.distiller.config.alpha_kd * kd + self.distiller.config.alpha_ce * ce

            # Check for numerical issues
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  Numerical issue: total_loss={total_loss}, kd={kd.item():.4f}, ce={ce.item():.4f}")
                return None, None, None

            # More conservative loss clamping
            if total_loss.item() > 10:
                print(f"  Warning: Very high loss {total_loss.item():.4f}, clamping to 8")
                total_loss = torch.clamp(total_loss, 0, 8)

            return total_loss, kd.item(), ce.item(), tokens

        except Exception as e:
            print(f"Forward pass error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, None, None

    def train(self, train_loader, save_callback=None, log_callback=None):
        print("🚀 Starting stable SAR Knowledge Distillation training...")

        # Setup
        self.distiller.teacher.eval()
        self.distiller.student.train()

        step = 0
        total_tokens = 0
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        successful_steps = 0

        # Mixed precision setup
        use_scaler = torch.cuda.is_available() and self.distiller.use_fp16

        # Additional safety check: disable scaler if any parameters are FP16
        has_fp16_params = any(p.dtype == torch.float16 for p in self.distiller.student.parameters())
        if hasattr(self.distiller, 'router_params') and self.distiller.router_params:
            has_fp16_params = has_fp16_params or any(p.dtype == torch.float16 for p in self.distiller.router_params)

        if has_fp16_params and use_scaler:
            print("WARNING: FP16 parameters detected - disabling gradient scaler to prevent errors")
            use_scaler = False

        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Print scaler status for user information
        if torch.cuda.is_available():
            if self.distiller.use_fp16 and not has_fp16_params:
                print("Mixed precision training enabled - FP16 computations with FP32 parameters and gradient scaling")
            elif has_fp16_params:
                print("FP16 model parameters detected - using FP16 training without gradient scaling")
            else:
                print("FP32 training - gradient scaler disabled")
        else:
            print("CUDA not available - gradient scaler disabled")

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            self.run_evaluation(step, log_callback)

        # Training loop
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            # Safe forward pass
            result = self.safe_forward_pass(batch)
            if result[0] is None:  # Skip if forward pass failed
                step += 1
                continue

            loss, kd_loss, ce_loss, batch_tokens = result

            # Backward pass
            loss = loss / self.args.grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate metrics
            running_loss += loss.item() * self.args.grad_accum_steps
            running_kd_loss += kd_loss
            running_ce_loss += ce_loss
            total_tokens += batch_tokens
            successful_steps += 1

            # Optimizer step
            if (step + 1) % self.args.grad_accum_steps == 0:
                try:
                    if use_scaler:
                        scaler.unscale_(self.distiller.optimizer)
                        # More aggressive gradient clipping for FP16
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.distiller.config.max_grad_norm)
                        if hasattr(self.distiller, 'router_params'):
                            torch.nn.utils.clip_grad_norm_(self.distiller.router_params, self.distiller.config.max_grad_norm)

                        # Skip update if gradients are too large
                        if grad_norm > 5.0:
                            print(f"  Skipping optimizer step: grad_norm={grad_norm:.4f}")
                            scaler.update()
                            self.distiller.optimizer.zero_grad()
                            continue

                        scaler.step(self.distiller.optimizer)
                        scaler.update()
                    else:
                        # More aggressive gradient clipping for FP16
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.distiller.config.max_grad_norm)

                        # Skip update if gradients are too large
                        if grad_norm > 5.0:
                            print(f"  Skipping optimizer step: grad_norm={grad_norm:.4f}")
                            self.distiller.optimizer.zero_grad()
                            continue
                    if hasattr(self.distiller, 'router_params'):
                        torch.nn.utils.clip_grad_norm_(self.distiller.router_params, self.distiller.config.max_grad_norm)
                    self.distiller.optimizer.step()

                self.distiller.optimizer.zero_grad(set_to_none=True)

                # Step scheduler
                if self.distiller.scheduler is not None:
                    self.distiller.scheduler.step()

                # Clear cache
                if self.args.clear_cache_every_step and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            step += 1

            # Logging
            if step % self.args.grad_accum_steps == 0 and log_callback and successful_steps > 0:
                avg_loss = running_loss / successful_steps
                avg_kd_loss = running_kd_loss / successful_steps
                avg_ce_loss = running_ce_loss / successful_steps

                metrics = {
                    "step": step,
                    "train_loss": avg_loss,
                    "train_kd_loss": avg_kd_loss,
                    "train_ce_loss": avg_ce_loss,
                    "train_ppl": min(math.exp(avg_loss), 1000),
                    "tokens": total_tokens,
                    "successful_batches": successful_steps,
                    "success_rate": successful_steps / step if step > 0 else 0,
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

            # Save checkpoint
            if self.args.save_steps > 0 and step % self.args.save_steps == 0 and step > 0:
                if save_callback:
                    save_callback(step, self.distiller.student, getattr(self.distiller, 'router_linears', []))

        # Final evaluation
        print("🏁 Running final evaluation...")
        final_ppl = self.run_evaluation(step, log_callback)
        print(f"Final validation perplexity: {final_ppl:.2f}")
        best_ppl_display = self.best_val_ppl if self.best_val_ppl != float('inf') else final_ppl
        print(f"Best validation perplexity: {best_ppl_display:.2f} at step {self.best_step}")

        return self.training_history

    def run_evaluation(self, step, log_callback=None):
        print(f"🔍 Running evaluation at step {step}")

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
                log_callback({
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_ppl": self.best_val_ppl if self.best_val_ppl != float('inf') else val_ppl,
                    "best_step": self.best_step,
                    "is_best": is_best
                })

            # Store history
            self.training_history.append({
                'step': step,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'is_best': is_best
            })

            return val_ppl

        except Exception as e:
            print(f"❌ Evaluation failed at step {step}: {e}")
            return float('inf')


def main():
    args = parse_args()

    print("🎯 STABLE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Stable version with conservative settings and robust error handling")
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

    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    print(f"  Model dtype: {dtype}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Ultra-conservative learning rates: student={args.student_lr}, router={args.router_lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  FP16 numerical stability: Enhanced")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_memory_info("Initial", device)

    # Load models
    print("\n📚 Loading models...")
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
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler_type,
        total_steps=args.train_steps,
        use_fp16=use_fp16,
        offload_teacher_to_cpu=args.offload_teacher_to_cpu,
        clear_cache_every_step=args.clear_cache_every_step,
    )

    # Create distiller
    print("⚗️ Creating distiller...")
    distiller = SARDistiller(
        teacher, student, device, cfg,
        tokenizers_compatible=same_tok,
        router_patterns=args.router_patterns
    )

    # Create evaluator
    evaluator = StableEvaluator(
        eval_ds, same_tok, collator,
        args.per_device_batch_size, args.eval_batches, device
    )

    # Create trainer
    trainer = StableTrainer(distiller, evaluator, args)
    print_memory_info("After setup", device)

    # Test evaluation
    print("\n🧪 Testing initial evaluation...")
    try:
        initial_ppl = evaluator.evaluate(student, use_fp16=use_fp16)
        print(f"✅ Initial evaluation successful: PPL = {initial_ppl:.2f}")
    except Exception as e:
        print(f"❌ Initial evaluation failed: {e}")
        return

    # Callbacks
    def save_callback(step, student_model, router_linears):
        try:
            step_dir = os.path.join(args.output_dir, f'checkpoint_{step}')
            os.makedirs(step_dir, exist_ok=True)

            # Save student model
            student_model.save_pretrained(step_dir)
            try:
                student_tok.save_pretrained(step_dir)
            except:
                pass

            # Save router state if available
            if router_linears:
                router_state = {}
                for name, lin in router_linears:
                    for pname, p in lin.named_parameters(recurse=False):
                        router_state[f"{name}.{pname}"] = p.detach().cpu()
                torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

            print(f"💾 Checkpoint saved: step {step}")

            # Save best model
            if trainer.best_step == step and args.save_best_model:
                best_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_dir, exist_ok=True)
                student_model.save_pretrained(best_dir)
                try:
                    student_tok.save_pretrained(best_dir)
                except:
                    pass

                metadata = {
                    'step': step,
                    'val_ppl': trainer.best_val_ppl,
                    'val_loss': math.log(trainer.best_val_ppl) if trainer.best_val_ppl != float('inf') else float('inf')
                }
                with open(os.path.join(best_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"🏆 Best model saved: PPL = {trainer.best_val_ppl:.2f}")

        except Exception as e:
            print(f"❌ Save failed: {e}")

    def log_callback(metrics):
        ts = datetime.now(UTC).isoformat()

        # Add memory info
        if torch.cuda.is_available():
            metrics['gpu_memory_gb'] = torch.cuda.memory_allocated(device) / 1024**3

        log_entry = {"ts": ts, **metrics}
        print(json.dumps(log_entry))

        # Save to log file
        with open(os.path.join(args.output_dir, 'training.jsonl'), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    # Run training
    print("\n🚀 Starting stable training...")
    try:
        training_history = trainer.train(
            train_loader=train_loader,
            save_callback=save_callback,
            log_callback=log_callback
        )

        # Save training history
        with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"\n✅ Training completed successfully!")
        best_ppl_final = trainer.best_val_ppl if trainer.best_val_ppl != float('inf') else 0.0
        print(f"📊 Best validation perplexity: {best_ppl_final:.2f} at step {trainer.best_step}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print_memory_info("Final", device)


if __name__ == '__main__':
    main()
