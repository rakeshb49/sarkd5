import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import math
import traceback
from datetime import datetime, UTC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student
from sar_kd.trainer import SARConfig, SARDistiller
from sar_kd.losses import (
    ce_loss,
    kd_kl_loss,
    router_anchor_l2,
    router_entropy_bonus,
    router_load_balance_loss,
)
from sar_kd.router_utils import snapshot_router_state


def parse_args():
    p = argparse.ArgumentParser(description="Complete SAR Knowledge Distillation with integrated training and evaluation")

    # Core training arguments
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

    # Learning parameters
    p.add_argument('--student_lr', type=float, default=1e-4)
    p.add_argument('--router_lr', type=float, default=5e-4)
    p.add_argument('--temperature', type=float, default=3.0)
    p.add_argument('--alpha_kd', type=float, default=0.85)
    p.add_argument('--alpha_ce', type=float, default=0.15)
    p.add_argument('--router_anchor_weight', type=float, default=1e-4)
    p.add_argument('--router_load_balance_weight', type=float, default=1e-3)
    p.add_argument('--router_entropy_weight', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--max_grad_norm', type=float, default=1.0)

    # Scheduler
    p.add_argument('--use_scheduler', action='store_true', default=True)
    p.add_argument('--warmup_steps', type=int, default=150)
    p.add_argument('--scheduler_type', type=str, default='cosine', choices=['linear', 'cosine'])
    p.add_argument('--router_patterns', nargs='*', help='Custom regex patterns for router discovery')

    # Memory optimizations
    p.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'])
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=False)
    p.add_argument('--clear_cache_every_step', action='store_true', default=True)

    # Evaluation settings
    p.add_argument('--eval_batches', type=int, default=50, help='Number of batches for evaluation')
    p.add_argument('--save_best_model', action='store_true', default=True)
    p.add_argument('--early_stopping_patience', type=int, default=3, help='Stop if no improvement for N evaluations')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd_complete')

    return p.parse_args()


def print_memory_info(stage="", device=None):
    """Print memory information"""
    if torch.cuda.is_available() and device is not None:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[{stage}] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


class IntegratedEvaluator:
    """Evaluator that works with the actual training state"""

    def __init__(self, eval_dataset, tokenizer_compatible, collator, batch_size, eval_batches, device):
        self.eval_dataset = eval_dataset
        self.tokenizer_compatible = tokenizer_compatible
        self.collator = collator
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.device = device

    def evaluate(self, student_model, use_fp16=False):
        """Run evaluation using current student model state"""
        if self.eval_dataset is None:
            return float('inf')

        # Create fresh evaluation loader with different sampling each time
        eval_size = min(len(self.eval_dataset), self.eval_batches * self.batch_size * 2)
        eval_indices = torch.randperm(len(self.eval_dataset))[:eval_size]
        eval_subset = torch.utils.data.Subset(self.eval_dataset, eval_indices)

        eval_loader = DataLoader(
            eval_subset,
            batch_size=self.batch_size,
            shuffle=True,
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

                            loss_fct = nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
                            losses = losses.view(targets.shape)

                            masked_losses = losses * valid_mask.float()
                            batch_loss = masked_losses.sum() / valid_mask.sum()

                            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                                continue

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
            return float('inf')

        avg_loss = total_loss / batch_count
        perplexity = math.exp(min(avg_loss, 20))
        return perplexity


class CompleteTrainer:
    """Complete trainer with integrated SAR distillation and evaluation"""

    def __init__(self, distiller, evaluator, args):
        self.distiller = distiller
        self.evaluator = evaluator
        self.args = args
        self.best_val_ppl = float('inf')
        self.best_step = 0
        self.patience_counter = 0
        self.training_history = []

        # Initialize router state for anchor loss
        self.router_init = snapshot_router_state(self.distiller.router_linears)

    def train(self, train_loader, save_callback=None, log_callback=None):
        """Complete training loop with SAR distillation and evaluation"""

        print("🚀 Starting complete SAR Knowledge Distillation training...")

        # Training setup
        self.distiller.teacher.eval()
        self.distiller.student.train()

        step = 0
        total_tokens = 0
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0

        # Mixed precision setup
        use_scaler = torch.cuda.is_available() and self.distiller.use_fp16
        has_fp16_params = any(p.dtype == torch.float16 for p in list(self.distiller.student.parameters()) + list(self.distiller.router_params))
        if has_fp16_params and use_scaler:
            use_scaler = False

        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        # Initial evaluation
        if step % self.args.eval_steps == 0:
            self.run_evaluation(step, log_callback)

        # Training loop
        for batch in train_loader:
            if step >= self.args.train_steps:
                break

            # Move teacher to GPU if CPU offloading enabled
            if self.args.offload_teacher_to_cpu:
                if self.distiller.teacher_device.type == 'cpu':
                    self.distiller.teacher.to(self.distiller.device)
                    self.distiller.teacher_device = self.distiller.device

            # Handle batch data
            try:
                if 'teacher_input_ids' in batch:
                    # Dual tokenizer path
                    t_ids = batch['teacher_input_ids'].to(self.distiller.device)
                    t_mask = batch['teacher_attention_mask'].to(self.distiller.device)
                    s_ids = batch['student_input_ids'].to(self.distiller.device)
                    s_mask = batch['student_attention_mask'].to(self.distiller.device)

                    with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                        # Forward passes
                        t_out = self.distiller.teacher(input_ids=t_ids, attention_mask=t_mask, output_hidden_states=True)
                        s_out = self.distiller.student(input_ids=s_ids, attention_mask=s_mask, output_hidden_states=True)

                        # Hidden state alignment for dual tokenizer case
                        t_h = t_out.hidden_states[-1]
                        s_h = s_out.hidden_states[-1]
                        if self.distiller.h_proj is not None:
                            t_h_proj = self.distiller.h_proj(t_h)
                        else:
                            t_h_proj = t_h

                        # Simplified alignment (MSE loss between final hidden states)
                        if t_h_proj.size(1) == s_h.size(1):  # Same sequence length
                            kd = nn.functional.mse_loss(s_h, t_h_proj.detach())
                        else:
                            # Use mean pooling if sequence lengths differ
                            t_mean = t_h_proj.mean(dim=1)
                            s_mean = s_h.mean(dim=1)
                            kd = nn.functional.mse_loss(s_mean, t_mean.detach())

                        # CE loss on student labels
                        s_logits = s_out.logits[:, :-1, :]
                        s_labels = batch['student_labels'][:, 1:].to(self.distiller.device)
                        if (s_labels != -100).any():
                            ce = ce_loss(s_logits, s_labels)
                        else:
                            ce = torch.tensor(0.0, device=self.distiller.device)

                else:
                    # Same tokenizer path
                    input_ids = batch["input_ids"].to(self.distiller.device)
                    labels = batch["labels"].to(self.distiller.device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.distiller.device)

                    with torch.amp.autocast('cuda', enabled=self.distiller.use_fp16):
                        t_out = self.distiller.teacher(input_ids=input_ids, attention_mask=attention_mask)
                        s_out = self.distiller.student(input_ids=input_ids, attention_mask=attention_mask)

                        teacher_logits = t_out.logits[:, :-1, :]
                        student_logits = s_out.logits[:, :-1, :]
                        target_labels = labels[:, 1:]

                        kd = kd_kl_loss(student_logits, teacher_logits, self.distiller.config.temperature)
                        ce = ce_loss(student_logits, target_labels)

                # Router regularization
                current_router_state = {}
                for name, lin in self.distiller.router_linears:
                    for pname, p in lin.named_parameters(recurse=False):
                        current_router_state[f"{name}.{pname}"] = p
                anchor = router_anchor_l2(current_router_state, self.router_init) * self.distiller.config.router_anchor_weight

                gate_probs = self.distiller.gating_stats.pop_and_reset()
                lb = router_load_balance_loss(gate_probs, self.distiller.device) * self.distiller.config.router_load_balance_weight
                ent = router_entropy_bonus(gate_probs, self.distiller.device) * self.distiller.config.router_entropy_weight

                # Total loss
                loss = self.distiller.config.alpha_kd * kd + self.distiller.config.alpha_ce * ce + anchor + lb + ent

                # Handle NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at step {step}, skipping")
                    continue

            except Exception as e:
                print(f"Error in forward pass at step {step}: {e}")
                continue

            # Move teacher back to CPU if offloading enabled
            if self.args.offload_teacher_to_cpu:
                self.distiller.teacher.cpu()
                self.distiller.teacher_device = torch.device('cpu')

            # Backward pass
            loss = loss / self.args.grad_accum_steps
            if loss.requires_grad:
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Accumulate metrics
            running_loss += loss.item() * self.args.grad_accum_steps
            running_kd_loss += kd.item() if torch.is_tensor(kd) else 0
            running_ce_loss += ce.item() if torch.is_tensor(ce) else 0

            # Count tokens
            if 'teacher_input_ids' in batch:
                s_labels = batch.get('student_labels')
                if s_labels is not None:
                    total_tokens += int((s_labels != -100).sum().item())
                else:
                    total_tokens += int(s_mask.sum().item())
            else:
                if labels is not None:
                    total_tokens += int((labels != -100).sum().item())
                else:
                    total_tokens += int(input_ids.numel())

            # Optimizer step
            if (step + 1) % self.args.grad_accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(self.distiller.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.distiller.config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.distiller.router_params, self.distiller.config.max_grad_norm)
                    scaler.step(self.distiller.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.distiller.student.parameters(), self.distiller.config.max_grad_norm)
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
            if step % self.args.grad_accum_steps == 0 and log_callback:
                avg_loss = running_loss / max(1, step)
                avg_kd_loss = running_kd_loss / max(1, step)
                avg_ce_loss = running_ce_loss / max(1, step)

                metrics = {
                    "step": step,
                    "train_loss": avg_loss,
                    "train_kd_loss": avg_kd_loss,
                    "train_ce_loss": avg_ce_loss,
                    "train_ppl": min(math.exp(avg_loss), 1000),
                    "tokens": total_tokens,
                    "lr": self.distiller.optimizer.param_groups[0]['lr'] if self.distiller.optimizer else 0
                }
                log_callback(metrics)

            # Evaluation
            if step % self.args.eval_steps == 0 and step > 0:
                val_ppl = self.run_evaluation(step, log_callback)

                # Early stopping check
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self.best_step = step
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.args.early_stopping_patience:
                    print(f"Early stopping: no improvement for {self.args.early_stopping_patience} evaluations")
                    break

            # Save checkpoint
            if self.args.save_steps > 0 and step % self.args.save_steps == 0 and step > 0:
                if save_callback:
                    save_callback(step, self.distiller.student, self.distiller.router_linears)

        # Final evaluation
        print("🏁 Running final evaluation...")
        final_ppl = self.run_evaluation(step, log_callback)
        print(f"Final validation perplexity: {final_ppl:.2f}")
        print(f"Best validation perplexity: {self.best_val_ppl:.2f} at step {self.best_step}")

        return self.training_history

    def run_evaluation(self, step, log_callback=None):
        """Run evaluation and track progress"""
        print(f"🔍 Running evaluation at step {step}")

        try:
            val_ppl = self.evaluator.evaluate(self.distiller.student, use_fp16=self.distiller.use_fp16)
            val_loss = math.log(val_ppl) if val_ppl != float('inf') else float('inf')

            print(f"  📊 Step {step}: Validation PPL = {val_ppl:.2f}")

            # Track best model
            is_best = val_ppl < self.best_val_ppl
            if is_best:
                print(f"  🏆 New best model! PPL: {val_ppl:.2f}")

            # Log results
            if log_callback:
                log_callback({
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_ppl": self.best_val_ppl,
                    "best_step": self.best_step,
                    "is_best": is_best,
                    "patience": self.patience_counter
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
            traceback.print_exc()
            return float('inf')


def main():
    args = parse_args()

    print("🎯 COMPLETE SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Fully integrated training with SAR distillation and working evaluation")
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
    print(f"  Eval every: {args.eval_steps} steps")
    print(f"  Early stopping patience: {args.early_stopping_patience}")

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
    evaluator = IntegratedEvaluator(
        eval_ds, same_tok, collator,
        args.per_device_batch_size, args.eval_batches, device
    )

    # Create trainer
    trainer = CompleteTrainer(distiller, evaluator, args)
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

            # Save router state
            router_state = {}
            for name, lin in router_linears:
                for pname, p in lin.named_parameters(recurse=False):
                    router_state[f"{name}.{pname}"] = p.detach().cpu()
            torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

            print(f"💾 Checkpoint saved: step {step}")

            # Save best model if this is the best
            if trainer.best_step == step and args.save_best_model:
                best_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_dir, exist_ok=True)
                student_model.save_pretrained(best_dir)
                try:
                    student_tok.save_pretrained(best_dir)
                except:
                    pass
                torch.save(router_state, os.path.join(best_dir, 'router_update.pt'))

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
    print("\n🚀 Starting complete training...")
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
        print(f"📊 Best validation perplexity: {trainer.best_val_ppl:.2f} at step {trainer.best_step}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()
    finally:
        print_memory_info("Final", device)


if __name__ == '__main__':
    main()
