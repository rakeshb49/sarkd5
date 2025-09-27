import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
from datetime import datetime, UTC

import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student
from sar_kd.trainer import SARConfig, SARDistiller


def parse_args():
    p = argparse.ArgumentParser(description="Quality-focused SAR Knowledge Distillation with conservative memory optimizations")

    # Core training arguments
    p.add_argument('--teacher_model', type=str, default='huihui-ai/Huihui-MoE-1B-A0.6B')
    p.add_argument('--student_model', type=str, default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--dataset_name', type=str, default='wikitext')
    p.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=512, help='Sequence length - 512 is good balance of quality/memory')
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=24, help='Balanced for memory and update frequency')
    p.add_argument('--train_steps', type=int, default=1000)
    p.add_argument('--eval_steps', type=int, default=200)
    p.add_argument('--save_steps', type=int, default=0, help='Save model every N steps (0 to disable)')

    # Learning parameters (quality-focused)
    p.add_argument('--student_lr', type=float, default=1e-4, help='Student learning rate')
    p.add_argument('--router_lr', type=float, default=5e-4, help='Router learning rate')
    p.add_argument('--temperature', type=float, default=3.0, help='Slightly higher temp for better knowledge transfer')
    p.add_argument('--alpha_kd', type=float, default=0.85, help='Knowledge distillation weight')
    p.add_argument('--alpha_ce', type=float, default=0.15, help='Cross-entropy weight')
    p.add_argument('--router_anchor_weight', type=float, default=1e-4)
    p.add_argument('--router_load_balance_weight', type=float, default=1e-3)
    p.add_argument('--router_entropy_weight', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01, help='Small weight decay for regularization')
    p.add_argument('--max_grad_norm', type=float, default=1.0)

    # Scheduler (recommended for quality)
    p.add_argument('--use_scheduler', action='store_true', default=True, help='Use learning rate scheduler (recommended)')
    p.add_argument('--no_scheduler', dest='use_scheduler', action='store_false', help='Disable scheduler')
    p.add_argument('--warmup_steps', type=int, default=150, help='Longer warmup for stability')
    p.add_argument('--scheduler_type', type=str, default='cosine', choices=['linear', 'cosine'])
    p.add_argument('--router_patterns', nargs='*', help='Custom regex patterns for router discovery')

    # Memory optimizations (conservative for quality)
    p.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'],
                   help='Use float16 for memory efficiency (minimal quality impact)')
    p.add_argument('--conservative_memory', action='store_true', default=True,
                   help='Use conservative memory optimizations that minimize quality impact')
    p.add_argument('--aggressive_memory', dest='conservative_memory', action='store_false',
                   help='Use more aggressive memory optimizations (may impact quality)')

    # Quality monitoring
    p.add_argument('--eval_samples', type=int, default=100, help='Number of samples for evaluation')
    p.add_argument('--save_best_model', action='store_true', help='Save model with best validation performance')

    # System
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd_quality_focused')

    return p.parse_args()


def print_memory_info(stage="", device=None):
    """Print memory information without being too verbose"""
    if torch.cuda.is_available() and device is not None:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[{stage}] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


class QualityFocusedTrainer:
    """
    A wrapper around SARDistiller that implements quality-focused training strategies
    """

    def __init__(self, distiller, config, eval_loader=None):
        self.distiller = distiller
        self.config = config
        self.eval_loader = eval_loader
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_with_quality_monitoring(self, train_loader, steps, grad_accum_steps,
                                    eval_every, save_callback=None, log_callback=None):
        """
        Enhanced training loop with quality monitoring
        """

        def enhanced_log_callback(metrics):
            # Add quality indicators to logging
            if 'train_loss' in metrics:
                # Calculate perplexity for better interpretability
                try:
                    ppl = torch.exp(torch.tensor(metrics['train_loss'])).item()
                    metrics['train_ppl'] = min(ppl, 1000)  # Cap at 1000 for display
                except:
                    pass

            # Memory monitoring (less verbose)
            if torch.cuda.is_available() and 'step' in metrics and metrics['step'] % (grad_accum_steps * 5) == 0:
                device = self.distiller.device
                allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
                metrics['gpu_memory_gb'] = round(allocated_gb, 1)

            if log_callback:
                log_callback(metrics)

        def enhanced_eval_callback(step):
            """Enhanced evaluation with quality metrics"""
            if self.eval_loader is not None and step % eval_every == 0:
                try:
                    print(f"🔍 Running evaluation at step {step}")
                    val_ppl = self.distiller.evaluate(self.eval_loader)

                    # Handle NaN/Inf validation results
                    if val_ppl is None or torch.isnan(torch.tensor(val_ppl)) or torch.isinf(torch.tensor(val_ppl)):
                        print(f"⚠️ Invalid validation perplexity: {val_ppl}, using fallback")
                        val_ppl = 1000.0  # Fallback value

                    val_loss = torch.log(torch.tensor(max(val_ppl, 1.01))).item()  # Prevent log(1) or log(<1)

                    # Track best model for quality
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if hasattr(self.config, 'save_best_model') and self.config.save_best_model:
                            # Save best model state
                            self.best_model_state = {
                                'step': step,
                                'student_state_dict': self.distiller.student.state_dict(),
                                'router_state': {f"{name}.{pname}": p.clone()
                                               for name, lin in self.distiller.router_linears
                                               for pname, p in lin.named_parameters(recurse=False)},
                                'val_loss': val_loss,
                                'val_ppl': val_ppl
                            }
                            print(f"💾 New best model saved (step {step}, val_ppl: {val_ppl:.2f})")

                    # Log validation metrics
                    if enhanced_log_callback:
                        enhanced_log_callback({
                            "step": step,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                            "best_val_ppl": torch.exp(torch.tensor(self.best_val_loss)).item()
                        })

                    return val_ppl
                except Exception as e:
                    print(f"❌ Evaluation failed at step {step}: {e}")
                    return None
            return None

        # Custom training loop with evaluation
        original_trainer = self.distiller
        step = 0

        for batch in train_loader:
            if step >= steps:
                break

            # Run evaluation periodically
            if step % eval_every == 0 and step > 0:
                enhanced_eval_callback(step)

            # Run save callback periodically
            if save_callback and step % eval_every == 0 and step > 0:
                save_callback(step, original_trainer.student, original_trainer.router_linears)

            step += 1

        # Run the actual distiller training with modified callbacks
        def modified_save_callback(step, student_model, router_linears):
            # Run evaluation before saving
            if step % eval_every == 0:
                enhanced_eval_callback(step)

            # Call original save callback
            if save_callback:
                save_callback(step, student_model, router_linears)

        # Use the distiller's train method with our enhancements
        self.distiller.train(
            train_loader=train_loader,
            eval_loader=None,  # We handle eval ourselves
            steps=steps,
            grad_accum_steps=grad_accum_steps,
            eval_every=eval_every,  # Enable for save timing
            save_every=0,  # We handle saving through eval
            save_callback=modified_save_callback,
            log_callback=enhanced_log_callback,
        )

        # Final evaluation and save best model
        if self.eval_loader is not None:
            try:
                print("🏁 Running final evaluation...")
                final_ppl = self.distiller.evaluate(self.eval_loader)
                if final_ppl is None or torch.isnan(torch.tensor(final_ppl)) or torch.isinf(torch.tensor(final_ppl)):
                    print(f"⚠️ Final evaluation returned invalid result: {final_ppl}")
                    final_ppl = 1000.0
                else:
                    print(f"🏁 Final validation perplexity: {final_ppl:.2f}")
            except Exception as e:
                print(f"❌ Final evaluation failed: {e}")
                final_ppl = 1000.0

            if hasattr(self.config, 'save_best_model') and self.config.save_best_model and self.best_model_state:
                print(f"🏆 Best model: step {self.best_model_state['step']}, "
                      f"val_ppl: {self.best_model_state['val_ppl']:.2f}")


def main():
    args = parse_args()

    print("🎯 Quality-Focused SAR Knowledge Distillation Training")
    print("=" * 70)
    print("This script prioritizes training quality while implementing")
    print("conservative memory optimizations that minimize quality impact.")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"📝 Training configuration saved to: {config_path}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = args.model_dtype == 'float16'
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"\n🔧 Configuration:")
    print(f"  Device: {device}")
    print(f"  Model dtype: {dtype}")
    print(f"  Sequence length: {args.block_size}")
    print(f"  Effective batch size: {args.per_device_batch_size * args.grad_accum_steps}")
    print(f"  Conservative memory mode: {args.conservative_memory}")

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print_memory_info("Initial", device)

    # Load models
    print(f"\n📚 Loading models...")
    teacher, student, teacher_tok, student_tok = load_teacher_student(
        args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
    )

    print_memory_info("After model loading", device)

    # Build datasets
    print(f"📊 Building datasets...")
    train_ds, eval_ds, same_tok = build_text_datasets(
        args.dataset_name, args.dataset_config_name,
        teacher_tok, student_tok, args.block_size
    )

    # Create data collator
    if same_tok:
        collator = DataCollatorForLanguageModeling(teacher_tok, mlm=False)
    else:
        collator = DualTokenizerCollator(teacher_tok, student_tok, args.block_size)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=args.conservative_memory,  # Enable pin_memory in conservative mode
        num_workers=0,
    )

    eval_loader = None
    if eval_ds is not None:
        # Limit eval dataset size for faster evaluation
        if len(eval_ds) > args.eval_samples:
            eval_indices = torch.randperm(len(eval_ds))[:args.eval_samples]
            eval_ds = torch.utils.data.Subset(eval_ds, eval_indices)

        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=args.conservative_memory,
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
        # Conservative memory settings
        offload_teacher_to_cpu=not args.conservative_memory,  # Only if not conservative
        clear_cache_every_step=not args.conservative_memory,  # Only if not conservative
    )

    # Add config extensions for quality focus
    cfg.save_best_model = args.save_best_model

    print(f"\n🧠 Memory optimizations:")
    print(f"  Teacher CPU offloading: {cfg.offload_teacher_to_cpu}")
    print(f"  Cache clearing: {cfg.clear_cache_every_step}")
    print(f"  Gradient checkpointing: Enabled")
    print(f"  Mixed precision: {use_fp16}")

    # Create distiller
    print(f"\n⚗️ Creating distiller...")
    distiller = SARDistiller(
        teacher, student, device, cfg,
        tokenizers_compatible=same_tok,
        router_patterns=args.router_patterns
    )

    print_memory_info("After distiller creation", device)

    # Optional memory cleanup
    if not args.conservative_memory:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_info("After cleanup", device)

    # Create quality-focused trainer
    quality_trainer = QualityFocusedTrainer(distiller, cfg, eval_loader)

    # Callbacks
    def save_callback(step, student_model, router_linears):
        """Quality-focused save callback"""
        # Save current model
        step_dir = os.path.join(args.output_dir, 'current')
        os.makedirs(step_dir, exist_ok=True)

        student_model.save_pretrained(step_dir)
        try:
            student_tok.save_pretrained(step_dir)
        except:
            pass

        # Save router updates
        router_state = {}
        for name, lin in router_linears:
            for pname, p in lin.named_parameters(recurse=False):
                router_state[f"{name}.{pname}"] = p.detach().cpu()
        torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

        # Save best model if we have one
        if quality_trainer.best_model_state is not None:
            try:
                best_dir = os.path.join(args.output_dir, 'best')
                os.makedirs(best_dir, exist_ok=True)

                # Save best student model
                current_state = student_model.state_dict()  # Backup current state
                student_model.load_state_dict(quality_trainer.best_model_state['student_state_dict'])
                student_model.save_pretrained(best_dir)
                student_model.load_state_dict(current_state)  # Restore current state

                try:
                    student_tok.save_pretrained(best_dir)
                except Exception as e:
                    print(f"Warning: Could not save tokenizer: {e}")

                # Save best router state
                torch.save(quality_trainer.best_model_state['router_state'],
                          os.path.join(best_dir, 'router_update.pt'))

                # Save metadata
                metadata = {
                    'step': quality_trainer.best_model_state['step'],
                    'val_loss': quality_trainer.best_model_state['val_loss'],
                    'val_ppl': quality_trainer.best_model_state['val_ppl'],
                }
                with open(os.path.join(best_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save best model: {e}")

    def log_callback(metrics):
        """Enhanced logging"""
        ts = datetime.now(UTC).isoformat()
        print(json.dumps({"ts": ts, **metrics}))

        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
        with open(metrics_path, 'a') as f:
            f.write(json.dumps({"ts": ts, **metrics}) + '\n')

    print(f"\n🚀 Starting training...")
    print(f"  Steps: {args.train_steps}")
    print(f"  Evaluation every: {args.eval_steps} steps")
    print(f"  Output directory: {args.output_dir}")

    try:
        # Use the quality-focused trainer
        quality_trainer.train_with_quality_monitoring(
            train_loader=train_loader,
            steps=args.train_steps,
            grad_accum_steps=args.grad_accum_steps,
            eval_every=args.eval_steps,
            save_callback=save_callback,
            log_callback=log_callback,
        )

        print(f"\n✅ Training completed successfully!")
        if quality_trainer.best_model_state:
            print(f"🏆 Best model saved with validation perplexity: "
                  f"{quality_trainer.best_model_state['val_ppl']:.2f}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n💥 CUDA Out of Memory Error: {e}")
        print("\n🔧 Suggestions to reduce memory usage:")
        print("  • Try: --aggressive_memory (less conservative optimizations)")
        print("  • Try: --block_size 256 (shorter sequences)")
        print("  • Try: --grad_accum_steps 48 (smaller memory per step)")
        print_memory_info("OOM Error", device)
        raise
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print_memory_info("Error", device)
        raise
    finally:
        # Final cleanup
        print_memory_info("Final", device)


if __name__ == '__main__':
    main()
