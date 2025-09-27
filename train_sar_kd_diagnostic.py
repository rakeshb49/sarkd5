import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import traceback
from datetime import datetime, UTC

import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student
from sar_kd.trainer import SARConfig, SARDistiller


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostic SAR Knowledge Distillation with robust error handling")

    # Core training arguments
    p.add_argument('--teacher_model', type=str, default='huihui-ai/Huihui-MoE-1B-A0.6B')
    p.add_argument('--student_model', type=str, default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--dataset_name', type=str, default='wikitext')
    p.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=512)
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=24)
    p.add_argument('--train_steps', type=int, default=200, help='Reduced for diagnostic run')
    p.add_argument('--eval_steps', type=int, default=50, help='More frequent evaluation for debugging')
    p.add_argument('--save_steps', type=int, default=50, help='Save frequently for debugging')

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
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--scheduler_type', type=str, default='cosine', choices=['linear', 'cosine'])
    p.add_argument('--router_patterns', nargs='*', help='Custom regex patterns for router discovery')

    # Memory optimizations
    p.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'])
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=False, help='Disabled for debugging')
    p.add_argument('--clear_cache_every_step', action='store_true', default=True)

    # Diagnostic options
    p.add_argument('--debug_eval', action='store_true', default=True, help='Enable detailed evaluation debugging')
    p.add_argument('--eval_samples', type=int, default=10, help='Small eval set for debugging')
    p.add_argument('--skip_eval_on_error', action='store_true', help='Continue training even if eval fails')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd_diagnostic')

    return p.parse_args()


def print_memory_info(stage="", device=None):
    """Enhanced memory information"""
    if torch.cuda.is_available() and device is not None:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        free = (torch.cuda.get_device_properties(device).total_memory / 1024**3) - reserved
        print(f"[{stage}] GPU Memory: {allocated:.1f}GB alloc, {reserved:.1f}GB rsv, {free:.1f}GB free")

        # Check for fragmentation
        if reserved - allocated > 2.0:  # More than 2GB fragmentation
            print(f"  ⚠️ Memory fragmentation detected: {reserved - allocated:.1f}GB")


def safe_evaluate(distiller, eval_loader, max_samples=None, debug=False):
    """
    Safe evaluation function with comprehensive error handling
    """
    if eval_loader is None:
        print("⚠️ No evaluation dataset available")
        return float('inf')

    print(f"🔍 Starting evaluation (debug={debug}, max_samples={max_samples})")

    try:
        # Set models to evaluation mode
        distiller.teacher.eval()
        distiller.student.eval()

        total_loss = 0.0
        total_samples = 0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if max_samples and total_samples >= max_samples:
                    print(f"  Reached max_samples limit ({max_samples})")
                    break

                try:
                    # Handle batch structure
                    if 'teacher_input_ids' in batch:
                        # Dual tokenizer mode
                        s_ids = batch['student_input_ids'].to(distiller.device)
                        s_mask = batch['student_attention_mask'].to(distiller.device)
                        s_labels = batch['student_labels'][:, 1:].to(distiller.device)

                        # Check for valid labels
                        valid_labels = (s_labels != -100)
                        if not valid_labels.any():
                            if debug:
                                print(f"  Batch {batch_idx}: No valid labels, skipping")
                            continue

                        # Student forward pass
                        with torch.amp.autocast('cuda', enabled=distiller.use_fp16):
                            s_out = distiller.student(input_ids=s_ids, attention_mask=s_mask)
                            s_logits = s_out.logits[:, :-1, :]  # Shift for next-token prediction

                            # Calculate loss only on valid labels
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1))
                            losses = losses.view(s_labels.shape)

                            # Mask invalid positions
                            masked_losses = losses * valid_labels.float()
                            batch_loss = masked_losses.sum() / valid_labels.sum()

                    else:
                        # Same tokenizer mode
                        input_ids = batch["input_ids"].to(distiller.device)
                        labels = batch["labels"][:, 1:].to(distiller.device)
                        attention_mask = batch.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(distiller.device)

                        # Check for valid labels
                        valid_labels = (labels != -100)
                        if not valid_labels.any():
                            if debug:
                                print(f"  Batch {batch_idx}: No valid labels, skipping")
                            continue

                        # Student forward pass
                        with torch.amp.autocast('cuda', enabled=distiller.use_fp16):
                            s_out = distiller.student(input_ids=input_ids, attention_mask=attention_mask)
                            s_logits = s_out.logits[:, :-1, :]

                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            losses = loss_fct(s_logits.view(-1, s_logits.size(-1)), labels.view(-1))
                            losses = losses.view(labels.shape)

                            masked_losses = losses * valid_labels.float()
                            batch_loss = masked_losses.sum() / valid_labels.sum()

                    # Check for NaN/Inf
                    if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                        print(f"  ⚠️ Batch {batch_idx}: Invalid loss ({batch_loss.item()}), skipping")
                        if debug:
                            print(f"    Valid labels: {valid_labels.sum().item()}")
                            print(f"    Logits range: [{s_logits.min().item():.3f}, {s_logits.max().item():.3f}]")
                        continue

                    total_loss += batch_loss.item()
                    total_samples += valid_labels.sum().item()
                    batch_count += 1

                    if debug and batch_idx < 3:
                        print(f"  Batch {batch_idx}: loss={batch_loss.item():.4f}, "
                              f"valid_tokens={valid_labels.sum().item()}")

                except Exception as e:
                    print(f"  ❌ Error in batch {batch_idx}: {e}")
                    if debug:
                        traceback.print_exc()
                    continue

        # Calculate final metrics
        if batch_count == 0:
            print("  ❌ No valid batches processed during evaluation")
            return float('inf')

        avg_loss = total_loss / batch_count
        perplexity = math.exp(min(avg_loss, 20))  # Cap at exp(20) to prevent overflow

        print(f"  ✅ Evaluation complete: {batch_count} batches, {total_samples} tokens")
        print(f"  📊 Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        return perplexity

    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        if debug:
            traceback.print_exc()
        return float('inf')
    finally:
        # Restore training mode
        distiller.student.train()


class DiagnosticTrainer:
    """Enhanced trainer with diagnostic capabilities"""

    def __init__(self, distiller, eval_loader, args):
        self.distiller = distiller
        self.eval_loader = eval_loader
        self.args = args
        self.best_val_loss = float('inf')
        self.best_step = 0
        self.evaluation_history = []

    def run_evaluation(self, step):
        """Run evaluation with comprehensive error handling"""
        print(f"\n🔬 Running evaluation at step {step}")
        try:
            val_ppl = safe_evaluate(
                self.distiller,
                self.eval_loader,
                max_samples=self.args.eval_samples,
                debug=self.args.debug_eval
            )

            val_loss = math.log(val_ppl) if val_ppl != float('inf') else float('inf')

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = step
                print(f"  🏆 New best model! Step {step}, PPL: {val_ppl:.2f}")

            # Store evaluation history
            self.evaluation_history.append({
                'step': step,
                'val_loss': val_loss,
                'val_ppl': val_ppl
            })

            return val_loss, val_ppl

        except Exception as e:
            print(f"❌ Evaluation failed at step {step}: {e}")
            if self.args.debug_eval:
                traceback.print_exc()

            if self.args.skip_eval_on_error:
                print("  ⏭️ Continuing training despite evaluation error")
                return float('inf'), float('inf')
            else:
                raise


def main():
    args = parse_args()

    print("🔬 DIAGNOSTIC SAR Knowledge Distillation Training")
    print("=" * 70)
    print("This diagnostic version includes comprehensive error handling")
    print("and detailed logging to identify and fix issues.")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'diagnostic_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set seeds
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
    print(f"  Eval samples: {args.eval_samples}")

    print_memory_info("Initial", device)

    # Load models
    print(f"\n📚 Loading models...")
    try:
        teacher, student, teacher_tok, student_tok = load_teacher_student(
            args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
        )
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return

    print_memory_info("After model loading", device)

    # Build datasets
    print(f"\n📊 Building datasets...")
    try:
        train_ds, eval_ds, same_tok = build_text_datasets(
            args.dataset_name, args.dataset_config_name,
            teacher_tok, student_tok, args.block_size
        )
        print(f"✅ Datasets built: train={len(train_ds)}, eval={len(eval_ds) if eval_ds else 0}")
    except Exception as e:
        print(f"❌ Dataset building failed: {e}")
        traceback.print_exc()
        return

    # Create data collator
    if same_tok:
        collator = DataCollatorForLanguageModeling(teacher_tok, mlm=False)
        print("📝 Using single tokenizer collator")
    else:
        collator = DualTokenizerCollator(teacher_tok, student_tok, args.block_size)
        print("📝 Using dual tokenizer collator")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )

    eval_loader = None
    if eval_ds is not None:
        # Create small eval subset for diagnostics
        eval_indices = torch.randperm(len(eval_ds))[:args.eval_samples * 5]  # Buffer for safety
        eval_subset = torch.utils.data.Subset(eval_ds, eval_indices)

        eval_loader = DataLoader(
            eval_subset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=False,
            num_workers=0,
        )
        print(f"📊 Evaluation loader created with {len(eval_subset)} samples")

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
    print(f"\n⚗️ Creating distiller...")
    try:
        distiller = SARDistiller(
            teacher, student, device, cfg,
            tokenizers_compatible=same_tok,
            router_patterns=args.router_patterns
        )
        print("✅ Distiller created successfully")
    except Exception as e:
        print(f"❌ Distiller creation failed: {e}")
        traceback.print_exc()
        return

    print_memory_info("After distiller creation", device)

    # Test initial evaluation
    print(f"\n🧪 Testing initial evaluation...")
    diagnostic_trainer = DiagnosticTrainer(distiller, eval_loader, args)
    try:
        initial_val_loss, initial_val_ppl = diagnostic_trainer.run_evaluation(0)
        print(f"✅ Initial evaluation successful: PPL = {initial_val_ppl:.2f}")
    except Exception as e:
        print(f"❌ Initial evaluation failed: {e}")
        if not args.skip_eval_on_error:
            return

    # Training callbacks
    def save_callback(step, student_model, router_linears):
        """Enhanced save callback with error handling"""
        try:
            step_dir = os.path.join(args.output_dir, f'checkpoint_{step}')
            os.makedirs(step_dir, exist_ok=True)

            # Save student model
            student_model.save_pretrained(step_dir)
            student_tok.save_pretrained(step_dir)

            # Save router updates
            router_state = {}
            for name, lin in router_linears:
                for pname, p in lin.named_parameters(recurse=False):
                    router_state[f"{name}.{pname}"] = p.detach().cpu()
            torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

            # Save evaluation history
            with open(os.path.join(step_dir, 'eval_history.json'), 'w') as f:
                json.dump(diagnostic_trainer.evaluation_history, f, indent=2)

            print(f"💾 Checkpoint saved: {step_dir}")

        except Exception as e:
            print(f"❌ Save failed at step {step}: {e}")

    def log_callback(metrics):
        """Enhanced logging callback"""
        ts = datetime.now(UTC).isoformat()

        # Add memory info periodically
        if 'step' in metrics and metrics['step'] % (args.grad_accum_steps * 2) == 0:
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                metrics.update({
                    'gpu_memory_allocated_gb': round(allocated_gb, 1),
                    'gpu_memory_reserved_gb': round(reserved_gb, 1),
                })

        # Log to console and file
        log_entry = {"ts": ts, **metrics}
        print(json.dumps(log_entry))

        # Append to log file
        log_path = os.path.join(args.output_dir, 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    # Custom training loop with evaluation
    print(f"\n🚀 Starting diagnostic training...")

    step = 0
    for batch in train_loader:
        if step >= args.train_steps:
            break

        # Run evaluation
        if step % args.eval_steps == 0 and step > 0:
            try:
                val_loss, val_ppl = diagnostic_trainer.run_evaluation(step)
                log_callback({
                    'step': step,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'best_val_ppl': math.exp(diagnostic_trainer.best_val_loss) if diagnostic_trainer.best_val_loss != float('inf') else float('inf')
                })
            except Exception as e:
                print(f"❌ Evaluation error at step {step}: {e}")
                if not args.skip_eval_on_error:
                    break

        # Save checkpoint
        if step % args.save_steps == 0 and step > 0:
            save_callback(step, distiller.student, distiller.router_linears)

        step += 1

    # Final evaluation
    print(f"\n🏁 Running final evaluation...")
    try:
        final_val_loss, final_val_ppl = diagnostic_trainer.run_evaluation(step)
        print(f"✅ Final validation perplexity: {final_val_ppl:.2f}")
        print(f"🏆 Best validation perplexity: {math.exp(diagnostic_trainer.best_val_loss):.2f} at step {diagnostic_trainer.best_step}")
    except Exception as e:
        print(f"❌ Final evaluation failed: {e}")

    # Save final results
    results = {
        'final_step': step,
        'final_val_ppl': final_val_ppl if 'final_val_ppl' in locals() else float('inf'),
        'best_val_ppl': math.exp(diagnostic_trainer.best_val_loss) if diagnostic_trainer.best_val_loss != float('inf') else float('inf'),
        'best_step': diagnostic_trainer.best_step,
        'evaluation_history': diagnostic_trainer.evaluation_history
    }

    results_path = os.path.join(args.output_dir, 'diagnostic_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Diagnostic results saved to: {results_path}")
    print_memory_info("Final", device)

    print("\n✅ Diagnostic training completed!")


if __name__ == '__main__':
    import math  # Add missing import
    main()
