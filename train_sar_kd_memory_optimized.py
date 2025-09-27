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
    p = argparse.ArgumentParser(description="Memory-optimized Student-Aware Router KD training")
    p.add_argument('--teacher_model', type=str, default='huihui-ai/Huihui-MoE-1B-A0.6B')
    p.add_argument('--student_model', type=str, default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--dataset_name', type=str, default='wikitext')
    p.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=512, help='Reduced from 1024 for memory efficiency')
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=32, help='Increased to maintain effective batch size')
    p.add_argument('--train_steps', type=int, default=1000)
    p.add_argument('--eval_steps', type=int, default=200)
    p.add_argument('--save_steps', type=int, default=0, help='Save model every N steps (0 to disable)')
    p.add_argument('--student_lr', type=float, default=1e-4)
    p.add_argument('--router_lr', type=float, default=5e-4)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--alpha_kd', type=float, default=0.9)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    p.add_argument('--router_anchor_weight', type=float, default=1e-4)
    p.add_argument('--router_load_balance_weight', type=float, default=1e-3)
    p.add_argument('--router_entropy_weight', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler with warmup and decay')
    p.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for learning rate scheduler')
    p.add_argument('--scheduler_type', type=str, default='cosine', choices=['linear', 'cosine'], help='Type of learning rate scheduler')
    p.add_argument('--router_patterns', nargs='*', help='Custom regex patterns for router discovery')

    # Memory optimization options - all enabled by default
    p.add_argument('--model_dtype', type=str, default='float16', choices=['float16', 'float32'],
                   help='Model precision: float16 for memory efficiency (default), float32 for stability')
    p.add_argument('--offload_teacher_to_cpu', action='store_true', default=True,
                   help='Move teacher model to CPU when not in use (default: enabled)')
    p.add_argument('--no_offload_teacher', dest='offload_teacher_to_cpu', action='store_false',
                   help='Disable teacher CPU offloading')
    p.add_argument('--clear_cache_every_step', action='store_true', default=True,
                   help='Clear GPU cache after each gradient update (default: enabled)')
    p.add_argument('--no_clear_cache', dest='clear_cache_every_step', action='store_false',
                   help='Disable cache clearing')
    p.add_argument('--aggressive_gc', action='store_true', default=True,
                   help='Run garbage collection frequently (default: enabled)')
    p.add_argument('--no_aggressive_gc', dest='aggressive_gc', action='store_false',
                   help='Disable aggressive garbage collection')

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd_mem_opt')
    return p.parse_args()


def print_memory_usage(stage=""):
    """Print current memory usage for debugging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print(f"[{stage}] CUDA not available")


def aggressive_memory_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryOptimizedDataLoader:
    """Wrapper around DataLoader that implements memory optimizations"""
    def __init__(self, dataloader, aggressive_gc=True):
        self.dataloader = dataloader
        self.aggressive_gc = aggressive_gc

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
            # Clean up after each batch if enabled
            if self.aggressive_gc:
                gc.collect()

    def __len__(self):
        return len(self.dataloader)


def main():
    args = parse_args()
    print("Starting memory-optimized SAR Knowledge Distillation training")
    print(f"Memory optimization settings:")
    print(f"  - Model dtype: {args.model_dtype}")
    print(f"  - Teacher CPU offloading: {args.offload_teacher_to_cpu}")
    print(f"  - Cache clearing: {args.clear_cache_every_step}")
    print(f"  - Aggressive GC: {args.aggressive_gc}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Gradient accumulation: {args.grad_accum_steps}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Set memory-optimized environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = args.model_dtype == 'float16'
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"Device: {device}")
    print(f"Model precision: {dtype}, Mixed precision training: {use_fp16}")

    print_memory_usage("Initial")

    # Enable memory-efficient settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        torch.set_float32_matmul_precision('high')

    # Load models with memory optimizations
    print("Loading models...")
    teacher, student, teacher_tok, student_tok = load_teacher_student(
        args.teacher_model, args.student_model, dtype, device, use_fp16=use_fp16
    )

    print_memory_usage("After model loading")

    if args.aggressive_gc:
        aggressive_memory_cleanup()
        print_memory_usage("After initial cleanup")

    # Build datasets
    print("Building datasets...")
    train_ds, eval_ds, same_tok = build_text_datasets(
        args.dataset_name, args.dataset_config_name, teacher_tok, student_tok, args.block_size
    )

    # Create data collator
    if same_tok:
        collator = DataCollatorForLanguageModeling(teacher_tok, mlm=False)
    else:
        collator = DualTokenizerCollator(teacher_tok, student_tok, args.block_size)

    # Create data loaders with memory optimization
    base_train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        num_workers=0,     # Single worker to avoid memory duplication
    )

    train_loader = MemoryOptimizedDataLoader(base_train_loader, args.aggressive_gc)

    eval_loader = None
    if eval_ds is not None:
        base_eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=False,
            num_workers=0,
        )
        eval_loader = MemoryOptimizedDataLoader(base_eval_loader, args.aggressive_gc)

    print_memory_usage("After data loading")

    # Create configuration with memory optimizations
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
    print("Creating distiller...")
    distiller = SARDistiller(teacher, student, device, cfg, tokenizers_compatible=same_tok, router_patterns=args.router_patterns)

    print_memory_usage("After distiller creation")

    if args.aggressive_gc:
        aggressive_memory_cleanup()
        print_memory_usage("After distiller cleanup")

    def save_callback(step, student_model, router_linears):
        """Memory-optimized save callback"""
        step_dir = os.path.join(args.output_dir)
        os.makedirs(step_dir, exist_ok=True)

        # Save student model
        student_dir = os.path.join(step_dir, 'student')
        os.makedirs(student_dir, exist_ok=True)
        student_model.save_pretrained(student_dir)

        # Save tokenizer
        try:
            student_tok.save_pretrained(student_dir)
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")

        # Save router updates
        router_state = {}
        for name, lin in router_linears:
            for pname, p in lin.named_parameters(recurse=False):
                router_state[f"{name}.{pname}"] = p.detach().cpu()
        torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

        # Clean up after saving
        if args.aggressive_gc:
            del router_state
            aggressive_memory_cleanup()

    def log_callback(metrics: dict):
        """Enhanced log callback with memory monitoring"""
        ts = datetime.now(UTC).isoformat()

        # Add memory usage to logs
        if torch.cuda.is_available():
            metrics.update({
                "gpu_mem_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_mem_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "gpu_mem_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                "gpu_mem_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            })

        print(json.dumps({"ts": ts, **metrics}))

    print("Starting training...")
    print_memory_usage("Training start")

    try:
        distiller.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            steps=args.train_steps,
            grad_accum_steps=args.grad_accum_steps,
            eval_every=args.eval_steps,
            save_every=args.save_steps,
            save_callback=save_callback,
            log_callback=log_callback,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OutOfMemoryError: {e}")
        print_memory_usage("OOM Error")
        # Try to recover some memory
        aggressive_memory_cleanup()
        print_memory_usage("After OOM cleanup")
        raise
    except Exception as e:
        print(f"Training error: {e}")
        print_memory_usage("Training error")
        raise
    finally:
        # Final cleanup
        print("Training completed, performing final cleanup...")
        aggressive_memory_cleanup()
        print_memory_usage("Final cleanup")


if __name__ == '__main__':
    main()
