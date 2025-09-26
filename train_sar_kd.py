import argparse
import json
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
from datetime import datetime, UTC

import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

from sar_kd.data import build_text_datasets, DualTokenizerCollator
from sar_kd.models import load_teacher_student
from sar_kd.trainer import SARConfig, SARDistiller


def parse_args():
    p = argparse.ArgumentParser(description="Student-Aware Router KD training")
    p.add_argument('--teacher_model', type=str, default='huihui-ai/Huihui-MoE-1B-A0.6B')
    p.add_argument('--student_model', type=str, default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--dataset_name', type=str, default='wikitext')
    p.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1')
    p.add_argument('--block_size', type=int, default=1024)
    p.add_argument('--per_device_batch_size', type=int, default=1)
    p.add_argument('--grad_accum_steps', type=int, default=8)
    p.add_argument('--train_steps', type=int, default=1000)
    p.add_argument('--eval_steps', type=int, default=200)
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
    p.add_argument('--model_dtype', type=str, default='float32', choices=['float16', 'float32'], help='Model precision: float16 for memory efficiency, float32 for stability')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='outputs/sar_kd')
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.model_dtype == 'float16' else torch.float32
    print(f"Using model dtype: {dtype}")
    # Enable TF32 when on CUDA and better matmul precision on CPU
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    teacher, student, teacher_tok, student_tok = load_teacher_student(
        args.teacher_model, args.student_model, dtype, device
    )

    train_ds, eval_ds, same_tok = build_text_datasets(
        args.dataset_name, args.dataset_config_name, teacher_tok, student_tok, args.block_size
    )

    if same_tok:
        collator = DataCollatorForLanguageModeling(teacher_tok, mlm=False)
    else:
        collator = DualTokenizerCollator(teacher_tok, student_tok, args.block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

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
    )

    distiller = SARDistiller(teacher, student, device, cfg, tokenizers_compatible=same_tok)

    def save_callback(step, student_model, router_linears):
        step_dir = os.path.join(args.output_dir)
        os.makedirs(step_dir, exist_ok=True)
        # save student
        student_dir = os.path.join(step_dir, 'student')
        os.makedirs(student_dir, exist_ok=True)
        student_model.save_pretrained(student_dir)
        # save tokenizer compatible with the student
        try:
            teacher_tok.save_pretrained(student_dir)
        except Exception:
            pass
        # save router update only
        router_state = {}
        for name, lin in router_linears:
            for pname, p in lin.named_parameters(recurse=False):
                router_state[f"{name}.{pname}"] = p.detach().cpu()
        torch.save(router_state, os.path.join(step_dir, 'router_update.pt'))

    def log_callback(metrics: dict):
        ts = datetime.now(UTC).isoformat()
        print(json.dumps({"ts": ts, **metrics}))

    distiller.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        steps=args.train_steps,
        grad_accum_steps=args.grad_accum_steps,
        eval_every=args.eval_steps,
        save_callback=save_callback,
        log_callback=log_callback,
    )


if __name__ == '__main__':
    main()
