# Student-Aware Router Knowledge Distillation (SAR-KD)

This project implements Knowledge Distillation from a Mixture-of-Experts (MoE) teacher using a Student-Aware Router (SAR). The router parameters in the teacher are the only trainable teacher parameters; all experts remain frozen. The student is trained jointly while the teacher's router adapts to produce mixtures that are easier for the student to learn.

Teacher: huihui-ai/Huihui-MoE-1B-A0.6B (MoE, ~1B total, ~0.6B active per token)
Student: HuggingFaceTB/SmolLM-135M (dense ~135M)

Key features:
- Joint optimization of student and teacher router parameters via KD.
- KL distillation on logits with temperature, plus optional CE to ground-truth tokens.
- Router regularization: anchor-to-initial L2, entropy bonus, and load-balancing penalty.
- GPU/CPU support, gradient accumulation, FP16, gradient checkpointing.

Tokenizer compatibility:
- If tokenizers are compatible (same class, vocab size, identical behavior on probes), the distiller performs logit-level KD (KL with temperature) and trains the teacher router.
- If tokenizers differ, the system automatically uses a sophisticated dual-tokenizer mode: it aligns teacher and student hidden states by character-span overlaps using offset mappings from fast tokenizers, and optimizes an MSE between the student hidden states and a projection of teacher hidden states. This preserves differentiability into the teacher's router. CE on the student labels is still applied.

Run (example on Kaggle P100):

```
python train_sar_kd.py \
  --teacher_model huihui-ai/Huihui-MoE-1B-A0.6B \
  --student_model HuggingFaceTB/SmolLM-135M \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --block_size 1024 \
  --train_steps 1000 \
  --per_device_batch_size 1 \
  --grad_accum_steps 8 \
  --student_lr 1e-4 \
  --router_lr 5e-4 \
  --temperature 2.0 \
  --alpha_kd 0.9 \
  --alpha_ce 0.1 \
  --router_anchor_weight 1e-4 \
  --router_load_balance_weight 1e-3 \
  --router_entropy_weight 1e-4 \
  --eval_steps 200 \
  --save_steps 500 \
  --output_dir outputs/sar_kd
```

On Kaggle P100 (16GB), this configuration keeps memory usage moderate by using small batch sizes with gradient accumulation. Increase steps as budget allows.

Outputs:
- Student model: saved to `--output_dir/student/`
- Router update (teacher side): `--output_dir/router_update.pt` (contains only router parameters); original teacher weights remain unchanged.

Evaluation:
- The script reports student validation perplexity during training.

Advanced features:
- Learning rate scheduling: Use `--use_scheduler` with `--warmup_steps` and `--scheduler_type` (linear/cosine) for improved convergence
- Custom router discovery: Use `--router_patterns` to provide custom regex patterns for finding router layers in models with non-standard naming
- Mixed precision: Automatic support for FP16 models with proper gradient scaling

Additional options:
```
--use_scheduler                    # Enable learning rate scheduler
--warmup_steps 100                # Number of warmup steps
--scheduler_type cosine           # Scheduler type: linear or cosine
--router_patterns gate router     # Custom regex patterns for router discovery
```

Caveats:
- Some MoE architectures name router layers differently. The router finder searches common patterns ("gate", "router", "moe") and explicitly logs what it found. If no router layers are detected, consider using `--router_patterns` to specify custom patterns.
- The dual-tokenizer mode automatically handles incompatible tokenizers through hidden-state alignment - no manual intervention required.