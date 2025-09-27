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

## Memory Optimization for Constrained Devices

For GPUs with limited memory (8GB or less), use these memory-saving techniques:

### Mixed Precision Training (FP16)
Use `--model_dtype float16` for mixed precision training (FP32 parameters, FP16 computations) to reduce memory usage by ~50%:

```bash
python train_sar_kd.py \
  --teacher_model huihui-ai/Huihui-MoE-1B-A0.6B \
  --student_model HuggingFaceTB/SmolLM-135M \
  --model_dtype float16 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --block_size 512 \
  --train_steps 1000 \
  --output_dir outputs/sar_kd_fp16
```

### Additional Memory Saving Options
- **Smaller batch size**: Use `--per_device_batch_size 1` with higher `--grad_accum_steps`
- **Shorter sequences**: Reduce `--block_size` from 1024 to 512 or 256
- **Gradient checkpointing**: Automatically enabled for supported models
- **Higher gradient accumulation**: Use `--grad_accum_steps 16` or higher to maintain effective batch size

### Memory Usage Estimates
- **FP32 (default)**: ~14-16GB for teacher + student + gradients
- **FP16**: ~8-10GB for teacher + student + gradients
- **FP16 + optimizations**: ~6-8GB total memory usage

### Example for 8GB GPU (RTX 2070, GTX 1080):
```bash
python train_sar_kd.py \
  --model_dtype float16 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --block_size 512 \
  --train_steps 1000
```

### Memory Estimation Tool
To estimate memory usage before training:

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def estimate_memory(model_name, dtype_str='float32'):
    # Mixed precision: parameters in FP32, computations in FP16
    dtype = torch.float32  # Parameters always in FP32 for mixed precision
    use_fp16 = dtype_str == 'float16'
    print(f'Estimating memory for {model_name} with {"mixed precision" if use_fp16 else "FP32"}...')
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map='cpu'
        )
        param_count = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4  # Always FP32 for parameters
        model_size_gb = (param_count * bytes_per_param) / (1024**3)
        
        # Mixed precision saves memory on activations and intermediate computations
        training_multiplier = 4 if not use_fp16 else 2.5
        total_estimate = model_size_gb * training_multiplier
        
        print(f'  Parameters: {param_count:,}')
        print(f'  Model size: {model_size_gb:.2f} GB')
        print(f'  Training estimate: {total_estimate:.2f} GB')
        return total_estimate
    except Exception as e:
        print(f'  Error: {e}')
        return None

# Estimate both models with mixed precision
teacher_mem = estimate_memory('huihui-ai/Huihui-MoE-1B-A0.6B', 'float16')
student_mem = estimate_memory('HuggingFaceTB/SmolLM-135M', 'float16')

if teacher_mem and student_mem:
    total = teacher_mem + student_mem
    print(f'Total estimated memory: {total:.2f} GB')
    if total > 8:
        print('Recommendation: Use additional optimizations for 8GB GPUs')
    elif total > 16:
        print('Recommendation: Use 24GB+ GPU or reduce model sizes')
"
```

### Quick Start Commands by GPU Memory

**For 6-8GB GPUs (RTX 3060, GTX 1060):**
```bash
python train_sar_kd.py \
  --model_dtype float16 \
  --per_device_batch_size 1 \
  --grad_accum_steps 32 \
  --block_size 256 \
  --train_steps 500
```

**For 10-12GB GPUs (RTX 3080, RTX 2080 Ti):**
```bash
python train_sar_kd.py \
  --model_dtype float16 \
  --per_device_batch_size 1 \
  --grad_accum_steps 16 \
  --block_size 512 \
  --train_steps 1000
```

**For 16GB+ GPUs (RTX 4090, V100):**
```bash
python train_sar_kd.py \
  --model_dtype float16 \
  --per_device_batch_size 2 \
  --grad_accum_steps 8 \
  --block_size 1024 \
  --train_steps 1000 \
  --use_scheduler --warmup_steps 100
```

### Advanced Memory Saving Tips
- **Mixed precision**: `--model_dtype float16` uses FP16 computations with FP32 parameters for stability
- **Gradient accumulation**: Use higher `--grad_accum_steps` to reduce per-step memory
- **Sequence length**: Start with `--block_size 256`, increase gradually
- **Batch size**: Keep `--per_device_batch_size 1` for maximum memory savings
- **Router-only training**: Only router parameters are trained, reducing optimizer memory
- **Gradient scaling**: Automatic gradient scaling prevents underflow in mixed precision

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
--model_dtype float16             # Use mixed precision (FP16 computations, FP32 parameters)
--save_steps 500                  # Save model every N steps
```

Caveats:
- Some MoE architectures name router layers differently. The router finder searches common patterns ("gate", "router", "moe") and explicitly logs what it found. If no router layers are detected, consider using `--router_patterns` to specify custom patterns.
- The dual-tokenizer mode automatically handles incompatible tokenizers through hidden-state alignment - no manual intervention required.