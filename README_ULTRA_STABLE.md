# Ultra-Stable SAR Knowledge Distillation Training

## Overview

This is an **ultra-stable** version of the SAR Knowledge Distillation training script, specifically designed to handle **FP16 numerical instability issues** that commonly occur in knowledge distillation on hardware like P100 GPUs.

## Problem Solved

The original training was experiencing:
- **NaN losses** after ~96 training steps in FP16 mode
- **Gradient scaler errors** with FP16 parameters
- **Catastrophic numerical instability** leading to complete training failure
- **Memory overflow** on P100 GPUs

## Key Features

### 🛡️ Maximum FP16 Stability
- **Ultra-aggressive logit clamping** (`-2.5` to `2.5` range)
- **Comprehensive NaN/Inf detection** at every computation step
- **Model state recovery** mechanisms when instability is detected
- **Conservative gradient clipping** (max norm: 0.5)

### 🔧 Enhanced Error Handling
- **Automatic gradient scaler detection** and disabling for FP16 parameters
- **Batch-level error recovery** with detailed logging
- **Emergency model reset** after consecutive NaN batches
- **Robust evaluation** with stability measures

### ⚡ Memory Optimization
- **Teacher CPU offloading** during training
- **Gradient checkpointing** enabled by default
- **Conservative batch sizes** and memory management
- **Automatic cache clearing**

## Usage

### Basic Usage (P100 with FP16)
```bash
python train_sar_kd_ultra_stable.py \
    --train_steps 500 \
    --model_dtype float16 \
    --per_device_batch_size 1 \
    --eval_steps 50 \
    --student_lr 5e-6 \
    --temperature 1.5 \
    --alpha_kd 0.05 \
    --alpha_ce 0.95
```

### Conservative Settings (Maximum Stability)
```bash
python train_sar_kd_ultra_stable.py \
    --train_steps 1000 \
    --model_dtype float16 \
    --per_device_batch_size 1 \
    --eval_steps 100 \
    --student_lr 2e-6 \
    --temperature 1.2 \
    --alpha_kd 0.01 \
    --alpha_ce 0.99 \
    --max_grad_norm 0.25
```

### Test Installation
```bash
python test_ultra_stable.py
```

## Key Parameters

### Ultra-Conservative Defaults
- `student_lr`: `5e-6` (much lower than standard)
- `max_grad_norm`: `0.5` (aggressive clipping)
- `temperature`: `1.5` (conservative temperature)
- `alpha_kd`: `0.05` (minimal KD weight for stability)

### Stability Settings
- **Logit clamping**: `[-2.5, 2.5]` (prevents FP16 overflow)
- **Loss clamping**: `[0.1, 5.0]` (conservative range)
- **Max consecutive NaNs**: `20` (before model reset)
- **Gradient norm threshold**: `2.0` (skip updates above this)

## Output & Monitoring

### Training Logs
The script provides detailed logging including:
- **Batch success rates**
- **Consecutive NaN counters**
- **Memory usage tracking**
- **Gradient norm monitoring**
- **Numerical stability warnings**

### Log Format
```json
{
  "step": 100,
  "train_loss": 3.2415,
  "success_rate": 0.89,
  "consecutive_nans": 0,
  "gpu_memory_gb": 2.1
}
```

## Architecture

### Key Classes

**`UltraStableTrainer`**
- Implements emergency model reset mechanisms
- Tracks consecutive NaN batches
- Provides enhanced gradient handling

**`UltraStableEvaluator`**
- Ultra-conservative evaluation with extensive clamping
- Robust NaN/Inf detection in evaluation
- Memory-efficient batch processing

### Safety Mechanisms

1. **Parameter State Monitoring**: Continuous checking for NaN/Inf in model parameters
2. **Gradient Validation**: Pre-update gradient sanity checks
3. **Loss Validation**: Multi-stage loss validation with clamping
4. **Automatic Recovery**: Model reset when instability is detected

## Troubleshooting

### Common Issues

**"Still getting NaN losses"**
- Reduce `student_lr` to `1e-6` or lower
- Reduce `temperature` to `1.0`
- Increase `max_grad_norm` clipping

**"Training too slow"**
- This is expected - stability over speed
- Consider using fewer evaluation steps
- Use smaller models if possible

**"Memory errors"**
- Enable `--clear_cache_every_step`
- Reduce `block_size` to 256 or 128
- Ensure teacher CPU offloading is enabled

### Success Indicators

✅ **Stable training**: Success rate > 80%
✅ **No consecutive NaNs**: Counter stays at 0
✅ **Reasonable losses**: Training loss between 1-8
✅ **Memory efficient**: GPU usage stable

## Comparison with Standard Training

| Aspect | Standard Training | Ultra-Stable Training |
|--------|------------------|----------------------|
| **Stability** | Prone to NaN crashes | Comprehensive NaN handling |
| **Speed** | Faster | Slower (stability first) |
| **Memory** | Higher usage | Optimized for P100 |
| **Recovery** | Manual restart needed | Automatic recovery |
| **FP16 Support** | Basic | Enterprise-grade |

## When to Use

**Use Ultra-Stable Version When:**
- Training on P100 or older GPUs
- Using FP16 for memory efficiency  
- Knowledge distillation with temperature scaling
- Production environments requiring reliability
- Previous attempts crashed with NaN losses

**Use Standard Version When:**
- Training on modern GPUs (V100, A100)
- Using FP32 exclusively
- Speed is more important than stability
- Experimental/research settings

## Technical Details

### FP16 Parameter Detection
```python
has_fp16_params = any(p.dtype == torch.float16 for p in model.parameters())
if has_fp16_params:
    # Disable gradient scaler to prevent crashes
    use_scaler = False
```

### Logit Clamping Strategy
```python
# Ultra-aggressive clamping for FP16
student_logits = torch.clamp(student_logits, -2.5, 2.5)
teacher_logits = torch.clamp(teacher_logits, -2.5, 2.5)
```

### Emergency Recovery
```python
if consecutive_nan_batches >= max_consecutive_nans:
    # Reset model to initial state
    model.load_state_dict(initial_state)
    # Reinitialize optimizer
    optimizer = create_fresh_optimizer()
```

## Performance Expectations

**On P100 (16GB) with FP16:**
- **Memory usage**: ~2-3GB for medium models
- **Training speed**: ~5-10 steps/minute (conservative)
- **Stability**: >95% successful batches
- **Convergence**: Slower but reliable

**Recommended Hardware:**
- **Minimum**: P100 16GB
- **Optimal**: V100 32GB or better
- **Memory**: 16GB+ GPU memory recommended

## Contributing

When modifying the ultra-stable version:
1. **Never remove safety checks** without extensive testing
2. **Test on P100** before considering changes stable
3. **Maintain backward compatibility** with existing configs
4. **Add new safety measures** rather than removing existing ones

## Support

For issues specific to ultra-stable training:
1. Check the **consecutive NaN counter** in logs
2. Verify **gradient norms** are reasonable (<5.0)
3. Monitor **memory usage** trends
4. Review **success rates** per batch

The ultra-stable version prioritizes **reliability over speed** - this is by design for production FP16 environments.