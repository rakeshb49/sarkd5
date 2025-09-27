# 🛡️ ULTRA-STABLE KAGGLE COMMAND

## Fixed Ultra-Stable SAR Knowledge Distillation Command for P100

Use this command in your Kaggle notebook to run the ultra-stable FP16 training:

```bash
!python /kaggle/working/sarkd5/train_sar_kd_ultra_stable.py \
    --teacher_model microsoft/DialoGPT-large \
    --student_model microsoft/DialoGPT-small \
    --model_dtype float16 \
    --train_steps 500 \
    --per_device_batch_size 1 \
    --eval_steps 50 \
    --student_lr 3e-6 \
    --temperature 1.2 \
    --alpha_kd 0.02 \
    --alpha_ce 0.98 \
    --max_grad_norm 0.3 \
    --router_anchor_weight 0.0005 \
    --router_load_balance_weight 0.0005 \
    --router_entropy_weight 0.0005 \
    --block_size 384 \
    --output_dir /kaggle/working/sar_outputs \
    --offload_teacher_to_cpu \
    --use_scheduler \
    --clear_cache_every_step
```

## 🔧 Key Differences from Original Command

| Parameter | Original | Ultra-Stable | Reason |
|-----------|----------|--------------|---------|
| `student_lr` | `1e-5` | `3e-6` | Prevent gradient explosion |
| `temperature` | `2.0` | `1.2` | Reduce numerical instability |
| `alpha_kd` | `0.1` | `0.02` | Minimize KD loss contribution |
| `alpha_ce` | `0.9` | `0.98` | Focus on stable CE loss |
| `max_grad_norm` | `1.0` | `0.3` | Aggressive gradient clipping |
| `block_size` | `512` | `384` | Reduce memory pressure |

## 📊 Expected Behavior

**✅ Success Indicators:**
- No "CE loss issue: ce=nan" messages
- Success rate > 90%
- Consecutive NaN counter stays at 0
- Stable GPU memory usage (~2-3GB)

**⚠️ Warning Signs:**
- Multiple "NaN/Inf in logits" warnings
- Success rate < 80% 
- "EMERGENCY: Resetting" messages
- Memory usage spikes

## 🔍 Monitoring Commands

Check training progress:
```bash
!tail -f /kaggle/working/sar_outputs/training.jsonl
```

Monitor GPU memory:
```bash
!nvidia-smi
```

## 🛠️ Troubleshooting

**If you still see NaN losses:**
1. Reduce `student_lr` to `1e-6`
2. Reduce `temperature` to `1.0`
3. Set `alpha_kd` to `0.01`

**If memory errors:**
1. Reduce `block_size` to `256`
2. Add `--grad_accum_steps 2` 
3. Ensure teacher CPU offloading is working

**If training is too slow:**
- This is expected - stability over speed
- Consider reducing `train_steps` to `300`
- Use fewer `eval_steps` (e.g., `100`)

## ⚡ Quick Test Command

For a quick test (50 steps):
```bash
!python /kaggle/working/sarkd5/train_sar_kd_ultra_stable.py \
    --train_steps 50 \
    --eval_steps 25 \
    --model_dtype float16 \
    --student_lr 2e-6 \
    --temperature 1.0 \
    --alpha_kd 0.01 \
    --alpha_ce 0.99
```

This ultra-stable version should eliminate the NaN cascade that occurred at step 96 in your original run!