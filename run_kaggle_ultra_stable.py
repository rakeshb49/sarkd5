#!/usr/bin/env python3
"""
Kaggle-Specific Ultra-Stable SAR Knowledge Distillation Runner
=============================================================

Optimized command for running ultra-stable FP16 training on Kaggle P100 environment.
This script automatically configures all the optimal settings for maximum stability.
"""

import subprocess
import sys
import os

def run_kaggle_ultra_stable():
    """Run ultra-stable training with Kaggle P100 optimized settings"""

    print("🎯 KAGGLE P100 Ultra-Stable SAR Knowledge Distillation")
    print("=" * 60)
    print("Automatically configured for maximum FP16 stability on P100")
    print("=" * 60)

    # Kaggle P100 optimized command
    cmd = [
        sys.executable, "/kaggle/working/sarkd5/train_sar_kd_ultra_stable.py",

        # Model configuration - same as original command
        "--teacher_model", "microsoft/DialoGPT-large",   # Same as original
        "--student_model", "microsoft/DialoGPT-small",   # Same as original
        "--model_dtype", "float16",                      # FP16 for memory efficiency

        # Ultra-conservative training settings
        "--train_steps", "500",
        "--per_device_batch_size", "1",                  # Minimal batch size
        "--grad_accum_steps", "1",                       # No accumulation for stability
        "--eval_steps", "50",
        "--save_steps", "250",
        "--clear_cache_every_step",                      # Aggressive cache clearing

        # Ultra-stable learning rates (even more conservative for Kaggle)
        "--student_lr", "3e-6",                          # Very low LR
        "--router_lr", "1e-5",                           # Conservative router LR
        "--weight_decay", "0.01",
        "--max_grad_norm", "0.3",                        # Very aggressive clipping

        # Knowledge distillation - maximum stability
        "--temperature", "1.2",                          # Low temperature
        "--alpha_kd", "0.02",                            # Minimal KD weight
        "--alpha_ce", "0.98",                            # Focus on CE loss

        # Router regularization - minimal for stability
        "--router_anchor_weight", "0.0005",
        "--router_load_balance_weight", "0.0005",
        "--router_entropy_weight", "0.0005",

        # Data configuration - conservative
        "--dataset_name", "wikitext",
        "--dataset_config_name", "wikitext-103-raw-v1",
        "--block_size", "384",                           # Moderate context length

        # System configuration
        "--output_dir", "/kaggle/working/sar_outputs",
        "--seed", "42",
        "--offload_teacher_to_cpu",                      # Essential for P100
        "--use_scheduler",
    ]

    print("🚀 Command to execute:")
    print(" ".join(cmd))
    print()

    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)

        print("✅ Training completed successfully!")
        print("\n📊 Training output:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print(f"\n📋 Error output:")
        print(e.stdout)
        return e.returncode

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0

def check_kaggle_environment():
    """Check if running in Kaggle environment and print system info"""
    print("🔍 Checking Kaggle environment...")

    # Check if we're in Kaggle
    is_kaggle = os.path.exists('/kaggle')
    print(f"  Kaggle environment: {'✅' if is_kaggle else '❌'}")

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  CUDA available: ✅")
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.1f}GB")
        else:
            print(f"  CUDA available: ❌")
    except ImportError:
        print(f"  PyTorch not available: ❌")
        return False

    # Check if our files exist
    script_exists = os.path.exists("/kaggle/working/sarkd5/train_sar_kd_ultra_stable.py")
    print(f"  Ultra-stable script: {'✅' if script_exists else '❌'}")

    if not script_exists:
        print("\n⚠️  Ultra-stable training script not found!")
        print("Make sure to upload the sarkd5 directory to /kaggle/working/")
        return False

    return True

def main():
    """Main entry point"""
    print("🎯 Kaggle Ultra-Stable SAR KD Runner")
    print("=" * 40)

    # Check environment
    if not check_kaggle_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return 1

    print("\n" + "=" * 40)

    # Ask for confirmation
    response = input("\n🚀 Ready to start ultra-stable training? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return 0

    # Run training
    return run_kaggle_ultra_stable()

if __name__ == "__main__":
    exit(main())
