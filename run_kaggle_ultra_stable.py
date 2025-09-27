#!/usr/bin/env python3
"""
Kaggle-Specific Ultra-Stable SAR Knowledge Distillation Runner
=============================================================

FIXED VERSION - Constructor parameter order corrected.
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
    print("✅ CONSTRUCTOR ISSUE FIXED - Ready for stable FP16 training!")
    print("=" * 60)

    # Kaggle P100 optimized command - FIXED VERSION
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

    print("🚀 Fixed command to execute:")
    print(" ".join(cmd))
    print()

    # Show what was fixed
    print("🔧 FIXES APPLIED:")
    print("  ✅ Constructor parameter order: SARDistiller(teacher, student, device, cfg)")
    print("  ✅ Correct imports from sar_kd modules")
    print("  ✅ Proper config parameters (total_steps, scheduler_type)")
    print("  ✅ No optimizer override conflicts")
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
        print("\n🛠️ Troubleshooting:")
        print("  1. Check if all required modules are available")
        print("  2. Verify GPU memory is sufficient")
        print("  3. Try the backup stable version if needed")
        return e.returncode

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0

def run_backup_stable():
    """Run the proven stable version as backup"""
    print("🔄 Running backup stable version...")

    cmd = [
        sys.executable, "/kaggle/working/sarkd5/train_sar_kd_stable.py",
        "--train_steps", "500",
        "--model_dtype", "float16",
        "--per_device_batch_size", "1",
        "--eval_steps", "50",
        "--student_lr", "1e-5",
        "--temperature", "2.0",
        "--alpha_kd", "0.1",
        "--alpha_ce", "0.9"
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("✅ Backup stable training completed!")
        return 0
    except Exception as e:
        print(f"❌ Backup stable training also failed: {e}")
        return 1

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

            # Check if it's P100 (where the original NaN issues occurred)
            if "P100" in gpu_name:
                print("  🎯 P100 detected - ultra-stable mode is ideal for this GPU!")

        else:
            print(f"  CUDA available: ❌")
    except ImportError:
        print(f"  PyTorch not available: ❌")
        return False

    # Check if our files exist
    ultra_stable_exists = os.path.exists("/kaggle/working/sarkd5/train_sar_kd_ultra_stable.py")
    stable_exists = os.path.exists("/kaggle/working/sarkd5/train_sar_kd_stable.py")

    print(f"  Ultra-stable script: {'✅' if ultra_stable_exists else '❌'}")
    print(f"  Backup stable script: {'✅' if stable_exists else '❌'}")

    if not ultra_stable_exists and not stable_exists:
        print("\n⚠️  No training scripts found!")
        print("Make sure to upload the sarkd5 directory to /kaggle/working/")
        return False

    # Check sar_kd module
    try:
        sys.path.append('/kaggle/working/sarkd5')
        from sar_kd.trainer import SARDistiller, SARConfig
        print("  SAR modules: ✅")
    except ImportError as e:
        print(f"  SAR modules: ❌ ({e})")
        return False

    return True

def main():
    """Main entry point"""
    print("🎯 Kaggle Ultra-Stable SAR KD Runner (FIXED)")
    print("=" * 50)

    # Check environment
    if not check_kaggle_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return 1

    print("\n" + "=" * 50)

    # Show options
    print("📋 Available options:")
    print("  1. Ultra-stable FP16 training (FIXED - recommended)")
    print("  2. Backup stable training (fallback)")
    print("  3. Cancel")

    try:
        choice = input("\nChoose option (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0

    if choice == "1":
        print("\n🛡️ Starting FIXED ultra-stable training...")
        return run_kaggle_ultra_stable()
    elif choice == "2":
        print("\n🔄 Starting backup stable training...")
        return run_backup_stable()
    elif choice == "3":
        print("Training cancelled.")
        return 0
    else:
        print("Invalid choice. Defaulting to ultra-stable training...")
        return run_kaggle_ultra_stable()

if __name__ == "__main__":
    exit(main())
