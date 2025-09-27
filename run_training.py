#!/usr/bin/env python3
"""
Simple launcher script for SAR Knowledge Distillation training
This script provides an easy way to run the training with optimal settings.
"""

import subprocess
import sys
import os

def run_training():
    """Run the SAR Knowledge Distillation training with optimal parameters"""

    print("🚀 Starting SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Using optimized parameters for stable training...")
    print()

    # Optimal parameters based on debugging results
    cmd = [
        sys.executable,  # Use current Python interpreter
        "train_sar_kd_final_safe.py",
        "--teacher_model", "microsoft/DialoGPT-medium",
        "--student_model", "distilgpt2",
        "--model_dtype", "float16",
        "--train_steps", "500",
        "--student_lr", "1e-6",          # Reduced for stability
        "--max_grad_norm", "15.0",       # Relaxed gradient clipping
        "--temperature", "2.0",
        "--alpha", "0.7",
        "--train_batch_size", "2",
        "--eval_batch_size", "2",
        "--grad_accum_steps", "4",
        "--eval_steps", "50",
        "--block_size", "512",
        "--warmup_steps", "50"
    ]

    print("Command:", " ".join(cmd))
    print()

    # Check if the training script exists
    if not os.path.exists("train_sar_kd_final_safe.py"):
        print("❌ Error: train_sar_kd_final_safe.py not found!")
        print("Make sure you're running this from the correct directory.")
        return False

    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def run_training_extended():
    """Run training with extended parameters for better results"""

    print("🚀 Starting Extended SAR Knowledge Distillation Training")
    print("=" * 60)
    print("Using extended parameters for better convergence...")
    print()

    cmd = [
        sys.executable,
        "train_sar_kd_final_safe.py",
        "--teacher_model", "microsoft/DialoGPT-medium",
        "--student_model", "distilgpt2",
        "--model_dtype", "float16",
        "--train_steps", "1000",         # More training steps
        "--student_lr", "2e-6",          # Slightly higher LR
        "--max_grad_norm", "10.0",       # Moderate clipping
        "--temperature", "1.5",          # Lower temperature
        "--alpha", "0.6",                # More balanced KD/CE
        "--train_batch_size", "2",
        "--eval_batch_size", "2",
        "--grad_accum_steps", "4",
        "--eval_steps", "100",           # Less frequent eval
        "--block_size", "512",
        "--warmup_steps", "100"          # More warmup
    ]

    print("Command:", " ".join(cmd))
    print()

    if not os.path.exists("train_sar_kd_final_safe.py"):
        print("❌ Error: train_sar_kd_final_safe.py not found!")
        return False

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Extended training completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def run_training_fast():
    """Run a quick training test to verify everything works"""

    print("🧪 Running Quick Training Test")
    print("=" * 60)
    print("Testing with minimal parameters...")
    print()

    cmd = [
        sys.executable,
        "train_sar_kd_final_safe.py",
        "--teacher_model", "microsoft/DialoGPT-medium",
        "--student_model", "distilgpt2",
        "--model_dtype", "float16",
        "--train_steps", "50",           # Very short test
        "--student_lr", "1e-6",
        "--max_grad_norm", "15.0",
        "--temperature", "2.0",
        "--alpha", "0.7",
        "--train_batch_size", "1",       # Smaller batch
        "--eval_batch_size", "1",
        "--grad_accum_steps", "2",
        "--eval_steps", "25",
        "--block_size", "256",           # Shorter sequences
        "--warmup_steps", "10"
    ]

    print("Command:", " ".join(cmd))
    print()

    if not os.path.exists("train_sar_kd_final_safe.py"):
        print("❌ Error: train_sar_kd_final_safe.py not found!")
        return False

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Quick test completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main function with menu"""

    print("🎯 SAR Knowledge Distillation Training Launcher")
    print("=" * 60)
    print()
    print("Choose training mode:")
    print("1. Standard Training (500 steps, optimized parameters)")
    print("2. Extended Training (1000 steps, better convergence)")
    print("3. Quick Test (50 steps, verify setup)")
    print("4. Exit")
    print()

    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                success = run_training()
                break
            elif choice == "2":
                success = run_training_extended()
                break
            elif choice == "3":
                success = run_training_fast()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                return
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue

        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return

    if success:
        print("\n🎉 All done! Check the output above for training results.")
    else:
        print("\n💡 If training failed, check the error messages above.")
        print("   Common fixes:")
        print("   - Ensure CUDA is available for GPU training")
        print("   - Check disk space for model downloads")
        print("   - Reduce batch size if out of memory")

if __name__ == "__main__":
    main()
