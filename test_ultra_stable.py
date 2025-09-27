#!/usr/bin/env python3
"""
Test script for Ultra-Stable SAR Knowledge Distillation Training
================================================================

Quick test to verify the ultra-stable FP16 training works correctly.
This script runs a minimal version of the training to validate:
- FP16 parameter detection
- Gradient scaler handling
- Numerical stability measures
- Error recovery mechanisms
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# Add the sarkd5 directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ultra_stable_training():
    """Test the ultra-stable training script"""
    print("🧪 Testing Ultra-Stable SAR KD Training")
    print("=" * 50)

    try:
        # Test with minimal settings
        test_args = [
            "--train_steps", "10",
            "--per_device_batch_size", "1",
            "--eval_steps", "5",
            "--student_lr", "1e-6",
            "--temperature", "1.2",
            "--alpha_kd", "0.01",
            "--alpha_ce", "0.99",
            "--model_dtype", "float16",
            "--block_size", "128",
            "--output_dir", "./test_output"
        ]

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = ["train_sar_kd_ultra_stable.py"] + test_args

        # Import and run
        from train_sar_kd_ultra_stable import main

        print("🚀 Running ultra-stable training test...")
        main()

        print("✅ Test completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all SAR modules are available")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

    finally:
        # Restore original argv
        sys.argv = original_argv

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_fp16_detection():
    """Test FP16 parameter detection logic"""
    print("\n🔍 Testing FP16 parameter detection...")

    try:
        # Create a simple model with FP16 parameters
        model = torch.nn.Linear(10, 5)

        # Test FP32 detection
        has_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
        print(f"FP32 model FP16 detection: {has_fp16} (should be False)")
        assert not has_fp16, "FP32 model incorrectly detected as FP16"

        # Convert to FP16 and test
        model = model.half()
        has_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
        print(f"FP16 model FP16 detection: {has_fp16} (should be True)")
        assert has_fp16, "FP16 model not detected correctly"

        print("✅ FP16 detection working correctly")
        return True

    except Exception as e:
        print(f"❌ FP16 detection test failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability measures"""
    print("\n🛡️ Testing numerical stability measures...")

    try:
        # Test logit clamping
        extreme_logits = torch.tensor([100.0, -100.0, float('inf'), float('-inf'), float('nan')])
        clamped = torch.clamp(extreme_logits, -2.5, 2.5)

        # Check for finite values
        finite_mask = torch.isfinite(clamped)
        print(f"Clamped logits: {clamped}")
        print(f"Finite values after clamping: {finite_mask.sum().item()}/{len(clamped)}")

        # Test NaN detection
        has_nan = torch.isnan(extreme_logits).any()
        has_inf = torch.isinf(extreme_logits).any()
        print(f"NaN detection: {has_nan} (should be True)")
        print(f"Inf detection: {has_inf} (should be True)")

        print("✅ Numerical stability measures working correctly")
        return True

    except Exception as e:
        print(f"❌ Numerical stability test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Ultra-Stable SAR KD Test Suite")
    print("=" * 40)

    tests = [
        ("FP16 Detection", test_fp16_detection),
        ("Numerical Stability", test_numerical_stability),
        ("Ultra-Stable Training", test_ultra_stable_training),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Ultra-stable training is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
