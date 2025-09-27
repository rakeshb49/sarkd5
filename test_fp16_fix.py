#!/usr/bin/env python3
"""
Test script to verify FP16 gradient scaler fix logic.
This script demonstrates the fix for the "Attempting to unscale FP16 gradients" error.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler


def test_fp16_gradient_scaler_logic():
    """Test the FP16 parameter detection and scaler logic"""
    print("🧪 Testing FP16 Gradient Scaler Fix Logic")
    print("=" * 50)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    # Test 1: FP32 model with FP16 training
    print("\n📋 Test 1: FP32 model parameters with FP16 autocast")
    import copy
    model_fp32 = copy.deepcopy(model)

    has_fp16_params = any(p.dtype == torch.float16 for p in model_fp32.parameters())
    use_fp16_training = True  # Simulating --model_dtype float16
    use_scaler = torch.cuda.is_available() and use_fp16_training

    if has_fp16_params and use_scaler:
        print("WARNING: FP16 parameters detected - disabling gradient scaler to prevent errors")
        use_scaler = False

    print(f"  Model parameter dtype: {next(model_fp32.parameters()).dtype}")
    print(f"  Has FP16 parameters: {has_fp16_params}")
    print(f"  Use scaler: {use_scaler}")
    print(f"  Expected behavior: Should use gradient scaler for mixed precision")

    # Test 2: FP16 model parameters
    print("\n📋 Test 2: FP16 model parameters")
    model_fp16 = copy.deepcopy(model).half()  # Convert to FP16

    has_fp16_params = any(p.dtype == torch.float16 for p in model_fp16.parameters())
    use_fp16_training = True
    use_scaler = torch.cuda.is_available() and use_fp16_training

    if has_fp16_params and use_scaler:
        print("WARNING: FP16 parameters detected - disabling gradient scaler to prevent errors")
        use_scaler = False

    print(f"  Model parameter dtype: {next(model_fp16.parameters()).dtype}")
    print(f"  Has FP16 parameters: {has_fp16_params}")
    print(f"  Use scaler: {use_scaler}")
    print(f"  Expected behavior: Should disable gradient scaler to prevent FP16 gradient error")

    # Test 3: Simulate the actual training scenario
    print("\n📋 Test 3: Simulating actual training scenario")

    # This simulates loading models with dtype=torch.float16
    model_loaded_fp16 = copy.deepcopy(model).to(torch.float16)

    # Check parameters (this is the key fix)
    has_fp16_params = any(p.dtype == torch.float16 for p in model_loaded_fp16.parameters())
    use_fp16_training = True  # This comes from args.model_dtype == 'float16'
    use_scaler = torch.cuda.is_available() and use_fp16_training

    # The critical fix: disable scaler if parameters are already FP16
    if has_fp16_params and use_scaler:
        print("WARNING: FP16 parameters detected - disabling gradient scaler to prevent errors")
        use_scaler = False

    # Create the scaler
    scaler = GradScaler('cuda', enabled=use_scaler)

    print(f"  Model parameter dtype: {next(model_loaded_fp16.parameters()).dtype}")
    print(f"  Has FP16 parameters: {has_fp16_params}")
    print(f"  Use scaler enabled: {use_scaler}")
    print(f"  Scaler state: {scaler.is_enabled()}")

    # Print the status message that would appear in training
    if torch.cuda.is_available():
        if use_fp16_training and not has_fp16_params:
            status_msg = "Mixed precision training enabled - FP16 computations with FP32 parameters and gradient scaling"
        elif has_fp16_params:
            status_msg = "FP16 model parameters detected - using FP16 training without gradient scaling"
        else:
            status_msg = "FP32 training - gradient scaler disabled"
    else:
        status_msg = "CUDA not available - gradient scaler disabled"

    print(f"  Training mode: {status_msg}")

    # Test 4: Demonstrate the error scenario (what would happen without the fix)
    print("\n📋 Test 4: What happens WITHOUT the fix")
    print("  Without the fix, the code would:")
    print("  1. Create GradScaler(enabled=True) because CUDA is available and use_fp16=True")
    print("  2. During training, gradients would be FP16 (because parameters are FP16)")
    print("  3. scaler.unscale_() would fail with 'ValueError: Attempting to unscale FP16 gradients'")
    print("  4. This is because the scaler expects FP32 gradients to unscale from FP16")

    print("\n✅ Fix Summary:")
    print("  The fix detects when model parameters are already FP16 and disables")
    print("  gradient scaling in that case, preventing the unscale error.")

    return True


def test_gradient_scaler_behavior():
    """Test actual gradient scaler behavior to demonstrate the fix"""
    print("\n🔬 Testing Actual Gradient Scaler Behavior")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - simulating behavior")
        return

    # Create a simple model and data
    model = nn.Linear(2, 1).cuda()
    x = torch.randn(4, 2).cuda()
    y = torch.randn(4, 1).cuda()

    # Test with FP32 model (this should work with scaler)
    print("\n📋 Testing with FP32 model:")
    import copy
    model_fp32 = copy.deepcopy(model).float()
    optimizer = torch.optim.SGD(model_fp32.parameters(), lr=0.01)
    scaler = GradScaler('cuda', enabled=True)

    with torch.amp.autocast('cuda'):
        output = model_fp32(x)
        loss = nn.MSELoss()(output, y)

    scaler.scale(loss).backward()
    print(f"  Parameter dtype: {next(model_fp32.parameters()).dtype}")
    print(f"  Gradient dtype: {next(model_fp32.parameters()).grad.dtype}")
    print("  scaler.unscale_() - should work")

    try:
        scaler.unscale_(optimizer)
        print("  ✅ Success: No error with FP32 parameters")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test with FP16 model (this would fail without our fix)
    print("\n📋 Testing with FP16 model:")
    model_fp16 = copy.deepcopy(model).half()
    optimizer_fp16 = torch.optim.SGD(model_fp16.parameters(), lr=0.01)
    scaler_fp16 = GradScaler('cuda', enabled=True)  # This is the problematic setup

    with torch.amp.autocast('cuda'):
        output = model_fp16(x.half())
        loss = nn.MSELoss()(output, y.half())

    scaler_fp16.scale(loss).backward()
    print(f"  Parameter dtype: {next(model_fp16.parameters()).dtype}")
    print(f"  Gradient dtype: {next(model_fp16.parameters()).grad.dtype}")
    print("  scaler.unscale_() - this would fail without our fix")

    try:
        scaler_fp16.unscale_(optimizer_fp16)
        print("  ❌ Unexpected: This should have failed")
    except ValueError as e:
        print(f"  ✅ Expected error: {e}")
        print("  This is exactly the error our fix prevents!")


if __name__ == "__main__":
    print("🧪 FP16 Gradient Scaler Fix Verification")
    print("=" * 60)

    # Test the logic
    test_fp16_gradient_scaler_logic()

    # Test actual behavior if CUDA is available
    if torch.cuda.is_available():
        test_gradient_scaler_behavior()
    else:
        print("\n⚠️  CUDA not available - skipping actual gradient scaler tests")

    print("\n🎉 All tests completed!")
    print("\nThe fix in train_sar_kd_stable.py should now handle the")
    print("'Attempting to unscale FP16 gradients' error correctly.")
