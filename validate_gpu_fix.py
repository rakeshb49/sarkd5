#!/usr/bin/env python3
"""
Quick GPU validation script for the tensor view/reshape fix
This script tests the specific issue that was causing evaluation failures.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

def test_gpu_tensor_operations():
    """Test tensor operations on GPU with mixed precision"""
    print("🔧 GPU Tensor Operations Validation")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU tests")
        return False

    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Load small model
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    # Test with realistic batch sizes and mixed precision
    test_texts = [
        "The machine learning model processes text efficiently.",
        "Knowledge distillation reduces model size while maintaining performance."
    ]

    print("\n🧪 Testing problematic operations...")

    with torch.no_grad():
        # Create batch similar to DataCollator output
        inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        labels = input_ids.clone()

        print(f"Batch shape: {input_ids.shape}")

        # Mixed precision forward pass (like in training)
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]  # Remove last token
            targets = labels[:, 1:]  # Remove first token

            print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
            print(f"Logits contiguous: {logits.is_contiguous()}")
            print(f"Targets contiguous: {targets.is_contiguous()}")

            # Test OLD approach (what was failing)
            print("\n❌ Testing OLD approach (.view())...")
            try:
                logits_flat_old = logits.view(-1, logits.size(-1))
                targets_flat_old = targets.view(-1)

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses_old = loss_fct(logits_flat_old, targets_flat_old)
                losses_reshaped_old = losses_old.view(targets.shape)  # This often fails!

                print(f"  ✅ OLD approach worked (surprisingly!)")

            except RuntimeError as e:
                if "view size is not compatible" in str(e):
                    print(f"  🎯 CONFIRMED: OLD approach failed with expected error")
                    print(f"     Error: {e}")
                else:
                    print(f"  ❓ OLD approach failed with different error: {e}")

            # Test NEW approach (the fix)
            print("\n✅ Testing NEW approach (.reshape())...")
            try:
                logits_flat_new = logits.reshape(-1, logits.size(-1))
                targets_flat_new = targets.reshape(-1)

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses_new = loss_fct(logits_flat_new, targets_flat_new)
                losses_reshaped_new = losses_new.reshape(targets.shape)  # This should work!

                # Calculate final loss like in evaluation
                valid_mask = (targets != -100)
                if valid_mask.any():
                    masked_losses = losses_reshaped_new * valid_mask.float()
                    final_loss = masked_losses.sum() / valid_mask.sum()

                    print(f"  ✅ NEW approach succeeded!")
                    print(f"     Final loss: {final_loss.item():.4f}")
                    return True

            except Exception as e:
                print(f"  ❌ NEW approach failed: {e}")
                return False

    return False

def simulate_failing_scenario():
    """Simulate the exact scenario that was failing in logs"""
    print("\n\n🎯 Simulating Original Failing Scenario")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    device = torch.device('cuda')

    # Create tensors that are more likely to be non-contiguous
    # This happens after certain GPU operations, slicing, etc.
    batch_size = 2
    seq_len = 127  # After removing last token
    vocab_size = 50257

    # Create some tensors that might not be contiguous
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float16)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Make tensors potentially non-contiguous through operations
    logits = logits.transpose(0, 1).transpose(0, 1)  # This can make non-contiguous
    targets = targets.transpose(0, 1).transpose(0, 1)

    print(f"Logits contiguous: {logits.is_contiguous()}")
    print(f"Targets contiguous: {targets.is_contiguous()}")

    # Test view vs reshape
    print("\n❌ Testing .view() on potentially non-contiguous tensors...")
    try:
        logits_flat_view = logits.view(-1, logits.size(-1))
        targets_flat_view = targets.view(-1)
        print("  ✅ .view() worked")
    except RuntimeError as e:
        if "view size is not compatible" in str(e):
            print("  🎯 CONFIRMED: .view() failed with memory layout issue")
        else:
            print(f"  ❓ .view() failed with: {e}")

    print("\n✅ Testing .reshape() on same tensors...")
    try:
        logits_flat_reshape = logits.reshape(-1, logits.size(-1))
        targets_flat_reshape = targets.reshape(-1)
        print("  ✅ .reshape() succeeded!")

        # Full loss calculation
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits_flat_reshape, targets_flat_reshape)
        losses_back = losses.reshape(targets.shape)
        print("  ✅ Full loss calculation succeeded!")

    except Exception as e:
        print(f"  ❌ .reshape() failed: {e}")

if __name__ == "__main__":
    print("🚀 GPU Tensor View/Reshape Fix Validation")
    print("=" * 60)

    success = test_gpu_tensor_operations()
    simulate_failing_scenario()

    print("\n" + "=" * 60)
    if success:
        print("🎉 VALIDATION SUCCESSFUL! The fix should work.")
        print("✅ Replace .view() with .reshape() in evaluation code.")
    else:
        print("❌ Validation had issues. Check CUDA availability and drivers.")

    print("\n💡 Key insight: .reshape() handles non-contiguous tensors automatically")
    print("   while .view() requires contiguous memory layout.")
