#!/usr/bin/env python3
"""
Test script to verify the tensor reshape fix for SAR Knowledge Distillation
This script isolates and tests the specific tensor operations that were failing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

def safe_tensor_reshape(tensor, target_shape):
    """Safely reshape tensor, handling non-contiguous memory layouts"""
    try:
        if tensor.is_contiguous():
            return tensor.view(target_shape)
        else:
            return tensor.reshape(target_shape)
    except Exception as e:
        print(f"Warning: Tensor reshape failed, making contiguous: {e}")
        return tensor.contiguous().view(target_shape)

def test_tensor_operations():
    """Test the problematic tensor operations with real model outputs"""
    print("🧪 Testing Tensor Reshape Operations")
    print("=" * 50)

    # Load a small model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    # Create some test data
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "Knowledge distillation helps compress large models effectively."
    ]

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Test individual operations
    print("\n1. Testing basic tensor operations...")

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: '{text[:30]}...'")

            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
            input_ids = inputs['input_ids'].to(device)
            labels = input_ids.clone()

            print(f"  Input shape: {input_ids.shape}")

            # Forward pass
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]  # Remove last token for prediction
            targets = labels[:, 1:]  # Remove first token for targets

            print(f"  Logits shape: {logits.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Logits contiguous: {logits.is_contiguous()}")
            print(f"  Targets contiguous: {targets.is_contiguous()}")

            # Test the problematic operations
            print("  Testing OLD approach (.view())...")
            try:
                logits_flat_old = logits.view(-1, logits.size(-1))
                targets_flat_old = targets.view(-1)
                print(f"    ✅ OLD view() succeeded")
                print(f"    Logits flat shape: {logits_flat_old.shape}")
                print(f"    Targets flat shape: {targets_flat_old.shape}")
            except Exception as e:
                print(f"    ❌ OLD view() failed: {e}")

            print("  Testing NEW approach (safe_tensor_reshape())...")
            try:
                logits_flat_new = safe_tensor_reshape(logits, (-1, logits.size(-1)))
                targets_flat_new = safe_tensor_reshape(targets, (-1,))
                print(f"    ✅ NEW reshape() succeeded")
                print(f"    Logits flat shape: {logits_flat_new.shape}")
                print(f"    Targets flat shape: {targets_flat_new.shape}")

                # Test loss calculation
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(logits_flat_new, targets_flat_new)
                print(f"    Loss shape: {losses.shape}")

                # Test reshaping losses back
                losses_reshaped = safe_tensor_reshape(losses, targets.shape)
                print(f"    ✅ Loss reshape succeeded: {losses_reshaped.shape}")

                # Calculate final loss
                valid_mask = (targets != -100)
                masked_losses = losses_reshaped * valid_mask.float()
                if valid_mask.any():
                    final_loss = masked_losses.sum() / valid_mask.sum()
                    print(f"    Final loss: {final_loss.item():.4f}")

            except Exception as e:
                print(f"    ❌ NEW reshape() failed: {e}")

def test_evaluation_pipeline():
    """Test the complete evaluation pipeline with the fix"""
    print("\n\n🔍 Testing Complete Evaluation Pipeline")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16)
    model.to(device)

    # Load a small dataset
    print("Loading test dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test[:100]')  # Just 100 samples

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=128)

    def group_texts(examples):
        block_size = 128
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Process dataset
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_dataset = tokenized.map(group_texts, batched=True)

    print(f"Test dataset size: {len(lm_dataset)}")

    # Create dataloader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(lm_dataset, batch_size=2, collate_fn=collator, shuffle=False)

    # Test evaluation
    model.eval()
    total_loss = 0.0
    batch_count = 0
    successful_batches = 0

    print("\nRunning evaluation batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= 5:  # Test just 5 batches
                break

            print(f"\nBatch {batch_idx}:")
            try:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                print(f"  Input shape: {input_ids.shape}")

                # Forward pass
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :]
                targets = labels[:, 1:]

                print(f"  Logits shape: {logits.shape}")
                print(f"  Targets shape: {targets.shape}")

                # Safe tensor operations
                logits_flat = safe_tensor_reshape(logits, (-1, logits.size(-1)))
                targets_flat = safe_tensor_reshape(targets, (-1,))

                # Calculate loss
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(logits_flat, targets_flat)

                # Reshape losses
                losses_reshaped = safe_tensor_reshape(losses, targets.shape)

                # Mask and reduce
                valid_mask = (targets != -100)
                if valid_mask.any():
                    masked_losses = losses_reshaped * valid_mask.float()
                    batch_loss = masked_losses.sum() / valid_mask.sum()

                    total_loss += batch_loss.item()
                    successful_batches += 1
                    print(f"  ✅ Batch loss: {batch_loss.item():.4f}")
                else:
                    print(f"  ⚠️ No valid tokens in batch")

                batch_count += 1

            except Exception as e:
                print(f"  ❌ Batch failed: {e}")
                continue

    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
        perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 10000)
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Successful batches: {successful_batches}/{batch_count}")
    else:
        print(f"\n❌ No successful batches!")

def test_mixed_precision():
    """Test mixed precision training scenario"""
    print("\n\n⚡ Testing Mixed Precision Scenario")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in FP16 but convert params to FP32 (mixed precision style)
    model = AutoModelForCausalLM.from_pretrained('distilgpt2', torch_dtype=torch.float16)
    model.to(device)

    # Convert parameters to FP32 (like in mixed precision training)
    print("Converting model parameters to FP32...")
    for param in model.parameters():
        param.data = param.data.float()

    print("Model parameter dtype after conversion:", next(model.parameters()).dtype)

    # Test forward pass with autocast
    test_text = "This is a test sentence for mixed precision evaluation."
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    labels = input_ids.clone()

    model.train()  # Set to training mode

    print("\nTesting mixed precision forward pass...")
    try:
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = labels[:, 1:]

            print(f"Logits dtype in autocast: {logits.dtype}")
            print(f"Logits shape: {logits.shape}")

            # Test safe reshape
            logits_flat = safe_tensor_reshape(logits, (-1, logits.size(-1)))
            targets_flat = safe_tensor_reshape(targets, (-1,))

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_flat, targets_flat)

            print(f"✅ Mixed precision forward pass successful!")
            print(f"Loss: {loss.item():.4f}")
            print(f"Loss dtype: {loss.dtype}")

    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")

if __name__ == "__main__":
    print("🚀 SAR Knowledge Distillation Tensor Reshape Fix Test")
    print("=" * 60)

    test_tensor_operations()
    test_evaluation_pipeline()
    test_mixed_precision()

    print("\n" + "=" * 60)
    print("🏁 Test completed! Check results above.")
