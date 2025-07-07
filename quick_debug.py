#!/usr/bin/env python3
"""
Quick debug script to identify embedding training issues
"""

import torch
import torch.nn as nn
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from datasets import load_dataset
from chatterbox.tts import punc_norm

def check_embedding_initialization():
    """Check if Hindi embeddings are properly initialized"""
    print("=" * 60)
    print("CHECKING EMBEDDING INITIALIZATION")
    print("=" * 60)
    
    # Load base model
    model = ChatterboxTTS.from_pretrained("hrusheekeshsawarkar/base-hi-tts", device="cpu")
    
    text_emb = model.t3.text_emb.weight.data
    text_head = model.t3.text_head.weight.data
    
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Text head shape: {text_head.shape}")
    
    # Check English vs Hindi embeddings
    english_emb = text_emb[:704]
    hindi_emb = text_emb[704:]
    
    print(f"\nEnglish embeddings (first 704):")
    print(f"  Mean: {english_emb.mean():.8f}")
    print(f"  Std: {english_emb.std():.8f}")
    print(f"  Min: {english_emb.min():.8f}")
    print(f"  Max: {english_emb.max():.8f}")
    
    print(f"\nHindi embeddings (704+):")
    print(f"  Mean: {hindi_emb.mean():.8f}")
    print(f"  Std: {hindi_emb.std():.8f}")
    print(f"  Min: {hindi_emb.min():.8f}")
    print(f"  Max: {hindi_emb.max():.8f}")
    
    # Check if Hindi embeddings are problematic
    if torch.allclose(hindi_emb, torch.zeros_like(hindi_emb), atol=1e-6):
        print("🚨 CRITICAL: Hindi embeddings are all zeros!")
        return False
    elif torch.allclose(hindi_emb, hindi_emb[0].unsqueeze(0), atol=1e-6):
        print("🚨 CRITICAL: All Hindi embeddings are identical!")
        return False
    elif hindi_emb.std() < 0.001:
        print("⚠️  WARNING: Hindi embeddings have very small variance!")
        return False
    else:
        print("✓ Hindi embeddings appear properly initialized")
        return True

def check_gradient_masking_bug():
    """Check if gradient masking is accidentally masking Hindi embeddings"""
    print("\n" + "=" * 60)
    print("CHECKING GRADIENT MASKING BUG")
    print("=" * 60)
    
    # Create a simple test
    vocab_size = 2000
    hidden_size = 1024
    freeze_vocab_size = 704
    
    # Create dummy embedding and head layers
    text_emb = nn.Embedding(vocab_size, hidden_size)
    text_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    # Print shapes
    print(f"Text embedding weight shape: {text_emb.weight.shape}")
    print(f"Text head weight shape: {text_head.weight.shape}")
    
    # Create dummy input
    dummy_input = torch.randn(2, 10, hidden_size)
    dummy_tokens = torch.randint(700, 1000, (2, 10))  # Hindi tokens
    
    # Forward pass
    logits = text_head(dummy_input)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), dummy_tokens.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check gradients before masking
    print(f"\nBefore masking:")
    print(f"  Text head grad shape: {text_head.weight.grad.shape}")
    print(f"  English head grad max: {text_head.weight.grad[:freeze_vocab_size].abs().max():.8f}")
    print(f"  Hindi head grad max: {text_head.weight.grad[freeze_vocab_size:].abs().max():.8f}")
    
    # Apply gradient masking (current implementation)
    text_head.weight.grad[:freeze_vocab_size] = 0
    
    print(f"\nAfter masking:")
    print(f"  English head grad max: {text_head.weight.grad[:freeze_vocab_size].abs().max():.8f}")
    print(f"  Hindi head grad max: {text_head.weight.grad[freeze_vocab_size:].abs().max():.8f}")
    
    # Check if Hindi gradients are still there
    if text_head.weight.grad[freeze_vocab_size:].abs().max() > 1e-6:
        print("✓ Hindi gradients preserved after masking")
        return True
    else:
        print("🚨 Hindi gradients were also masked!")
        return False

def check_learning_rate_issue():
    """Check if learning rate is too low for effective training"""
    print("\n" + "=" * 60)
    print("CHECKING LEARNING RATE EFFECTIVENESS")
    print("=" * 60)
    
    # Simulate parameter updates with different learning rates
    initial_std = 0.0176  # From debug output
    learning_rates = [5e-6, 5e-5, 1e-4, 5e-4]
    
    for lr in learning_rates:
        # Simulate gradient magnitude (typical for cross-entropy loss)
        grad_magnitude = 0.01  # Typical gradient magnitude
        
        # Simulate weight update
        weight_update = lr * grad_magnitude
        
        # Calculate relative change
        relative_change = weight_update / initial_std
        
        print(f"Learning rate {lr:.0e}:")
        print(f"  Weight update: {weight_update:.8f}")
        print(f"  Relative change: {relative_change:.6f}")
        
        if relative_change < 0.001:
            print("  ⚠️  Very small relative change - may be too low")
        elif relative_change > 0.1:
            print("  ⚠️  Large relative change - may be too high")
        else:
            print("  ✓ Reasonable relative change")
        print()

def check_batch_size_impact():
    """Check if batch size is causing gradient averaging issues"""
    print("\n" + "=" * 60)
    print("CHECKING BATCH SIZE IMPACT")
    print("=" * 60)
    
    # Load dataset to check token distribution
    dataset = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train")
    model = ChatterboxTTS.from_pretrained("hrusheekeshsawarkar/base-hi-tts", device="cpu")
    tokenizer = model.tokenizer
    
    # Sample different batch sizes
    batch_sizes = [2, 8, 16, 32, 56]
    
    for batch_size in batch_sizes:
        print(f"\nBatch size {batch_size}:")
        
        # Count Hindi tokens per batch
        hindi_token_counts = []
        for i in range(5):  # Sample 5 batches
            batch_hindi_tokens = 0
            for j in range(batch_size):
                sample_idx = (i * batch_size + j) % len(dataset)
                text = dataset[sample_idx]["text"]
                normalized = punc_norm(text)
                tokens = tokenizer.text_to_tokens(normalized).squeeze(0)
                hindi_tokens = [t for t in tokens.tolist() if t >= 704]
                batch_hindi_tokens += len(hindi_tokens)
            hindi_token_counts.append(batch_hindi_tokens)
        
        avg_hindi_tokens = sum(hindi_token_counts) / len(hindi_token_counts)
        print(f"  Average Hindi tokens per batch: {avg_hindi_tokens:.1f}")
        
        # Estimate gradient signal strength
        signal_strength = avg_hindi_tokens / (batch_size * 50)  # Rough estimate
        print(f"  Estimated gradient signal strength: {signal_strength:.3f}")
        
        if signal_strength < 0.1:
            print("  ⚠️  Weak gradient signal - large batch may dilute learning")
        elif signal_strength > 0.5:
            print("  ✓ Strong gradient signal")
        else:
            print("  ✓ Moderate gradient signal")

def suggest_fixes():
    """Suggest specific fixes based on analysis"""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)
    
    print("Based on the analysis, try these fixes in order:")
    
    print("\n1. **INCREASE LEARNING RATE (Most likely fix)**")
    print("   - Current: 5e-5 → Try: 1e-4 or 2e-4")
    print("   - Hindi embeddings need stronger signal to update")
    
    print("\n2. **REDUCE BATCH SIZE**")
    print("   - Current: 56 → Try: 16 or 24")
    print("   - Smaller batches = stronger per-sample gradients")
    
    print("\n3. **INCREASE GRADIENT ACCUMULATION**")
    print("   - Current: 2 → Try: 4 or 8")
    print("   - Maintains effective batch size with stronger gradients")
    
    print("\n4. **USE DIFFERENT LEARNING RATE FOR HINDI TOKENS**")
    print("   ```python")
    print("   # In finetune_t3.py, modify optimizer setup:")
    print("   hindi_params = [p for name, p in model.named_parameters() if 'text_emb' in name or 'text_head' in name]")
    print("   other_params = [p for name, p in model.named_parameters() if 'text_emb' not in name and 'text_head' not in name]")
    print("   optimizer = torch.optim.AdamW([")
    print("       {'params': hindi_params, 'lr': 1e-4},")
    print("       {'params': other_params, 'lr': 5e-5}")
    print("   ])")
    print("   ```")
    
    print("\n5. **ADD GRADIENT CLIPPING**")
    print("   - Add: --max_grad_norm 1.0")
    print("   - Prevents gradient explosion with higher learning rates")
    
    print("\n6. **MONITOR GRADIENTS DURING TRAINING**")
    print("   - Add gradient monitoring hooks")
    print("   - Watch for Hindi embedding gradient magnitudes")
    
    print("\n7. **REDUCE TRAINING EPOCHS**")
    print("   - Current: 30 → Try: 10-15")
    print("   - Prevent overfitting with higher learning rates")

def main():
    print("Quick Hindi Embedding Debug")
    print("=" * 60)
    
    # Run all checks
    init_ok = check_embedding_initialization()
    mask_ok = check_gradient_masking_bug()
    
    check_learning_rate_issue()
    check_batch_size_impact()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if not init_ok:
        print("🚨 CRITICAL: Embedding initialization problem!")
    elif not mask_ok:
        print("🚨 CRITICAL: Gradient masking bug!")
    else:
        print("✓ Embeddings and masking look OK")
        print("⚠️  Most likely cause: Learning rate too low + batch size too large")
    
    suggest_fixes()

if __name__ == "__main__":
    main() 