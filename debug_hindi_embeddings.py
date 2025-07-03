#!/usr/bin/env python3
"""
Debug Hindi embedding initialization and training
"""

import torch
import numpy as np
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

def analyze_embeddings(model_path: str, original_vocab_size: int = 704):
    """Analyze embedding weights before and after training"""
    print("=" * 60)
    print("ANALYZING HINDI EMBEDDINGS")
    print("=" * 60)
    
    try:
        if model_path.startswith("hrusheekeshsawarkar/"):
            # Load from HuggingFace
            from huggingface_hub import hf_hub_download
            
            download_dir = Path("./temp_model_analysis")
            download_dir.mkdir(exist_ok=True)
            
            files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
            for f in files:
                hf_hub_download(repo_id=model_path, filename=f, local_dir=download_dir)
            
            model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cpu")
        else:
            model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
            
        print(f"✓ Model loaded from: {model_path}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    t3_model = model.t3
    text_emb = t3_model.text_emb.weight.data
    text_head = t3_model.text_head.weight.data
    
    vocab_size = text_emb.shape[0]
    print(f"Total vocab size: {vocab_size}")
    print(f"Original vocab size: {original_vocab_size}")
    print(f"New Hindi tokens: {vocab_size - original_vocab_size}")
    
    if vocab_size <= original_vocab_size:
        print("⚠️  No vocab extension found!")
        return
    
    # Split embeddings
    english_emb = text_emb[:original_vocab_size]
    hindi_emb = text_emb[original_vocab_size:]
    
    english_head = text_head[:, :original_vocab_size]  
    hindi_head = text_head[:, original_vocab_size:]
    
    print("\n" + "=" * 40)
    print("EMBEDDING ANALYSIS")
    print("=" * 40)
    
    print(f"\nEnglish embeddings ({original_vocab_size} tokens):")
    print(f"  Shape: {english_emb.shape}")
    print(f"  Mean: {english_emb.mean():.6f}")
    print(f"  Std: {english_emb.std():.6f}")
    print(f"  Min: {english_emb.min():.6f}")
    print(f"  Max: {english_emb.max():.6f}")
    
    print(f"\nHindi embeddings ({vocab_size - original_vocab_size} tokens):")
    print(f"  Shape: {hindi_emb.shape}")
    print(f"  Mean: {hindi_emb.mean():.6f}")
    print(f"  Std: {hindi_emb.std():.6f}")
    print(f"  Min: {hindi_emb.min():.6f}")
    print(f"  Max: {hindi_emb.max():.6f}")
    
    # Check for problematic patterns
    print("\n" + "=" * 40)
    print("DIAGNOSTIC CHECKS")
    print("=" * 40)
    
    # Check 1: Are Hindi embeddings all zeros?
    if torch.allclose(hindi_emb, torch.zeros_like(hindi_emb), atol=1e-6):
        print("🚨 CRITICAL: Hindi embeddings are all zeros!")
        print("   This will cause gibberish output.")
        print("   FIX: Properly initialize Hindi embeddings.")
    
    # Check 2: Are Hindi embeddings all identical?
    elif torch.allclose(hindi_emb, hindi_emb[0].unsqueeze(0), atol=1e-6):
        print("🚨 CRITICAL: All Hindi embeddings are identical!")
        print("   This will cause repetitive gibberish.")
        print("   FIX: Properly initialize with random values.")
    
    # Check 3: Are Hindi embeddings too small?
    elif hindi_emb.std() < 0.01:
        print("⚠️  WARNING: Hindi embeddings have very small variance!")
        print(f"   Std: {hindi_emb.std():.6f}")
        print("   This might cause poor learning.")
    
    # Check 4: Are Hindi embeddings too large?
    elif hindi_emb.std() > 1.0:
        print("⚠️  WARNING: Hindi embeddings have very large variance!")
        print(f"   Std: {hindi_emb.std():.6f}")
        print("   This might cause unstable training.")
    
    # Check 5: Compare scales
    else:
        ratio = hindi_emb.std() / english_emb.std()
        print(f"✓ Hindi/English std ratio: {ratio:.3f}")
        if ratio < 0.5:
            print("⚠️  Hindi embeddings are much smaller than English")
        elif ratio > 2.0:
            print("⚠️  Hindi embeddings are much larger than English")
        else:
            print("✓ Hindi and English embeddings are similarly scaled")
    
    print("\n" + "=" * 40)
    print("HEAD LAYER ANALYSIS")
    print("=" * 40)
    
    print(f"\nEnglish head weights:")
    print(f"  Shape: {english_head.shape}")
    print(f"  Mean: {english_head.mean():.6f}")
    print(f"  Std: {english_head.std():.6f}")
    
    print(f"\nHindi head weights:")
    print(f"  Shape: {hindi_head.shape}")
    print(f"  Mean: {hindi_head.mean():.6f}")
    print(f"  Std: {hindi_head.std():.6f}")
    
    # Check head layer issues
    if torch.allclose(hindi_head, torch.zeros_like(hindi_head), atol=1e-6):
        print("🚨 CRITICAL: Hindi head weights are all zeros!")
    elif torch.allclose(hindi_head, hindi_head[:, 0:1], atol=1e-6):
        print("🚨 CRITICAL: All Hindi head weights are identical!")
    else:
        head_ratio = hindi_head.std() / english_head.std()
        print(f"✓ Hindi/English head std ratio: {head_ratio:.3f}")
    
    return model

def check_token_distributions(model_path: str, dataset_name: str = "SPRINGLab/IndicTTS-Hindi"):
    """Check what tokens are actually being used in training"""
    print("\n" + "=" * 60)
    print("CHECKING TOKEN DISTRIBUTIONS")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        from chatterbox.tts import punc_norm
        
        # Load model
        if model_path.startswith("hrusheekeshsawarkar/"):
            from huggingface_hub import hf_hub_download
            download_dir = Path("./temp_model_analysis")
            download_dir.mkdir(exist_ok=True)
            files = ["tokenizer.json"]
            for f in files:
                hf_hub_download(repo_id=model_path, filename=f, local_dir=download_dir)
            model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cpu")
        else:
            model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        # Analyze token usage
        all_tokens = []
        tokenizer = model.tokenizer
        
        print(f"Analyzing {min(100, len(dataset))} samples...")
        
        for i in range(min(100, len(dataset))):
            text = dataset[i]["text"]
            normalized = punc_norm(text)
            tokens = tokenizer.text_to_tokens(normalized).squeeze(0)
            all_tokens.extend(tokens.tolist())
        
        unique_tokens = set(all_tokens)
        print(f"✓ Total unique tokens used: {len(unique_tokens)}")
        
        # Check token ranges
        english_tokens = [t for t in unique_tokens if t < 704]
        hindi_tokens = [t for t in unique_tokens if t >= 704]
        
        print(f"✓ English tokens used: {len(english_tokens)}")
        print(f"✓ Hindi tokens used: {len(hindi_tokens)}")
        
        if not hindi_tokens:
            print("🚨 CRITICAL: No Hindi tokens found in dataset!")
            print("   This means Hindi text isn't being tokenized to new tokens.")
        else:
            print(f"✓ Hindi token range: {min(hindi_tokens)} - {max(hindi_tokens)}")
            
        return unique_tokens
        
    except Exception as e:
        print(f"✗ Error analyzing token distributions: {e}")
        return None

def generate_fixes():
    """Generate suggested fixes"""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)
    
    print("Based on the analysis, here are potential fixes:")
    
    print("\n1. **If Hindi embeddings are zeros/identical:**")
    print("   - Reinitialize Hindi embeddings with proper random values")
    print("   - Use Xavier/Kaiming initialization")
    print("   - Match the scale of English embeddings")
    
    print("\n2. **If Hindi embeddings are poorly scaled:**")
    print("   - Adjust initialization variance")
    print("   - Use lower learning rate for Hindi tokens")
    print("   - Add embedding layer normalization")
    
    print("\n3. **If no Hindi tokens are being used:**")
    print("   - Check tokenizer extension process")
    print("   - Verify vocab mapping is correct")
    print("   - Ensure Hindi text is being properly preprocessed")
    
    print("\n4. **Training adjustments:**")
    print("   - Use different learning rates for English vs Hindi")
    print("   - Add gradient clipping")
    print("   - Monitor per-token losses")
    
    print("\n5. **Debugging training:**")
    print("   - Add logging for Hindi token gradients")
    print("   - Check if Hindi embeddings are actually updating")
    print("   - Verify loss computation for Hindi tokens")

def main():
    print("Hindi Embedding Debug Script")
    print("=" * 60)
    
    # Analyze your base model
    base_model_path = "hrusheekeshsawarkar/base-hi-tts"
    
    print("Analyzing BASE model (before training):")
    analyze_embeddings(base_model_path)
    
    # Check token usage
    check_token_distributions(base_model_path)
    
    # If you have a trained model, analyze it too
    trained_model_path = "./checkpoints/chatterbox_finetuned_indictts"
    if Path(trained_model_path).exists():
        print("\n" + "=" * 60)
        print("Analyzing TRAINED model (after training):")
        analyze_embeddings(trained_model_path)
    
    generate_fixes()

if __name__ == "__main__":
    main() 