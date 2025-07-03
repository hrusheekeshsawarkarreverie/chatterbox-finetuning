#!/usr/bin/env python3
"""
Quick tokenizer test for Hindi support
"""

import torch
from chatterbox.tts import ChatterboxTTS, punc_norm

def test_tokenizer(model_path):
    """Test Hindi tokenization"""
    print("Loading model...")
    
    # Test both paths
    try:
        if model_path.startswith("hrusheekeshsawarkar/"):
            # Load from HuggingFace
            from huggingface_hub import hf_hub_download
            from pathlib import Path
            
            download_dir = Path("./temp_model_test")
            download_dir.mkdir(exist_ok=True)
            
            files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
            for f in files:
                hf_hub_download(repo_id=model_path, filename=f, local_dir=download_dir)
            
            model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cpu")
        else:
            # Load from local path
            model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
        
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Test tokenization
    test_texts = [
        "Hello world",
        "प्रसिद्द कबीर अध्येता",
        "यह एक परीक्षा है",
        "आधुनिक पांडित्य नहीं",
    ]
    
    tokenizer = model.tokenizer
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_size = len(vocab)
    unk_id = vocab.get("[UNK]", -1)
    
    print(f"Vocab size: {vocab_size}")
    print(f"UNK token ID: {unk_id}")
    
    print("\n" + "="*50)
    print("TOKENIZATION TEST")
    print("="*50)
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        # Normalize
        normalized = punc_norm(text)
        print(f"Normalized: '{normalized}'")
        
        # Tokenize
        tokens = tokenizer.text_to_tokens(normalized).squeeze(0)
        print(f"Tokens: {tokens.tolist()}")
        
        # Count unknowns
        if unk_id != -1:
            unk_count = (tokens == unk_id).sum().item()
            print(f"Unknown tokens: {unk_count}")
            if unk_count > 0:
                print("⚠️  HIGH UNKNOWN TOKEN COUNT!")
        
        # Decode
        decoded = tokenizer.decode(tokens.tolist())
        print(f"Decoded: '{decoded}'")
        
        # Check reversibility
        if normalized.strip() != decoded.strip():
            print("⚠️  TOKENIZATION NOT REVERSIBLE!")
        else:
            print("✓ Tokenization reversible")

if __name__ == "__main__":
    # Test your model
    model_path = "hrusheekeshsawarkar/base-hi-tts"  # or local path
    test_tokenizer(model_path)
