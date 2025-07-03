#!/usr/bin/env python3
"""
Debug script for Hindi TTS training issues
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from datasets import load_dataset
from chatterbox.tts import ChatterboxTTS, punc_norm
import json

def debug_tokenizer_extension(model_path: str):
    """Debug the tokenizer extension for Hindi support"""
    print("=" * 60)
    print("1. DEBUGGING TOKENIZER EXTENSION")
    print("=" * 60)
    
    # Load your extended model
    try:
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
        tokenizer = chatterbox_model.tokenizer
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Get vocab info
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"✓ Total vocabulary size: {vocab_size}")
    
    # Test English vs Hindi tokenization
    test_texts = [
        "Hello world, this is a test.",  # English
        "प्रसिद्द कबीर अध्येता, पुरुषोत्तम अग्रवाल का यह शोध आलेख",  # Hindi
        "यहाँ प्रस्तुत है, हिन्दी कवि कथाकार",  # Hindi
    ]
    
    print("\n--- Token Analysis ---")
    for i, text in enumerate(test_texts):
        lang = "English" if i == 0 else "Hindi"
        print(f"\n{lang} Text: '{text}'")
        
        # Normalize text
        normalized = punc_norm(text)
        print(f"Normalized: '{normalized}'")
        
        # Get tokens
        try:
            tokens = tokenizer.text_to_tokens(normalized).squeeze(0)
            print(f"Token count: {len(tokens)}")
            print(f"Token IDs: {tokens.tolist()[:20]}...")  # Show first 20
            
            # Decode back
            decoded = tokenizer.decode(tokens.tolist())
            print(f"Decoded: '{decoded}'")
            
            # Check for unknown tokens
            unk_token_id = vocab.get("[UNK]", -1)
            if unk_token_id != -1:
                unk_count = (tokens == unk_token_id).sum().item()
                print(f"Unknown tokens: {unk_count}")
                if unk_count > 0:
                    print(f"⚠️  High UNK count for {lang}!")
            
        except Exception as e:
            print(f"✗ Error tokenizing {lang} text: {e}")
    
    return chatterbox_model

def debug_dataset_processing(dataset_name: str, num_samples: int = 5):
    """Debug dataset processing"""
    print("\n" + "=" * 60)
    print("2. DEBUGGING DATASET PROCESSING")
    print("=" * 60)
    
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        print("\n--- Sample Analysis ---")
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            text = sample["text"]
            audio = sample["audio"]
            
            print(f"\nSample {i}:")
            print(f"Text: '{text[:100]}...'")
            print(f"Text length: {len(text)} chars")
            print(f"Audio type: {type(audio)}")
            
            if isinstance(audio, dict):
                print(f"Audio keys: {list(audio.keys())}")
                if "array" in audio:
                    print(f"Audio array shape: {np.array(audio['array']).shape}")
                    print(f"Audio sample rate: {audio.get('sampling_rate', 'N/A')}")
            
            # Check character distribution
            hindi_chars = sum(1 for c in text if ord(c) > 127)
            english_chars = sum(1 for c in text if ord(c) <= 127)
            print(f"Hindi chars: {hindi_chars}, English chars: {english_chars}")
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    return dataset

def debug_model_training_state(model_path: str, original_vocab_size: int = 704):
    """Debug model training state and freezing"""
    print("\n" + "=" * 60)
    print("3. DEBUGGING MODEL TRAINING STATE")
    print("=" * 60)
    
    try:
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
        t3_model = chatterbox_model.t3
        
        print(f"✓ Model loaded successfully")
        print(f"Text embedding shape: {t3_model.text_emb.weight.shape}")
        print(f"Text head shape: {t3_model.text_head.weight.shape}")
        
        # Check if gradients are enabled
        text_emb_requires_grad = t3_model.text_emb.weight.requires_grad
        text_head_requires_grad = t3_model.text_head.weight.requires_grad
        
        print(f"Text embedding requires_grad: {text_emb_requires_grad}")
        print(f"Text head requires_grad: {text_head_requires_grad}")
        
        # Analyze weight distributions
        text_emb_weights = t3_model.text_emb.weight.data
        
        if text_emb_weights.shape[0] > original_vocab_size:
            original_weights = text_emb_weights[:original_vocab_size]
            new_weights = text_emb_weights[original_vocab_size:]
            
            print(f"\nOriginal embeddings (first {original_vocab_size}):")
            print(f"  Mean: {original_weights.mean():.6f}")
            print(f"  Std: {original_weights.std():.6f}")
            
            print(f"\nNew embeddings (after {original_vocab_size}):")
            print(f"  Mean: {new_weights.mean():.6f}")
            print(f"  Std: {new_weights.std():.6f}")
            
            # Check if new embeddings are initialized
            if torch.allclose(new_weights, torch.zeros_like(new_weights)):
                print("⚠️  New embeddings are all zeros!")
            elif torch.allclose(new_weights, new_weights[0]):
                print("⚠️  New embeddings are all identical!")
        
        return chatterbox_model
        
    except Exception as e:
        print(f"✗ Error analyzing model state: {e}")
        return None

def debug_inference_comparison(model_path: str):
    """Debug inference comparison between English and Hindi"""
    print("\n" + "=" * 60)
    print("4. DEBUGGING INFERENCE COMPARISON")
    print("=" * 60)
    
    try:
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=model_path, device="cpu")
        
        test_cases = [
            ("English", "Hello world, this is a test."),
            ("Hindi", "यह एक परीक्षा है।"),
            ("Mixed", "Hello यह test है।"),
        ]
        
        for lang, text in test_cases:
            print(f"\n--- {lang} Test ---")
            print(f"Input text: '{text}'")
            
            try:
                # Tokenize
                normalized = punc_norm(text)
                tokens = chatterbox_model.tokenizer.text_to_tokens(normalized).squeeze(0)
                print(f"Tokenized: {len(tokens)} tokens")
                
                # Simple decoding test
                decoded = chatterbox_model.tokenizer.decode(tokens.tolist())
                print(f"Decoded: '{decoded}'")
                
                # Check for issues
                if "[UNK]" in decoded:
                    print("⚠️  Unknown tokens found!")
                
                if normalized != decoded.strip():
                    print("⚠️  Tokenization not reversible!")
                    print(f"  Original: '{normalized}'")
                    print(f"  Decoded:  '{decoded.strip()}'")
                
            except Exception as e:
                print(f"✗ Error processing {lang}: {e}")
    
    except Exception as e:
        print(f"✗ Error in inference comparison: {e}")

def generate_fixed_training_command(original_vocab_size: int = 704):
    """Generate corrected training command"""
    print("\n" + "=" * 60)
    print("5. SUGGESTED FIXES")
    print("=" * 60)
    
    print("Based on the analysis, here are the suggested fixes:")
    print("\n1. **Add freeze_text_embeddings parameter** (CRITICAL):")
    print("   This freezes the original English tokens during training.")
    
    print("\n2. **Use corrected training command:**")
    print("""
python finetune_t3.py \\
    --output_dir ./checkpoints/chatterbox_finetuned_indictts \\
    --model_name_or_path hrusheekeshsawarkar/base-hi-tts \\
    --dataset_name SPRINGLab/IndicTTS-Hindi \\
    --train_split_name train \\
    --eval_split_size 0.01 \\
    --num_train_epochs 10 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 1e-5 \\
    --warmup_steps 100 \\
    --logging_steps 10 \\
    --eval_strategy steps \\
    --eval_steps 500 \\
    --save_strategy steps \\
    --save_steps 1000 \\
    --save_total_limit 4 \\
    --fp16 True \\
    --report_to tensorboard \\
    --dataloader_num_workers 0 \\
    --do_train --do_eval \\
    --dataloader_pin_memory False \\
    --label_names labels_speech \\
    --text_column_name text \\
    --freeze_text_embeddings {original_vocab_size} \\
    --max_text_len 128 \\
    --max_speech_len 400
""".format(original_vocab_size=original_vocab_size))
    
    print("\n3. **Key changes made:**")
    print(f"   - Added --freeze_text_embeddings {original_vocab_size}")
    print("   - Reduced learning rate to 1e-5 (from 5e-6)")
    print("   - Reduced epochs to 10 (from 20)")
    print("   - Reduced max_text_len to 128 (from 256)")
    print("   - Reduced max_speech_len to 400 (from 800)")
    print("   - More frequent evaluation (steps 500 vs 1000)")
    
    print("\n4. **Additional recommendations:**")
    print("   - Monitor training loss closely")
    print("   - Check if Hindi text is being tokenized properly")
    print("   - Verify that new embeddings are being initialized")
    print("   - Consider using gradient accumulation for stability")

def main():
    """Main debugging function"""
    print("Hindi TTS Training Debug Script")
    print("=" * 60)
    
    # You need to provide these paths
    model_path = "./checkpoints/chatterbox_finetuned_indictts"  # Update this path
    dataset_name = "SPRINGLab/IndicTTS-Hindi"
    original_vocab_size = 704  # Update this if different
    
    print(f"Model path: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Original vocab size: {original_vocab_size}")
    
    # Run debugging steps
    model = debug_tokenizer_extension(model_path)
    if model:
        dataset = debug_dataset_processing(dataset_name)
        debug_model_training_state(model_path, original_vocab_size)
        debug_inference_comparison(model_path)
    
    generate_fixed_training_command(original_vocab_size)

if __name__ == "__main__":
    main() 