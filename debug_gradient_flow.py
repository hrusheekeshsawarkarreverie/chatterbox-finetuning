#!/usr/bin/env python3
"""
Debug script to check gradient flow during training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from transformers import HfArgumentParser, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import logging

# Import your training components
from src.finetune_t3 import (
    ModelArguments, DataArguments, CustomTrainingArguments,
    SpeechFineTuningDataset, SpeechDataCollator, T3ForFineTuning
)
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.t3.t3 import T3Cond

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradientMonitor:
    """Monitor gradients during training"""
    
    def __init__(self, model, freeze_vocab_size=704):
        self.model = model
        self.freeze_vocab_size = freeze_vocab_size
        self.gradient_stats = []
        
    def register_hooks(self):
        """Register hooks to monitor gradients"""
        
        def text_emb_hook(module, grad_input, grad_output):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad
                english_grad = grad[:self.freeze_vocab_size]
                hindi_grad = grad[self.freeze_vocab_size:]
                
                print(f"\nTEXT EMBEDDING GRADIENTS:")
                print(f"  English tokens (0-{self.freeze_vocab_size-1}):")
                print(f"    Mean: {english_grad.mean():.8f}")
                print(f"    Std: {english_grad.std():.8f}")
                print(f"    Max: {english_grad.abs().max():.8f}")
                print(f"  Hindi tokens ({self.freeze_vocab_size}+):")
                print(f"    Mean: {hindi_grad.mean():.8f}")
                print(f"    Std: {hindi_grad.std():.8f}")
                print(f"    Max: {hindi_grad.abs().max():.8f}")
                
                # Check if Hindi gradients are actually flowing
                if hindi_grad.abs().max() < 1e-8:
                    print("🚨 CRITICAL: Hindi embedding gradients are essentially zero!")
                elif hindi_grad.abs().max() < 1e-6:
                    print("⚠️  WARNING: Hindi embedding gradients are very small!")
                else:
                    print("✓ Hindi embedding gradients are flowing")
                    
        def text_head_hook(module, grad_input, grad_output):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad  # Shape: [vocab_size, hidden_size]
                english_grad = grad[:self.freeze_vocab_size]
                hindi_grad = grad[self.freeze_vocab_size:]
                
                print(f"\nTEXT HEAD GRADIENTS:")
                print(f"  English tokens (0-{self.freeze_vocab_size-1}):")
                print(f"    Mean: {english_grad.mean():.8f}")
                print(f"    Std: {english_grad.std():.8f}")
                print(f"    Max: {english_grad.abs().max():.8f}")
                print(f"  Hindi tokens ({self.freeze_vocab_size}+):")
                print(f"    Mean: {hindi_grad.mean():.8f}")
                print(f"    Std: {hindi_grad.std():.8f}")
                print(f"    Max: {hindi_grad.abs().max():.8f}")
                
                # Check if Hindi head gradients are actually flowing
                if hindi_grad.abs().max() < 1e-8:
                    print("🚨 CRITICAL: Hindi head gradients are essentially zero!")
                elif hindi_grad.abs().max() < 1e-6:
                    print("⚠️  WARNING: Hindi head gradients are very small!")
                else:
                    print("✓ Hindi head gradients are flowing")
        
        # Register hooks
        self.model.t3.text_emb.register_backward_hook(text_emb_hook)
        self.model.t3.text_head.register_backward_hook(text_head_hook)
        
    def check_gradient_masking(self):
        """Check if gradient masking is working correctly"""
        print("\n" + "=" * 60)
        print("CHECKING GRADIENT MASKING")
        print("=" * 60)
        
        # Create a simple forward pass
        batch_size = 2
        seq_len = 10
        hidden_size = self.model.t3.dim
        device = self.model.t3.device
        
        # Create dummy inputs on the correct device
        dummy_input = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True, device=device)
        
        # Forward pass through text_head
        logits = self.model.t3.text_head(dummy_input)
        
        # Create dummy targets (mix of English and Hindi tokens) on the correct device
        targets = torch.randint(0, 2000, (batch_size, seq_len), device=device)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        text_head_grad = self.model.t3.text_head.weight.grad
        
        if text_head_grad is not None:
            english_grad = text_head_grad[:self.freeze_vocab_size]
            hindi_grad = text_head_grad[self.freeze_vocab_size:]
            
            print(f"Text head gradient shape: {text_head_grad.shape}")
            print(f"English gradient (first {self.freeze_vocab_size} tokens):")
            print(f"  Max: {english_grad.abs().max():.8f}")
            print(f"  Mean: {english_grad.mean():.8f}")
            print(f"Hindi gradient (tokens {self.freeze_vocab_size}+):")
            print(f"  Max: {hindi_grad.abs().max():.8f}")
            print(f"  Mean: {hindi_grad.mean():.8f}")
            
            # Check if masking is working
            if english_grad.abs().max() < 1e-8:
                print("✓ English gradients are properly masked (frozen)")
            else:
                print("🚨 English gradients are NOT masked!")
                
            if hindi_grad.abs().max() > 1e-6:
                print("✓ Hindi gradients are flowing (not masked)")
            else:
                print("🚨 Hindi gradients are also being masked!")
        
        # Clear gradients
        self.model.zero_grad()

def debug_single_batch():
    """Debug a single training batch"""
    print("\n" + "=" * 60)
    print("DEBUGGING SINGLE BATCH")
    print("=" * 60)
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args = ModelArguments(
        model_name_or_path="hrusheekeshsawarkar/base-hi-tts",
        freeze_text_embeddings=704
    )
    data_args = DataArguments(
        dataset_name="SPRINGLab/IndicTTS-Hindi",
        max_text_len=96,
        max_speech_len=300,
        preprocessing_num_workers=1
    )
    training_args = CustomTrainingArguments(
        output_dir="./temp_debug",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=1000,
        do_train=True,
        do_eval=False,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Load model
    print("Loading model...")
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    
    # Download model files first
    download_dir = Path("./temp_model_debug")
    download_dir.mkdir(exist_ok=True)
    files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
    
    for f in files_to_download:
        try:
            hf_hub_download(
                repo_id=model_args.model_name_or_path,
                filename=f,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
                cache_dir=model_args.cache_dir
            )
        except Exception as e:
            print(f"Warning: Could not download {f}: {e}")
    
    # Try to download conds.pt as well
    try:
        hf_hub_download(
            repo_id=model_args.model_name_or_path,
            filename="conds.pt",
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            cache_dir=model_args.cache_dir
        )
    except:
        print("Note: conds.pt not found (optional)")
    
    # Load model from local directory
    chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cuda")
    
    t3_model = chatterbox_model.t3
    chatterbox_t3_config = t3_model.hp
    
    # Apply gradient masking
    if model_args.freeze_text_embeddings:
        freeze_vocab_size = model_args.freeze_text_embeddings
        
        def mask_old_token_gradients(module, grad_input, grad_output):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                module.weight.grad[:freeze_vocab_size] = 0
        
        t3_model.text_emb.register_backward_hook(mask_old_token_gradients)
        t3_model.text_head.register_backward_hook(mask_old_token_gradients)
        print(f"✓ Gradient masking applied for first {freeze_vocab_size} tokens")
    
    # Create model wrapper
    hf_model = T3ForFineTuning(t3_model, chatterbox_t3_config)
    
    # Setup gradient monitoring
    monitor = GradientMonitor(hf_model, freeze_vocab_size=704)
    monitor.register_hooks()
    
    # Check gradient masking
    monitor.check_gradient_masking()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(data_args.dataset_name, split="train")
    train_dataset = SpeechFineTuningDataset(
        data_args, chatterbox_model, chatterbox_t3_config, 
        dataset.select(range(10)), True  # Small subset for debugging
    )
    
    # Create data collator
    data_collator = SpeechDataCollator(
        chatterbox_t3_config,
        chatterbox_t3_config.stop_text_token,
        chatterbox_t3_config.stop_speech_token
    )
    
    # Create a small batch
    print("Creating batch...")
    batch_items = [train_dataset[i] for i in range(2)]
    batch_items = [item for item in batch_items if item is not None]
    
    if not batch_items:
        print("❌ No valid items in batch!")
        return
    
    batch = data_collator(batch_items)
    
    # Move to GPU
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    
    # Forward pass
    print("Running forward pass...")
    hf_model.train()
    outputs = hf_model(**batch)
    
    if isinstance(outputs, dict):
        loss = outputs['loss']
        print(f"✓ Loss: {loss.item():.6f}")
        
        # Backward pass
        print("Running backward pass...")
        loss.backward()
        
        print("✓ Gradients computed successfully")
        
        # Check parameter updates
        print("\nChecking parameter statistics:")
        text_emb_weight = hf_model.t3.text_emb.weight
        text_head_weight = hf_model.t3.text_head.weight
        
        print(f"Text embedding weight range: {text_emb_weight.min():.6f} to {text_emb_weight.max():.6f}")
        print(f"Text head weight range: {text_head_weight.min():.6f} to {text_head_weight.max():.6f}")
        
        # Check if gradients exist
        if text_emb_weight.grad is not None:
            print(f"Text embedding grad range: {text_emb_weight.grad.min():.8f} to {text_emb_weight.grad.max():.8f}")
        else:
            print("❌ No gradients for text embeddings!")
            
        if text_head_weight.grad is not None:
            print(f"Text head grad range: {text_head_weight.grad.min():.8f} to {text_head_weight.grad.max():.8f}")
        else:
            print("❌ No gradients for text head!")
            
    else:
        print(f"❌ Unexpected output type: {type(outputs)}")

def analyze_token_usage():
    """Analyze what tokens are actually being used"""
    print("\n" + "=" * 60)
    print("ANALYZING TOKEN USAGE")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset("SPRINGLab/IndicTTS-Hindi", split="train")
    
    # Load model for tokenizer
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    
    # Download tokenizer
    download_dir = Path("./temp_tokenizer_debug")
    download_dir.mkdir(exist_ok=True)
    
    try:
        hf_hub_download(
            repo_id="hrusheekeshsawarkar/base-hi-tts",
            filename="tokenizer.json",
            local_dir=download_dir,
            local_dir_use_symlinks=False
        )
        
        # Download minimal model files for tokenizer
        for f in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors"]:
            try:
                hf_hub_download(
                    repo_id="hrusheekeshsawarkar/base-hi-tts",
                    filename=f,
                    local_dir=download_dir,
                    local_dir_use_symlinks=False
                )
            except:
                pass
        
        model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cpu")
        tokenizer = model.tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return {}
    
    # Analyze first 100 samples
    token_counts = {}
    for i in range(min(100, len(dataset))):
        text = dataset[i]["text"]
        from chatterbox.tts import punc_norm
        normalized = punc_norm(text)
        tokens = tokenizer.text_to_tokens(normalized).squeeze(0)
        
        for token in tokens.tolist():
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Analyze token distribution
    english_tokens = {k: v for k, v in token_counts.items() if k < 704}
    hindi_tokens = {k: v for k, v in token_counts.items() if k >= 704}
    
    print(f"Total unique tokens: {len(token_counts)}")
    print(f"English tokens: {len(english_tokens)}")
    print(f"Hindi tokens: {len(hindi_tokens)}")
    
    if hindi_tokens:
        print(f"Most common Hindi tokens:")
        sorted_hindi = sorted(hindi_tokens.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_hindi[:10]:
            print(f"  Token {token}: {count} times")
    else:
        print("❌ No Hindi tokens found in dataset!")
    
    return token_counts

def main():
    print("Gradient Flow Debug Script")
    print("=" * 60)
    
    # Check token usage first
    token_counts = analyze_token_usage()
    
    # Debug single batch
    try:
        debug_single_batch()
    except Exception as e:
        print(f"❌ Error during batch debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 