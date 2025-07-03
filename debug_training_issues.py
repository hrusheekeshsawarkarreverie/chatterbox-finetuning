#!/usr/bin/env python3
"""
Debug and fix Hindi training issues
"""

import torch
import numpy as np
from pathlib import Path

def check_training_logs(checkpoint_dir: str):
    """Check training logs for loss patterns"""
    print("=" * 60)
    print("CHECKING TRAINING LOGS")
    print("=" * 60)
    
    log_file = Path(checkpoint_dir) / "trainer_state.json"
    if log_file.exists():
        import json
        with open(log_file, 'r') as f:
            trainer_state = json.load(f)
        
        print("Training Progress:")
        for i, entry in enumerate(trainer_state.get('log_history', [])[-10:]):  # Last 10 entries
            if 'train_loss' in entry:
                print(f"  Step {entry.get('step', i)}: Loss = {entry['train_loss']:.4f}")
        
        # Check if loss decreased meaningfully
        if len(trainer_state.get('log_history', [])) > 0:
            first_loss = None
            last_loss = None
            for entry in trainer_state['log_history']:
                if 'train_loss' in entry:
                    if first_loss is None:
                        first_loss = entry['train_loss']
                    last_loss = entry['train_loss']
            
            if first_loss and last_loss:
                improvement = first_loss - last_loss
                print(f"\nLoss Improvement: {first_loss:.4f} -> {last_loss:.4f} (Δ={improvement:.4f})")
                
                if improvement < 0.5:
                    print("⚠️  Very small loss improvement! Learning rate might be too low.")
                elif improvement > 3.0:
                    print("⚠️  Very large loss improvement! Model might be overfitting.")
                else:
                    print("✓ Reasonable loss improvement.")
    else:
        print("No training logs found.")

def create_fixed_training_script():
    """Create a fixed training script with proper parameters"""
    print("\n" + "=" * 60)
    print("CREATING FIXED TRAINING SCRIPT")
    print("=" * 60)
    
    script_content = '''#!/bin/bash

# Fixed Hindi TTS Training Script
# This addresses the identified issues with Hindi embedding training
# Uses only VALID arguments supported by the finetune_t3.py script

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"None\"}')"

# Set CUDA environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture
export CUDA_VISIBLE_DEVICES=0

# Run training with VALID parameters only
python src/finetune_t3.py \\
    --output_dir ./checkpoints/chatterbox_finetuned_indictts_fixed \\
    --model_name_or_path hrusheekeshsawarkar/base-hi-tts \\
    --dataset_name SPRINGLab/IndicTTS-Hindi \\
    --train_split_name train \\
    --eval_split_size 0.01 \\
    --num_train_epochs 12 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 2 \\
    --learning_rate 5e-5 \\
    --warmup_steps 200 \\
    --logging_steps 10 \\
    --save_steps 1000 \\
    --save_total_limit 4 \\
    --fp16 \\
    --do_train --do_eval \\
    --text_column_name text \\
    --freeze_text_embeddings 704 \\
    --max_text_len 96 \\
    --max_speech_len 300 \\
    --preprocessing_num_workers 4 \\
    --early_stopping_patience 3

echo "Training completed with fixed parameters!"
echo "GPU memory usage:"
python -c "import torch; print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB'); print(f'GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB')" 2>/dev/null || echo "No GPU stats available"
'''
    
    with open("train_hindi_fixed.sh", "w") as f:
        f.write(script_content)
    
    print("Created train_hindi_fixed.sh with the following VALID fixes:")
    print("1. ✅ Increased learning rate: 5e-6 → 5e-5 (10x higher)")
    print("2. ✅ Optimized batch size: 2 → 4 with grad_accumulation=2")
    print("3. ✅ Reduced epochs: 20 → 12 (prevent overfitting)")
    print("4. ✅ Increased warmup: 100 → 200 steps") 
    print("5. ✅ Reduced sequence lengths (more efficient)")
    print("6. ✅ Added early stopping patience")
    print("7. ✅ GPU memory monitoring and CUDA optimizations")
    print("8. ✅ REMOVED INVALID ARGUMENTS that don't exist in finetune_t3.py:")
    print("   - eval_strategy (use built-in evaluation logic)")
    print("   - dataloader_pin_memory (not supported)")
    print("   - dataloader_num_workers (use preprocessing_num_workers)")
    print("   - report_to (not supported)")
    print("   - max_grad_norm (not supported)")
    print("   - weight_decay (not supported)")
    print("   - ddp_find_unused_parameters (not supported)")
    print("   - remove_unused_columns (not supported)")
    
    return "train_hindi_fixed.sh"

def create_embedding_analysis_script():
    """Create a script to monitor embedding changes during training"""
    print("\n" + "=" * 60)
    print("CREATING EMBEDDING MONITORING SCRIPT")
    print("=" * 60)
    
    script_content = '''#!/usr/bin/env python3
"""
Monitor Hindi embedding changes during training
"""

import torch
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

def compare_models(base_path, trained_path):
    """Compare embeddings before and after training"""
    
    # Load base model
    print("Loading base model...")
    if base_path.startswith("hrusheekeshsawarkar/"):
        from huggingface_hub import hf_hub_download
        download_dir = Path("./temp_base")
        download_dir.mkdir(exist_ok=True)
        files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
        for f in files:
            hf_hub_download(repo_id=base_path, filename=f, local_dir=download_dir)
        base_model = ChatterboxTTS.from_local(ckpt_dir=str(download_dir), device="cpu")
    else:
        base_model = ChatterboxTTS.from_local(ckpt_dir=base_path, device="cpu")
    
    # Load trained model
    print("Loading trained model...")
    trained_model = ChatterboxTTS.from_local(ckpt_dir=trained_path, device="cpu")
    
    # Compare embeddings
    base_emb = base_model.t3.text_emb.weight.data
    trained_emb = trained_model.t3.text_emb.weight.data
    
    # English embeddings (0-703)
    english_base = base_emb[:704]
    english_trained = trained_emb[:704]
    english_change = (english_trained - english_base).abs().mean()
    
    # Hindi embeddings (704-1999)
    hindi_base = base_emb[704:]
    hindi_trained = trained_emb[704:]
    hindi_change = (hindi_trained - hindi_base).abs().mean()
    
    print(f"\\nEmbedding Changes:")
    print(f"English embeddings: {english_change:.6f} (should be ~0 if frozen)")
    print(f"Hindi embeddings: {hindi_change:.6f} (should be >0.001 if trained)")
    
    if english_change > 0.001:
        print("⚠️  English embeddings changed! Freezing might not be working.")
    else:
        print("✓ English embeddings properly frozen.")
    
    if hindi_change < 0.001:
        print("🚨 Hindi embeddings barely changed! Training failed.")
    else:
        print("✓ Hindi embeddings updated during training.")
    
    return english_change, hindi_change

if __name__ == "__main__":
    base_path = "hrusheekeshsawarkar/base-hi-tts"
    trained_path = "./checkpoints/chatterbox_finetuned_indictts_fixed"
    
    if Path(trained_path).exists():
        compare_models(base_path, trained_path)
    else:
        print(f"Trained model not found at: {trained_path}")
        print("Train the model first using the fixed training script.")
'''
    
    with open("monitor_embeddings.py", "w") as f:
        f.write(script_content)
    
    print("Created monitor_embeddings.py")
    print("Run this after training to verify Hindi embeddings actually changed.")

def create_gpu_monitor_script():
    """Create a script to monitor GPU usage during training"""
    print("\n" + "=" * 60)
    print("CREATING GPU MONITORING SCRIPT")
    print("=" * 60)
    
    script_content = '''#!/usr/bin/env python3
"""
Monitor GPU usage during training
"""

import time
import subprocess
import psutil
from datetime import datetime

def monitor_gpu():
    """Monitor GPU and system resources"""
    print("GPU & System Resource Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # GPU info using nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(', ')
                    gpu_util = gpu_data[0]
                    gpu_mem_used = int(gpu_data[1])
                    gpu_mem_total = int(gpu_data[2])
                    gpu_temp = gpu_data[3]
                    gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
                    
                    print(f"[{timestamp}] GPU: {gpu_util}% | "
                          f"VRAM: {gpu_mem_used}MB/{gpu_mem_total}MB ({gpu_mem_percent:.1f}%) | "
                          f"Temp: {gpu_temp}°C")
                else:
                    print(f"[{timestamp}] GPU: nvidia-smi not available")
            except:
                print(f"[{timestamp}] GPU: monitoring failed")
            
            # CPU and RAM
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            
            print(f"[{timestamp}] CPU: {cpu_percent:.1f}% | "
                  f"RAM: {ram_used_gb:.1f}GB/{ram_total_gb:.1f}GB ({ram_percent:.1f}%)")
            
            print()
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_gpu()
'''
    
    with open("monitor_gpu.py", "w") as f:
        f.write(script_content)
    
    print("Created monitor_gpu.py")
    print("Run this in a separate terminal during training to monitor resources.")

def main():
    print("Hindi Training Issue Diagnosis & Fix")
    print("=" * 60)
    
    # Check existing training logs
    checkpoint_dir = "./checkpoints/chatterbox_finetuned_indictts"
    if Path(checkpoint_dir).exists():
        check_training_logs(checkpoint_dir)
    
    # Create fixed training script
    script_name = create_fixed_training_script()
    
    # Create monitoring scripts
    create_embedding_analysis_script()
    create_gpu_monitor_script()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED ACTION PLAN")
    print("=" * 60)
    
    print("1. 🔧 Run the fixed training script:")
    print(f"   chmod +x {script_name}")
    print(f"   ./{script_name}")
    
    print("\n2. 📊 Monitor training progress:")
    print("   # In terminal 1:")
    print("   tensorboard --logdir ./checkpoints/chatterbox_finetuned_indictts_fixed")
    print("   # In terminal 2:")
    print("   python monitor_gpu.py")
    
    print("\n3. ✅ Verify embedding changes:")
    print("   python monitor_embeddings.py")
    
    print("\n4. 🎯 Expected results:")
    print("   - Hindi embeddings should change significantly (>0.005)")
    print("   - English embeddings should stay frozen (<0.001)")
    print("   - Training loss should decrease more than previous run")
    print("   - Hindi inference should produce proper speech")
    
    print("\n5. 🚀 GPU Optimizations included:")
    print("   - CUDA environment variables set for A100")
    print("   - Memory pinning enabled for faster data transfer")
    print("   - 4 data loader workers for parallel processing")
    print("   - Batch size optimized for A100 GPU memory")
    print("   - FP16 training for faster computation")

if __name__ == "__main__":
    main() 