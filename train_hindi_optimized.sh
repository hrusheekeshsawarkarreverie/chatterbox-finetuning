#!/bin/bash
set -e

echo "🚀 Optimized Hindi TTS Training Script"
echo "======================================"
echo "Key optimizations:"
echo "- Higher learning rate (1e-4) for better Hindi embedding updates"
echo "- Smaller batch size (16) for stronger gradient signals"
echo "- Gradient accumulation (4) to maintain effective batch size"
echo "- More aggressive early stopping to prevent overfitting"
echo "- Better monitoring with frequent evaluation"
echo ""

# Set CUDA environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p ./checkpoints/hindi_optimized_training

# Start GPU monitoring in background
python -c "
import subprocess
import time
import os

def monitor_gpu():
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    mem_used, mem_total, gpu_util = line.split(', ')
                    mem_percent = int(mem_used) / int(mem_total) * 100
                    print(f'GPU {i}: {mem_percent:.1f}% memory, {gpu_util}% utilization')
            time.sleep(30)
        except:
            break

if __name__ == '__main__':
    monitor_gpu()
" &
MONITOR_PID=$!

# Trap to kill monitor on exit
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

echo "Starting optimized training..."
echo "Monitor PID: $MONITOR_PID"

python src/finetune_t3.py \
    --output_dir ./checkpoints/hindi_optimized_training \
    --model_name_or_path hrusheekeshsawarkar/base-hi-tts \
    --dataset_name SPRINGLab/IndicTTS-Hindi \
    --train_split_name train \
    --eval_split_size 0.005 \
    --num_train_epochs 12 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_steps 300 \
    --logging_steps 5 \
    --save_steps 500 \
    --save_total_limit 3 \
    --fp16 \
    --do_train --do_eval \
    --text_column_name text \
    --freeze_text_embeddings 704 \
    --max_text_len 96 \
    --max_speech_len 300 \
    --preprocessing_num_workers 4 \
    --early_stopping_patience 2 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --eval_strategy steps \
    --eval_steps 250 \
    --save_strategy steps \
    --load_best_model_at_end True \
    --seed 42

echo ""
echo "✅ Training completed!"
echo "Check the results with: python debug_hindi_embeddings.py"
echo "Model saved to: ./checkpoints/hindi_optimized_training"

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "🔍 Quick model check:"
python -c "
from pathlib import Path
import torch
from chatterbox.tts import ChatterboxTTS

model_path = './checkpoints/hindi_optimized_training'
if Path(model_path).exists():
    try:
        model = ChatterboxTTS.from_local(model_path, device='cpu')
        text_emb = model.t3.text_emb.weight.data
        
        english_emb = text_emb[:704]
        hindi_emb = text_emb[704:]
        
        print(f'English embeddings std: {english_emb.std():.6f}')
        print(f'Hindi embeddings std: {hindi_emb.std():.6f}')
        
        change_ratio = hindi_emb.std() / english_emb.std()
        print(f'Hindi/English ratio: {change_ratio:.3f}')
        
        if change_ratio > 0.8:
            print('✅ Hindi embeddings appear well-trained!')
        else:
            print('⚠️  Hindi embeddings may need more training')
            
    except Exception as e:
        print(f'Error loading model: {e}')
else:
    print('Model directory not found')
" 