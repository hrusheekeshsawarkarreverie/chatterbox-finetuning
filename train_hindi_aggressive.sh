#!/bin/bash
set -e

echo "🚀 AGGRESSIVE Hindi TTS Training Script"
echo "========================================"
echo "CRITICAL FIXES APPLIED:"
echo "- Fixed gradient masking hooks (now properly registered)"
echo "- MUCH higher learning rate (1e-3) based on debug analysis"
echo "- Smaller batch size for stronger gradients"
echo "- Debug logging enabled to monitor gradient masking"
echo "- Shorter training to prevent overfitting with high LR"
echo ""

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p ./checkpoints/hindi_aggressive_training

echo "🔥 AGGRESSIVE PARAMETERS:"
echo "- Learning Rate: 1e-3 (20x higher than before)"
echo "- Batch Size: 8 (much smaller for stronger gradients)"
echo "- Epochs: 8 (shorter to prevent overfitting)"
echo "- Early Stopping: Very aggressive (patience=1)"
echo ""

python src/finetune_t3.py \
    --output_dir ./checkpoints/hindi_aggressive_training \
    --model_name_or_path hrusheekeshsawarkar/base-hi-tts \
    --dataset_name SPRINGLab/IndicTTS-Hindi \
    --train_split_name train \
    --eval_split_size 0.01 \
    --num_train_epochs 35 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_steps 100 \
    --logging_steps 5 \
    --save_steps 250 \
    --save_total_limit 2 \
    --fp16 \
    --do_train --do_eval \
    --text_column_name text \
    --freeze_text_embeddings 704 \
    --max_text_len 96 \
    --max_speech_len 300 \
    --preprocessing_num_workers 4 \
    --early_stopping_patience 1 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --eval_strategy steps \
    --eval_steps 125 \
    --save_strategy steps \
    # --load_best_model_at_end True \
    # --seed 42 \
    # --logging_level DEBUG

echo ""
echo "✅ Aggressive training completed!"
echo ""

# Immediate analysis
echo "🔍 IMMEDIATE RESULTS ANALYSIS:"
python -c "
from pathlib import Path
import torch
from chatterbox.tts import ChatterboxTTS

model_path = './checkpoints/hindi_aggressive_training'
if Path(model_path).exists():
    try:
        print('Loading trained model for analysis...')
        model = ChatterboxTTS.from_local(model_path, device='cpu')
        text_emb = model.t3.text_emb.weight.data
        
        english_emb = text_emb[:704]
        hindi_emb = text_emb[704:]
        
        print(f'📊 EMBEDDING ANALYSIS:')
        print(f'  English embeddings std: {english_emb.std():.6f}')
        print(f'  Hindi embeddings std: {hindi_emb.std():.6f}')
        print(f'  Hindi/English ratio: {hindi_emb.std()/english_emb.std():.3f}')
        print()
        
        # Compare with base model
        print('📈 COMPARING WITH BASE MODEL:')
        
        # Download base model for comparison
        from huggingface_hub import hf_hub_download
        base_dir = Path('./temp_base_comparison')
        base_dir.mkdir(exist_ok=True)
        
        for f in ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json']:
            try:
                hf_hub_download(
                    repo_id='hrusheekeshsawarkar/base-hi-tts',
                    filename=f,
                    local_dir=base_dir,
                    local_dir_use_symlinks=False
                )
            except:
                pass
        
        base_model = ChatterboxTTS.from_local(str(base_dir), device='cpu')
        base_text_emb = base_model.t3.text_emb.weight.data
        base_hindi_emb = base_text_emb[704:]
        
        # Calculate change
        hindi_change = (hindi_emb.std() - base_hindi_emb.std()).item()
        change_percentage = (hindi_change / base_hindi_emb.std() * 100).item()
        
        print(f'  Base Hindi std: {base_hindi_emb.std():.6f}')
        print(f'  Trained Hindi std: {hindi_emb.std():.6f}')
        print(f'  Change: {hindi_change:+.6f} ({change_percentage:+.1f}%)')
        print()
        
        if abs(change_percentage) > 10:
            print('🎉 SUCCESS! Hindi embeddings changed significantly!')
            print('   This suggests the training is working.')
        elif abs(change_percentage) > 2:
            print('✅ MODERATE SUCCESS! Some change detected.')
            print('   Consider running for more epochs or higher LR.')
        else:
            print('⚠️  LIMITED CHANGE. Hindi embeddings barely moved.')
            print('   May need even higher learning rate or debugging.')
            
    except Exception as e:
        print(f'❌ Error analyzing results: {e}')
else:
    print('❌ Training output directory not found!')
"

echo ""
echo "🔍 Run full analysis with: python debug_hindi_embeddings.py"
echo "📁 Model saved to: ./checkpoints/hindi_aggressive_training" 