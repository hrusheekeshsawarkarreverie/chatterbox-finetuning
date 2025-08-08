#!/bin/bash

# ENHANCED Cartesia Fine-tuning Script for ChatterboxTTS
# Features:
# - Periodic audio generation during training for monitoring progress
# - Model saving after each epoch for resumable training
# - Improved training parameters to prevent NaN loss

cd "$(dirname "$0")"

echo "🚀 Starting ENHANCED Cartesia Fine-tuning with audio monitoring"
echo ""
echo "🎵 Features enabled:"
echo "  - Audio generation every 250 steps + at epoch end"
echo "  - Model saving after each epoch (resumable training)"
echo "  - Test sentences in 4 languages: English, Hindi Devanagari, Hindi Roman, Hinglish"
echo "  - Much lower learning rate (5e-5 instead of 1e-3)"
echo "  - Gradient clipping and weight decay for stability"
echo ""

# Get current timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/cartesia_enhanced_${TIMESTAMP}"

echo "📁 Output directory: $OUTPUT_DIR"
echo "🎵 Audio samples will be saved to: $OUTPUT_DIR/training_audio_samples/"
echo ""

python src/finetune_t3_cartesia.py \
    --model_name_or_path hrusheekeshsawarkar/base-hi-tts-1100voc \
    --csv_file_path "/mnt/other/cartesia_data/cartesia_audio/audio_text_mapping_with_iisc_data_20250725_154437.csv" \
    --audio_dir_path "/mnt/other/cartesia_data/cartesia_audio/tts_audio_limale_81410" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --eval_split_size 0.001 \
    --num_train_epochs 55 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --max_text_len 4096 \
    --max_speech_len 8192 \
    --dataloader_num_workers 0 \
    --fp16 \
    --seed 42 \
    --report_to tensorboard \
    --logging_dir "$OUTPUT_DIR/logs"

echo ""
echo "✅ ENHANCED Fine-tuning completed!"
echo ""