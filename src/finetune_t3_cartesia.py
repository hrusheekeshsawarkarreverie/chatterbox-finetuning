import argparse
import logging
import os
import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

from chatterbox.tts import ChatterboxTTS, Conditionals, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE
from chatterbox.models.s3gen import S3GEN_SR

logger = logging.getLogger(__name__)

# --- Audio Monitoring Callback ---
class AudioGenerationCallback(TrainerCallback):
    """
    Custom callback to generate test audio samples during training
    for monitoring model learning progress across different languages/scripts
    """
    
    def __init__(self, chatterbox_model, output_dir, generation_interval=250):
        self.chatterbox_model = chatterbox_model
        self.output_dir = output_dir
        self.generation_interval = generation_interval
        
        # Test sentences for monitoring training progress
        self.test_sentences = {
            "english": [
                "Hello, how are you today?",
                "This is a test of the speech synthesis system.",
                "Thank you for calling customer service."
            ],
            "hindi_devanagari": [
                "नमस्ते, आप कैसे हैं?",
                "मैं आपकी सहायता कर सकता हूँ।",
                "धन्यवाद, आपका दिन शुभ हो।"
            ],
            "hindi_roman": [
                "namaste, aap kaise hain?",
                "main aapki sahayata kar sakta hun.",
                "dhanyawad, aapka din shubh ho."
            ],
            "hinglish": [
                "Hello, main aapki help kar sakta hun.",
                "aapka loan application process ho gaya hai.",
                "toh aapne jo payment kiya tha, woh successful hai."
            ]
        }
        
        # Create audio monitoring directory
        self.audio_dir = Path(output_dir) / "training_audio_samples"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate audio samples at regular intervals"""
        if state.global_step % self.generation_interval == 0 and state.global_step > 0:
            self._generate_monitoring_audio(state.global_step, state.epoch)
            
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Generate audio samples at the end of each epoch"""
        self._generate_monitoring_audio(state.global_step, state.epoch, epoch_end=True)
        
    def _generate_monitoring_audio(self, step, epoch, epoch_end=False):
        """Generate and save audio samples for all test sentences"""
        import torchaudio as ta
        
        prefix = f"epoch_{epoch:.0f}_step_{step}" if not epoch_end else f"epoch_{epoch:.0f}_final"
        
        logger.info(f"🎵 Generating monitoring audio samples at {prefix}")
        
        try:
            # Temporarily set model to eval mode for generation
            original_training_mode = self.chatterbox_model.training
            self.chatterbox_model.eval()
            
            for lang_type, sentences in self.test_sentences.items():
                lang_dir = self.audio_dir / lang_type
                lang_dir.mkdir(exist_ok=True)
                
                for i, text in enumerate(sentences):
                    try:
                        # Generate audio
                        wav = self.chatterbox_model.generate(text, temperature=0.7)
                        duration = wav.shape[1] / self.chatterbox_model.sr
                        
                        # Save audio file
                        filename = f"{prefix}_{lang_type}_sample{i+1}.wav"
                        filepath = lang_dir / filename
                        ta.save(str(filepath), wav, self.chatterbox_model.sr)
                        
                        logger.info(f"  ✅ {lang_type} sample {i+1}: {duration:.2f}s - '{text[:50]}{'...' if len(text) > 50 else ''}' → {filename}")
                        
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to generate {lang_type} sample {i+1}: {e}")
                        
            # Restore original training mode
            if original_training_mode:
                self.chatterbox_model.train()
                
            logger.info(f"🎵 Audio monitoring complete for {prefix}")
            
        except Exception as e:
            logger.error(f"❌ Audio monitoring failed at {prefix}: {e}")

# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping with specified patience. Default: None (disabled)."}
    )
    
    def __post_init__(self):
        # Import the enum for proper comparison
        from transformers.trainer_utils import IntervalStrategy
        
        # Set strategies BEFORE calling parent __post_init__
        # Force evaluation strategy if eval_steps is provided or early stopping is enabled
        if (self.eval_steps is not None and self.eval_steps > 0) or self.early_stopping_patience is not None:
            self.evaluation_strategy = IntervalStrategy.STEPS
            # Also set eval_strategy for compatibility with different transformers versions
            self.eval_strategy = IntervalStrategy.STEPS
            
        # Force save strategy if save_steps is provided  
        if self.save_steps is not None and self.save_steps > 0:
            self.save_strategy = IntervalStrategy.STEPS
                
        # Enable load_best_model_at_end if early stopping is used
        if self.early_stopping_patience is not None and self.early_stopping_patience > 0:
            self.load_best_model_at_end = True
            
        # Call parent __post_init__ AFTER setting our strategies
        super().__post_init__()

# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})
    freeze_text_embeddings: Optional[int] = field(default=None, metadata={"help": "Number of original text embedding tokens to freeze (e.g., 704 for original vocab size)."})

@dataclass
class DataArguments:
    # Cartesia dataset specific arguments
    csv_file_path: str = field(
        metadata={"help": "Path to the CSV file containing audio-text mappings (audio_text_mapping.csv)"}
    )
    audio_dir_path: str = field(
        metadata={"help": "Path to the directory containing audio files (tts_audio_81717)"}
    )
    
    # Original arguments for compatibility
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the CSV."})
    audio_column_name: str = field(default="audio_filename", metadata={"help": "The name of the audio filename column in the CSV."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0, metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    eval_split_size: float = field(
        default=0.1, metadata={"help": "Fraction of data to use for evaluation when splitting dataset."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use from the dataset. If None, use all samples."}
    )

def load_cartesia_dataset(data_args: DataArguments, training_args) -> tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """
    Load the Cartesia dataset from CSV and audio directory.
    Returns train and eval datasets as lists of dicts.
    """
    logger.info(f"Loading dataset from CSV: {data_args.csv_file_path}")
    logger.info(f"Audio directory: {data_args.audio_dir_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(data_args.csv_file_path)
        logger.info(f"Loaded CSV with {len(df)} rows")
        logger.info(f"CSV columns: {list(df.columns)}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {data_args.csv_file_path}: {e}")
    
    # Filter successful entries only
    if 'status' in df.columns:
        df = df[df['status'] == 'success']
        logger.info(f"Filtered to {len(df)} successful entries")
    
    # Filter for Voice_2 only
    if 'voice_name' in df.columns:
        df = df[df['voice_name'] == 'Voice_2']
        logger.info(f"Filtered to {len(df)} Voice_2 entries")
    else:
        logger.warning("voice_name column not found in CSV - proceeding without voice filtering")
    
    # Create dataset entries
    dataset_entries = []
    audio_dir = Path(data_args.audio_dir_path)
    
    for idx, row in df.iterrows():
        if data_args.max_samples and len(dataset_entries) >= data_args.max_samples:
            break
            
        audio_filename = row[data_args.audio_column_name]
        text = row[data_args.text_column_name]
        
        # Construct full audio path
        audio_path = audio_dir / audio_filename
        
        # Check if audio file exists
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}. Skipping.")
            continue
            
        dataset_entries.append({
            "audio": str(audio_path),
            "text": text,
            "unique_id": row.get('unique_id', f'sample_{idx}')
        })
    
    logger.info(f"Created dataset with {len(dataset_entries)} valid entries")
    
    # Split into train and eval if requested
    eval_dataset = None
    if training_args.do_eval and data_args.eval_split_size > 0:
        split_idx = int(len(dataset_entries) * (1 - data_args.eval_split_size))
        if split_idx == 0:
            split_idx = 1  # Ensure at least one for train
        if split_idx == len(dataset_entries):
            split_idx = len(dataset_entries) - 1  # Ensure at least one for eval
            
        # Shuffle before splitting
        np.random.seed(training_args.seed)
        np.random.shuffle(dataset_entries)
        
        train_dataset = dataset_entries[:split_idx]
        eval_dataset = dataset_entries[split_idx:]
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        train_dataset = dataset_entries
        
    return train_dataset, eval_dataset

# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: DataArguments,
                 chatterbox_model: ChatterboxTTS,
                 t3_config: T3Config,
                 dataset_entries: List[Dict[str, str]]):
        self.data_args = data_args
        self.chatterbox_model = chatterbox_model
        self.chatterbox_t3_config = t3_config
        self.dataset_entries = dataset_entries

        self.text_tokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve

        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)

    def __len__(self):
        return len(self.dataset_entries)

    def _load_audio_text_from_item(self, idx):
        item = self.dataset_entries[idx]
        audio_path = item["audio"]
        text = item["text"]
        
        try:
            wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
            return wav_16k, text
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0]).cpu()
        except Exception as e:
            logger.error(f"Error getting speaker embedding for item {idx}: {e}. Skipping.")
            return None

        normalized_text = punc_norm(text)
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0).cpu()
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token])])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        # Log text tokens for first few samples and periodically
        if idx < 5 or idx % 100 == 0:
            logger.info(f"📝 TOKENIZATION ANALYSIS - Sample {idx}")
            logger.info(f"  Original Text: '{text[:150]}{'...' if len(text) > 150 else ''}'")
            logger.info(f"  Normalized Text: '{normalized_text[:150]}{'...' if len(normalized_text) > 150 else ''}'")
            
            # Show detailed tokenization process
            try:
                # Show individual token decode for first 30 tokens
                individual_tokens = []
                for i, token_id in enumerate(raw_text_tokens[:30].tolist()):
                    token_text = self.text_tokenizer.decode([token_id])
                    individual_tokens.append(f"{token_id}:'{token_text}'")
                logger.info(f"  Raw Tokens (first 30): [{', '.join(individual_tokens)}]")
                
                # Show full tokens with BOS/EOS
                full_individual_tokens = []
                for i, token_id in enumerate(text_tokens[:30].tolist()):
                    token_text = self.text_tokenizer.decode([token_id])
                    if i == 0:
                        token_text += "(BOS)"
                    elif i == len(text_tokens) - 1:
                        token_text += "(EOS)"
                    full_individual_tokens.append(f"{token_id}:'{token_text}'")
                logger.info(f"  With BOS/EOS (first 30): [{', '.join(full_individual_tokens)}]")
                
            except Exception as e:
                logger.warning(f"  Error in detailed token logging: {e}")
            
            logger.info(f"  Token Counts: Raw={len(raw_text_tokens)}, With BOS/EOS={len(text_tokens)}, Max allowed={self.data_args.max_text_len}")
            logger.info(f"  Input IDs: {text_tokens.tolist()[:30]}{'...' if len(text_tokens) > 30 else ''}")
            logger.info(f"  Token Length Tensor: {text_token_len.item()}")
            logger.info(f"  🔄 Text round-trip test: '{self.text_tokenizer.decode(text_tokens.tolist())}'")
            logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()].cpu()
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None
            
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token])])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        # Log speech tokens for first few samples  
        if idx < 5:
            logger.info(f"Sample {idx} - Raw speech tokens: {raw_speech_tokens.tolist()[:20]}... (len={len(raw_speech_tokens)})")
            logger.info(f"Sample {idx} - Speech tokens with BOS/EOS: {speech_tokens.tolist()[:20]}... (len={len(speech_tokens)})")

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0).cpu()
            except Exception as e:
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len: 
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else: 
                cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)
        
        # Log conditioning prompt tokens for first few samples
        if idx < 5:
            logger.info(f"Sample {idx} - Conditioning prompt speech tokens: {cond_prompt_speech_tokens.tolist()[:20]}... (len={len(cond_prompt_speech_tokens)})")
        
        emotion_adv_scalar = 0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return_dict = {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }

        return return_dict

# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config  # Chatterbox T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features. Returning empty batch.")
            return {}
        features = valid_features

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        # Pad text tokens
        padded_text_tokens = torch.stack([
            F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
            for t in text_tokens_list
        ])  # shape: (B, max_text_len)

        # Pad speech tokens
        padded_speech_tokens = torch.stack([
            F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
            for s in speech_tokens_list
        ])  # shape: (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in features])      # (B,)
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in features])  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in features])             # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in features])  # (B, prompt_len)
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in features])  # (B, 1, 1)
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        # Shift off BOS from padded_text_tokens: new length = max_text_len - 1
        shifted_text = padded_text_tokens[:, 1:].contiguous()  # shape: (B, max_text_len - 1)
        T_text = shifted_text.size(1)

        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        arange_text = torch.arange(T_text, device=shifted_text.device)  # (T_text,)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]  # (B, T_text)

        labels_text = shifted_text.clone()           # (B, T_text)
        labels_text[mask_pad_text] = IGNORE_ID       # set pad/beyond to -100

        # --- Build labels_speech ---
        # Shift off BOS from padded_speech_tokens: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()  # shape: (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)

        # Mask positions t >= (speech_len - 1)
        speech_token_lens = speech_token_lens.to(shifted_speech.device)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        arange_speech = torch.arange(T_speech, device=shifted_speech.device)  # (T_speech,)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]  # (B, T_speech)

        # Mask positions t < prompt_len
        mask_prompt = arange_speech[None] < prompt_len  # (1, T_speech) -> broadcast to (B, T_speech)
        mask_prompt = mask_prompt.expand(batch_size, T_speech)

        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt  # (B, T_speech)

        labels_speech = shifted_speech.clone()          # (B, T_speech)
        labels_speech[mask_speech_total] = IGNORE_ID    # set prompt & pad to -100

        # Log batch information periodically
        if torch.rand(1).item() < 0.15:  # Log ~15% of batches to show tokenization details
            logger.info(f"🔄 BATCH TOKENIZATION - Size: {batch_size}, Max text len: {max_text_len}, Max speech len: {max_speech_len}")
            logger.info(f"  Text token lengths: {text_token_lens.tolist()}")
            logger.info(f"  Speech token lengths: {speech_token_lens.tolist()}")
            
            # Show detailed token information for first sample in batch
            try:
                first_sample_text_tokens = padded_text_tokens[0]
                first_sample_labels = labels_text[0]
                
                # Find non-padding tokens
                non_pad_mask = first_sample_text_tokens != self.text_pad_token_id
                actual_tokens = first_sample_text_tokens[non_pad_mask]
                
                logger.info(f"  📄 First sample in batch:")
                logger.info(f"    Padded text tokens: {first_sample_text_tokens.tolist()[:30]}{'...' if len(first_sample_text_tokens) > 30 else ''}")
                logger.info(f"    Actual tokens (no padding): {actual_tokens.tolist()[:30]}{'...' if len(actual_tokens) > 30 else ''}")
                logger.info(f"    Text labels: {first_sample_labels.tolist()[:30]}{'...' if len(first_sample_labels) > 30 else ''}")
                
                # Show token decoding for verification
                if hasattr(self, 'text_tokenizer'):
                    try:
                        decoded_text = self.text_tokenizer.decode(actual_tokens.tolist())
                        logger.info(f"    Decoded text: '{decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}'")
                    except Exception as e:
                        logger.info(f"    Decoding failed: {e}")
                
                # Show mask information
                pad_positions = (first_sample_text_tokens == self.text_pad_token_id).sum().item()
                ignore_positions = (first_sample_labels == -100).sum().item()
                logger.info(f"    Padding positions: {pad_positions}, Ignored label positions: {ignore_positions}")
                
            except Exception as e:
                logger.warning(f"  Error in batch token logging: {e}")
            
            logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return {
            "text_tokens": padded_text_tokens, 
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens, 
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,       # (B, max_text_len - 1) masked with -100
            "labels_speech": labels_speech,   # (B, max_speech_len - 1) masked with -100
        }

# --- Model Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = chatterbox_t3_config.text_tokens_dict_size
        hf_config_instance.speech_tokens_dict_size = chatterbox_t3_config.speech_tokens_dict_size
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = chatterbox_t3_config.speech_cond_prompt_len
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

    def forward(self,
                text_tokens,
                text_token_lens,
                speech_tokens,
                speech_token_lens,
                t3_cond_speaker_emb,
                t3_cond_prompt_speech_tokens,
                t3_cond_emotion_adv,
                labels_text=None,
                labels_speech=None):

        # Log input information periodically with detailed tokenization 
        if torch.rand(1).item() < 0.08:  # Log ~8% of forward passes with token details
            is_training = self.training
            phase = "🔥 TRAINING" if is_training else "📊 EVALUATION"
            
            logger.info(f"🎯 MODEL FORWARD {phase} - Batch size: {text_tokens.shape[0]}")
            logger.info(f"  Text tokens shape: {text_tokens.shape}, Speech tokens shape: {speech_tokens.shape}")
            logger.info(f"  Text token lengths: {text_token_lens.tolist()}")
            logger.info(f"  Speech token lengths: {speech_token_lens.tolist()}")
            
            # Show detailed token information for first sample in batch
            try:
                first_text_tokens = text_tokens[0]
                first_text_len = text_token_lens[0].item()
                actual_text_tokens = first_text_tokens[:first_text_len]
                
                logger.info(f"  📄 First sample tokens being processed:")
                logger.info(f"    Input text tokens: {actual_text_tokens.tolist()[:30]}{'...' if len(actual_text_tokens) > 30 else ''}")
                logger.info(f"    Full padded tokens: {first_text_tokens.tolist()[:30]}{'...' if len(first_text_tokens) > 30 else ''}")
                
                # Show labels if available (during training)
                if labels_text is not None:
                    first_text_labels = labels_text[0]
                    # Show non-ignored labels
                    valid_labels = first_text_labels[first_text_labels != -100]
                    logger.info(f"    Text labels (non-ignored): {valid_labels.tolist()[:30]}{'...' if len(valid_labels) > 30 else ''}")
                    
                if labels_speech is not None:
                    first_speech_labels = labels_speech[0]
                    valid_speech_labels = first_speech_labels[first_speech_labels != -100]
                    logger.info(f"    Speech labels (non-ignored): {valid_speech_labels.tolist()[:20]}{'...' if len(valid_speech_labels) > 20 else ''}")
                
                # Show conditioning information
                logger.info(f"    Speaker embedding shape: {t3_cond_speaker_emb[0].shape}")
                logger.info(f"    Conditioning prompt tokens: {t3_cond_prompt_speech_tokens[0].tolist()[:20]}{'...' if len(t3_cond_prompt_speech_tokens[0]) > 20 else ''}")
                logger.info(f"    Emotion advancement: {t3_cond_emotion_adv[0].item():.3f}")
                
            except Exception as e:
                logger.warning(f"  Error in model forward token logging: {e}")
            
            logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        current_t3_cond = T3Cond(
            speaker_emb=t3_cond_speaker_emb,
            cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv
        ).to(device=self.t3.device)

        loss_text, loss_speech, speech_logits = self.t3.loss(
            t3_cond=current_t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            labels_text=labels_text,
            labels_speech=labels_speech
        )
        
        total_loss = loss_text + loss_speech

        # Log loss information periodically with token analysis
        if torch.rand(1).item() < 0.08:  # Log ~8% of forward passes
            is_training = self.training
            phase = "🔥 TRAINING" if is_training else "📊 EVALUATION"
            
            logger.info(f"💯 LOSS ANALYSIS {phase}")
            logger.info(f"  Text loss: {loss_text.item():.6f}")
            logger.info(f"  Speech loss: {loss_speech.item():.6f}")
            logger.info(f"  Total loss: {total_loss.item():.6f}")
            
            # Add token analysis context
            try:
                batch_size = text_tokens.shape[0]
                avg_text_len = text_token_lens.float().mean().item()
                avg_speech_len = speech_token_lens.float().mean().item()
                
                logger.info(f"  Batch context: {batch_size} samples, avg text len: {avg_text_len:.1f}, avg speech len: {avg_speech_len:.1f}")
                
                # Show logits information
                if speech_logits is not None:
                    logger.info(f"  Speech logits shape: {speech_logits.shape}")
                    # Show prediction confidence (softmax of first few tokens)
                    with torch.no_grad():
                        first_logits = speech_logits[0, :5]  # First 5 predictions of first sample
                        probs = torch.softmax(first_logits, dim=-1)
                        max_probs, pred_tokens = torch.max(probs, dim=-1)
                        logger.info(f"  First 5 predicted tokens: {pred_tokens.tolist()} (confidence: {max_probs.tolist()})")
                
            except Exception as e:
                logger.warning(f"  Error in loss analysis logging: {e}")
            
            logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Return loss and additional outputs in a dict format for proper logging
        return {
            'loss': total_loss,
            'text_loss': loss_text,
            'speech_loss': loss_speech,
            'logits': speech_logits
        }

trainer_instance: Optional[Trainer] = None

def main():
    global trainer_instance

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    logger.info("Loading ChatterboxTTS model...")

    original_model_dir_for_copy: Optional[Path] = None
    if model_args.local_model_dir:
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        local_dir_path = Path(model_args.local_model_dir)
        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=str(local_dir_path), device="cuda")
        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(training_args.output_dir) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]

        from huggingface_hub import hf_hub_download as hf_download

        for f in files_to_download:
            try: 
                hf_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: 
                logger.warning(f"Could not download {f} from {repo_to_download}: {e}.")

        try: 
            hf_download(repo_id=repo_to_download, filename="conds.pt", local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
        except: 
            logger.info("conds.pt not found on Hub or failed to download for this model.")

        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cuda")
        original_model_dir_for_copy = download_dir

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    logger.info(f"🔧 T3 Model Configuration:")
    logger.info(f"  - Text vocab size: {chatterbox_t3_config_instance.text_tokens_dict_size}")
    logger.info(f"  - Speech vocab size: {chatterbox_t3_config_instance.speech_tokens_dict_size}")
    logger.info(f"  - Max text tokens: {chatterbox_t3_config_instance.max_text_tokens}")
    logger.info(f"  - Max speech tokens: {chatterbox_t3_config_instance.max_speech_tokens}")

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters(): 
            param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters(): 
            param.requires_grad = False
        logger.info("S3Gen model frozen.")
    for param in t3_model.parameters(): 
        param.requires_grad = True
    
    # Create model wrapper first
    hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)
    
    # Freeze original text embeddings if specified
    if model_args.freeze_text_embeddings is not None:
        freeze_vocab_size = model_args.freeze_text_embeddings
        current_vocab_size = chatterbox_t3_config_instance.text_tokens_dict_size
        if current_vocab_size > freeze_vocab_size:
            # We'll mask gradients in a training hook instead of setting requires_grad
            def mask_old_token_gradients(module, grad_input, grad_output):
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    module.weight.grad[:freeze_vocab_size] = 0
            
            t3_model.text_emb.register_backward_hook(mask_old_token_gradients)
            t3_model.text_head.register_backward_hook(mask_old_token_gradients)
            logger.info(f"Added gradient masking for original text embeddings (first {freeze_vocab_size} tokens)")
        else:
            logger.warning(f"Cannot freeze {freeze_vocab_size} tokens - current vocab size is only {current_vocab_size}")

    logger.info("T3 model set to trainable.")

    logger.info("Loading and processing Cartesia dataset...")
    
    # Load Cartesia dataset
    train_dataset_entries, eval_dataset_entries = load_cartesia_dataset(data_args, training_args)

    train_dataset = SpeechFineTuningDataset(
        data_args,
        chatterbox_model,
        chatterbox_t3_config_instance,
        train_dataset_entries
    )

    logger.info(f"📊 Training dataset loaded with {len(train_dataset)} samples")

    eval_dataset = None
    if eval_dataset_entries and training_args.do_eval:
        eval_dataset = SpeechFineTuningDataset(
            data_args,
            chatterbox_model,
            chatterbox_t3_config_instance,
            eval_dataset_entries
        )
        logger.info(f"📊 Evaluation dataset loaded with {len(eval_dataset)} samples")
    else:
        logger.info("📊 No evaluation dataset configured")

    data_collator = SpeechDataCollator(
        chatterbox_t3_config_instance, 
        chatterbox_t3_config_instance.stop_text_token,
        chatterbox_t3_config_instance.stop_speech_token
    )

    logger.info(f"📊 Data collator configured with:")
    logger.info(f"  - Text pad token ID: {chatterbox_t3_config_instance.stop_text_token}")
    logger.info(f"  - Speech pad token ID: {chatterbox_t3_config_instance.stop_speech_token}")
    logger.info(f"  - Speech conditioning prompt length: {chatterbox_t3_config_instance.speech_cond_prompt_len}")

    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    
    # Add audio generation callback for monitoring training progress
    audio_callback = AudioGenerationCallback(
        chatterbox_model=chatterbox_model,
        output_dir=training_args.output_dir,
        generation_interval=250  # Generate audio every 250 steps
    )
    callbacks.append(audio_callback)
    logger.info("🎵 Added audio generation callback - will generate test samples every 250 steps and at epoch end")

    trainer_instance = Trainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Set proper label names for the trainer
    if training_args.label_names is None: 
        trainer_instance.label_names = ["labels_text", "labels_speech"]

    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model()
        
        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = trainer_instance.model.t3 if hasattr(trainer_instance.model, 't3') else trainer_instance.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()
        
        output_t3_safetensor_path = Path(training_args.output_dir) / "t3_cfg.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        if original_model_dir_for_copy:
            import shutil
            for f_name in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]:
                src_path = original_model_dir_for_copy / f_name
                if src_path.exists(): 
                    shutil.copy2(src_path, Path(training_args.output_dir) / f_name)
            if (original_model_dir_for_copy / "conds.pt").exists():
                shutil.copy2(original_model_dir_for_copy / "conds.pt", Path(training_args.output_dir) / "conds.pt")
            logger.info(f"Full model components structured in {training_args.output_dir}")

        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("Finetuning script finished.")

if __name__ == "__main__":
    main()
