#!/usr/bin/env python3
"""
Script to download Whisper Hindi dataset, transliterate sentences to English,
and upload the new dataset to Hugging Face.
"""

import os
import json
import requests
import time
import pickle
import shutil
import tempfile
import gc
import pandas as pd
import csv
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datasets import load_dataset, Dataset, Audio, DatasetDict
from huggingface_hub import HfApi, login
import logging
from dotenv import load_dotenv

load_dotenv()

# Import configuration
try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    MAX_WORKERS = 8
    REQUESTS_PER_SECOND = 20.0
    BATCH_SIZE = 50
    CHUNK_SIZE = 500
    API_TIMEOUT = 30
    SAVE_DIR = "processed_dataset"
    TARGET_REPO_ID = "hrusheekeshsawarkar/whisper_hindi_small_T13N"
    LOG_LEVEL = "INFO"
    PROGRESS_LOG_INTERVAL = 10

# Import memory monitor
try:
    from memory_monitor import MemoryMonitor
except ImportError:
    MemoryMonitor = None

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HindiTransliterator:
    def __init__(self, max_workers: int = 5, requests_per_second: float = 10.0):
        self.api_endpoint = "https://revapi.reverieinc.com/"
        self.headers = {
            'Content-Type': 'application/json',
            'REV-API-KEY': '172c5bb5af18516905473091fd58d30afe740b3f',
            'REV-APP-ID': 'rev.transliteration',
            'src_lang': 'hi',
            'tgt_lang': 'en',
            'REV-APPNAME': 'transliteration',
            'domain': '1',
            'cnt_lang': 'hi'
        }
        self.max_workers = max_workers
        self.min_delay = 1.0 / requests_per_second  # Minimum delay between requests
        self.lock = Lock()  # Thread-safe operations
        self.last_request_time = 0
        
    def _rate_limit(self):
        """
        Thread-safe rate limiting
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()

    def transliterate_text(self, text: str) -> str:
        """
        Transliterate Hindi text to English using Reverie API with rate limiting
        """
        # Apply rate limiting
        self._rate_limit()
        
        payload = {
            "data": [text],
            "isBulk": False,
            "ignoreTaggedEntities": False,
            "convertNumber": "Words"
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle the correct Reverie API response structure
                if ('responseList' in result and 
                    len(result['responseList']) > 0):
                    
                    first_response = result['responseList'][0]
                    if isinstance(first_response, dict) and 'outString' in first_response:
                        out_string = first_response['outString']
                        if isinstance(out_string, list) and len(out_string) > 0:
                            transliterated = str(out_string[0]).strip()
                        else:
                            transliterated = str(out_string).strip()
                        
                        if transliterated and transliterated != text:
                            logger.debug(f"Successfully transliterated: '{text}' -> '{transliterated}'")
                            return transliterated
                        else:
                            logger.warning(f"Empty or same transliteration result for: {text[:50]}...")
                            return text
                    else:
                        logger.warning(f"Unexpected response structure for text: {text[:50]}...")
                        logger.debug(f"First response: {first_response}")
                        return text
                else:
                    logger.warning(f"Unexpected API response structure for text: {text[:50]}...")
                    logger.warning(f"Full response: {result}")
                    return text  # Return original text if transliteration fails
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return text  # Return original text if API fails
                
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out for text: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"Error transliterating text '{text[:50]}...': {str(e)}")
            return text
    
    def transliterate_batch(self, texts: List[str], batch_size: int = None) -> List[str]:
        """
        Transliterate a batch of texts using concurrent processing with rate limiting and memory management
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
            
        logger.info(f"Starting transliteration of {len(texts)} texts using {self.max_workers} workers")
        logger.info(f"Rate limit: {1/self.min_delay:.1f} requests per second")
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"Initial memory usage: {initial_memory:.1f} MB")
        
        transliterated = [None] * len(texts)  # Pre-allocate result list to maintain order
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch = texts[i:batch_end]
            batch_num = i//batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (samples {i+1}-{batch_end}) with {self.max_workers} workers")
            
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks for this batch
                future_to_index = {
                    executor.submit(self.transliterate_text, text): i + j 
                    for j, text in enumerate(batch)
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        transliterated[index] = result
                        completed += 1
                        
                        # Log progress every N completions
                        if completed % PROGRESS_LOG_INTERVAL == 0:
                            logger.debug(f"Batch {batch_num}: completed {completed}/{len(batch)} samples")
                            
                    except Exception as e:
                        logger.error(f"Error processing text at index {index}: {str(e)}")
                        # Use original text as fallback
                        transliterated[index] = texts[index]
                
                # Clear futures dictionary to free memory
                del future_to_index
            
            # Clear batch from memory
            del batch
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Monitor memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.debug(f"Memory usage after batch {batch_num}: {current_memory:.1f} MB")
            
            logger.info(f"Completed batch {batch_num}/{total_batches} ({batch_end}/{len(texts)} total samples)")
            
            # Brief pause between batches to be respectful to the API
            if i + batch_size < len(texts):
                time.sleep(1)
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage - Initial: {initial_memory:.1f} MB, Final: {final_memory:.1f} MB, Increase: {final_memory - initial_memory:.1f} MB")
        
        return transliterated

def download_dataset():
    """
    Download the original Hindi dataset from Hugging Face
    """
    logger.info("Downloading original dataset from Hugging Face...")
    
    try:
        dataset = load_dataset("shields/whisper-small-hindi", token=os.getenv("HF_TOKEN"))
        logger.info(f"Dataset downloaded successfully. Available splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def process_dataset_incremental(dataset, transliterator: HindiTransliterator, chunk_size: int = 500):
    """
    Process the dataset by transliterating all sentences with incremental saving to avoid memory issues
    """
    logger.info("Starting incremental transliteration process...")
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor() if MemoryMonitor else None
    if memory_monitor:
        logger.info("Memory monitoring enabled")
        initial_status = memory_monitor.check_memory_status()
        logger.info(f"Initial memory: Process {initial_status['process_mb']:.1f}MB, System {initial_status['system_percent']:.1f}%")
    
    # Create temporary directory for storing chunks
    temp_dir = tempfile.mkdtemp(prefix="transliteration_chunks_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        processed_splits = {}
        
        for split_name, split_data in dataset.items():
            logger.info(f"Processing split: {split_name} ({len(split_data)} samples)")
            
            # Create directory for this split's chunks
            split_temp_dir = os.path.join(temp_dir, split_name)
            os.makedirs(split_temp_dir, exist_ok=True)
            
            total_samples = len(split_data)
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            chunk_files = []
            
            # Process each chunk and save immediately
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, total_samples)
                
                logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} for split {split_name}")
                logger.info(f"Chunk range: {chunk_start+1}-{chunk_end}")
                
                # Extract chunk data
                chunk_data = split_data.select(range(chunk_start, chunk_end))
                sentences = chunk_data['sentence']
                
                # Transliterate sentences
                transliterated_sentences = transliterator.transliterate_batch(sentences)
                
                # Create chunk dataset
                chunk_dataset_dict = {
                    'audio': chunk_data['audio'],
                    'sentence_hindi': sentences,
                    'sentence_english': transliterated_sentences
                }
                
                chunk_dataset = Dataset.from_dict(chunk_dataset_dict)
                chunk_dataset = chunk_dataset.cast_column("audio", Audio())
                
                # Save chunk immediately
                chunk_file = os.path.join(split_temp_dir, f"chunk_{chunk_idx:04d}")
                chunk_dataset.save_to_disk(chunk_file)
                chunk_files.append(chunk_file)
                
                logger.info(f"Saved chunk {chunk_idx + 1}/{num_chunks} to disk")
                
                # Clear memory
                del chunk_data, sentences, transliterated_sentences, chunk_dataset_dict, chunk_dataset
                gc.collect()
                
                # Memory monitoring
                if memory_monitor:
                    status = memory_monitor.check_memory_status()
                    if status['warning']:
                        logger.warning(status['warning'])
                    
                    # Log memory status every 5 chunks
                    if (chunk_idx + 1) % 5 == 0:
                        logger.info(f"Memory status - Process: {status['process_mb']:.1f}MB, System: {status['system_percent']:.1f}%")
                    
                    # Critical memory check
                    if status['critical']:
                        logger.error("CRITICAL MEMORY USAGE - Consider stopping and reducing CHUNK_SIZE")
                        logger.error("Current process memory: {:.1f}MB".format(status['process_mb']))
                        logger.error("System memory usage: {:.1f}%".format(status['system_percent']))
            
            # Merge all chunks for this split
            logger.info(f"Merging {len(chunk_files)} chunks for split {split_name}")
            processed_splits[split_name] = merge_chunks(chunk_files)
            
            logger.info(f"Completed processing split: {split_name}")
        
        # Final memory report
        if memory_monitor:
            logger.info("Final memory report:")
            memory_monitor.print_memory_report()
        
        return processed_splits
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

def merge_chunks(chunk_files: List[str]) -> Dataset:
    """
    Merge multiple chunk files into a single dataset
    """
    logger.info(f"Merging {len(chunk_files)} chunks...")
    
    # Load first chunk to get the structure
    first_chunk = Dataset.load_from_disk(chunk_files[0])
    
    if len(chunk_files) == 1:
        return first_chunk
    
    # Collect all data
    all_audio = []
    all_hindi = []
    all_english = []
    
    for i, chunk_file in enumerate(chunk_files):
        logger.debug(f"Loading chunk {i+1}/{len(chunk_files)}")
        chunk = Dataset.load_from_disk(chunk_file)
        
        all_audio.extend(chunk['audio'])
        all_hindi.extend(chunk['sentence_hindi'])
        all_english.extend(chunk['sentence_english'])
        
        # Clear chunk from memory
        del chunk
        gc.collect()
    
    # Create merged dataset
    merged_data = {
        'audio': all_audio,
        'sentence_hindi': all_hindi,
        'sentence_english': all_english
    }
    
    merged_dataset = Dataset.from_dict(merged_data)
    merged_dataset = merged_dataset.cast_column("audio", Audio())
    
    logger.info(f"Merged dataset created with {len(merged_dataset)} samples")
    return merged_dataset

# Keep the old function for backward compatibility but use the new one
def process_dataset(dataset, transliterator: HindiTransliterator, chunk_size: int = 500):
    """
    Process the dataset - now uses incremental approach to avoid memory issues
    """
    return process_dataset_incremental(dataset, transliterator, chunk_size)

def save_dataset_as_csv(processed_dataset, save_dir: str = "processed_dataset"):
    """
    Save the processed dataset locally as CSV files
    """
    logger.info(f"Saving processed dataset as CSV files to: {save_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        csv_files = {}
        total_samples = 0
        splits_info = {}
        
        for split_name, split_data in processed_dataset.items():
            logger.info(f"Saving split '{split_name}' as CSV...")
            
            # Prepare data for CSV
            csv_data = []
            
            for i in range(len(split_data)):
                # Get audio file path or data
                audio_info = split_data[i]['audio']
                if isinstance(audio_info, dict):
                    # Audio is loaded as dict with 'array' and 'sampling_rate'
                    audio_path = f"audio_{split_name}_{i:06d}.wav"  # We'll save audio separately
                    sampling_rate = audio_info.get('sampling_rate', 16000)
                else:
                    audio_path = str(audio_info)
                    sampling_rate = 16000
                
                csv_row = {
                    'audio_path': audio_path,
                    'sampling_rate': sampling_rate,
                    'sentence_hindi': split_data[i]['sentence_hindi'],
                    'sentence_english': split_data[i]['sentence_english']
                }
                csv_data.append(csv_row)
            
            # Save CSV file
            csv_filename = f"{split_name}.csv"
            csv_path = os.path.join(save_dir, csv_filename)
            csv_files[split_name] = csv_path
            
            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['audio_path', 'sampling_rate', 'sentence_hindi', 'sentence_english']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            split_samples = len(csv_data)
            total_samples += split_samples
            splits_info[split_name] = split_samples
            
            logger.info(f"Saved {split_samples} samples for split '{split_name}' to {csv_filename}")
        
        # Save metadata
        metadata = {
            "total_samples": total_samples,
            "splits": splits_info,
            "csv_files": csv_files,
            "columns": ['audio_path', 'sampling_rate', 'sentence_hindi', 'sentence_english'],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "csv"
        }
        
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"CSV dataset saved successfully! Total samples: {total_samples}")
        logger.info(f"Splits: {splits_info}")
        logger.info(f"Files saved in: {save_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset as CSV: {str(e)}")
        return False

def save_dataset_locally(processed_dataset, save_dir: str = "processed_dataset", format: str = "csv"):
    """
    Save the processed dataset locally in specified format
    """
    if format == "csv":
        return save_dataset_as_csv(processed_dataset, save_dir)
    else:
        # Original HF dataset format
        logger.info(f"Saving processed dataset locally to: {save_dir}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save as Hugging Face dataset format
            dataset_dict = DatasetDict(processed_dataset)
            dataset_dict.save_to_disk(save_dir)
            
            # Also save metadata
            metadata = {
                "total_samples": sum(len(split) for split in processed_dataset.values()),
                "splits": {name: len(split) for name, split in processed_dataset.items()},
                "columns": list(processed_dataset[list(processed_dataset.keys())[0]].column_names),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "format": "hf_dataset"
            }
            
            with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset saved successfully! Total samples: {metadata['total_samples']}")
            logger.info(f"Splits: {metadata['splits']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving dataset locally: {str(e)}")
            return False

def load_dataset_locally(save_dir: str = "processed_dataset"):
    """
    Load the locally saved dataset
    """
    logger.info(f"Loading dataset from: {save_dir}")
    
    try:
        if not os.path.exists(save_dir):
            logger.error(f"Dataset directory not found: {save_dir}")
            return None
        
        # Check metadata to determine format
        metadata_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            dataset_format = metadata.get("format", "hf_dataset")
            logger.info(f"Dataset format: {dataset_format}")
            
            if dataset_format == "csv":
                # Load CSV files
                processed_dataset = {}
                for split_name, csv_path in metadata.get("csv_files", {}).items():
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        logger.info(f"Loaded {len(df)} samples for split '{split_name}' from CSV")
                        processed_dataset[split_name] = df
                    else:
                        logger.warning(f"CSV file not found: {csv_path}")
                
                logger.info(f"Loaded CSV dataset with {metadata['total_samples']} samples")
                logger.info(f"Splits: {metadata['splits']}")
                return processed_dataset
            else:
                # Load HF dataset format
                dataset_dict = DatasetDict.load_from_disk(save_dir)
                logger.info(f"Loaded HF dataset with {metadata['total_samples']} samples")
                logger.info(f"Splits: {metadata['splits']}")
                return dict(dataset_dict)
        else:
            # Try to load as HF dataset format (fallback)
            dataset_dict = DatasetDict.load_from_disk(save_dir)
            return dict(dataset_dict)
        
    except Exception as e:
        logger.error(f"Error loading dataset locally: {str(e)}")
        return None

def upload_csv_to_huggingface(save_dir: str, repo_id: str):
    """
    Upload CSV files to Hugging Face
    """
    logger.info(f"Uploading CSV dataset to Hugging Face repository: {repo_id}")
    
    try:
        # Login to Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. You may need to login manually.")
            try:
                login()  # This will prompt for token if not already logged in
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face: {str(e)}")
                logger.info("Please set the HF_TOKEN environment variable or run 'huggingface-cli login'")
                return False
        else:
            login(token=hf_token)
        
        # Initialize HF API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=hf_token)
            logger.info(f"Repository {repo_id} ready")
        except Exception as e:
            logger.warning(f"Repository creation warning: {str(e)}")
        
        # Upload all files in the save directory
        files_uploaded = []
        for file_name in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file_name)
            if os.path.isfile(file_path):
                logger.info(f"Uploading file: {file_name}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
                files_uploaded.append(file_name)
        
        logger.info(f"CSV dataset uploaded successfully! Files: {files_uploaded}")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading CSV dataset: {str(e)}")
        return False

def upload_to_huggingface(processed_dataset, repo_id: str, dataset_format: str = "hf"):
    """
    Upload the processed dataset to Hugging Face
    """
    if dataset_format == "csv":
        # This is for backward compatibility when called with processed pandas dataframes
        logger.warning("upload_to_huggingface called with CSV format data. Use upload_csv_to_huggingface instead.")
        return False
    
    logger.info(f"Uploading HF dataset to Hugging Face repository: {repo_id}")
    
    try:
        # Login to Hugging Face (you'll need to set your token)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. You may need to login manually.")
            try:
                login()  # This will prompt for token if not already logged in
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face: {str(e)}")
                logger.info("Please set the HF_TOKEN environment variable or run 'huggingface-cli login'")
                return False
        else:
            login(token=hf_token)
        
        # Upload each split
        for split_name, split_data in processed_dataset.items():
            logger.info(f"Uploading split: {split_name}")
            split_data.push_to_hub(
                repo_id=repo_id,
                split=split_name,
                token=hf_token
            )
        
        logger.info("Dataset uploaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        return False

def create_dataset_card(repo_id: str):
    """
    Create a dataset card for the new repository
    """
    dataset_card = """---
license: apache-2.0
task_categories:
- automatic-speech-recognition
language:
- hi
- en
tags:
- whisper
- hindi
- transliteration
- speech-recognition
- csv
size_categories:
- 1K<n<10K
---

# Whisper Small Hindi - Transliterated (CSV Format)

This dataset is derived from [shields/whisper-small-hindi](https://huggingface.co/datasets/shields/whisper-small-hindi) with Hindi sentences transliterated to English using the Reverie API.

## Dataset Description

This dataset contains audio recordings with corresponding transcriptions in both Hindi (Devanagari script) and transliterated English text. The data is provided in CSV format for easy processing and analysis.

## Files Structure

- `train.csv`: Training split data
- `test.csv`: Test split data (if available)
- `validation.csv`: Validation split data (if available)
- `metadata.json`: Dataset metadata and statistics

## CSV Columns

- `audio_path`: Path to audio file or audio identifier
- `sampling_rate`: Audio sampling rate (typically 16000 Hz)
- `sentence_hindi`: Original Hindi sentences in Devanagari script
- `sentence_english`: Transliterated English text using Reverie API

## Transliteration

The Hindi sentences were transliterated to English using the Reverie API transliteration service with the following configuration:
- API Endpoint: `https://revapi.reverieinc.com/`
- Source Language: Hindi (hi)
- Target Language: English (en)
- Domain: 1

## Usage

### Loading with pandas
```python
import pandas as pd

# Load training data
train_df = pd.read_csv("train.csv")
print(f"Training samples: {len(train_df)}")

# Access the data
for idx, row in train_df.iterrows():
    audio_path = row['audio_path']
    hindi_text = row['sentence_hindi']
    english_text = row['sentence_english']
    # Process your data
```

### Loading with Hugging Face datasets
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "test": "test.csv"  # if available
})
```

## Processing Information

This dataset was processed using chunk-by-chunk transliteration to handle large datasets efficiently:
- Concurrent API processing with rate limiting
- Memory-efficient chunk processing
- Fallback to original text on API failures
- Thread-safe operations

## Citation

Please cite the original dataset:
```
@dataset{shields_whisper_small_hindi,
  title={whisper-small-hindi},
  author={shields},
  year={2023},
  url={https://huggingface.co/datasets/shields/whisper-small-hindi}
}
```

## License

This dataset follows the same license as the original dataset (Apache 2.0).
"""
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        logger.info("Dataset card created successfully!")
    except Exception as e:
        logger.error(f"Error creating dataset card: {str(e)}")

def test_transliteration():
    """
    Test the transliteration with a few samples
    """
    logger.info("Testing transliteration with sample data...")
    
    # Test sentences
    test_sentences = [
        "हमने उसका जन्मदिन मनाया।",
        "वह भारत और चीन दोनो में बहुत मशहूर है।",
        "मुझे और वक़्त दो।",
        "क्या सवाल है!",
        "मेरे पास एक बंदूक है।"
    ]
    
    # Use fewer workers for testing
    transliterator = HindiTransliterator(max_workers=2, requests_per_second=5.0)
    
    print("\nTesting transliteration:")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        transliterated = transliterator.transliterate_text(sentence)
        print(f"{i}. Hindi: {sentence}")
        print(f"   English: {transliterated}")
        print()
    
    return True

def upload_local_dataset(save_dir: str = "processed_dataset", repo_id: str = "hrusheekeshsawarkar/whisper_hindi_small_T13N"):
    """
    Upload a locally saved dataset to Hugging Face
    """
    logger.info("Starting upload of locally saved dataset...")
    
    try:
        # Check if dataset exists
        if not os.path.exists(save_dir):
            logger.error(f"Dataset directory not found: {save_dir}")
            return False
        
        # Check metadata to determine format
        metadata_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            dataset_format = metadata.get("format", "hf_dataset")
            logger.info(f"Detected dataset format: {dataset_format}")
            
            if dataset_format == "csv":
                # Upload CSV files directly
                success = upload_csv_to_huggingface(save_dir, repo_id)
            else:
                # Load and upload HF dataset
                processed_dataset = load_dataset_locally(save_dir)
                if processed_dataset is None:
                    logger.error("Failed to load local dataset")
                    return False
                success = upload_to_huggingface(processed_dataset, repo_id, dataset_format="hf")
        else:
            # Fallback to HF dataset format
            logger.info("No metadata found, assuming HF dataset format")
            processed_dataset = load_dataset_locally(save_dir)
            if processed_dataset is None:
                logger.error("Failed to load local dataset")
                return False
            success = upload_to_huggingface(processed_dataset, repo_id, dataset_format="hf")
        
        if success:
            # Create dataset card
            create_dataset_card(repo_id)
            logger.info("Upload completed successfully!")
        else:
            logger.error("Upload failed")
            
        return success
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return False

def main():
    """
    Main execution function
    """
    import sys
    
    # Configuration (can be overridden by config.py)
    target_repo_id = TARGET_REPO_ID
    save_dir = SAVE_DIR
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\nHindi to English Transliteration Pipeline")
            print("="*50)
            print("Usage: python transliterate_hi_en.py [OPTION]")
            print("\nOptions:")
            print("  --test              Test transliteration with sample sentences")
            print("  --csv               Full pipeline: download -> transliterate -> save as CSV -> upload")
            print("  --process-only      Process and save locally as CSV only (no upload)")
            print("  --upload            Upload previously saved dataset to Hugging Face")
            print("  --upload-only       Same as --upload")
            print("  --help, -h          Show this help message")
            print("\nDefault (no arguments): Full pipeline with HF dataset format")
            print(f"\nConfiguration:")
            print(f"  Workers: {MAX_WORKERS}")
            print(f"  Rate limit: {REQUESTS_PER_SECOND} requests/second")
            print(f"  Batch size: {BATCH_SIZE}")
            print(f"  Chunk size: {CHUNK_SIZE}")
            print(f"  Save directory: {SAVE_DIR}")
            print(f"  Target repository: {TARGET_REPO_ID}")
            return
        elif sys.argv[1] == '--test':
            test_transliteration()
            return
        elif sys.argv[1] == '--upload':
            # Upload from local save
            upload_local_dataset(save_dir, target_repo_id)
            return
        elif sys.argv[1] == '--upload-only':
            # Upload from local save without processing
            upload_local_dataset(save_dir, target_repo_id)
            return
        elif sys.argv[1] == '--csv':
            # Full pipeline with CSV output
            logger.info("Starting Hindi to English transliteration pipeline (CSV format)...")
            logger.info(f"Configuration: {MAX_WORKERS} workers, {REQUESTS_PER_SECOND} req/sec, batch size {BATCH_SIZE}")
            try:
                # Step 1: Download the original dataset
                original_dataset = download_dataset()
                
                # Step 2: Initialize transliterator with multiprocessing
                transliterator = HindiTransliterator(max_workers=MAX_WORKERS, requests_per_second=REQUESTS_PER_SECOND)
                
                # Step 3: Process the dataset
                processed_dataset = process_dataset(original_dataset, transliterator, chunk_size=CHUNK_SIZE)
                
                # Step 4: Save locally as CSV
                save_success = save_dataset_locally(processed_dataset, save_dir, format="csv")
                
                if not save_success:
                    logger.error("Failed to save dataset locally")
                    return
                
                # Step 5: Upload CSV files to Hugging Face
                success = upload_csv_to_huggingface(save_dir, target_repo_id)
                
                if success:
                    # Step 6: Create dataset card
                    create_dataset_card(target_repo_id)
                    logger.info("CSV pipeline completed successfully!")
                else:
                    logger.error("CSV pipeline failed during upload step")
                    logger.info(f"Dataset is saved locally at: {save_dir}")
                    logger.info("To retry upload, run: python transliterate_hi_en.py --upload")
                    
            except Exception as e:
                logger.error(f"CSV pipeline failed: {str(e)}")
                raise
            return
        elif sys.argv[1] == '--process-only':
            # Process and save locally only, don't upload
            logger.info("Starting Hindi to English transliteration pipeline (process only)...")
            logger.info(f"Configuration: {MAX_WORKERS} workers, {REQUESTS_PER_SECOND} req/sec, batch size {BATCH_SIZE}")
            try:
                # Step 1: Download the original dataset
                original_dataset = download_dataset()
                
                # Step 2: Initialize transliterator with multiprocessing
                transliterator = HindiTransliterator(max_workers=MAX_WORKERS, requests_per_second=REQUESTS_PER_SECOND)
                
                # Step 3: Process the dataset
                processed_dataset = process_dataset(original_dataset, transliterator, chunk_size=CHUNK_SIZE)
                
                # Step 4: Save locally as CSV
                save_success = save_dataset_locally(processed_dataset, save_dir, format="csv")
                
                if save_success:
                    logger.info("Processing completed successfully!")
                    logger.info(f"Dataset saved as CSV files to: {save_dir}")
                    logger.info("To upload later, run: python transliterate_hi_en.py --upload")
                else:
                    logger.error("Processing failed during save step")
                    
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                raise
            return
    
    logger.info("Starting Hindi to English transliteration pipeline...")
    logger.info(f"Configuration: {MAX_WORKERS} workers, {REQUESTS_PER_SECOND} req/sec, batch size {BATCH_SIZE}")
    
    try:
        # Step 1: Download the original dataset
        original_dataset = download_dataset()
        
        # Step 2: Initialize transliterator with multiprocessing
        transliterator = HindiTransliterator(max_workers=MAX_WORKERS, requests_per_second=REQUESTS_PER_SECOND)
        
        # Step 3: Process the dataset
        processed_dataset = process_dataset(original_dataset, transliterator, chunk_size=CHUNK_SIZE)
        
        # Step 4: Save locally first as CSV
        save_success = save_dataset_locally(processed_dataset, save_dir, format="csv")
        
        if not save_success:
            logger.error("Failed to save dataset locally")
            return
        
        # Step 5: Upload CSV files to Hugging Face
        success = upload_csv_to_huggingface(save_dir, target_repo_id)
        
        if success:
            # Step 6: Create dataset card
            create_dataset_card(target_repo_id)
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Pipeline failed during upload step")
            logger.info(f"Dataset is saved locally at: {save_dir}")
            logger.info("To retry upload, run: python transliterate_hi_en.py --upload")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.info("If processing completed but upload failed, you can retry with: python transliterate_hi_en.py --upload")
        raise

if __name__ == "__main__":
    main()
