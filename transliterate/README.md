# Hindi to English Transliteration Pipeline

This script downloads the Whisper Hindi dataset, transliterates the Hindi sentences to English using the Reverie API, and uploads the processed dataset to Hugging Face.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Hugging Face authentication:**
   
   You need to authenticate with Hugging Face to upload the dataset. You can do this in one of two ways:
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```
   
   **Option B: CLI Login**
   ```bash
   huggingface-cli login
   ```
   
   To get your Hugging Face token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Write" permissions
   - Copy the token

3. **Verify repository access:**
   Make sure you have write access to the target repository:
   `https://huggingface.co/datasets/hrusheekeshsawarkar/whisper_hindi_small_T13N`

## Usage

**Test the transliteration first (recommended):**
```bash
python transliterate_hi_en.py --test
```

**Process dataset and save locally only (recommended for large datasets):**
```bash
python transliterate_hi_en.py --process-only
```

**Upload locally saved dataset to Hugging Face:**
```bash
python transliterate_hi_en.py --upload
```

**Check status of locally saved dataset:**
```bash
python check_dataset.py
```

**View current configuration settings:**
```bash
python config_info.py
```

**Monitor memory usage:**
```bash
python memory_monitor.py
```

**Run the full pipeline (process + upload):**
```bash
python transliterate_hi_en.py
```

### Available Commands:

- `--test`: Test transliteration with sample sentences
- `--process-only`: Download, transliterate, and save locally (no upload)
- `--upload`: Upload previously saved local dataset to Hugging Face
- No arguments: Run full pipeline (process + save locally + upload)

## Recommended Workflow (for large datasets):

1. **Test first**: `python transliterate_hi_en.py --test`
2. **Process and save locally**: `python transliterate_hi_en.py --process-only`
3. **Check saved dataset**: `python check_dataset.py`
4. **Upload when ready**: `python transliterate_hi_en.py --upload`

This approach prevents data loss if the process gets interrupted and allows you to verify the results before uploading.

## What the script does:

1. **Downloads** the original dataset from `shields/whisper-small-hindi`
2. **Transliterates** all Hindi sentences to English using the Reverie API
3. **Creates** a new dataset with three columns:
   - `audio`: Original audio files
   - `sentence_hindi`: Original Hindi sentences in Devanagari script
   - `sentence_english`: Transliterated English sentences
4. **Saves locally** to `processed_dataset/` directory (NEW: prevents data loss)
5. **Uploads** the processed dataset to `hrusheekeshsawarkar/whisper_hindi_small_T13N`
6. **Creates** a dataset card with metadata and documentation

## Local Dataset Storage:

The processed dataset is saved locally in the `processed_dataset/` directory with:
- Full dataset in Hugging Face format
- `metadata.json` with dataset statistics
- This allows you to retry uploads or inspect the data locally

## Features:

- **Multiprocessing**: Uses concurrent threads to speed up API calls (configurable)
- **Smart rate limiting**: Thread-safe rate limiting to respect API limits
- **Incremental processing**: Saves chunks immediately to prevent memory buildup
- **Memory monitoring**: Real-time memory usage tracking and warnings
- **Local saving**: Saves dataset locally before uploading (prevents data loss)
- **Memory efficient**: Processes data in chunks with automatic cleanup
- **Configurable**: Easy-to-modify configuration file for performance tuning
- **Error handling**: Gracefully handles API failures and network issues
- **Logging**: Detailed progress logging throughout the process
- **Batch processing**: Processes data in manageable batches
- **Recovery**: Can resume from saved local dataset if interrupted
- **Fallback**: If transliteration fails, keeps the original text

## Performance Configuration:

You can customize the processing speed and resource usage by editing `config.py`:

```python
# Multiprocessing configuration
MAX_WORKERS = 8  # Number of concurrent threads (reduced for stability)
REQUESTS_PER_SECOND = 20.0  # API rate limit (decreased if hitting limits)
BATCH_SIZE = 50  # Texts per batch (reduced for memory management)
CHUNK_SIZE = 500  # Samples per memory chunk (reduced to prevent memory issues)
```

**Performance Tips:**
- **Start conservative** - use default settings first
- **Increase MAX_WORKERS** (e.g., 10-15) if memory allows
- **Decrease REQUESTS_PER_SECOND** if you hit rate limits
- **Increase BATCH_SIZE** cautiously (watch memory usage)
- **Decrease CHUNK_SIZE** if getting killed (memory issues)

**Memory Management:**
- **Monitor memory**: `python memory_monitor.py`
- **Real-time tracking**: Memory usage logged during processing
- **Automatic warnings**: Alerts when memory usage gets high
- **Incremental saving**: Chunks saved immediately to prevent buildup

**Expected Performance:**
- Conservative settings: ~160 requests/second theoretical (8 workers × 20 req/sec)
- Full dataset (~9400 samples): ~1-2 minutes processing time
- Memory usage: Stays constant regardless of dataset size

## Dataset Structure:

The output dataset will have approximately 9,434 rows across train and test splits, with each row containing:
- Audio recording
- Original Hindi sentence
- Transliterated English sentence

## Notes:

- The script uses the Reverie API with the provided API key
- Processing time depends on the API response time (expect several hours for the full dataset)
- The script will preserve the original train/test splits from the source dataset

## Troubleshooting:

**If the process gets killed/interrupted:**
- Check if data was saved: `python check_dataset.py`
- If yes, upload directly: `python transliterate_hi_en.py --upload`
- If no, restart with: `python transliterate_hi_en.py --process-only`

**Memory issues (Process getting killed):**
- **NEW**: Script now uses incremental processing to prevent memory buildup
- **Monitor first**: `python memory_monitor.py` to check current usage
- **Reduce CHUNK_SIZE**: Try 250 or 100 in `config.py` (default: 500)
- **Reduce MAX_WORKERS**: Try 4-6 workers instead of 8
- **Reduce BATCH_SIZE**: Try 25 instead of 50
- **Check system memory**: Ensure you have at least 4GB free RAM

**API issues:**
- Test the API first: `python transliterate_hi_en.py --test`
- Check your API key is correct and has quota remaining
- If hitting rate limits, reduce `REQUESTS_PER_SECOND` in `config.py`
- If getting timeouts, reduce `MAX_WORKERS` to decrease concurrent load

**Performance tuning:**
- **Start conservative**: Use default settings first
- **Monitor memory**: Watch for warnings during processing
- **Gradual increases**: Only increase settings if memory allows
- **Watch for errors**: If you see many failures, reduce concurrency
- **Use debug logging**: Set `LOG_LEVEL = "DEBUG"` in `config.py`
- **Memory reports**: Script provides detailed memory usage reports 