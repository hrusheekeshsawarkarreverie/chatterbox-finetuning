# Hindi to English Transliteration Pipeline

This script downloads the `shields/whisper-small-hindi` dataset, transliterates Hindi sentences to English using the Reverie API, and uploads the processed dataset to Hugging Face.

## Features

- **Updated API Integration**: Uses the correct Reverie API endpoint and headers
- **CSV Output**: Saves datasets as CSV files for easy processing
- **Chunk-by-Chunk Processing**: Memory-efficient processing to avoid OOM issues
- **Concurrent Processing**: Thread-safe API calls with rate limiting
- **Robust Error Handling**: Fallback mechanisms and comprehensive logging
- **Flexible Workflows**: Multiple command options for different use cases

## Quick Start

### 1. Test the API
```bash
python transliterate_hi_en.py --test
```

### 2. Full Pipeline (Recommended)
Download → Transliterate → Save as CSV → Upload to HuggingFace:
```bash
python transliterate_hi_en.py --csv
```

### 3. Process Only (No Upload)
Download → Transliterate → Save as CSV locally:
```bash
python transliterate_hi_en.py --process-only
```

### 4. Upload Previously Processed Dataset
```bash
python transliterate_hi_en.py --upload
```

## Configuration

The script uses `config.py` for configuration. Default values:

- **Workers**: 10 concurrent threads
- **Rate Limit**: 50 requests/second
- **Batch Size**: 100 samples per batch
- **Chunk Size**: 1000 samples per chunk
- **Output Directory**: `processed_dataset/`
- **Target Repository**: `hrusheekeshsawarkar/whisper_hindi_small_T13N`

## API Configuration

The script uses the Reverie API with these settings:

- **Endpoint**: `https://revapi.reverieinc.com/`
- **API Key**: `172c5bb5af18516905473091fd58d30afe740b3f`
- **App ID**: `rev.transliteration`
- **Source Language**: Hindi (hi)
- **Target Language**: English (en)
- **Domain**: 1

## Output Format

### CSV Files Generated

The script generates CSV files with the following structure:

| Column | Description |
|--------|-------------|
| `audio_path` | Audio file path/identifier |
| `sampling_rate` | Audio sampling rate (16000 Hz) |
| `sentence_hindi` | Original Hindi text in Devanagari |
| `sentence_english` | Transliterated English text |

### Files Created

- `train.csv` - Training split data
- `test.csv` - Test split data (if available)
- `validation.csv` - Validation split data (if available)  
- `metadata.json` - Dataset statistics and configuration

## Example Usage

### Loading the CSV Data

```python
import pandas as pd

# Load the processed data
train_df = pd.read_csv('processed_dataset/train.csv')

print(f"Dataset size: {len(train_df)}")
print("\nSample data:")
for idx, row in train_df.head(3).iterrows():
    print(f"Hindi: {row['sentence_hindi']}")
    print(f"English: {row['sentence_english']}")
    print()
```

### Using with Hugging Face Datasets

```python
from datasets import load_dataset

# Load from CSV files
dataset = load_dataset('csv', data_files={
    'train': 'processed_dataset/train.csv',
    'test': 'processed_dataset/test.csv'
})

print(dataset)
```

## Requirements

Install dependencies:

```bash
pip install datasets huggingface_hub pandas requests python-dotenv psutil
```

## Environment Setup

1. **Hugging Face Token**: Set your HF token for uploading:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```
   
   Or create a `.env` file:
   ```
   HF_TOKEN=your_huggingface_token
   ```

2. **Configuration**: Create or modify `config.py` to adjust processing parameters:
   ```python
   MAX_WORKERS = 10
   REQUESTS_PER_SECOND = 50.0
   BATCH_SIZE = 100
   CHUNK_SIZE = 1000
   ```

## Command Options

```bash
# Show help
python transliterate_hi_en.py --help

# Test API with sample sentences
python transliterate_hi_en.py --test

# Full pipeline with CSV output
python transliterate_hi_en.py --csv

# Process only (save locally, no upload)
python transliterate_hi_en.py --process-only

# Upload previously processed dataset
python transliterate_hi_en.py --upload

# Default: Full pipeline with HF dataset format
python transliterate_hi_en.py
```

## Performance

The script is optimized for large datasets:

- **Memory Efficient**: Chunk-by-chunk processing prevents memory buildup
- **Rate Limited**: Respects API limits to avoid blocking
- **Concurrent**: Multi-threaded processing for faster execution
- **Cached**: Avoids redundant API calls for repeated text
- **Monitored**: Real-time memory and progress monitoring

## Troubleshooting

### Memory Issues
- Reduce `CHUNK_SIZE` in config.py
- Reduce `MAX_WORKERS` for lower memory usage
- Monitor memory with the built-in memory monitor

### API Issues
- Check API key and endpoint configuration
- Reduce `REQUESTS_PER_SECOND` if hitting rate limits
- Enable debug logging for detailed API responses

### Upload Issues
- Ensure HF_TOKEN is set correctly
- Check repository permissions
- Verify repository exists or can be created

## Sample Output

```
2025-06-30 16:52:03,896 - INFO - Starting Hindi to English transliteration pipeline (CSV format)...
2025-06-30 16:52:04,123 - INFO - Downloading original dataset from Hugging Face...
2025-06-30 16:52:08,456 - INFO - Processing split: train (2542 samples)
2025-06-30 16:52:08,789 - INFO - Processing chunk 1/3 for split train
2025-06-30 16:52:25,123 - INFO - Completed batch 1/10 (100/1000 total samples)
...
2025-06-30 16:58:42,456 - INFO - CSV dataset saved successfully! Total samples: 2542
2025-06-30 16:58:43,789 - INFO - Uploading CSV dataset to Hugging Face repository...
2025-06-30 16:59:15,123 - INFO - CSV pipeline completed successfully!
```

## License

This project follows the same license as the original dataset (Apache 2.0). 