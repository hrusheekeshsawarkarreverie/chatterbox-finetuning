# Transliteration-Enabled Fine-tuning Usage

## Overview

The fine-tuning script now supports on-the-fly Hindi to English transliteration during training using the Reverie API. This eliminates memory issues by processing sentences individually during training rather than pre-processing the entire dataset.

## Updated API Integration

- **Endpoint**: `https://revapi.reverieinc.com/`
- **Default API Key**: `172c5bb5af18516905473091fd58d30afe740b3f`
- **Default App ID**: `rev.transliteration`

## Usage Examples

### Basic Usage with Transliteration

```bash
python src/finetune_t3__t13n.py \
    --dataset_name "shields/whisper-small-hindi" \
    --enable_transliteration \
    --output_dir "./results" \
    --do_train \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5
```

### Advanced Configuration

```bash
python src/finetune_t3__t13n.py \
    --dataset_name "shields/whisper-small-hindi" \
    --enable_transliteration \
    --transliteration_api_key "your_custom_api_key" \
    --transliteration_app_id "rev.transliteration" \
    --transliteration_rate_limit 1.5 \
    --transliteration_cache_size 15000 \
    --transliteration_timeout 15.0 \
    --transliteration_fallback \
    --output_dir "./results" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 1000
```

## Transliteration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_transliteration` | `False` | Enable Hindi to English transliteration |
| `--transliteration_api_key` | `172c5bb5af18516905473091fd58d30afe740b3f` | Reverie API key |
| `--transliteration_app_id` | `rev.transliteration` | Reverie App ID |
| `--transliteration_rate_limit` | `2.0` | Max requests per second |
| `--transliteration_cache_size` | `10000` | LRU cache size |
| `--transliteration_timeout` | `10.0` | API timeout in seconds |
| `--transliteration_fallback` | `True` | Use original text on API failure |

## Key Features

### ✅ Memory Efficient
- On-the-fly transliteration during training
- No need to pre-process entire dataset
- Eliminates memory buildup issues

### ✅ Thread-Safe
- Safe for multi-worker data loading
- Thread-safe API calls with proper locking

### ✅ Rate Limited
- Configurable requests per second
- Respects API limits to avoid blocking

### ✅ Cached
- LRU cache for repeated sentences
- Avoids redundant API calls
- Configurable cache size

### ✅ Robust Error Handling
- Fallback to original text on failure
- Comprehensive error logging
- Configurable fallback behavior

### ✅ Performance Monitoring
- Cache statistics logging
- API response time tracking
- Training progress with transliteration metrics

## API Response Structure

The Reverie API returns responses in this format:
```json
{
  "responseList": [
    {
      "apiStatus": 2,
      "inString": "हमने उसका जन्मदिन मनाया।",
      "outString": ["humne uska janmadin manaya."]
    }
  ]
}
```

## Example Transliterations

| Original Hindi | Transliterated English |
|----------------|------------------------|
| हमने उसका जन्मदिन मनाया। | humne uska janmadin manaya. |
| नमस्ते, आप कैसे हैं? | namaste, aap kaise hain? |
| मुझे खुशी है कि आप यहाँ हैं। | mujhe khusi hai ki aap yahan hain. |
| यह एक परीक्षण वाक्य है। | yeh ek parikshan vakya hai. |

## Notes

1. **API Key**: The default API key is embedded in the script. You can override it with your own key.

2. **Fallback Behavior**: When `transliteration_fallback=True`, failed transliterations will use the original Hindi text. When `False`, failed samples are skipped.

3. **Rate Limiting**: The default rate limit is 2 requests/second. Adjust based on your API quota and needs.

4. **Cache Size**: The LRU cache stores transliterated results. Larger cache sizes reduce API calls for repeated text.

5. **Dataset Compatibility**: Works with any dataset containing Hindi text in the specified text column.

## Performance Tips

- Use higher cache sizes for datasets with repeated text
- Adjust rate limits based on your API quota
- Enable fallback for maximum data utilization
- Monitor cache hit rates in the logs for optimization 