#!/usr/bin/env python3
"""
Configuration settings for the Hindi transliteration pipeline
"""

# Multiprocessing configuration
MAX_WORKERS = 10  # Number of concurrent threads for API calls (reduced for stability)
REQUESTS_PER_SECOND = 50.0  # Rate limiting: requests per second across all workers
BATCH_SIZE = 100  # Number of texts to process in each batch (reduced for memory)
CHUNK_SIZE = 1000  # Number of samples to process in each memory chunk (reduced for memory)

# API configuration
API_TIMEOUT = 30  # Timeout for each API request in seconds

# Dataset configuration
SAVE_DIR = "processed_dataset"  # Directory to save processed dataset locally
TARGET_REPO_ID = "hrusheekeshsawarkar/whisper_hindi_small_T13N"  # Hugging Face repo

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
PROGRESS_LOG_INTERVAL = 10  # Log progress every N completions within a batch

# Performance tuning notes:
# - Increase MAX_WORKERS for faster processing (but watch API rate limits)
# - Decrease REQUESTS_PER_SECOND if you hit API rate limits
# - Increase BATCH_SIZE for better throughput (but uses more memory)
# - Decrease CHUNK_SIZE if you encounter memory issues 