#!/usr/bin/env python3
"""
Display current configuration settings for the transliteration pipeline
"""

try:
    from config import *
    config_found = True
except ImportError:
    # Default configuration if config.py is not found
    MAX_WORKERS = 5
    REQUESTS_PER_SECOND = 10.0
    BATCH_SIZE = 50
    CHUNK_SIZE = 1000
    API_TIMEOUT = 30
    SAVE_DIR = "processed_dataset"
    TARGET_REPO_ID = "hrusheekeshsawarkar/whisper_hindi_small_T13N"
    LOG_LEVEL = "INFO"
    PROGRESS_LOG_INTERVAL = 10
    config_found = False

def show_config():
    print("📋 Current Configuration Settings")
    print("=" * 50)
    
    if not config_found:
        print("⚠️  Using default configuration (config.py not found)")
    else:
        print("✅ Using configuration from config.py")
    
    print()
    print("🔧 Performance Settings:")
    print(f"   Max Workers: {MAX_WORKERS}")
    print(f"   Requests per Second: {REQUESTS_PER_SECOND}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    
    print()
    print("⚙️  System Settings:")
    print(f"   API Timeout: {API_TIMEOUT}s")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Progress Log Interval: {PROGRESS_LOG_INTERVAL}")
    
    print()
    print("📁 Dataset Settings:")
    print(f"   Save Directory: {SAVE_DIR}")
    print(f"   Target Repository: {TARGET_REPO_ID}")
    
    print()
    print("🚀 Expected Performance:")
    theoretical_speed = MAX_WORKERS * REQUESTS_PER_SECOND
    print(f"   Theoretical Max Speed: {theoretical_speed:.1f} requests/second")
    
    # Estimate time for full dataset (9434 samples)
    total_samples = 9434
    estimated_time_seconds = total_samples / theoretical_speed
    estimated_time_minutes = estimated_time_seconds / 60
    
    print(f"   Estimated Time for Full Dataset: {estimated_time_minutes:.1f} minutes")
    
    print()
    print("💡 Optimization Tips:")
    if MAX_WORKERS < 10:
        print("   - Consider increasing MAX_WORKERS for faster processing")
    if REQUESTS_PER_SECOND < 20:
        print("   - You might be able to increase REQUESTS_PER_SECOND")
    if BATCH_SIZE < 100:
        print("   - Increasing BATCH_SIZE could improve throughput")

if __name__ == "__main__":
    show_config() 