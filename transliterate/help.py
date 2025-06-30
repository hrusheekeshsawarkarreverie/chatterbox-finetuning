#!/usr/bin/env python3
"""
Help and command overview for the Hindi transliteration pipeline
"""

def show_help():
    print("🚀 Hindi to English Transliteration Pipeline")
    print("=" * 60)
    print()
    
    print("📝 AVAILABLE COMMANDS:")
    print("-" * 30)
    
    print("🧪 Testing & Diagnostics:")
    print("   python transliterate_hi_en.py --test     Test API with sample sentences")
    print("   python config_info.py                   Show current configuration")
    print("   python memory_monitor.py                 Check memory usage")
    print("   python check_dataset.py                  Check saved dataset status")
    print()
    
    print("⚡ Processing Commands:")
    print("   python transliterate_hi_en.py --process-only    Process & save locally only")
    print("   python transliterate_hi_en.py --upload          Upload saved dataset")
    print("   python transliterate_hi_en.py                   Full pipeline (process + upload)")
    print()
    
    print("🎛️  CONFIGURATION:")
    print("-" * 20)
    print("Edit config.py to customize:")
    print("   MAX_WORKERS = 8         # Concurrent threads")
    print("   REQUESTS_PER_SECOND = 20 # API rate limit")
    print("   BATCH_SIZE = 50         # Texts per batch")
    print("   CHUNK_SIZE = 500        # Memory chunk size")
    print()
    
    print("🔧 RECOMMENDED WORKFLOW:")
    print("-" * 25)
    print("1. python transliterate_hi_en.py --test              # Test API")
    print("2. python memory_monitor.py                          # Check memory")
    print("3. python config_info.py                             # Review settings")
    print("4. python transliterate_hi_en.py --process-only      # Process dataset")
    print("5. python check_dataset.py                           # Verify results")
    print("6. python transliterate_hi_en.py --upload            # Upload to HF")
    print()
    
    print("⚠️  MEMORY MANAGEMENT:")
    print("-" * 22)
    print("If process gets killed (memory issues):")
    print("   • Reduce CHUNK_SIZE (try 250 or 100)")
    print("   • Reduce MAX_WORKERS (try 4-6)")
    print("   • Reduce BATCH_SIZE (try 25)")
    print("   • Monitor with: python memory_monitor.py")
    print()
    
    print("🚀 PERFORMANCE TIPS:")
    print("-" * 19)
    print("   • Start with default settings")
    print("   • Monitor memory during processing")
    print("   • Increase settings gradually if stable")
    print("   • Expected processing time: 1-2 minutes")
    print("   • Theoretical speed: 160 requests/second")
    print()
    
    print("📊 EXPECTED RESULTS:")
    print("-" * 19)
    print("   • Dataset: ~9,400 samples")
    print("   • Splits: train (~6,540) + test (~2,890)")
    print("   • Columns: audio, sentence_hindi, sentence_english")
    print("   • Memory usage: Stays constant (~constant MB)")
    print()
    
    print("💡 For detailed help, see README.md")

if __name__ == "__main__":
    show_help() 