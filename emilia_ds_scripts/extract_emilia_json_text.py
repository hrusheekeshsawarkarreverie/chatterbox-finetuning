#!/usr/bin/env python3
"""Extract text data from Emilia YODAS extracted JSON files for tokenizer training."""

import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def find_extracted_json_files(base_dir: str = "chatterbox-project/Emilia-YODAS"):
    """Find all JSON files in extracted directories and return language -> files mapping."""
    datasets = {}
    
    for lang_dir in ["DE", "FR"]:
        lang_path = os.path.join(base_dir, lang_dir, f"{lang_dir}")
        
        if not os.path.exists(lang_path):
            print(f"Warning: {lang_path} not found, skipping {lang_dir}")
            continue
            
        json_files = []
        
        for batch_dir in os.listdir(lang_path):
            batch_path = os.path.join(lang_path, batch_dir)
            if os.path.isdir(batch_path):
                batch_json_files = glob.glob(os.path.join(batch_path, "*.json"))
                json_files.extend(batch_json_files)
        
        if json_files:
            datasets[lang_dir.lower()] = sorted(json_files)
            print(f"Found {len(json_files)} JSON files for {lang_dir}")
    
    return datasets


def process_json_file(json_file):
    """Extract text from a single JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = data.get('text', '').strip()
        if text:
            return text
        return None
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None


def extract_text_from_json_files(json_files: list, output_file: str, language: str = "unknown", use_multiprocessing: bool = True):
    """Extract text from all JSON files."""
    print(f"Processing {len(json_files)} JSON files for {language.upper()}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_texts = 0
        
        if use_multiprocessing and len(json_files) > 100:
            num_processes = min(cpu_count(), 8)
            print(f"Using {num_processes} processes for parallel processing")
            
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_json_file, json_files, chunksize=50),
                    total=len(json_files),
                    desc=f"Processing {language.upper()} JSON files"
                ))
            
            for text in results:
                if text:
                    f.write(text + '\n')
                    total_texts += 1
        else:
            for json_file in tqdm(json_files, desc=f"Processing {language.upper()} JSON files"):
                text = process_json_file(json_file)
                if text:
                    f.write(text + '\n')
                    total_texts += 1
    
    print(f"Extracted {total_texts} text entries for {language.upper()} to {output_file}")
    return total_texts


def get_dataset_stats(datasets):
    """Get statistics about the datasets."""
    stats = {}
    
    for language, json_files in datasets.items():
        stats[language] = {
            "total_files": len(json_files),
            "sample_files": json_files[:3] if json_files else [],
            "directories": set()
        }
        
        for json_file in json_files:
            batch_dir = os.path.basename(os.path.dirname(json_file))
            stats[language]["directories"].add(batch_dir)
        
        stats[language]["batch_count"] = len(stats[language]["directories"])
        stats[language]["directories"] = sorted(list(stats[language]["directories"]))
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract text from Emilia YODAS extracted JSON files")
    parser.add_argument("--base-dir", default="chatterbox-project/Emilia-YODAS", 
                       help="Base directory containing Emilia-YODAS data")
    parser.add_argument("--output-dir", default=".", help="Output directory for text files")
    parser.add_argument("--combined", action="store_true", help="Combine all languages into one file")
    parser.add_argument("--languages", nargs="*", choices=["de", "fr"], 
                       help="Specific languages to process (de, fr)")
    parser.add_argument("--no-multiprocessing", action="store_true", 
                       help="Disable multiprocessing for debugging")
    parser.add_argument("--stats-only", action="store_true", help="Show statistics only, don't extract")
    
    args = parser.parse_args()
    
    print("Searching for extracted JSON files...")
    datasets = find_extracted_json_files(args.base_dir)
    
    if not datasets:
        print("No extracted JSON files found!")
        print(f"Make sure the extraction has been completed in: {args.base_dir}")
        print("Expected structure: .../DE/extracted/DE-B000XXX/*.json")
        return
    
    if args.languages:
        datasets = {lang: files for lang, files in datasets.items() if lang in args.languages}
        if not datasets:
            print(f"No datasets found for requested languages: {args.languages}")
            return
    
    stats = get_dataset_stats(datasets)
    
    print(f"\n=== DATASET STATISTICS ===")
    for language, stat in stats.items():
        print(f"{language.upper()}:")
        print(f"  - Total JSON files: {stat['total_files']:,}")
        print(f"  - Extracted batches: {stat['batch_count']}")
        print(f"  - Batch directories: {', '.join(stat['directories'][:5])}")
        if len(stat['directories']) > 5:
            print(f"    ... and {len(stat['directories']) - 5} more")
    
    if args.stats_only:
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    total_extracted = 0
    use_mp = not args.no_multiprocessing
    
    if args.combined:
        output_file = os.path.join(args.output_dir, "emilia_yodas_all_languages_text.txt")
        print(f"\nCombining all languages into: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as combined_file:
            for language, json_files in datasets.items():
                print(f"\nProcessing {language.upper()} dataset...")
                
                temp_file = f"temp_{language}_yodas_text.txt"
                texts_extracted = extract_text_from_json_files(
                    json_files, temp_file, language, use_mp
                )
                
                if texts_extracted > 0:
                    with open(temp_file, 'r', encoding='utf-8') as temp:
                        combined_file.write(temp.read())
                    
                    os.remove(temp_file)
                    total_extracted += texts_extracted
    else:
        for language, json_files in datasets.items():
            print(f"\nProcessing {language.upper()} dataset...")
            
            output_file = os.path.join(args.output_dir, f"emilia_yodas_{language}_text.txt")
            print(f"Output file: {output_file}")
            
            texts_extracted = extract_text_from_json_files(
                json_files, output_file, language, use_mp
            )
            total_extracted += texts_extracted
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total text entries extracted: {total_extracted:,}")
    print(f"Languages processed: {list(datasets.keys())}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()