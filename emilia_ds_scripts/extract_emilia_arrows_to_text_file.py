#!/usr/bin/env python3
"""Extract text data from Emilia dataset .arrow files for BPE tokenizer training."""

import os
import glob
import argparse
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

def find_emilia_datasets(base_dir: str = "."):
    """Find all Emilia dataset directories and return language -> path mapping."""
    datasets = {}
    
    search_patterns = [
        "**/amphion___emilia-dataset*/default-*/0.0.0/*",
        "**/emilia-dataset*",
        "chatterbox-project/amphion___emilia-dataset*"
    ]
    
    for pattern in search_patterns:
        for path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
            if os.path.isdir(path):
                arrow_files = glob.glob(os.path.join(path, "emilia-dataset-*.arrow"))
                if arrow_files:
                    for arrow_file in arrow_files:
                        basename = os.path.basename(arrow_file)
                        if basename.startswith("emilia-dataset-"):
                            parts = basename.replace("emilia-dataset-", "").split("-")
                            if parts and len(parts[0]) <= 3:
                                lang = parts[0]
                                if lang not in datasets:
                                    datasets[lang] = path
                                    print(f"Found {lang} dataset: {path}")
                                break
    
    return datasets

def extract_text_from_arrow_files(dataset_dir: str, output_file: str, language: str = "unknown"):
    """Extract text from all .arrow files in dataset directory."""
    arrow_pattern = os.path.join(dataset_dir, "emilia-dataset-*.arrow")
    arrow_files = sorted(glob.glob(arrow_pattern))
    
    if not arrow_files:
        print(f"No .arrow files found in {dataset_dir}")
        return 0
    
    print(f"Found {len(arrow_files)} .arrow files for {language.upper()} dataset")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_texts = 0
        
        for arrow_file in tqdm(arrow_files, desc=f"Processing {language.upper()} .arrow files"):
            try:
                dataset = Dataset.from_file(arrow_file)
                
                if arrow_file == arrow_files[0]:
                    print(f"Columns in {language.upper()} dataset: {dataset.column_names}")
                
                for example in dataset:
                    text = None
                    
                    if 'json' in example and isinstance(example['json'], dict):
                        text = example['json'].get('text')
                    elif 'text' in example:
                        text = example['text']
                    
                    if text and text.strip():
                        f.write(text.strip() + '\n')
                        total_texts += 1
                        
            except Exception as e:
                print(f"Error processing {arrow_file}: {e}")
                continue
    
    print(f"Extracted {total_texts} text entries for {language.upper()} to {output_file}")
    return total_texts

def main():
    parser = argparse.ArgumentParser(description="Extract text from Emilia datasets")
    parser.add_argument("--base-dir", default=".", help="Base directory to search for datasets")
    parser.add_argument("--output-dir", default=".", help="Output directory for text files")
    parser.add_argument("--combined", action="store_true", help="Combine all languages into one file")
    parser.add_argument("--languages", nargs="*", help="Specific languages to process (e.g., ja en zh)")
    
    args = parser.parse_args()
    
    print("Searching for Emilia datasets...")
    datasets = find_emilia_datasets(args.base_dir)
    
    if not datasets:
        print("No Emilia datasets found!")
        print("Make sure you're running this script in the correct directory.")
        print("Expected directory structure: .../amphion___emilia-dataset/default-*/0.0.0/*")
        return
    
    if args.languages:
        datasets = {lang: path for lang, path in datasets.items() if lang in args.languages}
        if not datasets:
            print(f"No datasets found for requested languages: {args.languages}")
            return
    
    print(f"Found datasets for languages: {list(datasets.keys())}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    total_extracted = 0
    
    if args.combined:
        output_file = os.path.join(args.output_dir, "emilia_all_languages_text_data.txt")
        print(f"Combining all languages into: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as combined_file:
            for language, dataset_dir in datasets.items():
                print(f"\nProcessing {language.upper()} dataset...")
                
                temp_file = f"temp_{language}_text.txt"
                texts_extracted = extract_text_from_arrow_files(dataset_dir, temp_file, language)
                
                if texts_extracted > 0:
                    with open(temp_file, 'r', encoding='utf-8') as temp:
                        combined_file.write(f"# === {language.upper()} DATASET ===\n")
                        combined_file.write(temp.read())
                        combined_file.write(f"\n# === END {language.upper()} DATASET ===\n\n")
                    
                    os.remove(temp_file)
                    total_extracted += texts_extracted
    else:
        for language, dataset_dir in datasets.items():
            print(f"\nProcessing {language.upper()} dataset...")
            
            output_file = os.path.join(args.output_dir, f"emilia_{language}_text_data.txt")
            print(f"Output file: {output_file}")
            
            texts_extracted = extract_text_from_arrow_files(dataset_dir, output_file, language)
            total_extracted += texts_extracted
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total text entries extracted: {total_extracted}")
    print(f"Languages processed: {list(datasets.keys())}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()