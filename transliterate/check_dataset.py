#!/usr/bin/env python3
"""
Utility script to check the status of locally saved datasets
"""

import os
import json
from datasets import DatasetDict

def check_dataset_status(save_dir: str = "processed_dataset"):
    """
    Check the status of a locally saved dataset
    """
    print(f"Checking dataset in: {save_dir}")
    print("=" * 50)
    
    if not os.path.exists(save_dir):
        print("❌ No saved dataset found")
        return False
    
    try:
        # Check if it's a valid dataset
        dataset_dict = DatasetDict.load_from_disk(save_dir)
        
        print("✅ Valid dataset found!")
        print(f"Splits: {list(dataset_dict.keys())}")
        
        for split_name, split_data in dataset_dict.items():
            print(f"  - {split_name}: {len(split_data)} samples")
            print(f"    Columns: {split_data.column_names}")
        
        # Check metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            print(f"\n📊 Metadata:")
            print(f"  - Total samples: {metadata.get('total_samples', 'Unknown')}")
            print(f"  - Saved at: {metadata.get('saved_at', 'Unknown')}")
            print(f"  - Columns: {metadata.get('columns', 'Unknown')}")
        
        # Check a sample
        first_split = list(dataset_dict.keys())[0]
        sample = dataset_dict[first_split][0]
        
        print(f"\n🔍 Sample from {first_split}:")
        if 'sentence_hindi' in sample:
            print(f"  Hindi: {sample['sentence_hindi']}")
        if 'sentence_english' in sample:
            print(f"  English: {sample['sentence_english']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    save_dir = "processed_dataset"
    if len(sys.argv) > 1:
        save_dir = sys.argv[1]
    
    check_dataset_status(save_dir) 