#!/usr/bin/env python3
"""
Script to download and store Hindi sentences from the whisper-small-hindi dataset.
Downloads only the 'sentence' column from the train split and saves 500 samples to a txt file.
"""

import os
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_save_hindi_sentences(
    dataset_name: str = "shields/whisper-small-hindi",
    output_file: str = "hindi_sentences.txt",
    num_samples: int = 500,
    split: str = "train"
):
    """
    Download the Hindi dataset and save only the sentence column to a text file.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_file: Output text file path
        num_samples: Number of samples to save
        split: Dataset split to use ('train' or 'test')
    """
    
    try:
        # Load the dataset
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        # Check if we have enough samples
        total_samples = len(dataset)
        logger.info(f"Total samples in {split} split: {total_samples}")
        
        if num_samples > total_samples:
            logger.warning(f"Requested {num_samples} samples but only {total_samples} available. Using all available samples.")
            num_samples = total_samples
        
        # Extract sentences
        logger.info(f"Extracting {num_samples} sentences...")
        sentences = dataset["sentence"][:num_samples]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to text file
        logger.info(f"Saving sentences to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences, 1):
                f.write(f"{sentence}\n")
        
        logger.info(f"Successfully saved {len(sentences)} Hindi sentences to {output_file}")
        
        # Print some sample sentences
        logger.info("Sample sentences:")
        for i, sentence in enumerate(sentences[:5], 1):
            logger.info(f"  {i}: {sentence}")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

def main():
    """Main function to run the script."""
    
    # Configuration
    dataset_name = "shields/whisper-small-hindi"
    output_file = "data/hindi_sentences_500.txt"
    num_samples = 500
    split = "train"
    
    logger.info("Starting Hindi sentence extraction...")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Split: {split}")
    
    # Download and save
    download_and_save_hindi_sentences(
        dataset_name=dataset_name,
        output_file=output_file,
        num_samples=num_samples,
        split=split
    )
    
    logger.info("Script completed successfully!")

if __name__ == "__main__":
    main()
