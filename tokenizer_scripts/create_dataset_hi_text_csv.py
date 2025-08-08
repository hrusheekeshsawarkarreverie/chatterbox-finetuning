#!/usr/bin/env python3
"""
Script to extract text sentences from a local CSV file.
Loads text from the 'text' column and saves samples to a txt file.
"""

import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_save_sentences_from_csv(
    csv_file_path: str,
    output_file: str = "sentences.txt",
    num_samples: int = 500,
    text_column: str = "text"
):
    """
    Load a CSV file and save only the text column to a text file.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file: Output text file path
        num_samples: Number of samples to save (if None, save all)
        text_column: Name of the column containing text
    """
    
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Load the CSV file
        logger.info(f"Loading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Check if text column exists
        if text_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(f"Column '{text_column}' not found. Available columns: {available_columns}")
        
        # Remove any rows with missing text
        df = df.dropna(subset=[text_column])
        
        # Check if we have enough samples
        total_samples = len(df)
        logger.info(f"Total samples with valid text: {total_samples}")
        
        if num_samples and num_samples > total_samples:
            logger.warning(f"Requested {num_samples} samples but only {total_samples} available. Using all available samples.")
            num_samples = total_samples
        
        # Extract sentences
        if num_samples:
            logger.info(f"Extracting {num_samples} sentences...")
            sentences = df[text_column].head(num_samples).tolist()
        else:
            logger.info("Extracting all sentences...")
            sentences = df[text_column].tolist()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to text file
        logger.info(f"Saving sentences to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                # Clean the sentence (remove extra whitespace and newlines)
                cleaned_sentence = ' '.join(str(sentence).split())
                f.write(f"{cleaned_sentence}\n")
        
        logger.info(f"Successfully saved {len(sentences)} sentences to {output_file}")
        
        # Print some sample sentences
        logger.info("Sample sentences:")
        for i, sentence in enumerate(sentences[:5], 1):
            cleaned_sentence = ' '.join(str(sentence).split())
            logger.info(f"  {i}: {cleaned_sentence}")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

def main():
    """Main function to run the script."""
    
    # Configuration
    csv_file_path = "/home/hrusheekesh.sawarkar/Reverie/tts_data_collector/cartesia_audio_data/audio_text_mapping_81717_cleaned_final_no_digits_20250805_162548_no_digits_20250806_104158.csv"
    output_file = "data/sentences_97000.txt"
    num_samples = 97000
    text_column = "text"
    
    logger.info("Starting sentence extraction from CSV...")
    logger.info(f"CSV file: {csv_file_path}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Text column: {text_column}")
    
    # Load and save
    load_and_save_sentences_from_csv(
        csv_file_path=csv_file_path,
        output_file=output_file,
        num_samples=num_samples,
        text_column=text_column
    )
    
    logger.info("Script completed successfully!")

if __name__ == "__main__":
    main()
