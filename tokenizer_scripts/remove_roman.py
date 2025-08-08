#!/usr/bin/env python3
"""
Script to remove lines containing Roman/Latin characters from a text file.
Keeps only lines that contain pure Devanagari script (Hindi text).
"""

import re
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def has_roman_characters(text):
    """
    Check if text contains Roman/Latin characters.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text contains Roman characters, False otherwise
    """
    # Pattern to match Roman/Latin letters (A-Z, a-z)
    roman_pattern = r'[A-Za-z]'
    return bool(re.search(roman_pattern, text))

def remove_roman_lines(input_file, output_file):
    """
    Remove lines containing Roman characters from input file and save to output file.
    
    Args:
        input_file (str): Path to input text file
        output_file (str): Path to output text file
    """
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Processing file: {input_file}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        total_lines = 0
        kept_lines = 0
        removed_lines = 0
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Check if line contains Roman characters
                    if has_roman_characters(line):
                        removed_lines += 1
                        if removed_lines <= 5:  # Show first 5 removed lines as examples
                            logger.info(f"Removed line {line_num}: {line[:100]}...")
                    else:
                        # Line contains only non-Roman characters (Devanagari, punctuation, etc.)
                        kept_lines += 1
                        outfile.write(line + '\n')
                        
                        if kept_lines <= 5:  # Show first 5 kept lines as examples
                            logger.info(f"Kept line {line_num}: {line[:100]}...")
                    
                    # Progress update every 10000 lines
                    if total_lines % 10000 == 0:
                        logger.info(f"Processed {total_lines} lines...")
        
        logger.info(f"Processing completed!")
        logger.info(f"Total lines processed: {total_lines}")
        logger.info(f"Lines kept (no Roman characters): {kept_lines}")
        logger.info(f"Lines removed (contains Roman characters): {removed_lines}")
        logger.info(f"Output saved to: {output_file}")
        
        # Calculate percentage
        if total_lines > 0:
            kept_percentage = (kept_lines / total_lines) * 100
            removed_percentage = (removed_lines / total_lines) * 100
            logger.info(f"Kept: {kept_percentage:.1f}% | Removed: {removed_percentage:.1f}%")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

def main():
    """Main function to run the script."""
    
    # Configuration
    input_file = "/home/hrusheekesh.sawarkar/Reverie/chatterbox-finetuning/tokenizer_scripts/data/sentences_97000.txt"
    output_file = "data/sentences_hindi_only_97000.txt"
    
    logger.info("Starting Roman character removal...")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Process the file
    remove_roman_lines(input_file, output_file)
    
    logger.info("Script completed successfully!")

if __name__ == "__main__":
    main()
