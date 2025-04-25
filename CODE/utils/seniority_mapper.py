"""
Utility to map raw seniority levels to standardized categories.
"""

import json
import os
import pandas as pd
from fuzzywuzzy import process
import argparse

def load_seniority_mapping():
    """Load seniority mapping from JSON file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(current_dir, "seniority_mapping_new.json")
    with open(mapping_path, "r") as f:
        return json.load(f)

def clean_text(text):
    """Clean text by removing extra whitespace and special characters"""
    if not isinstance(text, str):
        return ""
    return ' '.join(text.strip().split())

def categorize_seniority(text, mapping=None):
    """Categorize raw seniority text into standardized categories using fuzzy matching"""
    if mapping is None:
        mapping = load_seniority_mapping()
    
    text = clean_text(text).lower()
    
    # First try exact matching
    for category, variations in mapping.items():
        for variation in variations:
            if variation.lower() in text:
                return category
    
    # If no exact match, use fuzzy matching
    # Create a list of all variations with their categories
    all_variations = []
    category_lookup = {}
    
    for category, variations in mapping.items():
        for variation in variations:
            all_variations.append(variation.lower())
            category_lookup[variation.lower()] = category
    
    # Find the best match on the list of all variations, always using the highest score
    best_match, _ = process.extractOne(text, all_variations)
    return category_lookup[best_match]


def add_mapped_seniority(df, input_col="y_true", output_col="y_true_mapped"):
    """
    Add a new column to DataFrame with mapped seniority values
    Args:
        df (pandas.DataFrame): DataFrame containing seniority data
        input_col (str): Name of column containing raw seniority values
        output_col (str): Name of column to create with mapped values
    Returns:
        pandas.DataFrame: DataFrame with new mapped column
    """
    mapping = load_seniority_mapping()
    df[output_col] = df[input_col].apply(lambda x: categorize_seniority(x, mapping))
    return df


def main():
    """Process a CSV file and add mapped seniority column"""
    parser = argparse.ArgumentParser(description="Map seniority values to standardized categories")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-file", help="Path to output CSV file (default: adds '_mapped' to input filename)")
    parser.add_argument("--input-col", default="y_true", help="Name of input column (default: 'y_true')")
    parser.add_argument("--output-col", default="y_true_mapped", help="Name of output column (default: 'y_true_mapped')")
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_mapped{ext}"
    
    # Read CSV, map seniority, and write result
    print(f"Reading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    print(f"Mapping seniority from column '{args.input_col}' to '{args.output_col}'...")
    df = add_mapped_seniority(df, args.input_col, args.output_col)
    
    print(f"Writing mapped data to {args.output_file}...")
    df.to_csv(args.output_file, index=False)
    
    print("Done!")


if __name__ == "__main__":
    main()