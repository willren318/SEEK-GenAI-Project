#!/usr/bin/env python3
"""
Script to analyze the distribution of seniority levels in the dataset.
Please run this script in the CODE/EDA directory with command `python3 seniority_distribution.py`
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def analyze_seniority_distribution(file_path, output_dir=None):
    """
    Analyze the distribution of seniority levels in the dataset.
    
    Args:
        file_path (str): Path to the seniority dataset
        output_dir (str, optional): Directory to save the visualization
    
    Returns:
        dict: Count of each seniority level
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    # Read the CSV file
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Count the distribution of y_true values
    distribution = df['y_true'].value_counts().to_dict()
    
    # Calculate total count
    total = sum(distribution.values())

    
    # Create a DataFrame for better presentation
    distribution_df = pd.DataFrame({
        'Seniority Level': list(distribution.keys()),
        'Count': list(distribution.values())
    })
    
    # Add percentage column
    distribution_df['Percentage'] = (distribution_df['Count'] / total * 100).round(2).astype(str) + '%'
    
    # Sort by count in descending order
    distribution_df = distribution_df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    
    # Create and save visualization into output_dir
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Sort distribution by count in descending order
        sorted_distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
        
        # Get sorted keys and values (reversed to show highest at top)
        levels = list(sorted_distribution.keys())
        counts = list(sorted_distribution.values())
        
        # Reverse the order so highest appears at the top
        levels.reverse()
        counts.reverse()
        
        # Plot distribution
        plt.figure(figsize=(30, 24))
        plt.barh(levels, counts)
        plt.title('Seniority Level Distribution', fontsize=16)
        plt.ylabel('Seniority Level', fontsize=14)
        plt.xlabel('Count', fontsize=14)
        plt.xlim(0, 1000)  # Set x-axis range to 0-250 to include count and percent label
        
        # Add count and percent labels to bars
        for i, (level, count) in enumerate(zip(levels, counts)):
            percentage = (count / total) * 100
            plt.text(count + 5, i, f"{count} ({percentage:.1f}%)", 
                     va='center', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'seniority_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nDistribution plot saved to: {output_path}")
        plt.close()
    
    # Print distribution to terminal
    print("\nSeniority Level Distribution:")
    print("-" * 60)
    print(f"{'Seniority Level':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    for _, row in distribution_df.iterrows():
        print(f"{row['Seniority Level']:<25} {row['Count']:<10} {row['Percentage']:<10}")
    print("-" * 60)
    print(f"Total: {total} entries")
    
    return distribution

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze seniority distribution in dataset")
    parser.add_argument("--file", type=str, 
                        # default="../../MISC/job_data_files/seniority_labelled_test_set.csv",
                        default="../../MISC/job_data_files/seniority_labelled_development_set.csv",
                        help="Path to the seniority dataset")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                        help="Directory to save visualization output")
    args = parser.parse_args()
    
    distribution = analyze_seniority_distribution(
        args.file, 
        args.output_dir
    )
    
    if distribution:
        print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main() 