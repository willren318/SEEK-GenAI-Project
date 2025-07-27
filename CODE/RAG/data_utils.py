import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

def clean_html(text):
    """
    Remove HTML tags from text.
    Args:
        text (str): Text that may contain HTML tags
    Returns:
        str: Clean text without HTML tags
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Use BeautifulSoup for HTML cleaning
    # Reference at https://www.crummy.com/software/BeautifulSoup/bs4/doc/#get-text
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    # Replace multiple spaces with a single space for better processing
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

def load_dataset(file_path):
    """Load a dataset from CSV file"""
    return pd.read_csv(file_path)

def prepare_job_text(row):
    """Combine relevant columns into a single text for embedding"""
    components = []
    
    # Add job title with higher weight by repeating it twice
    if not pd.isna(row.get('job_title', None)):
        components.append(f"Job Title: {row['job_title']} {row['job_title']}")

    # Add classification and subclassification
    if not pd.isna(row.get('classification_name', None)):
        components.append(f"Classification: {row['classification_name']}")
    if not pd.isna(row.get('subclassification_name', None)):
        components.append(f"Subclassification: {row['subclassification_name']}")
   
    # Add job summary
    if not pd.isna(row.get('job_summary', None)):
        components.append(f"Job Summary: {row['job_summary']}")
 
    # Add job ad details
    if not pd.isna(row.get('job_ad_details', None)):
        job_details = row['job_ad_details']
        # Clean HTML tags if present
        if isinstance(job_details, str) and ("<" in job_details and ">" in job_details):
            job_details = clean_html(job_details)
        components.append(f"Job Details: {job_details}")
    
    return "\n".join(components)

def prepare_dataset(df):
    """Create combined text field for each row in the dataset"""
    df['combined_text'] = df.apply(prepare_job_text, axis=1)
    return df
