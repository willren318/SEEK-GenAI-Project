"""
Seniority classification task.
"""

from prompts.templates import get_template
import pandas as pd


def get_additional_fields():
    """Return list of additional fields to initialize in results."""
    return ["original_labels"]


def get_ground_truth_column():
    """Return the column name containing ground truth for evaluation."""
    return "y_true_mapped"


def get_additional_columns():
    """Return additional columns to track in results."""
    return {"original_labels": "y_true"}  # Read y_true from dataset into original_labels


def prepare_job_ad(row):
    """Extract and format job data from a dataset row."""
    job_components = []
    
    # Add job title if available
    if "job_title" in row and not pd.isna(row["job_title"]):
        job_components.append(f"Job Title: {row['job_title']}")
    
    # Add job summary if available
    if "job_summary" in row and not pd.isna(row["job_summary"]):
        job_components.append(f"Job Summary: {row['job_summary']}")
    
    # Add classification name if available
    if "classification_name" in row and not pd.isna(row["classification_name"]):
        job_components.append(f"Classification: {row['classification_name']}")
    
    # Add job ad details if available
    if "job_ad_details" in row and not pd.isna(row["job_ad_details"]):
        job_components.append(f"Job Details: {row['job_ad_details']}")
    elif "job_ad" in row and not pd.isna(row["job_ad"]):
        job_components.append(f"Job Details: {row['job_ad']}")
    
    # Combine all components
    return "\n\n".join(job_components) 


def create_prompt(job_ad):
    """
    Create a prompt for seniority classification.
    Args:
        job_ad (str): Job advertisement text
    Returns:
        str: Formatted prompt
    """
    template = get_template("seniority")
    return template.format(job_ad=job_ad) 


def parse_prediction(prediction):
    """
    Parse the LLM's prediction to extract the seniority label.
    Args:
        prediction (str): Raw prediction from the LLM
    Returns:
        str: Parsed label (entry, mid, senior, executive)
    """
    prediction = prediction.strip()
    return prediction


def get_result_columns():
    """
    Return mapping of result column names to source fields.
    Returns:
        dict: {output_column_name: source_field_name}
    """
    return {"original_labels": "original_labels"}  # Include original_labels as is 