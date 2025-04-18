"""
Salary extraction task.
"""

import pandas as pd
from prompts.templates import get_template

def get_additional_fields():
    """Return list of additional fields to initialize in results."""
    # Include fields to be carried over to the output
    return ["y_true", "nation_short_desc", "salary_additional_text"]

def get_ground_truth_column():
    """Return the column name containing ground truth for evaluation."""
    return "y_true"

def get_additional_columns():
    """Return additional columns to track in results."""
    # Map dataset columns to be included in results
    return {
        "y_true": "y_true", 
        "nation_short_desc": "nation_short_desc",
        "salary_additional_text": "salary_additional_text"
    }

def prepare_job_ad(row):
    """Extract and format job data from a dataset row."""
    job_components = []
    
    # Add job title if available and not NaN
    if "job_title" in row and pd.notna(row["job_title"]):
        job_components.append(f"Job Title: {row['job_title']}")
        
    # Add job ad details if available and not NaN
    if "job_ad_details" in row and pd.notna(row["job_ad_details"]):
        job_components.append(f"Job Details: {row['job_ad_details']}")

    # Add nation_short_desc if available and not NaN
    if "nation_short_desc" in row and pd.notna(row["nation_short_desc"]):
        job_components.append(f"Nation: {row['nation_short_desc']}")

    # Add salary_additional_text if available and not NaN
    if "salary_additional_text" in row and pd.notna(row["salary_additional_text"]):
        job_components.append(f"Salary Additional Text: {row['salary_additional_text']}")

    # Combine all components
    return "\n\n".join(job_components)

def create_prompt(job_ad):
    """
    Create a prompt for salary extraction.
    Args:
        job_ad (str): Job advertisement text
    Returns:
        str: Formatted prompt
    """
    template = get_template("salary")
    return template.format(job_ad=job_ad)

def parse_prediction(prediction):
    """
    Parse the LLM's prediction to extract the salary information.
    Args:
        prediction (str): Raw prediction from the LLM
    Returns:
        str: Parsed salary information
    """
    prediction = prediction.strip()
    
    return prediction

def get_result_columns():
    """
    Return mapping of result column names to source fields.
    Returns:
        dict: {output_column_name: source_field_name}
    """
    # Ensure the tracked columns are mapped to the output
    return {
        "y_true": "y_true",
        "nation_short_desc": "nation_short_desc", 
        "salary_additional_text": "salary_additional_text"
    }
