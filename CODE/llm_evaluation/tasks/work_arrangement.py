"""
Work arrangement classification task.
"""

from prompts.templates import get_template


def get_additional_fields():
    """Return list of additional fields to initialize in results."""
    return []  # No additional fields needed


def get_ground_truth_column():
    """Return the column name containing ground truth for evaluation."""
    return "y_true"


def get_additional_columns():
    """Return additional columns to track in results."""
    return {}  # No additional columns needed


def prepare_job_ad(row):
    """Extract and format job data from a dataset row."""
    # Use either job_ad or job_ad_details column
    job_ad_col = "job_ad" if "job_ad" in row else "job_ad_details"
    return row[job_ad_col] if job_ad_col in row else ""


def create_prompt(job_ad):
    """
    Create a prompt for work arrangement classification.
    Args:
        job_ad (str): Job advertisement text
    Returns:
        str: Formatted prompt
    """
    template = get_template("work_arrangement")
    return template.format(job_ad=job_ad) # format the prompt with the job ad


def parse_prediction(prediction):
    """
    Parse the LLM's prediction to extract the work arrangement label.
    Args:
        prediction (str): Raw prediction from the LLM
    Return:
        str: Parsed label (Remote, Hybrid, OnSite)
    """
    prediction = prediction.strip()
    # valid_labels = ["Remote", "Hybrid", "OnSite"]
    
    # Check if prediction is directly one of the valid labels
    # if prediction in valid_labels:
    #     return prediction
    
    # If no valid label is found, default to onsite (most common work arrangement)
    # return "OnSite"
    # return "Unknown"
    return prediction


def get_result_columns():
    """
    Return mapping of result column names to source fields.
    Returns:
        dict: {output_column_name: source_field_name}
    """
    return {}  # No additional columns needed