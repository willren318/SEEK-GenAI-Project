"""
Work arrangement classification task.
"""

from prompts.templates import get_template


def parse_prediction(prediction):
    """
    Parse the LLM's prediction to extract the work arrangement label.
    Args:
        prediction (str): Raw prediction from the LLM
    Return:
        str: Parsed label (Remote, Hybrid, OnSite)
    """
    prediction = prediction.strip()
    valid_labels = ["Remote", "Hybrid", "OnSite"]
    
    # Check if prediction is directly one of the valid labels
    if prediction in valid_labels:
        return prediction
    
    # If no valid label is found, default to onsite (most common work arrangement)
    # return "OnSite"
    return "Unknown"


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