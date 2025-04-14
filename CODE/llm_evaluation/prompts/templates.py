"""
This file contains prompt templates for various LLM tasks related to job analysis.
"""

# Templates for different tasks
TEMPLATES = {
    # Work Arrangement classification
    "work_arrangement": """
    You are analyzing a job advertisement to determine its work arrangement type.
    Classify this job posting into exactly one of these categories:
    - Remote: Fully remote work with no requirement to go to a physical office
    - Hybrid: A mix of remote work and in-office(onsite) work
    - OnSite: Work that must be done at a physical location

    Job Advertisement:
    {job_ad}

    Provide your answer as a single word: Remote, Hybrid, or OnSite.
    """,
    
    # Salary extraction

    
    # Seniority classification
 


}

def get_template(task_name):
    """
    Get prompt template for a specific task.
    
    Args:
        task_name (str): Name of the task (work_arrangement, salary, seniority)
        
    Returns:
        str: Prompt template
    """
    return TEMPLATES.get(task_name, TEMPLATES["work_arrangement"]) 