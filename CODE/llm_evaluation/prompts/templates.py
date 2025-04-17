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
    "seniority": """
    Classify this job posting into exactly ONE of these seniority categories:
    
    - ENTRY_LEVEL: Entry-level jobs with little to no experience required (e.g., junior, graduate, trainee, apprentice, junior assistant, administrator, entry level, student)
    
    - MID_LEVEL: Mid-level jobs with some experience required (e.g., intermediate, assistant, qualified, mid-level, entry-level to intermediate, standard, level 2)
    
    - EXPERIENCED: Jobs requiring significant experience but not yet senior (e.g., experienced, associate, specialist, coordinator, intermediate to senior, post-doctoral)
    
    - SENIOR: Senior-level jobs with substantial experience required (e.g., senior, advanced, senior associate, mid-senior, senior assistant)
    
    - MANAGEMENT: Management roles with team leadership responsibilities (e.g., manager, supervisor, lead, assistant manager, middle management)
    
    - EXECUTIVE: Executive or high-level leadership positions (e.g., head, principal, executive, director, chief, deputy, board, senior lead, owner)

    Job title is the strongest indicator of seniority level. If the job title contains words matching examples above, prioritize that match.

    Job Advertisement:
    {job_ad}

    Provide your answer as a single word: ENTRY_LEVEL, MID_LEVEL, EXPERIENCED, SENIOR, MANAGEMENT, or EXECUTIVE.
    """
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