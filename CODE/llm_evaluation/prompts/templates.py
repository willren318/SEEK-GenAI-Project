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
    "salary": """Analyze the job advertisement text that will be provided at the end of these instructions and extract the salary information.

    Instructions:
    1. **Prioritize Salary Source:** First, look for salary information within the 'Salary Additional Text:' section of the Job Advertisement, if present. Use this information and ignore the 'Job Details:' and 'Job Title:' sections. If no salary details are found in the 'Salary Additional Text:' section, then analyze the 'Job Details:' and 'Job Title:' sections.
    2. **Extract Components:** Attempt to extract the minimum salary, maximum salary, currency, and pay period (e.g., HOURLY, MONTHLY, ANNUAL, WEEKLY).
    3. **Determine Currency from Nation:** Derive the 3-letter currency code DIRECTLY from the 'Nation:' section of the Job Advertisement. Use standard mappings (e.g., AU -> AUD, PH -> PHP, SG -> SGD, NZ -> NZD, MY -> MYR, HK -> HKD). 
    4. **Handle PHP Nation Exception:** If the 'Nation:' is 'PH', check the text within 'Job Details:'.
       - Look for the exact text pattern "Compensation" followed by a numerical value on the same or next line (e.g., "Compensation 16000" or "Compensation\n16000").
       - If you find this "Compensation" pattern with a number, use this numerical value *only*. Round it to the nearest integer and use it for both min_salary and max_salary.
       - In this case (Nation: PH and the "Compensation" pattern found), explicitly IGNORE any separate text like "Compensation Range" followed by a range (e.g., "Compensation Range ₱15,000 - ₱20,000").
       - If only a "Compensation Range" pattern is found (and no preceding "Compensation" pattern with a single number), then use the range provided after "Compensation Range".
    5. **Rounding Only Decimal Salary Values:** round any decimal salary values to the nearest integer.
    6. **Handle Tenure-Based Rates and Bonuses:** If multiple pay rates are listed based on duration of employment, use ONLY the starting rate for BOTH min_salary and max_salary. Explicitly EXCLUDE any separate bonuses or one-time payments from the salary range.
    7. **Formatting:** Format the output strictly as: min_salary-max_salary-currency-period
       - Use INTEGERS for min_salary and max_salary resulting from the rounding rule above.
       - Use standard 3-letter currency codes (e.g., MYR, HKD, AUD, NZD, SGD, PHP) as determined by rule 3.
       - Use standard period names: ONLY HOURLY, MONTHLY, ANNUAL, or WEEKLY are valid.
    8. **Single Value:** If only a single salary amount is mentioned (and the PHP rule didn't use a specific "Compensation" value), use its rounded integer value for both min_salary and max_salary.
    9. **Range:** If a salary range is specified (and the PHP rule didn't apply or used the "Compensation Range"), use the rounded integer values for min_salary and max_salary.
    10. **Strict Completeness Check:** AFTER applying rules 1-8, you MUST verify that you have determined ALL THREE parts: (A) a valid non-zero salary integer or range, (B) a valid 3-letter currency code, AND (C) a valid period that is *exactly* one of HOURLY, MONTHLY, ANNUAL, or WEEKLY. 
       - If component (A), (B), OR (C) is missing, cannot be determined, or is not valid according to the allowed values (e.g., the period is missing or something other than the 4 allowed words), then the entire extraction is invalid. 
       - In case of ANY invalidity, you MUST output exactly: 0-0-None-None. 
       - CRITICAL: Do NOT output partial results. For example, if you find USD and 20 but cannot determine a valid period from the allowed list, output 0-0-None-None, NOT 20-20-USD-None.

    CRITICAL FORMATTING REQUIREMENT:
    You MUST output ONLY the formatted salary string (e.g., 37-84-AUD-HOURLY or 16000-16000-PHP-MONTHLY or 0-0-None-None).
    DO NOT include ANY other text, reasoning, explanations, or thinking steps.
    Your entire response must consist of ONLY the formatted string.

    Job Advertisement:
    {job_ad}
    """,
    
    # Seniority classification
    "seniority": """
    Your task is to classify the job posting into exactly ONE of the following seniority categories. Follow the rules strictly.

    **Categories and Examples:**
    - ENTRY_LEVEL: Little to no experience required (e.g., junior, graduate, trainee, apprentice, junior assistant, administrator, entry level, student, receptionist)
    - MID_LEVEL: Some experience required, performs standard tasks (e.g., intermediate, assistant, qualified, mid-level, standard, level 2)
    - EXPERIENCED: Significant independent experience, often specialized (e.g., experienced, associate, specialist, coordinator, post-doctoral) - *Requires more than just standard tasks.* 
    - SENIOR: Substantial experience, high expertise, may mentor but not primarily manage (e.g., senior, advanced, senior associate, mid-senior, senior assistant)
    - MANAGEMENT: Explicit people/team management responsibilities (e.g., manager, supervisor, lead, assistant manager, middle management)
    - EXECUTIVE: High-level strategic leadership (e.g., head, principal, executive, director, chief, deputy, board, senior lead, owner)

    **Rules (Apply in Order):**
    1.  **Prioritize Job Title:** If the Job Title contains a keyword EXACTLY matching an example above (case-insensitive), assign that category immediately. This is the most important rule.
        *   Example: If title contains "Intermediate", classify as MID_LEVEL, even if description mentions many years experience.
        *   Example: If title contains "Assistant", classify as MID_LEVEL, unless combined with clear entry terms like "Receptionist".
    2.  **Explicit Level Mention:** If no title match, look for explicit phrases like "entry-level role", "senior position", "mid-level opportunity" in the description.
    3.  **Distinguish MANAGEMENT:** Assign MANAGEMENT *only* if the title clearly indicates management (Manager, Supervisor, Lead) OR the description explicitly mentions direct responsibility for managing/supervising people/teams. Do NOT infer management from words like "coordinate" or "lead" in a project sense.
    4.  **Distinguish EXPERIENCED vs MID_LEVEL:** Assign EXPERIENCED *only* if the role requires significant independent work, specialization, or >5 years experience *and* the title doesn't indicate MID_LEVEL.
    5.  **Years Experience (Guideline Only):** Use as a *fallback* guideline if other rules don't apply: 0-2 yrs (ENTRY), 2-5 yrs (MID), 5-8 yrs (EXP), 8+ yrs (SENIOR+).

    **Job Advertisement:**
    {job_ad}

    CRITICAL FORMATTING REQUIREMENT: 
    You MUST respond with ONLY ONE of these six category names: ENTRY_LEVEL, MID_LEVEL, EXPERIENCED, SENIOR, MANAGEMENT, or EXECUTIVE.
    Your entire response must be ONLY the category name with NO other text, explanation, periods, or any additional content.
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