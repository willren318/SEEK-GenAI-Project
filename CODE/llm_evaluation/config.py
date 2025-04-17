"""
Configuration settings for LLM evaluation.
"""

# Define tasks and their corresponding dataset paths
TASKS_DATAPATH = {    
    "work_arrangement": "../../MISC/job_data_files/work_arrangements_test_set.csv",
    # "work_arrangement": "../../MISC/label_unlabelled_data/lable_work_arrangement.csv",
    "salary": "../../MISC/label_unlabelled_data/lable_work_arrangement.csv", 
    "seniority": "../../MISC/seniority_data_mapped/seniority_labelled_test_set_mapped.csv"
}

# Define  model variants available for testing
MODELS_VARIANTS = {
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    # "llama-3-70b": "llama-3-70b-8192"
}

# Define model configurations
MODELS = {
    "claude": {
        "class": "ClaudeEvaluator",
        "name": "claude-3-7-sonnet-20250219",  # Default model
        "parameters": {}
    },
    "llama": {
        "class": "LlamaEvaluator",
        "name": "llama-3-70b",
        "parameters": {}
    }
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": None,
    "experiment_prefix": "llm_eval"
} 