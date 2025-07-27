"""
Main script to run LLM evaluation experiments with MLflow tracking.
"""

import os
import argparse
import mlflow
from models.claude_evaluator import ClaudeEvaluator
from models.llama_evaluator import LlamaEvaluator
from models.deepseek_evaluator import DeepSeekEvaluator
from models.togetherAI_evaluator import TogetherAIEvaluator
from config import TASKS_DATAPATH, MODELS, MLFLOW_CONFIG, MODELS_VARIANTS 

def run_experiment(model_type, task_name, dataset_path, sample_range=None, model_variant=None):
    """
    Run a single experiment with the specified model_type, model_variant and task.
    Examples:
    python run_experiments.py --model claude --task work_arrangement --start_index 0 --end_index 5 --model-variant claude-3-haiku
    python run_experiments.py --model claude --task seniority --start_index 0 --end_index 100 --model-variant claude-3-7-sonnet   
    python run_experiments.py --model togetherAI --task work_arrangement --start_index 0 --end_index 5 --model-variant meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
    Args:
        model_type (str): type of model to use (claude, llama, deepseek, togetherAI, etc.)
        task_name (str): name of the task (work_arrangement, salary, seniority)
        dataset_path (str): path to the dataset
        sample_range (tuple[int, int] | None): Process examples from start index (inclusive) to end index (exclusive). If None, process all.
        model_variant (str, optional): specific model variant to use (eg. claude-3-5-haiku)
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n=== Running experiment with {model_type} {model_variant} on {task_name} task ===")
    # Check if model_type exists in config
    if model_type not in MODELS:
        raise ValueError(f"unsupported model type: {model_type}")

    
    # Get model configuration
    model_config = MODELS[model_type].copy() 
    model_class_name = model_config['class']
    model_name = model_config['name']
    model_params = model_config.get('parameters', {})
    
    # Use specific model variant with releasing date
    if model_type == "claude" and model_variant:
        if model_variant in MODELS_VARIANTS:
            model_variant = MODELS_VARIANTS[model_variant]
            print(f"Using model variant: {model_variant}")
        else:
            print(f"Warning: Unknown model variant '{model_variant}'. Using default model variant.")
    
    # Get the correct evaluator class given model_class_name
    if model_class_name == "ClaudeEvaluator":
        model_class = ClaudeEvaluator
    elif model_class_name == "LlamaEvaluator":
        model_class = LlamaEvaluator
    elif model_class_name == "DeepSeekEvaluator":
        model_class = DeepSeekEvaluator
    elif model_class_name == "TogetherAIEvaluator":
        model_class = TogetherAIEvaluator
    else:
        raise ValueError(f"Model class '{model_class_name}' not supported yet")
    
    # Create evaluator instance
    evaluator = model_class(model_variant=model_variant, task_name=task_name, **model_params)
    
    # Run evaluation and get metrics
    metrics = evaluator.evaluate_dataset(
        dataset_path,
        sample_range=sample_range
    )
    
    # Print summary
    model_display = f"{model_type.upper()} ({model_variant})" if model_variant else model_type.upper()
    print(f"\n--- {model_display} on {task_name} ---")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"Total cost: ${metrics.get('total_cost', 0):.4f}")
    print(f"Average latency: {metrics.get('avg_latency', 0):.4f} seconds")
    print(f"Total tokens: {metrics.get('total_tokens', 0)}")
    print(f"Processed samples: {metrics.get('sample_count', 0)}")
    if 'error_rate' in metrics:
        print(f"Error rate: {metrics.get('error_rate', 0):.2%}")
    
    # Print cache hit information for DeepSeek models
    if model_class_name == "DeepSeekEvaluator" and "cache_hit_tokens" in metrics:
        print(f"Cache hit tokens: {metrics.get('cache_hit_tokens', 0)}")
        print(f"Cache miss tokens: {metrics.get('cache_miss_tokens', 0)}")
        if metrics.get('total_input_tokens', 0) > 0:
            cache_hit_rate = metrics.get('cache_hit_tokens', 0) / metrics.get('total_input_tokens', 0)
            print(f"Cache hit rate: {cache_hit_rate:.2%}")
    
    return metrics

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    
    # Update the choices to include togetherAI
    model_choices = list(MODELS.keys()) + ["togetherAI"] 
    
    parser.add_argument("--model", choices=model_choices, 
                        default="claude", help="Model to evaluate")
    parser.add_argument("--task", choices=list(TASKS_DATAPATH.keys()), 
                        required=True, help="Task to evaluate")
    parser.add_argument("--start_index", type=int, default=None,
                        help="0-based starting index of the sample range (inclusive)")
    parser.add_argument("--end_index", type=int, default=None,
                        help="0-based ending index of the sample range (exclusive)")
    parser.add_argument("--tracking-uri", type=str,
                        default=MLFLOW_CONFIG.get("tracking_uri") if MLFLOW_CONFIG else None,
                        help="MLflow tracking URI (default: from config)")
                        
    # Update model variant choices to include TogetherAI options
    all_model_variants = list(MODELS_VARIANTS.keys()) + [
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3-70B-Instruct"
    ]
    
    parser.add_argument("--model-variant", 
                        help="Specific model variant to use")
    args = parser.parse_args()
    
    # Get task and model from arguments
    model = args.model
    task = args.task
    start_index = args.start_index
    end_index = args.end_index
    model_variant = args.model_variant
    
    # Handle TogetherAI model separately since it's not in MODELS config
    if model == "togetherAI" and model not in MODELS:
        # Add TogetherAI configuration dynamically
        MODELS["togetherAI"] = {
            "class": "TogetherAIEvaluator",
            "name": "TogetherAI LLM",
            "parameters": {}
        }
        
        # Set default model variant if not specified
        if not model_variant:
            model_variant = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            print(f"No model variant specified for TogetherAI. Using default: {model_variant}")
        
    # Get dataset path from config
    dataset_path = TASKS_DATAPATH.get(task)
    if not dataset_path:
        print(f"Error: Task '{task}' not found in tasks configuration.")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at configured path: {dataset_path}.")
        return

    # Determine sample_range based on provided arguments
    sample_range = None
    if start_index is not None and end_index is not None:
        if start_index >= 0 and end_index > start_index:
            sample_range = (start_index, end_index)
            print(f"Processing samples from index {start_index} (inclusive) to {end_index} (exclusive).")
        else:
            print(f"Warning: Invalid range --start_index {start_index} --end_index {end_index}. Must have start >= 0 and end > start. Processing all examples.")
    elif start_index is not None or end_index is not None:
        print("Warning: Both --start_index and --end_index must be provided to specify a range. Processing all examples.")
    else:
        print("Processing all examples.")

    # Run the single experiment
    metrics = run_experiment(
        model, task, dataset_path, sample_range, model_variant
    )

    if metrics:
        print(f"\nExperiment complete. View results in MLflow UI.")
    else:
        print(f"\nExperiment failed or returned no metrics.")

if __name__ == "__main__":
    main() 