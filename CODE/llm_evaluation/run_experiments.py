"""
Main script to run LLM evaluation experiments with MLflow tracking.
"""

import os
import argparse
import mlflow
from models import ClaudeEvaluator
from config import TASKS_DATAPATH, MODELS, MLFLOW_CONFIG, MODELS_VARIANTS 

def run_experiment(model_type, task_name, dataset_path, limit=None, model_variant=None):
    """
    Run a single experiment with the specified model_type, model_variant and task.
    Example:
    python run_experiments.py --model claude --task work_arrangement --limit 5 --model-variant claude-3-haiku   
    Args:
        model_type (str): type of model to use (claude, llama, etc.)
        task_name (str): name of the task (work_arrangement, salary, seniority)
        dataset_path (str): path to the dataset
        limit (int, optional): limit the number of examples to process
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
    model_params = model_config['parameters']
    
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
        pass
    else:
        raise ValueError(f"Model class '{model_class_name}' not supported yet")
    
    # Create evaluator instance
    evaluator = model_class(model_variant=model_variant, task_name=task_name, **model_params)
    
    # Run evaluation and get metrics
    metrics = evaluator.evaluate_dataset(dataset_path, limit=limit)
    
    # Print summary
    model_display = f"{model_type.upper()} ({model_variant})" if model_variant else model_type.upper()
    print(f"\n--- {model_display} on {task_name} ---")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Total cost: ${metrics.get('total_cost', 0):.4f}")
    print(f"Average latency: {metrics.get('avg_latency', 0):.4f} seconds")
    print(f"Total tokens: {metrics.get('total_tokens', 0)}")
    
    return metrics

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    parser.add_argument("--model", choices=list(MODELS.keys()), 
                        default="claude", help="Model to evaluate")
    parser.add_argument("--task", choices=list(TASKS_DATAPATH.keys()), 
                        required=True, help="Task to evaluate")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit the number of examples to process")
    parser.add_argument("--tracking-uri", type=str, default=MLFLOW_CONFIG["tracking_uri"],
                        help="MLflow tracking URI (default: from config)")
    parser.add_argument("--model-variant", choices=list(MODELS_VARIANTS.keys()),
                        help="Specific model variant to use")
    args = parser.parse_args()
    
    # Get task and model from arguments
    model = args.model
    task = args.task
    limit = args.limit
    model_variant = args.model_variant
        
    # Get dataset path from config
    dataset_path = TASKS_DATAPATH[task]
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}.")
        return
        
    # Run the single experiment
    metrics = run_experiment(
        model, task, dataset_path, limit, model_variant
    )
    
    print(f"Experiment complete. View results in MLflow UI.")

if __name__ == "__main__":
    main() 