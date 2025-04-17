"""
Base class for all LLM evaluators with MLflow integration.
"""

import os
import time
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import re
from bs4 import BeautifulSoup
import datetime
import importlib

class LLMEvaluator:
    """Base class for all LLM evaluators with MLflow integration."""
    
    def __init__(self, model_variant, task_name):
        """
        Initialize the evaluator.
        Args:
            model_variant (str): Name of the LLM model variant
            task_name (str): Name of the task (work_arrangement, salary, seniority)
        """
        self.model_variant = model_variant
        self.task_name = task_name
        
        # Dynamically import the task module from tasks folder
        try:
            self.task_module = importlib.import_module(f"tasks.{task_name}")
        except ImportError:
            print(f"Warning: Task module for '{task_name}' not found")
            self.task_module = None
        
        # Initialize results dictionary to store results for each example
        self.results = {
            "predictions": [],
            "ground_truth": [],
            "latencies": [],
            "input_tokens": [],
            "output_tokens": [],
            "total_tokens": [],
            "costs": [],
            "job_ids": []  # Store job IDs for reference
        }
        
        # Initialize task-specific result fields
        if self.task_module and hasattr(self.task_module, 'get_additional_fields'):
            for field in self.task_module.get_additional_fields():
                self.results[field] = []
    
    def call_api(self, prompt):
        """Call the LLM API with the given prompt.
        This method should be implemented by subclasses given the LLM model variant.
        Args:
            prompt (str): The prompt to send to the LLM
        
        Returns:
            dict: Results including prediction, latency, and token counts
        """
        raise NotImplementedError("Subclasses must implement call_api")
    
    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate the cost based on token usage.
        This method should be implemented by subclasses given the LLM model variant.
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
        Returns:
            float: Calculated cost
        """
        raise NotImplementedError("Subclasses must implement calculate_cost")
    
    def clean_html(self, text):
        """Remove HTML tags from text.
        Args:
            text (str): Text that may contain HTML tags
        Returns:
            str: Clean text without HTML tags
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Use BeautifulSoup for HTML cleaning
        # Reference at https://www.crummy.com/software/BeautifulSoup/bs4/doc/#get-text
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
        
        # Replace multiple spaces with a single space for LLM better processing
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def create_prompt(self, job_ad):
        """Create a prompt based on task and job ad
        Args:
            job_ad (str): The job ad text
        Returns:
            str: Formatted prompt
        """
        # Use task-specific prompt creation if available
        if self.task_module and hasattr(self.task_module, 'create_prompt'):
            return self.task_module.create_prompt(job_ad)
        
    def parse_prediction(self, prediction):
        """Parse the prediction using the task-specific parser.
        Args:
            prediction (str): Raw prediction from the LLM
        Returns:
            str: Parsed prediction
        """
        if self.task_module and hasattr(self.task_module, 'parse_prediction'):
            return self.task_module.parse_prediction(prediction)
        return prediction.strip()
    
    def evaluate_dataset(self, dataset_path, mlflow_tracking=True, limit=None):
        """Evaluate the model on a dataset and track with MLflow.
        Args:
            dataset_path (str): Path to the CSV dataset
            mlflow_tracking (bool): Whether to track results with MLflow
            limit (int, optional): Limit the number of examples to process
        Returns:
            dict: Evaluation metrics
        """
        # End the previous run just in case
        mlflow.end_run()

        start_time = time.time()
        
        # Set up MLflow tracking
        if mlflow_tracking:
            mlflow.set_experiment(f"{self.task_name}_evaluation") # set experiment name
        

        with mlflow.start_run(run_name=f"{self.model_variant}_{self.task_name}"):
            if mlflow_tracking:
                # Log parameters
                mlflow.log_param("model_type", self.model_variant.split("-")[0])
                mlflow.log_param("model_variant", self.model_variant)
                mlflow.log_param("task_name", self.task_name)
                mlflow.log_param("dataset", os.path.basename(dataset_path))
                mlflow.log_param("evaluation_date", datetime.datetime.now().strftime("%Y-%m-%d"))
                mlflow.log_param("evaluation_time", datetime.datetime.now().strftime("%H:%M:%S"))
                
                if limit:
                    mlflow.log_param("sample_limit", limit)
                
                # Set tags
                mlflow.set_tag("model_family", self.model_variant.split("-")[0])
                mlflow.set_tag("task_type", self.task_name)
                mlflow.set_tag("model_variant", self.model_variant)

            
            # Read dataset
            df = pd.read_csv(dataset_path)
            
            # Limit examples if specified
            if limit is not None and limit > 0 and limit < len(df):
                df = df.head(limit)
                
            print(f"Processing {len(df)} examples...")
            
            # Determine column names based on task
            job_id_col = "id" if "id" in df.columns else "job_id"
            
            # Get ground truth column from task module
            truth_col = "y_true"  # Default
            if self.task_module and hasattr(self.task_module, 'get_ground_truth_column'):
                truth_col = self.task_module.get_ground_truth_column()
            
            # Get additional columns to track
            additional_columns = {}
            if self.task_module and hasattr(self.task_module, 'get_additional_columns'):
                additional_columns = self.task_module.get_additional_columns()
            
            # Process each example
            for idx, row in df.iterrows():
                try:
                    # Extract job id
                    job_id = str(row[job_id_col])
                    
                    # Extract job ad using task-specific method if available
                    if self.task_module and hasattr(self.task_module, 'prepare_job_ad'):
                        job_ad = self.task_module.prepare_job_ad(row)
                    else:
                        # Default fallback
                        job_ad_col = "job_ad" if "job_ad" in row else "job_ad_details"
                        job_ad = row[job_ad_col] if job_ad_col in row else ""
                    
                    # Clean job ad if it contains HTML
                    if "<" in str(job_ad) and ">" in str(job_ad):
                        job_ad = self.clean_html(str(job_ad))
                    
                    # Get ground truth
                    ground_truth = str(row[truth_col]) if truth_col in row else "unknown"
                    
                    # Track additional columns
                    for result_field, source_column in additional_columns.items():
                        if source_column in row:
                            self.results[result_field].append(str(row[source_column]))
                    
                    # Create prompt
                    prompt = self.create_prompt(job_ad)
                    
                    # Call API
                    result = self.call_api(prompt)
                    
                    # Parse prediction using task-specific parser
                    raw_prediction = result["prediction"]
                    prediction = self.parse_prediction(raw_prediction)
                    
                    # Store results
                    self.results["predictions"].append(prediction)
                    self.results["ground_truth"].append(ground_truth)
                    self.results["latencies"].append(result["latency"])
                    self.results["input_tokens"].append(result["input_tokens"])
                    self.results["output_tokens"].append(result["output_tokens"])
                    self.results["total_tokens"].append(result["total_tokens"])
                    self.results["costs"].append(result["cost"])
                    self.results["job_ids"].append(job_id)
                    
                    # Report progress every 5 examples
                    if (idx + 1) % 5 == 0:
                        print(f"Processed {idx + 1}/{len(df)} examples")
                        
                except Exception as e:
                    print(f"Error processing example {idx}: {e}")
                    # Add placeholder results for failed examples
                    self.results["predictions"].append("error")
                    self.results["ground_truth"].append(ground_truth if 'ground_truth' in locals() else "unknown")
                    self.results["latencies"].append(0)
                    self.results["input_tokens"].append(0)
                    self.results["output_tokens"].append(0)
                    self.results["total_tokens"].append(0)
                    self.results["costs"].append(0)
                    self.results["job_ids"].append(job_id if 'job_id' in locals() else str(idx))
                    
                    # Add placeholder for additional fields
                    for result_field in additional_columns.keys():
                        if result_field in self.results:
                            self.results[result_field].append("error")
            
            # Calculate metrics from self.results
            metrics = self._calculate_metrics()
            
            # Track total evaluation time
            total_evaluation_time = time.time() - start_time
            metrics["total_evaluation_time"] = total_evaluation_time
            
            if mlflow_tracking:
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log prompt example
                prompt_example = self.create_prompt("Example job advertisement text")
                mlflow.log_text(prompt_example, "prompt_example.txt")
                
                # Log artifacts
                self._log_artifacts()
            
            return metrics
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics.
        Returns:
            dict: Dictionary of metrics
        """
        # Filter out error cases
        valid_idx = [i for i, p in enumerate(self.results["predictions"]) if p != "error"]
        valid_predictions = [self.results["predictions"][i] for i in valid_idx]
        valid_ground_truth = [self.results["ground_truth"][i] for i in valid_idx]
        
        # Initialize metrics dict
        metrics = {}
        
        # Calculate classification metrics if we have ground truth
        if len(valid_ground_truth) > 0 and valid_ground_truth[0] != "unknown":
            metrics["accuracy"] = accuracy_score(valid_ground_truth, valid_predictions)
            
            # Calculate precision, recall, f1 for classification tasks
            if self.task_name in ["work_arrangement", "seniority"]: # how about salary task?
                labels = sorted(list(set(valid_ground_truth + valid_predictions)))
                
                metrics["macro_precision"] = precision_score(
                    valid_ground_truth, valid_predictions, labels=labels, average="macro", zero_division=0
                )
                metrics["macro_recall"] = recall_score(
                    valid_ground_truth, valid_predictions, labels=labels, average="macro", zero_division=0
                )
                metrics["macro_f1"] = f1_score(
                    valid_ground_truth, valid_predictions, labels=labels, average="macro", zero_division=0
                )
        else:
            metrics["accuracy"] = 0.0
            
        # Calculate token and cost metrics
        metrics["total_tokens"] = sum(self.results["total_tokens"])
        metrics["input_tokens"] = sum(self.results["input_tokens"])
        metrics["output_tokens"] = sum(self.results["output_tokens"])
        metrics["total_cost"] = sum(self.results["costs"])
        metrics["avg_cost_per_example"] = metrics["total_cost"] / len(self.results["costs"]) if self.results["costs"] else 0
        metrics["avg_latency"] = sum(self.results["latencies"]) / len(self.results["latencies"]) if self.results["latencies"] else 0
        metrics["max_latency"] = max(self.results["latencies"]) if self.results["latencies"] else 0
        metrics["min_latency"] = min(self.results["latencies"]) if self.results["latencies"] else 0
        metrics["avg_tokens_per_example"] = metrics["total_tokens"] / len(self.results["total_tokens"]) if self.results["total_tokens"] else 0
        metrics["sample_count"] = len(self.results["predictions"])
        metrics["error_count"] = self.results["predictions"].count("error")
        metrics["error_rate"] = metrics["error_count"] / metrics["sample_count"] if metrics["sample_count"] > 0 else 0
        
        return metrics
    
    def _log_artifacts(self):
        """Create and log artifacts to MLflow."""
        # Create results DataFrame
        results_dict = {
            "job_id": self.results["job_ids"],
            "prediction": self.results["predictions"],
            "ground_truth": self.results["ground_truth"],
            "latency": self.results["latencies"],
            "input_tokens": self.results["input_tokens"],
            "output_tokens": self.results["output_tokens"],
            "total_tokens": self.results["total_tokens"],
            "cost": self.results["costs"]
        }
        
        # Add task-specific columns to results
        if self.task_module and hasattr(self.task_module, 'get_result_columns'):
            additional_columns = self.task_module.get_result_columns()
            for output_col, source_field in additional_columns.items():
                if source_field in self.results:
                    results_dict[output_col] = self.results[source_field]
        
        # Create DataFrame
        results_df = pd.DataFrame(results_dict)
        
        # Save results as CSV
        results_path = f"{self.model_variant}_{self.task_name}_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        # Create confusion matrix if we have ground truth
        if len(self.results["ground_truth"]) > 0 and self.results["ground_truth"][0] != "unknown":
            # Filter out error cases
            valid_idx = [i for i, p in enumerate(self.results["predictions"]) if p != "error"]
            valid_predictions = [self.results["predictions"][i] for i in valid_idx]
            valid_ground_truth = [self.results["ground_truth"][i] for i in valid_idx]
            
            # Create confusion matrix
            labels = sorted(list(set(valid_ground_truth + valid_predictions)))
            cm = confusion_matrix(valid_ground_truth, valid_predictions, labels=labels)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {self.model_variant} on {self.task_name}')
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            
            # Add text annotations(exact values) for better visualization
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout(pad=1.1, rect=[0, 0.03, 1, 0.95])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save and log confusion matrix
            cm_path = f"{self.model_variant}_{self.task_name}_confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)

            # Find misclassified examples
            misclassified_indices = [i for i in range(len(self.results["ground_truth"])) 
                          if self.results["ground_truth"][i] != self.results["predictions"][i] 
                          and self.results["predictions"][i] != "error"]
            
            # Create misclassified DataFrame
            if misclassified_indices:
                misclassified_dict = {
                    "job_id": [self.results["job_ids"][i] for i in misclassified_indices],
                    "ground_truth": [self.results["ground_truth"][i] for i in misclassified_indices],
                    "prediction": [self.results["predictions"][i] for i in misclassified_indices]
                }
                
                # Add task-specific columns to misclassified results
                if self.task_module and hasattr(self.task_module, 'get_result_columns'):
                    additional_columns = self.task_module.get_result_columns()
                    for output_col, source_field in additional_columns.items():
                        if source_field in self.results:
                            misclassified_dict[output_col] = [self.results[source_field][i] for i in misclassified_indices]
                
                # Create DataFrame
                misclassified_df = pd.DataFrame(misclassified_dict)
                misclassified_path = f"{self.model_variant}_{self.task_name}_misclassified.csv"
                misclassified_df.to_csv(misclassified_path, index=False)
                mlflow.log_artifact(misclassified_path)

            # Generate classification report
            report = classification_report(valid_ground_truth, valid_predictions, labels=labels)
            report_path = f"{self.model_variant}_{self.task_name}_classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)


        # Clean up temporary files
        try:
            os.remove(results_path)
            if 'cm_path' in locals():
                os.remove(cm_path)
            if 'misclassified_path' in locals():
                os.remove(misclassified_path)
            if 'report_path' in locals():
                os.remove(report_path)
        except:
            pass 