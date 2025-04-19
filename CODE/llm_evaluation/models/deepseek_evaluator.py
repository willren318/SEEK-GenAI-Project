import os
import time
import pandas as pd
import mlflow
from openai import OpenAI
from models.base_evaluator import LLMEvaluator

class DeepSeekEvaluator(LLMEvaluator):
    """DeepSeek-specific implementation of the LLM evaluator."""
    
    def __init__(self, model_variant="deepseek-chat", task_name="work_arrangement"):
        """
        Initialize the DeepSeek evaluator.
        Args:
            model_variant (str): DeepSeek model variant
                - deepseek-chat: DeepSeek-V3 model
                - deepseek-reasoner: DeepSeek-R1 model
            task_name (str): Task name (work_arrangement, salary, seniority)
        """
        super().__init__(model_variant, task_name) # call the parent class constructor  
        
        # Initialize DeepSeek client using OpenAI format
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # Add cache hit/miss tracking to results dictionary
        self.results["cache_hit_tokens"] = []
        self.results["cache_miss_tokens"] = []
        
        # Set up pricing according to DeepSeek documentation https://api-docs.deepseek.com/
        # Standard prices (UTC 00:30-16:30) per 1M tokens
        self.standard_pricing = {
            "deepseek-chat": {
                "input_cache_hit": 0.07,
                "input_cache_miss": 0.27,
                "output": 1.10
            },
            "deepseek-reasoner": {
                "input_cache_hit": 0.14,
                "input_cache_miss": 0.55,
                "output": 2.19
            }
        }
        
        # Discount prices (UTC 16:30-00:30) per 1M tokens
        self.discount_pricing = {
            "deepseek-chat": {
                "input_cache_hit": 0.035, # 50% off
                "input_cache_miss": 0.135, # 50% off
                "output": 0.55 # 50% off
            },
            "deepseek-reasoner": {
                "input_cache_hit": 0.035, # 75% off
                "input_cache_miss": 0.135, # 75% off
                "output": 0.55 # 75% off
            }
        }
        
        # Determine if we're in discount hours (UTC 16:30-00:30)
        current_hour_utc = time.gmtime().tm_hour
        current_min_utc = time.gmtime().tm_min
        current_time_decimal = current_hour_utc + current_min_utc / 60.0
        
        # Check if current time is in discount period (16:30-00:30 UTC)
        self.is_discount_period = (current_time_decimal >= 16.5) or (current_time_decimal < 0.5)
    
    def call_api(self, prompt):
        """Call the DeepSeek API with the given prompt.
        Args:
            prompt (str): The prompt to send to the DeepSeek model
            
        Returns:
            dict: Results including prediction, latency, and token counts
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_variant,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely. When asked for a specific format or exact answer, provide ONLY that format or answer WITHOUT explanation, reasoning, or additional text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.0,
                stream=False
            )
            
            end_time = time.time()
            
            # Extract the prediction
            prediction = response.choices[0].message.content.strip()
            
            # Extract token counts including cache information
            # From https://api-docs.deepseek.com/news/news0802
            # DeepSeek provides prompt_cache_hit_tokens and prompt_cache_miss_tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Direct access to cache metrics from response.usage
            try:
                # These properties are directly available in the CompletionUsage object
                cache_hit_tokens = response.usage.prompt_cache_hit_tokens
                cache_miss_tokens = response.usage.prompt_cache_miss_tokens
                
                # Add detailed debug output for cache metrics
                if cache_hit_tokens > 0:
                    hit_rate = cache_hit_tokens / (cache_hit_tokens + cache_miss_tokens)
                    print(f"✅ CACHE HIT! Hit: {cache_hit_tokens}, Miss: {cache_miss_tokens}, Hit Rate: {hit_rate:.2%}")
                else:
                    print(f"Cache metrics - hit: {cache_hit_tokens}, miss: {cache_miss_tokens}")
            except AttributeError:
                # If properties don't exist or there's an attribute error
                print(f"Cache metrics not available in response")
                cache_hit_tokens = 0
                cache_miss_tokens = input_tokens
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Error accessing cache metrics: {str(e)}")
                cache_hit_tokens = 0
                cache_miss_tokens = input_tokens
            
            result = {
                "prediction": prediction,
                "latency": round(end_time - start_time, 6),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "cache_miss_tokens": cache_miss_tokens
            }
            
            # Calculate cost using actual cache hit/miss information
            result["cost"] = self.calculate_cost(
                result["cache_hit_tokens"],
                result["cache_miss_tokens"],
                result["output_tokens"]
            )
            
            return result
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            # Return default values on error
            return {
                "prediction": f"ERROR: {str(e)}",
                "latency": round(time.time() - start_time, 6),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
                "cost": 0
            }
    
    def calculate_cost(self, cache_hit_tokens, cache_miss_tokens, output_tokens):
        """Calculate the cost based on DeepSeek pricing using actual cache hit/miss information.
        Args:
            cache_hit_tokens (int): Number of input tokens served from cache
            cache_miss_tokens (int): Number of input tokens not served from cache
            output_tokens (int): Number of output tokens
            
        Returns:
            float: Calculated cost in USD, rounded to 6 decimal places
        """
        # Determine which pricing to use (standard or discount)
        pricing = self.discount_pricing if self.is_discount_period else self.standard_pricing
        
        if self.model_variant in pricing:
            # Calculate costs based on actual cache hits and misses
            hit_cost = (cache_hit_tokens / 1_000_000) * pricing[self.model_variant]["input_cache_hit"]
            miss_cost = (cache_miss_tokens / 1_000_000) * pricing[self.model_variant]["input_cache_miss"]
            output_cost = (output_tokens / 1_000_000) * pricing[self.model_variant]["output"]
            
            return round(hit_cost + miss_cost + output_cost, 6)
        else:
            # If pricing for the specific model is not found, use default
            print(f"Warning: Pricing information not found for model: {self.model_variant}. Using deepseek-chat pricing.")
            default_pricing = pricing.get("deepseek-chat")
            
            hit_cost = (cache_hit_tokens / 1_000_000) * default_pricing["input_cache_hit"]
            miss_cost = (cache_miss_tokens / 1_000_000) * default_pricing["input_cache_miss"]
            output_cost = (output_tokens / 1_000_000) * default_pricing["output"]
            
            return round(hit_cost + miss_cost + output_cost, 6)
            
    def evaluate_dataset(self, dataset_path, mlflow_tracking=True, sample_range=None):
        """Evaluate the model on a dataset and track with MLflow.
        Overriding to properly track cache hit/miss tokens.
        Args:
            dataset_path (str): Path to the CSV dataset
            mlflow_tracking (bool): Whether to track results with MLflow
            sample_range (int | tuple[int, int] | None): 
                - If int: Limit the number of examples to process from the beginning.
                - If tuple (start, end): Process examples from start index (inclusive) to end index (exclusive).
                - If None: Process all examples.
        Returns:
            dict: Evaluation metrics
        """
        # We need to override this method to store the cache hit/miss tokens
        # Much of this code is from the parent class - we need to modify it to handle cache tokens
        
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
                mlflow.log_param("evaluation_date", time.strftime("%Y-%m-%d"))
                mlflow.log_param("evaluation_time", time.strftime("%H:%M:%S"))
                
                # Log sample range/limit
                if isinstance(sample_range, int):
                    mlflow.log_param("sample_limit", sample_range)
                elif isinstance(sample_range, (list, tuple)) and len(sample_range) == 2:
                    mlflow.log_param("sample_start_index", sample_range[0])
                    mlflow.log_param("sample_end_index", sample_range[1])
                elif sample_range is not None:
                     print(f"Warning: Invalid sample_range format: {sample_range}. Processing all examples.")
                
                # Set tags
                mlflow.set_tag("model_family", self.model_variant.split("-")[0])
                mlflow.set_tag("task_type", self.task_name)
                mlflow.set_tag("model_variant", self.model_variant)

            
            # Read dataset
            df_full = pd.read_csv(dataset_path)
            df = df_full # Default to using the full dataframe
            
            # Apply sample range or limit
            if isinstance(sample_range, int):
                limit = sample_range
                if limit > 0 and limit < len(df_full):
                    df = df_full.head(limit)
                    print(f"Limiting to first {len(df)} examples.")
                elif limit <= 0:
                     print(f"Warning: Sample limit ({limit}) must be positive. Processing all examples.")
                elif limit >= len(df_full):
                    print(f"Warning: Sample limit ({limit}) is >= dataset size ({len(df_full)}). Processing all examples.")
            
            elif isinstance(sample_range, (list, tuple)) and len(sample_range) == 2:
                start, end = sample_range
                if 0 <= start < end <= len(df_full):
                    df = df_full[start:end]
                    print(f"Processing examples from index {start} to {end} (exclusive). Total: {len(df)} examples.")
                else:
                    print(f"Warning: Invalid sample range ({start}, {end}) for dataset size {len(df_full)}. Processing all examples.")
                    # Reset df to full if range is invalid
                    df = df_full
            
            elif sample_range is not None:
                 # Already warned during MLflow logging
                 pass # Process all examples

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
                    
                    # Store DeepSeek-specific cache metrics
                    self.results["cache_hit_tokens"].append(result["cache_hit_tokens"])
                    self.results["cache_miss_tokens"].append(result["cache_miss_tokens"])
                    
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
                    
                    # Add DeepSeek-specific placeholder cache data
                    self.results["cache_hit_tokens"].append(0)
                    self.results["cache_miss_tokens"].append(0)
                    
                    # Add placeholder for additional fields
                    for result_field in additional_columns.keys():
                        if result_field in self.results:
                            self.results[result_field].append("error")
            
            # Calculate metrics from self.results
            metrics = self._calculate_metrics()
            
            # Add cache-specific metrics
            total_cache_hit_tokens = sum(self.results["cache_hit_tokens"])
            total_cache_miss_tokens = sum(self.results["cache_miss_tokens"])
            total_input_tokens = total_cache_hit_tokens + total_cache_miss_tokens
            
            metrics["cache_hit_tokens"] = total_cache_hit_tokens
            metrics["cache_miss_tokens"] = total_cache_miss_tokens
            metrics["total_input_tokens"] = total_input_tokens
            
            # Calculate cache hit rate
            if total_input_tokens > 0:
                metrics["cache_hit_rate"] = total_cache_hit_tokens / total_input_tokens
            else:
                metrics["cache_hit_rate"] = 0
            
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
                
                # Create results DataFrame with cache metrics included
                results_dict = {
                    "job_id": self.results["job_ids"],
                    "prediction": self.results["predictions"],
                    "ground_truth": self.results["ground_truth"],
                    "latency": self.results["latencies"],
                    "input_tokens": self.results["input_tokens"],
                    "output_tokens": self.results["output_tokens"],
                    "total_tokens": self.results["total_tokens"],
                    "cache_hit_tokens": self.results["cache_hit_tokens"],
                    "cache_miss_tokens": self.results["cache_miss_tokens"],
                    "cost": self.results["costs"]
                }
                
                # Add task-specific columns to results
                if self.task_module and hasattr(self.task_module, 'get_result_columns'):
                    additional_columns = self.task_module.get_result_columns()
                    for output_col, source_field in additional_columns.items():
                        if source_field in self.results:
                            results_dict[output_col] = self.results[source_field]
                
                # Create DataFrame and save as CSV
                results_df = pd.DataFrame(results_dict)
                results_path = f"{self.model_variant}_{self.task_name}_results.csv"
                results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path)
                
                # Clean up temporary file
                try:
                    os.remove(results_path)
                except Exception as e:
                    print(f"Warning: Error removing temporary file: {e}")
                
                # Continue with other MLflow logging as in parent class
                self._log_artifacts()
            
            return metrics 