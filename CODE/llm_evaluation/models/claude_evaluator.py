import os
import time
import anthropic
from models.base_evaluator import LLMEvaluator

class ClaudeEvaluator(LLMEvaluator):
    """Claude-specific implementation of the LLM evaluator."""
    
    def __init__(self, model_variant="claude-3-7-sonnet-20250219", task_name="work_arrangement"):
        """
        Initialize the Claude evaluator.
        Args:
            model_variant (str): model variant (such as claude-3-haiku-20240307)
            task_name (str): Task name (work_arrangement, salary, seniority)
        """
        super().__init__(model_variant, task_name) # call the parent class constructor  
        
        # Initialize Claude client  
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Claude API pricing per 1M tokens (https://www.anthropic.com/pricing?utm_source=chatgpt.com#api)
        # 50% discount with batch processing (https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#pricing)
        self.pricing = {
            "claude-3-haiku": {"input": 0.25, "output": 1.25},   # $0.25 per 1M input, $1.25 per 1M output    # $0.8 per 1M input, $4 per 1M output
            "claude-3-5-sonnet": {"input": 2.4, "output": 12},   # $2.4 per 1M input, $12 per 1M output
            "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},   # $3 per 1M input, $15 per 1M output
            "claude-3-opus": {"input": 15.00, "output": 75.00},   # $15 per 1M input, $75 per 1M output
        }
        
        # Extract base model for pricing
        if "claude-3-haiku" in model_variant:
            self.base_model = "claude-3-haiku"
        elif "claude-3-5-haiku" in model_variant:
            self.base_model = "claude-3-5-haiku"
        elif "claude-3-5-sonnet" in model_variant:
            self.base_model = "claude-3-5-sonnet"
        elif "claude-3-7-sonnet" in model_variant:
            self.base_model = "claude-3-7-sonnet"
        elif "claude-3-opus" in model_variant:
            self.base_model = "claude-3-opus"
        else:
            # Default to claude-3-7-sonnet pricing if model not recognized
            self.base_model = "claude-3-7-sonnet"
    
    def call_api(self, prompt):
        """Call the Claude API with the given prompt.
        Args:
            prompt (str): The prompt to send to Claude
            
        Returns:
            dict: Results including prediction, latency, and token counts
        """
        start_time = time.time()
        
        # Call the Claude API, reference at https://docs.anthropic.com/en/api/messages
        try:
            response = self.client.messages.create(
                model=self.model_variant,
                max_tokens=500, # Increased max output tokens
                temperature=0.0,  # use deterministic output
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            end_time = time.time()
            
            # Extract the prediction (first line of the response)
            prediction = response.content[0].text.strip()
            
            result = {
                "prediction": prediction,
                "latency": round(end_time - start_time, 6),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Calculate cost
            result["cost"] = self.calculate_cost(
                result["input_tokens"], 
                result["output_tokens"]
            )
            
            return result
            
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            # Return default values on error
            return {
                "prediction": "ERROR",
                "latency": round(time.time() - start_time, 6),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0
            }
    
    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate the cost based on Claude pricing.
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            
        Returns:
            float: Calculated cost in USD, rounded to 6 decimal places
        Raises:
            ValueError: If pricing information for the base model is not found.
        """
        if self.base_model in self.pricing:
            input_cost = (input_tokens / 1_000_000) * self.pricing[self.base_model]["input"]
            output_cost = (output_tokens / 1_000_000) * self.pricing[self.base_model]["output"]
            return round(input_cost + output_cost, 6)
        else:
            # If pricing for the specific base model is not found, raise an error.
            raise ValueError(f"Pricing information not found for base model: {self.base_model}") 