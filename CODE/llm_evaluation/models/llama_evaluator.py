import os
import time
import json
from llamaapi import LlamaAPI
from models.base_evaluator import LLMEvaluator

class LlamaEvaluator(LLMEvaluator):
    """Llama-specific implementation of the LLM evaluator using llamaapi."""

    def __init__(self, model_variant="llama3.1-70b", task_name="work_arrangement"):
        """
        Initialize the Llama evaluator.
        Args:
            model_variant (str): Llama model variant (e.g., llama3.1-70b)
                - llama3.1-8b (small): Good balance of performance and speed
                - llama3.1-70b (medium): High quality for most evaluation tasks
                - llama4-maverick (large): Highest quality, best for complex evaluation
            task_name (str): Task name (work_arrangement, salary, seniority)
        """
        super().__init__(model_variant, task_name) # call the parent class constructor

        # Initialize LlamaAPI client
        api_key = os.environ.get("LLAMA_API_KEY")
        if not api_key:
            raise ValueError("LLAMA_API_KEY environment variable not set")

        self.client = LlamaAPI(api_key)

        # model price can be found at https://console.llmapi.com/en/dashboard/ai-model after login
        self.pricing = {
            "llama3.1-8b": {"input": 0.4, "output": 0.4},
            "llama3.1-70b": {"input": 2.8, "output": 2.8},
            "llama4-maverick": {"input": 9.0, "output": 9.0},
        }

        # Determine base model for pricing
        self.base_model = None
        for model_key in self.pricing:
            if model_key in model_variant:
                self.base_model = model_key
                break
                
        # If no specific match, use llama3.1-70b as default
        if not self.base_model:
            print(f"Warning: Unknown model variant '{model_variant}'. Using pricing for llama3.1-70b as default.")
            self.base_model = "llama3.1-70b"

    def call_api(self, prompt):
        """Call the Llama API with the given prompt.
        Args:
            prompt (str): The prompt to send to the Llama model
        Returns:
            dict: Results including prediction, latency, and token counts (if available)
        """
        start_time = time.time()

        api_request_json = {
            "model": self.model_variant,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that follows instructions precisely. When asked for a specific format or exact answer, provide ONLY that format or answer WITHOUT explanation, reasoning, or additional text."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "max_token": 500, 
            "temperature": 0.0,
            # Force more concise, direct responses
            "top_p": 0.1,
            # "response_format" parameter to enforce structure
            "response_format": {"type": "text"}
        }

        try:
            response_raw = self.client.run(api_request_json)
            response = response_raw.json() # Get JSON content from response

            end_time = time.time()


            prediction = "ERROR: Response format not recognized" # Default error message
            if 'choices' in response and len(response['choices']) > 0:
                 message = response['choices'][0].get('message', {})
                 if 'content' in message:
                     prediction = message['content'].strip()
                 elif 'function_call' in message:
                     prediction = f"Function Call: {json.dumps(message['function_call'])}"


            # Token counts: LlamaAPI response format
            input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response.get("usage", {}).get("completion_tokens", 0)
            total_tokens = input_tokens + output_tokens

            result = {
                "prediction": prediction,
                "latency": round(end_time - start_time, 6),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }

            # Calculate cost
            result["cost"] = self.calculate_cost(
                result["input_tokens"],
                result["output_tokens"]
            )

            return result

        except Exception as e:
            print(f"Error calling Llama API: {str(e)}")
            # Return default values on error
            return {
                "prediction": f"ERROR: {str(e)}",
                "latency": round(time.time() - start_time, 6),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0
            }

    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate the cost based on Llama API pricing.
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens

        Returns:
            float: Calculated cost in USD, rounded to 6 decimal places
        """
        pricing_info = self.pricing.get(self.base_model)
        
        if pricing_info:
            input_cost = (input_tokens / 1_000_000) * pricing_info["input"]
            output_cost = (output_tokens / 1_000_000) * pricing_info["output"]
            return round(input_cost + output_cost, 6)
        else:
            # have set a default model
            print(f"Warning: Pricing information not found for base model: {self.base_model}. Using llama3.1-70b pricing.")
            default_pricing = self.pricing.get("llama3.1-70b")
            input_cost = (input_tokens / 1_000_000) * default_pricing["input"]
            output_cost = (output_tokens / 1_000_000) * default_pricing["output"]
            return round(input_cost + output_cost, 6) 