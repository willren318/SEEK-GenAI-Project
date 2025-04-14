"""
Models package for LLM evaluators.
"""

from models.base_evaluator import LLMEvaluator
from models.claude_evaluator import ClaudeEvaluator

# List of available model evaluator classes
__all__ = [
    'LLMEvaluator',
    'ClaudeEvaluator'
] 