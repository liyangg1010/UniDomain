"""LLM utilities for HD (High-Definition / Human Demonstration) data processing.

This module provides utility functions for handling LLM outputs specific to HD data,
including thinking tag removal and custom model configurations.
"""

import re
from typing import Union, List


def exclude_thinking(llm_output: Union[str, List[str]]) -> Union[str, List[str]]:
    """Remove <think>...</think> blocks from LLM outputs.

    Some LLM models (like Qwen3-thinking variants) include reasoning process
    within <think> tags. This function strips those tags to get clean output.

    Args:
        llm_output: LLM output string or list of strings

    Returns:
        The same type as input with thinking blocks removed

    Examples:
        >>> exclude_thinking("<think>reasoning...</think>Answer: 42")
        'Answer: 42'
        >>> exclude_thinking(["<think>A</think>Ans1", "Ans2"])
        ['Ans1', 'Ans2']
    """
    # Regex pattern to match <think>...</think> blocks (including multiline)
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

    def _remove_think_tags(text: str) -> str:
        """Remove think tags from a single string."""
        cleaned = think_pattern.sub('', text)
        # Clean up extra whitespace that may be left
        return cleaned.strip()

    if isinstance(llm_output, str):
        return _remove_think_tags(llm_output)
    elif isinstance(llm_output, list):
        return [_remove_think_tags(output) for output in llm_output]
    else:
        return llm_output


# Extended model costs for HD-specific models
# Format: (input_cost_per_1M_tokens, output_cost_per_1M_tokens)
HD_MODEL_COSTS = {
    "Qwen3-VL-235B-A22B-Instruct": (1.0, 1.0),
    "Qwen3-235B-A22B": (1.0, 1.0),
    "gpt-5": (1.25, 10.0),
    "Qwen3-235B-A22B-thinking": (1.0, 1.0),
    "Qwen3-235B-A22B-Instruct-2507": (1.0, 1.0),
    "Qwen3-32B": (1.0, 1.0),
    "GLM-4.7": (1.0, 1.0),
}


def get_extended_model_costs():
    """Get combined model costs including both default and HD-specific models.

    Returns:
        Dict mapping model names to (input_cost, output_cost) tuples
    """
    from unidomain.configs.llm import MODEL_COSTS

    extended_costs = MODEL_COSTS.copy()
    extended_costs.update(HD_MODEL_COSTS)
    return extended_costs
