"""
LLM Configuration
=================

This module defines configuration schemas for Large Language Model (LLM) agents
and task execution policies. It acts as the central registry for model pricing,
instantiation parameters, and reliability settings (retries/delays).

Architecture Position:
    [Configuration]
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

# Pricing Registry. Unit: USD per 1 Million tokens.
# Format: "model_name": (Input Price, Output Price)
# Note: These prices should be updated regularly as providers adjust rates.
MODEL_COSTS = {
    "gpt-4o": (2.5, 10),
    "o1": (15, 60),
    "o3-mini": (1.1, 4.4),
    "chatgpt-4o-latest": (5, 15),
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "o3": (4, 10),
    "o4-mini": (1.1, 4.4)
}


class LLMAgentConfig(BaseModel):
    """Configuration for instantiating a specific LLM Agent.

    Attributes:
        model (str): The model identifier string (e.g., "gpt-4o").
        kwargs (Dict[str, Any]): Additional keyword arguments passed directly
            to the API client (e.g., temperature, top_p, max_tokens).
            Defaults to an empty dictionary.
    """
    model: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class LLMTasksConfig(BaseModel):
    """Configuration for LLM task execution reliability and retry policies.

    Attributes:
        call_max_retries (int): Max attempts for raw API calls (handling
            RateLimitError, APIConnectionError, etc.). Defaults to 10.
        call_retry_delay (int): Seconds to wait before retrying a failed API call.
            Defaults to 2.
        
        validator_max_retries (int): Max attempts for output parsing/validation
            (handling JSONDecodeError, ValidationError). Defaults to 5.
        validator_retry_delay (int): Seconds to wait before retrying validation.
            Defaults to 2.
    """
    # Network/API level retries
    call_max_retries: int = 10
    call_retry_delay: int = 2

    # Logic/Parsing level retries
    validator_max_retries: int = 5
    validator_retry_delay: int = 2
