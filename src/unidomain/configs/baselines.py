"""
Baselines Configuration
=======================

This module defines configuration settings specifically for baseline comparison methods.
It sets up the default LLM parameters used when running benchmark comparisons
(e.g., VLM-CoT, IVML) to ensure fair and consistent evaluation.

Architecture Position:
    [Configuration] -> Used by settings.py
"""

from pydantic import BaseModel

from unidomain.configs.llm import LLMAgentConfig

# Pre-defined model configurations for baselines
GPT_4_1_CONFIG = LLMAgentConfig(model="gpt-4.1", kwargs={"temperature": 0.0})

class BaselinesConfig(BaseModel):
    """Configuration for Baseline experiments.

    Attributes:
        llm_model_config (LLMAgentConfig): The LLM agent settings to use for
            baseline tasks. Defaults to GPT-4.1 with temperature 0.0 for reproducibility.
    """
    llm_model_config: LLMAgentConfig = GPT_4_1_CONFIG
