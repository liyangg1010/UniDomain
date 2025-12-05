"""
Core Module Configurations
==========================

This module defines the hyperparameter and model configurations for the four
main pillars of the system: Keyframe Extraction, Atomic Domain Generation,
Domain Fusion, and Task Planning.

Architecture Position:
    [Configuration] -> Used by settings.py to aggregate global settings.
"""

from typing import Callable

from pydantic import BaseModel

from unidomain.configs.llm import LLMAgentConfig

# Pre-defined Model Configs
GPT_4_1_CONFIG = LLMAgentConfig(model="gpt-4.1", kwargs={"temperature": 0.0})
GPT_O3_CONFIG = LLMAgentConfig(model="o3")


# ==========================================
# Keyframe Config Logic
# ==========================================
def hyperparam_delta(num_frames: int) -> int:
    """Calculate the adaptive window step (delta) for energy-based extraction.

    This empirical formula adjusts the locality range based on video length.
    - Shorter videos need finer granularity (smaller delta).
    - Longer videos need coarser granularity (larger delta) to avoid over-segmentation.

    Args:
        num_frames (int): Total number of frames in the video.

    Returns:
        int: The calculated window radius (delta).
    """
    if num_frames <= 100:
        window_step = 10
    elif num_frames <= 150:
        window_step = 20
    elif num_frames <= 200:
        window_step = 30
    else:
        # Empirical scaling for long videos
        window_step = 40 + ((num_frames - 501) // 200 + 1) * 10

    return window_step


class KeyframeConfig(BaseModel):
    """Configuration for the Keyframe Extraction module."""
    # A callable that takes num_frames and returns the window step (delta)
    hyperparam_delta: Callable[[int], int] = hyperparam_delta


# ==========================================
# Atomic Domain Config
# ==========================================
class AtomicDomainConfig(BaseModel):
    """Configuration for the Atomic Domain Generation pipeline.

    Attributes:
        refine_max_attempts (int): Maximum iterations for the Check-Verify-Refine loop.
        num_test_problems (int): Number of test problems to generate for solvability checks.
        solvability_check_threshold (float): Success rate (0.0-1.0) required to pass the planner check.
        pddl_planner_num_workers (int): Parallel workers for running the PDDL planner.
        vlm_model_config (LLMAgentConfig): Model for visual perception (Initial Domain).
        llm_model_config (LLMAgentConfig): Model for reasoning and coding (Revision/Refinement).
    """
    refine_max_attempts: int = 5
    num_test_problems: int = 5
    solvability_check_threshold: float = 0.6
    pddl_planner_num_workers: int = 5

    # Defaults: VLM uses GPT-4.1 (Multimodal), Logic uses GPT-O3 (Reasoning)
    vlm_model_config: LLMAgentConfig = GPT_4_1_CONFIG
    llm_model_config: LLMAgentConfig = GPT_O3_CONFIG


# ==========================================
# Domain Fusion Config
# ==========================================
class DomainFusionConfig(BaseModel):
    """Configuration for the Domain Fusion module.

    Attributes:
        predicate_threshold (float): Cosine similarity threshold for merging predicates.
        operator_threshold (float): Cosine similarity threshold for merging operators.
        llm_model_config (LLMAgentConfig): LLM used for verification and merging logic.
    """
    predicate_threshold: float = 0.3
    operator_threshold: float = 0.3
    llm_model_config: LLMAgentConfig = GPT_4_1_CONFIG


# ==========================================
# Task Planner Config
# ==========================================
class TaskPlannerConfig(BaseModel):
    """Configuration for the Task Planner module.

    Attributes:
        llm_model_config (LLMAgentConfig): LLM used for problem grounding (VLM capabilities required).
    """
    llm_model_config: LLMAgentConfig = GPT_4_1_CONFIG
