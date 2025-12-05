"""
Global Settings
===============

This module aggregates all sub-configurations into a single, unified settings object.
It serves as the central source of truth for hyperparameters, file paths, and
model configurations across the entire project.

Architecture Position:
    [Configuration / Entry Point] -> Imported by logic modules to access `config`.
"""

from pydantic import BaseModel, Field

from unidomain.configs.baselines import BaselinesConfig
from unidomain.configs.llm import LLMTasksConfig
from unidomain.configs.prompt_path import PromptPathConfig
from unidomain.configs.unidomain import AtomicDomainConfig, DomainFusionConfig, KeyframeConfig, TaskPlannerConfig


class UniDomainSettings(BaseModel):
    """The root configuration schema for the UniDomain project.

    This class composes specific configuration sections for each functional module.
    Default values are instantiated from their respective config classes.

    Attributes:
        keyframe (KeyframeConfig): Settings for video processing and frame extraction.
        atomic_domain (AtomicDomainConfig): Settings for VLM perception and PDDL generation.
        domain_fusion (DomainFusionConfig): Settings for merging logic and thresholds.
        task_planner (TaskPlannerConfig): Settings for the planning engine.
        baselines (BaselinesConfig): Settings for benchmark comparison methods.
        prompt_path (PromptPathConfig): Registry of file paths for prompt templates.
        llm_tasks (LLMTasksConfig): Global policies for LLM retries and validation.
    """

    # Core Modules
    keyframe: KeyframeConfig = Field(default_factory=KeyframeConfig)
    atomic_domain: AtomicDomainConfig = Field(default_factory=AtomicDomainConfig)
    domain_fusion: DomainFusionConfig = Field(default_factory=DomainFusionConfig)
    task_planner: TaskPlannerConfig = Field(default_factory=TaskPlannerConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)

    # Infrastructure & Resources
    prompt_path: PromptPathConfig = Field(default_factory=PromptPathConfig)
    llm_tasks: LLMTasksConfig = Field(default_factory=LLMTasksConfig)

# Singleton instance used throughout the application
# Usage: from unidomain.configs.settings import config
config = UniDomainSettings()
