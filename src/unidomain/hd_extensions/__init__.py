"""HD (Human Demonstration) Extensions for UniDomain.

This module provides enhanced functionality for processing HD data where:
- Frames are extracted from narrations with temporal gaps
- Type annotations need to be removed for compatibility
- Custom LLM configurations are required

The HD extensions maintain compatibility with the original UniDomain API while
providing specialized handling for datasets with discontinuous temporal structure.

Main Components:
    - Atomic Domain Pipeline: Frame-by-frame learning adapted for temporal gaps
    - Domain Fusion Pipeline: Type-safe merging with conflict resolution
    - LLM Utilities: Thinking tag removal and extended model support

Usage:
    >>> from unidomain.hd_extensions import (
    ...     atomic_domain_pipeline_for_hd,
    ...     domain_fusion_pipeline_hd
    ... )
    >>>
    >>> # Generate atomic domain from HD data
    >>> success = atomic_domain_pipeline_for_hd(
    ...     data_path="tasks.json",
    ...     save_dir="output/atomic",
    ...     vlm_model="gpt-5",
    ...     llm_model="gpt-4o-mini"
    ... )
    >>>
    >>> # Fuse multiple atomic domains
    >>> domain_fusion_pipeline_hd(
    ...     domain_dir="atomic_domains",
    ...     output_dir="output/fused",
    ...     llm_model="gpt-4o-mini"
    ... )
"""

from .atomic_domain_pipeline_hd import atomic_domain_pipeline_for_hd
from .domain_fusion_pipeline_hd import delete_type, domain_fusion_pipeline_hd
from .initial_domain_hd import (
    initiate_domain_from_keyframes_for_hd,
    run_initial_domain_step_for_hd,
)
from .llm_utils_hd import (
    HD_MODEL_COSTS,
    exclude_thinking,
    get_extended_model_costs,
)
from .operator_merging_hd import merge_operators_hd
from .predicate_merging_hd import merge_predicates_hd
from .runner_hd import fuse_two_domains_hd, run_domain_fusion_hd

__all__ = [
    # Atomic Domain
    "atomic_domain_pipeline_for_hd",
    "initiate_domain_from_keyframes_for_hd",
    "run_initial_domain_step_for_hd",
    # Domain Fusion
    "domain_fusion_pipeline_hd",
    "delete_type",
    "merge_predicates_hd",
    "merge_operators_hd",
    "fuse_two_domains_hd",
    "run_domain_fusion_hd",
    # LLM Utilities
    "exclude_thinking",
    "HD_MODEL_COSTS",
    "get_extended_model_costs",
]

__version__ = "0.1.0"
__author__ = "LEAP Project"
