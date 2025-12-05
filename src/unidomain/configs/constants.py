"""
Global Project Constants
========================

This module defines project-wide constants using namespace classes.
It centralizes filenames, directory structures, data keys, environment
variables, and internal logger names to ensure consistency across the
entire codebase.

By using these constants instead of hardcoded strings, the project enforces
a strict "Single Source of Truth" for system protocols and file conventions.

Architecture Position:
    [Configuration / Constants] -> Shared static definitions used by
    all pipelines and infrastructure modules.
"""

from typing import Final

__all__ = ["File", "Dir", "JSONKey", "EnvVar", "LoggerName"]


class File:
    """Standard filenames for output artifacts."""
    
    # Atomic Domain Generation
    INITIAL_DOMAIN_STEM: Final[str] = "initial_domain"
    REVISED_DOMAIN_STEM: Final[str] = "revised_domain"
    REFINED_DOMAIN_STEM: Final[str] = "refined_domain"

    INITIAL_DOMAIN_PDDL: Final[str] = f"{INITIAL_DOMAIN_STEM}.pddl"
    REVISED_DOMAIN_PDDL: Final[str] = f"{REVISED_DOMAIN_STEM}.pddl"
    REFINED_DOMAIN_PDDL: Final[str] = f"{REFINED_DOMAIN_STEM}.pddl"

    INITIAL_DOMAIN_JSON: Final[str] = f"{INITIAL_DOMAIN_STEM}.json"
    REVISED_DOMAIN_JSON: Final[str] = f"{REVISED_DOMAIN_STEM}.json"
    REFINED_DOMAIN_JSON: Final[str] = f"{REFINED_DOMAIN_STEM}.json"
    
    TEST_PROBLEM_PREFIX: Final[str] = "test_problem_"
    PLAN_PREFIX: Final[str] = "plan_"
    REASONING: Final[str] = "reasoning.txt"

    # Domain Fusion
    META_DOMAIN_GRAPH: Final[str] = "meta_domain_graph"
    ATOMIC_DOMAIN_GRAPH: Final[str] = "atomic_domain_graph"
    DOMAIN_FUSION_TREE: Final[str] = "binary_domain_fusion_tree.png"
    MAPPING_TABLE: Final[str] = "mapping_table.json"

    # Atomic & Meta
    ATOMIC_DOMAIN_PDDL: Final[str] = "atomic_domain.pddl"
    META_DOMAIN_PDDL: Final[str] = "meta_domain.pddl"

    ATOMIC_DOMAIN_JSON: Final[str] = "atomic_domain.json"
    META_DOMAIN_JSON: Final[str] = "meta_domain.json"
    
    # PDDL (Intermediate)
    DOMAIN_PDDL: Final[str] = "domain.pddl"
    PROBLEM_PDDL: Final[str] = "problem.pddl"
    PROBLEM_JSON: Final[str] = "problem.json"
    OUTPUTS: Final[str] = "outputs.txt"
    RESULTS_PREFIX: Final[str] = "results-"

    # Task Planner
    FINAL_METRICS: Final[str] = "summary.json"
    FINAL_RESULTS: Final[str] = "solution.txt"
    GROUP_PREDICATES: Final[str] = "group_predicates.txt"

    # Logs
    EXEC_LOG: Final[str] = "execution.log"
    USAGE_LOG: Final[str] = "llm_usage.log"

    # Checklist
    CHECKLIST_NAME: Final[str] = "checklist.json"
    ERROR_LOG_NAME: Final[str] = "error.txt"


class Dir:
    """Standard directory names for pipeline stages."""

    # For Atomic Domain Generation
    INITIAL_DOMAIN: Final[str] = "initial_domain"
    REVISED_DOMAIN: Final[str] = "revised_domain"
    CHECK_SUFFIX: Final[str] = ": solvability check"
    VERIFY_SUFFIX: Final[str] = ": solution verification"


class JSONKey:
    """JSON keys for statistics aggregation."""

    # For Stat Summary
    TOTAL_CALLS: Final[str] = "nums_llm_call"
    TOTAL_COSTS: Final[str] = "costs"
    TOTAL_INPUT_TOKENS: Final[str] = "input_tokens"
    TOTAL_OUTPUT_TOKENS: Final[str] = "output_tokens"
    TOTAL_THINKING_TIME: Final[str] = "thinking_time"
    TOTAL_PLANNER_TIME: Final[str] = "pddl_planner_time"

    # For Checklist
    PATH: Final[str] = "path"
    INSTRUCTION: Final[str] = "instruction"

    # For Mapping Table
    COMMENT: Final[str] = "_comment"


class EnvVar:
    """Keys for Environment Variables loaded from .env or system env."""
    
    # LLM setup
    API_KEY: Final[str] = "API_KEY"
    BASE_URL: Final[str] = "BASE_URL"
    CLIENT_TYPE: Final[str] = "CLIENT_TYPE"
    API_VERSION: Final[str] = "API_VERSION"  # For Azure
    
    # External Tools
    FAST_DOWNWARD_PATH: Final[str] = "FAST_DOWNWARD_PATH"
    
    # Model Paths
    EMBEDDING_MODEL_PATH: Final[str] = "EMBEDDING_MODEL_PATH"


class LoggerName:
    """Internal logger names used to retrieve specific logging instances."""
    EXECUTION: Final[str] = "execution"
    LLM_USAGE: Final[str] = "llm_usage"
