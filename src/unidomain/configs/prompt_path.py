"""
Prompt Template Path Configuration
==================================

This module defines the file system paths for all LLM prompt templates used in the project.
It acts as a central registry, mapping logical task names to physical `.txt` files
located in the `prompt_templates/` directory.

Architecture Position:
    [Configuration] -> Used by settings.py to initialize the global config.
"""

from pathlib import Path

from pydantic import BaseModel

# Locate the source code root relative to this config file
SOURCE_CODE_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = SOURCE_CODE_ROOT / "prompt_templates"

# Sub-directories for specific modules
ATOMIC_DOMAIN_DIR = PROMPT_DIR / "atomic_domain"
DOMAIN_FUSION_DIR = PROMPT_DIR / "domain_fusion"
TASK_PLANNER_DIR = PROMPT_DIR /"task_planner"
BASELINE_DIR = PROMPT_DIR / "baselines"


class PromptPathConfig(BaseModel):
    """Configuration schema for prompt template paths.

    This Pydantic model automatically sets default paths based on the project structure.
    All paths are absolute `pathlib.Path` objects.
    """

    # ==========================================
    # Atomic Domain Generation
    # ==========================================
    generate_test_problems: Path = ATOMIC_DOMAIN_DIR / "generate_test_problems.txt"
    holistic_revise_domain: Path = ATOMIC_DOMAIN_DIR /  "holistic_revise_domain.txt"
    initial_domain: Path = ATOMIC_DOMAIN_DIR / "initial_domain.txt"
    refine_from_solution_verification: Path = ATOMIC_DOMAIN_DIR / "refine_from_solution_verification.txt"
    refine_from_solvability_check: Path = ATOMIC_DOMAIN_DIR / "refine_from_solvability_check.txt"
    verify_solution: Path = ATOMIC_DOMAIN_DIR / "verify_solution.txt"

    # ==========================================
    # Domain Fusion
    # ==========================================
    check_merge_actions: Path = DOMAIN_FUSION_DIR / "check_merge_actions.txt"
    check_merge_predicates: Path = DOMAIN_FUSION_DIR / "check_merge_predicates.txt"
    update_action_string: Path = DOMAIN_FUSION_DIR / "update_action_string.txt"

    # ==========================================
    # Task Planner
    # ==========================================
    generate_problem: Path = TASK_PLANNER_DIR / "generate_problem.txt"
    
    # ==========================================
    # Baselines
    # ==========================================
    vlm_cot: Path = BASELINE_DIR / "vlm_cot.txt"
    vlm_cot_pddl: Path = BASELINE_DIR / "vlm_cot_pddl.txt"

    # IVML methods
    ivml_BoN: Path = BASELINE_DIR / "ivml-BoN.txt"
    ivml_initialization: Path = BASELINE_DIR / "ivml-initialization.txt"
    ivml_opt: Path = BASELINE_DIR / "ivml-opt.txt"
    ivml_problem: Path = BASELINE_DIR / "ivml-problem.txt"
    ivml_update: Path = BASELINE_DIR / "ivml-update.txt"
    
    # Code as Policies
    cap_generate_code: Path = BASELINE_DIR / "cap_generate_code.txt"
    cap_fix_bugs: Path = BASELINE_DIR / "cap_fix_bugs.txt"
