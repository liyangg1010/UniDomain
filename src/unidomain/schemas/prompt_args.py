"""
Prompt Argument Schemas
=======================

This module defines the Pydantic models used to validate and structure the
input arguments for LLM prompt templates.

These classes serve as a contract between the Python logic and the text files
in "prompt_templates/". The field names here MUST match the placeholders
(e.g., "{domain}") inside the corresponding ".txt" template files.
"""

from pydantic import BaseModel


class PromptArgs(BaseModel):
    """Base class for all prompt argument schemas."""
    pass


# ==========================================
# Atomic Domain Generation
# ==========================================

class InitialDomainPromptArgs(PromptArgs):
    """Args for initial_domain.txt (VLM Generation)."""
    task_description: str
    available_predicates: str
    available_operators: str

class HolisticReviseDomainPromptArgs(PromptArgs):
    """Args for holistic_revise_domain.txt."""
    instructions: str
    domain: str

class RefineFromSolvabilityCheckPromptArgs(PromptArgs):
    """Args for refine_from_solvability_check.txt."""
    domain: str
    test_problem: str
    solver_error: str

class RefineFromSolutionVerificationPromptArgs(PromptArgs):
    """Args for refine_from_solution_verification.txt."""
    domain: str
    reasoning: str
    instructions: str

class GenerateTestProblemsPromptArgs(PromptArgs):
    """Args for generate_test_problems.txt."""
    instructions: str
    predicates: str
    num_problems: int

class VerifySolutionPromptArgs(PromptArgs):
    """Args for verify_solution.txt."""
    domain: str
    problem: str
    action_sequence: str


# ==========================================
# Domain Fusion
# ==========================================

class CheckMergePredicatesPromptArgs(PromptArgs):
    """Args for check_merge_predicates.txt."""
    predicate_in_domain_1: str
    predicate_in_domain_2: str

class UpdateActionStringPromptArgs(PromptArgs):
    """Args for update_action_string.txt."""
    action_string: str
    predicates_in_action: str
    old_predicate: str
    new_predicate: str

class CheckMergeActionsPromptArgs(PromptArgs):
    """Args for check_merge_actions.txt."""
    action_in_domain_1: str
    predicates_in_domain_1: str
    action_in_domain_2: str
    predicates_in_domain_2: str


# ==========================================
# Task Planner
# ==========================================

class GenerateProblemPromptArgs(PromptArgs):
    """Args for generate_problem.txt (Task Planning)."""
    instructions: str
    available_predicates: str


# ==========================================
# Baselines
# ==========================================

class VLMCoTPromptArgs(PromptArgs):
    instructions: str


class VLMCoTPDDLPromptArgs(PromptArgs):
    instructions: str


# IVML Baselines
class IVMLBoNPromptArgs(PromptArgs):
    instructions: str
    domains: str


class IVMLInitializationPromptArgs(PromptArgs):
    instructions: str


class IVMLOptPromptArgs(PromptArgs):
    instructions: str
    thought: str
    domain: str


class IVMLProblemPromptArgs(PromptArgs):
    instructions: str
    domain: str


class IVMLUpdatePromptArgs(PromptArgs):
    instructions: str
    thought: str
    domain: str
    feedback: str
    

# Code as Policies
class CaPGenerateCodePromptArgs(PromptArgs):
    instructions: str


class CaPFixBugsPromptArgs(PromptArgs):
    code: str
    error: str
