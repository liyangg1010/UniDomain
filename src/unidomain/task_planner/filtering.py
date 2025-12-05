"""
Domain Filtering Module
=======================

This module implements the pre-processing logic for the task planner.
It analyzes the specific problem (initial state and goals) to identify
relevant predicates and operators, effectively pruning the domain to a
smaller, task-specific subset.

This reduction serves two main purposes:
1. It improves the accuracy of LLM-based problem grounding by eliminating
   irrelevant domain elements that might distract the LLM.
2. It increases the scale of problems solvable by the PDDL planner by
   reducing the search space.

Architecture Position:
    [Core Algorithms / Functional Level] -> Called by task_planner.pipeline
"""

import re
from typing import Set

from unidomain.pddl.parser import analyze_pddl_action, get_predicate_name
from unidomain.schemas.typings import Goal, Init, Operators, Predicates, Problems

__all__ = ["extract_operators_from_predicates", "predicate_filtering"]

# Standard PDDL keywords to ignore when extracting predicate names via regex
_IGNORED_KEYWORDS = {
    "and", "or", "not", "imply", "exists", "forall", "when",
    "define", "domain", "problem", "metric", "minimize", "maximize"
}
_UBIQUITOUS_PREDICATES = {"holding", "hand_free"}

def _extract_candidate_predicates_from_problem(inits: Init, goals: Goal) -> Set[str]:
    """Extract all predicate names appearing in the initial state and goal.

    Args:
        inits (Init): List of initial state strings (e.g., ["(at a b)", ...]).
        goals (Goal): Goal string (e.g., "(:goal (and (on c d)))").

    Returns:
        Set[str]: A set of predicate names found in the problem.
    """

    candidates: Set[str] = set()

    # Regex to capture the first word inside parentheses: (word ...
    pattern = re.compile(r"\(([\w-]+)")

    # Get predicate names in "inits" part
    for init in inits:
        match = pattern.search(init)
        if match:
            name = match.group(1)
            if name not in _IGNORED_KEYWORDS:
                candidates.add(name)

    # Get predicate names in "goals" part
    matches = pattern.findall(goals)
    for name in matches:
        if name not in _IGNORED_KEYWORDS:
            candidates.add(name)

    return candidates


def _extract_actual_predicates(candidate_names: Set[str], all_predicates: Predicates) -> Predicates:
    """Filter the full domain predicates to keep only those requested.

    Args:
        candidate_names (Set[str]): Set of predicate names to keep.
        all_predicates (Predicates): The full dictionary of domain predicates.

    Returns:
        Predicates: A dictionary of the filtered predicates.
    """

    actual_predicates: Predicates = {}

    # Create a mapping from clean name to full definition key, e.g., "at" -> "(at ?x ?y)"
    name_to_key = {get_predicate_name(k): k for k in all_predicates.keys()}

    # Extract the actual predicates from the full set
    for name in candidate_names:
        if name in name_to_key:
            full_key = name_to_key[name]
            actual_predicates[full_key] = all_predicates[full_key]

    return actual_predicates


def extract_operators_from_predicates(
    whole_operators: Operators,
    chosen_predicates: Predicates
) -> Operators:
    """Extract relevant operators based on the chosen predicates.

    An operator is selected if it involves any of the chosen predicates in its
    preconditions or effects.

    Note:
        Common predicates like 'holding' or 'hand_free' are skipped during the
        dependency check to prevent selecting every single operator (since almost
        all operators might use them), ensuring a more focused pruning.

    Args:
        whole_operators (Operators): The complete set of operators.
        chosen_predicates (Predicates): The subset of predicates active in the problem.

    Returns:
        Operators: A filtered dictionary of operators.
    """

    operators: Operators = {}
    for predicate in chosen_predicates:
        predicate_name = get_predicate_name(predicate)
        
        # Skip ubiquitous predicates to ensure effective pruning
        # Names here can be adapted if they have different names
        if predicate_name in _UBIQUITOUS_PREDICATES:
            continue

        # Extract operators
        for operator_name, operator in whole_operators.items():
            # Skip checking operators already selected again
            if operator_name in operators:
                continue

            # Analyze the operator to get used predicates
            action_info = analyze_pddl_action(operator)
            used_predicates = action_info["precondition_predicates"] + action_info["effect_predicates"]
            if predicate_name in used_predicates:
                operators[operator_name] = operator

    return operators


def predicate_filtering(
    problems: Problems,
    all_predicates: Predicates
) -> Predicates:
    """Filter predicates that appear in a set of problems.

    Args:
        problems (Problems): A list of problem tuples.
        all_predicates (Predicates): The full domain predicates.

    Returns:
        Predicates: The subset of predicates relevant to the provided problems.
    """
    
    candidate_names = set()

    # Aggregate predicate names from all problems
    for problem in problems:
        init = problem[1]
        goal = problem[2]
        names = _extract_candidate_predicates_from_problem(init, goal)
        candidate_names.update(names)

    chosen_predicates = _extract_actual_predicates(candidate_names, all_predicates)
    return chosen_predicates

