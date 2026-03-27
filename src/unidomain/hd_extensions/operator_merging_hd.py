"""Operator merging for HD domain fusion.

Enhanced operator merging with improved semantic matching and filtering.
"""

from typing import Dict, List, Optional, Tuple

from unidomain.domain_fusion.operator_merging import _check_merge_actions
from unidomain.domain_fusion.similarity import filter_and_sort_strings
from unidomain.pddl import parser
from unidomain.schemas.prompt_args import CheckMergeActionsPromptArgs
from unidomain.schemas.typings import Domain
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.logger import get_task_logger

__all__ = ["merge_operators_hd"]


def merge_operators_hd(
    domain_1: Domain,
    domain_2: Domain,
    llm_agent: LLMAgent,
    threshold: float
) -> Domain:
    """Merge actions from two domains with the same predicates into a unified set of operators.

    This function iterates through operators in the second domain and attempts to
    find semantically similar operators in the first domain to merge with.
    Operators that cannot be merged are simply added to the set.

    Args:
        domain_1: The first domain (predicates, operators) tuple
        domain_2: The second domain (predicates, operators) tuple
        llm_agent: The LLM agent service
        threshold: Similarity threshold (0.0 to 1.0) for action name matching

    Returns:
        Domain: A tuple (merged_predicates, merged_operators) representing the new domain

    Merging Strategy:
        1. For each operator in domain 2, find similar operators in domain 1
        2. Use LLM to check if they can be semantically merged
        3. If merged, replace old operators with new unified operator
        4. If not merged, add domain 2's operator to the final set
        5. Operators from domain 1 that never matched are kept as-is
    """
    execution_logger = get_task_logger(llm_agent.log_dir)

    # Merge predicates of two domains (should be mostly aligned after predicate merging)
    predicates_1, operators_1 = domain_1[0], domain_1[1]
    predicates_2, operators_2 = domain_2[0], domain_2[1]
    merged_predicates = {**predicates_1, **predicates_2}

    # Track replacement history: [old_operator_name (or None), new_operator_name, new_operator]
    replaced_operators: List[Tuple[Optional[str], str, str]] = []

    # Track which operators from Domain 1 have been merged
    merged_operators: Dict[str, bool] = {}

    # Merge operators loop
    for operator_2_name, operator_2 in operators_2.items():
        # Pre-filtering and sort based on semantic similarity
        candidate_actions = filter_and_sort_strings(
            operator_2_name,
            list(operators_1.keys()),
            threshold
        )

        # Recording filtering messages
        filtering_messages = []
        for operator_1_name in operators_1.keys():
            if operator_1_name not in candidate_actions:
                filtering_messages.append(
                    f"similarity between {operator_1_name} and {operator_2_name} "
                    f"is less than threshold {threshold}"
                )
        execution_logger.info("\n".join(filtering_messages))

        for operator_1_name in candidate_actions:
            # One operator can only be merged once
            if operator_1_name in merged_operators:
                continue

            # Merge actions
            operator_1 = operators_1[operator_1_name]

            # Get explanations of predicates used in these actions
            predicates_in_domain_1 = parser.get_predicates_in_action(
                merged_predicates,
                operator_1
            )
            predicates_in_domain_2 = parser.get_predicates_in_action(
                merged_predicates,
                operator_2
            )
            prompt_args = CheckMergeActionsPromptArgs(
                action_in_domain_1=operator_1,
                predicates_in_domain_1=str(predicates_in_domain_1),
                action_in_domain_2=operator_2,
                predicates_in_domain_2=str(predicates_in_domain_2),
            )

            merge_flag, new_operator_name, new_operator = _check_merge_actions(
                llm_agent,
                prompt_args
            )

            # Record merging history
            if merge_flag:
                merged_operators[operator_1_name] = True
                replaced_operators.append([operator_1_name, new_operator_name, new_operator])
                execution_logger.info(
                    f"Merged operators: {operator_1_name} and {operator_2_name} "
                    f"-> {new_operator_name}"
                )
                break  # Only one action can be merged

        # If one action cannot be merged, add it to the new domain directly
        else:
            replaced_operators.append([None, operator_2_name, operator_2])

    # Update operators
    final_operators = operators_1.copy()

    for replaced_operator in replaced_operators:
        old_operator_name, new_operator_name, new_operator = replaced_operator

        # Remove old operator if it was merged
        if old_operator_name is not None:
            del final_operators[old_operator_name]

        # Add new/merged operator
        final_operators[new_operator_name] = new_operator

    return merged_predicates, final_operators
