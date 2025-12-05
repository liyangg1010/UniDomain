"""
Operator Merging Module
=======================

This module implements the logic for merging operators (actions) from two PDDL domains.
It assumes that the predicates have already been unified. It attempts to merge actions
that are semantically similar and structurally compatible, relying on LLM for the
final merge decision.

Architecture Position:
    [Core Algorithms] -> Called by runner.py, uses similarity.py.
"""

from typing import Dict, List, Optional, Tuple

from unidomain.configs.settings import config
from unidomain.domain_fusion.similarity import filter_and_sort_strings
from unidomain.pddl import parser
from unidomain.schemas.llm_task_outputs import CheckMergeActionsOutputs
from unidomain.schemas.prompt_args import CheckMergeActionsPromptArgs
from unidomain.schemas.typings import Domain
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.logger import get_task_logger

__all__ = ["merge_operators"]

def _check_merge_actions(
    llm_agent: LLMAgent,
    prompt_args: CheckMergeActionsPromptArgs,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Call the LLM to check if two actions should be merged.

    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (CheckMergeActionsPromptArgs): Arguments for LLM prompt.

    Returns:
        Tuple[bool, Optional[str], Optional[str]]: A tuple containing:
            - A boolean flag indicating if the actions were merged.
            - The new operator name (if merged, else None).
            - The new operator string (if merged, else None).
    """

    # Call LLM and parse JSON into specified format
    outputs: CheckMergeActionsOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.check_merge_actions,
        prompt_args=prompt_args,
        validator=CheckMergeActionsOutputs,
    )

    # return necessary information
    merge_flag = outputs.merge_flag
    new_operator_name = outputs.new_operator_name
    new_operator = outputs.new_operator

    return merge_flag, new_operator_name, new_operator


def merge_operators(
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
        domain_1 (Domain): The first domain.
        domain_2 (Domain): The second domain.
        llm_agent (LLMAgent): The LLM agent service.
        threshold (float): Similarity threshold (0.0 to 1.0) for action name matching.

    Returns:
        Domain: A tuple (merged_predicates, merged_operators) representing the new domain.
    """

    execution_logger = get_task_logger(llm_agent.log_dir)

    # Merge predicates of two domains
    predicates_1, operators_1 = domain_1[0], domain_1[1]
    predicates_2, operators_2 = domain_2[0], domain_2[1]
    merged_predicates = {**predicates_1, **predicates_2}
    
    # [old_operator_name (or None), new_operator_name, new_operator]
    replaced_operators: List[Tuple[Optional[str], str, str]] = []

    # Track which operators from Domain 1 have been merged
    merged_operators: Dict[str, bool] = {}

    # Merge operators loop
    for operator_2_name, operator_2 in operators_2.items():

        # Pre-filtering and sort based on semantic similarity
        candidate_actions = filter_and_sort_strings(operator_2_name, list(operators_1.keys()), threshold)
        
        # Recording filtering messages
        filtering_messages = []
        for operator_1_name in operators_1.keys():
            if operator_1_name not in candidate_actions:
                filtering_messages.append(
                    f"similarity between {operator_1_name} and {operator_2_name} is less than threshold {threshold}"
                )
        execution_logger.info("\n".join(filtering_messages))

        for operator_1_name in candidate_actions:
            # One operator can only be unioned once
            if operator_1_name in merged_operators:
                continue
            
            # Merge actions
            operator_1 = operators_1[operator_1_name]
            
            # Get explanations of predicates used in these actions
            predicates_in_domain_1 = parser.get_predicates_in_action(merged_predicates, operator_1)
            predicates_in_domain_2 = parser.get_predicates_in_action(merged_predicates, operator_2)
            prompt_args = CheckMergeActionsPromptArgs(
                action_in_domain_1=operator_1,
                predicates_in_domain_1=str(predicates_in_domain_1),
                action_in_domain_2=operator_2,
                predicates_in_domain_2=str(predicates_in_domain_2),
            )

            merge_flag, new_operator_name, new_operator = _check_merge_actions(llm_agent, prompt_args)
            
            # Record merging history
            if merge_flag:
                merged_operators[operator_1_name] = True
                replaced_operators.append([operator_1_name, new_operator_name, new_operator])
                break   # Only one action can be merged

        # if one action can not be merged, add it to the new domain directly
        else:
            replaced_operators.append([None, operator_2_name, operator_2])

    # Update operators
    final_operators = operators_1.copy()

    for replaced_operator in replaced_operators:
        old_operator_name, new_operator_name, new_operator = replaced_operator

        # Remove old operators
        if old_operator_name is not None:
            del final_operators[old_operator_name]

        final_operators[new_operator_name] = new_operator
 
    return merged_predicates, final_operators
