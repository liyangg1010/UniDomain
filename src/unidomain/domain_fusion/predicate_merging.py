"""
Predicate Merging Module
========================

This module implements the logic for merging predicates between two PDDL domains.
It identifies semantically similar predicates using vector embeddings and LLM
verification, and then updates the corresponding operators (actions) in both domains
to reflect the merged predicate definitions.

Architecture Position:
    [Core Algorithms] -> Called by runner.py, uses similarity.py.
"""

from typing import List, Optional, Tuple

from unidomain.configs.settings import config
from unidomain.domain_fusion.similarity import filter_and_sort_strings
from unidomain.pddl import parser
from unidomain.pddl.io import load_domain
from unidomain.schemas.llm_task_outputs import CheckMergePredicatesOutputs, UpdateActionStringOutputs
from unidomain.schemas.prompt_args import CheckMergePredicatesPromptArgs, UpdateActionStringPromptArgs
from unidomain.schemas.typings import Domain, Operators, Paths, Predicates
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.logger import get_task_logger

__all__ = ["merge_predicates"]

def _check_update_action_string(origin_predicate: str, new_predicate: str) -> bool:
    """Check if the structure (arguments) of a predicate has changed.

    If the arguments change (e.g., from (at ?x) to (at ?x ?y)), simple string
    replacement is insufficient, and an LLM update is required.

    Args:
        origin_predicate (str): The original predicate string (e.g., "(at ?x)").
        new_predicate (str): The new predicate string.

    Returns:
        bool: True if the argument signature differs, requiring an LLM update.
    """

    # Simple parsing to compare argument lists
    origin_parts = origin_predicate.strip("() ").split()
    new_parts = new_predicate.strip("() ").split()

    # Compare arguments (skipping the predicate name at index 0)
    origin_args = origin_parts[1:]
    new_args = new_parts[1:]

    return origin_args != new_args


def _check_merge_predicates(
    llm_agent: LLMAgent,
    prompt_args: CheckMergePredicatesPromptArgs,
) -> Tuple[bool, Optional[List[str]]]:
    """Call the LLM to check if two semantically similar predicates should be merged.
    
    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (CheckMergePredicatesPromptArgs): Arguments for LLM prompt.

    Returns:
        Tuple[bool, Optional[List[str]]]: A tuple containing:
            - A boolean flag indicating if merge is approved.
            - A list of the merged predicates if merged, otherwise None.
    """

    # Call LLM and parse JSON into specified format
    outputs: CheckMergePredicatesOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.check_merge_predicates,
        prompt_args=prompt_args,
        validator=CheckMergePredicatesOutputs,
    )

    # Return necessary information
    merge_flag = outputs.merge_flag
    new_predicate = outputs.new_predicate

    return merge_flag, new_predicate


def _update_action_string(
    llm_agent: LLMAgent,
    prompt_args: UpdateActionStringPromptArgs,
) -> str:
    """Call the LLM to rewrite a PDDL action string when predicate signatures change.
    
    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (UpdateActionStringPromptArgs): Arguments for LLM prompt.

    Returns:
        str: The updated PDDL action string.
    """

    # Call LLM and parse JSON into specified format
    outputs: UpdateActionStringOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.update_action_string,
        prompt_args=prompt_args,
        validator=UpdateActionStringOutputs,
    )

    # Return necessary information
    updated_action_string = outputs.updated_action

    return updated_action_string


def _update_domain_operators(
    operators: Operators,
    predicates: Predicates,
    old_predicate: str,
    new_predicate: str,
    llm_agent: LLMAgent
) -> None:
    """Utility function to update operators based on predicate changes.

    This function iterates through all operators in a domain. If an operator uses
    the "old_predicate", it updates the operator to use "new_predicate". This may involve
    simple string replacement or an LLM-based rewrite if the predicate signature changed.

    Args:
        operators (Operators): The operators to update (modified in-place).
        predicates (Predicates): The current domain predicates.
        old_predicate (str): The old predicate string to be replaced.
        new_predicate (str): The new predicate string to replace with.
        llm_agent (LLMAgent): The LLM agent service.
    """
    # Update operators loop
    for operator_name, operator in list(operators.items()):
        if parser.get_predicate_name(old_predicate) in operator:
            # Check smart LLM update or simple replace
            if _check_update_action_string(old_predicate, new_predicate):
                prompt_args = UpdateActionStringPromptArgs(
                    action_string=operator,
                    predicates_in_action=str(parser.get_predicates_in_action(predicates, operator)),
                    old_predicate=old_predicate,
                    new_predicate=new_predicate,
                )
                new_action_string = _update_action_string(llm_agent, prompt_args)
            else:
                new_action_string = operator.replace(old_predicate, new_predicate)

            # Remove old operators because the action name itself might change
            if operator_name in operators:
                del operators[operator_name]

            # Update operators
            new_action_name = parser.get_action_name(new_action_string)
            operators[new_action_name] = new_action_string


def merge_predicates(
    domain_1_path: Paths,
    domain_2_path: Paths,
    llm_agent: LLMAgent,
    threshold: float
) -> Tuple[Domain, Domain]:
    """Merge predicates between two domains and updates operators accordingly.
    
    This is a core fusion step. It loads two domains, identifies equivalent predicates
    based on semantic similarity and LLM verification, and unifies them.

    Args:
        domain_1_path (Paths): File paths for the first domain.
        domain_2_path (Paths): File paths for the second domain.
        llm_agent (LLMAgent): The LLM agent service.
        threshold (float): Similarity threshold (0.0 to 1.0) for considering merges.

    Returns:
        Tuple[Domain, Domain]: A tuple containing the two modified domains.
    """

    execution_logger = get_task_logger(llm_agent.log_dir)

    # Load domains
    predicates_1, operators_1 = load_domain(domain_1_path)
    predicates_2, operators_2 = load_domain(domain_2_path)

    # Tracking merge history: [old_p1, old_p2, new_p_name, new_p_def]
    replaced_predicates_info = []
    merged_status = {}  # Cache to prevent double merging

    # Merge predicates loop
    for predicate_1 in predicates_1.keys():

        # Pre-filtering and sort based on semantic similarity
        candidate_predicates = filter_and_sort_strings(predicate_1, list(predicates_2.keys()), threshold)
        
        # Recording filtering messages
        filtering_messages = []
        for predicate_2 in predicates_2.keys():
            if predicate_2 not in candidate_predicates:
                filtering_messages.append(
                    f"similarity between {predicate_1} and {predicate_2} is less than threshold {threshold}"
                )
        execution_logger.info("\n".join(filtering_messages))

        for predicate_2 in candidate_predicates:
            # One predicate can only be merged once
            if predicate_1 in merged_status or predicate_2 in merged_status:
                continue

            # Merge predicates
            predicate_in_domain_1 = f"{predicate_1} ; {predicates_1[predicate_1]}"
            predicate_in_domain_2 = f"{predicate_2} ; {predicates_2[predicate_2]}"
            prompt_args = CheckMergePredicatesPromptArgs(
                predicate_in_domain_1=predicate_in_domain_1,
                predicate_in_domain_2=predicate_in_domain_2
            )
            merge_flag, new_predicate = _check_merge_predicates(llm_agent, prompt_args)
            
            # record merging history
            if merge_flag:
                merged_status[predicate_1] = True
                merged_status[predicate_2] = True
                replaced_predicates_info.append([
                    predicate_1, predicate_2, new_predicate[0], new_predicate[1]
                ])
                break   # One predicate can only be merged once
    
    # update predicates and operators
    for info in replaced_predicates_info:
        old_p1, old_p2, new_p_name, new_p_def = info

        # predicates_1
        del predicates_1[old_p1]
        predicates_1[new_p_name] = new_p_def
    
        # predicates_2
        del predicates_2[old_p2]
        predicates_2[old_p2] = new_p_def

        # operators_1
        _update_domain_operators(operators_1, predicates_1, old_p1, new_p_name, llm_agent)

        # operators_2
        _update_domain_operators(operators_2, predicates_2, old_p2, new_p_name, llm_agent)
    
    # return
    merged_domain_1 = (predicates_1, operators_1)
    merged_domain_2 = (predicates_2, operators_2)
    return merged_domain_1, merged_domain_2
