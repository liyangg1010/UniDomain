"""Predicate merging for HD domain fusion with conflict resolution.

Enhanced predicate merging that handles name conflicts by adding hash suffixes
when predicates have the same name but different semantics.
"""

import hashlib
from typing import List, Optional, Tuple

from unidomain.configs.settings import config
from unidomain.domain_fusion.predicate_merging import (
    _check_merge_predicates,
    _update_domain_operators
)
from unidomain.domain_fusion.similarity import filter_and_sort_strings
from unidomain.pddl import parser
from unidomain.pddl.io import load_domain
from unidomain.schemas.prompt_args import CheckMergePredicatesPromptArgs
from unidomain.schemas.typings import Domain, Operators, Paths, Predicates
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.logger import get_task_logger

__all__ = ["merge_predicates_hd"]


def merge_predicates_hd(
    domain_1_path: Paths,
    domain_2_path: Paths,
    llm_agent: LLMAgent,
    threshold: float
) -> Tuple[Domain, Domain]:
    """Merge predicates between two domains with name conflict resolution.

    This enhanced version adds hash suffixes to predicates when they have
    the same name but different semantics (fail to merge but have identical names).

    This is a core fusion step. It loads two domains, identifies equivalent predicates
    based on semantic similarity and LLM verification, and unifies them.

    Args:
        domain_1_path: File path for the first domain
        domain_2_path: File path for the second domain
        llm_agent: The LLM agent service
        threshold: Similarity threshold (0.0 to 1.0) for considering merges

    Returns:
        Tuple[Domain, Domain]: A tuple containing the two modified domains
            Each domain is a tuple of (predicates, operators)

    Name Conflict Handling:
        When two predicates have the same name but fail to merge (different semantics),
        the predicate from domain 2 is renamed by appending a 3-character hash suffix.
        Example: "on(?x, ?y)" in domain 2 might become "on_a3f(?x, ?y)"
    """
    execution_logger = get_task_logger(llm_agent.log_dir)

    # Load domains
    predicates_1, operators_1 = load_domain(domain_1_path)
    predicates_2, operators_2 = load_domain(domain_2_path)

    # Tracking merge history: [old_p1, old_p2, new_p_name, new_p_def]
    replaced_predicates_info = []
    merged_status = {}  # Cache to prevent double merging

    # Track predicate name conflicts for renaming
    # Format: [domain_id, old_predicate, old_predicate_name, new_predicate_name]
    rename_info = []

    # Merge predicates loop
    for predicate_1 in predicates_1.keys():
        # Pre-filtering and sort based on semantic similarity
        candidate_predicates = filter_and_sort_strings(
            predicate_1,
            list(predicates_2.keys()),
            threshold
        )

        # Recording filtering messages
        filtering_messages = []
        for predicate_2 in predicates_2.keys():
            if predicate_2 not in candidate_predicates:
                filtering_messages.append(
                    f"similarity between {predicate_1} and {predicate_2} "
                    f"is less than threshold {threshold}"
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

            # Record merging history
            if merge_flag:
                merged_status[predicate_1] = True
                merged_status[predicate_2] = True
                replaced_predicates_info.append([
                    predicate_1, predicate_2, new_predicate[0], new_predicate[1]
                ])
                execution_logger.info(
                    f"Merged: {predicate_1} and {predicate_2} -> {new_predicate[0]}"
                )
                break  # One predicate can only be merged once

            # Handle name conflict: same name but failed to merge
            elif parser.get_predicate_name(predicate_1) == parser.get_predicate_name(predicate_2):
                predicate_name = parser.get_predicate_name(predicate_1)

                # Generate unique name with hash suffix
                new_predicate_name = (
                    f"{predicate_name}_{hashlib.md5(predicate_2.encode()).hexdigest()[:3]}"
                )

                rename_info.append([2, predicate_2, predicate_name, new_predicate_name])

    # Update same-name predicates to resolve conflicts
    for _, old_predicate, old_predicate_name, new_predicate_name in rename_info:
        old_predicate_key = old_predicate
        new_predicate_key = old_predicate.replace(old_predicate_name, new_predicate_name)

        # Update predicate in domain 2
        predicates_2[new_predicate_key] = predicates_2[old_predicate_key]
        predicates_2.pop(old_predicate_key)

        # Update operators in domain 2
        _update_domain_operators(
            operators_2,
            predicates_2,
            old_predicate_key,
            new_predicate_key,
            llm_agent
        )

    # Update predicates and operators for successful merges
    for info in replaced_predicates_info:
        old_p1, old_p2, new_p_name, new_p_def = info

        # Update predicates_1
        del predicates_1[old_p1]
        predicates_1[new_p_name] = new_p_def

        # Update predicates_2
        del predicates_2[old_p2]
        predicates_2[new_p_name] = new_p_def

        # Update operators_1
        _update_domain_operators(operators_1, predicates_1, old_p1, new_p_name, llm_agent)

        # Update operators_2
        _update_domain_operators(operators_2, predicates_2, old_p2, new_p_name, llm_agent)

    # Return merged domains
    merged_domain_1 = (predicates_1, operators_1)
    merged_domain_2 = (predicates_2, operators_2)
    return merged_domain_1, merged_domain_2
