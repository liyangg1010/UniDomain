"""Initial domain generation for HD (Human Demonstration) data.

For HD data from narrations, frames may be far apart temporally.
The original UniDomain approach of incremental operator learning frame-by-frame
is not suitable. This module provides adapted functions for HD data characteristics
where temporal continuity may be broken between frames.
"""

from pathlib import Path
from typing import Dict

from unidomain.atomic_domain.initial_domain import _learn_incremental_domain_from_keyframes
from unidomain.atomic_domain.post_process import post_process_domain
from unidomain.configs.constants import File
from unidomain.pddl.io import save_domain
from unidomain.schemas.prompt_args import InitialDomainPromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = [
    "initiate_domain_from_keyframes_for_hd",
    "run_initial_domain_step_for_hd",
]


def initiate_domain_from_keyframes_for_hd(
    llm_agent: LLMAgent,
    tasks: Dict,
    save_domain_path: Paths,
) -> None:
    """Iterate through keyframes to learn the initial domain from scratch for HD data.

    Unlike the original function, this version handles multiple tasks where each task
    contains its own keyframes path and instruction. Frames within each task may have
    temporal gaps, so incremental learning is applied within each task.

    Args:
        llm_agent: The VLM agent service
        tasks: Dict mapping task IDs to task info containing:
            - 'path': Directory with keyframe images
            - 'instruction': Task description
        save_domain_path: Path to save the incrementally learned JSON domain

    Returns:
        None
    """
    available_predicates = {}
    available_operators = {}
    execution_logger = get_task_logger(llm_agent.log_dir)

    for task_id, task_info in tasks.items():
        keyframes_path = task_info['path']
        task_description = task_info['instruction']

        # Load and filter keyframes (ignore hidden files like .DS_Store)
        keyframes_dir = setup_path(keyframes_path, mkdir=False)
        keyframes = sorted([
            p for p in keyframes_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ])

        if len(keyframes) < 2:
            execution_logger.warning(
                f"Task {task_id}: Not enough keyframes to learn transitions "
                f"(need at least 2, got {len(keyframes)}). Skipping."
            )
            continue

        # Incremental learning loop
        # Iterate from the second frame to compare with the previous frame
        for i in range(1, len(keyframes)):
            execution_logger.info(
                f"Task {task_id} Step {i}/{len(keyframes)-1}: "
                f"Learning transition from frame {i-1} to {i}"
            )

            # Get images
            image_path_1 = keyframes_dir / keyframes[i - 1]
            image_path_2 = keyframes_dir / keyframes[i]

            # Update domain incrementally from two keyframes
            prompt_args = InitialDomainPromptArgs(
                task_description=task_description,
                available_predicates=str(available_predicates),
                available_operators=str(available_operators)
            )
            invented_predicates, new_action_name, new_action = _learn_incremental_domain_from_keyframes(
                image_path_list=[image_path_1, image_path_2],
                llm_agent=llm_agent,
                prompt_args=prompt_args
            )

            # Update available predicates
            for predicate, explanation in invented_predicates:
                if predicate:
                    available_predicates[predicate] = explanation

            # Update available operators
            if new_action_name:
                available_operators[new_action_name] = new_action

    save_domain(available_predicates, available_operators, save_domain_path)


def run_initial_domain_step_for_hd(
    llm_agent: LLMAgent,
    tasks: Dict,
    output_dir: Paths
) -> None:
    """Execute the initial domain generation step for HD data.

    This wrapper sets up paths, invokes the learning loop for multiple tasks,
    and performs final post-processing (cleanup and PDDL generation).

    Args:
        llm_agent: The VLM agent service
        tasks: Dict mapping task IDs to task info (see initiate_domain_from_keyframes_for_hd)
        output_dir: Directory to save output files

    Returns:
        None
    """
    output_dir = setup_path(output_dir)
    json_path = output_dir / File.INITIAL_DOMAIN_JSON
    pddl_path = output_dir / File.INITIAL_DOMAIN_PDDL

    # Core Logic: Learn from keyframes across multiple tasks
    initiate_domain_from_keyframes_for_hd(
        llm_agent=llm_agent,
        tasks=tasks,
        save_domain_path=json_path
    )

    # Post-process: Clean up unused predicates and format PDDL
    post_process_domain(json_path, pddl_path)
