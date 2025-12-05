"""
Initial Domain Generation Step
==============================

This module implements the first stage of the atomic domain generation pipeline.
It utilizes a Vision-Language Model (VLM) to incrementally learn PDDL predicates
and operators by observing transitions between consecutive keyframes.

Architecture Position:
    [Functional Steps / Middle Level] -> Called by atomic_domain.pipeline
    Uses post_process.py for cleanup.
"""


from typing import List, Tuple

from unidomain.atomic_domain.post_process import post_process_domain
from unidomain.configs.constants import File
from unidomain.configs.settings import config
from unidomain.pddl.io import save_domain
from unidomain.schemas.llm_task_outputs import InitialDomainOutputs
from unidomain.schemas.prompt_args import InitialDomainPromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["run_initial_domain_step"]


def _learn_incremental_domain_from_keyframes(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: InitialDomainPromptArgs,
) -> Tuple[List[Tuple[str, str]], str, str]:
    """Generate incremental domain updates from two keyframes using VLM.

    This function asks the VLM to identify what changed between two images and
    formulate corresponding PDDL predicates and actions.
    
    Args:
        image_path_list (List[Paths]): A list containing paths to the two keyframes.
        llm_agent (LLMAgent): The VLM agent service.
        prompt_args (InitialDomainPromptArgs): Arguments for VLM prompt.

    Returns:
        Tuple[List[Tuple[str, str]], str, str]: A tuple containing:
            - A list of invented predicates. [(predicate_definition, explanation), ...].
            - The name of the learned action.
            - The PDDL definition of the learned action.
    """

    # Call VLM and parse JSON into specified format
    outputs: InitialDomainOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.initial_domain,
        prompt_args=prompt_args,
        validator=InitialDomainOutputs,
        image_path_list=image_path_list,
    )

    # Return necessary information
    invented_predicates = outputs.invented_predicates
    new_action_name = outputs.PDDL_action_name
    new_action = outputs.PDDL_action
    return invented_predicates, new_action_name, new_action


def _initiate_domain_from_keyframes(
    llm_agent: LLMAgent,
    keyframes_dir: Paths,
    task_description: str,
    save_domain_path: Paths,
) -> None:
    """Iterate through keyframes to learn the initial domain from scratch.
    
    Args:
        llm_agent (LLMAgent): The VLM agent service.
        keyframes_dir (Paths): Directory containing sequential keyframe images.
        task_description (str): High-level description of the task.
        save_domain_path (Paths): Path to save the incrementally learned JSON domain.
    
    Returns:
        None
    """

    available_predicates = {}
    available_operators = {}
    execution_logger = get_task_logger(llm_agent.log_dir)

    # Load and filter keyframes like .DS_Store
    keyframes_dir = setup_path(keyframes_dir, mkdir=False)
    keyframes = sorted([p for p in keyframes_dir.iterdir() if p.is_file() and not p.name.startswith(".")])

    if len(keyframes) < 2:
        execution_logger.warning("Not enough keyframes to learn transitions (need at least 2).")
        return

    # Incremental learning loop
    # Iterate from the second frame to compare with the previous frame
    for i in range(1, len(keyframes)):
        execution_logger.info(f"Step {i}/{len(keyframes)-1}: Learning transition from frame {i-1} to {i}")

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


def run_initial_domain_step(
    llm_agent: LLMAgent,
    keyframes_dir: Paths,
    instructions: str,
    output_dir: Paths
) -> None:
    """Execute the initial domain generation step.

    This wrapper sets up paths, invokes the learning loop, and performs
    final post-processing (cleanup and PDDL generation).

    Args:
        llm_agent (LLMAgent): The VLM agent service.
        keyframes_dir (Paths): Directory containing keyframes.
        instructions (str): The task instructions.
        output_dir (Paths): Directory to save output files.
    """

    output_dir = setup_path(output_dir)
    json_path = output_dir / File.INITIAL_DOMAIN_JSON
    pddl_path = output_dir / File.INITIAL_DOMAIN_PDDL

    # Core Logic: Learn from keyframes
    _initiate_domain_from_keyframes(
        llm_agent=llm_agent,
        keyframes_dir=keyframes_dir,
        task_description=instructions,
        save_domain_path=json_path
    )
    
    # Post-process: Clean up unused predicates and format PDDL
    post_process_domain(json_path, pddl_path)
