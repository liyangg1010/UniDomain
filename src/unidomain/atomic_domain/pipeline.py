"""
Atomic Domain Generation Pipeline
=================================

This module orchestrates the complete lifecycle of atomic domain generation.
It coordinates the VLM-based initialization, holistic revision, and the
iterative refinement loop based on solvability checks and semantic verification.

Architecture Position:
    [Orchestrator / Top Level] -> Entry point for the atomic_domain module.
"""

import shutil
from pathlib import Path
from typing import Dict

from unidomain.atomic_domain.initial_domain import run_initial_domain_step
from unidomain.atomic_domain.revise_domain import run_holistic_revision_step
from unidomain.atomic_domain.solution_verification import run_verification_refinement_step, run_verification_step
from unidomain.atomic_domain.solvability_check import run_solvability_check_step, run_solvability_refinement_step
from unidomain.configs.constants import Dir, File
from unidomain.configs.settings import config
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["atomic_domain_pipeline", "atomic_domain_batch_pipeline"]


def _update_current_domain(source_dir: Path, root_dir: Path, source_name: str) -> None:
    """Utility function to update the 'current' atomic domain pointer to the latest result.

    Copies the generated JSON and PDDL files from the step directory to the
    root directory, standardizing their names to 'File.ATOMIC_DOMAIN_STEM.json/pddl'.
    """
    shutil.copy(source_dir / f"{source_name}.json", root_dir / File.ATOMIC_DOMAIN_JSON)
    shutil.copy(source_dir / f"{source_name}.pddl", root_dir / File.ATOMIC_DOMAIN_PDDL)


def atomic_domain_pipeline(
    keyframes_dir: Paths,
    instructions: str,
    save_dir: Paths,
) -> bool:
    """Execute the complete atomic domain generation pipeline.

    The pipeline consists of the following stages:
    1. Initial Domain Learning (VLM): Learn from visual keyframes.
    2. Holistic Revision (LLM): Fix syntax and obvious errors.
    3. Refinement Loop (LLM + Planner):
       - Solvability Check: Verify if test problems can be solved.
       - Solution Verification: Verify if generated plans make semantic sense.
       - Refinement: Fix the domain based on error logs or verification reasoning.

    Args:
        keyframes_dir (Paths): The directory containing keyframes. The order of keyframes is recognized as
            lexicographical order of images' name (str). A recommended name style is like:
                000.jpg, 001.jpg, 002.jpg, ..., or
                0001.jpg, 0127.jpg, 1346.jpg, ...
        instructions (str): the instructions for the atomic domain.
            In robot manipulation datasets like DROID, instructions are generally available.
            Instructions can also be description of the video, or the task.
            Instructions are supplementary to atomic domain generation, improving the quality.
            Instructions can be an empty string (or "None", etc.), if not necessary or not available.
        save_dir (Paths): The directory to store the generated atomic domain, including
            initial domain, revised domain, test problems, refined domain and final atomic domain.
            An empty directory is recommanded.

    Returns:
        bool: True if atomic domain generation is successful; False if it failed after max attempts for check.
    """
    save_dir = setup_path(save_dir)
    execution_logger = get_task_logger(save_dir)

    # vlm_agent for initial domain, llm_agent for other modulars
    vlm_agent = LLMAgent(config.atomic_domain.vlm_model_config, log_dir=save_dir)
    llm_agent = LLMAgent(config.atomic_domain.llm_model_config, log_dir=save_dir)

    # Pointers to the current "best" domain in the root directory
    current_json = save_dir / File.ATOMIC_DOMAIN_JSON
    current_pddl = save_dir / File.ATOMIC_DOMAIN_PDDL

    # Call VLM to learn initial domain from keyframes
    execution_logger.info(f"Call {vlm_agent.model} to learn initial domain from {keyframes_dir}...")
    step_dir = save_dir / Dir.INITIAL_DOMAIN
    run_initial_domain_step(vlm_agent, keyframes_dir, instructions, step_dir)
    _update_current_domain(step_dir, save_dir, source_name=File.INITIAL_DOMAIN_STEM)

    # Revise initial domain holistically
    execution_logger.info(f"Call {llm_agent.model} to revise initial domain holistically...")
    step_dir = save_dir / Dir.REVISED_DOMAIN
    run_holistic_revision_step(llm_agent, current_pddl.read_text(encoding="utf-8"), instructions, step_dir)
    _update_current_domain(step_dir, save_dir, source_name=File.REVISED_DOMAIN_STEM)

    # Loop for solvability check and solution verification
    max_attempts = config.atomic_domain.refine_max_attempts
    attempts = 0
    execution_logger.info(f"Entering refinement loop (Max attempts: {max_attempts})...")
    while attempts < max_attempts:
        
        check_dir = save_dir / f"{attempts}{Dir.CHECK_SUFFIX}"
        success, last_prob, last_info = run_solvability_check_step(
            llm_agent, current_pddl, current_json, instructions, check_dir
        )

        # Refine if solvability check failed
        if not success:
            execution_logger.info("Solvability check failed!")
            if attempts + 1 >= max_attempts:
                break

            execution_logger.info(f"Call {llm_agent.model} to refine from solvability check.")
            run_solvability_refinement_step(
                llm_agent, current_pddl.read_text(encoding="utf-8"), last_prob, last_info, check_dir
            )
            _update_current_domain(check_dir, save_dir, source_name=File.REFINED_DOMAIN_STEM)
            attempts += 1
            continue

        # Solution verification if solvability check is successful
        execution_logger.info("Solvability check succeeds. Verifying solution...")

        verify_dir = save_dir / f"{attempts}{Dir.VERIFY_SUFFIX}"
        is_valid, reasoning = run_verification_step(
            llm_agent, current_pddl.read_text(encoding="utf-8"), last_prob, last_info, verify_dir
        )

        if is_valid:
            execution_logger.info("Solution verification succeeded! Atomic domain generation complete!")
            return True
        
        # Refine if solution verification failed
        else:
            execution_logger.info("Solution verification failed!")
            if attempts + 1 >= max_attempts:
                break

            run_verification_refinement_step(
                llm_agent, current_pddl.read_text(encoding="utf-8"), reasoning, instructions, verify_dir
            )
            _update_current_domain(verify_dir, save_dir, source_name=File.REFINED_DOMAIN_STEM)

            attempts += 1
            continue

    execution_logger.info(f"Atomic domain generation failed after {max_attempts} attempts!")
    return False


def atomic_domain_batch_pipeline(
    data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1
) -> None:
    """Execute atomic domain generation in batch mode.
    
    Args:
        data_path (Paths): The JSON file to the atomic domain generation, in the following format:
            {
                "name_1": {"instruction": "...", "path": "path/to/keyframes_dir"},
                "name_2": {"instruction": "...", "path": "path/to/keyframes_dir"},
                ...
            }
            keyframes directory can be whether absolute path, or relative path to the JSON file's directory.

        save_dir (Paths): The path to the saved directory.
                          After generation, the directory will be structured as follows:
            name_1/
                ...
            name_2/
                ...
            ...
            checklist.json (recording final status of each generation. Reruning can be tried for failed ones)
            {
                "name_1": true, # "name_1" generation succeeded.
                "name_2": false, # "name_2" generation failed.
                "name_3": null, # Error occurred during "name_3" generation, such as network error, etc.
                ...
            }

        num_workers (int, optional): Number of parallel threads to execute all tasks.
    
    Returns:
        None
    """

    # Load tasks
    tasks = load_batch_inputs(data_path)

    # Define Adapter/Closure
    # Adapts the specific arguments of atomic_domain_pipeline to the generic interface
    def _worker(name: str, info: Dict, sub_dir: Path):
        return atomic_domain_pipeline(
            keyframes_dir=info["path"],
            instructions=info["instruction"],
            save_dir=sub_dir
        )

    # Delegate to generic batch runner
    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
