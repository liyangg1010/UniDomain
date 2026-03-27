"""Atomic domain pipeline for HD (Human Demonstration) data.

Modified pipeline for datasets where frames are extracted from narrations
and may have temporal gaps, making the original frame-by-frame operator learning unsuitable.
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Dict

from unidomain.atomic_domain.revise_domain import run_holistic_revision_step
from unidomain.atomic_domain.solution_verification import (
    run_verification_refinement_step,
    run_verification_step
)
from unidomain.atomic_domain.solvability_check import (
    run_solvability_check_step,
    run_solvability_refinement_step
)
from unidomain.configs.constants import Dir, File
from unidomain.configs.settings import config
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.batch_runner import load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

from .initial_domain_hd import run_initial_domain_step_for_hd

__all__ = ["atomic_domain_pipeline_for_hd"]


def _suppress_submodule_logs():
    """Context manager to temporarily suppress logs from sub-modules.

    Raises the execution logger level to CRITICAL to suppress INFO/DEBUG logs
    from sub-modules while keeping current function logs visible.
    """
    from unidomain.utils.logger import execution_logger as root_execution_logger

    prev_level = root_execution_logger.level
    try:
        # Raise level to CRITICAL to suppress INFO/DEBUG logs from sub-modules
        root_execution_logger.setLevel(logging.CRITICAL)
        yield
    finally:
        # Restore original level
        root_execution_logger.setLevel(prev_level)


def _update_current_domain(source_dir: Path, root_dir: Path, source_name: str) -> None:
    """Update the 'current' atomic domain pointer to the latest result.

    Copies domain files from a step's output directory to the root directory,
    making them the "current" best domain for subsequent steps.

    Args:
        source_dir: Directory containing the domain files to promote
        root_dir: Root directory where current domain pointers live
        source_name: Base name for the source files (without extension)

    Returns:
        None
    """
    import shutil

    # Source files
    src_json = source_dir / f"{source_name}.json"
    src_pddl = source_dir / f"{source_name}.pddl"

    # Target files
    dest_json = root_dir / File.ATOMIC_DOMAIN_JSON
    dest_pddl = root_dir / File.ATOMIC_DOMAIN_PDDL

    # Copy files if they exist
    if src_json.exists():
        shutil.copy(src_json, dest_json)
    if src_pddl.exists():
        shutil.copy(src_pddl, dest_pddl)


def atomic_domain_pipeline_for_hd(
    data_path: Path,
    save_dir: Paths,
    vlm_model: str,
    llm_model: str,
    log_enabled: bool = True,
    vlm_client = None,
    llm_client = None,
) -> bool:
    """Generate atomic domain from HD narration data with temporal gaps between frames.

    For datasets where frames are extracted from narrations, frames may be far apart
    temporally. The original UniDomain incremental frame-by-frame learning is not
    suitable. This pipeline adapts the process for such data characteristics.

    Args:
        data_path: Path to batch inputs (JSON mapping task IDs to {path, instruction})
        save_dir: Directory to save all outputs
        vlm_model: VLM model name for initial domain extraction
        llm_model: LLM model name for revision and verification
        log_enabled: Whether to show detailed logs from sub-modules
        vlm_client: Optional pre-configured VLM client (bypasses environment variable requirement)
        llm_client: Optional pre-configured LLM client (bypasses environment variable requirement)

    Returns:
        bool: True if domain generation succeeded, False otherwise

    File Structure:
        save_dir/
            atomic_domain.json          # Current best domain (JSON)
            atomic_domain.pddl          # Current best domain (PDDL)
            0_initial_domain/
                initial_domain.json
                initial_domain.pddl
            1_revised_domain/
                revised_domain.json
                revised_domain.pddl
            0_check/                    # Solvability check iteration 0
                ...
            0_verify/                   # Solution verification iteration 0
                ...
            success.log / fail.log      # Final status marker
    """
    # Load tasks
    tasks = load_batch_inputs(data_path)
    instructions = '. '.join([v["instruction"] for _, v in tasks.items()])

    save_dir = setup_path(save_dir)
    execution_logger = get_task_logger(save_dir)

    # Initialize LLM agents
    # If custom clients are provided, use them directly to bypass environment variable checks
    if vlm_client is not None and llm_client is not None:
        # Temporarily set a dummy API_KEY to bypass environment variable check in LLMAgent.__init__
        # The actual client will be overridden immediately after initialization
        original_api_key = os.environ.get('API_KEY')
        os.environ['API_KEY'] = 'dummy_key_will_be_replaced'

        try:
            from unidomain.configs.llm import LLMAgentConfig
            vlm_agent = LLMAgent(LLMAgentConfig(model=vlm_model, kwargs={"temperature": 0.0}), log_dir=save_dir)
            llm_agent = LLMAgent(LLMAgentConfig(model=llm_model), log_dir=save_dir)
            # Override the clients that were initialized during __init__
            vlm_agent._client = vlm_client
            llm_agent._client = llm_client
        finally:
            # Restore original environment variable state
            if original_api_key is None:
                os.environ.pop('API_KEY', None)
            else:
                os.environ['API_KEY'] = original_api_key
    else:
        # Use default initialization from config (requires environment variables)
        # Note: You may need to customize the agent initialization based on your setup
        # For example, using custom API keys or base URLs via environment variables
        vlm_agent = LLMAgent(config.atomic_domain.vlm_model_config, log_dir=save_dir)
        llm_agent = LLMAgent(config.atomic_domain.llm_model_config, log_dir=save_dir)

    # Pointers to the current "best" domain in the root directory
    current_json = save_dir / File.ATOMIC_DOMAIN_JSON
    current_pddl = save_dir / File.ATOMIC_DOMAIN_PDDL

    # Step 1: Call VLM to learn initial domain from keyframes
    execution_logger.info(f"Call {vlm_agent.model} to learn initial domain from {data_path}...")
    step_dir = save_dir / Dir.INITIAL_DOMAIN
    with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
        run_initial_domain_step_for_hd(vlm_agent, tasks, step_dir)
    _update_current_domain(step_dir, save_dir, source_name=File.INITIAL_DOMAIN_STEM)

    # Step 2: Revise initial domain holistically
    execution_logger.info(f"Call {llm_agent.model} to revise initial domain holistically...")
    step_dir = save_dir / Dir.REVISED_DOMAIN
    with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
        run_holistic_revision_step(
            llm_agent,
            current_pddl.read_text(encoding="utf-8"),
            instructions,
            step_dir
        )
    _update_current_domain(step_dir, save_dir, source_name=File.REVISED_DOMAIN_STEM)

    # Step 3: Loop for solvability check and solution verification
    max_attempts = config.atomic_domain.refine_max_attempts
    attempts = 0
    execution_logger.info(f"Entering refinement loop (Max attempts: {max_attempts})...")

    while attempts < max_attempts:
        # Solvability check
        check_dir = save_dir / f"{attempts}{Dir.CHECK_SUFFIX}"
        with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
            success, last_prob, last_info = run_solvability_check_step(
                llm_agent, current_pddl, current_json, instructions, check_dir
            )

        # Refine if solvability check failed
        if not success:
            execution_logger.info("Solvability check failed!")
            if attempts + 1 >= max_attempts:
                break

            execution_logger.info(f"Call {llm_agent.model} to refine from solvability check.")
            with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
                run_solvability_refinement_step(
                    llm_agent,
                    current_pddl.read_text(encoding="utf-8"),
                    last_prob,
                    last_info,
                    check_dir
                )
            _update_current_domain(check_dir, save_dir, source_name=File.REFINED_DOMAIN_STEM)
            attempts += 1
            continue

        # Solution verification if solvability check is successful
        execution_logger.info("Solvability check succeeds. Verifying solution...")
        verify_dir = save_dir / f"{attempts}{Dir.VERIFY_SUFFIX}"
        with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
            is_valid, reasoning = run_verification_step(
                llm_agent,
                current_pddl.read_text(encoding="utf-8"),
                last_prob,
                last_info,
                verify_dir
            )

        if is_valid:
            execution_logger.info(
                "Solution verification succeeded! Atomic domain generation complete!"
            )
            (save_dir / 'success.log').touch()
            return True

        # Refine if solution verification failed
        else:
            execution_logger.info("Solution verification failed!")
            if attempts + 1 >= max_attempts:
                break

            with _suppress_submodule_logs() if not log_enabled else contextlib.nullcontext():
                run_verification_refinement_step(
                    llm_agent,
                    current_pddl.read_text(encoding="utf-8"),
                    reasoning,
                    instructions,
                    verify_dir
                )
            _update_current_domain(verify_dir, save_dir, source_name=File.REFINED_DOMAIN_STEM)

            attempts += 1
            continue

    execution_logger.info(f"Atomic domain generation failed after {max_attempts} attempts!")
    (save_dir / 'fail.log').touch()
    return False
