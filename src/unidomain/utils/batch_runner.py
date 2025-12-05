"""
Batch Execution Utilities
=========================

This module provides generic infrastructure for executing tasks in batch mode.
It handles input loading (JSON), path resolution, and multi-threaded execution
with standardized error logging and status checkpointing.

Architecture Position:
    [Shared Infrastructure / Utility] -> Generic utility for parallel task execution.
"""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict

from unidomain.configs.constants import File, JSONKey
from unidomain.schemas.typings import BatchCheckList, BatchInputs, Paths
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import register_force_exit, setup_path

__all__ = ["load_batch_inputs", "execute_batch_tasks"]


def load_batch_inputs(data_path: Paths) -> BatchInputs:
    """Load and resolve batch task definitions from a JSON file.

    This function reads a JSON file defining a batch of tasks. It automatically
    resolves relative paths found within the task definitions relative to the
    JSON file's directory, ensuring portability.

    Args:
        data_path (Paths): Path to the JSON configuration file.
            Expected format:
            {
                "task_name": {"path": "relative/or/abs/path", "instruction": "..."},
                ...
            }

    Returns:
        BatchInputs: A dictionary mapping task names to their configuration dicts.

    Raises:
        FileNotFoundError, json.JSONDecodeError: If the data_path does not exist.
        ValueError: If the file is not valid JSON or if a task is missing the 'path' key.
    """
    data_path = setup_path(data_path, is_file=True, mkdir=False)

    # Load JSON configuration
    try:
        with data_path.open("r", encoding="utf-8") as f:
            batch_inputs: Dict[str, Dict[str, str]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to decode JSON from '{data_path}': {e}") from e
    
    # Resolve paths relative to the JSON file location
    json_file_dir = data_path.parent

    for name, input_data in batch_inputs.items():
        # Validate required fields
        if JSONKey.PATH not in input_data:
            raise ValueError(f"Task '{name}' in '{data_path}' is missing the required '{JSONKey.PATH}' field.")

        raw_path = Path(input_data[JSONKey.PATH])
        # If path is relative, make it absolute relative to the JSON file
        if not raw_path.is_absolute():
            input_data[JSONKey.PATH] = (json_file_dir / raw_path).resolve()
        else:
            input_data[JSONKey.PATH] = raw_path.resolve()

    return batch_inputs

def execute_batch_tasks(
    tasks: BatchInputs,
    save_dir: Paths,
    worker_func: Callable[[str, str, Paths], bool],
    num_workers: int,
) -> None:
    """Execute a set of tasks in parallel using a thread pool.

    This generic engine handles directory creation, task submission, result
    collection, and error logging. It produces a checklist JSON file summarizing
    the status (Success/Fail/Error) of all tasks.

    Resume Logic:
        If a checklist file exists, tasks marked as True (Success) in the checklist
        will be skipped. Tasks marked as False (Fail) or None (Error), or tasks not
        present in the checklist, will be executed.

    Args:
        tasks (BatchInputs): Dictionary of task definitions (name -> info).
        save_dir (Paths): Root directory to save outputs for all tasks.
        worker_func (Callable): The adapter function to execute a single task.
            Signature: (task_name: str, task_info: Dict, sub_save_dir: Path) -> bool
        num_workers (int): Number of parallel threads.

    Return:
        None
    """

    register_force_exit()
    save_dir = setup_path(save_dir)
    
    error_path = save_dir / File.ERROR_LOG_NAME
    checklist_path = save_dir / File.CHECKLIST_NAME
    execution_logger = get_task_logger(save_dir)

    # Try to resume from existing checklist
    checklist: BatchCheckList = {}
    if checklist_path.exists():
        try:
            with checklist_path.open("r", encoding="utf-8") as f:
                checklist: BatchCheckList = json.load(f)
            execution_logger.info(f"Found existing checklist at {checklist_path}. Checking for resume...")
        except json.JSONDecodeError:
            execution_logger.warning("Existing checklist is corrupted. Starting fresh.")

    tasks_to_run = {}
    skipped_count = 0

    for name, info in tasks.items():
        # If task exists in checklist AND was successful (True), skip it
        if checklist.get(name) is True:
            skipped_count += 1
            continue
        tasks_to_run[name] = info

    if skipped_count > 0:
        execution_logger.info(f"Skipping {skipped_count} already successful tasks.")

    if not tasks_to_run:
        execution_logger.info("All tasks are already completed successfully. Nothing to do.")
        return
    
    execution_logger.info(f"Starting execution for {len(tasks_to_run)} tasks with {num_workers} workers.")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        for name, info in tasks_to_run.items():
            # Prepare sub-directory for the specific task
            sub_save_dir = setup_path(save_dir / name)

            # Submit task
            future = executor.submit(worker_func, name, info, sub_save_dir)
            futures[future] = name

        execution_logger.info("\n--- Tasks submitted. Waiting for completion... ---")
        
        # Collect results as they complete
        for future in as_completed(futures):
            name = futures[future]
            try:
                # Get the result from the worker function (bool)
                result = future.result()
                checklist[name] = result

                status_str = "Success" if result else "Failed"
                execution_logger.info(f"Task '{name}' finished: {status_str}")

            except Exception as e:
                # Catch unhandled exceptions in the worker thread
                checklist[name] = None  # None indicates System Error / Exception

                error_msg = f"Task: {name}\n{traceback.format_exc()}\n{'='*40}\n"
                execution_logger.error(f"Error in task {name}: {e}")

                # Write to error log
                with error_path.open("a", encoding="utf-8") as f:
                    f.write(error_msg)

            # Save intermediate checklist, in case of crash
            try:
                with checklist_path.open("w", encoding="utf-8") as f:
                    json.dump(checklist, f, ensure_ascii=False, indent=4, sort_keys=True)
            except Exception as e:
                execution_logger.error(f"Failed to save checklist to disk: {e}")

    # Save final checklist
    with checklist_path.open("w", encoding="utf-8") as f:
        json.dump(checklist, f, ensure_ascii=False, indent=4, sort_keys=True)
    
    execution_logger.info(f"\n--- Batch tasks finished. Check {checklist_path} for results. ---")
