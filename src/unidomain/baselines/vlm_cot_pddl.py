"""
Baseline: VLM-CoT-PDDL
======================

This module implements the "VLM-CoT-PDDL" baseline, which serves as a dual
counterpart to the naive VLM-CoT baseline.

Methodology:
    1.  **Perception & Logic**: Instead of generating a plan directly in natural language,
        the VLM is prompted to generate full PDDL files (Domain and Problem) based
        on the visual observation and instruction.
    2.  **Execution**: These PDDL files are then fed into a symbolic planner (Fast Downward)
        to find a solution.

Contrast:
    - UniDomain uses the same meta domain for every new task, fused from atomic domains.
    - This baseline attempts "One-Shot" PDDL generation from scratch for every new task.

Architecture Position:
    [Baselines] -> A Neuro-Symbolic baseline without pre-training/fusion.
"""


from pathlib import Path
from typing import Dict, List, Tuple

from unidomain.baselines.utils import pddl_planning
from unidomain.configs.constants import File, JSONKey
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import VLMCoTPDDLOutputs
from unidomain.schemas.prompt_args import VLMCoTPDDLPromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["run_vlm_cot_pddl", "run_vlm_cot_pddl_batch"]


def _generate_pddl(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: VLMCoTPDDLPromptArgs,
    n: int
) -> Tuple[str, str]:
    """Call VLM to generate full PDDL Domain and Problem strings.

    Args:
        image_path_list (List[Paths]): List of observation images.
        llm_agent (LLMAgent): The VLM agent service.
        prompt_args (VLMCoTPDDLPromptArgs): Context arguments including instructions.
        n (int): Number of variations to generate (for ensembling).

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - List of generated PDDL Domain strings.
            - List of generated PDDL Problem strings.
    """
    
    # Call VLM and parse the output into JSON format
    outputs: List[VLMCoTPDDLOutputs] = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.vlm_cot_pddl,
        prompt_args=prompt_args,
        validator=VLMCoTPDDLOutputs,
        image_path_list=image_path_list,
        n=n
    )

    # Normalize single output to list for uniform processing
    if not isinstance(outputs, list):
        outputs = [outputs]

    # Extract domains and problems
    domains = [output.domain for output in outputs]
    problems = [output.problem for output in outputs]

    return domains, problems


def run_vlm_cot_pddl(
    image_path: Paths,
    instruction: str,
    save_dir: Paths,
    parallelism: int = 3,
) -> bool:
    """Execute the VLM-CoT-PDDL baseline for a single task.

    Args:
        image_path (Paths): Path to the initial observation image.
        instruction (str): Natural language task instruction.
        save_dir (Paths): Directory to save results and metrics.
        parallelism (int): Number of PDDL candidates to generate per task.
            Defaults to 3.

    Returns:
        bool: Always returns True if execution finishes without exception.
    """
    save_dir = setup_path(save_dir)
    image_path = setup_path(image_path, is_file=True, mkdir=False)

    execution_logger = get_task_logger(save_dir)
    llm_agent = LLMAgent(config.baselines.llm_model_config, save_dir)

    execution_logger.info(
        "Runing Baseline: VLM-CoT-PDDL "
        f"Task Path: {image_path} "
        f"Instruction: {instruction}"
    )

    # Generate PDDL candidates
    prompt_args = VLMCoTPDDLPromptArgs(instructions=instruction)
    domains, problems = _generate_pddl([image_path], llm_agent, prompt_args, parallelism)
    
    # Planning
    pddl_planning(domains, problems, save_dir)
    llm_agent.write_record(save_dir / File.FINAL_METRICS)
    return True


def run_vlm_cot_pddl_batch(
    task_data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1,
    parallelism: int = 3,
) -> None:
    """Execute VLM-CoT-PDDL in batch mode."""
    tasks = load_batch_inputs(task_data_path)

    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return run_vlm_cot_pddl(
            image_path=info[JSONKey.PATH],
            instruction=info[JSONKey.INSTRUCTION],
            save_dir=sub_dir,
            parallelism=parallelism,
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
