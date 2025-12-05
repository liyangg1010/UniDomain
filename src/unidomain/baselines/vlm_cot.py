"""
Baseline: VLM-CoT (Vision-Language Model with Chain-of-Thought)
===============================================================

This module implements a naive baseline that directly prompts a VLM to generate
a plan (sequence of actions) from visual observations using Chain-of-Thought (CoT)
reasoning.

Unlike UniDomain, this method does not produce intermediate symbolic representations
(PDDL) or perform search-based planning. It relies entirely on the internal
reasoning capabilities of the VLM.

Architecture Position:
    [Baselines] -> A direct end-to-end planning baseline.
"""


from pathlib import Path
from typing import Dict, List

from unidomain.configs.constants import File, JSONKey
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import VLMCoTOutputs
from unidomain.schemas.prompt_args import VLMCoTPromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["run_vlm_cot", "run_vlm_cot_batch"]


def _generate_plan(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: VLMCoTPromptArgs,
) -> List[str]:
    """Call VLM to directly generate a sequence of actions (plan).
    
    Args:
        image_path_list (List[Paths]): List of observation images (usually just one).
        llm_agent (LLMAgent): The VLM agent service.
        prompt_args (VLMCoTPromptArgs): Context arguments including instructions.
        
    Returns:
        List[str]: A list of action strings representing the plan.
    """
    
    # Call VLM and parse JSON into specified format
    outputs: VLMCoTOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.vlm_cot,
        prompt_args=prompt_args,
        validator=VLMCoTOutputs,
        image_path_list=image_path_list
    )

    # Get plans
    plans = outputs.plan_sequence
    return plans


def run_vlm_cot(
    image_path: Paths,
    instruction: str,
    save_dir: Paths
) -> bool:
    """Execute the VLM-CoT baseline for a single task.
    
    Args:
        image_path (Paths): Path to the initial observation image.
        instruction (str): Natural language task instruction.
        save_dir (Paths): Directory to save results and metrics.
        
    Returns:
        bool: Always returns True if execution finishes without exception.
    """
    
    save_dir = setup_path(save_dir)
    image_path = setup_path(image_path, is_file=True, mkdir=False)
    execution_logger = get_task_logger(save_dir)
    llm_agent = LLMAgent(config.baselines.llm_model_config, save_dir)

    execution_logger.info(
        "Runing Baseline: VLM-CoT "
        f"Task Path: {image_path} "
        f"Instruction: {instruction}"
    )

    # Inference
    prompt_args = VLMCoTPromptArgs(instructions=instruction)
    plans = _generate_plan([image_path], llm_agent, prompt_args)
    
    # Save Results
    save_path = save_dir / File.FINAL_RESULTS
    save_path.write_text("\n".join(plans), encoding="utf-8")

    llm_agent.write_record(save_dir / File.FINAL_METRICS)
    return True

def run_vlm_cot_batch(
    task_data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1,
) -> None:
    """Execute VLM-CoT in batch mode."""
    tasks = load_batch_inputs(task_data_path)

    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return run_vlm_cot(
            image_path=info[JSONKey.PATH],
            instruction=info[JSONKey.INSTRUCTION],
            save_dir=sub_dir,
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
