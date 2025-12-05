"""
Baseline: IVML (Instance Verbalized Machine Learning)
=====================================================

This module implements an adapted version of the IVML algorithm for PDDL generation.
**Reference:** "Generating Symbolic World Models via Test-time Scaling of Large Language Models".

**Algorithm Overview:**
1.  **Initialization**: Sample N candidate domains from the VLM.
2.  **Best-of-N (BoN)**: Select the best candidate domain.
    - *Original Paper*: Uses logits/probabilities from open-source models.
    - *Adaptation*: Uses a VLM to evaluate and vote for the best domain index,
      as we rely on closed-source APIs (OpenAI).
3.  **Iterative Optimization**:
    - **f_opt**: Critic model provides feedback on the current domain.
    - **f_update**: Actor model refines the domain based on feedback.
4.  **Planning**: (Adapted) Generate PDDL problems using the final domain and execute the plan.

Architecture Position:
    [Baselines] -> A sophisticated iterative baseline for domain generation.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from unidomain.baselines.utils import pddl_planning
from unidomain.configs.constants import File, JSONKey
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import (
    IVMLBoNOutputs,
    IVMLInitializationOutputs,
    IVMLOptOutputs,
    IVMLProblemOutputs,
    IVMLUpdateOutputs,
)
from unidomain.schemas.prompt_args import (
    IVMLBoNPromptArgs,
    IVMLInitializationPromptArgs,
    IVMLOptPromptArgs,
    IVMLProblemPromptArgs,
    IVMLUpdatePromptArgs,
)
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["run_ivml", "run_ivml_batch"]


def _generate_problem(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: IVMLProblemPromptArgs,
    n: int
) -> List[str]:
    """Generate PDDL problem definitions from the visual task."""

    # Call VLM and parse the output into JSON format
    outputs: List[IVMLProblemOutputs] = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.ivml_problem,
        prompt_args=prompt_args,
        validator=IVMLProblemOutputs,
        image_path_list=image_path_list,
        n=n
    )

    if n == 1:
        outputs = [outputs]

    problems = [output.problem for output in outputs]
    return problems


def _initialization(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: IVMLInitializationPromptArgs,
    n: int = 3
) -> List[str]:
    """Step 1: Sample N initial candidate domains."""

    # Call VLM and parse JSON into specified format
    outputs: List[IVMLInitializationOutputs] = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.ivml_initialization,
        prompt_args=prompt_args,
        validator=IVMLInitializationOutputs,
        image_path_list=image_path_list,
        n=n
    )

    # Get initial domains
    domains = [output.domain for output in outputs]
    return domains

def _best_of_N(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: IVMLBoNPromptArgs,
    n: int = 3
) -> int:
    """Step 2: Select the best domain index using VLM voting."""

    # Call VLM and parse JSON into specified format
    outputs: List[IVMLBoNOutputs] = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.ivml_BoN,
        prompt_args=prompt_args,
        validator=IVMLBoNOutputs,
        image_path_list=image_path_list,
        n=n
    )
    
    # get output
    votes = [output.domain_index for output in outputs]

    counter = Counter(votes)
    max_count = max(counter.values())
    modes = [item for item, count in counter.items() if count == max_count]

    return random.choice(modes)

def _f_opt(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: IVMLOptPromptArgs,
) -> str:
    """Step 3a: Critic (f_opt) provides feedback on the domain."""

    # Call VLM and parse JSON into specified format
    outputs: IVMLOptOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.ivml_opt,
        prompt_args=prompt_args,
        validator=IVMLOptOutputs,
        image_path_list=image_path_list
    )

    # Get feedback
    feedback = outputs.feedback

    return feedback


def _f_update(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: IVMLUpdatePromptArgs,
) -> str:
    """Step 3b: Actor (f_update) refines the domain based on feedback."""

    # Call VLM and parse JSON into specified format
    outputs: IVMLUpdateOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.ivml_update,
        prompt_args=prompt_args,
        validator=IVMLUpdateOutputs,
        image_path_list=image_path_list
    )

    # Get thought and domain
    thought = outputs.thought
    domain = outputs.domain

    return thought, domain


def run_ivml(
    image_path: Paths,
    instruction: str,
    save_dir: Paths,
    N: int = 8,
    choosen_parallelism: int = 3,
    iteration: int = 5,
    problem_parallelism: int = 3,
) -> bool:
    """Execute the full IVML pipeline for a single task.

    Args:
        image_path (Paths): Path to visual observation.
        instruction (str): Task instruction.
        save_dir (Paths): Directory to save outputs.
        N (int): Number of initial domains to sample. Defaults to 8.
        choosen_parallelism (int): Number of votes for Best-of-N selection. Defaults to 3.
        iteration (int): Number of optimization steps (f_opt <-> f_update). Defaults to 5.
        problem_parallelism (int): Number of PDDL problems to generate for final planning.

    Returns:
        bool: Always returns True if execution finishes.
    """
    
    save_dir = setup_path(save_dir)
    image_path = setup_path(image_path, is_file=True, mkdir=False)
    execution_logger = get_task_logger(save_dir)
    llm_agent = LLMAgent(config.baselines.llm_model_config, save_dir)

    execution_logger.info(
        "Runing Baseline: IVML "
        f"Task Path: {image_path} "
        f"Instruction: {instruction}"
    )

    # Initialization
    execution_logger.info(f"Step 1: Initialization (Sampling {N} domains)...")
    prompt_args = IVMLInitializationPromptArgs(instructions=instruction)
    candidate_domains = _initialization([image_path], llm_agent, prompt_args, N)

    # Prepare context for BoN
    # Construct a string listing all candidates: "Index: 0 \n content... \n Index: 1 ..."
    domains_context = ""
    for i, candidate in enumerate(candidate_domains):
        domains_context += f"Domain Index: {i}\n{candidate}\n\n"

    # Best-of-N Selection
    execution_logger.info("Step 2: Best-of-N Selection...")
    prompt_args = IVMLBoNPromptArgs(instructions=instruction, domains=domains_context)
    best_domain_index = _best_of_N([image_path], llm_agent, prompt_args, choosen_parallelism)
    domain = candidate_domains[int(best_domain_index)]

    # Iterative Optimization
    execution_logger.info(f"Step 3: Optimization Loop ({iteration} iterations)...")
    thought = "Initial thought: None"
    for i in range(iteration):
        prompt_args = IVMLOptPromptArgs(instructions=instruction, thought=thought, domain=domain)
        feedback = _f_opt([image_path], llm_agent, prompt_args)

        prompt_args = IVMLUpdatePromptArgs(instructions=instruction, domain=domain, feedback=feedback, thought=thought)
        thought, domain = _f_update([image_path], llm_agent, prompt_args)

        execution_logger.info(f"Iteration {i+1}/{iteration} complete.")

    # Generate Problems & Plan
    execution_logger.info("Step 4: Problem Generation & Planning...")
    prompt_args = IVMLProblemPromptArgs(instructions=instruction, domain=domain)
    problems = _generate_problem([image_path], llm_agent, prompt_args, problem_parallelism)

    domains = [domain] * len(problems)
    pddl_planning(domains, problems, save_dir)
    llm_agent.write_record(save_dir / File.FINAL_METRICS)

    return True


def run_ivml_batch(
    task_data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1,
    N: int = 8,
    choosen_parallelism: int = 3,
    iteration: int = 5,
    problem_parallelism: int = 3,
) -> None:
    """Execute IVML in batch mode."""
    tasks = load_batch_inputs(task_data_path)

    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return run_ivml(
            image_path=info[JSONKey.PATH],
            instruction=info[JSONKey.INSTRUCTION],
            save_dir=sub_dir,
            N=N,
            choosen_parallelism=choosen_parallelism,
            iteration=iteration,
            problem_parallelism=problem_parallelism
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
