"""
Task Planner Pipeline
=====================

This module orchestrates the complete task planning workflow. It serves as the
entry point for solving specific tasks using the synthesized meta-domain.

The pipeline typically involves:
1. (Optional) Domain Filtering: Pruning the meta-domain to relevant predicates/operators.
2. Problem Generation: Using VLM to ground visual observations into PDDL problems.
3. Plan Generation: Invoking the PDDL planner to find executable action sequences.

Architecture Position:
    [Orchestrator / Top Level] -> Entry point for the task_planner module.
"""

from pathlib import Path
from typing import Dict, List

from unidomain.configs.constants import File, JSONKey
from unidomain.configs.settings import config
from unidomain.pddl.io import load_domain, save_domain
from unidomain.pddl.parser import extract_domain_from_pddl
from unidomain.schemas.llm_task_outputs import GenerateProblemOutputs
from unidomain.schemas.prompt_args import GenerateProblemPromptArgs
from unidomain.schemas.typings import Paths, Problems
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.task_planner.filtering import extract_operators_from_predicates, predicate_filtering
from unidomain.task_planner.planning import planning
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["task_planner_pipeline", "task_planner_batch_pipeline"]


def _generate_problem(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: GenerateProblemPromptArgs,
    n: int
) -> Problems:
    """Call VLM to generate PDDL problem definitions (Objects, Init, Goal).
    
    Args:
        image_path_list (List[Paths]): List of image paths, but generally only one image.
        llm_agent (LLMAgent): The VLM agent service.
        prompt_args (GenerateProblemPromptArgs): Arguments for VLM prompt.
        n (int): Number of problems to generate.
        
    Returns:
        Problems: A list of tuples, each containing (Objects, Init, Goal).
    """

    # Call VLM and parse JSON into specified format
    outputs: List[GenerateProblemOutputs] = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.generate_problem,
        prompt_args=prompt_args,
        validator=GenerateProblemOutputs,
        image_path_list=image_path_list,
        n=n
    )

    # Normalize output to a list even if n=1
    outputs = [outputs] if n == 1 else outputs
    problems = [(output.objects, output.init, output.goal) for output in outputs]

    return problems

def task_planner_pipeline(
    image_path: Paths,
    instruction: str,
    meta_domain_dir: Paths,
    save_dir: Paths,
    require_filtering: bool = True,
    filter_parallelism: int = 1,
    problem_parallelism: int = 1,
    filtering_with_operators: bool = True,
    planning_with_operators: bool = True,
) -> bool:
    """Execute the task planning pipeline for a single (image, instruction) pair.
    
    Args:
        image_path (Paths): Path to the input image.
        instruction (str): Natural language task instruction.
        meta_domain_dir (Paths): Directory containing the meta domain, containing:
            (1) group_predicates.txt: a file including grouped predicates.
                Grouping can be performed manually or automatically (by LLM).
                If not grouped, just a copy of predicates will also work.
            (2) meta_domain.json/pddl: a file including the planning domain.
                It can be retrieved from the root node of the binary domain fusion tree.
                JSON file is used if available.
                If JSON file is not provided, a JSON format will be generated from the PDDL file.

        save_dir (Paths): The path to the saved directory.
        require_filtering (bool): Whether to perform predicate and operator filtering. Defaults to True.
            Performing filtering will enhance the stablity and quality of the generated problems by LLM,
            at the cost of more tokens and thinking time due to two calls of LLM.
        filter_parallelism (int): Number of outputs from LLM for predicate and operator filtering.
            Ignored if require_filtering is False.
        problem_parallelism (int): Number of problems generated from LLM for PDDL planning.
        filtering_with_operators (bool): Whether to use operators as input to LLM when filtering.
            It appears that operators are not necessary as input to LLM when generating problems,
            because they do not exist in the problem file. However, using operators as input to LLM will
            enhance the quality of the generated problems, probably due to training data in GPT-4.1-like models.
            This will cost more tokens. Defaults to True.
            Ignored if require_filtering is False.
        planning_with_operators (bool): Whether to use operators as input to LLM when planning.
            Similar to filtering_with_operators, but used for actual generated problems. Defaults to True.
    
    Returns:
        bool: True if the pipeline completed execution (regardless of whether a
              valid plan was found). Returns None on unhandled errors.
    """
    llm_agent = LLMAgent(config.task_planner.llm_model_config, save_dir)
    execution_logger = get_task_logger(save_dir)
    save_dir = setup_path(save_dir)

    # Prepare meta domain
    meta_domain_dir = setup_path(meta_domain_dir, mkdir=False)
    group_predicates_path = meta_domain_dir / File.GROUP_PREDICATES
    meta_domain_path = meta_domain_dir / File.META_DOMAIN_JSON

    # If meta_domain.json does not exist, generate it from meta_domain.pddl
    if not meta_domain_path.exists():
        meta_domain_pddl = meta_domain_dir / File.META_DOMAIN_PDDL
        if not meta_domain_pddl.exists():
            raise FileNotFoundError(f"Meta domain not found in {meta_domain_dir}")
        
        domain_content = meta_domain_pddl.read_text(encoding="utf-8")
        predicates, operators = extract_domain_from_pddl(domain_content)
        save_domain(predicates, operators, meta_domain_path)

    # Logging
    execution_logger.info(
        "Runing Task Planner Pipeline \n"
        f"Meta domain: {meta_domain_dir} \n"
        f"Task Path: {image_path} \n"
        f"Instruction: {instruction}\n"
    )

    # Load resources
    group_predicates = group_predicates_path.read_text(encoding="utf-8")
    available_predicates, available_operators = load_domain(meta_domain_path)

    # Predicate and operator Filtering
    # Generate initial problems based on the task
    if require_filtering:
        execution_logger.info("Starting domain filtering...")

        # Construct prompt context
        domain_context = group_predicates
        if filtering_with_operators:
            # If operators are used as input to LLM, they are also included in the context
            operator_str = "\n".join(list(available_operators.values()))
            domain_context = f"Predicates:\n{group_predicates}\nOperators:\n{operator_str}"
        
        prompt_args = GenerateProblemPromptArgs(
            instructions=instruction,
            available_predicates=domain_context,
        )

        # Generate initial problems purely for the purpose of identifying necessary predicates
        filter_problems = _generate_problem(
            image_path_list=[image_path],
            llm_agent=llm_agent,
            prompt_args=prompt_args,
            n=filter_parallelism
        )

        # Filter predicates based on usage in these generated problems
        chosen_predicates = predicate_filtering(filter_problems, available_predicates)
        execution_logger.info(f"Chosen Predicates: {chosen_predicates}")

    # Plan Generation
    # Generate final problems based on the task and the filtered domain (or the original meta domain)
    active_predicates = chosen_predicates if require_filtering else available_predicates
    final_domain_context = str(active_predicates)

    if planning_with_operators:
        chosen_operators = extract_operators_from_predicates(available_operators, active_predicates)
        operator_str = "\n".join(list(chosen_operators.values()))
        final_domain_context = f"Predicates:\n{active_predicates}\nOperators:\n{operator_str}"
    
    # Generate the actual PDDL problems for planning
    prompt_args = GenerateProblemPromptArgs(
        instructions=instruction,
        available_predicates=final_domain_context,
    )
    final_problems = _generate_problem(
        image_path_list=[image_path],
        llm_agent=llm_agent,
        prompt_args=prompt_args,
        n=problem_parallelism,
    )

    # Record metrics and solve PDDL problems
    llm_agent.write_record(save_dir / File.FINAL_METRICS)
    best_plan_path = planning(meta_domain_path, final_problems, save_dir)

    # Copy final planning results to save_dir
    final_results_path = save_dir / File.FINAL_RESULTS

    if best_plan_path is not None:
        plan_content = best_plan_path.read_text(encoding="utf-8")
        final_results_path.write_text(plan_content, encoding="utf-8")
        execution_logger.info(f"Final Result:\n{plan_content}")
    else:
        final_results_path.write_text("No solution found.", encoding="utf-8")
        execution_logger.info("No solution found.")

    return True

def task_planner_batch_pipeline(
    task_data_path: Paths,
    meta_domain_dir: Paths,
    save_dir: Paths,
    num_workers: int = 1,
    require_filtering: bool = True,
    filter_parallelism: int = 1,
    problem_parallelism: int = 1,
    filtering_with_operators: bool = True,
    planning_with_operators: bool = True,
) -> None:
    """Execute task planning in batch mode for multiple tasks.
    
    Args:
        task_data_path (Paths): The JSON file to the tasks, in the following format:
            {
                task_name_1: {"instruction": "task_1_instruction", "path": "task_1_image_path"},
                task_name_2: {"instruction": "task_2_instruction", "path": "task_2_image_path"},
                ...
            }
            Image path can be whether absolute path or relative path to the JSON file's directory.

        meta_domain_dir (Paths): Directory containing the meta domain, containing:
            (1) group_predicates.txt: a file including grouped predicates.
                Grouping can be performed manually or automatically (by LLM).
                If not grouped, just a copy of predicates will also work.
            (2) meta_domain.json/pddl: a file including the planning domain.
                It can be retrieved from the root node of the binary domain fusion tree.
                JSON file is used if available.
                If JSON file is not provided, a JSON format will be generated from the PDDL file.
                
        save_dir (Paths): The path to the saved directory, in the following format:
            task_name_1/
                ...
            task_name_2/
                ...

        num_workers (int, optional): The number of threadings to execute all tasks.
        require_filtering (bool): Whether to perform predicate and operator filtering. Defaults to True.
            Performing filtering will enhance the stablity and quality of the generated problems by LLM,
            at the cost of more tokens and thinking time due to two calls of LLM.
        filter_parallelism (int): Number of outputs from LLM for predicate and operator filtering.
            Ignored if require_filtering is False.
        problem_parallelism (int): Number of problems generated from LLM for PDDL planning.
        filtering_with_operators (bool): Whether to use operators as input to LLM when filtering.
            It appears that operators are not necessary as input to LLM when generating problems,
            because they do not exist in the problem file. However, using operators as input to LLM will
            enhance the quality of the generated problems, probably due to training data in GPT-4.1-like models.
            This will cost more tokens. Defaults to True.
            Ignored if require_filtering is False.
        planning_with_operators (bool): Whether to use operators as input to LLM when planning.
            Similar to filtering_with_operators, but used for actual generated problems. Defaults to True.
    
    Returns:
        None
    """

    tasks = load_batch_inputs(task_data_path)

    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return task_planner_pipeline(
            image_path=info[JSONKey.PATH],
            instruction=info[JSONKey.INSTRUCTION],
            save_dir=sub_dir,
            meta_domain_dir=meta_domain_dir,
            require_filtering=require_filtering,
            filter_parallelism=filter_parallelism,
            problem_parallelism=problem_parallelism,
            filtering_with_operators=filtering_with_operators,
            planning_with_operators=planning_with_operators,
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
