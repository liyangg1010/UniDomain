"""
Solvability Check & Refinement Step
===================================

This module implements the third stage of the atomic domain generation pipeline.
It validates the generated domain by creating test problems and attempting to solve
them using a standard PDDL planner. If validation fails, it triggers an LLM-based
refinement process using the planner's error logs.

Architecture Position:
    [Functional Steps / Middle Level] -> Called by atomic_domain.pipeline
    Uses PDDLPlanner for validation.
"""

import re
from pathlib import Path
from typing import List, Tuple

from unidomain.atomic_domain.post_process import save_and_post_process
from unidomain.configs.constants import File
from unidomain.configs.settings import config
from unidomain.pddl.io import load_domain
from unidomain.pddl.planner import PDDLPlanner
from unidomain.schemas.llm_task_outputs import GenerateTestProblemsOutputs, RefineFromSolvabilityCheckOutputs
from unidomain.schemas.prompt_args import GenerateTestProblemsPromptArgs, RefineFromSolvabilityCheckPromptArgs
from unidomain.schemas.typings import Paths, PDDLPlannerFiles
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.runtime import setup_path

__all__ = ["run_solvability_check_step", "run_solvability_refinement_step"]


def _generate_test_problems(
    llm_agent: LLMAgent,
    prompt_args: GenerateTestProblemsPromptArgs
) -> List[str]:
    """Call LLM to generate test problems based on domain predicates and instructions.
    
    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (GenerateProblemPromptArgs): Arguments for LLM prompt.

    Returns:
        List[str]: A list of generated PDDL problem strings.

    """

    # Call LLM and parse JSON into specified format
    outputs: GenerateTestProblemsOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.generate_test_problems,
        prompt_args=prompt_args,
        validator=GenerateTestProblemsOutputs,
    )

    # Return necessary information
    test_problems = [p.strip() for p in outputs.problems]
    return test_problems


def _check_solvability(files: PDDLPlannerFiles) -> Tuple[bool, str, str]:
    """Execute the PDDL planner to check the solvability of generated test problems.

    Args:
        files (PDDLPlannerFiles): A list of task tuples. Each tuple contains:
            1. The path to the domain for check, in PDDL format.
            2. The path to the test problem for check, in PDDL format.
            3. The path to the output file from PDDL planner.

    Returns:
        Tuple[bool, str, str]: A tuple of three elements:
            1. bool: True if the success rate exceeds the threshold.
            2. str: The content of the representative problem file (last success or last failure).
            3. str: The planner output corresponding to that problem (plan or error log).
    """

    if not files:
        return False, "", "No test problems generated."
    
    # Solve all PDDL problems
    pddl_planner = PDDLPlanner()
    num_workers = config.atomic_domain.pddl_planner_num_workers
    pddl_planner.parallel_solve_pddl(files, num_workers=num_workers)

    # Analyze planner results
    last_success_index = -1
    last_failure_index = -1
    success_count = 0
    for idx, (_, _, plan_path) in enumerate(files):
        planner_file_path = Path(plan_path).resolve()
        with planner_file_path.open("r", encoding="utf-8") as f:
            solver_output = "\n".join([line.strip() for line in f.readlines() if line.strip()])

        # Determine success based on output content
        # Note: Fast Downward typically writes "INFO" logs to stdout/file on failure
        # or if no plan is found. A clean plan file usually doesn't contain "INFO".
        if not solver_output or re.search(r"\bINFO\b", solver_output):
            last_failure_index = idx
        else:
            last_success_index = idx
            success_count += 1

    # Success if the ratio of solvable test problems > threshold
    threshold = config.atomic_domain.solvability_check_threshold
    success = True if success_count / len(files) >= threshold else False
    last_index = last_success_index if success else last_failure_index
    
    # For no success
    if last_index == -1 and not success:
        last_index = last_failure_index

    # Return the last problem and planner output
    last_problem = files[last_index][1].read_text(encoding="utf-8")
    last_planner_output = files[last_success_index][2].read_text(encoding="utf-8")
    return success, last_problem, last_planner_output


def run_solvability_check_step(
    llm_agent: LLMAgent,
    domain_path: Paths,
    domain_json_path: Paths,
    instructions: str,
    output_dir: Paths
) -> Tuple[bool, str, str]:
    """Orchestrate the solvability check process: generate, run, and analyze.

    Args:
        llm_agent (LLMAgent): The LLM agent service.
        domain_path (Paths): Path to the PDDL domain file.
        domain_json_path (Paths): Path to the JSON domain file.
        instructions (str): Task instructions.
        output_dir (Paths): Directory to save test problems and logs.

    Returns:
        Tuple[bool, str, str]: (Success flag, Representative problem, Planner output/Error).
    """
    output_dir = setup_path(output_dir)

    # Generate Test Problems
    predicates, _ = load_domain(domain_json_path)
    prompt_args = GenerateTestProblemsPromptArgs(
        instructions=instructions,
        predicates=str(predicates),
        num_problems=config.atomic_domain.num_test_problems
    )
    problems = _generate_test_problems(llm_agent, prompt_args)

    # Prepare Files for Planner
    files = []
    for idx, problem_str in enumerate(problems):
        test_problem_path = output_dir / f"{File.TEST_PROBLEM_PREFIX}{idx}.pddl"
        test_problem_path.write_text(problem_str, encoding="utf-8")
        plan_path = output_dir / f"{File.PLAN_PREFIX}{idx}.txt"
        files.append([domain_path, test_problem_path, plan_path])
    
    # Run Solvability Check
    return _check_solvability(files)


def _refine_from_solvability_check(
    llm_agent: LLMAgent,
    prompt_args: RefineFromSolvabilityCheckPromptArgs,
) -> str:
    """Call LLM to refine the atomic domain based on solvability errors.
    
    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (RefineFromSolvabilityCheckPromptArgs): Arguments for LLM prompt.

    Returns:
        str: the refined domain string in PDDL format.
    """

    # Call LLM and parse JSON into specified format
    outputs: RefineFromSolvabilityCheckOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.refine_from_solvability_check,
        prompt_args=prompt_args,
        validator=RefineFromSolvabilityCheckOutputs,
    )

    # Return necessary information
    refined_domain = outputs.refined_domain
    return refined_domain


def run_solvability_refinement_step(
    llm_agent: LLMAgent,
    domain_content: str,
    last_problem: str,
    solver_error: str,
    output_dir: Paths
) -> None:
    """Execute the refinement step when solvability check fails.

    Args:
        llm_agent (LLMAgent): The LLM agent service.
        domain_content (str): The current failing domain content.
        last_problem (str): The problem file that caused failure.
        solver_error (str): The error log from the planner.
        output_dir (Paths): Directory to save the refined result.
    """
    output_dir = setup_path(output_dir)
    
    # Call LLM for refinement
    prompt_args = RefineFromSolvabilityCheckPromptArgs(
        domain=domain_content,
        test_problem=last_problem,
        solver_error=solver_error,
    )
    refined_str = _refine_from_solvability_check(llm_agent, prompt_args)

    save_and_post_process(refined_str, output_dir, stem=File.REFINED_DOMAIN_STEM)
