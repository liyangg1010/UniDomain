"""
Baseline Utilities
==================

Shared helper functions for PDDL-based baselines (e.g., VLM-CoT-PDDL, IVML).
These utilities handle the execution of the PDDL planner and the aggregation
of results from multiple generated samples.
"""

import random
from collections import Counter
from typing import List

from unidomain.configs.constants import File
from unidomain.pddl.planner import PDDLPlanner, parse_plans
from unidomain.schemas.typings import Paths, PDDLPlannerFiles
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["pddl_planning"]


def _ensembling(results: List[Paths]) -> Paths:
    """Select the most consistent plan using Majority Voting (Mode of plan length)."""

    # Calculate costs (plan lengths) for all results
    valid_results = []
    costs = []

    for result_path in results:
        plan_steps = parse_plans(result_path)
        cost = len(plan_steps)
        if plan_steps:
            valid_results.append((result_path, cost))
            costs.append(cost)

    if not valid_results:
        return None

    # Majority Voting: find the most frequent cost (Mode)
    cost_counter = Counter(costs)
    max_frequency = max(cost_counter.values())
    
    # Identify costs that appear with max frequency
    most_common_costs = [cost for cost, count in cost_counter.items() if count == max_frequency]

    # Randomly select a target cost from the modes (usually there's only one)
    chosen_cost = random.choice(most_common_costs)

    # Filter candidates that match the chosen cost
    candidate_paths = [path for (path, cost) in valid_results if cost == chosen_cost]

    return random.choice(candidate_paths)


def pddl_planning(
    domains: List[str],
    problems: List[str],
    save_dir: Paths
) -> None:
    """Execute PDDL planning for a batch of domain/problem pairs and save the best result.

    This function writes the domain/problem strings to files, runs the planner
    in parallel, and uses ensembling to select the final output.

    Args:
        domains (List[str]): List of PDDL domain strings.
        problems (List[str]): List of PDDL problem strings.
        save_dir (Paths): Directory to save intermediate and final results.
    """

    save_dir = setup_path(save_dir)
    execution_logger = get_task_logger(save_dir)

    # Prepare PDDL files
    files: PDDLPlannerFiles = []
    for i, (domain, problem) in enumerate(zip(domains, problems)):

        result_dir = setup_path(save_dir / f"{File.RESULTS_PREFIX}{i}")
        domain_path = result_dir / File.DOMAIN_PDDL
        problem_path = result_dir / File.PROBLEM_PDDL
        result_path = result_dir / File.OUTPUTS

        domain_path.write_text(domain, encoding="utf-8")
        problem_path.write_text(problem, encoding="utf-8")

        files.append((domain_path, problem_path, result_path))

    # Run Planner
    pddl_planner = PDDLPlanner()
    pddl_planner.parallel_solve_pddl(files)

    # Ensemble Results
    result_paths = [result_path for _, _, result_path in files]
    best_plan_path = _ensembling(result_paths)

    # Save Final Result
    final_results_path = save_dir / File.FINAL_RESULTS
    if best_plan_path is not None:
        plan_content = best_plan_path.read_text(encoding="utf-8")
        final_results_path.write_text(plan_content, encoding="utf-8")
        execution_logger.info(f"Final Result:\n{plan_content}")
    else:
        final_results_path.write_text("No solution found.", encoding="utf-8")
        execution_logger.info("No solution found.")

    pddl_planner.write_record(save_dir / File.FINAL_METRICS)
