"""
Planning Module
======================

This module serves as the execution engine for task planning. It prepares the
environment (converting domains/problems), invokes the PDDL planner, and
processes the results. It also includes strategies for cleaning up problem
definitions and ensembling multiple planning results.

Architecture Position:
    [Core Algorithms / Functional Level] -> Called by task_planner.pipeline
"""

import random
import re
from collections import Counter
from typing import List, Set

from unidomain.configs.constants import File
from unidomain.pddl.generator import domain2file, problem2file
from unidomain.pddl.io import load_domain, load_problem, save_problem
from unidomain.pddl.parser import get_predicate_name
from unidomain.pddl.planner import PDDLPlanner, parse_plans
from unidomain.schemas.typings import Paths, PDDLPlannerFiles, Problems

__all__ = ["planning"]


def _remove_undefined_init_in_problem(domain_path: Paths, problem_path: Paths) -> None:
    """Remove initial predicates from the problem if they are not defined in the domain.

    This step is crucial for handling LLM-generated problems. LLMs often "hallucinate"
    predicates that seem plausible but do not strictly exist in the domain definition.
    These hallucinated predicates are typically non-essential for the actual planning
    logic (redundant details) but will cause standard PDDL planners to crash due to
    undefined symbols. Removing them ensures PDDL compliance without affecting solvability.

    Args:
        domain_path (Paths): Path to the domain JSON file.
        problem_path (Paths): Path to the problem JSON file.
    """

    # Load the domain and problem
    predicates, _ = load_domain(domain_path)
    objects, init, goal = load_problem(problem_path)

    valid_predicate_names: Set[str] = {
            get_predicate_name(k) for k in predicates.keys()
        }

    # Regex to safely capture the predicate name: (name ...
    pattern = re.compile(r"\(([\w-]+)")

    # Check whether predicates in "init" are defined in domain file
    new_inits = []
    for check_init in init:
        match = pattern.search(check_init)
        if match:
            predicate_name = match.group(1)
            # Only keep if NAME exists in the domain definition
            if predicate_name in valid_predicate_names:
                new_inits.append(check_init)

    # Save the cleaned problem back to the same path
    save_problem(objects, new_inits, goal, problem_path)

def _ensembling(results: List[Paths]) -> Paths:
    """Select the most consistent plan from multiple planner outputs using Majority Voting.

    This function calculates the cost (length) of each generated plan and selects
    one from the group of plans that have the most frequent cost (Mode).

    Note:
        Experimental analysis suggests that ensembling offers limited gains for this task,
        though it is retained here for experimental completeness:

        1. At "temperature=0.0", despite the inherent stochasticity of LLMs (outputs
           are not strictly deterministic), the variance remains minimal, resulting
           in highly similar plans across the batch.
        2. At higher temperatures, outputs become unstable. This instability manifests
           primarily as **Grounding errors** (e.g., hallucinated objects, invalid
           predicate-argument bindings). Consequently, the divergence tends to lead to
           invalid logic rather than diverse, valid solutions.
        
    Args:
        results (List[Paths]): List of paths to PDDL solver outputs.

    Returns:
        Optional[Paths]: Path to the selected PDDL solver output.
            Returns None if no solution was found in any result.
    """

       
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

def planning(
    meta_domain_path: Paths,
    problems: Problems,
    save_dir: Paths
) -> Paths:
    """Execute the PDDL planning process for a set of problems.

    This function prepares the files, cleans up undefined predicates (handling
    LLM hallucinations), runs the PDDL planner in parallel, and selects the
    final result via ensembling.

    Args:
        meta_domain_path (Paths): Path to the meta domain (JSON).
        problems (Problems): A list of problem definitions (objects, init, goal).
        save_dir (Paths): Directory to save intermediate and final results.

    Returns:
        Optional[Paths]: Path to the final selected plan output file.
    """

    # Construct PDDL files
    files: PDDLPlannerFiles = []
    for index, problem in enumerate(problems):
        objects, init, goal = problem

        # Create paths for each problem
        results_dir = save_dir / f"{File.RESULTS_PREFIX}{index}"
        domain_save_path =  results_dir / File.DOMAIN_PDDL
        problem_save_path = results_dir / File.PROBLEM_PDDL
        output_save_path = results_dir / File.OUTPUTS
        problem_json_path = results_dir / File.PROBLEM_JSON

        # Convert and postprocess the problem
        domain2file(meta_domain_path, domain_save_path)
        save_problem(objects, init, goal, problem_json_path)
        _remove_undefined_init_in_problem(meta_domain_path, problem_json_path)
        problem2file(problem_json_path, problem_save_path)

        files.append((domain_save_path, problem_save_path, output_save_path))
    
    # Solve PDDL problems in parallel
    pddl_planner = PDDLPlanner()
    pddl_planner.parallel_solve_pddl(files)
    
    # Get the final plans
    output_paths = [output_path for _, _, output_path in files]
    final_result_path = _ensembling(output_paths)
    pddl_planner.write_record(save_dir / File.FINAL_METRICS)

    return final_result_path
