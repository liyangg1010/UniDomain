"""
Baseline: ISR-LLM (Iterative Self-Refined Large Language Model)
===============================================================

This module adapts the ISR-LLM baseline for the UniDomain benchmark.
**Reference:** "ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning".

**Algorithm Overview:**
1.  **Translator**: Translates natural language instructions into PDDL Domain/Problem definitions.
2.  **Planner**: Uses an LLM to generate a plan sequence based on the PDDL description.
3.  **Validator (Self-Refinement)**: The LLM validates the generated plan. If incorrect, 
    it provides feedback to refine the plan iteratively.

**Modifications & Adaptations:**
-   **Zero-Shot Adaptation**: The original ISR-LLM leverages task-specific few-shot 
    examples. In our setting, we do not provide these specific examples because 
    ground-truth annotations are assumed to be unavailable for the novel/unseen real-world
    tasks. Instead, we use a **single, generic prompt template** (serving as a basic format
    guide) to evaluate the model's true generalization capability.
-   **VLM Integration**: The original text-only "Translator" has been adapted to accept 
    visual inputs, enabling it to perceive the environment state.
-   **Wrapper Interface**: Encapsulated the original implementation within the standard 
    UniDomain execution pipeline.

Architecture Position:
    [Baselines] -> A Neuro-Symbolic baseline emphasizing iterative self-correction.
"""

import os
from typing import Dict

from unidomain.baselines.isr_llm.Planner.Planner import Planner
from unidomain.baselines.isr_llm.Translator.Translator import Translator
from unidomain.baselines.isr_llm.utils import *
from unidomain.baselines.isr_llm.Validator.Validator import Validator
from unidomain.configs.constants import File
from unidomain.configs.settings import config
from unidomain.schemas.typings import Paths
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs


# LLM planning with PDDL translator and using self feedback
def test_LLM_trans_self_feedback(
    image_path: str,
    instruction: str,
    max_num_refine: str, 
    test_log_file_path: str,
    LLM_Translator: Translator,
    LLM_Planner: Planner,
    LLM_Validator: Validator
) -> None:

    # generate description
    description = f"Now instruction is {instruction}."
    print(description)

    with open(test_log_file_path, "a") as f:
        f.write(description +"\n")

    # LLM translator
    planning_problem = LLM_Translator.query(description, image_path, is_append = False)
    print(planning_problem)

    # initial and goal state in pddl
    pddl_init_state, pddl_goal_state = extract_state_pddl(planning_problem)

    # refine loop
    for j in range(max_num_refine + 1):

        with open(test_log_file_path, "a") as f:
            f.write("Attempt: " + str(j) + "\n")

        # LLM planner
        temperature = 0
        action_sequence = LLM_Planner.query(planning_problem, is_append = True, temperature = temperature)

        # simulate actions
        print("Attempt", j)
        print(action_sequence)
        with open(test_log_file_path, "a") as f:
            f.write(action_sequence +"\n")
            f.write("Analysis: " +"\n")

        # self-evaluate actions
        action_description = extract_action_description(action_sequence)

        validate_question = "Question:\ninitial state:\n" + pddl_init_state + "\nGoal state:\n" + pddl_goal_state + "\nExamined action sequence:\n" + action_description

        print(validate_question)
        with open(test_log_file_path, "a") as f:
            f.write(validate_question +"\n")

        response_validator_content = LLM_Validator.query(validate_question, is_append=False)
        with open(test_log_file_path, "a") as f:
            f.write(response_validator_content +"\n")

        # Answer
        valid_result = response_validator_content.split('Final answer:', 1)
        if len(valid_result) == 1: # no action returned
            break
        else:
            valid_result = valid_result[1]

        if 'Yes' in valid_result:

            print("Self-evaluation suggests a solution.")
            with open(test_log_file_path, "a") as f:
                f.write("Self-evaluation suggests a solution.\n")
            # exit self-refine loop
            break

        elif 'No' in valid_result:

            print("Self-evaluation suggests a failure.")
            error_description = "Goal is not satisfied." #Error analysis:" + summary_content 
            planning_problem = error_description + " Please find a new plan by considering the goals."
            
            print(planning_problem)

            with open(test_log_file_path, "a") as f:
                f.write(planning_problem +"\n")

        else:
            print("Unknown results:", valid_result)

    # final results
    print(action_sequence)
    final_results_path = os.path.join(os.path.dirname(test_log_file_path), File.FINAL_RESULTS)
    with open(final_results_path, "w") as f:
        f.write(action_sequence)

def run_isr_llm(
    image_path: str,
    instruction: str,
    logdir: str,
    max_num_refine: int = 10,
):
    """Execute the ISR-LLM baseline pipeline for a single task.

    It orchestrates the three components (Translator, Planner, Validator) 
    to solve the task through iterative self-refinement.

    Args:
        image_path (Paths): Path to the visual observation.
        instruction (str): The task instruction.
        logdir (Paths): Directory to save logs, PDDL files, and metrics.
        max_num_refine (int): Maximum number of self-correction iterations. Defaults to 10.

    Returns:
        bool: True if the process finishes (regardless of plan success).
    """

    # Test log
    logdir = str(logdir)
    test_log_file_path = logdir + "/test_log.txt"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(test_log_file_path, "w") as f:
        f.write("Test log for " + "\n")

    print("Runing Baseline: ISR-LLM")
    print(f"Task Path: {image_path}")
    print(f"Instruction: {instruction}")

    model = config.baselines.llm_model_config.model
    test_LLM_trans_self_feedback(
        image_path=str(image_path),
        instruction=instruction,
        max_num_refine=max_num_refine,
        test_log_file_path=test_log_file_path,
        LLM_Translator=Translator(model, logdir, is_log_example=True),
        LLM_Planner=Planner(model, logdir, is_log_example=True),
        LLM_Validator=Validator(model, logdir, is_log_example=True)
    )

    write_llm_metric(os.path.join(logdir, File.FINAL_METRICS))


def run_isr_llm_batch(
    task_data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1,
) -> None:
    """Execute ISR-LLM in batch mode using the standard UniDomain runner."""
    tasks = load_batch_inputs(task_data_path)
    
    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return run_isr_llm(
            image_path=info["path"],
            instruction=info["instruction"],
            logdir=sub_dir,
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
