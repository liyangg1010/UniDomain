"""
Baseline: ReAct / Reflexion (Interactive Vision-Language Planning)
==================================================================

This module implements an interactive baseline based on **ReAct** and **Reflexion**.
**References:**
1.  "ReAct: Synergizing Reasoning and Acting in Language Models"
2.  "Reflexion: Language Agents with Verbal Reinforcement Learning"

**Algorithm Overview:**
The agent operates in an interleaved manner: Observation -> Thought -> Action -> Feedback.
It generates an action, receives feedback from the physical environment (via human/camera),
and performs self-correction if necessary.

**Modifications & Adaptations:**
-   **Single-Trial Setting (ReAct Mode)**: While the code supports multi-trial self-reflection
    (Reflexion), we predominantly use `num_trials=1`. Real-world robotic tasks typically
    cannot be reset cheaply for trial-and-error learning. Thus, the method functions
    primarily as a **Vision-Language ReAct Agent**.
-   **Physical Feedback Loop**: Unlike simulation baselines, this method requires **live
    hardware integration**.
    -   **Success**: Verified by human teleoperation; returns a new image via the camera (`video_index`).
    -   **Failure**: Returns a generic "Fail" signal without specific reasons, triggering LLM reflection.
-   **Evaluation Note**: This baseline is more lenient than others. Generating an invalid
    action does not immediately terminate the episode; the agent is allowed to reflect
    and retry until `max_steps` is reached.

Architecture Position:
    [Baselines] -> An interactive, human-in-the-loop baseline.
"""


import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from unidomain.baselines.reflexion.camera_trial import run_trial
from unidomain.baselines.reflexion.generate_reflections import update_memory
from unidomain.baselines.reflexion.utils import write_llm_metric
from unidomain.configs.settings import config


def run_react(
    image_path: str,
    instruction: str,
    save_dir: str,
    num_trials: int = 3,
    max_steps: int = 30,
    video_index: int = 0
) -> None:
    """Execute the ReAct/Reflexion pipeline with live camera feedback.

    This function starts an interactive loop where the model proposes actions,
    and a human expert (or simulator wrapper) executes them and provides visual feedback.

    Args:
        image_path (str): Path to the **initial** observation image. Subsequent observations
            will be captured dynamically from the camera.
        instruction (str): The task description.
        save_dir (str): Directory to save the interaction logs and images.
        
        num_trials (int): The number of attempts for the task.
            - If `1`: Runs as standard **ReAct** (Single-shot execution).
            - If `>1`: Runs as **Reflexion**, where the agent uses memory from previous
              failed trials to guide the next trial. (Note: Rarely used in real-robot
              settings due to reset costs).
        
        max_steps (int): The maximum number of interaction turns allowed. If the task
            is not completed within these steps, it is marked as failed.
        
        video_index (int): The Linux video device ID (e.g., `0` for `/dev/video0`).
            Used to capture the state of the environment after each successful action.

    Returns:
        None
    """

    print("Runing Baseline: ReAct")
    print(f"Task Path: {image_path}")
    print(f"Instruction: {instruction}")
    task_name = "ReAct"
 
    save_dir = str(save_dir)
    run_name = os.path.join(str(save_dir), "reflexion_run_logs")  
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    logging_dir = run_name

    # Initialize environment
    env_configs: List[Dict[str, Any]] = [{
        'name': task_name,
        'memory': [],
        'is_success': False,
        'skip': False
    }]
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # Print start status to user
    use_memory = True
    print(f"""
    -----
    Starting run with the following parameters:
    Run name: {logging_dir}
    task: {task_name}
    Number of trials: {num_trials}
    Use memory: {use_memory}

    Sending all logs to {run_name}
    -----
    """)

    # run trials
    # capture the first frame using the camera
    trial_idx = 0
    while trial_idx < num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        model = config.baselines.llm_model_config.model
        result_path = os.path.join(save_dir, f"solution_{trial_idx}.txt")
        if os.path.exists(result_path):
            os.remove(result_path)
        run_trial(
            trial_log_path,
            world_log_path,
            trial_idx,
            env_configs,
            use_memory,
            model,
            task_name,
            instruction,
            max_steps,
            os.path.join(save_dir, "video_images"),
            video_index
        )

        # update memory if needed
        if use_memory:
            env_configs = update_memory(trial_log_path, env_configs)

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1

        write_llm_metric(os.path.join(save_dir, f"summary_{trial_idx}.json"))
        