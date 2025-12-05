from pathlib import Path

from unidomain.baselines import (
                                 run_code_as_policies_batch,
                                 run_isr_llm_batch,
                                 run_ivml_batch,
                                 run_react,
                                 run_vlm_cot_batch,
                                 run_vlm_cot_pddl_batch,
)

root_dir = Path(__file__).parent.parent.resolve()
task_data_path = root_dir / "data" / "tasks" / "task_data.json"
baseline_dir = root_dir / "outputs" / "baselines"


# =========================================================================
# ⚠️ SAFETY LOCK: Configuration Required
# =========================================================================
# By default, all runs are commented out to prevent accidental massive costs.
# Please uncomment the specific method you want to evaluate below.
# =========================================================================

# # Run VLM-CoT Baseline in Batch Mode
# run_vlm_cot_batch(task_data_path, baseline_dir / "vlm_cot_batch", num_workers=10)

# # Run VLM-CoT-PDDL Baseline in Batch Mode
# run_vlm_cot_pddl_batch(task_data_path, baseline_dir / "vlm_cot_pddl_batch", num_workers=10)

# # Run IVML Baseline in Batch Mode
# run_ivml_batch(task_data_path, baseline_dir / "ivml_batch", num_workers=10)

# # Run Code as Policies Baseline in Batch Mode
# run_code_as_policies_batch(task_data_path, baseline_dir / "code_as_policies_batch", num_workers=10)

# # Run ISR-LLM Baseline in Batch Mode
# run_isr_llm_batch(task_data_path, baseline_dir / "isr_llm_batch", num_workers=10)

# # If you want to run baselines above for a single task, please import the single mode function
# # Example:
# # from unidomain.baselines import run_vlm_cot
# # image_path, instruction, save_dir = ..., ..., ...
# # run_vlm_cot(image_path, instruction, save_dir)


# # Run ReAct Baseline. ReAct requires iteration, not supporting batch mode.
# # num_trials = 1 for ReAct
# # num_trials > 1 for Reflexion
# # max_steps specifies the max running step of Reflexion
# # video_index specifies the video device index on Ubuntu
# # Running ReAct requires other dependencies (opencv-python, etc.), using "pip install -e .[baselines] to install"
# # Note: ReAct is not supported on our DockerFile, please run locally (or setup X11 server yourself).
# image_path = root_dir / "data" / "tasks" / "BlockWorld" / "task_1.jpg"
# instruction = "Arrange all blocks in a single stack (from top to bottom): 1, 2, 3, 4, 5, 6, 7, 8."
# run_react(image_path, instruction, baseline_dir / "react", num_trials=1, max_steps=40, video_index=0)

# --- Safety Prompt ---
print("\n[Info] No baseline selected.")
print("Please open 'scripts/run_baselines.py' and uncomment the method you want to run.")
