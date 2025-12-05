from pathlib import Path

from unidomain import task_planner_batch_pipeline

root_dir = Path(__file__).parent.parent.resolve()


task_data_path = root_dir / "data" / "tasks" / "task_data.json"
save_dir = root_dir / "ouptuts" / "task_planner" / "100_tasks"
meta_domain_dir = root_dir / "data" / "meta_domain"

# =========================================================================
# ⚠️ SAFETY LOCK: Cost Warning
# =========================================================================
# Running this script will execute the Task Planner on 100 tasks in parallel.
# This involves intensive LLM calls.
# Please UNCOMMENT the function below to proceed.
# =========================================================================
# # Run task planner for 100 tasks in Batch Mode
# # If you want to run a single task, please use "task_planner_pipeline"
# task_planner_batch_pipeline(
#     task_data_path=task_data_path,
#     meta_domain_dir=meta_domain_dir,
#     save_dir=save_dir,
#     num_workers=10,
# )

# --- Safety Prompt ---
print("\n[Info] Task execution is locked by default to prevent accidental costs.")
print(f"Target: Run 100 tasks from {task_data_path}")
print("Action: Please open 'scripts/run_tasks.py' and uncomment the 'task_planner_batch_pipeline' call.")
