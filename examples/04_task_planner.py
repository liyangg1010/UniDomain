from pathlib import Path

from unidomain import task_planner_pipeline

root_dir = Path(__file__).parent.parent.resolve()


image_path = root_dir / "data" / "meta_domain" / "overview_task.jpg"
instruction = "Move the corn from the pot into the orange bowl, " \
              "wipe the table with the towel in the yellow drawer and " \
              "put it back to the closed yellow drawer."

meta_domain_dir = root_dir / "data" / "meta_domain"
if not meta_domain_dir.exists():
    raise FileNotFoundError(
        f"Demo-Task Planner: {meta_domain_dir} does not exist. "
        "Please run 'unidomain download meta' to download meta domain data first."
    )

save_dir = root_dir / "outputs" / "task_planner" / "overview_task"
task_planner_pipeline(
    image_path=image_path,
    instruction=instruction,
    meta_domain_dir=meta_domain_dir,
    save_dir=save_dir,
)
