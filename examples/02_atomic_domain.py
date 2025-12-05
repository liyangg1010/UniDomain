from pathlib import Path

from unidomain import atomic_domain_batch_pipeline

root_dir = Path(__file__).parent.parent.resolve()

# Batch Mode
data_path = root_dir / "data" / "tutorial" / "data.json"
if not data_path.exists():
    raise FileNotFoundError(
        f"Tutorial-Atomic Domain: {data_path} does not exist. "
        "Please run 'unidomain download tutorial' to download tutorial data.")

tutorial_keyframes_dir = root_dir / "outputs" / "01_keyframes"
if not tutorial_keyframes_dir:
    raise FileNotFoundError(
        f"Tutorial-Atomic Domain: {tutorial_keyframes_dir} does not exist. "
        "Please run 'python examples/01_keyframes.py' to extract keyframes."
    )

save_dir = root_dir / "outputs" / "02_atomic_domain"
atomic_domain_batch_pipeline(data_path, save_dir, num_workers=4)
