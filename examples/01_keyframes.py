from pathlib import Path

from unidomain import keyframes_pipeline

root_dir = Path(__file__).parent.parent.resolve()

input_paths = root_dir / "data" / "tutorial" / "videos"
if not input_paths.exists():
    raise FileNotFoundError(
        f"Tutorial-Keyframes: {input_paths} does not exist. "
        "Please run 'unidomain download tutorial' to download tutorial data."
    )
output_dir = root_dir / "outputs" / "01_keyframes"

keyframes_pipeline(input_paths, output_dir)
