"""
UniDomain Command Line Interface
================================

This module provides the main entry point for the UniDomain system.
It utilizes `typer` to expose the four core pipelines (Keyframe, Atomic, Fusion, Planner)
as accessible command-line utilities.

Usage:
    python cli.py [COMMAND] [ARGS]...
    python cli.py --help
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated, List, Optional

import typer

# Initialize the main application
app = typer.Typer(
    help="UniDomain CLI: Pre-trained PDDL Domains from Visual Demonstrations.",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

# Sub-commands for modules with multiple modes
atomic_app = typer.Typer(help="Atomic domain generation module (single/batch).")
planner_app = typer.Typer(help="Task planning module (single/batch).")

app.add_typer(atomic_app, name="atomic")
app.add_typer(planner_app, name="planner")


def version_callback(value: bool):
    import unidomain
    if value:
        print(f"UniDomain Version: {unidomain.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option(
        "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the application version and exit."
    )] = None
):
    """
    UniDomain Framework Configuration
    """
    pass


# ==========================================
# Module: Keyframe (Direct Command)
# ==========================================
@app.command("keyframe")
def keyframe_entry(
    input_paths: Annotated[List[Path], typer.Option(
        "--input", "-i",
        help="Input video file(s) or directory(s). Can be used multiple times.",
        exists=True
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save extracted keyframes."
    )],
):
    """
    [bold green]Extract[/bold green] keyframes from videos or image sequences.
    
    This command calculates pixel-wise energy to select significant frames.
    It supports batch processing of directories.
    """
    from unidomain.keyframes.pipeline import keyframes_pipeline
    
    keyframes_pipeline(input_paths=input_paths, output_dir=output_dir)


# ==========================================
# Module: Fusion (Direct Command)
# ==========================================
@app.command("fusion")
def fusion_entry(
    domain_dir: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Root directory containing atomic domains (0/, 1/, ...).",
        exists=True, file_okay=False, dir_okay=True
    )],
    output_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save the intermediate and fused meta-domains."
    )],
    num_workers: Annotated[int, typer.Option(
        "--workers", "-n",
        help="Number of parallel workers for binary tree merging."
    )] = 2,
):
    """
    [bold green]Fuse[/bold green] multiple atomic domains into a unified meta-domain.
    
    This command executes the parallel binary tree fusion strategy.
    It supports resumable execution if interrupted.
    """
    from unidomain.domain_fusion.pipeline import domain_fusion_pipeline
    
    domain_fusion_pipeline(
        domain_dir=domain_dir,
        output_dir=output_dir,
        num_workers=num_workers
    )


# ==========================================
# Module: Atomic Domain (Sub-commands)
# ==========================================
@atomic_app.command("run")
def atomic_run(
    keyframes_dir: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Directory containing sorted keyframes.",
        exists=True, file_okay=False, dir_okay=True
    )],
    instructions: Annotated[str, typer.Option(
        "--instruct",
        help="Task instructions or video description."
    )],
    save_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save the generated atomic domain."
    )],
):
    """
    Generate an atomic domain from a [bold]single[/bold] video sequence.
    """
    from unidomain.atomic_domain.pipeline import atomic_domain_pipeline
    
    atomic_domain_pipeline(
        keyframes_dir=keyframes_dir,
        instructions=instructions,
        save_dir=save_dir
    )


@atomic_app.command("batch")
def atomic_batch(
    data_path: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Path to the JSON dataset index file.",
        exists=True, dir_okay=False
    )],
    save_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Root directory to save generated domains."
    )],
    num_workers: Annotated[int, typer.Option(
        "--workers", "-n",
        help="Number of parallel workers."
    )] = 1,
):
    """
    Batch process atomic domain generation from a [bold]JSON index[/bold].
    """
    from unidomain.atomic_domain.pipeline import atomic_domain_batch_pipeline
    
    atomic_domain_batch_pipeline(
        data_path=data_path,
        save_dir=save_dir,
        num_workers=num_workers
    )


# ==========================================
# Module: Task Planner (Sub-commands)
# ==========================================
@planner_app.command("run")
def planner_run(
    image_path: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Path to the initial observation image.",
        exists=True, dir_okay=False
    )],
    instruction: Annotated[str, typer.Option(
        "--instruct",
        help="Instruction for the target task."
    )],
    meta_domain_dir: Annotated[Path, typer.Option(
        "--meta", "-m",
        help="Directory containing the meta domain (json/pddl).",
        exists=True, file_okay=False, dir_okay=True
    )],
    save_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save planning results."
    )],
    # Advanced Options
    require_filtering: Annotated[bool, typer.Option(
        help="Enable domain filtering/pruning."
    )] = True,
    filter_parallelism: Annotated[int, typer.Option(
        help="Number of LLM samples for filtering."
    )] = 1,
    problem_parallelism: Annotated[int, typer.Option(
        help="Number of LLM samples for problem generation."
    )] = 1,
):
    """
    Run task planner for a [bold]single[/bold] problem instance.
    """
    from unidomain.task_planner.pipeline import task_planner_pipeline
    
    task_planner_pipeline(
        image_path=image_path,
        instruction=instruction,
        meta_domain_dir=meta_domain_dir,
        save_dir=save_dir,
        require_filtering=require_filtering,
        filter_parallelism=filter_parallelism,
        problem_parallelism=problem_parallelism,
    )


@planner_app.command("batch")
def planner_batch(
    task_data_path: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Path to the JSON task data file.",
        exists=True, dir_okay=False
    )],
    meta_domain_dir: Annotated[Path, typer.Option(
        "--meta", "-m",
        help="Directory containing the meta domain.",
        exists=True, file_okay=False, dir_okay=True
    )],
    save_dir: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Directory to save planning results."
    )],
    num_workers: Annotated[int, typer.Option(
        "--workers", "-n",
        help="Number of parallel workers."
    )] = 1,
    # Advanced Options
    require_filtering: Annotated[bool, typer.Option(
        help="Enable domain filtering/pruning."
    )] = True,
):
    """
    Run task planner in [bold]batch mode[/bold].
    """
    from unidomain.task_planner.pipeline import task_planner_batch_pipeline
    
    task_planner_batch_pipeline(
        task_data_path=task_data_path,
        meta_domain_dir=meta_domain_dir,
        save_dir=save_dir,
        num_workers=num_workers,
        require_filtering=require_filtering
    )


# ==========================================
# Module: Download Data (Direct Command)
# ==========================================
@app.command("download")
def download_entry(
    mode: str = typer.Argument(..., help="tutorial | meta | tasks | results | unified | all")
):
    """
    [bold green]Download[/bold green] datasets from Hugging Face.
    (This is a shortcut for running 'python scripts/download_data.py')
    """

    # Locate download script
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    script_path = project_root / "scripts" / "download_data.py"

    if not script_path.exists():
        print(f"[Error] Could not find download script at: {script_path}")
        print("Are you running this from the source repository?")
        raise typer.Exit(code=1)

    # Construct command
    cmd = [sys.executable, str(script_path), "--mode", mode]

    print(f"Executing: {' '.join(cmd)}")
    
    # Call download script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[Error] Download failed.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
