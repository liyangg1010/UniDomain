"""
Keyframes Pipeline
============================

This module acts as the unified entry point for the keyframe extraction system.
It intelligently handles various input formats (single videos, directories of videos,
image sequences) and dispatches them to the appropriate processing functions.

Architecture Position:
    [Orchestrator / Top Level] -> Entry point for the keyframes module.
"""

from pathlib import Path
from typing import List, Union

from unidomain.keyframes.processors import keyframe_pipeline_from_video, process_image_sequence
from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import get_sorted_paths, is_image_file, is_video_file, setup_path

__all__ = ["keyframes_pipeline"]


def keyframes_pipeline(
    input_paths: Union[Paths, List[Paths]],
    output_dir: Paths
) -> None:
    """Execute the keyframe extraction pipeline supporting flexible input formats.

    This function automatically detects the type of input and routes it to the
    correct processor (video extractor or sequence selector).

    Supported Input Formats:
        1. Single video file path.
        2. Directory containing multiple video files.
        3. Directory containing a single image sequence (frames).
        4. Directory containing multiple sub-directories of image sequences.
        5. List of mixed paths (videos or sequence directories).

    Args:
        input_paths (Union[Paths, List[Paths]]): The input source(s).
        output_dir (Paths): Root directory to save all extracted keyframes.

    Returns:
        None
    """

    output_dir = setup_path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Warning: Output directory '{output_dir}' is not empty.")
        
    if not isinstance(input_paths, (list, tuple)):
        # If a single path, determine if it's a "container" (Case 2, 4) or a "single unit" (Case 1, 3).

        root_path = Path(input_paths).resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {root_path}")
    
        # Case 1: single video file
        if root_path.is_file():
            if is_video_file(root_path):
                keyframe_pipeline_from_video(root_path, output_dir / root_path.stem)
            else:
                raise ValueError(f"Input file is not a recognized video: {root_path}")
            return

        # Handle directory inputs
        if root_path.is_dir():
            children = get_sorted_paths(root_path)
            if not children:
                print(f"Warning: Input directory is empty: {root_path}")
                return

            # Analyze content to decide strategy
            has_subdirs = any(p.is_dir() for p in children)
            has_videos = any(is_video_file(p) for p in children)
            has_images = any(is_image_file(p) for p in children)

            # Case 4: directory of sequence directories (e.g., data/seq1, data/seq2)
            if has_subdirs:
                print(f"Detected batch image sequences in: {root_path}")
                for sub_dir in children:
                    if sub_dir.is_dir():
                        process_image_sequence(sub_dir, output_dir / sub_dir.name)
                return

            # Case 2: directory of videos (e.g., data/vid1.mp4, data/vid2.mp4)
            if has_videos:
                print(f"Detected batch videos in: {root_path}")
                for video_path in children:
                    if is_video_file(video_path):
                        keyframe_pipeline_from_video(video_path, output_dir / video_path.stem)
                return

            # Case 3: single image sequence (e.g., data/frame_00.jpg, ...)
            if has_images:
                print(f"Detected single image sequence in: {root_path}")
                process_image_sequence(root_path, output_dir / root_path.name)
                return
            
            raise ValueError(f"Could not determine content type of directory: {root_path}")

    
    # Case 5: explicit list of inputs
    else:
        print(f"Processing list of {len(input_paths)} inputs...")
        for item in input_paths:
            path = Path(item).resolve()
            if not path.exists():
                print(f"Warning: Skipping missing path: {path}")
                continue

            if path.is_dir():
                # Assume list elements pointing to dirs are image sequences
                process_image_sequence(path, output_dir / path.name)
            
            elif path.is_file() and is_video_file(path):
                keyframe_pipeline_from_video(path, output_dir / path.stem)
            
            else:
                print(f"Warning: Skipping unsupported file type: {path}")
