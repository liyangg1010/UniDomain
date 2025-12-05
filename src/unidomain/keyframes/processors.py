"""
Media Processing Logic
======================

This module handles the intermediate media processing tasks for keyframe extraction.
It includes functionality for decoding videos into frames (via FFmpeg) and
coordinating the energy-based selection logic to extract valid image sequences.

Architecture Position:
    [Media Processing / Middle Level] -> Called by keyframes.pipeline
    Uses keyframes.energy for calculation.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from unidomain.configs.settings import config
from unidomain.keyframes.energy import get_keyframe_indices
from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import get_sorted_paths, is_image_file, setup_path

__all__ = ["process_image_sequence", "keyframe_pipeline_from_video"]


def _extract_frames_from_video(video_path: Paths, output_frames_dir: Paths, quality: int = 2) -> None:
    """Extract frames from a video file using FFmpeg.

    Args:
        video_path (Paths): Path to the source video file.
        output_frames_dir (Paths): Directory to store extracted frames.
        quality (int): JPEG quality scale (2-31, lower is better). Defaults to 2.

    Returns:
        None

    Raises:
        FileNotFoundError: If FFmpeg is not installed on the system.
        subprocess.CalledProcessError: If FFmpeg execution fails.
    """

    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("FFmpeg tool is not found. Please install FFmpeg to process videos.")
    
    # Construct command: -q:v controls JPEG quality
    command = [
        'ffmpeg',
        '-i', str(video_path),
        '-q:v', str(quality),
        f'{output_frames_dir}/%08d.jpg'
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8") if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg failed to extract frames: {error_msg}")
    
    print(f"Extract frames from {video_path} to {output_frames_dir} successfully!")


def process_image_sequence(image_dir: Paths, output_dir: Paths) -> None:
    """Select and extract keyframes from a directory of images.

    This function calculates the adaptive window step based on sequence length,
    identifies keyframe indices using energy profiles, and copies the selected
    files to the output directory.

    Args:
        image_dir (Paths): Directory containing the raw image sequence.
        output_dir (Paths): Directory to save the selected keyframes.

    Returns:
        None.
    """

    image_dir = setup_path(image_dir)
    output_dir = setup_path(output_dir)

    # Count number of frames and calculate hyper parameter of energy-based keyframe extraction
    print(f"--- Extracting keyframes from {image_dir.name} ---")
    image_files = get_sorted_paths(image_dir, filter_func=is_image_file)
    num_images = len(image_files)

    if num_images == 0:
        print(f"Warning: there are no images in {image_dir}")
        return

    # Get hyperparameter config for the window step
    window_step = config.keyframe.hyperparam_delta(num_images)
    print(f"Found {num_images} frames in {image_dir.name}. Set window step to {window_step}.")

    # Get keyframe indices
    keyframe_indices = get_keyframe_indices(image_dir, delta=window_step)
    print(f"Keyframe indices: {keyframe_indices}")

    # Copy keyframes to output directory
    print(f"Copying keyframes to {output_dir}")
    for index in keyframe_indices:
        if 0 <= index < num_images:
            source_path = image_files[index]
            destination_path = output_dir / source_path.name
            shutil.copy2(source_path, destination_path)
        else:
            raise ValueError(f"Index {index} out of bounds for {num_images} images.")

    print(f"Successfully extracted {len(keyframe_indices)} keyframes to {output_dir}")
    print(f"--- Keyframe extraction of {image_dir.name} completed! ---")


def keyframe_pipeline_from_video(video_path: Paths, output_dir: Paths) -> None:
    """Extract keyframes directly from a video file.

    This is a high-level wrapper that handles temporary frame extraction and cleanup.

    Args:
        video_path (Paths): Path to the source video file.
        output_dir (Paths): Directory to save the final keyframes.

    Returns:
        None.
    """

    video_path = setup_path(video_path, is_file=True, mkdir=False)
    output_dir = setup_path(output_dir)

    # Extract frames to temporary directory
    print(f"--- Extracting keyframes from video: {video_path.name} ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        print(f"Extracting raw frames using FFmpeg from {video_path}")
        _extract_frames_from_video(video_path, temp_dir)

        # Extract keyframes from the frames
        process_image_sequence(temp_dir, output_dir)
    
    print(f"--- --- Keyframe extraction of video {video_path.name} completed! ---")
