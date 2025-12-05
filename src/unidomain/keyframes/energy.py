"""
Keyframe Energy Calculation
===========================

This module implements the core mathematical algorithms for identifying keyframes
based on image energy profiles. It calculates the pixel-wise energy of image
sequences and identifies local extremes (maxima and minima) to select representative frames.

Architecture Position:
    [Core Algorithm / Bottom Level] -> Called by keyframes.processors
"""

from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import get_sorted_paths, is_image_file, setup_path

__all__ = ["get_keyframe_indices"]


def _gray_image_energy(gray_image: NDArray) -> float:
    """Calculate the energy of a grayscale image.

    Energy is defined as the sum of squared pixel values: E = sum_{x,y} |f(x,y)|^2.
    This metric captures the overall intensity/information content of the frame.

    Theoretical Note:
        According to Parseval's Theorem, this spatial domain energy is equivalent
        to the energy in the frequency (Fourier) domain. Therefore, this metric
        effectively quantifies the total signal power or information content
        of the frame, serving as a robust indicator for keyframe selection.

    Args:
        gray_image (Image.Image): A PIL Image object in 'L' (grayscale) mode.

    Returns:
        float: The calculated total energy.
    """

    # Energy of image f = \sum_{x,y} |f(x,y)|^2
    return np.sum(np.array(gray_image, dtype=np.float64) ** 2)


def _calculate_sequence_energies(frame_path_list: List[Path]) -> NDArray:
    """Compute the energy profile for a sequence of image files.

    Args:
        frame_path_list (List[Path]): Sorted list of paths to image frames.

    Returns:
        NDArray: A 1D numpy array containing the energy value for each frame.
    """
    energies = []
    
    for frame_path in tqdm(frame_path_list, desc="Calculating pixel-space energies"):
        with Image.open(frame_path) as img:
            gray_image = img.convert("L")
        
        energy = _gray_image_energy(gray_image)
        energies.append(energy)
    
    return np.array(energies)


def _select_keyframe_indices(energies: NDArray, delta: int) -> NDArray:
    """Identify keyframe indices based on local energy extremes.

    Keyframes are selected at local maxima (peaks of change/intensity) and
    local minima (valleys of stability) within a sliding window defined by delta.

    Args:
        energies (NDArray): 1D array of frame energies.
        delta (int): The radius of the local neighborhood window.

    Returns:
        NDArray: Sorted, unique indices of selected keyframes.
    """

    total_frames = len(energies)
    if total_frames == 0:
        return np.array([], dtype=int)

    # Always include the start and end frames
    keyframe_indices = [0]  # The first keyframe is the first frame

    # Iterate through the sequence to find local extremes
    # Note: A monotonic stack could optimize this to O(N), but the sliding
    # window O(N * delta) is sufficient given typical video frame counts.
    for center in range(total_frames):
        # Define local neighborhood boundary
        left = max(center - delta, 0)
        right = min(center + delta, total_frames - 1)

        window = energies[left:right + 1]

        # Extract the extreme point as the keyframe index
        if window.argmax() == center - left:
            keyframe_indices.append(center)
        if window.argmin() == center - left:
            keyframe_indices.append(center)

    keyframe_indices.append(total_frames - 1)    # The last keyframe is the last frame
    
    # Remove duplicates (e.g., if index 0 is also a local max) and sort
    return np.unique(keyframe_indices)


def get_keyframe_indices(frames_dir: Paths, delta: int) -> NDArray:
    """Calculate energy profile and select keyframes from a directory of images.

    This is the main entry point for the energy-based selection algorithm.
    Energies are calculated by the sum of square of pixel values of the original image.
    Keyframes are selected by the extreme point of the energies.
    Frames in the directory should be named as "00001.png", "00002.png", etc.
       
    Args:
        frames_dir (Paths): Directory containing the extracted video frames.
        delta (int): The local neighborhood range for extremum detection.
        
    Returns:
        NDArray: An array of integer indices representing the selected keyframes.
    """
    frames_dir = setup_path(frames_dir, mkdir=False)

    # Filter and sort frame paths
    # Keep only images, filtering out other files (e.g., .DS_Store)
    frame_path_list = get_sorted_paths(frames_dir, filter_func=is_image_file)

    if not frame_path_list:
        return np.array([])

    # calculate energies and select indices
    energies = _calculate_sequence_energies(frame_path_list)
    indices = _select_keyframe_indices(energies, delta)
    
    return indices
