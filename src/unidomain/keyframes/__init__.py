"""
Keyframe Extraction Module
==========================

This module implements the pre-processing stage that converts raw video data
or image sequences into a set of discrete keyframes. These keyframes serve as
the foundational visual observations for the atomic domain generation.

Architecture Dependency Graph
-----------------------------
The following diagram illustrates the call hierarchy and data flow within this module.
"A -> B" means A imports and calls B.

    [Orchestrator / Top Level]
           pipeline.py
                |
                | (Input dispatching, batch handling & configuration)
                v
    +-------------------------------------------------------+
    | [Media Processing / Middle Level]                     |
    |                                                       |
    |   processors.py                                       |
    |   (Video decoding via FFmpeg, frame selection logic)  |
    +-------------------------------------------------------+
                |
                | (Calculates pixel-wise energy profile)
                v
    [Core Algorithm / Bottom Level]
           energy.py

Component Descriptions
----------------------
- **pipeline.py**: The unified entry point. It intelligently detects input
  formats (video files, directories, lists) and dispatches them to the
  appropriate processing routines.

- **processors.py**: Handles the heavy lifting of media processing. It invokes
  FFmpeg for video decoding and coordinates the selection of frames based on
  energy metrics.
  
- **energy.py**: The mathematical core. It implements the pixel-wise energy
  calculation (based on Parseval's theorem) and local extremum detection to
  identify significant visual changes.
"""

from unidomain.keyframes.pipeline import keyframes_pipeline

__all__ = ["keyframes_pipeline"]
