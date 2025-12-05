"""
Atomic Domain Generation Module
===============================

This module is responsible for generating an executable PDDL atomic domain from
visual observations (keyframes). It employs a Neuro-Symbolic pipeline that first
leverages Vision-Language Models (VLMs) to perceive environmental dynamics,
followed by an iterative Large Language Model (LLM) refinement loop that ensures
syntactic correctness, solvability, and semantic validity.

Architecture Dependency Graph
-----------------------------
The following diagram illustrates the call hierarchy and data flow within this module.
"A -> B" means A imports and calls B.

    [Orchestrator / Top Level]
           pipeline.py
                |
                | (coordinates execution flow)
                v
    +-----------------------------------------------------------------------+
    | [Functional Steps / Middle Level]                                     |
    | (Each handles a specific stage of atomic domain generation)           |
    |                                                                       |
    |  1. initial_domain.py        2. revise_domain.py                      |
    |     (VLM Generation)            (Holistic Revision)                   |
    |                                                                       |
    |  3. solvability_check.py     4. solution_verification.py              |
    |     (Check & Refine)            (Verify & Refine)                     |
    +-----------------------------------------------------------------------+
                |
                | (uses shared logic for IO, cleaning, renaming)
                v
    [Shared Infrastructure / Bottom Level]
           post_process.py

Component Descriptions
----------------------
- **pipeline.py**: The only entry point for external calls. Manages state loops.

- **Functional Steps**: Independent modules containing specific LLM logic.

- **post_process.py**: Low-level services for PDDL file manipulation, used by all steps.
"""

from unidomain.atomic_domain.pipeline import atomic_domain_batch_pipeline, atomic_domain_pipeline

__all__ = ["atomic_domain_pipeline", "atomic_domain_batch_pipeline"]
