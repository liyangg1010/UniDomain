"""
Task Planner Module
===================

This module serves as the execution engine for solving downstream tasks using
the fused domain. It features an intelligent domain filtering mechanism to
prune irrelevant operators for specific tasks, and orchestrates the interaction
between LLMs (for problem grounding) and symbolic PDDL planners to generate
executable action sequences.

Architecture Dependency Graph
-----------------------------
The following diagram illustrates the call hierarchy and data flow within this module.
"A -> B" means A imports and calls B.

    [Orchestrator / Top Level]
           pipeline.py
                |
                | (coordinates execution flow)
                v
    +---------------------------------------------------------------+
    | [Core Algorithms / Functional Level]                          |
    | (Implementation of specific planning sub-routines)            |
    |                                                               |
    |   1. filtering.py               2. planning.py                |
    |      (Domain Pruning)              (Plan Generation)          |
    +---------------------------------------------------------------+

Component Descriptions
----------------------
- **pipeline.py**: The unified entry point. It orchestrates the workflow by
  invoking the filtering mechanism to optimize the domain, followed by the
  planning mechanism to solve the task.

- **filtering.py**: A functional module responsible for identifying and
  retaining only the relevant operators and predicates for a specific task to
  reduce search space.
  
- **planning.py**: A functional module responsible for interfacing with the PDDL
  planner to generate valid action sequences based on the (potentially filtered)
  domain.
"""

from unidomain.task_planner.pipeline import task_planner_batch_pipeline, task_planner_pipeline

__all__ = ["task_planner_pipeline", "task_planner_batch_pipeline"]
