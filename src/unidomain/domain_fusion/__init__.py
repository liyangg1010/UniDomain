"""
Domain Fusion Module
====================

This module implements the merging layer, designed to integrate multiple
atomic domains into a single "meta domain". It utilizes a parallel binary
tree merging strategy to efficiently reconcile predicate definitions and
operator logic across different source domains, ensuring a consistent and
comprehensive knowledge base.

Architecture Dependency Graph
-----------------------------
The following diagram illustrates the call hierarchy and data flow within this module.
"A -> B" means A imports and calls B.

    [Orchestrator / Top Level]
           pipeline.py
                |
                | (pre-processing & pipeline orchestration)
                v
    [Execution Engine / High Level]
           runner.py       ------------->        tree.py
    (Binary Tree Fusion Strategy)     (Tree Structure & Visualization)
                |
                | (invokes specific merging algorithms)
                v
    +---------------------------------------------------------------+
    | [Core Algorithms / Middle Level]                              |
    | (Implementation of specific PDDL component merging logic)     |
    |                                                               |
    |   1. predicate_merging.py        2. operator_merging.py       |
    |      (Predicate Mapping)            (Action Unification)      |
    +---------------------------------------------------------------+
                |
                | (computes semantic similarity & embeddings)
                v
    [Shared Infrastructure / Bottom Level]
           similarity.py

Component Descriptions
----------------------
- **pipeline.py**: The unified entry point. Handles pre-processing and calls the
  runner to execute the fusion pipeline.

- **runner.py & tree.py**: The execution engine. `runner.py` orchestrates the
  merging process (often pairwise), while `tree.py` encapsulates the binary tree
  logic for parallel multi-domain fusion.

- **predicate_merging.py & operator_merging.py**: The core logic modules that
  define how actions and predicates from different domains are merged.

- **similarity.py**: Low-level utility for calculating vector similarities, used
  extensively by the merging algorithms.
"""

from unidomain.domain_fusion.pipeline import domain_fusion_pipeline

__all__ = ["domain_fusion_pipeline"]
