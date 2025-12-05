"""
PDDL Utilities Module
=====================

This module acts as the "Toolbox" for PDDL manipulation. It provides low-level
functions to parse, generate, analyze, and visualize PDDL files (Domains and Problems).
It abstracts away the string processing complexity of the Lisp-like PDDL syntax.

Module Overview
---------------
- **io.py**: Handles serialization/deserialization between Python objects and
  JSON intermediate representations.

- **parser.py**: "Reader". Parses raw PDDL strings into structural dictionaries,
  extracting predicates, actions, and logic expressions.

- **generator.py**: "Writer". Converts internal data structures back into
  standard, indented PDDL file format.

- **planner.py**: "Solver". Wrapper for external planners (Fast Downward),
  handling process execution, timeouts, and result parsing.
  
- **domain_graph.py**: "Visualizer". Renders the dependency graph of predicates
  and operators using Graphviz.
"""

from unidomain.pddl.domain_graph import visualize_domain_graph
from unidomain.pddl.generator import domain2file, problem2file
from unidomain.pddl.io import load_domain, load_problem, save_domain, save_problem
from unidomain.pddl.parser import (
    analyze_pddl_action,
    extract_domain_from_pddl,
    get_action_name,
    get_predicate_name,
    get_predicates_in_action,
)
from unidomain.pddl.planner import PDDLPlanner, parse_plans

__all__ = [
    # IO
    "load_domain", "save_domain", "load_problem", "save_problem",
    # Parser
    "analyze_pddl_action", "extract_domain_from_pddl", "get_action_name",
    "get_predicate_name", "get_predicates_in_action",
    # Generator
    "domain2file", "problem2file",
    # Planner
    "PDDLPlanner", "parse_plans",
    # Visualization
    "visualize_domain_graph",
]
