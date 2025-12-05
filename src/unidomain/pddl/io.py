"""
PDDL I/O Utilities
==================

This module handles the input/output operations for PDDL structures.
It serves as the persistence layer, serializing the internal Python representations
of Domains and Problems into JSON format for intermediate storage, and deserializing
them back for processing.

Architecture Position:
    [Tools / Persistence] -> Used by pipelines to save/load intermediate states.
"""

import json

from unidomain.schemas.typings import Domain, Goal, Init, Objects, Operators, Paths, Predicates, Problem
from unidomain.utils.runtime import setup_path

__all__ = ["load_domain", "load_problem", "save_domain", "save_problem"]

def load_domain(load_path: Paths) -> Domain:
    """Load a PDDL domain definition from a JSON file.

    The JSON file is expected to contain "predicates" and "operators" keys.

    Args:
        load_path (Paths): The path to the source JSON file.

    Returns:
        Domain: A tuple containing (predicates_dict, operators_dict).
    """
    
    load_path = setup_path(load_path, is_file=True, mkdir=False)

    with load_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    predicates = data["predicates"]
    operators = data["operators"]

    return predicates, operators
    
def save_domain(
    predicates: Predicates,
    operators: Operators,
    save_path: Paths
) -> None:
    """Serialize the PDDL domain structure to a JSON file.

    Args:
        predicates (Predicates): The dictionary of domain predicates.
        operators (Operators): The dictionary of domain operators/actions.
        save_path (Paths): The destination path for the JSON file.

    Returns:
        None
    """
    save_path = setup_path(save_path, is_file=True)

    json_domain = {"predicates": predicates, "operators": operators}
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(json_domain, f, ensure_ascii=False, indent=4)


def load_problem(load_path: Paths) -> Problem:
    """Load a PDDL problem definition from a JSON file.

    The JSON file is expected to contain "objects", "init", and "goal" keys.

    Args:
        load_path (Paths): The path to the source JSON file.

    Returns:
        Problem: A tuple containing (objects_list, init_list, goal_string).
    """
    
    load_path = setup_path(load_path, is_file=True, mkdir=False)
    with load_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data["objects"]
    init = data["init"]
    goal = data["goal"]

    return objects, init, goal

def save_problem(
    objects: Objects,
    init: Init,
    goal: Goal,
    save_path: Paths
) -> None:
    """Serialize the PDDL problem structure to a JSON file.

    Args:
        objects (Objects): List of object definitions.
        init (Init): List of initial state predicates.
        goal (Goal): The PDDL goal string.
        save_path (Paths): The destination path for the JSON file.

    Returns:
        None
    """

    save_path = setup_path(save_path, is_file=True)

    json_problem = {"objects": objects, "init": init, "goal": goal}
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(json_problem, f, ensure_ascii=False, indent=4)
