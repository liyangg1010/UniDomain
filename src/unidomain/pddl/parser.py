"""
PDDL Parser
===========

This module provides utilities for parsing PDDL domain strings and extracting
structural information such as predicates, actions, preconditions, and effects.
It handles the Lisp-like nested parenthesis structure of PDDL.

Architecture Position:
    [Tools / Parser] -> Used to analyze PDDL content.
"""

import json
from typing import Any, Dict, List, OrderedDict, Union

from unidomain.schemas.typings import Domain, Predicates

__all__ = [
    "analyze_pddl_action",
    "get_predicates_in_action",
    "get_predicate_name",
    "get_action_name",
    "extract_domain_from_pddl",
]


def _parse_pddl(s: str) -> Union[List[Any], str]:
    """Parse a PDDL string into a nested list structure (Lisp-style tokenizer).

    Args:
        s (str): The raw PDDL string.

    Returns:
        Union[List[Any], str]: A nested list representing the syntax tree.
    """

    # Simple tokenizer: separate parens and split by whitespace
    tokens = s.replace("(", " ( ").replace(")", " ) ").split()

    stack = []
    current = []
    for token in tokens:
        if token == "(":
            stack.append(current)
            current = []
        elif token == ")":
            if stack:
                parent = stack.pop()
                parent.append(current)
                current = parent
        else:
            current.append(token)
    
    # Return the first element if it"s a single root list
    return current[0] if current else current


def _extract_predicates(expr: Union[List[Any], str]) -> List[str]:
    """Recursively extract predicate names from a parsed expression.

    Handles logical operators (and, or, not, imply, exists, forall).

    Args:
        expr: A nested list or string from _parse_pddl.

    Returns:
        List[str]: A list of predicate names found in the expression.
    """
    predicates = []
    if not expr or not isinstance(expr, list):
        return predicates
    
    operator = expr[0]

    # Handle logical operators
    if operator in ["and", "or", "not", "imply"]:
        for sub_expr in expr[1:]:
            predicates.extend(_extract_predicates(sub_expr))

            # Special handling for "not": verify negation
            if operator == "not" and predicates:
                predicates[-1] = f"not {predicates[-1]}"
    
    # Handle quantifiers
    elif operator in ["exists", "forall"]:
        # Structure: (exists (?x ...) (expression))
        # The expression is usually the 3rd element (index 2)
        if len(expr) > 2:
            predicates.extend(_extract_predicates(expr[2]))
    
    # Base case: It"s a predicate (e.g., ["at", "?x", "?y"])
    else:
        predicates.append(operator)

    return predicates


def analyze_pddl_action(pddl_str: str) -> Dict[str, Union[str, List[str]]]:
    """Analyze a PDDL action string to extract its name and components.

    Args:
        pddl_str (str): The full PDDL action definition string.

    Returns:
        dict: A dictionary with keys:
            - "action_name": str
            - "precondition_predicates": List[str]
            - "effect_predicates": List[str]
    """
    parsed = _parse_pddl(pddl_str)
    
    result = {
        "action_name": "",
        "precondition_predicates": [],
        "effect_predicates": []
    }
    
    # Iterate through top-level tokens of the action definition
    i = 0
    while i < len(parsed):
        if parsed[i] == ":action":
            result["action_name"] = parsed[i+1]
            i += 2

        elif parsed[i] == ":precondition":
            expr = parsed[i+1]
            preds = _extract_predicates(expr)
            result["precondition_predicates"] = list(OrderedDict.fromkeys(preds))
            i += 2

        elif parsed[i] == ":effect":
            expr = parsed[i+1]
            preds = _extract_predicates(expr)
            result["effect_predicates"] = list(OrderedDict.fromkeys(preds))
            i += 2

        else:
            i += 1
    
    return result


def get_predicates_in_action(predicates: Predicates, action: str) -> Predicates:
    """Identify which global predicates are used within a specific action.

    Args:
        predicates (Predicates): The global dictionary of available predicates.
        action (str): The PDDL action string to analyze.

    Returns:
        Predicates: A subset dictionary containing only the used predicates.
    """

    result = analyze_pddl_action(action)

    # Combine preconditions and effects
    predicate_names = result["precondition_predicates"] + result["effect_predicates"]

    # Clean names (remove 'not' and spaces)
    predicate_names = [predicate.replace("not", "").lstrip() for predicate in predicate_names]
    
    return_predicates = {}
    for predicate_name in predicate_names:
        for predicate in predicates.keys():
            if predicate_name in predicate:
                return_predicates[predicate] = predicates[predicate]

    return return_predicates


def get_predicate_name(predicate: str) -> str:
    """Extract the bare name of a predicate from its definition.

    Example:
        Input: "(at ?x ?y)"
        Output: "at"

    Args:
        predicate (str): The PDDL predicate string.

    Returns:
        str: The name of the predicate.
    """
    # Remove parens and take the first token
    predicate_name = predicate.strip("() ").split()[0]
    return predicate_name


def get_action_name(action_string: str) -> str:
    """Extract the name of an action from its definition string.

    Args:
        action_string (str): The PDDL action definition.

    Returns:
        str: The action name.
    """

    # Usually the first line contains (:action name)
    action_name = action_string.split("\n")[0].replace("(:action", "").strip()
    return action_name

def extract_domain_from_pddl(domain_string: str) -> Domain:
    """Parse a complete PDDL domain file content into structural components.

    It extracts predicates (with comments/explanations) and operators.
    Also supports parsing a JSON representation if provided as a string.

    Args:
        domain_string (str): The file content of domain.pddl (or json).

    Returns:
        Domain: A tuple (predicates_dict, operators_dict).
    """
    
    # Try parsing as JSON first (Hybrid support)
    try:
        domain_data = json.loads(domain_string)
        if isinstance(domain_data, dict) and "predicates" in domain_data and "operators" in domain_data:
            return domain_data["predicates"], domain_data["operators"]
    except Exception:
        pass    # Fallback to PDDL parsing

    # Extract Predicates
    stack = []

    # Locate the (:predicates ...) block
    predicates_pos = domain_string.find("predicates")
    stack.append("(")
    predicates = ""
    for i in range(predicates_pos, len(domain_string)):
        if domain_string[i] == "(":
            stack.append(domain_string[i])
        elif domain_string[i] == ")":
            stack.pop()
            # Stop when stack is empty
            if len(stack) == 0:
                break
        predicates += domain_string[i]

    # Remove "predicates" and split by line
    predicates = predicates.replace("predicates", "").strip()
    predicates = predicates.split("\n")

    # Extract predicate defination and explanation
    predicates_dict = {}
    for predicate in predicates:
        predicate = predicate.strip()
        key, value = predicate.split(";")
        predicates_dict[key.strip()] = value.strip()

    # Extract operators from PDDL domain string
    # Locate all "(:actions ...)" blocks in domain string
    all_action_pos = []
    start = 0
    while True:
        action_pos = domain_string.find("(:action", start)
        if action_pos == -1:
            break
        all_action_pos.append(action_pos)
        start = action_pos + 1

    # Extract each action string from domain string
    all_action_pos.append(len(domain_string))
    actions: list[str] = []
    for i in range(len(all_action_pos) - 1):
        action = domain_string[all_action_pos[i]:all_action_pos[i+1]].rstrip()
        actions.append(action)
    actions[-1] = actions[-1].rstrip(")").rstrip()

    # Extract action name and return as a dict
    actions_dict = {}
    for action in actions:
        actions_dict[action.split("\n")[0].replace("(:action", "").strip()] = action.replace("\t", "")

    return predicates_dict, actions_dict
