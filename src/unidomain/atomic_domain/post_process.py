"""
Post-processing Utilities
=========================

This module provides low-level infrastructure for manipulating PDDL files and
intermediate JSON domain representations. It serves as the shared IO and
cleaning layer for the atomic domain generation pipeline.

Architecture Position:
    [Shared Infrastructure / Bottom Level] -> Called by initial_domain, revise_domain, etc.
"""

import re
from pathlib import Path

from unidomain.pddl.generator import domain2file
from unidomain.pddl.io import load_domain, save_domain
from unidomain.pddl.parser import extract_domain_from_pddl, get_predicate_name
from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import setup_path

__all__ = ["post_process_domain", "save_and_post_process"]


def _rename_domain(pddl_domain_path: Paths, new_domain_name: str = "domain_name") -> None:
    """Rename the domain name in a PDDL domain file.
    
    Args:
        pddl_domain_path (Paths): The path to the PDDL domain file.
        new_domain_name (str): The new domain name. Defaults to "domain_name".

    Returns:
        None
    """

    # Get PDDL string from the domain file
    pddl_domain_path = setup_path(pddl_domain_path, is_file=True, mkdir=False)
    with pddl_domain_path.open("r", encoding="utf-8") as f:
        domain = f.read()
    
    # Locate and replace the domain name in the PDDL domain string
    # \(domain\s+ : Matches literal "(domain" followed by whitespace
    # ([^)\s]+)   : Group 1, matches the name (anything not ')' or whitespace)
    pattern = re.compile(r"\(domain\s+([^)\s]+)", re.IGNORECASE)
    match = pattern.search(domain)

    if match:
        start_idx, end_idx = match.span(1)
        domain = domain[:start_idx] + new_domain_name + domain[end_idx:]

    # wirte back to the PDDL domain file
    with pddl_domain_path.open("w", encoding="utf-8") as f:
        f.write(domain)


def _rename_problem(
    pddl_problem_path: Paths,
    new_domain_name: str = "domain_name",
    new_problem_name: str = "problem_name"
) -> None:
    """Rename the domain name and problem name in a PDDL problem file.

    Args:
        pddl_problem_path (Paths): The path to the PDDL problem file.
        new_domain_name (str): The new domain name. Defaults to "domain_name".
        new_problem_name (str): The new problem name. Defaults to "problem_name".

    Returns:
        None
    """

    # Get PDDL string from the problem file
    pddl_problem_path = setup_path(pddl_problem_path, is_file=True, mkdir=False)
    with pddl_problem_path.open("r", encoding="utf-8") as f:
        problem = f.read()
    
    # Rename Problem Definition: (problem <name>)
    pattern = re.compile(r"\(problem\s+([^)\s]+)", re.IGNORECASE)
    match = pattern.search(problem)
    if match:
        start_idx, end_idx = match.span(1)
        problem = problem[:start_idx] + new_problem_name + problem[end_idx:]

    # Rename Domain Reference: (:domain <name>)
    pattern = re.compile(r"\(:domain\s+([^)\s]+)", re.IGNORECASE)
    match = pattern.search(problem)
    if match:
        start_idx, end_idx = match.span(1)
        problem = problem[:start_idx] + new_domain_name + problem[end_idx:]

    # Wirte back to the PDDL problem file
    with pddl_problem_path.open("w", encoding="utf-8") as f:
        f.write(problem)


def _remove_unused_predicates(domain_path: Paths) -> None:
    """Remove predicates that are not used in any operator from the JSON domain.

    Args:
        domain_path (Paths): The path to the domain file (JSON).

    Returns:
        None
    """

    # Load domain
    predicates, operators = load_domain(domain_path)
    unused_predicates = []

    # Get unused predicates
    for predicate in predicates:
        predicate_name = get_predicate_name(predicate)

        # Use regex to ensure the token boundary is matched.
        safe_name = re.escape(predicate_name)
        pattern = re.compile(rf"\({safe_name}[\s)]")

        for operator in operators.values():
            if pattern.search(operator):
                break
        else:
            unused_predicates.append(predicate)

    # Remove unused predicates
    for unused_predicate in unused_predicates:
        del predicates[unused_predicate]
    
    # Save back
    save_domain(predicates, operators, domain_path)


def post_process_domain(json_path: Paths, pddl_path: Paths) -> None:
    """Perform cleanup and formatting on the generated domain files.

    This includes removing unused predicates, converting JSON to PDDL,
    and ensuring standardized naming.

    Args:
        json_path (Paths): Path to the intermediate JSON domain file.
        pddl_path (Paths): Path to the output PDDL domain file.

    Returns:
        None
    """
    _remove_unused_predicates(json_path)
    domain2file(json_path, pddl_path)
    _rename_domain(pddl_path)


def save_and_post_process(domain_str: str, output_dir: Paths, stem: str) -> None:
    """Coordinate the saving, converting, and post-processing of a raw LLM output.

    Args:
        domain_str (str): The raw PDDL domain string generated by LLM.
        output_dir (Paths): The directory to save files.
        stem (str): The filename stem (without extension).

    Returns:
        None
    """
    output_dir = setup_path(output_dir)
    json_path = Path(output_dir / f"{stem}.json").resolve()
    pddl_path = Path(output_dir / f"{stem}.pddl").resolve()
    
    # Extracts structure from raw text string and save to JSON
    predicates, operators = extract_domain_from_pddl(domain_str)
    save_domain(predicates, operators, json_path)
    
    # Post-process (Clean, Rename & Generate final PDDL file)
    post_process_domain(json_path, pddl_path)
