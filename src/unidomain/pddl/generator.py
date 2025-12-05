"""
PDDL Generator
==============

This module is responsible for generating valid PDDL files (domain.pddl and
problem.pddl) from internal Python data structures or intermediate JSON files.
It handles the formatting, indentation, and assembly of PDDL blocks.

Architecture Position:
    [Tools / Generator] -> Used by pipelines to convert JSON intermediate states
    into executable PDDL files for the planner.
"""

import textwrap
from dataclasses import dataclass
from typing import List, Set

from unidomain.pddl.io import load_domain, load_problem
from unidomain.schemas.typings import Operators, Paths, Predicates
from unidomain.utils.runtime import setup_path

__all__ = ["domain2file", "problem2file"]

_DOMAIN_TEMPLATE = """
(define (domain {domain_name})
    (:requirements :strips :typing :equality :negative-preconditions :disjunctive-preconditions)
    (:predicates
{predicates}
    )

{operators}
)
"""

_PROBLEM_TEMPLATE = """
(define (problem {problem_name})
       (:domain {domain_name})
       (:objects
{objects}
       )
       (:init
{init}
       )
       (:goal
            {goal}
       )
)
"""

@dataclass
class _PDDLDomain():
    domain_name: str
    predicates: Predicates
    operators: Operators

@dataclass
class _PDDLProblem():
    problem_name: str
    objects: Set[str]
    init: Set[str]
    goal: str

class _PDDLGenerator:
    """Internal helper to assemble and write PDDL files."""

    def __init__(self, domain_name: str = "domain_name", problem_name: str = "problem_name") -> None:
        self._domain = _PDDLDomain(domain_name=domain_name, predicates={}, operators={})
        self._problem = _PDDLProblem(problem_name=problem_name, objects=set(), init=set(), goal="")
        
    def update_predicate(self, predicate_dict: Predicates) -> None:
        """Update PDDL domain predicates."""
        for name, comments in predicate_dict.items():
            self._domain.predicates[name] = comments

    def update_operator(self, operator_dict: Operators) -> None:
        """Update PDDL domain operators."""
        for name, code in operator_dict.items():
            self._domain.operators[name] = code

    def write_domain_file(self, path: Paths) -> None:
        """Format and write the PDDL domain file."""

        # Format predicates: "(name ...) ; comment"
        predicate = [f"{p[0]} ; {p[1]}" for p in self._domain.predicates.items()]
        predicates = "\n".join(predicate)
        predicates = textwrap.indent(predicates, "\t\t")

        # Format operators
        operators = "\n\n".join(list(self._domain.operators.values()))
        operators = textwrap.indent(operators, "\t")

        # Assemble content
        domain = _DOMAIN_TEMPLATE.format(
            domain_name=self._domain.domain_name,
            predicates=predicates,
            operators=operators
        )
        domain = domain.lstrip("\n")

        # Write to file
        domain_file = setup_path(path, is_file=True)
        with domain_file.open("w", encoding="utf-8") as f:
            f.write(domain)

    def update_objects(self, objects: List[str]) -> None:
        """Update PDDL problem objects."""
        self._problem.objects.update(objects)
    
    def set_init(self, predicates: List[str]) -> None:
        """Set PDDL problem init state."""
        self._problem.init.update(predicates)

    def set_goal(self, goal: str) -> None:
        """Set PDDL problem goal."""
        self._problem.goal = goal.strip()

    def write_problem_file(self, path: Paths) -> None:
        """Format and write the PDDL problem file."""
        
        # Format objects: wrap text to 80 chars for readability
        objects = " ".join(list(self._problem.objects))
        objects = textwrap.wrap(objects, width=80)
        objects = "\n".join(objects)
        objects = textwrap.indent(objects, "\t\t\t")

        # Format init: one predicate per line
        init = "\n".join(list(self._problem.init))
        init = textwrap.indent(init, "\t\t\t")

        # Format goal
        goal = self._problem.goal

        # Assemble content
        problem = _PROBLEM_TEMPLATE.format(
            domain_name=self._domain.domain_name,
            problem_name=self._problem.problem_name,
            objects=objects,
            init=init,
            goal=goal,
        )
        problem = problem.lstrip("\n")

        # Write to file
        problem_file = setup_path(path, is_file=True)
        with problem_file.open("w", encoding="utf-8") as f:
            f.write(problem)


def domain2file(
    json_path: Paths,
    save_path: Paths,
) -> None:
    """Convert a JSON domain representation to a PDDL domain file.
    
    Args:
        json_path (Paths): Path to the source JSON file.
        save_path (Paths): Path to save the generated .pddl file.

    Returns:
        None
    """

    json_path = setup_path(json_path, is_file=True, mkdir=False)
    save_path = setup_path(save_path, is_file=True)

    # Generate PDDL domain
    predicates, operators = load_domain(json_path)
    generator = _PDDLGenerator()
    generator.update_operator(operators)
    generator.update_predicate(predicates)

    # write PDDL domain
    generator.write_domain_file(save_path)

def problem2file(
    json_path: Paths,
    save_path: Paths
) -> None:
    """Convert a JSON problem representation to a PDDL problem file.

    Args:
        json_path (Paths): Path to the source JSON file.
        save_path (Paths): Path to save the generated .pddl file.

    Returns:
        None
    """

    json_path = setup_path(json_path, is_file=True, mkdir=False)
    save_path = setup_path(save_path, is_file=True)

    # Generate PDDL problem
    objects, init, goal = load_problem(json_path)
    generator = _PDDLGenerator()
    generator.update_objects(objects)
    generator.set_init(init)
    generator.set_goal(goal)

    # write PDDL problem
    generator.write_problem_file(save_path)
