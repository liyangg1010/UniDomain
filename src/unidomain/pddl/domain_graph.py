"""
Domain Graph Visualization
==========================

This module converts the logical structure of a PDDL domain (Predicates and Operators)
into a visual directed graph using Graphviz. It visualizes the causal flow:
Preconditions (State) -> Operators (Action) -> Effects (State Change).

Architecture Position:
    [Tools / Visualization] -> Used by pipelines to generate PDF/PNG visualizations of the domain.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List

from graphviz import Digraph

from unidomain.pddl import parser
from unidomain.pddl.io import load_domain
from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import setup_path

__all__ = ["visualize_domain_graph"]


@dataclass
class _DomainGraph:
    """Internal representation of the domain connectivity graph.
    
    Attributes:
        precondition_map: Mapping from Predicate -> List of Operators that require it.
                          Represents edges: Predicate -> Operator.
        effect_map: Mapping from Predicate -> List of Operators that produce it.
                    Represents edges: Operator -> Predicate.
    """
    precondition_map: DefaultDict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    effect_map: DefaultDict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def add_precondition(self, predicate: str, operator: str) -> None:
        """Add a dependency edge: Predicate -> Operator."""
        self.precondition_map[predicate].append(operator)

    def add_effect(self, predicate: str, operator: str) -> None:
        """Add a production edge: Operator -> Predicate."""
        self.effect_map[predicate].append(operator)


def _domain2graph(domain_path: Paths) -> _DomainGraph:
    """Parse a PDDL JSON domain into a graph structure.

    Args:
        domain_path (Paths): Path to the PDDL domain file (JSON format).

    Returns:
        _DomainGraph: The constructed connectivity graph.
    """
    _, operators = load_domain(domain_path)
    graph = _DomainGraph()

    for operator_str in operators.values():
        # Analyze syntax to extract dependencies
        result = parser.analyze_pddl_action(operator_str)
        action_name = result["action_name"]
        preconditions = result["precondition_predicates"]
        effects = result["effect_predicates"]
        
        for precondition in preconditions:
            graph.add_precondition(precondition, action_name)
        for effect in effects:
            graph.add_effect(effect, action_name)

    return graph


def _render_graph(domain_graph: _DomainGraph, save_path: Paths) -> None:
    """Render the domain graph using Graphviz and save to file.

    The graph is organized hierarchically:
    - Top Level: Preconditions (Input State)
    - Mid Level: Operators (Actions)
    - Bottom Level: Effects (Output State, excluding those that are also inputs)

    Args:
        domain_graph (_DomainGraph): The graph data structure.
        save_path (Paths): Destination path for the rendered image.
    
    Returns:
        None
    """
    save_path = setup_path(save_path, is_file=True)

    # Handle filename extension logic for Graphviz
    # Digraph.render(filename) adds the format extension automatically.
    # Strip the extension from save_path to avoid double extensions (e.g., .pdf.pdf)
    output_format = save_path.suffix.lstrip(".") or "pdf"
    output_stem = save_path.parent / save_path.stem

    # Initialize Graphviz Digraph
    dot = Digraph(engine="dot", format=output_format)
    dot.attr(
        rankdir="TB",
        size="20, 15",
        nodesep="0.5",
        ranksep="1.0",
    )

    # Collect all operator nodes
    all_operators = set()
    for ops in domain_graph.precondition_map.values():
        all_operators.update(ops)
    for ops in domain_graph.effect_map.values():
        all_operators.update(ops)

    # --- Level 1: Preconditions (Inputs) ---
    with dot.subgraph() as pre:
        # pre.attr(label="Preconditions", style="dashed", color="gray")
        for predicate in domain_graph.precondition_map:
            pre.node(predicate, label=predicate, shape="ellipse", style="filled", fillcolor="lightgreen")
        pre.attr(rank="same")

    # --- Level 2: Operators (Actions) ---
    with dot.subgraph() as mid:
        # mid.attr(label="Actions", style="dashed", color="gray")
        for op in all_operators:
            mid.node(op, label=op, shape="box", style="filled", fillcolor="lightblue")
        mid.attr(rank="same")

    # --- Level 3: Effects (Outputs) ---
    # Only show predicates that are purely effects (not also inputs) in the bottom layer
    # to highlight state changes.
    effect_only_predicates = set(domain_graph.effect_map.keys()) - set(domain_graph.precondition_map.keys())
    with dot.subgraph() as post:
        # post.attr(label="Effects", style="dashed", color="gray")
        for predicate in effect_only_predicates:
            post.node(predicate, label=predicate, shape="ellipse", style="filled", fillcolor="yellow")
        post.attr(rank="same")

    # --- Edges ---
    # Precondition -> Operator (Black)
    for predicate, ops in domain_graph.precondition_map.items():
        for op in ops:
            dot.edge(predicate, op, color="black")

    # Operator -> Effect (Red)
    for predicate, ops in domain_graph.effect_map.items():
        for op in ops:
            dot.edge(op, predicate, color="red")

    # --- Legend ---
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(
            label="Legend",
            labelloc="b",
            rank="sink",
            style="rounded,filled",
            color="lightgray",
            fontsize="6"
        )

    dot.render(str(output_stem), cleanup=True, view=False)


def visualize_domain_graph(domain_path: Paths, save_path: Paths) -> None:
    """Generate and save a visual graph of the PDDL domain structure.

    Args:
        domain_path (Paths): Path to the PDDL domain file (JSON format).
        save_path (Paths): Path where the visualization (PDF/PNG) will be saved.

    Returns:
        None
    """
    graph = _domain2graph(domain_path)
    _render_graph(graph, save_path)
