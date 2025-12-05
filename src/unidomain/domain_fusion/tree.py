"""
Domain Fusion Tree Structure
============================

This module defines the data structures and serialization logic for the Domain Fusion Tree.
It uses a binary tree structure to manage the parallel merging of multiple domains.

Architecture Position:
    [Execution Engine / High Level] -> Used by runner.py to manage fusion state.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from anytree import NodeMixin
from anytree.exporter import DotExporter

from unidomain.pddl.io import load_domain
from unidomain.schemas.typings import Paths
from unidomain.utils.runtime import setup_path

__all__ = [
    "DomainFusionNode",
    "load_domain_fusion_tree",
    "save_domain_fusion_tree",
    "calculate_domain_size",
    "visualize_domain_fusion_tree",
]


@dataclass
class DomainFusionNode():
    """Data model representing a node in the domain fusion binary tree."""

    node_id: int
    domain_path: Path
    number_of_predicates: int = 0
    number_of_actions: int = 0
    left: Optional[DomainFusionNode] = None
    right: Optional[DomainFusionNode] = None


def load_domain_fusion_tree(load_path: Paths) -> DomainFusionNode:
    """Load the root node of the domain fusion tree from a pickle file.
    
    Args:
        load_path (Paths): The path to the pickle file.
        
    Returns:
        DomainFusionNode: The root node of the restored domain fusion tree.
    """
    load_path = setup_path(load_path, is_file=True, mkdir=False)
    with load_path.open("rb") as f:
        return pickle.load(f)


def save_domain_fusion_tree(root_node: DomainFusionNode, save_path: Paths) -> None:
    """Save the root node of the domain fuion tree to a pickle file.

    Args:
        root_node (DomainFusionNode): The root node of the tree to save.
        save_path (Paths): The path to save the pickle file.
    
    Returns:
        None
    """
    save_path = setup_path(save_path, is_file=True)
    with save_path.open("wb") as f:
        pickle.dump(root_node, f)


def calculate_domain_size(root_node: DomainFusionNode) -> None:
    """Recursively calculate and update predicate/action counts for the tree.
    
    This function performs an in-place update on the provided node and its children.
    It loads the PDDL domain file referenced by "domain_path" to count elements.

    Args:
        root_node (DomainFusionNode): The root node of the domain fusion tree.

    Returns:
        None
    """
    # Parse the PDDL domain to get statistics
    predicates, operators = load_domain(root_node.domain_path)
    root_node.number_of_actions = len(operators)
    root_node.number_of_predicates = len(predicates)

    # Recursive calls
    if root_node.left:
        calculate_domain_size(root_node.left)
    if root_node.right:
        calculate_domain_size(root_node.right)
    

class _AnyTreeNode(NodeMixin):
    """Anytree node for visualization.

    Wraps a DomainFusionNode to provide the tree structure required by "anytree".
    """
    def __init__(self, node: DomainFusionNode, parent: Optional[_AnyTreeNode] = None) -> None:
        self.name = node.node_id
        self.node = node
        self.parent = parent


def _build_anytree(node: DomainFusionNode, parent: Optional[_AnyTreeNode] = None) -> _AnyTreeNode:
    """Convert a DomainFusionNode into an anytree node structure for visualization.
    
    Args:
        node (DomainFusionNode): The current node in the domain fusion tree.
        parent (Optional[_AnyTreeNode]): The parent node of the current node. Defaults to None.
        
    Returns:
        _AnyTreeNode: The root of the constructed anytree structure.
    """
    tree_node = _AnyTreeNode(node, parent)
    if node.left:
        _build_anytree(node.left, parent=tree_node)
    if node.right:
        _build_anytree(node.right, parent=tree_node)
    return tree_node


def visualize_domain_fusion_tree(root_node: DomainFusionNode, save_path: Paths) -> None:
    """Render the domain fusion tree structure to an image file.
    
    Requires Graphviz to be installed on the system.

    Args:
        root_node (DomainFusionNode): The root node of the domain fusion tree to visualize.
        save_path (Paths): The output path for the image (e.g., "tree.png").
    
    Returns:
        None
    """
    def node_attr(node: _AnyTreeNode) -> str:
        # Create an HTML-like label for Graphviz
        html_label = (
            f'<<FONT COLOR="black">{node.node.node_id}</FONT><BR/>'
            f'<FONT COLOR="red">p: {node.node.number_of_predicates}</FONT><BR/>'
            f'<FONT COLOR="red">a: {node.node.number_of_actions}</FONT>>'
        )
        return f'label={html_label}'
    
    def edge_attr(node: _AnyTreeNode, child: _AnyTreeNode) -> str:
        return "dir=back"

    save_path = setup_path(save_path, is_file=True)
    anytree_root = _build_anytree(root_node)
    DotExporter(anytree_root, nodeattrfunc=node_attr, edgeattrfunc=edge_attr).to_picture(str(save_path))
