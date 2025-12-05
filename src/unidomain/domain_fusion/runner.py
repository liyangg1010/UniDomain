"""
Domain Fusion Runner
====================

This module orchestrates the execution of the domain fusion process. It manages
the parallel merging of multiple atomic domains using a balanced binary tree strategy.

Architecture Dependency Graph:
    [Top Level] pipeline.py
         |
         v
    [This Module] runner.py  <-- Orchestrates threading and state
         |
         +--> operator_merging.py (Core Logic)
         +--> predicate_merging.py (Core Logic)
         +--> tree.py (Data Structure)
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Callable, List, TypeVar

from unidomain.configs.constants import File
from unidomain.configs.settings import config
from unidomain.domain_fusion.operator_merging import merge_operators
from unidomain.domain_fusion.predicate_merging import merge_predicates
from unidomain.domain_fusion.tree import DomainFusionNode, calculate_domain_size, visualize_domain_fusion_tree
from unidomain.pddl.domain_graph import visualize_domain_graph
from unidomain.pddl.generator import domain2file
from unidomain.pddl.io import save_domain
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import register_force_exit, setup_path

__all__ = ["fuse_two_domains", "run_domain_fusion"]

R = TypeVar("R")


def fuse_two_domains(
    domain_1_path: Paths,
    domain_2_path: Paths,
    llm_agent: LLMAgent,
    save_path: Paths,
    predicate_threshold: float,
    operator_threshold: float,
) -> None:
    """Merge two domains into one domain.

    This process involves two steps:
    1. Predicate Merging: Merging semantically equivalent predicates.
    2. Operator Merging: Merging actions based on merged predicates.

    Args:
        domain_1_path (Paths): Path to the first domain file.
        domain_2_path (Paths): Path to the second domain file.
        llm_agent (LLMAgent): The LLM agent service.
        save_path (Paths): The path to save the merged domain (JSON).
        predicate_threshold (float): Similarity threshold for predicate matching.
        operator_threshold (float): Similarity threshold for operator matching.
    
    Returns:
        None
    """

    # Merge predicates
    domain_1, domain_2 = merge_predicates(
        domain_1_path=domain_1_path,
        domain_2_path=domain_2_path,
        llm_agent=llm_agent,
        threshold=predicate_threshold
    )

    # Merge operators
    predicates, operators = merge_operators(
        domain_1=domain_1,
        domain_2=domain_2,
        llm_agent=llm_agent,
        threshold=operator_threshold
    )

    # Save result
    save_domain(predicates, operators, save_path)


class _IDCounter:
    """Thread-safe ID counter that guarantees unique incremental values."""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def get_and_increment(self) -> int:
        """Atomically retrieve the current ID and increment it by one."""
        with self._lock:
            current_value = self._value
            self._value += 1
            return current_value


def _domain_fusion_node_wrapper(
    left_node: DomainFusionNode,
    right_node: DomainFusionNode,
    base_dir: Paths,
    counter: _IDCounter,
) -> DomainFusionNode:
    """Wrapper function for processing a single fusion node in a thread.

    This function handles the file system setup, logging, and execution of the
    pairwise fusion logic.

    Args:
        left_node (DomainFusionNode): The left child node.
        right_node (DomainFusionNode): The right child node.
        base_dir (Paths): The base directory to store merged results.
        counter (_IDCounter): Thread-safe ID counter for generating new node IDs.

    Returns:
        DomainFusionNode: A new node representing the merged result.
    """
    base_dir = setup_path(base_dir)
    execution_logger = get_task_logger(base_dir)

    # Generate new unique ID
    new_node_id = counter.get_and_increment()
    execution_logger.info(f"Merging node {left_node.node_id} and {right_node.node_id} to new node {new_node_id}")

    # Setup directory structure in new node
    new_node_dir = base_dir / str(new_node_id)
    new_node_dir.mkdir(exist_ok=True, parents=True)
    merged_domain_path = new_node_dir / File.META_DOMAIN_JSON

    # Check resume logic
    if not merged_domain_path.exists():
        # Clear directory to ensure no stale files exist
        for file in new_node_dir.iterdir():
            file.unlink()

        # Execute Core Fusion Logic
        fuse_two_domains(
            domain_1_path=left_node.domain_path,
            domain_2_path=right_node.domain_path,
            llm_agent=LLMAgent(config.domain_fusion.llm_model_config, log_dir=new_node_dir),
            save_path=merged_domain_path,
            predicate_threshold=config.domain_fusion.predicate_threshold,
            operator_threshold=config.domain_fusion.operator_threshold
        )

        # Generate PDDL and Visualization
        domain2file(merged_domain_path, new_node_dir / File.META_DOMAIN_PDDL)
        visualize_domain_graph(merged_domain_path, new_node_dir / File.META_DOMAIN_GRAPH)

    else:
        # Resume from interrupted domain fusion
        execution_logger.info(f"   - Node {new_node_id} already exists. Skipping.")

    # Create a new result node
    new_node = DomainFusionNode(
        node_id=new_node_id,
        domain_path=merged_domain_path,
        left=left_node,
        right=right_node
    )
    execution_logger.info(f"Merge node {left_node.node_id} and {right_node.node_id} to new node {new_node_id}")
    return new_node


def _merge_wrapper(merge_func: Callable[..., R], future_a: Future, future_b: Future) -> R:
    """Wrapper function to allows ThreadPoolExecutor to chain dependencies."""

    result_a = future_a.result()
    result_b = future_b.result()
    return merge_func(result_a, result_b)

def run_domain_fusion(
    domain_dir: Paths,
    leaf_node_counts: int,
    num_workers: int = 2,
) -> None:
    """Execute the domain fusion process using a multi-threaded binary tree strategy.

    This function initializes leaf nodes from existing atomic domains and iteratively
    fuses them until a single root domain remains.

    Args:
        domain_dir (Paths): The directory containing subdirectories of domains to be fused.
            Required structure:
                domain_dir/
                    0/
                        atomic_domain.json
                        ...
                    1/
                        atomic_domain.json
                        ...
                    ...
            Notes: (1) Sub-directory name MUST be named like 0, 1, 2, ...
                   (2) Other files in the sub-directory will be ignored.
                   (3) Atomic_domain.json in sub-directory is required.
                   (4) Number of leaf nodes (atomic domains) MUST be provided
            All original atomic domains and fused domains wiil be stored in domain_dir.
            The structure:
                output_dir/
                    0/  # Sub-directory provided from leaf nodes
                        atomic_domain.json
                        ...
                    1/
                        atomic_domain.json
                        ...
                    ...
                    n/  # New created directory for fused domain
                        meta_domain.json    # JSON format that can be loaded by domain fusion parser
                        meta_domain.pddl    # PDDL format that can be loaded by PDDL parser
                        meta_domain_graph.pdf   # Visualized domain graph
                        content.log # Recroding algorithm process of this node
                        metrics.log # Recroding LLM metrics of this node
                    n+1/
                        ...
                    binary_domain_fusion_tree.png   # Visualized domain fusion tree with each node's
                                                    # number of predicates and operators.
                    content.log # algorithm process out of each node.
        leaf_node_counts (int): The total number of initial atomic domains in the domain_dir.
                                This can be used for resuming the domain fusion process.
        num_workers (int): Number of parallel threads to use. Defaults to 2.

    Returns:
        None.
    """

    # Register signal handler for clean exit during multi-threading
    register_force_exit()

    domain_dir = setup_path(domain_dir, mkdir=False)
    execution_logger = get_task_logger(domain_dir)

    # Get domain paths of only leaf nodes
    sub_domain_dirs = [domain_dir / str(i) for i in range(leaf_node_counts)]

    # Initialize leaf nodes
    initial_nodes = [
        DomainFusionNode(node_id=i, domain_path=p / File.ATOMIC_DOMAIN_JSON)
        for i, p in enumerate(sub_domain_dirs)
    ]

    # Check for initial nodes
    if not initial_nodes:
        execution_logger.info("No initial nodes found for domain fusion.")
        return None
    if len(initial_nodes) == 1:
        execution_logger.info("Only one initial node found; no fusion needed.")
        return None
        
    # Initialize thread-safe ID counter starting after the last leaf node ID
    node_counter = _IDCounter(initial_value=len(initial_nodes))

    # Multi-threaded balanced binary tree merging
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Wrap initial node into pre-completed Futures
        current_layer_futures: List[Future] = [executor.submit(lambda n=node: n) for node in initial_nodes]
        
        # Prepare partial function with shared context
        merge_with_context = partial(
            _domain_fusion_node_wrapper,
            base_dir=domain_dir,
            counter=node_counter,
        )

        # Define the merging tasks for the next layer
        layer_num = 1
        while len(current_layer_futures) > 1:

            execution_logger.info(f"\n----- Constructing domain fusion task in {layer_num} layer -----")
            next_layer_futures = []

            # Separate the odd one out (if exists)
            odd_one_out = None
            if len(current_layer_futures) % 2 != 0:
                odd_one_out = current_layer_futures.pop()

            # Merge the remaining even pairs
            # Length of current_layer_futures must be even
            for i in range(0, len(current_layer_futures), 2):
                future1 = current_layer_futures[i]
                future2 = current_layer_futures[i+1]

                # Submit task depending on two previous futures
                new_future = executor.submit(_merge_wrapper, merge_with_context, future1, future2)
                next_layer_futures.append(new_future)

            # Re-insert the odd node at the beginning of the next layer
            # This ensures it gets merged as soon as possible to keep tree balanced
            if odd_one_out:
                current_layer_futures = [odd_one_out] + next_layer_futures
            else:
                current_layer_futures = next_layer_futures
            
            layer_num += 1
            execution_logger.info(f"--> Layer {layer_num} has {len(current_layer_futures)} nodes")

        execution_logger.info("\n----- All merge task dependencies have been submitted, waiting for final results... -----")
        # Wait for the final root node
        root_node = current_layer_futures[0].result()

    # Post-process tree visualization
    calculate_domain_size(root_node)
    visualize_domain_fusion_tree(root_node, domain_dir / File.DOMAIN_FUSION_TREE)
    return None
