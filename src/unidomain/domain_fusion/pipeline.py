"""
Domain Fusion Pipeline Module
=============================

This is the main entry point (Orchestrator) for the domain fusion process.
It handles workspace preparation, file mapping, and invokes the execution engine
to perform the hierarchical fusion of atomic domains.

Architecture Dependency Graph:
    [This Module] pipeline.py
         |
         v
    [Runner] runner.py
         |
         v
    [Core & Utils] (operator_merging, predicate_merging, tree, etc.)
"""


import json
import shutil

from unidomain.configs.constants import File, JSONKey
from unidomain.domain_fusion.runner import run_domain_fusion
from unidomain.pddl.domain_graph import visualize_domain_graph
from unidomain.pddl.generator import domain2file
from unidomain.pddl.io import save_domain
from unidomain.pddl.parser import extract_domain_from_pddl
from unidomain.schemas.typings import MappingTable, Paths
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import get_sorted_paths, setup_path

__all__ = ["domain_fusion_pipeline"]


def _prepare_domain_fusion_workspace(
    domain_dir: Paths,
    output_dir: Paths,
) -> int:
    """Prepare the workspace for domain fusion by setting up the output directory.

    This function handles two scenarios:
    1. New Run: Creates a standardized directory structure (0, 1, 2...) from the
       source domains, converts formats if necessary, and generates a mapping table.
    2. Resume: Checks if an existing output directory is valid and consistent with
       the source directory to allow resuming interrupted jobs.

    Args:
        domain_dir (Paths): The source directory containing atomic domains.
        output_dir (Paths): The target directory for the fusion process.

    Returns:
        int: The number of leaf nodes (atomic domains) prepared for fusion.

    Raises:
        ValueError: If the existing output directory structure does not match the
            source directory during a resume attempt.
    """

    execution_logger = get_task_logger(domain_dir)

    # Check if domain fusion starts from scratch (empty or non-existent output dir)
    is_new_run = (not output_dir.exists()) or (len(list(output_dir.iterdir())) == 0)
    
    if is_new_run:
        execution_logger.info("Run domain fusion from scratch. Creating new output directory.")

        # Filter and sort source subdirectories, Ignores hidden files like .DS_Store
        # Sorting strategy: numeric if possible, otherwise string sort
        original_subdirs = get_sorted_paths(domain_dir, filter_func=lambda x: x.is_dir())
        
        # Copy and map original sub-directories to standardized indices (0, 1, 2...)
        mapping_table: MappingTable = {}
        for idx, original_subdir in enumerate(original_subdirs):

            # Create directory with the same name
            original_name = original_subdir.name
            new_name = str(idx)
            dest_dir = output_dir / new_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy only atomic_domain.json/pddl
            src_pddl_path = original_subdir / File.ATOMIC_DOMAIN_PDDL
            src_json_path = original_subdir / File.ATOMIC_DOMAIN_JSON
            dest_json_path = dest_dir / File.ATOMIC_DOMAIN_JSON
            dest_pddl_path = dest_dir / File.ATOMIC_DOMAIN_PDDL

            # Prefer JSON source; fallback to PDDL source and convert
            if src_json_path.exists():
                shutil.copy(src_json_path, dest_dir)
                domain2file(src_json_path, dest_pddl_path)

            elif src_pddl_path.exists():
                # convert JSON format from .pddl if JSON not exists
                shutil.copy(src_pddl_path, dest_dir)
                domain_str = src_pddl_path.read_text(encoding="utf-8")
                predicates, operators = extract_domain_from_pddl(domain_str)
                save_domain(predicates, operators, dest_json_path)

            else:
                execution_logger.warning(f"Skipping {original_name}: No atomic_domain.json or .pddl found.")
                continue

            # visualize atomic domain graph in the directory
            visualize_domain_graph(dest_json_path, dest_dir / File.ATOMIC_DOMAIN_GRAPH)

            mapping_table[original_name] = idx

        # Save mapping table with metadata
        final_mapping_data = {
            JSONKey.COMMENT: f"Source Directory: {domain_dir}",
            **mapping_table,
        }
        mapping_table_path = output_dir / File.MAPPING_TABLE
        with mapping_table_path.open("w", encoding="utf-8") as f:
            json.dump(final_mapping_data, f, ensure_ascii=False, indent=4)

        return len(mapping_table)
    
    # Output directory exists, try to resume
    else:
        # Get recorded mapping table
        mapping_table_path = output_dir / File.MAPPING_TABLE
        if not mapping_table_path.exists():
            raise FileNotFoundError(f"Output directory exists but missing '{File.MAPPING_TABLE}': {output_dir}")
        
        with mapping_table_path.open("r", encoding="utf-8") as f:
            mapping_table: MappingTable = json.load(f)

        # Check whether output directory matches the original domain directory
        # Here only check the consistency of the directory names and mapping table
        # Strict consistency check can be performed using hashing
        original_subdirs = set([p.name for p in domain_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
        output_subdirs = set([dir_name for dir_name in mapping_table.keys() if dir_name != JSONKey.COMMENT])

        if original_subdirs != output_subdirs:
            error_message = (
                "The output directory already exists, but does not match "
                "the structure of the input directory. Unable to resume. "
                "Please check the structure or specify an empty output directory."
            )
            execution_logger.info(error_message)
            raise ValueError(error_message)
        
        execution_logger.info("Resume domain fusion from the existing output directory.")

        return len(output_subdirs)


def domain_fusion_pipeline(
    domain_dir: Paths,
    output_dir: Paths,
    num_workers: int = 2,
) -> None:
    """Execute the complete domain fusion pipeline using a binary tree strategy.

    This function manages the lifecycle of the fusion process: preparing the workspace,
    handling file conversions, and dispatching the multi-threaded fusion runner.

    Args:
        domain_dir (Paths): The directory containing subdirectories of domains to be fused.
            Required structure:
                domain_dir/
                    name_1/
                        atomic_domain.json/pddl
                        ...
                    name_2/
                        atomic_domain.json/pddl
                        ...
                    ...
            Notes: (1) Sub-directory names can be arbitrary strings; they will be mapped internally.
                       But 0, 1, 2, ... are preferred.
                   (2) Other files in the sub-directory will be ignored.
                   (3) JSON files are prioritized. If missing, PDDL files are converted.

        output_dir (Paths): The directory to store the all original atomic domains and fused domains.
            - If empty/non-existent: A new domain fusion process will be started;
            - If exists: The domain fusion process will resume.

            The structure:
                output_dir/
                    0/  # Sub-directory copied from original domain_dir with mapped name
                        atomic_domain.json/pddl
                        ...
                    1/
                        atomic_domain.json/pddl
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
                    content.log # Algorithm process out of each node.
                    mapping_table.json  # Recording the mapping of original domain name to the node number.
                                        # e.g. {"_comment": "Source Directory: path/to/domain_dir",
                                        #       "name_1": 0, "name_2": 1, ...}
        num_workers (int): The number of worker threads for parallel fusion. Defaults to 2.

    Returns:
        None.
    """

    domain_dir = setup_path(domain_dir, mkdir=False)
    output_dir = setup_path(output_dir)

    # Prepare workspace and get the number of leaf nodes
    leaf_node_counts = _prepare_domain_fusion_workspace(domain_dir, output_dir)

    # Execute the runner
    run_domain_fusion(output_dir, leaf_node_counts, num_workers)

    return None
