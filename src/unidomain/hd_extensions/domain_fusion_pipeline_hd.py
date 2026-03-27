"""Domain fusion pipeline for HD data with type removal and custom preparation.

Enhanced pipeline that:
1. Removes PDDL type annotations (converts "?p - person" to "?p")
2. Provides custom workspace preparation
3. Supports custom LLM configurations
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

from unidomain.configs.constants import File, JSONKey
from unidomain.pddl.domain_graph import visualize_domain_graph
from unidomain.pddl.generator import domain2file
from unidomain.pddl.io import load_domain, save_domain
from unidomain.pddl.parser import extract_domain_from_pddl
from unidomain.schemas.typings import MappingTable, Paths
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import get_sorted_paths, setup_path

from .runner_hd import run_domain_fusion_hd

__all__ = ["domain_fusion_pipeline_hd", "delete_type"]


def delete_type(
    predicates: Dict[str, str],
    operators: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Delete type annotations from predicates and operators.

    Traverses predicates and operators to find parameters with type annotations
    (e.g., "?p - person") and removes the type annotation, keeping only the
    variable name (e.g., "?p").

    Args:
        predicates: Dict mapping predicate definitions to descriptions
        operators: Dict mapping action names to action content

    Returns:
        Tuple of (updated_predicates, updated_operators)

    Examples:
        >>> preds = {"on(?x - block, ?y - block)": "x is on y"}
        >>> ops = {"move": "parameters: ?b - block\\n..."}
        >>> delete_type(preds, ops)
        ({"on(?x, ?y)": "x is on y"}, {"move": "parameters: ?b\\n..."})
    """
    # Regular expression to match type annotations: ?variable - type
    # Pattern: question mark + variable name + spaces + dash + spaces + type name
    type_pattern = re.compile(r'\?(\w+)\s+-\s+\w+')

    # Process predicates: delete type annotations in keys
    updated_predicates = {}
    for pred_key, pred_value in predicates.items():
        # Replace "?p - person" with "?p"
        updated_key = type_pattern.sub(r'?\1', pred_key)
        updated_predicates[updated_key] = pred_value

    # Process operators: delete type annotations in values
    updated_operators = {}
    for op_name, op_content in operators.items():
        # Replace "?p - person" with "?p"
        updated_content = type_pattern.sub(r'?\1', op_content)
        updated_operators[op_name] = updated_content

    return updated_predicates, updated_operators


def _prepare_domain_fusion_workspace_hd(
    domain_dir: Paths,
    output_dir: Paths,
) -> int:
    """Prepare the workspace for domain fusion with type removal.

    This function handles two scenarios:
    1. New Run: Creates a standardized directory structure (0, 1, 2...) from the
       source domains, converts formats if necessary, removes type annotations,
       and generates a mapping table.
    2. Resume: Checks if an existing output directory is valid and consistent with
       the source directory to allow resuming interrupted jobs.

    Args:
        domain_dir: The source directory containing atomic domains
        output_dir: The target directory for the fusion process

    Returns:
        int: The number of leaf nodes (atomic domains) prepared for fusion

    Raises:
        ValueError: If the existing output directory structure does not match the
            source directory during a resume attempt
        FileNotFoundError: If mapping table is missing during resume
    """
    execution_logger = get_task_logger(domain_dir)

    # Check if domain fusion starts from scratch (empty or non-existent output dir)
    is_new_run = (not output_dir.exists()) or (len(list(output_dir.iterdir())) == 0)

    if is_new_run:
        execution_logger.info("Run domain fusion from scratch. Creating new output directory.")

        # Filter and sort source subdirectories, ignoring hidden files like .DS_Store
        # Sorting strategy: numeric if possible, otherwise string sort
        original_subdirs = get_sorted_paths(
            domain_dir,
            filter_func=lambda x: x.is_dir()
        )

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
                # Convert JSON format from .pddl if JSON does not exist
                shutil.copy(src_pddl_path, dest_dir)
                domain_str = src_pddl_path.read_text(encoding="utf-8")
                execution_logger.info(f"Converting PDDL to JSON for {src_pddl_path}")
                predicates, operators = extract_domain_from_pddl(domain_str)
                save_domain(predicates, operators, dest_json_path)

            else:
                execution_logger.warning(
                    f"Skipping {original_name}: No atomic_domain.json or .pddl found."
                )
                continue

            # Remove type annotations from domain
            predicates, operators = load_domain(dest_json_path)
            predicates, operators = delete_type(predicates, operators)
            save_domain(predicates, operators, dest_json_path)
            domain2file(dest_json_path, dest_pddl_path)

            # Optional: Visualize atomic domain graph in the directory
            # visualize_domain_graph(dest_json_path, dest_dir / File.ATOMIC_DOMAIN_GRAPH)

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
            raise FileNotFoundError(
                f"Output directory exists but missing '{File.MAPPING_TABLE}': {output_dir}"
            )

        with mapping_table_path.open("r", encoding="utf-8") as f:
            mapping_table: MappingTable = json.load(f)

        # Check whether output directory matches the original domain directory
        # Here only check the consistency of the directory names and mapping table
        # Strict consistency check can be performed using hashing
        original_subdirs = set([
            p.name for p in domain_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ])
        output_subdirs = set([
            dir_name for dir_name in mapping_table.keys()
            if dir_name != JSONKey.COMMENT
        ])

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


def domain_fusion_pipeline_hd(
    domain_dir: Paths,
    output_dir: Paths,
    num_workers: int = 2,
    llm_model: str = "gpt-4o-mini",
    custom_client_factory=None,
) -> None:
    """Execute the complete domain fusion pipeline for HD data.

    This pipeline manages the lifecycle of the fusion process with enhancements for HD data:
    - Removes PDDL type annotations for compatibility
    - Handles file conversions
    - Dispatches multi-threaded fusion runner
    - Supports custom LLM configurations

    Args:
        domain_dir: The directory containing subdirectories of domains to be fused
            Required structure:
                domain_dir/
                    name_1/
                        atomic_domain.json/pddl
                        ...
                    name_2/
                        atomic_domain.json/pddl
                        ...
                    ...
            Notes:
                - Sub-directory names can be arbitrary strings; they will be mapped internally
                - 0, 1, 2, ... are preferred but not required
                - Other files in the sub-directory will be ignored
                - JSON files are prioritized. If missing, PDDL files are converted

        output_dir: The directory to store all original atomic domains and fused domains
            - If empty/non-existent: A new domain fusion process will be started
            - If exists: The domain fusion process will resume

            The structure:
                output_dir/
                    0/  # Sub-directory copied from original domain_dir with mapped name
                        atomic_domain.json/pddl (type annotations removed)
                        ...
                    1/
                        atomic_domain.json/pddl
                        ...
                    ...
                    n/  # New created directory for fused domain
                        meta_domain.json    # JSON format
                        meta_domain.pddl    # PDDL format
                        content.log         # Recording algorithm process
                        metrics.log         # Recording LLM metrics
                    n+1/
                        ...
                    mapping_table.json      # Original name -> node number mapping

        num_workers: The number of worker threads for parallel fusion. Defaults to 2

        llm_model: Model name to use for fusion. Defaults to "gpt-4o-mini"

        custom_client_factory: Optional callable(model_name) -> OpenAI client
            If provided, will be used to create custom client for each fusion node

    Returns:
        None

    Example with Custom Client:
        >>> from openai import OpenAI
        >>> def my_client_factory(model):
        ...     return OpenAI(api_key="...", base_url="...")
        >>> domain_fusion_pipeline_hd(
        ...     "data/atomic_domains",
        ...     "output/fused",
        ...     llm_model="GLM-4.7",
        ...     custom_client_factory=my_client_factory
        ... )
    """
    domain_dir = setup_path(domain_dir, mkdir=False)
    output_dir = setup_path(output_dir)

    # Prepare workspace and get the number of leaf nodes
    leaf_node_counts = _prepare_domain_fusion_workspace_hd(domain_dir, output_dir)

    # Execute the runner
    run_domain_fusion_hd(
        output_dir,
        leaf_node_counts,
        num_workers,
        llm_model,
        custom_client_factory
    )

    return None
