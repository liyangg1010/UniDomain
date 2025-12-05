"""
Holistic Domain Revision Step
=============================

This module implements the second stage of the atomic domain generation pipeline.
It performs a holistic revision of the initially learned domain, using the LLM's
internal knowledge to correct inconsistencies, syntax errors, or logical gaps
before moving to strict verification steps.

Architecture Position:
    [Functional Steps / Middle Level] -> Called by atomic_domain.pipeline
    Uses post_process.py for cleanup.
"""

from unidomain.atomic_domain.post_process import save_and_post_process
from unidomain.configs.constants import File
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import HolisticReviseDomainOutputs
from unidomain.schemas.prompt_args import HolisticReviseDomainPromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.runtime import setup_path

__all__ = ["run_holistic_revision_step"]


def _holistic_revise_domain(
    llm_agent: LLMAgent,
    prompt_args: HolisticReviseDomainPromptArgs,
) -> str:
    """Call LLM to revise the initial domain holistically.
    
    This function leverages the LLM to refine the PDDL domain generated in the
    previous step, aiming to fix obvious syntax errors or logical inconsistencies.

    Args:
        llm_agent (LLMAgent): The LLM agent service.
        prompt_args (HolisticReviseDomainPromptArgs): Arguments for LLM prompt.

    Returns:
        str: The revised domain in PDDL format.
    """
    # Call LLM and parse JSON into specified format
    outputs: HolisticReviseDomainOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.holistic_revise_domain,
        prompt_args=prompt_args,
        validator=HolisticReviseDomainOutputs,
    )

    # Return necessary information
    revised_domain = outputs.revised_domain.strip()
    return revised_domain


def run_holistic_revision_step(
    llm_agent: LLMAgent,
    domain_content: str,
    instructions: str,
    output_dir: Paths
) -> None:
    """Execute the holistic revision step and save the results.

    Args:
        llm_agent (LLMAgent): The LLM agent service.
        domain_content (str): The raw string content of the domain to be revised.
        instructions (str): The task instructions to guide the revision.
        output_dir (Paths): The directory to save the revised domain file.
    """
    output_dir = setup_path(output_dir)
    
    # Call LLM
    prompt_args = HolisticReviseDomainPromptArgs(domain=domain_content, instructions=instructions)
    revised_domain_str = _holistic_revise_domain(llm_agent, prompt_args)

    # Save & Post-process (Generate JSON/PDDL and Clean)
    save_and_post_process(revised_domain_str, output_dir, stem=File.REVISED_DOMAIN_STEM)
