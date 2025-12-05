"""
Baselines & Comparative Methods
===============================

This module implements a diverse set of baseline methods to evaluate UniDomain against
state-of-the-art approaches. These baselines span the spectrum from direct end-to-end
generation to iterative neuro-symbolic reasoning.

**Experimental Setting Note: Zero-Shot Adaptation**
To ensure a fair comparison with UniDomain's zero-shot capabilities, **we do not provide
task-specific in-context examples** (few-shot prompts) to any baseline (also for UniDomain).
Instead, we provide only a single, generic example to guide format compliance.

Baseline Overview
-----------------

**1. Direct / End-to-End Approaches**
*   **VLM-CoT**: *Natural Language (Naive)*.
    Directly prompts the VLM to generate a natural language plan using Chain-of-Thought
    reasoning based on visual observations. No formal domain definitions involved.

*   **Code as Policies (CaP)**: *Programmatic (Python)*.
    Treats planning as code generation. The model generates executable Python code
    calling primitive robot APIs to solve the task.

**2. Neuro-Symbolic / PDDL Approaches**
*   **VLM-CoT-PDDL**: *Symbolic (Naive)*.
    The "dual" of VLM-CoT. Prompts the VLM to generate full PDDL Domain/Problem files
    in one shot, then solves them using a classical planner.
    
*   **ISR-LLM**: *Iterative Self-Refinement*.
    Translates instructions to PDDL, generates a plan, and uses a self-verification
    loop to iteratively refine the PDDL definition based on planner feedback.
    
*   **IVML**: *Iterative Optimization*.
    Treats domain generation as a machine learning optimization problem. It uses a
    "Best-of-N" selection strategy followed by a Critic-Actor (f_opt/f_update) loop
    to optimize the domain.

**3. Interactive Approaches**
*   **ReAct**: *Interactive Embodied*.
    Interleaves reasoning and acting. Unlike the offline methods above, this agent
    executes actions step-by-step in the physical environment, receiving real-time
    success/failure feedback (Human-in-the-loop).
"""

from unidomain.baselines.code_as_policies import run_code_as_policies, run_code_as_policies_batch
from unidomain.baselines.isr_llm.isr_llm import run_isr_llm, run_isr_llm_batch
from unidomain.baselines.ivml import run_ivml, run_ivml_batch
from unidomain.baselines.vlm_cot import run_vlm_cot, run_vlm_cot_batch
from unidomain.baselines.vlm_cot_pddl import run_vlm_cot_pddl, run_vlm_cot_pddl_batch

try:
    from unidomain.baselines.reflexion.reflexion import run_react
except ImportError:
    def run_react(*args, **kwargs):
        raise ImportError("ReAct baseline requires 'opencv-python'. Please install it via 'pip install -e .[baselines]'.")

__all__ = [
    # Code as Policies
    "run_code_as_policies", "run_code_as_policies_batch",
    # ISR-LLM
    "run_isr_llm", "run_isr_llm_batch",
    # IVML
    "run_ivml", "run_ivml_batch",
    # ReAct
    "run_react",
    # VLM-CoT
    "run_vlm_cot", "run_vlm_cot_batch",
    # VLM-CoT-PDDL
    "run_vlm_cot_pddl", "run_vlm_cot_pddl_batch",
]
