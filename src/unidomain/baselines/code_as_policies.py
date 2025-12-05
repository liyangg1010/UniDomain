"""
Baseline: Modified Code as Policies (CaP)
=========================================

This module implements an **adapted version** of the "Code as Policies" baseline.
**Reference:** "Code as Policies: Language Model Programs for Embodied Control".

**Modifications & Adaptations:**
1.  **VLM Integration**: Unlike the original implementation which often relies on
    textual state descriptions or specific perception APIs, this version directly
    utilizes **Vision-Language Models (VLMs)** to perceive environmental dynamics
    from input keyframes.
2.  **Simplified Execution**: The implementation preserves the core mechanism
    (generating executable Python code for control) but simplifies auxiliary
    structures to align with the UniDomain experimental setting.

Architecture Position:
    [Baselines] -> A comparative method adapted to evaluate against UniDomain.
"""

import contextlib
import io
from pathlib import Path
from typing import Dict, List

import numpy as np

from unidomain.configs.constants import File, JSONKey
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import CaPFixBugsOutputs, CaPGenerateCodeOutputs
from unidomain.schemas.prompt_args import CaPFixBugsPromptArgs, CaPGenerateCodePromptArgs
from unidomain.schemas.typings import Paths
from unidomain.services.llm_agent import LLMAgent, execute_llm_task
from unidomain.utils.batch_runner import execute_batch_tasks, load_batch_inputs
from unidomain.utils.logger import get_task_logger
from unidomain.utils.runtime import setup_path

__all__ = ["run_code_as_policies", "run_code_as_policies_batch"]


# Mock Primitives for Execution
def pick_up(obj): print(f"pick_up({obj})")
def put_down(obj): print(f"put_down({obj})")
def open(obj):    print(f"open({obj})")
def close(obj):   print(f"close({obj})")
def fold(obj):    print(f"fold({obj})")
def wipe(obj):    print(f"wipe({obj})")
def scrunch(obj): print(f"scrunch({obj})")
def stir(obj):    print(f"stir({obj})")
def push(obj):    print(f"push({obj})")
def slide(obj):   print(f"slide({obj})")
def press(obj):   print(f"press({obj})")
def turn_on(obj): print(f"turn_on({obj})")
def turn_off(obj):print(f"turn_off({obj})")
def pull(obj):    print(f"pull({obj})")
def pour(obj):    print(f"pour({obj})")
def lean(obj1, obj2):    print(f"lean({obj1, obj2})")


def _generate_code(
    image_path_list: List[Paths],
    llm_agent: LLMAgent,
    prompt_args: CaPGenerateCodePromptArgs,
) -> str:
    """Prompt the VLM to generate Python code based on the visual task."""

    # Call VLM and parse the output into JSON format
    outputs: CaPGenerateCodeOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.cap_generate_code,
        prompt_args=prompt_args,
        validator=CaPGenerateCodeOutputs,
        image_path_list=image_path_list,
    )

    code = outputs.code
    return code


def fix_bugs(
    llm_agent: LLMAgent,
    prompt_args: CaPFixBugsPromptArgs,
) -> str:
    """Prompt the LLM to fix the code based on the execution error message."""
    # Call LLM and parse the output into JSON format
    outputs: CaPFixBugsOutputs = execute_llm_task(
        llm_agent=llm_agent,
        prompt_template_path=config.prompt_path.cap_fix_bugs,
        prompt_args=prompt_args,
        validator=CaPFixBugsOutputs,
    )

    fixed_code = outputs.fixed_code
    return fixed_code


def _exec_safe(code_str: str, exec_globals: dict) -> None:
    """Execute the generated code within a restricted scope.

    Security Note:
        This function strips out "import" statements to prevent the LLM from
        accessing system resources or arbitrary libraries. It relies on the
        provided `exec_globals` for all necessary functionality.
    
    Args:
        code_str (str): The Python code to execute.
        exec_globals (dict): The global namespace containing mock primitives.
    """
    lines = code_str.splitlines()

    # Basic sandbox: disable imports
    filtered = [line for line in lines if not line.strip().startswith(("import", "from"))]
    
    # Execute the cleaned code
    # Any exception here will be caught by the caller for retry logic
    exec("\n".join(filtered), exec_globals)
    

def run_code_as_policies(
    image_path: Paths,
    instruction: str,
    save_dir: Paths
) -> bool:
    """Run the CaP pipeline for a single task.

    Pipeline Steps:
    1. VLM generates code based on the image and instruction.
    2. Code is executed in a sandbox with `contextlib.redirect_stdout`.
    3. If execution succeeds, the printed actions are saved as the plan.
    4. If execution fails, the error is fed back to the LLM for one round of repair.

    Args:
        image_path (Paths): Path to the initial observation image.
        instruction (str): Natural language task instruction.
        save_dir (Paths): Directory to save the execution logs and results.

    Returns:
        bool: True (always returns True as it attempts to run regardless of success).
    """

    save_dir = setup_path(save_dir)
    image_path = setup_path(image_path, is_file=True, mkdir=False)
    execution_logger = get_task_logger(save_dir)
    llm_agent = LLMAgent(config.baselines.llm_model_config, save_dir)

    execution_logger.info(
        "Runing Baseline: Code as Policies "
        f"Task Path: {image_path} "
        f"Instruction: {instruction}"
    )

    # Generate Initial Code
    execution_logger.info("Generating Code ...")
    prompt_args = CaPGenerateCodePromptArgs(instructions=instruction)
    code = _generate_code([image_path], llm_agent, prompt_args)

    # Prepare execution environment
    buffer = io.StringIO()
    exec_globals = {
        'np': np,
        'pick_up': pick_up,
        'put_down': put_down,
        'open': open,
        'close': close,
        'fold': fold,
        'wipe': wipe,
        'scrunch': scrunch,
        'stir': stir,
        'push': push,
        'slide': slide,
        'press': press,
        'turn_on': turn_on,
        'turn_off': turn_off,
        'pull': pull,
        'pour': pour,
        'lean': lean
    }

    # Execute with Capture and Retry
    plans = []
    with contextlib.redirect_stdout(buffer):
        try:
            _exec_safe(code, exec_globals)  # Try to execute the generated code
            plans.append(buffer.getvalue())  # If successful, append the output

        except Exception as e:
            execution_logger.warning(f"Execution Error: {e}. Attempting to fix bugs...")

            try:
                # If execution fails, try to fix bugs and re-execute
                prompt_args = CaPFixBugsPromptArgs(code=code, error=str(e))
                fixed_code = fix_bugs(llm_agent, prompt_args)

                # Reset buffer for the second run
                buffer.truncate(0)
                buffer.seek(0)

                # Execute fixed code
                _exec_safe(fixed_code, exec_globals)
                plans.append(buffer.getvalue())
                
            except Exception as retry_e:
                execution_logger.info(f"Retry failed: {retry_e}")
                execution_logger.info("Both attempts failed. Skipping this task.")

    # Save Results
    save_path = save_dir / File.FINAL_RESULTS
    save_path.write_text("\n".join(plans), encoding="utf-8")

    llm_agent.write_record(save_dir / File.FINAL_METRICS)
    return True


def run_code_as_policies_batch(
    task_data_path: Paths,
    save_dir: Paths,
    num_workers: int = 1,
) -> None:
    """Execute Code as Policies in batch mode."""
    tasks = load_batch_inputs(task_data_path)

    # Closure adapter to capture configuration arguments
    def _worker(name: str, info: Dict, sub_dir: Path) -> bool:
        return run_code_as_policies(
            image_path=info[JSONKey.PATH],
            instruction=info[JSONKey.INSTRUCTION],
            save_dir=sub_dir,
        )

    execute_batch_tasks(
        tasks=tasks,
        save_dir=save_dir,
        worker_func=_worker,
        num_workers=num_workers
    )
