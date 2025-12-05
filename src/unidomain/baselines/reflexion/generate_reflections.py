from unidomain.baselines.reflexion.utils import get_completion, get_chat
from pathlib import Path
from typing import List, Dict, Any

root_dir = Path(__file__).parent.parent
few_shot_examples_dir = root_dir / "reflexion" /"reflexion_few_shot_examples.txt"
with open(few_shot_examples_dir, 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    if "Here is the task:" in s:
        return s.split("Here is the task:")[-1].strip()
    elif "Instruction:" in s:
        return s.split("Instruction:")[-1].strip()
    else:
        return s.split("#####\n\ntask_name")[-1].strip()

def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query

def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
        
    env = env_configs[0]
    
    # If unsolved, get reflection and update env config
    if not env['is_success'] and not env['skip']:
        if len(env['memory']) > 3:
            memory: List[str] = env['memory'][-3:]
        else:
            memory: List[str] = env['memory']
        
        reflection_query: str = _generate_reflection_query(full_log, memory)
        reflection: str = get_chat(reflection_query) 
        env_configs[0]['memory'] += [reflection]
                
    return env_configs