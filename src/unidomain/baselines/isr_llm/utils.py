import json
import os
import re
import time
from pathlib import Path

from unidomain.configs.constants import JSONKey
from unidomain.services.llm_agent import setup_client

client = setup_client()

class LLMTimer:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.costs = 0
        self.thinking_time = 0
        self.nums_llm_call = 0

timer = LLMTimer()
def update_timer(
    input_tokens: int,
    output_tokens: int,
    thinking_time: float
) -> None:
    timer.input_tokens += input_tokens
    timer.output_tokens += output_tokens
    timer.thinking_time += thinking_time
    timer.nums_llm_call += 1
    timer.costs += (input_tokens * 2 + output_tokens * 8) / 1e6

def clear_timer() -> None:
    timer.input_tokens = 0
    timer.output_tokens = 0
    timer.costs = 0
    timer.thinking_time = 0
    timer.nums_llm_call = 0

def write_llm_metric(path) -> None:
    path = Path(path)
    metrics = {
        JSONKey.TOTAL_INPUT_TOKENS: timer.input_tokens,
        JSONKey.TOTAL_OUTPUT_TOKENS: timer.output_tokens,
        JSONKey.TOTAL_THINKING_TIME: timer.thinking_time,
        JSONKey.TOTAL_CALLS: timer.nums_llm_call,
        JSONKey.TOTAL_COSTS: timer.costs
    }
    with path.open("w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
        
    clear_timer()

# extract init and goal state from generated PDDL file
def extract_state_pddl(pddl_problem):

    def extract_from_pddl(pddl_problem, start_flag="(:init"):
        stack = ["("]
        pddl_string = "("
        start_index = pddl_problem.find(start_flag)

        for i in range(start_index + 1, len(pddl_problem)):
            pddl_string += pddl_problem[i]
            if pddl_problem[i] == "(":
                stack.append("(")
            elif pddl_problem[i] == ")":
                stack.pop()
                if not stack:
                    break

        return pddl_string

    pddl_init_state = extract_from_pddl(pddl_problem, start_flag="(:init")
    pddl_goal_state = extract_from_pddl(pddl_problem, start_flag="(:goal")

    return pddl_init_state, pddl_goal_state


# extract action description from response
def extract_action_description(action_sequence):

    actions = re.findall(r'\(.*?\)', action_sequence)
    num_actions = len(actions)
    action_description = ""
    for i in range(num_actions):
        action_description = action_description + actions[i] + '\n'

    return action_description

def call_openai(model, messages, temperature):
    repeat_counts = 10
    count = 0
    success = False
    while not success:
        try:
            start_time = time.perf_counter_ns()
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            llm_outputs = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            thinking_time = (time.perf_counter_ns() - start_time) / 1e9
            update_timer(input_tokens, output_tokens, thinking_time)

            success = True
        except Exception as e:
            print(e)
            count += 1
            time.sleep(2)

            # reach repeat_counts, raise exception
            if count > repeat_counts:
                raise e
            
            continue

    return llm_outputs