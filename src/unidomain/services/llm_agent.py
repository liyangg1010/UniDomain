"""
LLM Agent Service
=================

This module encapsulates interactions with Large Language Models (LLMs) and
Vision-Language Models (VLMs). It provides a unified interface for model
invocation, cost tracking, input formatting (including images), and structured
output validation.

Architecture Position:
    [Services / Infrastructure] -> The foundational layer for all AI capabilities.
"""

import base64
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, OpenAI, RateLimitError
from pydantic import ValidationError

from unidomain.configs.constants import EnvVar, JSONKey
from unidomain.configs.llm import MODEL_COSTS, LLMAgentConfig
from unidomain.configs.settings import config
from unidomain.schemas.llm_task_outputs import LLMTaskOutputs
from unidomain.schemas.prompt_args import PromptArgs
from unidomain.schemas.typings import Paths
from unidomain.utils.logger import execution_logger, get_task_logger, llm_usage_logger
from unidomain.utils.runtime import get_mime_type, setup_path

load_dotenv()

__all__ = ["LLMAgent", "execute_llm_task", "setup_client"]


class _LLMRecorder:
    """Internal helper to track token usage, costs, and latency."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_time_seconds = 0
        self.total_costs = 0

        if model_name not in MODEL_COSTS:
            llm_usage_logger.warning(f"Cost of {model_name} is not set, failed to record costs.")

    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        # Prices are usually per 1M tokens
        prices = MODEL_COSTS.get(model_name, [0.0, 0.0])
        cost = (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
        return cost

    def record(
        self,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        thinking_time_seconds: float,
        log_dir: Paths
    ) -> None:
        """Record execution and usage for a single LLM call."""
        cost = self._calculate_cost(self.model_name, input_tokens, output_tokens)

        # Update accumulative usage
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_thinking_time_seconds += thinking_time_seconds
        self.total_costs += cost

        # Log structured usage (for analysis)
        call_id = str(uuid.uuid4())
        llm_usage_data = {
            "call_id": call_id,
            "model_name": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_time_seconds": thinking_time_seconds,
            "cost": cost,
            "total_costs_to_now": self.total_costs,
            "log_dir": log_dir
        }
        llm_usage_logger.info("", extra=llm_usage_data)

        # Log execution content (for debugging/audit)
        execution_data = {
            "log_type": "llm_agent",
            "call_id": call_id,
            "prompt": prompt,
            "response": str(response),
            "log_dir": log_dir
        }
        execution_logger.info("", extra=execution_data)
        
    def clear_record(self) -> None:
        """Reset accumulative usage."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_time_seconds = 0
        self.total_costs = 0

    def write_record(self, save_path: Paths) -> None:
        """Save accumulative usage to a JSON file and reset counters."""
        save_path = setup_path(save_path, is_file=True)

        # Load existing usage to append/accumulate if necessary
        usage = {}
        if save_path.exists():
            try:
                with save_path.open("r", encoding="utf-8") as f:
                    usage = json.load(f)
            except json.JSONDecodeError:
                pass

        # Update usage
        usage[JSONKey.TOTAL_CALLS] = self.total_calls
        usage[JSONKey.TOTAL_INPUT_TOKENS] = self.total_input_tokens
        usage[JSONKey.TOTAL_OUTPUT_TOKENS] = self.total_output_tokens
        usage[JSONKey.TOTAL_THINKING_TIME] = self.total_thinking_time_seconds
        usage[JSONKey.TOTAL_COSTS] = self.total_costs

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(usage, f, ensure_ascii=False, indent=4)

        self.clear_record()


def _image_to_base64(path: Paths) -> str:
    """Read an image file and encode it as a Base64 string."""
    path = setup_path(path, is_file=True, mkdir=False)
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _llm_post_process(llm_output: str) -> str:
    """Clean LLM output by stripping Markdown code blocks."""
    # Common pattern: ```json ... ```
    cleaned = llm_output.replace("```json", "").replace("```", "")
    return cleaned.strip()


def setup_client() -> Union[OpenAI, AzureOpenAI]:
    """Factory function to instantiate an OpenAI-compatible client.

    This function reads configuration from environment variables to set up
    either a standard OpenAI client or an Azure OpenAI client. It serves as
    the centralized authentication entry point for the project.

    Environment Variables:
        - API_KEY (required): The authentication key.
        - BASE_URL (optional): The API endpoint URL (required for Azure or custom proxies).
        - CLIENT_TYPE (optional): The provider type, either "OpenAI" (default) or "AzureOpenAI".
        - API_VERSION (required if CLIENT_TYPE is "AzureOpenAI"): The Azure API version string.

    Returns:
        Union[OpenAI, AzureOpenAI]: An initialized client instance ready for API calls.

    Raises:
        ValueError: If 'API_KEY' is missing, or if 'API_VERSION' is missing when
            using AzureOpenAI, or if an unsupported 'CLIENT_TYPE' is specified.
    """

    api_key = os.environ.get(EnvVar.API_KEY)
    base_url = os.environ.get(EnvVar.BASE_URL)
    client_type = os.environ.get(EnvVar.CLIENT_TYPE, "OpenAI")

    if not api_key:
        raise ValueError(f"Environment variable '{EnvVar.API_KEY}' is not set.")
    
    if client_type == "AzureOpenAI":
        api_version = os.environ.get(EnvVar.API_VERSION)
        if not api_version:
            raise ValueError(
                f"Environment variable '{EnvVar.API_VERSION}' is required when using AzureOpenAI."
            )
        return AzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=api_version)
    
    elif client_type == "OpenAI":
        return OpenAI(api_key=api_key, base_url=base_url)
    
    else:
        raise ValueError(f"Unsupported {EnvVar.CLIENT_TYPE}: {client_type}")


class LLMAgent:
    """Service wrapper for interacting with LLM providers (OpenAI / Azure)."""

    def __init__(self, llm_model_config: LLMAgentConfig, log_dir: Paths):
        self.model = llm_model_config.model
        self.log_dir = log_dir
        self._kwargs = llm_model_config.kwargs
        self._client = setup_client()
        self._recorder = _LLMRecorder(self.model)
        
    def _wrap_input(self, prompt: str, image_path_list: Optional[List[Paths]] = None) -> List[Dict[str, Any]]:
        """Construct the message payload for Chat Completion API."""

        # VLM Mode (Multimodal)
        if image_path_list:
            content_list: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            
            for image_path in image_path_list:
                mime_type = get_mime_type(image_path)
                if not mime_type.startswith("image/"):
                    raise ValueError(f"Invalid image file or unknown mime type: {image_path}")
                
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{_image_to_base64(image_path)}"}
                })
            messages = [{"role": "user", "content": content_list}]

        # Standard Text Mode
        else:
            messages = [{"role": "user", "content": prompt}]

        return messages

    def call(
        self,
        prompt: str,
        image_path_list: Optional[List[Paths]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """Execute a raw LLM call with internal retry logic for connection issues.

        Args:
            prompt (str): The text prompt.
            image_path_list (Optional[List[Paths]]): Images for VLM tasks.
            n (int): Number of completion choices to generate.

        Returns:
            Union[str, List[str]]: The generated text(s).
        """

        messages = self._wrap_input(prompt=prompt, image_path_list=image_path_list)
        max_retries = config.llm_tasks.call_max_retries
        task_execution_logger = get_task_logger(self.log_dir)

        # Retry loop for Network/API errors
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.perf_counter_ns()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=n,
                    **self._kwargs
                )
                thinking_time = (time.perf_counter_ns() - start_time) / 1e9

                # Extract outputs
                raw_outputs = [choice.message.content for choice in response.choices]

                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                break

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                task_execution_logger.error(f"Calling LLM failed: {e}. {attempt}/{max_retries}")
                
                if attempt < max_retries:
                    task_execution_logger.info("Retrying to call LLM...")
                    time.sleep(config.llm_tasks.call_retry_delay)
                else:
                    task_execution_logger.critical(f"Calling LLM failed after {max_retries} attempts, aborting!")
                    raise e

        
        # Post-process (clean Markdown)
        cleaned_outputs = [_llm_post_process(out) for out in raw_outputs]
        llm_outputs = cleaned_outputs[0] if n == 1 else cleaned_outputs

        self._recorder.record(
            prompt=prompt,
            response=llm_outputs,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_time_seconds=thinking_time,
            log_dir=self.log_dir
        )

        return llm_outputs

    def write_record(self, save_path: Paths) -> None:
        """Dump usage stats to file."""
        self._recorder.write_record(save_path)


def _get_prompt_from_template(path: Paths, **kwargs) -> str:
    """Load a text template and fill in variables."""
    path = setup_path(path, is_file=True, mkdir=False)
    with path.open("r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)
    
def execute_llm_task(
    llm_agent: LLMAgent,
    prompt_template_path: Paths,
    prompt_args: PromptArgs,
    validator: LLMTaskOutputs,
    image_path_list: Optional[List[Paths]] = None,
    n: int = 1,
) -> Union[LLMTaskOutputs, List[LLMTaskOutputs]]:
    """Execute a structured LLM task with validation and retry logic.

    This high-level function handles:
    1. Loading and formatting the prompt template.
    2. Invoking the LLM Agent.
    3. Parsing and Validating the output against a Pydantic schema.
    4. Retrying if validation fails (e.g., malformed JSON).

    Args:
        llm_agent (LLMAgent): The configured agent.
        prompt_template_path (Paths): Path to the text template.
        prompt_args (PromptArgs): Schema defining arguments for the template.
        validator (LLMTaskOutputs): Pydantic model class for validation.
        image_path_list (Optional[List[Paths]]): Images for VLM tasks.
        n (int): Number of outputs to generate.

    Returns:
        Union[LLMTaskOutputs, List[LLMTaskOutputs]]: The validated output object(s).
    """
    task_execution_logger = get_task_logger(llm_agent.log_dir)
    max_retries = config.llm_tasks.validator_max_retries

    prompt_args = prompt_args.model_dump()
    prompt = _get_prompt_from_template(prompt_template_path, **prompt_args)

    # Retry loop for Validation/Parsing errors
    for attempt in range(1, max_retries + 1):
        llm_outputs = None
        try:
            llm_outputs = llm_agent.call(prompt, image_path_list, n)

            # Validation
            if n == 1:
                # llm_outputs is str
                validated = validator.model_validate_json(llm_outputs)
                return validated
            else:
                # llm_outputs is List[str]
                validated_list = [validator.model_validate_json(llm_output) for llm_output in llm_outputs]
            return validated_list

        except (ValidationError, json.JSONDecodeError) as e:
            task_execution_logger.error(
                f"Parsing {validator} failed: {e}. {attempt}/{max_retries}\n "
                f"Output of LLMAgent:\n "
                f"{llm_outputs}"
            )

            if attempt < max_retries:
                task_execution_logger.info("Retrying LLM call due to validation error...")
                time.sleep(config.llm_tasks.validator_retry_delay)
            else:
                task_execution_logger.critical(f"Parsing {validator} failed after {max_retries} attempts, aborting!")
                raise e
