import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from openai import OpenAI
from tenacity import stop_after_attempt  # type: ignore
from tenacity import wait_random_exponential  # type: ignore
from tenacity import retry
from unidomain.configs.settings import config
from unidomain.services.llm_agent import setup_client
from unidomain.configs.constants import JSONKey

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]

client = setup_client()
model = config.baselines.llm_model_config.model

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].message.content

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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(
    prompt: str, 
    model = model, 
    temperature: float = 0.0, 
    stop_strs: Optional[List[str]] = None, 
    current_image_path: Optional[str] = None,
    history_images: Optional[List[str]] = None
) -> str:
    """
    Get a completion from a multimodal model that accepts both text and images.
    
    Args:
        prompt: Text prompt to send to the model
        model: Model to use (should be a multimodal model like gpt-4.1)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        stop_strs: Stop sequences for generation
        is_batched: Whether this is a batched request
        current_image_path: Path to the current image to analyze
        history_images: List of paths to previous images for context
        
    Returns:
        Generated text response
    """
    assert model != "text-davinci-003"
    
    # Initialize the content list for the message
    content = []
    
    # Add the text prompt
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # Add the current image 
    if current_image_path and os.path.isfile(current_image_path):
        base64_image = encode_image_to_base64(current_image_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"  # High detail for the current image
                }
            })
        else:
            print(f"Warning: Failed to encode current image {current_image_path}")
    
    # Add historical images 
    if history_images:
        for img_path in history_images:
            if os.path.isfile(img_path):
                base64_image = encode_image_to_base64(img_path)
                if base64_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"  # Low detail for historical images to save tokens
                        }
                    })
                else:
                    print(f"Warning: Failed to encode history image {img_path}")
    
    # Create and send the message
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    success = False
    while not success:
        try:
            start_time = time.perf_counter_ns()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=stop_strs,
                temperature=temperature,
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            thinking_time = (time.perf_counter_ns() - start_time) / 1e9
            update_timer(input_tokens, output_tokens, thinking_time)

            success = True
        except Exception as e:
            print(e)
            time.sleep(2)
            continue
    
    return response.choices[0].message.content

def save_image_with_timestamp(image: np.ndarray, base_dir: str = "./images") -> str:
    """Save an image with a timestamp filename and return the path."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"image_{timestamp}.jpg"
    filepath = os.path.join(base_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from the specified path."""
    if not os.path.exists(image_path):
        print(f"Warning: Image path {image_path} does not exist.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image from {image_path}.")
    
    return image

def get_image_description(image_path: str) -> str:
    """Get a description of the image using the multimodal model."""
    prompt = "Please describe what you see in this image in detail."
    description = get_chat(
        prompt=prompt,
        model=model,
        current_image_path=image_path
    )
    return description

def log_interaction(
    action: str, 
    image_path: str, 
    feedback: str, 
    log_file: str = "./interaction_log.json"
) -> None:
    """Log each interaction to a JSON file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "image_path": image_path,
        "feedback": feedback
    }
    
    # Load existing log if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
    else:
        log_data = []
    
    # Append new entry
    log_data.append(log_entry)
    
    # Save updated log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)