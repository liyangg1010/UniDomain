import os
import sys
import json
import cv2
import time
import numpy as np
from unidomain.baselines.reflexion.utils import get_chat, get_completion, save_image_with_timestamp
from unidomain.baselines.reflexion.env_history import EnvironmentHistory
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

root_dir = Path(__file__).parent.parent
FOLDER = root_dir / "reflexion" / "prompts"
PROMPT_FILE = 'camera_prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    prompts = json.load(f)

def llm(prompt: str, model, stop: List[str] = ["\n"], current_image_path: Optional[str] = None, history_images: Optional[List[str]] = None):
    try:
        cur_try = 0
        while cur_try < 6:
            if model == "text-davinci-003":
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            else:
                text = get_chat(
                    prompt=prompt, 
                    model=model, 
                    temperature=cur_try * 0.2, 
                    stop_strs=stop, 
                    current_image_path=current_image_path,
                    history_images=history_images
                )
            # Check if we got a reasonable response
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def capture_image(output_dir: str = "./temp_captures", video_index: int = 0) -> Optional[str]:
    """Capture an image from the camera when space is pressed"""

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    print("Press SPACE to capture an image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break
            
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Wait for key
        key = cv2.waitKey(1)
        if key == 32:  # Space key
            timestamp = int(time.time())
            temp_img_path = os.path.join(output_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(temp_img_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"Image captured and saved to {temp_img_path}")
            return temp_img_path
        
def get_human_feedback():
    """Get feedback from human: SPACE for success, ESC for invalid action, ENTER for success+completion"""

    print("Was the action executed successfully?")
    print("Press SPACE for YES ; ESC for NO ; ENTER for task completed")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
    
    # Keep showing camera feed until a key is pressed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display a simple feedback window
        feedback_frame = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(feedback_frame, "Was the action successful?", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(feedback_frame, "SPACE = YES | ESC = NO | ENTER = Task complete", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Feedback', feedback_frame)
        
        key = cv2.waitKey(1)
        if key == 32:  # Space key
            cap.release()
            cv2.destroyAllWindows()
            return "Action successfully executed."
        elif key == 13:  # Enter key
            cap.release()
            cv2.destroyAllWindows()
            return "Task completed!"
        elif key == 27:  # Escape key
            cap.release()
            cv2.destroyAllWindows()
            return "Action is invalid or could not be executed."
    
def camera_run(
    base_prompt, 
    memory: List[str], 
    first_frame_path: str,
    instruction: str,
    to_print=True, 
    model="text-davinci-003",
    max_steps=30,
    save_dir: str = "./temp_captures",
    video_index: int = 0,
    trial_idx: int = 1
) -> Tuple[EnvironmentHistory, bool]:
    """Run the interaction loop with camera and human feedback"""
    
    first_frame_description = f"Instruction: {instruction}"
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, first_frame_description, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, first_frame_description, memory, [])
    
    env_history.reset()

    # Add the first image to the environment history
    env_history.add("observation", f"Initial scene: {first_frame_path}")
    
    if to_print:
        sys.stdout.flush()
    
    cur_step = 0
    current_image_path = first_frame_path
    is_success = False
    
    image_history = []
    
    while cur_step < max_steps:  
        print(f"Step: {cur_step}/{max_steps}")
        history_images = image_history.copy()
        
        action = llm(
            str(env_history) + ">", 
            stop=['\n'], 
            model=model, 
            current_image_path=current_image_path,
            history_images=history_images
        ).strip()
        
        env_history.add("action", action)
        
        # Check if the action is a thinking step
        if action.startswith('think:'):
            observation = 'OK.'
            env_history.add("observation", observation)
            if to_print:
                print(f'> {action}\n{observation}')
            cur_step += 1
            continue
            
        # # Wait for human to execute and capture the result
        print(f"LLM outputs the action: {action}")
        print("Press SPACE when done to capture the result...")
        results_path = os.path.join(os.path.dirname(save_dir), f"results_{trial_idx}.txt")
        with open(results_path, "a") as f:
            f.write(f"{action}\n")
        captured_image_path = capture_image(save_dir, video_index)

        human_feedback = get_human_feedback()

        # Add the observation to the environment history
        current_image_path = captured_image_path
        image_history.append(captured_image_path)
        observation = f"Image: {current_image_path}\nFeedback: {human_feedback}"
        env_history.add("observation", observation)
        
        if to_print:
            print(f'> {action}\n{observation}')
            
        # Check if the task is completed
        if human_feedback == "Task completed!":
            return env_history, True
        
        elif env_history.check_is_exhausted():
            return env_history, False  
        cur_step += 1
    

    return env_history, is_success

def run_trial(
    trial_log_path: str,
    world_log_path: str,
    trial_idx: int,
    env_configs: List[Dict[str, Any]],
    use_memory: bool,
    model: str,
    task_name: str,
    instruction: str,
    max_steps: int,
    image_dir: str,
    video_index: int
) -> List[Dict[str, Any]]:

    env_config = env_configs[0]
    if env_config["is_success"]:

        # Log to world log
        with open(world_log_path, 'a') as wf:
            wf.write(f'task_name{task_name} Trial #{trial_idx}: SUCCESS\n')
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\ntask_name{task_name}: Success\n\n#####\n')
        return env_configs
    
    # Capture the first frame 
    print(f"Now trial {trial_idx}, Please set up the initial scene and press SPACE when ready to capture the first frame.")
    first_frame_path = capture_image(image_dir, video_index)

    base_prompt = prompts["camera_base_prompt"]
    base_prompt += " If you need to think for the answer first, please give your thinking process after 'think:'. If you received 'OK.' as the last of the interaction history, please answer with one action directly."
    base_prompt += " Please don't generate '>' or anything else in front of your answer."
    
    final_env_history, is_success = camera_run(
        base_prompt, 
        env_config["memory"] if use_memory else [], 
        to_print=True, 
        first_frame_path=first_frame_path,
        instruction=instruction,
        model=model,
        max_steps=max_steps,
        save_dir=image_dir,
        video_index=video_index,
        trial_idx=trial_idx
    )
    
    # Update env config
    if is_success:
        status_str: str = f'task_name{task_name} Trial #{trial_idx}: SUCCESS'
        env_configs[0]['is_success'] = True
    else:
        status_str: str = f'task_name{task_name} Trial #{trial_idx}: FAIL'
    
    # Log to world log
    with open(world_log_path, 'a') as f:
        f.write(status_str + '\n')
    
    # Log env results to trial log
    with open(trial_log_path, 'a') as wf:
        wf.write(f'\n#####\n\ntask_name{task_name}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
    
    # Log trial results to trial and world logs
    log_str: str = f"""
                    -----
                    SUCCESS: {1 if is_success else 0}
                    ADDITIONAL SUCCESS: {1 if is_success and not env_config["is_success"] else 0}
                    FAIL: {0 if is_success else 1}
                    TOTAL: 1
                    ACCURACY: {1 if is_success else 0}
                    -----"""
    
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')
    
    return env_configs