import os
import base64

from unidomain.baselines.isr_llm.utils import call_openai

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class Translator(object):
    """
    LLM Translator: it translates description to PDDL domain file and problem file
    arg: argument parameters
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, model, logdir, is_log_example = False, temperature = 0):
        self.model = model
        self.temperature = temperature
        self.messages = None
        self.log_dir = logdir
        self.log_file_path = self.log_dir + "/translator_log.txt"
        self.is_log_example = is_log_example
        
        # root for prompt examples
        self.num_examples = 2
        self.prompt_example_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocksworld_examples")
        
        # initialize messages
        self.init_messages()

    # Write content to file 
    def write_content(self, content, is_append):

        if is_append == False:

            with open(self.log_file_path, "w") as f:
                f.write(content+"\n")

        else:

            with open(self.log_file_path, "a") as f:
                f.write(content+"\n")

    # Initialize messages include opening and few-shot examples
    def init_messages(self, is_reinitialize = False):

        # opening setup
        file_path = self.prompt_example_root + "/opening.txt"
        with open(file_path, 'r') as f:
            contents = f.read()
            opening_message =  {"role": "system", "content": contents}
            self.messages = [opening_message]
            # record content
            if self.is_log_example == True and is_reinitialize == False:
                self.write_content(content= contents, is_append=False)

        # load few-shot examples
        for i in range(self.num_examples):

            file_path = self.prompt_example_root + "/example"+str(i)+".txt"
            with open(file_path, 'r') as f:
                contents = f.read().split('\n', 1)
                question = contents[0]
                answer = contents[1]
                
                question_message = {"role": "system", "name": "example_user", "content": question}
                self.messages.append(question_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= question, is_append=True)

                answer_message = {"role": "system", "name": "example_assistant", "content": answer}
                self.messages.append(answer_message)   
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= answer, is_append=True)

    # Query question message
    def query(self, content, image_path, is_append = False):
        base64_image = image_to_base64(image_path)
        question_message = {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": content,
                },
            ]
        }
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
        question_message["content"].append(image_message)

        if is_append == False:
            question = self.messages.copy()
        else:
            question = self.messages
        question.append(question_message)
        self.write_content(content=content, is_append=True)

        response = call_openai(model=self.model, messages=question, temperature=self.temperature)

        self.write_content(content=response, is_append=True)

        return response