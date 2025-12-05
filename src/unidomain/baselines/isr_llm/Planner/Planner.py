import os

from unidomain.baselines.isr_llm.utils import call_openai

class Planner(object):
    """
    LLM Planner: generate plans
    is_log_example: if the few-shot examples are recorded in the log file
    temperature: default temperature value for LLM
    """
    def __init__(self, model, logdir, is_log_example = False, temperature = 0):
        self.model = model
        self.temperature = temperature
        self.messages = None
        self.log_dir = logdir
        self.log_file_path = self.log_dir + "/planner_log.txt"
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
                contents = f.read().split('Action Sequence', 1)
                question = contents[0]
                answer = 'Action Sequence' + contents[1]
                
                question_message = {"role": "system", "name": "example_user", "content": question}
                self.messages.append(question_message)
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= question, is_append=True)

                answer_message = {"role": "system", "name": "example_assistant", "content": answer}
                self.messages.append(answer_message)  
                if self.is_log_example == True and is_reinitialize == False:
                    self.write_content(content= answer, is_append=True)

    # Query question message
    def query(self, content, is_append = False, temperature = None):

        # add new question to message list
        question_message = {"role": "user", "content": content}
        if is_append == False:
            question = self.messages.copy()
        else:
            question = self.messages
        question.append(question_message)
        self.write_content(content= content, is_append=True)

        if temperature == None:
            response = call_openai(model=self.model, messages=question, temperature=self.temperature)
        else:
            response = call_openai(model=self.model, messages=question, temperature=temperature)

        self.write_content(content=response, is_append=True)

        return response
