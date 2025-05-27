import time
import json
import os

class Player:
    def __init__(self, model, id, prompt, records=None, utility=None, tokens=None, valuation=None):
        self.model = model
        self.id = id
        self.prompt = prompt
        self.records = records if records else []
        self.utility = utility if utility else []
        self.tokens = tokens if tokens else []
        self.valuation = valuation if valuation else []
        
    def request(self, round_id, inputs, request_key="option"):
        if self.model == "user":
            return self.user_request(inputs, request_key)
        elif self.model.startswith("specified"):
            return self.specified_request(round_id, request_key)
        else:
            return self.gpt_request(inputs)
        
    def user_request(self, outputs, request_key):
        output_str = '\n'.join([prompt["content"] for prompt in outputs])
        response = input(f"{output_str}\nPlease input the answer for {request_key}:")
        response = f'{{"{request_key}": "{response}"}}'
        return response
    
    def specified_request(self, round_id, request_key):
        options = self.model.split("=")[1].split('/')
        option_num = len(options)
        response = options[(round_id - 1) % option_num]
        response = f'{{"{request_key}": "{response}"}}'
        return response

    def gpt_request(self, inputs):
        # Assuming chat function that interacts with models
        response = chat(self.model, inputs).strip()
        return response
