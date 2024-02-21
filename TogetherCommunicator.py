import together
class TogetherCommunicator:
    def __init__(self, API_KEY) -> None:
        together.api_key = API_KEY
        
    def get_messages(self, systemPrompt: str, userPrompt: str,):
            messages = [{"role": "system", "content": systemPrompt}, ]
            
            messages.append({"role": "user", "content": userPrompt})

            return messages  

    def generate_llama_v2_chat_prompt(self, messages):
            prompt = "[INST] <<SYS>> <</SYS>>"
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt = f"[INST]<<SYS>>\n{content}\n<</SYS>>"
                elif role == "user":
                    prompt += f"\n\n{content} [/INST]"
                elif role == "assistant":
                    prompt += f"\n\n{content} [INST]"
                else:
                    raise Exception('Messages format incorrect')

            return prompt

    def generate_mistral_chat_prompt(self, messages):
            prompt = '<s>'
            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == 'system':
                    prompt += f'[INST]System: {content} '
                elif message['role'] == 'user':
                    if prompt.endswith('[/INST]'):
                        prompt += f'[INST]{content}[/INST]'
                    else:
                        prompt += f'{content}[/INST]'
                elif message['role'] == 'assistant':
                    prompt += f'{content}</s><s>'
                else:
                    raise Exception('Messages format incorrect')
            return prompt


    def together_chat_api(self, prompt, model):
        try:
            response = together.Complete.create(
                prompt = prompt, 
                model = model, 
                max_tokens = 500,
                temperature = 0.5,
                stop = ['[INST]', '</s>', '\n\n'],
            )
            return response['output']['choices'][0]['text']
        except:
            return ''