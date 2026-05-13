from openai import OpenAI
from flux.backends.base import AbstractBackend

class OllamaBackend(AbstractBackend):
    """
    Backend for Ollama (OpenAI-compatible API).
    Ollama must be running at http://localhost:11434.
    """
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, model_name, prompt, max_new_tokens=256, temperature=0.7,
                 stop_sequences=None, **kwargs):
        completion = self.client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_sequences,
            **kwargs
        )
        return {
            "text": completion.choices[0].text,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens
            }
        }
