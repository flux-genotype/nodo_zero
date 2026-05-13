import logging
from typing import Dict, Any, Optional
from .base import AbstractBackend

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class VLLMBackend(AbstractBackend):
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "not-needed"):
        if OpenAI is None:
            raise ImportError("openai package required. Install with: pip install openai")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        try:
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
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
