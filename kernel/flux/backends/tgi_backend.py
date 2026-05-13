import logging
import requests
from typing import Dict, Any, Optional
from .base import AbstractBackend

logger = logging.getLogger(__name__)

class TGIBackend(AbstractBackend):
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "stop": stop_sequences or [],
                "details": True
            },
            **kwargs
        }
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                generated = data[0]["generated_text"]
                details = data[0].get("details", {})
            else:
                generated = data["generated_text"]
                details = data.get("details", {})

            prompt_tokens = 0
            completion_tokens = 0
            if details:
                tokens_list = details.get("best_of_sequences", [{}])[0].get("tokens", [])
                completion_tokens = len(tokens_list)
                prompt_tokens = max(1, len(prompt) // 4)
            else:
                prompt_tokens = max(1, len(prompt) // 4)
                completion_tokens = max(1, len(generated) // 4)

            return {
                "text": generated,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"TGI generation failed: {e}")
            raise
