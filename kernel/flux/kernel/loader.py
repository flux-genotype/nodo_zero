from typing import Dict, Any, Optional
from flux.backends.base import AbstractBackend

class FluxModelLoader:
    def __init__(self, backend: AbstractBackend, model_mapping: Optional[Dict[str, str]] = None,
                 model_contexts: Optional[Dict[str, int]] = None):
        self.backend = backend
        self.model_mapping = model_mapping or {}
        self.model_contexts = model_contexts or {}

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        max_context = self.model_contexts.get(model_key, 32768)
        return {"max_context": max_context}

    def generate(self, model_key: str, prompt: str,
                 max_new_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        real_model = self.model_mapping.get(model_key, model_key)
        return self.backend.generate(
            model_name=real_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
