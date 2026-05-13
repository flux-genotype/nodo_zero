from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AbstractBackend(ABC):
    @abstractmethod
    def generate(
        self,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_sequences: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Returns {'text': str, 'usage': {'prompt_tokens': int, 'completion_tokens': int}}"""
        ...
