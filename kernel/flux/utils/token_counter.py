import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed, falling back to heuristic.")

    def count(self, text: str) -> Tuple[int, Optional[List[int]]]:
        if self.enc:
            tokens = self.enc.encode(text)
            return len(tokens), tokens
        else:
            return int(len(text) / 4) + 1, None

    def decode(self, tokens: List[int]) -> str:
        if self.enc:
            return self.enc.decode(tokens)
        else:
            return ""
