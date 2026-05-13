import logging
import numpy as np
from typing import Tuple
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class ConfidenceEvaluator:
    def __init__(self):
        self.history = defaultdict(list)

    def extract_score(self, judge_response: str) -> float:
        judge_response = judge_response.strip()
        matches = re.findall(r'(\d+(?:\.\d+)?)', judge_response)
        if matches:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
        lower = judge_response.lower()
        if "perfect" in lower or "excellent" in lower:
            return 0.95
        if "good" in lower:
            return 0.75
        if "average" in lower:
            return 0.5
        if "bad" in lower:
            return 0.25
        logger.warning(f"Could not extract score from judge response: {judge_response[:200]}")
        return 0.5

    def record_score(self, key: str, score: float):
        if score < 0.0:
            return
        self.history[key].append(score)

    def get_trend(self, window: int = 5, key: str = "default") -> float:
        if key not in self.history or len(self.history[key]) < 2:
            return 0.0
        recent = self.history[key][-window:]
        n = len(recent)
        if n < 2:
            return 0.0
        x = np.arange(n)
        y = np.array(recent)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        return slope
