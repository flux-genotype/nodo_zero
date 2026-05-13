import threading
from collections import deque
from enum import Enum
from flux.kernel.logger import FluxLogger

class KernelMode(Enum):
    GROWTH = "growth"
    PRODUCTION = "production"
    SIMULATION = "simulation"

class GrowthSupervisor:
    def __init__(self, logger: FluxLogger, stability_runs: int = 10, min_confidence: float = 0.85, max_cost_per_request: float = 0.1):
        self.logger = logger
        self.mode = KernelMode.GROWTH
        self.stability_runs = stability_runs
        self.min_confidence = min_confidence
        self.max_cost_per_request = max_cost_per_request
        self.confidence_history = deque(maxlen=10000)
        self.cost_history = deque(maxlen=10000)
        self.last_mutation_run = 0
        self.run_counter = 0
        self._lock = threading.Lock()

    def record_run(self, confidence: float, cost: float, mutation_applied: bool):
        with self._lock:
            self.confidence_history.append(confidence)
            self.cost_history.append(cost)
            if mutation_applied:
                self.last_mutation_run = self.run_counter
            self.run_counter += 1
            if self.mode == KernelMode.GROWTH:
                if self._should_transition():
                    self.mode = KernelMode.PRODUCTION
                    self.logger.log_event("transition_to_production", {
                        "reason": "stability criteria met",
                        "run": self.run_counter
                    })

    def _should_transition(self) -> bool:
        if len(self.confidence_history) < self.stability_runs:
            return False
        recent_conf = list(self.confidence_history)[-self.stability_runs:]
        recent_cost = list(self.cost_history)[-self.stability_runs:]
        stable_conf = all(c >= self.min_confidence for c in recent_conf)
        low_cost = (sum(recent_cost)/len(recent_cost)) <= self.max_cost_per_request
        no_recent_mutation = (self.run_counter - self.last_mutation_run) > self.stability_runs
        return stable_conf and low_cost and no_recent_mutation
