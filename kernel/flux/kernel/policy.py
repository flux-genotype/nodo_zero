import threading
from collections import OrderedDict

class FluxPolicy:
    def __init__(self, config: dict, logger, tenant_id: str = "default", tenant_budget: float = float("inf")):
        self.logger = logger
        self.tenant_id = tenant_id
        self.max_cost_per_request = config.get("MAX_COST_PER_REQUEST", float("inf"))
        self.max_latency_ms = config.get("MAX_LATENCY_MS", 60000)
        self.max_human_iterations = config.get("MAX_HUMAN_ITERATIONS", 3)
        self.tenant_budget_limit = tenant_budget
        self.tenant_budget_used = 0.0
        self.human_interactions = OrderedDict()
        self._lock = threading.Lock()

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self.tenant_budget_used

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int = 0) -> float:
        return (prompt_tokens + completion_tokens) * 0.000002

    def try_commit_cost(self, estimated_cost: float, run_id: str) -> bool:
        with self._lock:
            if estimated_cost > self.max_cost_per_request:
                self.logger.log_event("budget_exceeded_per_request", {
                    "run_id": run_id,
                    "estimated_cost": estimated_cost,
                    "max_per_request": self.max_cost_per_request
                })
                return False
            if self.tenant_budget_used + estimated_cost > self.tenant_budget_limit:
                self.logger.log_event("tenant_budget_exceeded", {
                    "run_id": run_id,
                    "tenant_id": self.tenant_id,
                    "estimated_cost": estimated_cost,
                    "budget_used": self.tenant_budget_used
                })
                return False
            self.tenant_budget_used += estimated_cost
            return True

    def check_latency(self, elapsed_ms: float, run_id: str) -> bool:
        return elapsed_ms <= self.max_latency_ms

    def check_human_iterations(self, run_id: str) -> bool:
        with self._lock:
            return self.human_interactions.get(run_id, 0) < self.max_human_iterations

    def record_human_interaction(self, run_id: str):
        with self._lock:
            self.human_interactions[run_id] = self.human_interactions.get(run_id, 0) + 1

    def clean_run(self, run_id: str):
        with self._lock:
            self.human_interactions.pop(run_id, None)
