class BudgetExceededError(Exception):
    pass

class LatencyExceededError(Exception):
    pass

class EmptyResponseError(Exception):
    pass

class GenerationFailedError(Exception):
    def __init__(self, message, cost=0.0):
        super().__init__(message)
        self.cost = cost

class HumanIterationExceededError(GenerationFailedError):
    def __init__(self, message="Maximum human iterations exceeded for this run"):
        super().__init__(message)
