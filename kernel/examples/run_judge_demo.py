import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flux.parser import FluxParser
from flux.kernel import EcosystemInterpreter, FluxModelLoader, FluxLogger, FluxMetrics
from prometheus_client import start_http_server

class MockBackend:
    def __init__(self):
        self.counter = 0

    def generate(self, model_name, prompt, **kwargs):
        if model_name == "judge-model":
            self.counter += 1
            if self.counter % 2 == 0:
                return {"text": "0.9", "usage": {"prompt_tokens": 50, "completion_tokens": 2}}
            else:
                return {"text": "0.4", "usage": {"prompt_tokens": 50, "completion_tokens": 2}}
        else:
            return {
                "text": f"Answer from {model_name}: Paris is the capital of France.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}
            }

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus metrics server started on :8000")

    parser = FluxParser()
    ecosystem = parser.parse_file("examples/judge_demo.flux")
    backend = MockBackend()
    loader = FluxModelLoader(backend, model_mapping={"fast": "fast", "judge-model": "judge-model"})
    logger = FluxLogger("judge_demo.jsonl")
    metrics = FluxMetrics()
    interpreter = EcosystemInterpreter(ecosystem, loader, logger, metrics=metrics)

    for i in range(5):
        print(f"\nRun {i+1}")
        result, cost = interpreter.run_attractor("qa", "What is the capital of France?")
        print(f"Result: {result[:200]}")

    print("\nCurrent metrics:")
    print(metrics.get_metrics().decode())
