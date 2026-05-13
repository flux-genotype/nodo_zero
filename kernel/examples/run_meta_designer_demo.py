import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flux.parser import FluxParser
from flux.kernel import (
    EcosystemInterpreter, FluxModelLoader, FluxLogger, FluxMetrics,
    MetaDesigner, EcosystemStore, GrowthSupervisor
)
from flux.kernel.growth_supervisor import KernelMode

class MockBackend:
    def __init__(self):
        self.counter = 0
        self.models = {
            "fast": ("Paris is the capital of France.", {"prompt_tokens": 10, "completion_tokens": 20}),
            "accurate": ("The capital of France is Paris, known for its culture.", {"prompt_tokens": 10, "completion_tokens": 30}),
            "judge-model": ("0.85", {"prompt_tokens": 50, "completion_tokens": 2}),
        }

    def generate(self, model_name, prompt, **kwargs):
        if model_name in self.models:
            text, usage = self.models[model_name]
            if model_name == "fast":
                if "capital" in prompt.lower():
                    text = "Paris is the capital of France."
                else:
                    text = "I'm not sure."
            return {"text": text, "usage": usage}
        return {"text": "error", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

if __name__ == "__main__":
    parser = FluxParser()
    ecosystem = parser.parse_file("examples/meta_designer_demo.flux")
    backend = MockBackend()
    loader = FluxModelLoader(backend, model_mapping={
        "fast": "fast",
        "accurate": "accurate",
        "judge-model": "judge-model"
    })
    logger = FluxLogger("meta_designer_demo.jsonl")
    metrics = FluxMetrics()
    store = EcosystemStore("./ecosystems")

    interpreter = EcosystemInterpreter(
        ecosystem, loader, logger, metrics=metrics, mode=KernelMode.GROWTH
    )
    supervisor = GrowthSupervisor(logger, stability_runs=5, min_confidence=0.8, max_cost_per_request=0.05)
    meta_designer = MetaDesigner(
        interpreter=interpreter,
        store=store,
        judge_entity_name="judge",
        simulation_runs=3,
        improvement_threshold=0.1
    )
    interpreter.growth_supervisor = supervisor
    interpreter.meta_designer = meta_designer

    prompts = [
        "What is the capital of France?",
        "Name the capital of Italy.",
        "What is the capital of Germany?"
    ]
    for p in prompts:
        print(f"\nProcessing: {p}")
        result, conf, cost = interpreter.run_attractor_with_confidence("qa", p)
        print(f"Result: {result[:100]} | Confidence: {conf:.2f} | Cost: {cost:.6f}")
        time.sleep(0.1)

    print("\n=== Starting Growth Cycle ===")
    for i in range(2):
        meta_designer.growth_cycle(interpreter.historical_inputs)
        time.sleep(0.5)

    print(f"\nCurrent mode: {supervisor.mode}")
    print("Active ecosystem versions:", store.get_active_version())
