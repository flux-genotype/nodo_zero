#!/usr/bin/env python3
"""
Test FLUX Kernel with tinyllama (generator) + llama3.2:3b (judge) + hermes3:8b (architect)
Debug version with Prometheus metrics server
"""

import sys, os, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("DEBUG: Script avviato", flush=True)

# ---------- Import del kernel e dipendenze ----------
try:
    from flux.parser import FluxParser
    from flux.kernel import EcosystemInterpreter, FluxModelLoader, FluxLogger, FluxMetrics
    from flux.kernel.growth_supervisor import GrowthSupervisor, KernelMode
    from flux.kernel.meta_designer import MetaDesigner
    from flux.kernel.ecosystem_store import EcosystemStore
    from backend_ollama import OllamaBackend
    from test_prompts import test_prompts
    print("DEBUG: Tutti gli import completati con successo", flush=True)
except Exception as e:
    print(f"ERRORE durante gli import: {e}", flush=True)
    sys.exit(1)

# ---------- Avvia il server metriche SUBITO (prima di main) ----------
from prometheus_client import start_http_server
from flux.kernel.monitoring import FluxMetrics
FluxMetrics._shared_instance = None
start_http_server(8000, registry=FluxMetrics._shared_registry)
print("Prometheus metrics server started on http://localhost:8000/metrics", flush=True)

# ---------- Definizione di main() (invariata) ----------
def main():
    print("DEBUG: Entrato in main()", flush=True)

    # ---- Load ecosystem ----
    eco_path = os.path.join("examples", "judge_demo.flux")
    if not os.path.exists(eco_path):
        print(f"ERROR: file {eco_path} not found.")
        sys.exit(1)
    parser = FluxParser()
    ecosystem = parser.parse_file(eco_path)
    print(f"Ecosystem '{ecosystem.name}' loaded with {len(ecosystem.attractors)} attractor(s).")

    # ---- Ollama Backend ----
    backend = OllamaBackend()
    loader = FluxModelLoader(backend,
        model_mapping={
            "fast": "tinyllama:latest",               # intentionally weak
            "judge-model": "llama3.2:3b",             # fixed judge
            "architect": "hermes3:8b"                 # (or deepseek-coder:6.7b for more speed)
        }
    )

    # ---- Logger and metrics ----
    logger = FluxLogger("test.jsonl")
    metrics = FluxMetrics()

    # ---- Interpreter in GROWTH mode ----
    interpreter = EcosystemInterpreter(
        ecosystem, loader, logger, metrics=metrics,
        mode=KernelMode.GROWTH
    )

    # ---- Supervisor ----
    supervisor = GrowthSupervisor(logger, stability_runs=3, min_confidence=0.6, max_cost_per_request=0.1)
    interpreter.growth_supervisor = supervisor

    # ---- MetaDesigner with zero threshold ----
    store = EcosystemStore("./eco_store")
    meta = MetaDesigner(
        interpreter=interpreter,
        store=store,
        judge_entity_name="judge",
        architect_entity_name="architect",   # <-- uses the separate entity
        simulation_runs=2,
        improvement_threshold=-0.1,          # apply even the slightest improvement
        max_growth_iterations=5
    )
    interpreter.meta_designer = meta

    # ---- Phase 1: Data collection ----
    print("\n=== Phase 1: Data collection ===")
    for i, item in enumerate(test_prompts):
        prompt = item["prompt"]
        attractor = item.get("attractor", "qa")
        print(f"\n--- Run {i+1}: {prompt} ---")
        try:
            out, conf, cost = interpreter.run_attractor_with_confidence(attractor, prompt)
            print(f"Response  : {out.strip()[:120]}")
            print(f"Confidence: {conf:.2f} | Cost: {cost:.6f} | Mode: {supervisor.mode}")
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(0.3)

    # ---- Phase 2: MetaDesigner (growth) ----
    print("\n=== Phase 2: MetaDesigner (growth) ===")
    for ciclo in range(5):
        print(f"\n--- Growth cycle {ciclo+1} ---")
        success = meta.growth_cycle(interpreter.historical_inputs)
        if success:
            print("✔ New version applied!")
        else:
            print("✘ No mutation applied (likely generation error).")
        active = store.get_active_version()
        if active:
            print(f"Active version: {active}")
        time.sleep(1)

    print(f"\n=== Test completed. Final mode: {supervisor.mode} ===")
    print("Log saved in test.jsonl")
    print("Ecosystem versions in ./eco_store/")

# ---------- Esecuzione protetta ----------
if __name__ == "__main__":
    print("DEBUG: if __name__ == '__main__' raggiunto", flush=True)
    try:
        main()
    except Exception as e:
        print(f"ERRORE in main(): {e}", flush=True)

    # Mantiene il processo vivo per permettere a Prometheus lo scraping
    import time
    while True:
        time.sleep(1)
