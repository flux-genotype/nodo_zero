import time
import os
import tempfile
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from flux.core.entities import Ecosystem
from flux.kernel.interpreter import EcosystemInterpreter
from flux.kernel.ecosystem_store import EcosystemStore
from flux.kernel.growth_supervisor import KernelMode
from flux.kernel.human_interface import DummyHumanInterface
from flux.kernel.monitoring import NoopMetrics
from flux.parser.flux_serializer import serialize_ecosystem


class MetaDesigner:
    def __init__(
        self,
        interpreter: EcosystemInterpreter,
        store: EcosystemStore,
        judge_entity_name: str,
        architect_entity_name: Optional[str] = None,
        simulation_runs: int = 10,
        improvement_threshold: float = 0.05,
        max_growth_iterations: int = 20
    ):
        self.interpreter = interpreter
        self.logger = interpreter.logger
        self.loader = interpreter.loader
        self.policy = interpreter.policy
        self.metrics = interpreter.metrics
        self.store = store
        self.judge_entity_name = judge_entity_name
        self.architect_entity_name = architect_entity_name or judge_entity_name
        self.simulation_runs = simulation_runs
        self.improvement_threshold = improvement_threshold
        self.max_growth_iterations = max_growth_iterations
        self.growth_iterations = 0

    def evaluate_ecosystem(self, eco: Ecosystem, historical_inputs: List[Dict[str, str]]) -> float:
        if not eco.attractors:
            self.logger.log_event("evaluation_skipped", {"reason": "no attractors"})
            return 0.0
        if self.judge_entity_name not in eco.entities:
            self.logger.log_event("evaluation_missing_judge", {
                "ecosystem": eco.name,
                "judge_entity": self.judge_entity_name
            })
            return 0.5

        dummy_human = DummyHumanInterface()
        temp_interpreter = EcosystemInterpreter(
            ecosystem=eco,
            loader=self.loader,
            logger=self.logger,
            human_interface=dummy_human,
            tenant_id=self.interpreter.tenant_id,
            tenant_budget=float("inf"),
            metrics=NoopMetrics(),
            mode=KernelMode.SIMULATION
        )
        temp_interpreter.judge_entity_name = self.judge_entity_name
        temp_interpreter.require_judge = True
        confidences = []
        for inp in historical_inputs[:self.simulation_runs]:
            prompt = inp["prompt"]
            attractor_name = inp.get("attractor", next(iter(eco.attractors.keys())))
            try:
                _, conf, _ = temp_interpreter.run_attractor_with_confidence(attractor_name, prompt)
                confidences.append(conf)
            except Exception as e:
                self.logger.log_event("simulation_run_error", {"error": str(e)})
                confidences.append(0.0)
        return sum(confidences) / len(confidences) if confidences else 0.0

    def propose_mutation(self) -> Ecosystem:
        current_eco = self.interpreter.ecosystem
        eco_desc = serialize_ecosystem(current_eco)
        recent_conf = self.interpreter.confidence_evaluator.history.get("__global__", [])
        avg_conf = sum(recent_conf[-20:]) / min(20, len(recent_conf)) if recent_conf else 0.5
        trend = self.interpreter.confidence_evaluator.get_trend(window=10, key="__global__")

        prompt = (
            "You are an AI system architect. Output ONLY a FLUX ecosystem definition in .flux format.\n"
            "Do NOT use markdown code fences (```). Do NOT add any explanation.\n"
            "Start exactly with the word ECOSYSTEM followed by a space and the ecosystem name in double quotes.\n"
            "Use the exact same syntax as the example below.\n\n"
            "Current ecosystem:\n"
            f"{eco_desc}\n\n"
            f"Average confidence: {avg_conf:.2f}. Trend: {trend:+.2f}.\n"
            "Propose a modified version that may improve confidence, reduce cost or latency.\n"
            "You can change entity assignments, add/remove stages, adjust temperatures, change judge entity, etc.\n"
        )
        prompt += (
            "\nSUGGESTED IMPROVEMENT STRATEGIES (choose at least one):\n"
            "- Modify the 'Answer' stage OBSERVE to request more detailed, factual, and well-structured responses.\n"
            "- For example: OBSERVE: \"Provide a thorough and accurate answer with supporting details.\"\n"
            "- Lower the TEMPERATURE of the 'Answer' stage to 0.3–0.4 to reduce randomness and improve factual precision.\n"
            "- Increase MAX_NEW_TOKENS to at least 1024 to allow complete and informative answers.\n"
            "- If the model used for 'fast' is weak, replace MODEL_KEY with a more capable one (e.g., change to a larger model).\n"
            "- Add a verification stage that checks facts, but ensure it uses a reliable entity and does not degrade confidence.\n"
            "- Do NOT add stages that merely rephrase without adding factual value or that will be heavily penalized by the judge.\n"
        )
        prompt += (
            "\nAVAILABLE MODEL KEYS (you can assign to any entity's MODEL_KEY):\n"
            "- tinyllama:latest (637 MB, very fast but low accuracy)\n"
            "- llama3.2:3b (2.0 GB, good balance of speed and accuracy)\n"
            "- qwen2.5-coder:3b (1.9 GB, strong at code and reasoning)\n"
            "- deepseek-coder:6.7b (3.8 GB, excellent at structured generation)\n"
            "- hermes3:8b (4.7 GB, very capable general-purpose model)\n"
            "The current 'fast' entity uses 'tinyllama:latest'. To improve confidence, replace it with a more capable model like 'llama3.2:3b' or 'deepseek-coder:6.7b'.\n"
        )
        prompt += (
            "\nCRITICAL SYNTAX RULES:\n"
            "- The POLICY block must be at the top level, not inside a STAGE or ENTITY.\n"
            "- Inside a STAGE you can ONLY have: EXECUTE, TEMPERATURE, MAX_NEW_TOKENS, PROMPT_TRANSFORM, OBSERVE, GROUND_TRUTH.\n"
            "- Do NOT put MAX_COST_PER_REQUEST, MAX_LATENCY_MS, REQUIRE_JUDGE, JUDGE_ENTITY inside a STAGE.\n"
            "- Each ENTITY must have exactly one MODEL_KEY.\n"
            "- The ecosystem definition must end with a closing brace after POLICY.\n"
            "- Do NOT duplicate ENTITY names.\n"
        )
        prompt += (
            "\nCRITICAL SYNTAX RULES:\n"
            "- PROMPT_TRANSFORM must be a single quoted string, NOT a list. Example: PROMPT_TRANSFORM: \"Make the answer friendlier.\"\n"
            "- Do NOT use square brackets [ ] inside a STAGE definition.\n"
            "- GROUND_TRUTH must be a single quoted string (e.g., GROUND_TRUTH: \"4\") or omitted entirely. Do NOT use true/false.\n"
            "- TEMPERATURE, MAX_NEW_TOKENS, OBSERVE, GROUND_TRUTH are all simple values, not lists.\n"
            "- The POLICY block must be at top level, not inside a STAGE.\n"
            "- Each ENTITY name must be unique.\n"
            "- The ecosystem definition must end with a closing brace after POLICY.\n"
            "- Do NOT add any text or comments after the final closing brace of the ecosystem.\n"
            "- The output must end immediately after the last '}' character.\n"
        )
        # Create debug folder if it doesn't exist
        debug_dir = "debug_architect"
        os.makedirs(debug_dir, exist_ok=True)

        # Save the prompt
        prompt_file = os.path.join(debug_dir, f"prompt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt")
        with open(prompt_file, "w") as f:
            f.write(prompt)

        try:
            new_flux_code, usage, cost = self.interpreter._generate_with_retry_and_fallback(
                self.architect_entity_name,
                prompt,
                temperature=0.1,
                max_new_tokens=2048,
                run_id=f"meta_architect_{int(time.time())}"
            )
            new_flux_code = new_flux_code.strip()
            print(f"[MetaDesigner] Architect response ({len(new_flux_code)} characters)")

            # ALWAYS save the RAW output (before any cleaning)
            raw_file = os.path.join(debug_dir, f"raw_output_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt")
            with open(raw_file, "w") as f:
                f.write(new_flux_code)
            print(f"[MetaDesigner] Raw output saved in {raw_file}")

            # === BEGIN ROBUST CLEANING ===

            # 1. Remove any markdown code fences (``` ... ```)
            if "```" in new_flux_code:
                parts = new_flux_code.split("```")
                extracted = None
                for i in range(1, len(parts), 2):
                    candidate = parts[i].strip()
                    if candidate.lower().startswith("flux"):
                        candidate = candidate[4:].strip()
                    if candidate.upper().startswith("ECOSYSTEM"):
                        extracted = candidate
                        break
                if extracted:
                    new_flux_code = extracted
            # 1.5 Remove special model tokens (e.g. <|begin_of_sentence|>)
            new_flux_code = re.sub(r'<[^>]+>', '', new_flux_code)
            # 2. Search for the word ECOSYSTEM and cut everything before it
            match = re.search(r'ECOSYSTEM\s+"', new_flux_code, re.IGNORECASE)
            if match:
                new_flux_code = new_flux_code[match.start():]
            else:
                raise ValueError("The response does not contain a valid FLUX ecosystem definition.")

            # 3. Truncate everything after the last closing curly brace
            last_brace = new_flux_code.rfind('}')
            if last_brace != -1:
                new_flux_code = new_flux_code[:last_brace+1]
            else:
                raise ValueError("No closing brace found in FLUX code.")

            new_flux_code = new_flux_code.strip()

            # 4. Remove ALL lines starting with "- " (dash space) and section header lines
            lines = new_flux_code.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip empty lines? better keep them for now, but remove unwanted lines
                if (stripped.startswith('- ') or
                    stripped.startswith('Average confidence:') or
                    stripped.startswith('Trend:') or
                    stripped.startswith('Proposed changes:') or
                    stripped.startswith('Propose a modified version') or
                    stripped.startswith('You can change entity assignments') or
                    stripped.startswith('SUGGESTED IMPROVEMENT STRATEGIES') or
                    stripped.startswith('CRITICAL SYNTAX RULES') or
                    stripped.startswith('Do NOT') or
                    stripped.startswith('The POLICY block must be') or
                    stripped.startswith('Each ENTITY') or
                    stripped.startswith('Inside a STAGE you can ONLY') or
                    stripped.startswith('PROMPT_TRANSFORM must be') or
                    stripped.startswith('GROUND_TRUTH must be') or
                    stripped.startswith('TEMPERATURE, MAX_NEW_TOKENS') or
                    stripped.startswith('Add a verification stage') or
                    stripped.startswith('If the model used for')):
                    continue
                cleaned_lines.append(line)
            new_flux_code = '\n'.join(cleaned_lines).strip()

            # 5. Further cleaning with regex as backup
            patterns_to_remove = [
                r'\bAverage confidence:.*',
                r'\bTrend:.*',
                r'\bProposed changes:.*',
                r'\bPropose a modified version.*',
                r'\bYou can change entity assignments.*',
                r'\bSUGGESTED IMPROVEMENT STRATEGIES.*',
                r'\bCRITICAL SYNTAX RULES.*',
            ]
            for pattern in patterns_to_remove:
                new_flux_code = re.sub(pattern, '', new_flux_code, flags=re.IGNORECASE | re.MULTILINE)
            new_flux_code = new_flux_code.strip()

            # === END OF CLEANING ===
            print(f"[DEBUG] Length after cleaning: {len(new_flux_code)} characters. First 100: {new_flux_code[:100]}") #<------------DEBUG================

            # Save the cleaned version
            clean_file = os.path.join(debug_dir, f"cleaned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.flux")
            with open(clean_file, "w") as f:
                f.write(new_flux_code)
            print(f"[MetaDesigner] Cleaned version saved in {clean_file}")

            # Parsing
            from flux.parser import FluxParser
            parser = FluxParser()
            tf = tempfile.NamedTemporaryFile(mode='w', suffix='.flux', delete=False)
            try:
                tf.write(new_flux_code)
                tf.flush()
                tf.close()
                new_eco = parser.parse_file(tf.name)
            except Exception as parse_err:
                self.logger.log_event("mutation_proposal_parse_error", {"error": str(parse_err), "file": clean_file})
                print(f"[MetaDesigner] PARSING ERROR: {parse_err}")
                raise
            finally:
                os.unlink(tf.name)

            self.logger.log_event("mutation_proposed", {"new_ecosystem": new_eco.name})
            return new_eco

        except Exception as e:
            self.logger.log_event("mutation_proposal_failed", {"error": str(e)})
            print(f"[MetaDesigner] Proposal failure: {e}")
            raise

    def growth_cycle(self, historical_inputs: List[Dict[str, str]]):
        current_eco = self.interpreter.ecosystem
        current_score = self.evaluate_ecosystem(current_eco, historical_inputs)
        self.logger.log_event("meta_designer_cycle_start", {
            "iteration": self.growth_iterations,
            "current_score": current_score
        })
        print(f"[MetaDesigner] Cycle {self.growth_iterations} - Current score: {current_score:.3f}")
        self.growth_iterations += 1
        if self.metrics:
            self.metrics.set_meta_designer_score("current", current_score)
        try:
            new_eco = self.propose_mutation()
        except Exception as e:
            self.logger.log_event("mutation_proposal_failed_in_cycle", {"error": str(e)})
            print(f"[MetaDesigner] Proposal failed: {e}")
            return False

        try:
            new_score = self.evaluate_ecosystem(new_eco, historical_inputs)
        except Exception as e:
            self.logger.log_event("mutation_evaluation_failed", {"error": str(e)})
            print(f"[MetaDesigner] Evaluation failed: {e}")
            return False

        improvement = new_score - current_score
        self.logger.log_event("mutation_evaluated", {
            "new_ecosystem": new_eco.name,
            "score": new_score,
            "improvement": improvement
        })
        print(f"[MetaDesigner] New score: {new_score:.3f} (improvement: {improvement:+.3f})")
        if self.metrics:
            self.metrics.set_meta_designer_score("improvement", improvement)
        if improvement > self.improvement_threshold:
            self.apply_mutation(new_eco, new_score)
            print(f"[MetaDesigner] ✅ Mutation APPLIED.")
            return True
        else:
            self.logger.log_event("mutation_rejected", {"reason": "insufficient improvement"})
            print(f"[MetaDesigner] ❌ Mutation rejected (insufficient improvement).")
            return False

    def apply_mutation(self, new_eco: Ecosystem, new_score: float):
        self.store.save_ecosystem(new_eco, version=f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
        self.interpreter.replace_ecosystem(new_eco)
        if self.interpreter.growth_supervisor:
            self.interpreter.growth_supervisor.last_mutation_run = self.interpreter.growth_supervisor.run_counter
        self.logger.log_event("ecosystem_replaced", {
            "new_name": new_eco.name,
            "score": new_score
        })

    def run_growth_phase(self, historical_inputs: List[Dict[str, str]], max_iterations: int = None):
        max_iter = max_iterations or self.max_growth_iterations
        consecutive_failures = 0
        for i in range(max_iter):
            success = self.growth_cycle(historical_inputs)
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    self.logger.log_event("growth_phase_early_stop", {
                        "reason": f"too many consecutive failures ({consecutive_failures})"
                    })
                    break
        self.logger.log_event("growth_phase_complete", {"total_iterations": self.growth_iterations})
