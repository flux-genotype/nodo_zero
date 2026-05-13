import time
import random
import threading
from typing import Optional, Tuple, Dict, Any
from flux.core.entities import Ecosystem, Attractor, Stage, EntityType, Entity
from flux.kernel.loader import FluxModelLoader
from flux.kernel.logger import FluxLogger
from flux.kernel.policy import FluxPolicy
from flux.kernel.mutation import MutationEngine
from flux.kernel.recursive_engine import RecursiveEngine, RecursiveConfig
from flux.kernel.confidence import ConfidenceEvaluator
from flux.kernel.human_interface import CLIHumanInterface, HumanInterface
from flux.kernel.exceptions import (
    BudgetExceededError, LatencyExceededError, EmptyResponseError, GenerationFailedError,
    HumanIterationExceededError
)
from flux.kernel.monitoring import FluxMetrics
from flux.kernel.growth_supervisor import KernelMode
from flux.utils.token_counter import TokenCounter
from flux.utils.rwlock import RWLock
from flux.core.constants import (
    PHI, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BACKOFF_FACTOR, DEFAULT_RETRY_BASE_DELAY_SEC,
    CONFIDENCE_TARGET, AUTO_TUNE_WINDOW_SIZE, AUTO_TUNE_GAIN,
    AUTO_TUNE_MIN_FACTOR, AUTO_TUNE_MAX_FACTOR, AUTO_TUNE_EMA_ALPHA
)

class EcosystemInterpreter:
    def __init__(self, ecosystem: Ecosystem, loader: FluxModelLoader, logger: FluxLogger,
                 human_interface: HumanInterface = None, tenant_id: str = "default",
                 tenant_budget: float = float("inf"), metrics: Optional[FluxMetrics] = None,
                 mode: KernelMode = KernelMode.GROWTH,
                 growth_supervisor: Optional['GrowthSupervisor'] = None,
                 meta_designer: Optional['MetaDesigner'] = None):
        self.ecosystem = ecosystem
        self.loader = loader
        self.logger = logger
        self.tenant_id = tenant_id
        self.policy = FluxPolicy(ecosystem.policy, logger, tenant_id=tenant_id, tenant_budget=tenant_budget)
        self.mutation_engine = MutationEngine(ecosystem, logger)
        self.token_counter = TokenCounter()
        self.recursive_engine = None
        self.confidence_evaluator = ConfidenceEvaluator()
        self.confidence_history = {}
        self.confidence_ema = {}
        self.human_interface = human_interface or CLIHumanInterface()
        self.metrics = metrics if metrics is not None else FluxMetrics.get_shared()
        self._lock = threading.Lock()
        self._ecosystem_lock = RWLock()
        self._recursive_lock = threading.Lock()

        self.judge_entity_name = ecosystem.policy.get("JUDGE_ENTITY", None)
        self.require_judge = ecosystem.policy.get("REQUIRE_JUDGE", True)

        self.mode = mode
        self.growth_supervisor = growth_supervisor
        self.meta_designer = meta_designer
        self.historical_inputs = []
        self.max_historical_inputs = 1000

    def replace_ecosystem(self, new_ecosystem: Ecosystem):
        with self._ecosystem_lock.writer():
            self.ecosystem = new_ecosystem
            new_policy = new_ecosystem.policy
            self.policy.max_cost_per_request = new_policy.get("MAX_COST_PER_REQUEST", float("inf"))
            self.policy.max_latency_ms = new_policy.get("MAX_LATENCY_MS", 30000)
            self.policy.max_human_iterations = new_policy.get("MAX_HUMAN_ITERATIONS", 3)
            self.mutation_engine.ecosystem = new_ecosystem
            self.recursive_engine = None
            with self._lock:
                self.confidence_evaluator = ConfidenceEvaluator()
                self.confidence_ema = {}
                self.confidence_history = {}
            self.judge_entity_name = new_ecosystem.policy.get("JUDGE_ENTITY", None)
            self.require_judge = new_ecosystem.policy.get("REQUIRE_JUDGE", True)

    def run_attractor_with_confidence(self, attractor_name: str, user_prompt: str,
                                      intent: Optional[str] = None,
                                      ground_truth: Optional[str] = None) -> Tuple[str, float, float]:
        with self._ecosystem_lock.reader():
            output, run_cost = self.run_attractor(attractor_name, user_prompt, intent, ground_truth)
        conf_list = self.confidence_evaluator.history.get(attractor_name, [])
        last_conf = 0.5
        if conf_list:
            last_conf = conf_list[-1]
        return output, last_conf, run_cost

    def _adaptive_temperature(self, base_temp: float, attractor_name: str) -> float:
        with self._lock:
            ema = self.confidence_ema.get(attractor_name, None)
        if ema is None:
            return base_temp
        error = CONFIDENCE_TARGET - ema
        factor = 1.0 + AUTO_TUNE_GAIN * error
        factor = max(AUTO_TUNE_MIN_FACTOR, min(AUTO_TUNE_MAX_FACTOR, factor))
        adapted = base_temp * factor
        self.logger.log_event("adaptive_temperature", {
            "attractor": attractor_name,
            "confidence_ema": ema,
            "target": CONFIDENCE_TARGET,
            "error": error,
            "factor": factor,
            "base_temp": base_temp,
            "adapted_temp": adapted
        })
        return adapted

    def _should_use_recursive_mode(self, prompt: str, stage: Stage) -> bool:
        entity = self.ecosystem.entities.get(stage.entity_name)
        if not entity:
            return False
        if entity.type == EntityType.RECURSIVE_ENSEMBLE:
            return True
        model_key = entity.model_key or entity.name
        model_info = self.loader.get_model_info(model_key)
        max_context = model_info.get("max_context", 8192)
        tokens, _ = self.token_counter.count(prompt)
        return tokens > (max_context * 0.8)

    def _init_recursive_engine(self, entity: Optional[Entity] = None):
        with self._recursive_lock:
            if self.recursive_engine is None:
                config = RecursiveConfig()
                if entity and entity.recursive_config:
                    rec_conf = entity.recursive_config
                    config.max_chunk_tokens = rec_conf.get("max_chunk_tokens", config.max_chunk_tokens)
                    config.chunk_overlap = rec_conf.get("chunk_overlap", config.chunk_overlap)
                    config.sub_model_key = rec_conf.get("sub_model_key", config.sub_model_key)
                    config.root_model_key = rec_conf.get("root_model_key", config.root_model_key)
                    config.max_workers = rec_conf.get("max_workers", 4)
                self.recursive_engine = RecursiveEngine(
                    loader=self.loader,
                    logger=self.logger,
                    policy=self.policy,
                    token_counter=self.token_counter,
                    config=config
                )

    def _is_transient_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        transient_indicators = [
            "timeout", "connection refused", "too many requests", "rate limit",
            "overloaded", "temporary", "service unavailable", "internal server error",
            "500", "502", "503", "504"
        ]
        return any(ind in error_str for ind in transient_indicators)

    def _generate_with_retry_and_fallback(
        self,
        entity_name: str,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        run_id: str,
        depth: int = 0
    ) -> Tuple[str, Dict[str, Any], float]:
        entity = self.ecosystem.entities.get(entity_name)
        if not entity:
            raise ValueError(f"Entity '{entity_name}' not found for fallback chain")

        model_key = entity.model_key or entity.name
        max_retries = DEFAULT_MAX_RETRIES
        total_cost = 0.0
        last_non_transient_error = None

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response_data = self.loader.generate(
                    model_key, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                elapsed_ms = (time.time() - start_time) * 1000

                usage = response_data.get("usage", {})
                real_cost = self.policy.estimate_cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0)
                )
                if not self.policy.try_commit_cost(real_cost, run_id):
                    raise BudgetExceededError(f"Budget exceeded for tenant {self.tenant_id}")

                if not self.policy.check_latency(elapsed_ms, run_id):
                    raise LatencyExceededError(f"Latency {elapsed_ms:.2f}ms exceeds limit")

                response_text = response_data.get("text", "")
                if not response_text or len(response_text.strip()) == 0:
                    raise EmptyResponseError("Generated response is empty")

                self.logger.log_event("generation_success", {
                    "run_id": run_id,
                    "entity": entity_name,
                    "attempt": attempt,
                    "latency_ms": elapsed_ms,
                    "cost": real_cost
                })
                return response_text, usage, real_cost

            except (BudgetExceededError, LatencyExceededError, EmptyResponseError) as e:
                self.logger.log_event("non_transient_error", {
                    "run_id": run_id,
                    "entity": entity_name,
                    "error": str(e),
                    "attempt": attempt
                })
                last_non_transient_error = e
                break

            except Exception as e:
                if attempt < max_retries and self._is_transient_error(e):
                    delay = DEFAULT_RETRY_BASE_DELAY_SEC * (DEFAULT_RETRY_BACKOFF_FACTOR ** attempt)
                    delay = delay * (0.5 + random.random())
                    self.logger.log_event("retry_attempt", {
                        "run_id": run_id,
                        "entity": entity_name,
                        "attempt": attempt + 1,
                        "delay_sec": delay,
                        "error": str(e)
                    })
                    time.sleep(delay)
                    continue
                else:
                    self.logger.log_event("entity_failure", {
                        "run_id": run_id,
                        "entity": entity_name,
                        "error": str(e),
                        "attempt": attempt
                    })
                    last_non_transient_error = e
                    break

        if entity.fallback and depth <= 5:
            if isinstance(last_non_transient_error, (BudgetExceededError, LatencyExceededError)):
                raise GenerationFailedError(
                    f"Fallback blocked due to {type(last_non_transient_error).__name__} for {entity_name}",
                    cost=total_cost
                )
            fallback_entity_name = entity.fallback
            if fallback_entity_name not in self.ecosystem.entities:
                self.logger.log_event("fallback_entity_missing", {
                    "run_id": run_id,
                    "from_entity": entity_name,
                    "expected_fallback": fallback_entity_name
                })
                raise GenerationFailedError(
                    f"Fallback entity '{fallback_entity_name}' not found in ecosystem for '{entity_name}'",
                    cost=total_cost
                )
            self.logger.log_event("fallback_trigger", {
                "run_id": run_id,
                "from_entity": entity_name,
                "to_entity": fallback_entity_name
            })
            try:
                fb_text, fb_usage, fb_cost = self._generate_with_retry_and_fallback(
                    fallback_entity_name, prompt, temperature, max_new_tokens, run_id, depth + 1
                )
                total_cost += fb_cost
                return fb_text, fb_usage, total_cost
            except GenerationFailedError as e:
                total_cost += e.cost
                raise GenerationFailedError(f"Fallback chain exhausted for {entity_name}", cost=total_cost) from e
        else:
            if last_non_transient_error:
                raise GenerationFailedError(f"Generation failed for {entity_name}: {last_non_transient_error}", cost=total_cost)
            else:
                raise GenerationFailedError(f"Generation failed for {entity_name}", cost=total_cost)

    def _execute_judge_evaluation(self, generated: str, reference: Optional[str],
                                  source_prompt: Optional[str], attractor_name: str,
                                  run_id: str) -> float:
        if not self.judge_entity_name:
            if self.require_judge:
                raise RuntimeError("No JUDGE_ENTITY configured in POLICY, but REQUIRE_JUDGE is true.")
            self.logger.log_event("judge_missing_skip", {"attractor": attractor_name})
            return -1.0

        max_gen_chars = 2000
        truncated_generated = generated[:max_gen_chars] + ("..." if len(generated) > max_gen_chars else "")
        truncated_reference = (reference[:max_gen_chars] + "...") if reference and len(reference) > max_gen_chars else reference

        prompt = (
            "You are an impartial judge. Evaluate the quality of the following answer on a scale from 0 to 1, "
            "where 1 is perfectly correct, relevant, and well-formatted. Consider the original question and any reference answer.\n\n"
        )
        if source_prompt:
            prompt += f"Question: {source_prompt[:1000]}\n\n"
        if truncated_reference:
            prompt += f"Reference answer: {truncated_reference}\n\n"
        prompt += f"Answer to evaluate: {truncated_generated}\n\n"
        prompt += "Provide only the numeric score (0.0 to 1.0) and nothing else."

        try:
            judge_response, usage, cost = self._generate_with_retry_and_fallback(
                self.judge_entity_name,
                prompt,
                temperature=0.0,
                max_new_tokens=32,
                run_id=run_id + "_judge"
            )
        except Exception as e:
            self.logger.log_event("judge_generation_failed", {
                "attractor": attractor_name,
                "error": str(e)
            })
            if self.require_judge:
                raise RuntimeError(f"Judge evaluation failed for '{attractor_name}': {e}") from e
            return -1.0

        score = self.confidence_evaluator.extract_score(judge_response)
        self.logger.log_event("judge_evaluation", {
            "attractor": attractor_name,
            "judge_response": judge_response,
            "extracted_score": score
        })
        return score

    def _execute_stage(self, stage: Stage, prompt: str, run_id: str, attractor_name: str) -> Tuple[str, float]:
        entity = self.ecosystem.entities.get(stage.entity_name)
        if not entity:
            raise ValueError(f"Entity {stage.entity_name} not found")

        max_new_tokens = stage.max_new_tokens
        adapted_temp = self._adaptive_temperature(stage.temperature, attractor_name)

        need_human = (entity.type == EntityType.HUMAN_IN_THE_LOOP)
        if need_human and not self.policy.check_human_iterations(run_id):
            raise HumanIterationExceededError(f"Human iteration limit reached for run {run_id}")

        start = time.time()
        success = True
        error_type = None
        stage_cost = 0.0
        usage = {}
        response_text = ""

        try:
            response_text, usage, stage_cost = self._generate_with_retry_and_fallback(
                stage.entity_name, prompt, adapted_temp, max_new_tokens, run_id
            )
            if need_human:
                self.policy.record_human_interaction(run_id)
                response_text = self._human_interaction(stage, response_text, run_id, remaining_time_ms=None)
                self.logger.log_event("hitl_interaction", {
                    "run_id": run_id,
                    "stage": stage.name,
                    "entity_type": entity.type.value,
                    "final_response_len": len(response_text)
                })
        except GenerationFailedError as e:
            stage_cost = e.cost
            response_text = f"[ERROR: {str(e)}]"
            success = False
            error_type = "GenerationFailedError"
        except Exception as e:
            self.logger.log_event("stage_unexpected_error", {
                "run_id": run_id,
                "stage": stage.name,
                "error": str(e)
            })
            response_text = f"[CRITICAL ERROR: {str(e)}]"
            success = False
            error_type = type(e).__name__
            stage_cost = 0.0

        duration = time.time() - start
        self.metrics.record_request(self.tenant_id, attractor_name, stage.name,
                                    duration, success, error_type)
        self.metrics.add_cost(self.tenant_id, stage_cost)
        self.metrics.set_stage_temperature(self.tenant_id, attractor_name, stage.name, adapted_temp)

        return response_text, stage_cost

    def _human_interaction(self, stage, ai_output, run_id, remaining_time_ms=None):
        context = {
            "run_id": run_id,
            "stage": stage.name,
            "original_query": stage.observe,
            "ai_response": ai_output[:500]
        }
        if remaining_time_ms is not None:
            context["remaining_time_ms"] = remaining_time_ms
        approval = self.human_interface.approve_continue(context)
        if approval:
            return ai_output
        else:
            correction = self.human_interface.request_input(
                "Enter correction or exact answer:", context
            )
            return correction if correction else ai_output

    def run_attractor(self, attractor_name: str, user_prompt: str, intent: Optional[str] = None,
                      ground_truth: Optional[str] = None) -> Tuple[str, float]:
        run_id = self.logger._generate_run_id()
        attractor = self.ecosystem.attractors.get(attractor_name)
        if not attractor:
            raise ValueError(f"Attractor {attractor_name} not found")

        self.logger.log_event("attractor_start", {
            "run_id": run_id,
            "attractor": attractor_name,
            "intent": intent,
            "prompt_length": len(user_prompt)
        })

        cost_before = self.policy.total_cost
        pipeline_input = user_prompt
        stage_outputs = []
        run_cost = 0.0

        try:
            for stage in attractor.stages:
                self.logger.log_event("stage_start", {"run_id": run_id, "stage": stage.name})
                if self._should_use_recursive_mode(pipeline_input, stage):
                    entity = self.ecosystem.entities.get(stage.entity_name)
                    self._init_recursive_engine(entity)
                    query = stage.observe if stage.observe else intent if intent else "Extract and summarize."
                    adapted_temp = self._adaptive_temperature(stage.temperature, attractor_name)
                    start_rec = time.time()
                    output, rec_cost = self.recursive_engine.execute_recursive_pipeline(
                        pipeline_input, query, run_id,
                        sub_temperature=adapted_temp, root_temperature=adapted_temp
                    )
                    duration = time.time() - start_rec
                    self.metrics.record_request(self.tenant_id, attractor_name, stage.name, duration, True)
                    self.metrics.add_cost(self.tenant_id, rec_cost)
                    self.metrics.set_stage_temperature(self.tenant_id, attractor_name, stage.name, adapted_temp)
                    stage_cost = rec_cost
                else:
                    output, stage_cost = self._execute_stage(stage, pipeline_input, run_id, attractor_name)

                if stage.prompt_transform:
                    output = stage.prompt_transform.replace("{input}", output)
                pipeline_input = output
                stage_outputs.append(output)
                run_cost += stage_cost
                self.logger.log_event("stage_complete", {"run_id": run_id, "stage": stage.name,
                                                         "output_length": len(output)})

            ref = attractor.stages[-1].ground_truth if attractor.stages and attractor.stages[-1].ground_truth else ground_truth
            try:
                confidence = self._execute_judge_evaluation(
                    generated=pipeline_input,
                    reference=ref,
                    source_prompt=user_prompt,
                    attractor_name=attractor_name,
                    run_id=run_id
                )
            except Exception as e:
                self.logger.log_event("judge_evaluation_fatal", {
                    "attractor": attractor_name,
                    "error": str(e)
                })
                if self.require_judge:
                    raise GenerationFailedError(f"Judge evaluation required but failed: {e}") from e
                confidence = -1.0

            with self._lock:
                if confidence >= 0.0:
                    self.logger.log_event("confidence_evaluation", {
                        "run_id": run_id,
                        "confidence": confidence,
                        "reference_used": ref is not None
                    })
                    self.metrics.set_confidence(self.tenant_id, attractor_name, confidence)

                    if attractor_name not in self.confidence_ema:
                        self.confidence_ema[attractor_name] = confidence
                    else:
                        self.confidence_ema[attractor_name] = (
                            AUTO_TUNE_EMA_ALPHA * confidence +
                            (1 - AUTO_TUNE_EMA_ALPHA) * self.confidence_ema[attractor_name]
                        )

                    self.confidence_evaluator.record_score(attractor_name, confidence)
                    self.confidence_evaluator.record_score("__global__", confidence)
                    hist = self.confidence_history.setdefault(attractor_name, [])
                    hist.append(confidence)
                    if len(hist) > 10:
                        hist.pop(0)
                    avg_conf = sum(hist) / len(hist) if hist else 0.5

                    trend = self.confidence_evaluator.get_trend(window=5, key=attractor_name)
                    mutation_threshold = 1 / PHI

                    if avg_conf < mutation_threshold or (trend < 0 and avg_conf < 0.7):
                        self.mutation_engine.propose_mutation(attractor, avg_conf)
                        self.logger.log_event("mutation_triggered", {
                            "run_id": run_id,
                            "attractor": attractor_name,
                            "avg_confidence": avg_conf,
                            "trend": trend,
                            "threshold": mutation_threshold
                        })
                else:
                    avg_conf = 0.5
                    trend = 0.0

            run_cost = self.policy.total_cost - cost_before

            if self.mode == KernelMode.GROWTH and self.growth_supervisor is not None:
                with self._lock:
                    self.historical_inputs.append({
                        "prompt": user_prompt,
                        "attractor": attractor_name,
                        "intent": intent,
                        "ground_truth": ground_truth
                    })
                    if len(self.historical_inputs) > self.max_historical_inputs:
                        self.historical_inputs = self.historical_inputs[-self.max_historical_inputs:]
                mutation_applied = False
                self.growth_supervisor.record_run(
                    confidence if confidence >= 0 else 0.5,
                    run_cost,
                    mutation_applied
                )
                if self.growth_supervisor.mode == KernelMode.PRODUCTION:
                    self.mode = KernelMode.PRODUCTION
                    self.logger.log_event("entering_production_mode")

            self.logger.log_event("attractor_complete", {
                "run_id": run_id,
                "attractor": attractor_name,
                "final_output_length": len(pipeline_input),
                "average_confidence": avg_conf,
                "confidence_trend": trend,
                "run_cost": run_cost
            })
            self.policy.clean_run(run_id)
            return pipeline_input, run_cost

        except GenerationFailedError as e:
            run_cost = self.policy.total_cost - cost_before
            error_msg = f"Generation failed: {str(e)}"
            self.logger.log_event("attractor_error", {"run_id": run_id, "error": str(e), "cost": run_cost})
            self.policy.clean_run(run_id)
            return error_msg, run_cost
