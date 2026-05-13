import concurrent.futures
import time
import random
import threading
from typing import List, Dict, Any, Optional, Tuple
from flux.utils.token_counter import TokenCounter
from flux.kernel.logger import FluxLogger
from flux.kernel.policy import FluxPolicy
from flux.kernel.loader import FluxModelLoader
from flux.kernel.exceptions import BudgetExceededError

class RecursiveConfig:
    def __init__(
        self,
        max_chunk_tokens: int = 40000,
        chunk_overlap: int = 200,
        sub_model_key: str = "fast",
        root_model_key: str = "accurate",
        max_workers: int = 4
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.sub_model_key = sub_model_key
        self.root_model_key = root_model_key
        self.max_workers = max_workers

class RecursiveEngine:
    def __init__(
        self,
        loader: FluxModelLoader,
        logger: FluxLogger,
        policy: FluxPolicy,
        token_counter: TokenCounter,
        config: Optional[RecursiveConfig] = None
    ):
        self.loader = loader
        self.logger = logger
        self.policy = policy
        self.token_counter = token_counter
        self.config = config or RecursiveConfig()

    def _split_prompt_into_chunks(self, prompt: str, max_chunk_tokens: Optional[int] = None) -> List[str]:
        limit = max_chunk_tokens if max_chunk_tokens is not None else self.config.max_chunk_tokens
        tokens_all, token_list = self.token_counter.count(prompt)
        if tokens_all <= limit:
            return [prompt]
        paragraphs = prompt.split('\n\n')
        chunks = []
        current_tokens = 0
        current_text = ""

        for para in paragraphs:
            para_tokens, para_token_list = self.token_counter.count(para)
            if para_tokens > limit:
                if current_text:
                    chunks.append(current_text)
                chunks.append(para)
                current_text = ""
                current_tokens = 0
                continue

            if current_tokens + para_tokens > limit:
                if current_text:
                    chunks.append(current_text)
                overlap_text = ""
                if chunks and self.config.chunk_overlap > 0:
                    last_chunk = chunks[-1]
                    last_tokens, last_token_list = self.token_counter.count(last_chunk)
                    if last_token_list and last_tokens >= self.config.chunk_overlap:
                        overlap_tokens = last_token_list[-self.config.chunk_overlap:]
                        overlap_text = self.token_counter.decode(overlap_tokens) + "\n\n"
                    else:
                        overlap_text = last_chunk[-self.config.chunk_overlap * 4:] + "\n\n"
                current_text = overlap_text + para
                current_tokens, _ = self.token_counter.count(current_text)
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                current_tokens += para_tokens

        if current_text:
            chunks.append(current_text)
        return chunks

    def _truncate_chunk_to_fit(self, chunk: str, sub_query: str, model_key: str) -> str:
        model_info = self.loader.get_model_info(model_key)
        max_ctx = model_info.get("max_context", 8192)
        sub_tokens, _ = self.token_counter.count(sub_query)
        available = max_ctx - sub_tokens - 100
        if available <= 0:
            return ""
        chunk_tokens, token_list = self.token_counter.count(chunk)
        if chunk_tokens <= available:
            return chunk
        truncated_tokens = token_list[:available]
        return self.token_counter.decode(truncated_tokens)

    def _sub_call(self, chunk: str, sub_query: str, run_id: str, idx: int,
                  temperature: float = 0.3) -> Dict[str, Any]:
        start_time = time.time()
        max_retries = 2

        truncated_chunk = self._truncate_chunk_to_fit(chunk, sub_query, self.config.sub_model_key)
        if not truncated_chunk:
            self.logger.log_event("recursive_chunk_truncated_to_empty", {
                "run_id": run_id, "chunk_idx": idx
            })
            return {"error": "Chunk too large for sub model context", "chunk_idx": idx, "cost": 0.0}

        for attempt in range(max_retries + 1):
            try:
                prompt = f"{sub_query}\n\nCONTEXT:\n{truncated_chunk}"
                response_data = self.loader.generate(
                    self.config.sub_model_key, prompt,
                    max_new_tokens=512, temperature=temperature
                )
                response = response_data["text"]
                usage = response_data.get("usage", {})
                real_cost = self.policy.estimate_cost(usage.get("prompt_tokens", 0),
                                                      usage.get("completion_tokens", 0))
                if not self.policy.try_commit_cost(real_cost, run_id):
                    return {"error": "Budget exceeded", "chunk_idx": idx, "cost": 0.0}

                elapsed = time.time() - start_time
                if not self.policy.check_latency(elapsed * 1000, run_id):
                    return {"error": "Latency exceeded", "chunk_idx": idx}

                self.logger.log_event("recursive_subcall", {
                    "run_id": run_id,
                    "chunk_idx": idx,
                    "latency_ms": elapsed * 1000,
                    "model_key": self.config.sub_model_key,
                    "chunk_length": len(truncated_chunk),
                    "attempt": attempt
                })
                return {"chunk_idx": idx, "response": response, "latency": elapsed, "cost": real_cost}
            except Exception as e:
                if attempt < max_retries:
                    delay = 0.5 * (2 ** attempt) + random.random()
                    time.sleep(delay)
                    continue
                else:
                    self.logger.log_event("recursive_error", {"run_id": run_id, "chunk_idx": idx, "error": str(e), "attempt": attempt})
                    return {"error": str(e), "chunk_idx": idx, "cost": 0.0}
        return {"error": "Max retries exceeded", "chunk_idx": idx, "cost": 0.0}

    def _aggregate_large_results(self, concatenated: str, query: str, run_id: str,
                                 root_temperature: float) -> Tuple[str, float]:
        model_info = self.loader.get_model_info(self.config.root_model_key)
        max_ctx = model_info.get("max_context", 8192)
        query_tokens, _ = self.token_counter.count(query)
        available = max_ctx - query_tokens - 150
        if available <= 0:
            return "", 0.0
        concat_tokens, _ = self.token_counter.count(concatenated)
        if concat_tokens <= available:
            return concatenated, 0.0

        safe_chunk_tokens = available // 2
        paragraphs = concatenated.split("\n\n---\n\n")
        chunks = []
        current_text = ""
        current_tokens = 0
        for para in paragraphs:
            para_tokens, _ = self.token_counter.count(para)
            if current_tokens + para_tokens > safe_chunk_tokens:
                if current_text:
                    chunks.append(current_text)
                current_text = para
                current_tokens = para_tokens
            else:
                if current_text:
                    current_text += "\n\n---\n\n" + para
                else:
                    current_text = para
                current_tokens += para_tokens
        if current_text:
            chunks.append(current_text)

        sub_summaries = []
        total_cost = 0.0
        for idx, chunk in enumerate(chunks):
            prompt = (
                f"From the following extracted information, list the key facts relevant to this query: {query}\n\n"
                f"{chunk}\n\nKey facts (concise):"
            )
            try:
                resp_data = self.loader.generate(
                    self.config.root_model_key, prompt,
                    max_new_tokens=256, temperature=root_temperature * 0.8
                )
                text = resp_data["text"]
                usage = resp_data.get("usage", {})
                cost = self.policy.estimate_cost(usage.get("prompt_tokens", 0),
                                                 usage.get("completion_tokens", 0))
                if self.policy.try_commit_cost(cost, run_id):
                    total_cost += cost
                else:
                    cost = 0.0  # budget exceeded, but we still use the chunk
                sub_summaries.append(text)
            except Exception as e:
                self.logger.log_event("recursive_aggregation_error", {
                    "run_id": run_id,
                    "chunk_idx": idx,
                    "error": str(e)
                })
                sub_summaries.append(chunk)
        return "\n\n---\n\n".join(sub_summaries), total_cost

    def execute_recursive_pipeline(self, prompt: str, query: str, run_id: str,
                                   sub_temperature: float = 0.3,
                                   root_temperature: float = 0.7) -> Tuple[str, float]:
        total_cost = 0.0
        chunks = self._split_prompt_into_chunks(prompt)
        self.logger.log_event("recursive_start", {
            "run_id": run_id,
            "total_chunks": len(chunks),
            "prompt_tokens": self.token_counter.count(prompt)[0]
        })

        if len(chunks) == 1:
            model_info = self.loader.get_model_info(self.config.root_model_key)
            max_ctx = model_info.get("max_context", 8192)
            full_prompt = query + "\n\n" + prompt
            full_tokens, _ = self.token_counter.count(full_prompt)
            if full_tokens > max_ctx:
                self.logger.log_event("recursive_single_chunk_overflow", {
                    "run_id": run_id,
                    "tokens": full_tokens,
                    "max_context": max_ctx
                })
                local_config = RecursiveConfig(
                    max_chunk_tokens=max_ctx - self.token_counter.count(query)[0] - 100,
                    chunk_overlap=self.config.chunk_overlap,
                    sub_model_key=self.config.sub_model_key,
                    root_model_key=self.config.root_model_key,
                    max_workers=self.config.max_workers
                )
                chunks = self._split_prompt_into_chunks(prompt, max_chunk_tokens=local_config.max_chunk_tokens)
            else:
                response_data = self.loader.generate(
                    self.config.root_model_key, full_prompt,
                    max_new_tokens=1024, temperature=root_temperature
                )
                usage = response_data.get("usage", {})
                cost = self.policy.estimate_cost(usage.get("prompt_tokens", 0),
                                                 usage.get("completion_tokens", 0))
                if not self.policy.try_commit_cost(cost, run_id):
                    raise BudgetExceededError("Budget exceeded during single-chunk generation")
                total_cost += cost
                return response_data["text"], total_cost

        sub_query = f"Extract key information from the text to answer: {query}"
        sub_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._sub_call, chunk, sub_query, run_id, idx, sub_temperature): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                result = future.result()
                if "error" not in result:
                    sub_results.append(result)
                    total_cost += result.get("cost", 0.0)
        sub_results.sort(key=lambda x: x["chunk_idx"])
        concatenated = "\n\n---\n\n".join([r["response"] for r in sub_results])

        agg_text = concatenated
        agg_cost = 0.0
        while True:
            agg_prompt = (
                f"Synthesize the following extracted information to answer the query precisely.\n\n"
                f"QUERY: {query}\n\nEXTRACTED INFO:\n{agg_text}\n\n"
                "Final answer:"
            )
            tokens_agg, _ = self.token_counter.count(agg_prompt)
            model_info = self.loader.get_model_info(self.config.root_model_key)
            max_ctx = model_info.get("max_context", 8192)
            if tokens_agg <= max_ctx - 50:
                break
            new_agg, new_cost = self._aggregate_large_results(agg_text, query, run_id, root_temperature)
            agg_cost += new_cost
            if new_agg == agg_text or len(new_agg) <= len(agg_text):
                agg_text = agg_text[: (max_ctx - 50) * 4]
                agg_prompt = (
                    f"Synthesize the following extracted information to answer the query precisely.\n\n"
                    f"QUERY: {query}\n\nEXTRACTED INFO:\n{agg_text}\n\n"
                    "Final answer:"
                )
                break
            agg_text = new_agg

        response_data = self.loader.generate(
            self.config.root_model_key, agg_prompt,
            max_new_tokens=1024, temperature=root_temperature
        )
        usage = response_data.get("usage", {})
        cost = self.policy.estimate_cost(usage.get("prompt_tokens", 0),
                                         usage.get("completion_tokens", 0))
        if not self.policy.try_commit_cost(cost, run_id):
            raise BudgetExceededError("Budget exceeded during final aggregation")
        total_cost += cost + agg_cost

        self.logger.log_event("recursive_aggregation", {
            "run_id": run_id,
            "sub_calls": len(sub_results),
            "total_cost": total_cost
        })
        return response_data["text"], total_cost
