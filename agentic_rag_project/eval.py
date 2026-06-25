"""LLMOps, tracing, and RAG evaluation utilities."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _overlap_score(left: str, right: str) -> float:
    left_terms = set(_tokens(left))
    right_terms = set(_tokens(right))
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


@dataclass
class TraceStep:
    """One observable agent execution step."""

    name: str
    kind: str
    started_at: float
    ended_at: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    """One full user-query trace."""

    trace_id: str
    query: str
    started_at: float
    ended_at: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    steps: List[TraceStep] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


class TracingEngine:
    """JSONL tracing for latency, token usage, cost, and agent steps."""

    def __init__(self, log_dir: str = "data/traces"):
        self.log_dir = log_dir
        self.current_trace: Optional[TraceRecord] = None
        self.traces: List[TraceRecord] = []
        os.makedirs(self.log_dir, exist_ok=True)

    def start_trace(self, query: str) -> TraceRecord:
        trace_id = f"trace-{int(time.time() * 1000)}"
        self.current_trace = TraceRecord(trace_id=trace_id, query=query, started_at=time.time())
        return self.current_trace

    def start_step(self, name: str, kind: str, metadata: Optional[Dict[str, Any]] = None) -> TraceStep:
        if self.current_trace is None:
            self.start_trace("")
        step = TraceStep(
            name=name,
            kind=kind,
            started_at=time.time(),
            metadata=metadata or {},
        )
        self.current_trace.steps.append(step)
        return step

    def end_step(
        self,
        step: TraceStep,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceStep:
        step.ended_at = time.time()
        step.latency_ms = max((step.ended_at - step.started_at) * 1000, 0.001)
        step.input_tokens = input_tokens
        step.output_tokens = output_tokens
        step.cost_usd = cost_usd
        if metadata:
            step.metadata.update(metadata)

        if self.current_trace is not None:
            self.current_trace.total_input_tokens += input_tokens
            self.current_trace.total_output_tokens += output_tokens
            self.current_trace.total_cost_usd += cost_usd
        return step

    def end_trace(self, scores: Optional[Dict[str, float]] = None) -> TraceRecord:
        if self.current_trace is None:
            raise RuntimeError("No active trace to end")
        trace = self.current_trace
        trace.ended_at = time.time()
        trace.total_latency_ms = max((trace.ended_at - trace.started_at) * 1000, 0.001)
        trace.scores = scores or trace.scores
        self.traces.append(trace)
        self._write_trace(trace)
        self.current_trace = None
        return trace

    def get_summary(self) -> Dict[str, Any]:
        total_queries = len(self.traces)
        total_latency = sum(trace.total_latency_ms for trace in self.traces)
        return {
            "total_queries": total_queries,
            "average_latency_ms": total_latency / total_queries if total_queries else 0.0,
            "total_input_tokens": sum(trace.total_input_tokens for trace in self.traces),
            "total_output_tokens": sum(trace.total_output_tokens for trace in self.traces),
            "total_cost_usd": sum(trace.total_cost_usd for trace in self.traces),
        }

    def _write_trace(self, trace: TraceRecord) -> None:
        path = os.path.join(self.log_dir, "traces.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), default=str) + "\n")


class RAGTriadEvaluator:
    """Context relevance, groundedness, and answer relevance evaluator."""

    def __init__(self, llm: Any = None):
        self.llm = llm

    def evaluate(self, query: str, context: str, generated_answer: str) -> Dict[str, float]:
        context_relevance = _overlap_score(query, context)
        groundedness = self._groundedness(context, generated_answer)
        answer_relevance = _overlap_score(query, generated_answer)
        overall = (context_relevance + groundedness + answer_relevance) / 3
        return {
            "context_relevance": round(context_relevance, 4),
            "groundedness": round(groundedness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "overall": round(overall, 4),
        }

    def _groundedness(self, context: str, answer: str) -> float:
        answer_terms = set(_tokens(answer))
        context_terms = set(_tokens(context))
        if not answer_terms:
            return 0.0
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "is",
            "are",
            "to",
            "of",
            "in",
            "for",
            "with",
            "that",
            "this",
        }
        meaningful = {term for term in answer_terms if term not in stop}
        if not meaningful:
            meaningful = answer_terms
        return len(meaningful & context_terms) / len(meaningful)


class HallucinationDetector:
    """Detect claims weakly grounded in retrieved context."""

    def __init__(self, llm: Any = None, threshold: float = 0.45):
        self.llm = llm
        self.threshold = threshold

    def detect(self, context: str, generated_answer: str) -> Dict[str, Any]:
        grounding_score = RAGTriadEvaluator(self.llm)._groundedness(context, generated_answer)
        unsupported_claims = []
        context_terms = set(_tokens(context))
        for sentence in re.split(r"(?<=[.!?])\s+", generated_answer):
            terms = {term for term in _tokens(sentence) if len(term) > 3}
            if terms and len(terms & context_terms) / len(terms) < self.threshold:
                unsupported_claims.append(sentence)
        return {
            "has_potential_hallucination": grounding_score < self.threshold,
            "grounding_score": round(grounding_score, 4),
            "unsupported_claims": unsupported_claims,
        }


class GoldenDataset:
    """CI-friendly golden dataset benchmark runner."""

    DEFAULT_ENTRIES = [
        {
            "query": "What is Retrieval Augmented Generation?",
            "context": "Retrieval Augmented Generation, or RAG, retrieves relevant context before generation.",
            "expected_answer": "RAG retrieves relevant context before generating an answer.",
        },
        {
            "query": "Why use hybrid search?",
            "context": "Hybrid search combines keyword retrieval such as BM25 with semantic vector search.",
            "expected_answer": "Hybrid search combines keyword and vector search for better retrieval.",
        },
        {
            "query": "What does groundedness measure?",
            "context": "Groundedness measures whether the answer is supported by retrieved context.",
            "expected_answer": "Groundedness checks that the answer is supported by the context.",
        },
    ]

    def __init__(self, dataset_path: str = "data/golden_dataset.json"):
        self.dataset_path = dataset_path
        self.entries: List[Dict[str, str]] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as handle:
                self.entries = json.load(handle)
            return
        self.entries = list(self.DEFAULT_ENTRIES)
        with open(self.dataset_path, "w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, indent=2)

    def run_benchmark(self, evaluator: RAGTriadEvaluator) -> Dict[str, Any]:
        results = []
        totals: Dict[str, float] = defaultdict_float()
        for entry in self.entries:
            scores = evaluator.evaluate(
                entry["query"],
                entry.get("context", ""),
                entry.get("expected_answer", ""),
            )
            results.append({**entry, "scores": scores})
            for key, value in scores.items():
                totals[key] += value
        count = len(results) or 1
        averages = {key: round(value / count, 4) for key, value in totals.items()}
        return {
            "dataset_path": self.dataset_path,
            "entry_count": len(results),
            "average_scores": averages,
            "results": results,
        }


def defaultdict_float() -> Dict[str, float]:
    return {"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0, "overall": 0.0}


def evaluate_rag_triad(query: str, context: str, generated_answer: str) -> Dict[str, float]:
    """Compatibility wrapper for the original scaffold API."""

    return RAGTriadEvaluator().evaluate(query, context, generated_answer)


def inject_guardrails(prompt: str) -> str:
    """Compatibility wrapper: sanitize prompt injection attempts and mask PII."""

    try:
        from security import create_security_stack

        processed, _ = create_security_stack().process_input(prompt)
        return processed
    except Exception:
        return prompt.replace("SSN", "[MASKED]")
