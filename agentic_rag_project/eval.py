"""
LLMOps, Observability & Advanced Evaluation Module

Implements:
  - LLM-as-a-Judge (automated accuracy scoring)
  - Observability / Tracing (latency, token usage, cost, execution steps)
  - Golden Dataset (benchmark test sets)
  - Latency tracking
  - Hallucination detection
  - Guardrails evaluation
  
  Advanced Evaluation Frameworks (RAG Triad):
  - Context Relevance
  - Groundedness / Faithfulness
  - Answer Relevance
  - Frameworks: Ragas / TruLens compatible
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Observability / Tracing
# ─────────────────────────────────────────────

@dataclass
class TraceStep:
    """A single step in the agent execution trace."""
    step_name: str
    step_type: str                      # retrieve, grade, generate, route, etc.
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: Dict = field(default_factory=dict)
    status: str = "pending"             # pending, running, success, error
    error_message: str = ""


@dataclass
class ExecutionTrace:
    """
    Observability / Tracing:
    Full execution trace tracking latency, token usage, cost,
    and agent execution steps.
    """
    trace_id: str = ""
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    steps: List[TraceStep] = field(default_factory=list)
    evaluation_scores: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = hashlib.md5(
                f"{time.time()}:{self.query}".encode()
            ).hexdigest()[:12]


class TracingEngine:
    """
    Observability / Tracing Engine:
    Tracks latency, token usage, cost, and agent execution steps.
    Compatible with LangSmith / Arize AI patterns.
    """
    
    # Cost per 1K tokens (approximate, per model)
    TOKEN_COSTS = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "ollama": {"input": 0.0, "output": 0.0},  # Local = free
    }
    
    def __init__(self, log_dir: str = "data/traces", model_name: str = "ollama"):
        self.log_dir = log_dir
        self.model_name = model_name
        self.traces: List[ExecutionTrace] = []
        self.current_trace: Optional[ExecutionTrace] = None
        os.makedirs(log_dir, exist_ok=True)
    
    def start_trace(self, query: str) -> ExecutionTrace:
        """Start a new execution trace."""
        trace = ExecutionTrace(query=query, start_time=time.time())
        self.current_trace = trace
        logger.info(f"Trace [{trace.trace_id}] started for query: {query[:50]}...")
        return trace
    
    def start_step(self, step_name: str, step_type: str) -> TraceStep:
        """Start a new step within the current trace."""
        step = TraceStep(
            step_name=step_name,
            step_type=step_type,
            start_time=time.time(),
            status="running",
        )
        if self.current_trace:
            self.current_trace.steps.append(step)
        return step
    
    def end_step(self, step: TraceStep, input_tokens: int = 0,
                  output_tokens: int = 0, metadata: Dict = None):
        """Complete a step and record metrics."""
        step.end_time = time.time()
        step.latency_ms = (step.end_time - step.start_time) * 1000
        step.input_tokens = input_tokens
        step.output_tokens = output_tokens
        step.status = "success"
        step.metadata = metadata or {}
        
        # Calculate cost
        cost_info = self.TOKEN_COSTS.get(self.model_name, self.TOKEN_COSTS["ollama"])
        step.cost_usd = (
            (input_tokens / 1000) * cost_info["input"] +
            (output_tokens / 1000) * cost_info["output"]
        )
        
        logger.info(
            f"Step '{step.step_name}' completed: "
            f"{step.latency_ms:.1f}ms, {input_tokens}+{output_tokens} tokens, "
            f"${step.cost_usd:.6f}"
        )
    
    def end_step_with_error(self, step: TraceStep, error: str):
        """Complete a step with an error."""
        step.end_time = time.time()
        step.latency_ms = (step.end_time - step.start_time) * 1000
        step.status = "error"
        step.error_message = error
        logger.error(f"Step '{step.step_name}' failed: {error}")
    
    def end_trace(self, evaluation_scores: Dict = None) -> ExecutionTrace:
        """Complete the current trace."""
        if not self.current_trace:
            return ExecutionTrace()
        
        trace = self.current_trace
        trace.end_time = time.time()
        trace.total_latency_ms = (trace.end_time - trace.start_time) * 1000
        trace.total_input_tokens = sum(s.input_tokens for s in trace.steps)
        trace.total_output_tokens = sum(s.output_tokens for s in trace.steps)
        trace.total_cost_usd = sum(s.cost_usd for s in trace.steps)
        
        if evaluation_scores:
            trace.evaluation_scores = evaluation_scores
        
        self.traces.append(trace)
        self._save_trace(trace)
        
        logger.info(
            f"Trace [{trace.trace_id}] completed: "
            f"latency={trace.total_latency_ms:.1f}ms, "
            f"tokens={trace.total_input_tokens}+{trace.total_output_tokens}, "
            f"cost=${trace.total_cost_usd:.6f}"
        )
        
        self.current_trace = None
        return trace
    
    def _save_trace(self, trace: ExecutionTrace):
        """Save trace to disk for later analysis."""
        trace_data = {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "total_latency_ms": trace.total_latency_ms,
            "total_input_tokens": trace.total_input_tokens,
            "total_output_tokens": trace.total_output_tokens,
            "total_cost_usd": trace.total_cost_usd,
            "evaluation_scores": trace.evaluation_scores,
            "steps": [
                {
                    "step_name": s.step_name,
                    "step_type": s.step_type,
                    "latency_ms": s.latency_ms,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": s.cost_usd,
                    "status": s.status,
                    "error_message": s.error_message,
                }
                for s in trace.steps
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        trace_file = os.path.join(self.log_dir, f"trace_{trace.trace_id}.json")
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all traces."""
        if not self.traces:
            return {"total_queries": 0}
        
        return {
            "total_queries": len(self.traces),
            "avg_latency_ms": sum(t.total_latency_ms for t in self.traces) / len(self.traces),
            "total_tokens": sum(t.total_input_tokens + t.total_output_tokens for t in self.traces),
            "total_cost_usd": sum(t.total_cost_usd for t in self.traces),
            "avg_steps_per_query": sum(len(t.steps) for t in self.traces) / len(self.traces),
            "error_rate": sum(
                1 for t in self.traces if any(s.status == "error" for s in t.steps)
            ) / len(self.traces),
        }


# ─────────────────────────────────────────────
# RAG Triad Evaluation (Context Relevance, Groundedness, Answer Relevance)
# ─────────────────────────────────────────────

class RAGTriadEvaluator:
    """
    Advanced Evaluation Frameworks (RAG Triad):
    
    1. Context Relevance: Do the retrieved docs contain the answer?
    2. Groundedness / Faithfulness: Is the response strictly from context?
    3. Answer Relevance: Does the response answer the original question?
    
    Uses LLM-as-a-Judge pattern or keyword-based fallback.
    Compatible with Ragas / TruLens frameworks.
    """
    
    def __init__(self, llm=None):
        self.llm = llm  # Optional LLM for LLM-as-a-Judge
    
    def evaluate(self, query: str, context: str,
                  generated_answer: str) -> Dict[str, float]:
        """
        Run the full RAG Triad evaluation.
        Returns scores for context_relevance, groundedness, and answer_relevance.
        """
        scores = {}
        
        # 1. Context Relevance
        scores["context_relevance"] = self._evaluate_context_relevance(query, context)
        
        # 2. Groundedness / Faithfulness
        scores["groundedness"] = self._evaluate_groundedness(context, generated_answer)
        
        # 3. Answer Relevance
        scores["answer_relevance"] = self._evaluate_answer_relevance(query, generated_answer)
        
        # Overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _evaluate_context_relevance(self, query: str, context: str) -> float:
        """
        Context Relevance:
        Measures whether the retrieved documents actually contain
        the answers needed to satisfy the query.
        """
        if self.llm:
            return self._llm_judge_score(
                f"On a scale of 0-1, how relevant is the following context to the query?\n"
                f"Query: {query}\n"
                f"Context: {context[:2000]}\n"
                f"Score (0-1):"
            )
        
        # Keyword overlap fallback
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & context_words) / len(query_words)
        return min(overlap, 1.0)
    
    def _evaluate_groundedness(self, context: str, answer: str) -> float:
        """
        Groundedness / Faithfulness:
        Evaluates if the LLM's response is strictly derived only from
        the retrieved context (zero hallucination).
        """
        if self.llm:
            return self._llm_judge_score(
                f"On a scale of 0-1, is the following answer entirely grounded in the context?\n"
                f"Context: {context[:2000]}\n"
                f"Answer: {answer[:1000]}\n"
                f"Score (0-1, where 1 means fully grounded with no hallucination):"
            )
        
        # Check how many answer words appear in context
        answer_words = set(answer.lower().split()) - {"the", "a", "an", "is", "are", "was", "were",
                                                        "in", "on", "at", "to", "for", "of", "and",
                                                        "or", "but", "with", "this", "that", "it"}
        context_words = set(context.lower().split())
        if not answer_words:
            return 1.0
        grounded = len(answer_words & context_words) / len(answer_words)
        return min(grounded, 1.0)
    
    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance:
        Checks if the generated response actually answers the
        original question asked by the user.
        """
        if self.llm:
            return self._llm_judge_score(
                f"On a scale of 0-1, does the following answer properly address the query?\n"
                f"Query: {query}\n"
                f"Answer: {answer[:1000]}\n"
                f"Score (0-1):"
            )
        
        # Simple heuristic: check query keyword presence in answer
        query_words = set(query.lower().split()) - {"what", "how", "why", "when", "where",
                                                      "who", "which", "is", "are", "do", "does",
                                                      "can", "the", "a", "an"}
        answer_words = set(answer.lower().split())
        if not query_words:
            return 0.5
        relevance = len(query_words & answer_words) / len(query_words)
        return min(relevance, 1.0)
    
    def _llm_judge_score(self, prompt: str) -> float:
        """
        LLM-as-a-Judge:
        Uses a capable model to evaluate and score outputs.
        """
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract numerical score from response
            import re
            numbers = re.findall(r'(\d+\.?\d*)', content)
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"LLM-as-a-Judge error: {e}")
            return 0.5


# ─────────────────────────────────────────────
# Hallucination Detection
# ─────────────────────────────────────────────

class HallucinationDetector:
    """
    Hallucination:
    Detects instances where the AI generates factually incorrect
    or fabricated information not present in the source context.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def detect(self, context: str, answer: str) -> Dict:
        """
        Detect potential hallucinations in the answer.
        Returns a report with detection results.
        """
        report = {
            "has_potential_hallucination": False,
            "confidence": 0.0,
            "flagged_segments": [],
            "grounding_score": 0.0,
        }
        
        # Split answer into sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        context_lower = context.lower()
        
        ungrounded = []
        for sentence in sentences:
            # Check if key words from the sentence appear in context
            words = set(sentence.lower().split()) - {
                "the", "a", "an", "is", "are", "was", "were", "in", "on",
                "at", "to", "for", "of", "and", "or", "but", "with",
                "this", "that", "it", "can", "will", "would", "should",
            }
            if not words:
                continue
            
            found = sum(1 for w in words if w in context_lower)
            ratio = found / len(words)
            
            if ratio < 0.3:
                ungrounded.append({
                    "sentence": sentence,
                    "grounding_ratio": ratio,
                })
        
        if ungrounded:
            report["has_potential_hallucination"] = True
            report["flagged_segments"] = ungrounded
            report["confidence"] = len(ungrounded) / (len(sentences) or 1)
        
        grounded_count = len(sentences) - len(ungrounded)
        report["grounding_score"] = grounded_count / (len(sentences) or 1)
        
        return report


# ─────────────────────────────────────────────
# Golden Dataset
# ─────────────────────────────────────────────

class GoldenDataset:
    """
    Golden Dataset:
    A verified test set of queries and expected answers used to
    benchmark the system's performance.
    """
    
    def __init__(self, dataset_path: str = "data/golden_dataset.json"):
        self.dataset_path = dataset_path
        self.entries: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load golden dataset from disk."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            logger.info(f"Golden Dataset: Loaded {len(self.entries)} entries")
        else:
            self.entries = self._create_default()
            self.save()
    
    def _create_default(self) -> List[Dict]:
        """Create a default golden dataset with sample entries."""
        return [
            {
                "id": "gd_001",
                "query": "What is Retrieval Augmented Generation?",
                "expected_answer": "RAG is a technique that combines information retrieval with text generation, allowing LLMs to access external knowledge bases.",
                "expected_context_keywords": ["retrieval", "augmented", "generation", "LLM", "knowledge"],
                "difficulty": "easy",
                "category": "definitions",
            },
            {
                "id": "gd_002",
                "query": "Explain the difference between semantic and keyword search.",
                "expected_answer": "Semantic search understands the meaning and intent behind queries using embeddings, while keyword search matches exact terms using algorithms like BM25.",
                "expected_context_keywords": ["semantic", "keyword", "embeddings", "BM25", "meaning"],
                "difficulty": "medium",
                "category": "retrieval",
            },
            {
                "id": "gd_003",
                "query": "What are the components of the RAG Triad?",
                "expected_answer": "The RAG Triad consists of Context Relevance, Groundedness (Faithfulness), and Answer Relevance.",
                "expected_context_keywords": ["context", "relevance", "groundedness", "faithfulness", "answer"],
                "difficulty": "medium",
                "category": "evaluation",
            },
        ]
    
    def save(self):
        """Save golden dataset to disk."""
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(self, query: str, expected_answer: str,
                   keywords: List[str] = None, category: str = "general"):
        """Add a new entry to the golden dataset."""
        entry = {
            "id": f"gd_{len(self.entries) + 1:03d}",
            "query": query,
            "expected_answer": expected_answer,
            "expected_context_keywords": keywords or [],
            "difficulty": "custom",
            "category": category,
        }
        self.entries.append(entry)
        self.save()
    
    def run_benchmark(self, evaluator: RAGTriadEvaluator,
                       retrieve_fn=None, generate_fn=None) -> Dict:
        """
        Run benchmark against the golden dataset.
        Returns aggregate scores.
        """
        results = []
        
        for entry in self.entries:
            query = entry["query"]
            expected = entry["expected_answer"]
            
            # Use provided functions or simulate
            if retrieve_fn:
                context = retrieve_fn(query)
            else:
                context = expected  # Use expected answer as context for simulation
            
            if generate_fn:
                generated = generate_fn(query, context)
            else:
                generated = expected  # Simulate perfect generation
            
            scores = evaluator.evaluate(query, context, generated)
            results.append({
                "entry_id": entry["id"],
                "query": query,
                "scores": scores,
            })
        
        # Aggregate
        if results:
            avg_scores = {
                key: sum(r["scores"][key] for r in results) / len(results)
                for key in results[0]["scores"]
            }
        else:
            avg_scores = {}
        
        return {
            "total_entries": len(results),
            "average_scores": avg_scores,
            "per_entry_results": results,
        }


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_evaluation_stack(config=None, llm=None):
    """Create the full evaluation and observability stack."""
    from config import RAGConfig
    
    cfg = config or RAGConfig()
    
    tracing_engine = TracingEngine(
        log_dir=cfg.observability.trace_log_dir,
        model_name=cfg.llm.model_name,
    )
    
    rag_evaluator = RAGTriadEvaluator(llm=llm)
    hallucination_detector = HallucinationDetector(llm=llm)
    golden_dataset = GoldenDataset(dataset_path=cfg.observability.golden_dataset_path)
    
    return {
        "tracing_engine": tracing_engine,
        "rag_evaluator": rag_evaluator,
        "hallucination_detector": hallucination_detector,
        "golden_dataset": golden_dataset,
    }
