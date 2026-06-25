"""Agent orchestration graph for enterprise Agentic RAG."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except Exception:  # pragma: no cover - fallback keeps import smoke tests portable.
    @dataclass
    class HumanMessage:  # type: ignore
        content: str

    @dataclass
    class AIMessage:  # type: ignore
        content: str

    @dataclass
    class SystemMessage:  # type: ignore
        content: str

from config import RAGConfig
from eval import RAGTriadEvaluator, TracingEngine
from retrieval import (
    ContextWindowManager,
    EmbeddingEngine,
    HybridSearchEngine,
    KnowledgeGraph,
    SemanticRouter,
)
from security import Guardrails, create_security_stack
from tools import ToolRegistry, create_tool_registry


SPECIALIST_AGENTS: Dict[str, Dict[str, str]] = {
    "retrieval_specialist": {
        "domain": "retrieval",
        "goal": "Plan hybrid, metadata-filtered, parent-document, and graph retrieval.",
    },
    "security_specialist": {
        "domain": "security",
        "goal": "Apply PII masking, prompt-injection defense, RBAC, and HITL policy.",
    },
    "analysis_specialist": {
        "domain": "analysis",
        "goal": "Judge context relevance and groundedness before generation.",
    },
    "tool_specialist": {
        "domain": "tooling",
        "goal": "Select safe tools and request human approval for critical actions.",
    },
}


@dataclass
class AgentState:
    """State carried across DAG and cyclic self-correction edges."""

    query: str
    messages: List[Any] = field(default_factory=list)
    sanitized_query: str = ""
    guardrail_report: Dict[str, Any] = field(default_factory=dict)
    route: str = "general"
    route_score: float = 0.0
    context: str = ""
    graph_context: str = ""
    retrieved_docs: List[Tuple[Any, float]] = field(default_factory=list)
    relevance_score: float = 0.0
    retries: int = 0
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    eval_scores: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class AgenticRAGGraph:
    """Production-shaped local graph for Agentic RAG orchestration."""

    def __init__(
        self,
        llm: Any = None,
        config: Optional[RAGConfig] = None,
        hybrid_search: Optional[HybridSearchEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        guardrails: Optional[Guardrails] = None,
        memory_manager: Any = None,
        tracing_engine: Optional[TracingEngine] = None,
        evaluator: Optional[RAGTriadEvaluator] = None,
        semantic_router: Optional[SemanticRouter] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        context_manager: Optional[ContextWindowManager] = None,
        hitl_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.config = config or RAGConfig()
        self.hybrid_search = hybrid_search
        self.guardrails = guardrails or create_security_stack(self.config)
        self.memory_manager = memory_manager
        self.tracing = tracing_engine or TracingEngine(self.config.observability.trace_dir)
        self.evaluator = evaluator or RAGTriadEvaluator(llm=None)
        self.context_manager = context_manager or ContextWindowManager(
            self.config.generation.context_window_tokens
        )
        self.semantic_router = semantic_router or SemanticRouter(EmbeddingEngine())
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.tool_registry = tool_registry or create_tool_registry(
            hybrid_search=self.hybrid_search,
            context_manager=self.context_manager,
        )
        self.hitl_callback = hitl_callback
        self.stream_callback = stream_callback
        self._init_routes()

    def _init_routes(self) -> None:
        self.semantic_router.add_route(
            "retrieval",
            "Questions that require document search, semantic search, hybrid search, or citations.",
            ["Find policy details", "Search indexed documents", "What does the document say?"],
        )
        self.semantic_router.add_route(
            "security",
            "Questions about RBAC, prompt injection, PII masking, or guardrails.",
            ["Is this secure?", "Mask PII", "Explain access control"],
        )
        self.semantic_router.add_route(
            "tooling",
            "Questions requiring a calculator, SQL, file listing, or external tool.",
            ["Calculate 22 * 19", "List documents", "Query the database"],
        )

    def run(self, query: str, messages: Optional[List[Any]] = None) -> AgentState:
        self.tracing.start_trace(query)
        state = AgentState(query=query, messages=messages or [HumanMessage(content=query)])
        try:
            self._node_route(state)
            while True:
                self._node_retrieve(state)
                self._node_grade(state)
                if state.relevance_score >= self.config.evaluation.context_relevance_threshold:
                    break
                if state.retries >= self.config.agent_orchestration.max_retries:
                    break
            self._node_tools(state)
            self._node_generate(state)
            self._node_evaluate(state)
            self._record_memory(state)
        finally:
            if self.tracing.current_trace is not None:
                self.tracing.end_trace(state.eval_scores)
        return state

    def stream(self, inputs: Dict[str, Any], thread_config: Optional[Dict[str, Any]] = None):
        """Compatibility stream API matching the original LangGraph CLI loop."""

        messages = list(inputs.get("messages", []))
        query = self._last_message_content(messages)
        self.tracing.start_trace(query)
        state = AgentState(query=query, messages=messages or [HumanMessage(content=query)])
        try:
            yield {"route": self._node_route(state)}
            while True:
                yield {"retrieve": self._node_retrieve(state)}
                yield {"grade": self._node_grade(state)}
                if state.relevance_score >= self.config.evaluation.context_relevance_threshold:
                    break
                if state.retries >= self.config.agent_orchestration.max_retries:
                    break
            yield {"tools": self._node_tools(state)}
            yield {"generate": self._node_generate(state)}
            yield {"evaluate": self._node_evaluate(state)}
            self._record_memory(state)
        finally:
            if self.tracing.current_trace is not None:
                self.tracing.end_trace(state.eval_scores)

    def _node_route(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("route", "multi_agent_routing")
        sanitized, report = self.guardrails.process_input(
            state.query, self.config.security.default_user_id
        )
        state.sanitized_query = sanitized
        state.guardrail_report = report
        state.route, state.route_score = self.semantic_router.route(sanitized)
        self.tracing.end_step(
            step,
            input_tokens=self._estimate_tokens(state.query),
            output_tokens=self._estimate_tokens(state.route),
            metadata={"route": state.route, "route_score": state.route_score},
        )
        return {
            "route": state.route,
            "route_score": state.route_score,
            "specialist": SPECIALIST_AGENTS.get(f"{state.route}_specialist", SPECIALIST_AGENTS["retrieval_specialist"]),
            "guardrail_report": report,
        }

    def _node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("retrieve", "hybrid_parent_graph_retrieval")
        metadata_filter = self.guardrails.rbac_manager.get_metadata_filter(
            self.config.security.default_user_id
        )

        if self.hybrid_search is None:
            results = []
            graph_context = ""
        else:
            results = self.hybrid_search.search(
                state.sanitized_query,
                top_k=self.config.retrieval.top_k,
                metadata_filter=metadata_filter,
            )
            for chunk, _ in results:
                self.knowledge_graph.add_chunk(chunk)
            graph_context = self.knowledge_graph.get_context_for_query(state.sanitized_query)

        state.retrieved_docs = results
        state.graph_context = graph_context
        state.context = self.context_manager.compress_context(
            results,
            state.sanitized_query,
            graph_context=graph_context,
        )
        self.tracing.end_step(
            step,
            input_tokens=self._estimate_tokens(state.sanitized_query),
            output_tokens=self._estimate_tokens(state.context),
            metadata={"retrieved": len(results), "metadata_filter": metadata_filter},
        )
        return {
            "context": state.context,
            "retrieved_docs": results,
            "graph_context": state.graph_context,
            "retries": state.retries,
        }

    def _node_grade(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("grade", "self_reflection_llm_as_judge")
        state.relevance_score = self._heuristic_grade(state.sanitized_query, state.context)
        state.retries += 1
        self.tracing.end_step(
            step,
            input_tokens=self._estimate_tokens(state.sanitized_query + " " + state.context),
            output_tokens=1,
            metadata={"relevance_score": state.relevance_score, "retries": state.retries},
        )
        return {"relevance_score": state.relevance_score, "retries": state.retries}

    def _node_tools(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("tools", "react_tool_use_function_calling")
        state.tool_results = []

        # ReAct: reason about whether an action is needed, then act with tools.
        math_expr = self._extract_math_expression(state.sanitized_query)
        if math_expr:
            state.tool_results.append(
                self.tool_registry.execute_tool("calculator", expression=math_expr)
            )

        if state.route == "tooling" and re.search(r"\b(list|files|documents)\b", state.sanitized_query, re.I):
            state.tool_results.append(
                self.tool_registry.execute_tool("file_operations", action="list")
            )

        self.tracing.end_step(
            step,
            input_tokens=self._estimate_tokens(state.sanitized_query),
            output_tokens=self._estimate_tokens(str(state.tool_results)),
            metadata={"tool_count": len(state.tool_results)},
        )
        return {"tool_results": state.tool_results}

    def _node_generate(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("generate", "react_grounded_generation")
        system_prompt = self._build_system_prompt(state)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.sanitized_query),
        ]

        try:
            # Streaming Outputs: stream when the provider supports it, otherwise invoke.
            if self.config.generation.streaming_outputs and hasattr(self.llm, "stream"):
                chunks = []
                for chunk in self.llm.stream(messages):
                    token = self._message_content(chunk)
                    if token:
                        chunks.append(token)
                        if self.stream_callback:
                            self.stream_callback(token)
                answer = "".join(chunks).strip()
            else:
                response = self.llm.invoke(messages) if self.llm is not None else None
                answer = self._message_content(response)
        except Exception as exc:
            state.errors.append(f"generation_error: {exc}")
            answer = self._fallback_answer(state)

        if not answer:
            answer = self._fallback_answer(state)

        state.answer = self.guardrails.process_output(answer)
        response_message = AIMessage(content=state.answer)
        self.tracing.end_step(
            step,
            input_tokens=self._estimate_tokens(system_prompt + state.sanitized_query),
            output_tokens=self._estimate_tokens(state.answer),
            metadata={"errors": state.errors},
        )
        return {"messages": [response_message], "answer": state.answer}

    def _node_evaluate(self, state: AgentState) -> Dict[str, Any]:
        step = self.tracing.start_step("evaluate", "rag_triad")
        state.eval_scores = self.evaluator.evaluate(
            state.sanitized_query,
            state.context,
            state.answer,
        )
        self.tracing.end_step(step, metadata={"scores": state.eval_scores})
        return {"eval_scores": state.eval_scores}

    def _build_system_prompt(self, state: AgentState) -> str:
        """Build the grounded generation prompt.

        The prompt asks the model to use step-by-step / Chain-of-Thought style
        reasoning privately, then return only the final grounded answer.
        """

        return (
            "You are an enterprise Agentic RAG assistant. Use ReAct privately: "
            "reason about the request, decide whether retrieved context or tools are needed, "
            "then answer. Use step-by-step Chain-of-Thought internally, but do not reveal "
            "hidden reasoning. Ground every factual claim in the supplied context. "
            "If the context is insufficient, say what is missing.\n\n"
            f"Route: {state.route} (score {state.route_score:.3f})\n\n"
            f"Retrieved context:\n{state.context or '[No retrieved context]'}\n\n"
            f"Tool results:\n{state.tool_results or '[No tool results]'}"
        )

    def _heuristic_grade(self, query: str, context: str) -> float:
        query_terms = {term for term in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(term) > 2}
        context_terms = set(re.findall(r"[A-Za-z0-9_]+", context.lower()))
        if not query_terms:
            return 0.0
        lexical = len(query_terms & context_terms) / len(query_terms)
        length_bonus = min(len(context) / 1200, 0.25)
        return round(min(1.0, lexical + length_bonus), 4)

    def _fallback_answer(self, state: AgentState) -> str:
        if state.context:
            first_context = state.context.splitlines()[0:8]
            return (
                "I found relevant local context, but the LLM provider was unavailable. "
                "Here is the best grounded summary from retrieval:\n"
                + "\n".join(first_context)
            )
        return (
            "I do not have enough retrieved context to answer this reliably. "
            "Ingest documents first or configure an available Ollama/external model."
        )

    def _record_memory(self, state: AgentState) -> None:
        if not self.memory_manager:
            return
        try:
            if not getattr(self.memory_manager, "current_thread_id", ""):
                self.memory_manager.start_session("enterprise-session-1")
            self.memory_manager.record_turn("user", state.sanitized_query)
            self.memory_manager.record_turn("assistant", state.answer)
            self.memory_manager.save_agent_state(
                {
                    "query": state.sanitized_query,
                    "route": state.route,
                    "relevance_score": state.relevance_score,
                    "eval_scores": state.eval_scores,
                    "errors": state.errors,
                },
                "complete",
            )
        except Exception as exc:
            state.errors.append(f"memory_error: {exc}")

    def _extract_math_expression(self, text: str) -> str:
        match = re.search(r"(?<!\w)([\d\s+\-*/().]{3,})(?!\w)", text)
        return match.group(1).strip() if match else ""

    def _last_message_content(self, messages: Iterable[Any]) -> str:
        last = ""
        for message in messages:
            last = self._message_content(message)
        return last

    def _message_content(self, message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", message))

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(str(text).split())) if text else 0


def init_agent_graph(
    llm: Any,
    config: Optional[Any] = None,
    hybrid_search: Optional[HybridSearchEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
    guardrails: Optional[Guardrails] = None,
    memory_manager: Any = None,
    tracing_engine: Optional[TracingEngine] = None,
    evaluator: Optional[RAGTriadEvaluator] = None,
    hitl_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> AgenticRAGGraph:
    """Factory retained for the existing CLI."""

    rag_config = config if isinstance(config, RAGConfig) else RAGConfig()
    return AgenticRAGGraph(
        llm=llm,
        config=rag_config,
        hybrid_search=hybrid_search,
        tool_registry=tool_registry,
        guardrails=guardrails,
        memory_manager=memory_manager,
        tracing_engine=tracing_engine,
        evaluator=evaluator,
        hitl_callback=hitl_callback,
        stream_callback=stream_callback,
    )
