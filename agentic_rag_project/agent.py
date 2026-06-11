"""
Agents & Orchestration Module - The Core Agent Graph

Implements:
  - Agentic Frameworks (LangGraph-style orchestration)
  - Autonomous Agents (goal-driven, multi-step reasoning)
  - Reason + Act (ReAct) pattern (thought → action loops)
  - Tool Use / Function Calling
  - Human-in-the-Loop (HITL)
  - Multi-Agent Routing (specialized domain agents)
  - Self-Reflection / Self-Correction (evaluate & retry)
  - Directed Acyclic Graph (DAG) & Cyclic Graphs
  - Stateful Agents (memory-aware)
  - Chain-of-Thought (CoT) Prompting
  - Context Compression / Lost-in-the-Middle Mitigation
  - Streaming Outputs
"""

import re
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field

from memory import AgentMemoryManager
from security import Guardrails
from tools import ToolRegistry
from eval import TracingEngine, RAGTriadEvaluator, HallucinationDetector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Agent State (for DAG/Graph)
# ─────────────────────────────────────────────

@dataclass
class AgentState:
    """
    Stateful Agents:
    Maintains state across the agent execution graph.
    Tracks messages, context, scores, retries, tool calls, etc.
    """
    messages: List[Dict] = field(default_factory=list)
    context: str = ""
    graph_context: str = ""             # Graph RAG context
    relevance_score: float = 0.0
    groundedness_score: float = 0.0
    retries: int = 0
    current_node: str = ""
    tool_results: List[Dict] = field(default_factory=list)
    route: str = "default"
    needs_hitl: bool = False
    hitl_approved: bool = False
    generated_answer: str = ""
    evaluation_scores: Dict = field(default_factory=dict)
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "messages": self.messages,
            "context": self.context[:500],
            "relevance_score": self.relevance_score,
            "retries": self.retries,
            "current_node": self.current_node,
            "route": self.route,
            "generated_answer": self.generated_answer[:500],
        }


# ─────────────────────────────────────────────
# Specialized Domain Agents (Multi-Agent Routing)
# ─────────────────────────────────────────────

class SpecializedAgent:
    """
    Multi-Agent Routing:
    Specialized agents assigned to distinct domains.
    """
    
    def __init__(self, name: str, domain: str, system_prompt: str,
                 tools: List[str] = None):
        self.name = name
        self.domain = domain
        self.system_prompt = system_prompt
        self.tools = tools or []
    
    def get_system_prompt(self) -> str:
        return self.system_prompt


# Pre-defined specialist agents
SPECIALIST_AGENTS = {
    "technical": SpecializedAgent(
        name="Technical Specialist",
        domain="technical",
        system_prompt=(
            "You are a technical expert specializing in software engineering, "
            "AI/ML, and system architecture. Provide detailed, accurate technical answers. "
            "Always cite your sources from the retrieved context."
        ),
        tools=["document_search", "web_search", "calculator"],
    ),
    "security": SpecializedAgent(
        name="Security Analyst",
        domain="security",
        system_prompt=(
            "You are a cybersecurity specialist. Focus on security implications, "
            "best practices, and threat analysis. Be conservative and err on the side of caution."
        ),
        tools=["document_search", "sql_query"],
    ),
    "general": SpecializedAgent(
        name="General Assistant",
        domain="general",
        system_prompt=(
            "You are a helpful general-purpose AI assistant. "
            "Answer questions clearly and accurately using the provided context."
        ),
        tools=["document_search", "web_search", "calculator", "file_operations"],
    ),
    "data": SpecializedAgent(
        name="Data Analyst",
        domain="data",
        system_prompt=(
            "You are a data analysis specialist. Help with queries about data, "
            "databases, statistics, and analytical insights."
        ),
        tools=["document_search", "sql_query", "calculator"],
    ),
}


# ─────────────────────────────────────────────
# The Agentic RAG Graph (DAG with Cyclic Self-Correction)
# ─────────────────────────────────────────────

class AgenticRAGGraph:
    """
    Agentic Framework / Orchestration:
    
    The core agent graph implementing:
      - DAG structure with cyclic self-correction loops
      - ReAct (Reason + Act) pattern
      - Multi-Agent Routing
      - Self-Reflection / Self-Correction
      - Tool Use / Function Calling
      - Human-in-the-Loop (HITL)
      - Chain-of-Thought (CoT) Prompting
      - Context Compression & Lost-in-the-Middle Mitigation
    
    Graph nodes:
      1. route    → Semantic Router (intent classification)
      2. retrieve → Hybrid Search (BM25 + Semantic + Graph RAG)
      3. grade    → Self-Reflection (LLM-as-a-Judge)
      4. tools    → Tool Use / Function Calling
      5. generate → ReAct + CoT Generation
      6. evaluate → RAG Triad Evaluation
    
    Edges:
      - route → retrieve
      - retrieve → grade
      - grade → retrieve (self-correction loop / Cyclic Graph)
      - grade → tools (if tool use needed)
      - grade → generate
      - tools → generate
      - generate → evaluate
      - evaluate → END
    """
    
    def __init__(self, llm, config, retrieval_engine, memory_manager,
                 guardrails, tool_registry, tracing_engine,
                 rag_evaluator, hallucination_detector):
        self.llm = llm
        self.config = config
        self.retrieval = retrieval_engine
        self.memory = memory_manager
        self.guardrails = guardrails
        self.tools = tool_registry
        self.tracing = tracing_engine
        self.evaluator = rag_evaluator
        self.hallucination_detector = hallucination_detector
        
        # Agent graph configuration
        self.max_retries = config.orchestration.max_self_correction_retries
        self.specialist_agents = SPECIALIST_AGENTS
    
    def run(self, query: str, user_id: str = "default",
            hitl_callback=None, stream_callback=None) -> Dict:
        """
        Execute the full Agentic RAG pipeline.
        
        This is the main orchestration method implementing the DAG
        with cyclic self-correction loops.
        
        Args:
            query: User's input query
            user_id: For RBAC
            hitl_callback: Function to call for HITL approval (returns bool)
            stream_callback: Function to call for streaming output tokens
        
        Returns:
            Dict with answer, evaluation scores, and trace info
        """
        # Initialize state
        state = AgentState(
            messages=[{"role": "user", "content": query}],
        )
        
        # Start tracing
        trace = self.tracing.start_trace(query)
        
        # Record in memory
        self.memory.record_turn("user", query, user_id)
        
        # Input guardrails
        processed_query, guard_report = self.guardrails.process_input(query, user_id)
        if not guard_report.get("injection_safe", True):
            logger.warning(f"Prompt injection detected: {guard_report}")
        
        steps_output = []
        
        try:
            # ─── Node 1: ROUTE (Semantic Router / Multi-Agent Routing) ───
            step = self.tracing.start_step("route", "routing")
            state = self._node_route(state, processed_query)
            self.tracing.end_step(step, metadata={"route": state.route})
            steps_output.append({"step": "route", "route": state.route})
            self.memory.save_agent_state(state.to_dict(), "route")
            
            # ─── Node 2: RETRIEVE (Hybrid Search + Graph RAG) ───
            step = self.tracing.start_step("retrieve", "retrieval")
            state = self._node_retrieve(state, processed_query, user_id)
            self.tracing.end_step(step, metadata={"context_length": len(state.context)})
            steps_output.append({"step": "retrieve", "context_length": len(state.context)})
            self.memory.save_agent_state(state.to_dict(), "retrieve")
            
            # ─── Node 3: GRADE (Self-Reflection / Self-Correction) ───
            # This is the CYCLIC part - can loop back to retrieve
            while state.retries <= self.max_retries:
                step = self.tracing.start_step(f"grade_attempt_{state.retries}", "grading")
                state = self._node_grade(state, processed_query)
                self.tracing.end_step(step, metadata={
                    "relevance_score": state.relevance_score,
                    "retries": state.retries,
                })
                steps_output.append({
                    "step": "grade",
                    "relevance_score": state.relevance_score,
                    "retry": state.retries,
                })
                self.memory.save_agent_state(state.to_dict(), "grade")
                
                # Self-correction: if score is too low, retry retrieval
                if state.relevance_score >= 0.6 or state.retries >= self.max_retries:
                    break
                
                # Cyclic Graph: loop back to retrieve
                logger.info(f"Self-Correction: Score {state.relevance_score:.2f} too low, retrying... (attempt {state.retries})")
                state.retries += 1
                step = self.tracing.start_step(f"retrieve_retry_{state.retries}", "retrieval")
                state = self._node_retrieve(state, processed_query, user_id)
                self.tracing.end_step(step)
                steps_output.append({"step": "retrieve_retry", "attempt": state.retries})
            
            # ─── Node 4: TOOLS (Tool Use / Function Calling) ───
            if self._should_use_tools(state, processed_query):
                step = self.tracing.start_step("tools", "tool_use")
                
                # HITL check for critical tools
                if hitl_callback and state.needs_hitl:
                    state.hitl_approved = hitl_callback(
                        f"Agent wants to use tools. Approve? (Tools: {self._detect_needed_tools(processed_query)})"
                    )
                    if not state.hitl_approved:
                        steps_output.append({"step": "hitl", "approved": False})
                        self.tracing.end_step(step, metadata={"hitl_blocked": True})
                    else:
                        state = self._node_tools(state, processed_query)
                        self.tracing.end_step(step, metadata={"tool_results": len(state.tool_results)})
                        steps_output.append({"step": "tools", "results": len(state.tool_results)})
                else:
                    state = self._node_tools(state, processed_query)
                    self.tracing.end_step(step, metadata={"tool_results": len(state.tool_results)})
                    steps_output.append({"step": "tools", "results": len(state.tool_results)})
                
                self.memory.save_agent_state(state.to_dict(), "tools")
            
            # ─── Node 5: GENERATE (ReAct + CoT + Streaming) ───
            step = self.tracing.start_step("generate", "generation")
            state = self._node_generate(state, processed_query, user_id, stream_callback)
            
            # Estimate tokens
            input_tokens = len(processed_query + state.context) // 4
            output_tokens = len(state.generated_answer) // 4
            self.tracing.end_step(step, input_tokens=input_tokens, output_tokens=output_tokens)
            steps_output.append({"step": "generate", "answer_length": len(state.generated_answer)})
            self.memory.save_agent_state(state.to_dict(), "generate")
            
            # ─── Node 6: EVALUATE (RAG Triad) ───
            step = self.tracing.start_step("evaluate", "evaluation")
            state = self._node_evaluate(state, processed_query)
            self.tracing.end_step(step, metadata={"scores": state.evaluation_scores})
            steps_output.append({"step": "evaluate", "scores": state.evaluation_scores})
            
            # Output guardrails
            state.generated_answer = self.guardrails.process_output(state.generated_answer)
            
            # Record in memory
            self.memory.record_turn("assistant", state.generated_answer, user_id)
            self.memory.long_term.record_interaction(
                user_id, query, state.generated_answer, state.evaluation_scores
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            state.error = str(e)
            state.generated_answer = f"I encountered an error processing your query: {str(e)}"
            self.memory.record_failure("pipeline", str(e))
        
        # End trace
        trace = self.tracing.end_trace(state.evaluation_scores)
        
        return {
            "answer": state.generated_answer,
            "evaluation_scores": state.evaluation_scores,
            "steps": steps_output,
            "trace_id": trace.trace_id if trace else "",
            "latency_ms": trace.total_latency_ms if trace else 0,
            "total_tokens": (trace.total_input_tokens + trace.total_output_tokens) if trace else 0,
            "cost_usd": trace.total_cost_usd if trace else 0,
            "route": state.route,
            "retries": state.retries,
            "error": state.error,
        }
    
    # ─────────────────────────────────────────
    # Graph Node Implementations
    # ─────────────────────────────────────────
    
    def _node_route(self, state: AgentState, query: str) -> AgentState:
        """
        Semantic Router / Multi-Agent Routing:
        Analyze the user's intent and route to the correct specialist agent.
        """
        state.current_node = "route"
        
        # Use semantic router if available
        semantic_router = self.retrieval.get("semantic_router")
        if semantic_router and semantic_router.routes:
            route_name, confidence = semantic_router.route(query)
            state.route = route_name
        else:
            # Keyword-based routing fallback
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["security", "vulnerability", "attack", "threat", "rbac"]):
                state.route = "security"
            elif any(kw in query_lower for kw in ["data", "database", "sql", "statistics", "analytics"]):
                state.route = "data"
            elif any(kw in query_lower for kw in ["code", "api", "architecture", "system", "algorithm"]):
                state.route = "technical"
            else:
                state.route = "general"
        
        logger.info(f"Route: Query routed to '{state.route}' agent")
        return state
    
    def _node_retrieve(self, state: AgentState, query: str,
                        user_id: str) -> AgentState:
        """
        Hybrid Search + Hierarchical Retrieval + Graph RAG:
        Retrieve relevant context using the full retrieval engine.
        """
        state.current_node = "retrieve"
        
        # Get RBAC metadata filter
        metadata_filter = self.guardrails.rbac_manager.get_metadata_filter(user_id)
        
        # Hybrid search (BM25 + Semantic + Re-ranking + Hierarchical)
        hybrid_search = self.retrieval.get("hybrid_search")
        context_manager = self.retrieval.get("context_manager")
        knowledge_graph = self.retrieval.get("knowledge_graph")
        
        context_parts = []
        
        if hybrid_search:
            results = hybrid_search.search(
                query,
                top_k=self.config.retrieval.top_k,
                metadata_filter=metadata_filter,
                use_reranking=self.config.retrieval.reranking_enabled,
                use_hierarchical=self.config.retrieval.hierarchical_retrieval,
            )
            
            # Context Compression & Lost-in-the-Middle Mitigation
            if context_manager and results:
                compressed = context_manager.compress_context(results, query)
                context_parts.append(compressed)
        
        # Graph RAG: Add entity-relationship context
        if knowledge_graph and self.config.retrieval.graph_rag_enabled:
            graph_context = knowledge_graph.get_context_for_query(query)
            if graph_context:
                context_parts.append(f"\n[Graph Knowledge]\n{graph_context}")
                state.graph_context = graph_context
        
        # Conversation history for context
        history = self.memory.short_term.get_formatted_history(last_n=5)
        if history:
            context_parts.append(f"\n[Conversation History]\n{history}")
        
        # Past failures for self-correction
        failure_context = self.memory.get_failure_context()
        if failure_context:
            context_parts.append(f"\n{failure_context}")
        
        state.context = "\n\n".join(context_parts)
        return state
    
    def _node_grade(self, state: AgentState, query: str) -> AgentState:
        """
        Self-Reflection / Self-Correction:
        Evaluate the quality of retrieved context.
        Uses LLM-as-a-Judge pattern.
        """
        state.current_node = "grade"
        
        if not state.context.strip():
            state.relevance_score = 0.0
            return state
        
        # Use LLM-as-a-Judge if available, otherwise keyword overlap
        if self.llm:
            try:
                score = self.evaluator._evaluate_context_relevance(query, state.context)
                state.relevance_score = score
            except Exception as e:
                logger.warning(f"LLM grading failed: {e}, using heuristic")
                state.relevance_score = self._heuristic_grade(query, state.context)
        else:
            state.relevance_score = self._heuristic_grade(query, state.context)
        
        logger.info(f"Grade: Context relevance = {state.relevance_score:.3f}")
        return state
    
    def _heuristic_grade(self, query: str, context: str) -> float:
        """Heuristic context grading based on keyword overlap."""
        query_words = set(re.findall(r'\w+', query.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        if not query_words:
            return 0.0
        overlap = len(query_words & context_words) / len(query_words)
        return min(overlap, 1.0)
    
    def _should_use_tools(self, state: AgentState, query: str) -> bool:
        """Determine if the query requires tool use."""
        tool_triggers = [
            "search", "find", "look up", "calculate", "compute",
            "query", "database", "file", "read", "list",
        ]
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in tool_triggers)
    
    def _detect_needed_tools(self, query: str) -> List[str]:
        """Detect which tools might be needed."""
        tools = []
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["search", "find", "look up"]):
            tools.append("document_search")
        if any(kw in query_lower for kw in ["web", "internet", "online"]):
            tools.append("web_search")
        if any(kw in query_lower for kw in ["database", "sql", "query"]):
            tools.append("sql_query")
        if any(kw in query_lower for kw in ["calculate", "compute", "math"]):
            tools.append("calculator")
        if any(kw in query_lower for kw in ["file", "read", "directory"]):
            tools.append("file_operations")
        
        return tools or ["document_search"]
    
    def _node_tools(self, state: AgentState, query: str) -> AgentState:
        """
        Tool Use / Function Calling:
        Execute tools based on query analysis.
        """
        state.current_node = "tools"
        needed_tools = self._detect_needed_tools(query)
        
        for tool_name in needed_tools:
            tool = self.tools.get_tool(tool_name)
            if not tool:
                continue
            
            # Check HITL for critical tools
            if tool.requires_hitl:
                state.needs_hitl = True
                if not state.hitl_approved:
                    logger.info(f"Tool '{tool_name}' requires HITL approval - skipping")
                    continue
            
            try:
                if tool_name == "document_search":
                    result = tool.execute(query=query)
                elif tool_name == "web_search":
                    result = tool.execute(query=query)
                elif tool_name == "sql_query":
                    result = tool.execute(query=f"SELECT * FROM documents WHERE content LIKE '%{query[:20]}%' LIMIT 5")
                elif tool_name == "calculator":
                    # Extract math expression from query
                    numbers = re.findall(r'[\d.+\-*/()]+', query)
                    expr = numbers[0] if numbers else "0"
                    result = tool.execute(expression=expr)
                elif tool_name == "file_operations":
                    result = tool.execute(action="list")
                else:
                    result = tool.execute(query=query)
                
                state.tool_results.append(result)
                
                # Add tool results to context
                if result.get("status") == "success":
                    tool_context = json.dumps(result, indent=2, default=str)[:2000]
                    state.context += f"\n\n[Tool: {tool_name}]\n{tool_context}"
                
            except Exception as e:
                logger.error(f"Tool '{tool_name}' error: {e}")
                state.tool_results.append({
                    "status": "error", "tool": tool_name, "error": str(e)
                })
        
        return state
    
    def _node_generate(self, state: AgentState, query: str,
                        user_id: str, stream_callback=None) -> AgentState:
        """
        Reason + Act (ReAct) + Chain-of-Thought (CoT) Generation.
        
        Implements:
          - CoT Prompting (step-by-step reasoning)
          - Lost in the Middle Mitigation (strategic context ordering)
          - Streaming Outputs (real-time token delivery)
          - Speculative Decoding awareness
        """
        state.current_node = "generate"
        
        # Get specialist agent prompt
        specialist = self.specialist_agents.get(state.route, self.specialist_agents["general"])
        
        # Build the ReAct + CoT system prompt
        system_prompt = self._build_system_prompt(specialist, state)
        
        # Build messages
        messages = [
            ("system", system_prompt),
            ("human", query),
        ]
        
        if self.llm is None:
            # No LLM available - return context-based response
            state.generated_answer = self._generate_without_llm(query, state.context)
            return state
        
        try:
            if stream_callback and self.config.generation.streaming_outputs:
                # Streaming Outputs: deliver tokens in real-time
                full_response = ""
                for chunk in self.llm.stream(messages):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += token
                    if stream_callback:
                        stream_callback(token)
                state.generated_answer = full_response
            else:
                # Standard generation
                response = self.llm.invoke(messages)
                state.generated_answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            state.generated_answer = self._generate_without_llm(query, state.context)
            state.error = str(e)
        
        return state
    
    def _build_system_prompt(self, specialist: SpecializedAgent,
                              state: AgentState) -> str:
        """
        Build the system prompt with:
          - Specialist agent instructions
          - Chain-of-Thought (CoT) instructions
          - Retrieved context (with Lost-in-the-Middle mitigation applied)
          - Tool results
          - Available tools description
        """
        parts = [
            specialist.get_system_prompt(),
            "",
            "## Instructions",
            "1. Think step-by-step before answering (Chain-of-Thought).",
            "2. Base your answer ONLY on the provided context (Groundedness).",
            "3. If the context doesn't contain the answer, say so honestly.",
            "4. Cite specific parts of the context when possible.",
            "",
        ]
        
        # Context (already compressed and reordered for Lost-in-the-Middle)
        if state.context:
            parts.append("## Retrieved Context")
            parts.append(state.context)
            parts.append("")
        
        # Tool results
        if state.tool_results:
            parts.append("## Tool Results")
            for result in state.tool_results:
                parts.append(json.dumps(result, indent=2, default=str)[:1000])
            parts.append("")
        
        # Available tools
        tools_desc = self.tools.get_tools_description()
        if tools_desc:
            parts.append(f"## {tools_desc}")
            parts.append("")
        
        return "\n".join(parts)
    
    def _generate_without_llm(self, query: str, context: str) -> str:
        """
        Generate a response without an LLM (fallback mode).
        Returns a context-based summary.
        """
        if not context.strip():
            return (
                "I don't have enough context to answer this question. "
                "Please ingest some documents first using the 'ingest' command, "
                "or configure an LLM provider."
            )
        
        # Return relevant portions of the context
        return (
            f"Based on the available context for your query '{query}':\n\n"
            f"{context[:2000]}\n\n"
            f"Note: This is a direct context excerpt. Configure an LLM for "
            f"synthesized answers."
        )
    
    def _node_evaluate(self, state: AgentState, query: str) -> AgentState:
        """
        RAG Triad Evaluation:
          - Context Relevance
          - Groundedness / Faithfulness
          - Answer Relevance
          - Hallucination Detection
        """
        state.current_node = "evaluate"
        
        # RAG Triad
        triad_scores = self.evaluator.evaluate(
            query, state.context, state.generated_answer
        )
        
        # Hallucination Detection
        hallucination_report = self.hallucination_detector.detect(
            state.context, state.generated_answer
        )
        
        state.evaluation_scores = {
            **triad_scores,
            "hallucination_detected": hallucination_report.get("has_potential_hallucination", False),
            "grounding_score": hallucination_report.get("grounding_score", 0.0),
        }
        
        return state


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_agent_graph(llm, config, retrieval_engine, memory_manager,
                        guardrails, tool_registry, eval_stack) -> AgenticRAGGraph:
    """Create the full agentic RAG graph."""
    return AgenticRAGGraph(
        llm=llm,
        config=config,
        retrieval_engine=retrieval_engine,
        memory_manager=memory_manager,
        guardrails=guardrails,
        tool_registry=tool_registry,
        tracing_engine=eval_stack["tracing_engine"],
        rag_evaluator=eval_stack["rag_evaluator"],
        hallucination_detector=eval_stack["hallucination_detector"],
    )
