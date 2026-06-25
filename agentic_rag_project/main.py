"""Interactive CLI for the Enterprise Agentic RAG project."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table
except Exception:  # pragma: no cover - used when rich is not installed.
    class FallbackConsole:
        def print(self, *values, **kwargs):
            print(*values)

    class Prompt:
        @staticmethod
        def ask(message, choices=None, default=None, password=False):
            suffix = f" [{default}]" if default else ""
            value = input(f"{message}{suffix}: ").strip()
            return value or default

    class Table:
        def __init__(self, title=None):
            self.title = title
            self.columns = []
            self.rows = []

        def add_column(self, name):
            self.columns.append(name)

        def add_row(self, *values):
            self.rows.append(values)

        def __str__(self):
            lines = [self.title or ""]
            lines.append(" | ".join(self.columns))
            lines.extend(" | ".join(row) for row in self.rows)
            return "\n".join(line for line in lines if line)

    Console = FallbackConsole

from agent import init_agent_graph
from config import LLMConfig, RAGConfig
from eval import GoldenDataset, HallucinationDetector, RAGTriadEvaluator, TracingEngine
from ingestion import create_ingestion_pipeline
from llm_provider import LLMProvider, detect_ollama_models
from memory import create_memory_manager
from retrieval import ContextWindowManager, EmbeddingEngine, HybridSearchEngine, KnowledgeGraph
from security import create_security_stack
from tools import create_tool_registry

try:
    from langchain_core.messages import HumanMessage
except Exception:  # pragma: no cover
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content


console = Console()

SAMPLE_DOCUMENT = """Enterprise Agentic RAG reference

Agentic Frameworks manage autonomous agents through a graph of route,
retrieve, grade, tool, generate, and evaluate steps. ReAct alternates between
private reasoning and action. Human-in-the-loop approval is required for
critical tools such as SQL execution. Multi-agent routing sends security,
retrieval, tool, and analysis work to specialist agents.

The retrieval engine uses embeddings, semantic search, BM25 keyword search,
hybrid search, metadata filtering, parent-document retrieval, reranking,
semantic routing, context compression, lost-in-the-middle mitigation, and
Graph RAG entity context.

LLMOps tracks latency, token usage, cost estimates, traces, golden datasets,
LLM-as-a-judge scores, hallucination risk, context relevance, groundedness,
and answer relevance. Security includes prompt injection defense, PII masking,
RBAC, guardrails, and air-gapped Ollama deployment.
"""


def configure_llm_interactive(base_config: Optional[RAGConfig] = None) -> RAGConfig:
    """Ask the user whether to use a downloaded Ollama model or an external API."""

    config = base_config or RAGConfig()
    console.print("[bold green]Enterprise Agentic RAG Setup[/bold green]")
    models = detect_ollama_models()

    if models:
        console.print("[bold cyan]Downloaded Ollama models detected:[/bold cyan]")
        table = Table(title="Local Ollama Models")
        table.add_column("#")
        table.add_column("Model")
        table.add_column("Size")
        for index, model in enumerate(models, start=1):
            table.add_row(str(index), model.get("name", ""), model.get("size", ""))
        console.print(table)
        use_local = Prompt.ask(
            "Do you want to use a locally downloaded Ollama model?",
            choices=["yes", "no"],
            default="yes",
        )
    else:
        console.print("[yellow]No downloaded Ollama models were detected.[/yellow]")
        use_local = Prompt.ask(
            "Use an external chatbot/API provider instead?",
            choices=["yes", "no"],
            default="yes",
        )
        use_local = "no" if use_local == "yes" else "yes"

    if use_local == "yes" and models:
        selected = Prompt.ask("Select a local model number", default="1")
        try:
            model = models[int(selected) - 1]["name"]
        except (ValueError, IndexError):
            model = models[0]["name"]
        config.llm = LLMConfig(provider_type="ollama", model_name=model, air_gapped_mode=True)
        config.security.air_gapped_mode = True
        console.print(f"[green]Using local Ollama model: {model}[/green]")
        return config

    provider = Prompt.ask(
        "Select external API provider",
        choices=["openai", "anthropic", "gemini", "custom"],
        default="gemini",
    ).lower()
    default_model = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
        "gemini": "gemini-1.5-flash",
        "custom": "local-openai-compatible-model",
    }[provider]
    model_name = Prompt.ask("Model name", default=default_model)
    key_label = "Google API key" if provider == "gemini" else "API key"
    api_key = Prompt.ask(f"{key_label} (leave blank if your custom endpoint does not require one)", password=True, default="")
    api_base_url = ""
    if provider == "custom":
        api_base_url = Prompt.ask(
            "OpenAI-compatible base URL",
            default=os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1"),
        )

    console.print(
        "[yellow]External API selected: prompt injection defense, PII masking, "
        "RBAC metadata filtering, and output guardrails remain enabled.[/yellow]"
    )
    config.llm = LLMConfig(
        provider_type=provider,
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
        air_gapped_mode=False,
    )
    config.security.air_gapped_mode = False
    return config


def configure_llm() -> Dict[str, str]:
    """Backward-compatible wrapper for earlier versions of the project."""

    config = configure_llm_interactive()
    return {
        "type": config.llm.provider_type,
        "provider": config.llm.provider_type,
        "model": config.llm.model_name,
        "api_key": config.llm.api_key,
        "api_base_url": config.llm.api_base_url,
        "air_gapped_mode": str(config.llm.air_gapped_mode),
    }


def ensure_sample_document(config: RAGConfig) -> str:
    os.makedirs(config.documents_dir, exist_ok=True)
    sample_path = os.path.join(config.documents_dir, "agentic_rag_reference.md")
    if not os.path.exists(sample_path):
        with open(sample_path, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE_DOCUMENT)
    return sample_path


def ingest_documents(
    config: RAGConfig,
    directory: Optional[str] = None,
) -> Tuple[HybridSearchEngine, KnowledgeGraph, List[Any]]:
    """Parse, chunk, index, and graph documents."""

    directory = directory or config.documents_dir
    ensure_sample_document(config)
    pipeline = create_ingestion_pipeline(config)
    documents = pipeline.ingest_directory(directory, use_hierarchy=True)

    chunks = []
    parents = []
    graph = KnowledgeGraph()
    for document in documents:
        chunks.extend(document.chunks)
        parents.extend(document.parent_chunks)
        for chunk in document.parent_chunks or document.chunks:
            graph.add_chunk(chunk)

    search = HybridSearchEngine.from_chunks(
        chunks,
        parent_chunks=parents,
        embedding_engine=EmbeddingEngine(),
        persist_dir=os.path.join(config.data_dir, "vectordb"),
    )
    search.vector_store.persist()
    console.print(f"[green]Indexed {len(chunks)} child chunks and {len(parents)} parent chunks.[/green]")
    return search, graph, documents


def hitl_callback(action: str, payload: Dict[str, Any]) -> bool:
    """Human-in-the-loop approval callback for critical tool actions."""

    decision = Prompt.ask(
        f"HITL approval required for '{action}'. Allow?",
        choices=["yes", "no"],
        default="no",
    )
    return decision == "yes"


def stream_callback(token: str) -> None:
    """Token streaming callback used by providers that support streaming."""

    print(token, end="", flush=True)


def show_help() -> None:
    console.print(
        """
[bold]Commands[/bold]
help               Show this command list
ingest [path]      Ingest documents from a folder
status             Show system status and observability summary
benchmark          Run the golden dataset benchmark
history            Show recent short-term memory
tools              List registered function-calling tools
security           Show guardrail and RBAC status
traces             Show trace summary
quit               Save state and exit
"""
    )


def show_system_status(runtime: Dict[str, Any]) -> None:
    config: RAGConfig = runtime["config"]
    summary = runtime["tracing"].get_summary()
    table = Table(title="Agentic RAG Status")
    table.add_column("Area")
    table.add_column("Value")
    table.add_row("Provider", f"{config.llm.provider_type}:{config.llm.model_name}")
    table.add_row("air_gapped_mode", str(config.security.air_gapped_mode))
    table.add_row("Hybrid search", str(runtime.get("hybrid_search") is not None))
    table.add_row("Traces", str(summary.get("total_queries", 0)))
    table.add_row("Avg latency ms", f"{summary.get('average_latency_ms', 0):.2f}")
    console.print(table)


def run_benchmark(runtime: Dict[str, Any]) -> None:
    dataset = GoldenDataset(runtime["config"].observability.golden_dataset_path)
    report = dataset.run_benchmark(runtime["evaluator"])
    console.print(f"[green]Golden dataset entries: {report['entry_count']}[/green]")
    console.print(report["average_scores"])


def show_history(runtime: Dict[str, Any]) -> None:
    memory = runtime["memory"]
    console.print(memory.short_term.get_formatted_history(last_n=10) or "[dim]No history yet.[/dim]")


def show_tools(runtime: Dict[str, Any]) -> None:
    console.print(runtime["tools"].get_tools_description())


def show_security(runtime: Dict[str, Any]) -> None:
    guardrails = runtime["guardrails"]
    table = Table(title="Security")
    table.add_column("Control")
    table.add_column("Status")
    table.add_row("PII Masking", "enabled")
    table.add_row("Prompt Injection Defense", "enabled")
    table.add_row("RBAC Filter", str(guardrails.rbac_manager.get_metadata_filter("default")))
    table.add_row("Critical HITL Actions", ", ".join(guardrails.CRITICAL_ACTIONS))
    console.print(table)


def show_traces(runtime: Dict[str, Any]) -> None:
    console.print(runtime["tracing"].get_summary())


def build_runtime(config: RAGConfig) -> Dict[str, Any]:
    """Create all runtime components and wire the graph."""

    ensure_sample_document(config)
    hybrid_search, graph, documents = ingest_documents(config)
    guardrails = create_security_stack(config)
    memory = create_memory_manager(config)
    memory.start_session("enterprise-session-1")
    tracing = TracingEngine(config.observability.trace_dir)
    evaluator = RAGTriadEvaluator()
    context_manager = ContextWindowManager(config.generation.context_window_tokens)
    tools = create_tool_registry(hybrid_search=hybrid_search, context_manager=context_manager)

    try:
        provider = LLMProvider.from_config(config)
        llm = provider
        console.print(f"[green]LLM initialized: {provider.get_info()}[/green]")
    except Exception as exc:
        llm = None
        console.print(f"[yellow]LLM unavailable, retrieval-only fallback enabled: {exc}[/yellow]")

    app = init_agent_graph(
        llm,
        config,
        hybrid_search=hybrid_search,
        tool_registry=tools,
        guardrails=guardrails,
        memory_manager=memory,
        tracing_engine=tracing,
        evaluator=evaluator,
        hitl_callback=hitl_callback,
        stream_callback=None,
    )
    return {
        "config": config,
        "app": app,
        "hybrid_search": hybrid_search,
        "knowledge_graph": graph,
        "documents": documents,
        "guardrails": guardrails,
        "memory": memory,
        "tracing": tracing,
        "evaluator": evaluator,
        "tools": tools,
    }


def initialize_agentic_rag(config: Optional[RAGConfig] = None) -> None:
    """Initialize components and start the interactive chat loop."""

    runtime = build_runtime(config or RAGConfig())
    thread_config = {"configurable": {"thread_id": "enterprise-session-1"}}
    show_help()

    while True:
        query = Prompt.ask("\nUser Query", default="status")
        command, _, argument = query.partition(" ")
        command = command.lower().strip()

        if command in {"quit", "exit", "q"}:
            runtime["memory"].long_term.save()
            console.print("[green]State saved. Goodbye.[/green]")
            break
        if command == "help":
            show_help()
            continue
        if command == "ingest":
            runtime["hybrid_search"], runtime["knowledge_graph"], runtime["documents"] = ingest_documents(
                runtime["config"], argument.strip() or None
            )
            continue
        if command == "status":
            show_system_status(runtime)
            continue
        if command == "benchmark":
            run_benchmark(runtime)
            continue
        if command == "history":
            show_history(runtime)
            continue
        if command == 'tools':
            show_tools(runtime)
            continue
        if command == 'security':
            show_security(runtime)
            continue
        if command == 'traces':
            show_traces(runtime)
            continue

        try:
            for event in runtime["app"].stream({"messages": [HumanMessage(content=query)]}, thread_config):
                for key, value in event.items():
                    if key == "grade":
                        console.print(f"[dim]Context relevance: {value.get('relevance_score')}[/dim]")
                    if key == "generate":
                        messages = value.get("messages", [])
                        if messages:
                            console.print(f"\n[bold green]Agentic RAG Output[/bold green]\n{messages[-1].content}")
                    if key == "evaluate":
                        console.print(f"[dim]RAG triad: {value.get('eval_scores')}[/dim]")
        except Exception as exc:
            console.print(f"[red]Pipeline error: {exc}[/red]")


if __name__ == "__main__":
    initialize_agentic_rag(configure_llm_interactive())
