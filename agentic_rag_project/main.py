"""
Enterprise Agentic RAG - Main Application Entry Point

This is the main interactive application that orchestrates the full
Agentic RAG pipeline. It provides:

1. Interactive LLM setup (Ollama local / External API)
2. Document ingestion pipeline
3. Interactive chat with the Agentic RAG system
4. Evaluation and benchmarking commands
5. System status and observability dashboard

Architecture Terms Implemented:
  - Agents & Orchestration (ReAct, HITL, Multi-Agent Routing, Self-Correction)
  - Advanced State Management (DAG, Cyclic Graphs, Short/Long-Term Memory, Persistence)
  - Deep Data Engineering (Parsing, OCR, Multi-Modal, Chunking, Hierarchical Retrieval)
  - Retrieval & Memory Engine (Vector DB, Hybrid Search, Re-ranking, Graph RAG, Semantic Router)
  - Advanced Generation (CoT, Context Compression, Lost-in-the-Middle, Streaming)
  - LLMOps & Observability (Tracing, LLM-as-a-Judge, Golden Dataset, Latency)
  - Security & Guardrails (PII Masking, Prompt Injection Defense, RBAC, Air-Gapped)
  - Evaluation (RAG Triad, Hallucination Detection)
"""

import os
import sys
import time
import json
import logging
from typing import Optional, Dict

# ─── Setup Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("AgenticRAG")

# ─── Rich Console for Beautiful Output ───
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Create console
if HAS_RICH:
    console = Console()
else:
    class FallbackConsole:
        def print(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            # Strip rich markup
            import re
            clean = re.sub(r'\[/?[^\]]*\]', '', text)
            print(clean)
    console = FallbackConsole()


# ─────────────────────────────────────────────
# Module Imports
# ─────────────────────────────────────────────

from config import RAGConfig, LLMConfig
from llm_provider import LLMProvider, detect_ollama_models, is_ollama_running
from security import create_security_stack, UserProfile
from ingestion import create_ingestion_pipeline
from retrieval import create_retrieval_engine
from memory import create_memory_manager
from tools import create_tool_registry
from eval import create_evaluation_stack
from agent import create_agent_graph


# ─────────────────────────────────────────────
# Interactive LLM Configuration
# ─────────────────────────────────────────────

def configure_llm_interactive() -> LLMConfig:
    """
    Interactive LLM configuration.
    Asks the user if they want to use:
      1. Local Ollama model (Air-Gapped Deployment)
      2. External API (OpenAI, Anthropic, Custom)
    
    Auto-detects installed Ollama models.
    """
    llm_config = LLMConfig()
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Enterprise Agentic RAG System[/bold cyan]\n"
        "[dim]Comprehensive AI Pipeline with Full Observability[/dim]",
        border_style="cyan",
    )) if HAS_RICH else console.print("=== Enterprise Agentic RAG System ===")
    
    console.print("\n[bold yellow]Step 1: Configure LLM Provider[/bold yellow]")
    console.print("[dim]Choose between local (Air-Gapped) or external LLM.[/dim]\n")
    
    # Ask about local model
    if HAS_RICH:
        use_local = Prompt.ask(
            "Do you want to use a locally downloaded Ollama model? (Air-Gapped Deployment)",
            choices=["yes", "no"],
            default="yes"
        )
    else:
        use_local = input("Use local Ollama model? (yes/no) [yes]: ").strip().lower() or "yes"
    
    if use_local == "yes":
        return _configure_ollama(llm_config)
    else:
        return _configure_external(llm_config)


def _configure_ollama(llm_config: LLMConfig) -> LLMConfig:
    """Configure Ollama local model."""
    console.print("\n[bold green]🔍 Detecting Ollama models...[/bold green]")
    
    # Check if Ollama is running
    if not is_ollama_running():
        console.print("[yellow]⚠ Ollama server doesn't seem to be running.[/yellow]")
        console.print("[dim]Start it with: ollama serve[/dim]")
    
    # Detect installed models
    models = detect_ollama_models()
    
    if not models:
        console.print("[bold red]❌ No Ollama models detected.[/bold red]")
        console.print("Install Ollama from https://ollama.com and download a model:")
        console.print("  ollama pull llama3")
        console.print("  ollama pull mistral")
        console.print("  ollama pull gemma2")
        console.print("\n[yellow]Falling back to external API configuration...[/yellow]")
        return _configure_external(llm_config)
    
    # Display detected models
    if HAS_RICH:
        table = Table(title="Detected Ollama Models", box=box.ROUNDED)
        table.add_column("#", style="cyan", justify="right")
        table.add_column("Model Name", style="green")
        table.add_column("Model ID", style="dim")
        table.add_column("Size", style="yellow")
        
        for i, model in enumerate(models, 1):
            table.add_row(str(i), model["name"], model.get("id", ""), model.get("size", ""))
        
        console.print(table)
    else:
        console.print("Detected models:")
        for i, model in enumerate(models, 1):
            console.print(f"  {i}. {model['name']}")
    
    # Select model
    if HAS_RICH:
        choice = Prompt.ask("Select model number", default="1")
    else:
        choice = input(f"Select model (1-{len(models)}) [1]: ").strip() or "1"
    
    try:
        idx = int(choice) - 1
        selected = models[idx]
        llm_config.provider_type = "ollama"
        llm_config.model_name = selected["name"]
        console.print(f"\n[bold green]✅ Selected: {selected['name']} (Local/Air-Gapped)[/bold green]")
    except (ValueError, IndexError):
        console.print("[red]Invalid selection. Using first model.[/red]")
        llm_config.provider_type = "ollama"
        llm_config.model_name = models[0]["name"]
    
    return llm_config


def _configure_external(llm_config: LLMConfig) -> LLMConfig:
    """Configure external API provider."""
    console.print("\n[bold cyan]🌐 Configure External LLM API[/bold cyan]")
    console.print("[yellow]⚠ Warning: Using external APIs means data will leave your network.[/yellow]")
    console.print("[yellow]  PII Masking and Guardrails will be enabled automatically.[/yellow]\n")
    
    # Select provider
    if HAS_RICH:
        provider = Prompt.ask(
            "Select API provider",
            choices=["OpenAI", "Anthropic", "Custom"],
            default="OpenAI"
        )
    else:
        provider = input("API provider (OpenAI/Anthropic/Custom) [OpenAI]: ").strip() or "OpenAI"
    
    # Get API key
    if HAS_RICH:
        api_key = Prompt.ask(f"Enter your {provider} API Key", password=True)
    else:
        api_key = input(f"Enter {provider} API key: ").strip()
    
    # Get model name
    default_models = {
        "OpenAI": "gpt-4o",
        "Anthropic": "claude-3-5-sonnet-20241022",
        "Custom": "custom-model",
    }
    
    if HAS_RICH:
        model_name = Prompt.ask(
            "Enter model name",
            default=default_models.get(provider, "gpt-4o")
        )
    else:
        default = default_models.get(provider, "gpt-4o")
        model_name = input(f"Model name [{default}]: ").strip() or default
    
    llm_config.provider_type = provider.lower()
    llm_config.model_name = model_name
    llm_config.api_key = api_key
    
    # Custom base URL for custom providers
    if provider == "Custom":
        if HAS_RICH:
            base_url = Prompt.ask("Enter API base URL", default="http://localhost:8000/v1")
        else:
            base_url = input("API base URL [http://localhost:8000/v1]: ").strip() or "http://localhost:8000/v1"
        llm_config.api_base_url = base_url
    
    console.print(f"\n[bold green]✅ Configured: {provider} ({model_name})[/bold green]")
    return llm_config


# ─────────────────────────────────────────────
# System Initialization
# ─────────────────────────────────────────────

def initialize_system(config: RAGConfig) -> Dict:
    """
    Initialize the full Agentic RAG pipeline.
    
    Components initialized:
      1. Security Stack (PII Masking, Prompt Injection, RBAC, Guardrails)
      2. Ingestion Pipeline (Parsing, OCR, Chunking, Hierarchical)
      3. Retrieval Engine (Embeddings, Vector DB, BM25, Hybrid Search, etc.)
      4. Memory Manager (Short/Long Term, Persistence, Time-Travel)
      5. Tool Registry (Web Search, SQL, File, Calculator, Doc Search)
      6. Evaluation Stack (Tracing, RAG Triad, Hallucination, Golden Dataset)
      7. LLM Provider (Ollama / External)
      8. Agent Graph (Full DAG with Self-Correction loops)
    """
    config.ensure_directories()
    
    console.print("\n[bold magenta]━━━ Initializing Agentic RAG Enterprise Pipeline ━━━[/bold magenta]")
    
    components = {}
    
    # 1. Security Stack
    console.print("[cyan][1/8][/cyan] 🔒 Security Stack (PII Masking, Prompt Injection Defense, RBAC, Guardrails)...")
    guardrails = create_security_stack(config)
    components["guardrails"] = guardrails
    
    # Register a default user with RBAC
    guardrails.rbac_manager.register_user(UserProfile(
        user_id="admin",
        username="admin",
        clearance_level=4,
        department="general",
        roles=["admin", "viewer", "editor"],
    ))
    console.print("  [green]✓[/green] RBAC configured with clearance levels 0-4")
    console.print("  [green]✓[/green] PII Masking: SSN, Credit Card, Email, Phone, IP")
    console.print("  [green]✓[/green] Prompt Injection Defense: 12 detection patterns")
    
    # 2. Ingestion Pipeline
    console.print("[cyan][2/8][/cyan] 📄 Ingestion Pipeline (Parsing, OCR, Multi-Modal, Chunking)...")
    ingestor = create_ingestion_pipeline(config)
    components["ingestor"] = ingestor
    console.print(f"  [green]✓[/green] Chunking: {config.data_engineering.chunking_strategy} "
                  f"(size={config.data_engineering.chunk_size}, overlap={config.data_engineering.chunk_overlap})")
    console.print(f"  [green]✓[/green] Hierarchical Retrieval: parent={config.retrieval.parent_chunk_size}, "
                  f"child={config.retrieval.child_chunk_size}")
    
    # 3. Retrieval Engine
    console.print("[cyan][3/8][/cyan] 🔎 Retrieval Engine (Embeddings, Vector DB, Hybrid Search, Graph RAG)...")
    retrieval_engine = create_retrieval_engine(config)
    components["retrieval_engine"] = retrieval_engine
    
    # Try to load existing vector store
    vs = retrieval_engine["vector_store"]
    if vs.load():
        # Also index BM25
        retrieval_engine["bm25"].index(vs.chunks)
        console.print(f"  [green]✓[/green] Loaded existing vector store: {len(vs.chunks)} chunks")
    else:
        console.print("  [yellow]⚠[/yellow] No existing vector store found. Use 'ingest' to add documents.")
    
    console.print(f"  [green]✓[/green] Hybrid Search: BM25({config.retrieval.bm25_weight}) + "
                  f"Semantic({config.retrieval.semantic_weight})")
    console.print(f"  [green]✓[/green] Re-ranking: {config.retrieval.reranking_model}")
    console.print(f"  [green]✓[/green] Graph RAG: {'Enabled' if config.retrieval.graph_rag_enabled else 'Disabled'}")
    
    # Setup semantic router with default routes
    router = retrieval_engine["semantic_router"]
    router.add_route(
        "technical", "Technical and engineering questions",
        ["How does this algorithm work?", "Explain the architecture", "What API should I use?"]
    )
    router.add_route(
        "security", "Security and privacy questions",
        ["Is this data encrypted?", "What are the security risks?", "How to prevent attacks?"]
    )
    router.add_route(
        "data", "Data analysis and database questions",
        ["Show me the statistics", "Query the database", "What does the data show?"]
    )
    console.print(f"  [green]✓[/green] Semantic Router: 3 routes configured")
    
    # 4. Memory Manager
    console.print("[cyan][4/8][/cyan] 🧠 Memory Manager (Short-Term, Long-Term, Persistence, Time-Travel)...")
    memory_manager = create_memory_manager(config)
    memory_manager.start_session("enterprise-session-1")
    components["memory_manager"] = memory_manager
    console.print(f"  [green]✓[/green] Persistence Layer: SQLite at {config.state_management.persistence_db_path}")
    console.print(f"  [green]✓[/green] Time-Travel / State Rewinding: Enabled")
    
    # 5. LLM Provider
    console.print("[cyan][5/8][/cyan] 🤖 LLM Provider...")
    llm_provider = LLMProvider(
        provider_type=config.llm.provider_type,
        model_name=config.llm.model_name,
        api_key=config.llm.api_key,
        api_base_url=config.llm.api_base_url,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )
    components["llm_provider"] = llm_provider
    
    llm = llm_provider.llm
    is_local = config.llm.provider_type == "ollama"
    console.print(f"  [green]✓[/green] Provider: {config.llm.provider_type.upper()}")
    console.print(f"  [green]✓[/green] Model: {config.llm.model_name}")
    console.print(f"  [green]✓[/green] Air-Gapped: {'Yes' if is_local else 'No (External API - Guardrails Active)'}")
    
    if llm is None:
        console.print("  [yellow]⚠[/yellow] LLM not initialized. System will use context-only responses.")
    
    # 6. Tool Registry
    console.print("[cyan][6/8][/cyan] 🔧 Tool Registry (Web Search, SQL, File Ops, Calculator, Doc Search)...")
    tool_registry = create_tool_registry(
        hybrid_search=retrieval_engine.get("hybrid_search"),
        context_manager=retrieval_engine.get("context_manager"),
    )
    components["tool_registry"] = tool_registry
    console.print(f"  [green]✓[/green] Registered tools: {', '.join(tool_registry.tools.keys())}")
    
    # 7. Evaluation Stack
    console.print("[cyan][7/8][/cyan] 📊 Evaluation Stack (Tracing, RAG Triad, Hallucination, Golden Dataset)...")
    eval_stack = create_evaluation_stack(config, llm)
    components["eval_stack"] = eval_stack
    console.print(f"  [green]✓[/green] Tracing: Local (traces saved to {config.observability.trace_log_dir})")
    console.print(f"  [green]✓[/green] RAG Triad: Context Relevance, Groundedness, Answer Relevance")
    console.print(f"  [green]✓[/green] Hallucination Detection: Enabled")
    console.print(f"  [green]✓[/green] Golden Dataset: {len(eval_stack['golden_dataset'].entries)} benchmark entries")
    
    # 8. Agent Graph
    console.print("[cyan][8/8][/cyan] 🤖 Agent Graph (DAG + Cyclic Self-Correction, ReAct, Multi-Agent)...")
    agent_graph = create_agent_graph(
        llm=llm,
        config=config,
        retrieval_engine=retrieval_engine,
        memory_manager=memory_manager,
        guardrails=guardrails,
        tool_registry=tool_registry,
        eval_stack=eval_stack,
    )
    components["agent_graph"] = agent_graph
    console.print(f"  [green]✓[/green] Graph: route → retrieve → grade ⟲ → tools → generate → evaluate")
    console.print(f"  [green]✓[/green] Self-Correction: Max {config.orchestration.max_self_correction_retries} retries")
    console.print(f"  [green]✓[/green] Multi-Agent Routing: technical, security, data, general")
    console.print(f"  [green]✓[/green] HITL: Enabled for critical actions")
    
    console.print("\n[bold green]━━━ System Ready! ━━━[/bold green]\n")
    
    return components


# ─────────────────────────────────────────────
# Document Ingestion Command
# ─────────────────────────────────────────────

def ingest_documents(components: Dict, config: RAGConfig, path: str = ""):
    """Ingest documents into the RAG pipeline."""
    ingestor = components["ingestor"]
    retrieval_engine = components["retrieval_engine"]
    
    doc_dir = path or config.documents_dir
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir, exist_ok=True)
        console.print(f"[yellow]Created documents directory: {doc_dir}[/yellow]")
        console.print(f"[dim]Place your documents (PDF, TXT, MD, CSV, JSON, HTML, DOCX) in this directory.[/dim]")
        
        # Create a sample document
        sample_path = os.path.join(doc_dir, "sample_rag_guide.md")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_DOCUMENT)
        console.print(f"[green]Created sample document: {sample_path}[/green]")
    
    console.print(f"\n[bold cyan]📄 Ingesting documents from: {doc_dir}[/bold cyan]")
    
    # Ingest all documents
    documents = ingestor.ingest_directory(doc_dir, use_hierarchy=True)
    
    if not documents:
        console.print("[yellow]No documents found to ingest.[/yellow]")
        return
    
    # Add to retrieval engine
    vector_store = retrieval_engine["vector_store"]
    bm25 = retrieval_engine["bm25"]
    knowledge_graph = retrieval_engine["knowledge_graph"]
    
    all_chunks = []
    all_parent_chunks = []
    
    for doc in documents:
        all_chunks.extend(doc.chunks)
        all_parent_chunks.extend(doc.parent_chunks)
        
        # Extract entities for Graph RAG
        for chunk in doc.chunks:
            knowledge_graph.extract_entities_from_text(chunk.content, chunk.chunk_id)
    
    # Index in vector store
    vector_store.add_chunks(all_chunks, all_parent_chunks)
    vector_store.save()
    
    # Index in BM25
    bm25.index(vector_store.chunks)
    
    # Display summary
    if HAS_RICH:
        table = Table(title="Ingestion Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents Processed", str(len(documents)))
        table.add_row("Total Chunks (Child)", str(len(all_chunks)))
        table.add_row("Parent Chunks", str(len(all_parent_chunks)))
        table.add_row("Graph Entities", str(len(knowledge_graph.entities)))
        table.add_row("Graph Relations", str(len(knowledge_graph.relations)))
        table.add_row("Vector Store Size", str(len(vector_store.chunks)))
        
        console.print(table)
    else:
        console.print(f"  Documents: {len(documents)}")
        console.print(f"  Chunks: {len(all_chunks)}")
        console.print(f"  Parent Chunks: {len(all_parent_chunks)}")
        console.print(f"  Graph Entities: {len(knowledge_graph.entities)}")


# ─────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────

def display_result(result: Dict):
    """Display the agent's response with evaluation metrics."""
    
    # Answer
    console.print(f"\n[bold green]🤖 Agentic RAG Response:[/bold green]")
    console.print(f"\n{result['answer']}\n")
    
    # Evaluation scores
    scores = result.get("evaluation_scores", {})
    if scores and HAS_RICH:
        table = Table(title="RAG Triad Evaluation", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Status", style="bold")
        
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                status = "[green]✓ PASS[/green]" if value >= 0.6 else "[red]✗ FAIL[/red]"
                table.add_row(key.replace("_", " ").title(), f"{value:.3f}", status)
            elif isinstance(value, bool):
                status = "[red]⚠ YES[/red]" if value else "[green]✓ NO[/green]"
                table.add_row(key.replace("_", " ").title(), str(value), status)
        
        console.print(table)
    
    # Pipeline metadata
    if HAS_RICH:
        meta_parts = []
        if result.get("route"):
            meta_parts.append(f"Route: [cyan]{result['route']}[/cyan]")
        if result.get("latency_ms"):
            meta_parts.append(f"Latency: [yellow]{result['latency_ms']:.0f}ms[/yellow]")
        if result.get("total_tokens"):
            meta_parts.append(f"Tokens: [magenta]{result['total_tokens']}[/magenta]")
        if result.get("cost_usd"):
            meta_parts.append(f"Cost: [green]${result['cost_usd']:.6f}[/green]")
        if result.get("retries"):
            meta_parts.append(f"Retries: [red]{result['retries']}[/red]")
        if result.get("trace_id"):
            meta_parts.append(f"Trace: [dim]{result['trace_id']}[/dim]")
        
        if meta_parts:
            console.print(f"[dim]{'  |  '.join(meta_parts)}[/dim]")
    
    # Steps
    steps = result.get("steps", [])
    if steps:
        step_names = " → ".join(s.get("step", "?") for s in steps)
        console.print(f"[dim]Pipeline: {step_names}[/dim]")


# ─────────────────────────────────────────────
# System Status Command
# ─────────────────────────────────────────────

def show_system_status(components: Dict, config: RAGConfig):
    """Display comprehensive system status."""
    
    console.print("\n[bold cyan]━━━ System Status Dashboard ━━━[/bold cyan]\n")
    
    if HAS_RICH:
        # LLM Info
        llm_info = components["llm_provider"].get_info()
        table = Table(title="LLM Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        for k, v in llm_info.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
        
        # Retrieval Engine
        vs = components["retrieval_engine"]["vector_store"]
        kg = components["retrieval_engine"]["knowledge_graph"]
        table = Table(title="Retrieval Engine", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_row("Vector Store Chunks", str(len(vs.chunks)))
        table.add_row("Parent Chunks", str(len(vs.parent_chunks)))
        table.add_row("BM25 Indexed", str(components["retrieval_engine"]["bm25"].N))
        table.add_row("Graph Entities", str(len(kg.entities)))
        table.add_row("Graph Relations", str(len(kg.relations)))
        table.add_row("Semantic Router Routes", str(len(components["retrieval_engine"]["semantic_router"].routes)))
        console.print(table)
        
        # Observability
        summary = components["eval_stack"]["tracing_engine"].get_summary()
        table = Table(title="Observability Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in summary.items():
            if isinstance(v, float):
                table.add_row(k.replace("_", " ").title(), f"{v:.2f}")
            else:
                table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
        
        # Tools
        table = Table(title="Registered Tools", box=box.ROUNDED)
        table.add_column("Tool", style="cyan")
        table.add_column("HITL Required", style="yellow")
        for name, tool in components["tool_registry"].tools.items():
            table.add_row(name, "Yes" if tool.requires_hitl else "No")
        console.print(table)
        
        # Security
        table = Table(title="Security Configuration", box=box.ROUNDED)
        table.add_column("Feature", style="cyan")
        table.add_column("Status", style="green")
        table.add_row("PII Masking", "✓ Active")
        table.add_row("Prompt Injection Defense", "✓ Active")
        table.add_row("RBAC", f"✓ Active ({len(components['guardrails'].rbac_manager.users)} users)")
        table.add_row("Air-Gapped Mode", "✓ Yes" if config.llm.provider_type == "ollama" else "✗ No (External API)")
        table.add_row("Output Guardrails", "✓ Active")
        console.print(table)


# ─────────────────────────────────────────────
# Benchmark Command
# ─────────────────────────────────────────────

def run_benchmark(components: Dict):
    """Run the golden dataset benchmark."""
    console.print("\n[bold cyan]📊 Running Golden Dataset Benchmark...[/bold cyan]\n")
    
    golden = components["eval_stack"]["golden_dataset"]
    evaluator = components["eval_stack"]["rag_evaluator"]
    
    results = golden.run_benchmark(evaluator)
    
    if HAS_RICH:
        table = Table(title="Benchmark Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        
        for metric, score in results.get("average_scores", {}).items():
            table.add_row(metric.replace("_", " ").title(), f"{score:.3f}")
        
        table.add_row("Total Test Entries", str(results.get("total_entries", 0)))
        console.print(table)
    else:
        console.print(f"Results: {json.dumps(results.get('average_scores', {}), indent=2)}")


# ─────────────────────────────────────────────
# Help Command
# ─────────────────────────────────────────────

def show_help():
    """Display help information."""
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [green]<query>[/green]       Ask any question - the full Agentic RAG pipeline will process it
  [green]ingest[/green]        Ingest documents from the documents directory
  [green]ingest <path>[/green] Ingest documents from a specific path
  [green]status[/green]        Show system status dashboard (Observability)
  [green]benchmark[/green]     Run Golden Dataset benchmark
  [green]history[/green]       Show conversation history (Short-Term Memory)
  [green]tools[/green]         List available tools (Function Calling)
  [green]security[/green]      Show security configuration
  [green]traces[/green]        Show recent execution traces
  [green]help[/green]          Show this help message
  [green]quit[/green]          Exit the application

[bold cyan]Architecture Components:[/bold cyan]

  🤖 Agents:     ReAct, Multi-Agent Routing, Self-Correction, HITL
  🧠 Memory:     Short-Term, Long-Term, Persistence, Time-Travel
  📄 Ingestion:  Parsing, OCR, Chunking, Hierarchical Retrieval
  🔎 Retrieval:  Hybrid Search, Re-ranking, Graph RAG, Semantic Router
  ✨ Generation:  CoT Prompting, Context Compression, Streaming
  📊 Evaluation: RAG Triad, Hallucination Detection, Golden Dataset
  🔒 Security:   PII Masking, Prompt Injection, RBAC, Air-Gapped
  📈 Observability: Tracing, Latency, Token Usage, Cost
"""
    console.print(help_text)


# ─────────────────────────────────────────────
# Main Interactive Loop
# ─────────────────────────────────────────────

def main():
    """Main entry point for the Enterprise Agentic RAG system."""
    
    # Step 1: Configure LLM
    llm_config = configure_llm_interactive()
    
    # Step 2: Build config
    config = RAGConfig()
    config.llm = llm_config
    
    # Auto-enable air-gapped mode for local models
    if llm_config.provider_type == "ollama":
        config.security.air_gapped_mode = True
    
    # Step 3: Initialize system
    components = initialize_system(config)
    
    # Step 4: Auto-ingest if documents exist
    if os.path.exists(config.documents_dir) and os.listdir(config.documents_dir):
        vs = components["retrieval_engine"]["vector_store"]
        if not vs.chunks:  # Only auto-ingest if not already loaded
            ingest_documents(components, config)
    
    # Show help
    show_help()
    
    # HITL callback
    def hitl_callback(message: str) -> bool:
        console.print(f"\n[bold yellow]⚠ HITL Alert:[/bold yellow] {message}")
        if HAS_RICH:
            return Confirm.ask("Approve this action?", default=True)
        else:
            resp = input("Approve? (yes/no) [yes]: ").strip().lower()
            return resp != "no"
    
    # Stream callback
    def stream_callback(token: str):
        if HAS_RICH:
            console.print(token, end="", highlight=False)
        else:
            print(token, end="", flush=True)
    
    # Interactive loop
    console.print("[bold green]Type your query to begin (type 'help' for commands).[/bold green]\n")
    
    while True:
        try:
            if HAS_RICH:
                query = Prompt.ask("\n[bold cyan]User Query[/bold cyan]")
            else:
                query = input("\nUser Query: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            query_lower = query.lower().strip()
            
            if query_lower in ["quit", "exit", "q"]:
                console.print("[bold yellow]Shutting down... Saving state.[/bold yellow]")
                # Save vector store
                components["retrieval_engine"]["vector_store"].save()
                components["memory_manager"].long_term.save()
                console.print("[green]State saved. Goodbye![/green]")
                break
            
            elif query_lower == "help":
                show_help()
                continue
            
            elif query_lower == "status":
                show_system_status(components, config)
                continue
            
            elif query_lower.startswith("ingest"):
                path = query[6:].strip() if len(query) > 6 else ""
                ingest_documents(components, config, path)
                continue
            
            elif query_lower == "benchmark":
                run_benchmark(components)
                continue
            
            elif query_lower == "history":
                history = components["memory_manager"].short_term.get_context(last_n=20)
                if history:
                    for turn in history:
                        role = turn["role"].upper()
                        content = turn["content"][:200]
                        console.print(f"  [{role}] {content}")
                else:
                    console.print("[dim]No conversation history yet.[/dim]")
                continue
            
            elif query_lower == "tools":  # command: 'tools'
                console.print(components["tool_registry"].get_tools_description())
                continue
            
            elif query_lower == "security":  # command: 'security'
                guardrails = components["guardrails"]
                console.print(f"\n[bold]PII Patterns:[/bold] {len(guardrails.pii_masker.patterns)}")
                console.print(f"[bold]Injection Patterns:[/bold] {len(guardrails.injection_defense.patterns)}")
                console.print(f"[bold]RBAC Users:[/bold] {len(guardrails.rbac_manager.users)}")
                console.print(f"[bold]Critical Actions:[/bold] {', '.join(guardrails.CRITICAL_ACTIONS)}")
                continue
            
            elif query_lower == "traces":  # command: 'traces'
                traces = components["eval_stack"]["tracing_engine"].traces
                if traces:
                    for trace in traces[-5:]:
                        console.print(
                            f"  [{trace.trace_id}] {trace.query[:50]}... | "
                            f"{trace.total_latency_ms:.0f}ms | "
                            f"{trace.total_input_tokens + trace.total_output_tokens} tokens"
                        )
                else:
                    console.print("[dim]No traces yet.[/dim]")
                continue
            
            # ─── Process Query Through Agent Pipeline ───
            console.print("\n[dim italic]🔄 Agentic Pipeline Executing...[/dim italic]")
            
            result = components["agent_graph"].run(
                query=query,
                user_id="default",
                hitl_callback=hitl_callback,
                stream_callback=stream_callback if config.generation.streaming_outputs else None,
            )
            
            display_result(result)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            logger.exception("Unexpected error")
            continue


# ─────────────────────────────────────────────
# Sample Document
# ─────────────────────────────────────────────

SAMPLE_DOCUMENT = """# Comprehensive Guide to Agentic RAG Architecture

## 1. What is Retrieval Augmented Generation (RAG)?

Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) 
by providing them with external knowledge retrieved from a vector database. Instead of relying 
solely on the model's training data, RAG retrieves relevant documents and injects them into the 
prompt context, enabling more accurate, up-to-date, and grounded responses.

## 2. Agentic RAG vs Traditional RAG

Traditional RAG follows a simple retrieve-then-generate pattern. Agentic RAG introduces 
autonomous agents that can reason about the quality of retrieved context, decide which tools 
to use, and self-correct through iterative loops.

Key differences:
- **Autonomous Decision-Making**: Agents choose retrieval strategies dynamically
- **Self-Correction**: If retrieved context is poor, the agent retries with refined queries
- **Tool Use**: Agents can execute external APIs, SQL queries, and web searches
- **Multi-Agent Routing**: Specialized agents handle domain-specific queries

## 3. Hybrid Search: Combining BM25 and Semantic Search

Hybrid Search combines two complementary retrieval methods:

1. **BM25 (Keyword Search)**: Uses term frequency and inverse document frequency to find 
   documents containing exact query keywords. Excellent for specific terms and names.

2. **Semantic Search (Dense Vector Search)**: Uses embeddings to find documents with similar 
   meaning, even if they don't share exact keywords. Powered by models like sentence-transformers.

The Ensemble Retriever combines both with configurable weights (e.g., 40% BM25, 60% semantic).

## 4. The RAG Triad: Evaluation Framework

The RAG Triad measures three critical aspects:

1. **Context Relevance**: Do the retrieved documents contain the information needed?
2. **Groundedness (Faithfulness)**: Is the generated answer supported by the context?
3. **Answer Relevance**: Does the answer actually address the user's question?

Frameworks like Ragas and TruLens automate these evaluations using LLM-as-a-Judge.

## 5. Security Considerations

Enterprise RAG systems must implement:
- **PII Masking**: Remove personal data before sending to external APIs
- **Prompt Injection Defense**: Detect and block malicious prompts
- **RBAC**: Restrict document access based on user clearance levels
- **Air-Gapped Deployment**: Run entirely on-premise with local models (Ollama)

## 6. Knowledge Graphs and Graph RAG

Graph RAG enriches traditional vector search by adding entity-relationship awareness.
When a query mentions entities, the system can traverse knowledge graph connections to 
find highly interconnected and contextually relevant information that pure vector 
similarity might miss.

## 7. Chain-of-Thought Prompting

Chain-of-Thought (CoT) prompting instructs the LLM to reason step-by-step before 
generating a final answer. This improves accuracy on complex questions by making 
the reasoning process explicit and auditable.

## 8. Chunking Strategies

Documents must be broken into chunks for indexing. Strategies include:
- **Fixed-Size Chunking**: Split by character count with overlap
- **Recursive Chunking**: Split by natural separators (paragraphs, sentences)
- **Semantic Chunking**: Split at semantic boundaries where topic shifts

Hierarchical Retrieval uses small child chunks for precise search, but retrieves 
the larger parent document for generation context.

## 9. Observability and Tracing

Production RAG systems need comprehensive observability:
- **Latency tracking**: Per-step and end-to-end response times
- **Token usage**: Input and output token counts per query
- **Cost tracking**: Estimated API costs per query
- **Execution traces**: Full audit trail of agent decisions

## 10. Self-Reflection and Self-Correction

The most advanced RAG agents evaluate their own outputs:
1. Retrieve context for the query
2. Grade the context quality (LLM-as-a-Judge)
3. If quality is low, refine the query and retry (Cyclic Graph)
4. Generate the response only when context is sufficient
5. Evaluate the response against the RAG Triad

This creates a robust, self-improving pipeline that catches and corrects errors.
"""


if __name__ == "__main__":
    main()
