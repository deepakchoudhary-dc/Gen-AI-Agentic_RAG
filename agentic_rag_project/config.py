"""
Enterprise Agentic RAG Configuration
Covers all architectural parameters:
  - Agents & Orchestration
  - Advanced State Management & Graph Architecture
  - Deep Data Engineering & Ingestion Pipeline
  - Retrieval & Memory Engine
  - Advanced Generation & Post-Retrieval Tactics
  - LLMOps & Observability
  - Security, Privacy, & Enterprise Guardrails
  - Advanced Evaluation Frameworks (RAG Triad)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class AgentOrchestrationConfig:
    """1. Agents & Orchestration"""
    agent_framework: str = "LangGraph"              # Agentic Framework
    autonomous_mode: bool = True                      # Autonomous Agents
    react_enabled: bool = True                        # Reason + Act (ReAct) pattern
    tool_use_enabled: bool = True                     # Tool Use / Function Calling
    hitl_enabled: bool = True                         # Human-in-the-Loop (HITL)
    multi_agent_routing: bool = True                  # Multi-Agent Routing
    self_reflection_enabled: bool = True              # Self-Reflection / Self-Correction
    max_self_correction_retries: int = 3              # Max retries for self-correction loops


@dataclass
class StateManagementConfig:
    """2. Advanced State Management & Graph Architecture"""
    use_dag: bool = True                              # Directed Acyclic Graph (DAG)
    allow_cyclic: bool = True                         # Cyclic Graphs (allow loops)
    short_term_memory: bool = True                    # Short-Term Memory (conversation)
    long_term_memory: bool = True                     # Long-Term Memory (user profiles)
    persistence_backend: str = "sqlite"               # Persistence Layer (sqlite/redis/postgres)
    persistence_db_path: str = "data/agent_state.db"  # DB file path
    time_travel_enabled: bool = True                  # Time-Travel / State Rewinding


@dataclass
class DataEngineeringConfig:
    """3. Deep Data Engineering & Ingestion Pipeline"""
    # Data Parsing
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "txt", "md", "csv", "docx", "html", "json"
    ])
    ocr_enabled: bool = True                          # OCR (Optical Character Recognition)
    multi_modal_ingestion: bool = True                # Multi-Modal Ingestion
    
    # Chunking Strategy
    chunk_size: int = 1000                            # Fixed-Size Chunking
    chunk_overlap: int = 200                          # Chunk Overlap
    chunking_strategy: str = "recursive"              # recursive / semantic / fixed
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"        # Sentence-transformer model
    embedding_dimension: int = 384                    # Embedding dimension
    vector_quantization: bool = False                 # Vector Quantization (compression)
    embedding_fine_tuning: bool = False               # Embedding Fine-Tuning


@dataclass
class RetrievalConfig:
    """4. The Retrieval & Memory Engine"""
    vector_db: str = "chroma"                         # Vector Database
    vector_db_persist_dir: str = "data/vectordb"      # Persist directory
    
    # Search
    use_hybrid_search: bool = True                    # Hybrid Search (BM25 + Vector)
    bm25_weight: float = 0.4                          # BM25 keyword weight
    semantic_weight: float = 0.6                      # Semantic vector weight
    top_k: int = 5                                    # Number of results to retrieve
    
    # Advanced Retrieval
    hierarchical_retrieval: bool = True               # Hierarchical / Parent-Document Retrieval
    parent_chunk_size: int = 2000                     # Parent document chunk size
    child_chunk_size: int = 400                       # Child chunk size
    
    reranking_enabled: bool = True                    # Re-ranking
    reranking_model: str = "cross-encoder"            # Re-ranking model type
    
    metadata_filtering: bool = True                   # Metadata Filtering
    semantic_router_enabled: bool = True              # Semantic Router
    
    context_window_limit: int = 4096                  # Context Window limit
    graph_rag_enabled: bool = True                    # Graph RAG


@dataclass
class GenerationConfig:
    """5. Advanced Generation & Post-Retrieval Tactics"""
    context_compression: bool = True                  # Context Compression / Prompt Compression
    lost_in_middle_mitigation: bool = True            # Lost in the Middle Mitigation
    cot_prompting: bool = True                        # Chain-of-Thought (CoT) Prompting
    speculative_decoding: bool = False                # Speculative Decoding
    streaming_outputs: bool = True                    # Streaming Outputs
    temperature: float = 0.2                          # LLM temperature
    max_tokens: int = 2048                            # Max output tokens


@dataclass
class ObservabilityConfig:
    """6. LLMOps & Observability"""
    tracing_enabled: bool = True                      # Observability / Tracing
    tracing_backend: str = "local"                    # local / langsmith / arize
    llm_as_judge_enabled: bool = True                 # LLM-as-a-Judge
    golden_dataset_path: str = "data/golden_dataset.json"  # Golden Dataset
    log_latency: bool = True                          # Latency tracking
    log_token_usage: bool = True                      # Token usage tracking
    log_cost: bool = True                             # Cost tracking
    hallucination_detection: bool = True              # Hallucination detection
    trace_log_dir: str = "data/traces"                # Trace log directory


@dataclass
class SecurityConfig:
    """7. Security, Privacy, & Enterprise Guardrails"""
    pii_masking_enabled: bool = True                  # Data Anonymization / PII Masking
    prompt_injection_defense: bool = True             # Prompt Injection Defense
    rbac_enabled: bool = True                         # Role-Based Access Control (RBAC)
    air_gapped_mode: bool = False                     # Air-Gapped Deployment
    guardrails_enabled: bool = True                   # Guardrails
    
    # PII patterns to mask
    pii_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{3}-\d{2}-\d{4}\b',                    # SSN
        r'\b\d{16}\b',                                # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',            # Phone
    ])
    
    # Prompt injection patterns to block
    injection_patterns: List[str] = field(default_factory=lambda: [
        r'ignore\s+(previous|above|all)\s+instructions',
        r'system\s*prompt',
        r'you\s+are\s+now',
        r'forget\s+(everything|all)',
        r'reveal\s+(your|the)\s+(system|internal)',
    ])
    
    # RBAC clearance levels
    clearance_levels: Dict[str, int] = field(default_factory=lambda: {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
        "top_secret": 4,
    })


@dataclass
class EvaluationConfig:
    """8. Advanced Evaluation Frameworks (RAG Triad)"""
    eval_framework: str = "builtin"                   # builtin / ragas / trulens
    context_relevance_threshold: float = 0.7          # Context Relevance threshold
    groundedness_threshold: float = 0.7               # Groundedness / Faithfulness threshold
    answer_relevance_threshold: float = 0.7           # Answer Relevance threshold
    auto_eval_enabled: bool = True                    # Auto-evaluate every response


@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    provider_type: str = "ollama"                     # ollama / openai / anthropic / other
    model_name: str = ""                              # Model name
    api_key: str = ""                                 # API key for external providers
    api_base_url: str = ""                            # Custom API base URL
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class RAGConfig:
    """Master Configuration aggregating all sub-configs"""
    orchestration: AgentOrchestrationConfig = field(default_factory=AgentOrchestrationConfig)
    state_management: StateManagementConfig = field(default_factory=StateManagementConfig)
    data_engineering: DataEngineeringConfig = field(default_factory=DataEngineeringConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Data directories
    data_dir: str = "data"
    documents_dir: str = "data/documents"
    
    def ensure_directories(self):
        """Create necessary data directories."""
        dirs = [
            self.data_dir,
            self.documents_dir,
            self.retrieval.vector_db_persist_dir,
            self.observability.trace_log_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
