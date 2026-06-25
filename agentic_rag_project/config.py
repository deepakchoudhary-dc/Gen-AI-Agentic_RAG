"""Typed configuration for the Agentic RAG project.

The project can run fully local with Ollama, or call an external
OpenAI-compatible/hosted provider when the user opts into it. The config
objects are intentionally explicit so the architecture is visible in code,
tests, and CLI status output.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AgentOrchestrationConfig(BaseModel):
    """Agents & orchestration settings."""

    agent_framework: str = "LangGraph-compatible local graph"
    autonomous_mode: bool = True
    react_enabled: bool = True
    tool_use_enabled: bool = True
    hitl_enabled: bool = True
    multi_agent_routing: bool = True
    self_reflection_enabled: bool = True
    max_retries: int = 2
    specialists: List[str] = Field(
        default_factory=lambda: [
            "retrieval_specialist",
            "security_specialist",
            "analysis_specialist",
            "tool_specialist",
        ]
    )


class StateManagementConfig(BaseModel):
    """Advanced state management and graph architecture settings."""

    use_dag: bool = True
    allow_cyclic: bool = True
    short_term_memory: bool = True
    long_term_memory: bool = True
    persistence_backend: str = "sqlite"
    time_travel_enabled: bool = True
    max_turns: int = 50


class DataEngineeringConfig(BaseModel):
    """Deep ingestion pipeline settings."""

    supported_formats: List[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".html",
            ".htm",
            ".pdf",
            ".docx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
        ]
    )
    ocr_enabled: bool = True
    multi_modal_ingestion: bool = True
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "local-hashing-embedding"
    vector_quantization: bool = False
    embedding_fine_tuning: bool = False


class RetrievalConfig(BaseModel):
    """Retrieval and memory engine settings."""

    vector_db: str = "local-json-vector-store"
    use_hybrid_search: bool = True
    semantic_search: bool = True
    keyword_search: bool = True
    reranking_enabled: bool = True
    metadata_filtering: bool = True
    semantic_router_enabled: bool = True
    graph_rag_enabled: bool = True
    parent_document_retrieval: bool = True
    parent_chunk_size: int = 2000
    child_chunk_size: int = 400
    top_k: int = 5


class GenerationConfig(BaseModel):
    """Advanced generation and post-retrieval settings."""

    context_compression: bool = True
    lost_in_middle_mitigation: bool = True
    cot_prompting: bool = True
    speculative_decoding: bool = False
    streaming_outputs: bool = True
    context_window_tokens: int = 4096
    temperature: float = 0.2
    max_tokens: int = 2048


class SecurityConfig(BaseModel):
    """Security, privacy, and enterprise guardrail settings."""

    pii_masking: bool = True
    prompt_injection_defense: bool = True
    rbac_enabled: bool = True
    guardrails_enabled: bool = True
    air_gapped_mode: bool = True
    require_hitl_for_critical_actions: bool = True
    default_user_id: str = "default"


class EvaluationConfig(BaseModel):
    """RAG evaluation settings."""

    eval_framework: str = "local-ragas-trulens-compatible"
    context_relevance_threshold: float = 0.55
    groundedness_threshold: float = 0.55
    answer_relevance_threshold: float = 0.55
    auto_eval_enabled: bool = True


class ObservabilityConfig(BaseModel):
    """LLMOps observability settings."""

    tracing_enabled: bool = True
    llm_as_judge_enabled: bool = True
    golden_dataset_path: str = "data/golden_dataset.json"
    log_latency: bool = True
    log_token_usage: bool = True
    log_cost: bool = True
    hallucination_detection: bool = True
    trace_dir: str = "data/traces"


class LLMConfig(BaseModel):
    """LLM provider settings."""

    provider_type: str = "ollama"
    model_name: str = "llama3"
    api_key: str = ""
    api_base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    air_gapped_mode: bool = True
    external_provider: str = "custom"
    extra_headers: Dict[str, str] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Top-level project configuration."""

    data_dir: str = "data"
    documents_dir: str = "data/documents"
    agent_orchestration: AgentOrchestrationConfig = Field(
        default_factory=AgentOrchestrationConfig
    )
    state_management: StateManagementConfig = Field(default_factory=StateManagementConfig)
    data_engineering: DataEngineeringConfig = Field(default_factory=DataEngineeringConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @property
    def agent_framework(self) -> str:
        return self.agent_orchestration.agent_framework

    @property
    def hitl_enabled(self) -> bool:
        return self.agent_orchestration.hitl_enabled

    @property
    def use_hybrid_search(self) -> bool:
        return self.retrieval.use_hybrid_search

    @property
    def reranking_enabled(self) -> bool:
        return self.retrieval.reranking_enabled

    @property
    def streaming_outputs(self) -> bool:
        return self.generation.streaming_outputs

    @property
    def air_gapped_mode(self) -> bool:
        return self.security.air_gapped_mode


def merge_llm_selection(config: RAGConfig, selection: Dict[str, str]) -> RAGConfig:
    """Apply interactive LLM selection to a config object."""

    provider = selection.get("type") or selection.get("provider_type") or "ollama"
    if provider == "external":
        provider = selection.get("provider", "custom").lower()

    config.llm.provider_type = provider.lower()
    config.llm.model_name = selection.get("model", selection.get("model_name", config.llm.model_name))
    config.llm.api_key = selection.get("api_key", config.llm.api_key)
    config.llm.api_base_url = selection.get("api_base_url", config.llm.api_base_url)
    config.llm.external_provider = selection.get("provider", config.llm.external_provider).lower()
    config.llm.air_gapped_mode = config.llm.provider_type == "ollama"
    config.security.air_gapped_mode = config.llm.air_gapped_mode
    return config
