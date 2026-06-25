"""Streamlit web interface for the Enterprise Agentic RAG project."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import LLMConfig, RAGConfig
from eval import GoldenDataset, RAGTriadEvaluator
from llm_provider import detect_ollama_models
from main import build_runtime, ingest_documents
from retrieval import ContextWindowManager
from tools import create_tool_registry


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "data" / "web_uploads"
SUPPORTED_UPLOAD_TYPES = [
    "txt",
    "md",
    "csv",
    "json",
    "html",
    "htm",
    "pdf",
    "docx",
    "png",
    "jpg",
    "jpeg",
    "tiff",
    "bmp",
]


def init_session() -> None:
    st.session_state.setdefault("config", RAGConfig())
    st.session_state.setdefault("runtime", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("uploaded_paths", [])
    st.session_state.setdefault("last_build_key", "")


def provider_config_panel(config: RAGConfig) -> RAGConfig:
    st.sidebar.header("Model")
    provider_label = st.sidebar.radio(
        "Provider",
        [
            "Ollama local",
            "Google Gemini",
            "OpenAI",
            "Anthropic",
            "Custom OpenAI-compatible",
            "Retrieval-only fallback",
        ],
        index=0,
    )

    if provider_label == "Ollama local":
        models = detect_ollama_models()
        model_names = [model["name"] for model in models if model.get("name")]
        if model_names:
            model_name = st.sidebar.selectbox("Downloaded Ollama model", model_names)
            st.sidebar.caption("Detected from Ollama on this machine.")
        else:
            model_name = st.sidebar.text_input("Ollama model", value=config.llm.model_name or "llama3")
            st.sidebar.warning("No downloaded Ollama models were detected. You can still type a model name.")
        config.llm = LLMConfig(provider_type="ollama", model_name=model_name, air_gapped_mode=True)
        config.security.air_gapped_mode = True
        return config

    if provider_label == "Retrieval-only fallback":
        config.llm = LLMConfig(provider_type="none", model_name="retrieval-only", air_gapped_mode=False)
        config.security.air_gapped_mode = False
        return config

    provider_map = {
        "Google Gemini": ("gemini", "gemini-1.5-flash", "Google Gemini API key"),
        "OpenAI": ("openai", "gpt-4o-mini", "OpenAI API key"),
        "Anthropic": ("anthropic", "claude-3-5-sonnet-latest", "Anthropic API key"),
        "Custom OpenAI-compatible": ("custom", "local-openai-compatible-model", "API key"),
    }
    provider_type, default_model, key_label = provider_map[provider_label]
    model_name = st.sidebar.text_input("Model", value=default_model)
    api_key = st.sidebar.text_input(key_label, type="password")
    api_base_url = ""
    if provider_type == "custom":
        api_base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000/v1")

    config.llm = LLMConfig(
        provider_type=provider_type,
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
        air_gapped_mode=False,
    )
    config.security.air_gapped_mode = False
    return config


def retrieval_config_panel(config: RAGConfig) -> RAGConfig:
    st.sidebar.header("Retrieval")
    config.retrieval.top_k = st.sidebar.slider("Top K", min_value=1, max_value=12, value=config.retrieval.top_k)
    config.generation.context_window_tokens = st.sidebar.slider(
        "Context window tokens",
        min_value=1024,
        max_value=16384,
        value=config.generation.context_window_tokens,
        step=512,
    )
    config.data_engineering.chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=300,
        max_value=2500,
        value=config.data_engineering.chunk_size,
        step=100,
    )
    config.data_engineering.chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=min(config.data_engineering.chunk_overlap, 500),
        step=25,
    )
    return config


def runtime_key(config: RAGConfig) -> str:
    values = [
        config.llm.provider_type,
        config.llm.model_name,
        config.llm.api_base_url,
        str(bool(config.llm.api_key)),
        str(config.retrieval.top_k),
        str(config.generation.context_window_tokens),
        str(len(st.session_state.get("uploaded_paths", []))),
    ]
    return "|".join(values)


def save_uploads(files: List[Any]) -> List[str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded in files:
        safe_name = Path(uploaded.name).name
        target = UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
        with open(target, "wb") as handle:
            handle.write(uploaded.getbuffer())
        saved_paths.append(str(target))
    st.session_state.uploaded_paths.extend(saved_paths)
    return saved_paths


def ensure_runtime(config: RAGConfig, force: bool = False) -> Optional[Dict[str, Any]]:
    key = runtime_key(config)
    if force or st.session_state.runtime is None or st.session_state.last_build_key != key:
        with st.spinner("Building Agentic RAG runtime..."):
            try:
                st.session_state.runtime = build_runtime(config)
                st.session_state.last_build_key = key
            except Exception as exc:
                st.error(f"Runtime build failed: {exc}")
                return None
    return st.session_state.runtime


def ingest_uploaded_documents(config: RAGConfig, runtime: Optional[Dict[str, Any]]) -> None:
    if not st.session_state.uploaded_paths:
        st.info("Upload files first, then ingest them.")
        return
    with st.spinner("Ingesting uploaded files with parsing, OCR fallback, chunking, hybrid search, and Graph RAG..."):
        upload_dir = str(UPLOAD_DIR)
        hybrid_search, graph, documents = ingest_documents(config, upload_dir)
        if runtime is not None:
            runtime["hybrid_search"] = hybrid_search
            runtime["knowledge_graph"] = graph
            runtime["documents"] = documents
            runtime["app"].hybrid_search = hybrid_search
            runtime["app"].knowledge_graph = graph
            context_manager = ContextWindowManager(config.generation.context_window_tokens)
            runtime["tools"] = create_tool_registry(
                hybrid_search=hybrid_search,
                context_manager=context_manager,
            )
            runtime["app"].tool_registry = runtime["tools"]
        st.success(f"Ingested {len(documents)} document(s) from uploads.")


def render_status(runtime: Optional[Dict[str, Any]], config: RAGConfig) -> None:
    st.subheader("System Status")
    if runtime is None:
        st.info("Runtime has not been built yet.")
        return
    summary = runtime["tracing"].get_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provider", config.llm.provider_type)
    col2.metric("Model", config.llm.model_name)
    col3.metric("Indexed docs", len(runtime.get("documents", [])))
    col4.metric("Traces", summary.get("total_queries", 0))
    st.json(
        {
            "air_gapped_mode": config.security.air_gapped_mode,
            "retrieval": {
                "hybrid_search": True,
                "top_k": config.retrieval.top_k,
                "context_window_tokens": config.generation.context_window_tokens,
            },
            "observability": summary,
        }
    )


def render_chat(runtime: Optional[Dict[str, Any]]) -> None:
    st.subheader("Ask The Agent")
    if runtime is None:
        st.warning("Build the runtime first.")
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("scores"):
                st.caption(f"RAG triad: {message['scores']}")

    prompt = st.chat_input("Ask a question about your uploaded and indexed documents")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Routing, retrieving, grading, generating, and evaluating..."):
            state = runtime["app"].run(prompt)
        placeholder.markdown(state.answer)
        st.caption(f"Route: {state.route} | relevance: {state.relevance_score} | scores: {state.eval_scores}")
        with st.expander("Retrieved context"):
            st.text(state.context or "No context retrieved.")
        if state.tool_results:
            with st.expander("Tool results"):
                st.json(state.tool_results)
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": state.answer,
            "scores": state.eval_scores,
        }
    )


def render_operations(runtime: Optional[Dict[str, Any]], config: RAGConfig) -> None:
    st.subheader("Operations")
    left, right = st.columns(2)
    with left:
        if st.button("Run golden dataset benchmark", use_container_width=True):
            dataset = GoldenDataset(config.observability.golden_dataset_path)
            report = dataset.run_benchmark(runtime["evaluator"] if runtime else RAGTriadEvaluator())
            st.json(report["average_scores"])
    with right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if runtime:
        with st.expander("Tools"):
            st.text(runtime["tools"].get_tools_description())
        with st.expander("Security"):
            st.json(
                {
                    "rbac_filter": runtime["guardrails"].rbac_manager.get_metadata_filter("default"),
                    "critical_hitl_actions": runtime["guardrails"].CRITICAL_ACTIONS,
                    "pii_masking": True,
                    "prompt_injection_defense": True,
                }
            )
        with st.expander("Traces"):
            st.json(runtime["tracing"].get_summary())


def main() -> None:
    st.set_page_config(page_title="Enterprise Agentic RAG", layout="wide")
    init_session()

    st.title("Enterprise Agentic RAG")
    st.caption("Agentic orchestration, hybrid retrieval, Graph RAG, guardrails, observability, and Ollama/Gemini/API model support.")

    config = st.session_state.config
    config = provider_config_panel(config)
    config = retrieval_config_panel(config)
    st.session_state.config = config

    st.sidebar.header("Runtime")
    force_rebuild = st.sidebar.button("Build / Rebuild Runtime", use_container_width=True)
    runtime = ensure_runtime(config, force=force_rebuild)

    st.sidebar.header("Uploads")
    uploads = st.sidebar.file_uploader(
        "Upload documents, tables, PDFs, DOCX, or images",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    if uploads and st.sidebar.button("Save uploads", use_container_width=True):
        saved = save_uploads(uploads)
        st.sidebar.success(f"Saved {len(saved)} file(s).")

    if st.sidebar.button("Ingest saved uploads", use_container_width=True):
        runtime = ensure_runtime(config)
        ingest_uploaded_documents(config, runtime)

    tab_chat, tab_data, tab_ops = st.tabs(["Chat", "Data & Status", "Ops"])
    with tab_chat:
        render_chat(runtime)
    with tab_data:
        render_status(runtime, config)
        st.subheader("Saved Uploads")
        if st.session_state.uploaded_paths:
            st.dataframe([{"file": path} for path in st.session_state.uploaded_paths], use_container_width=True)
        else:
            st.info("No files uploaded yet.")
    with tab_ops:
        render_operations(runtime, config)


if __name__ == "__main__":
    main()
