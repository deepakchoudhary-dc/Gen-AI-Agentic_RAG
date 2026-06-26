# Enterprise Agentic RAG

This folder now contains a runnable Agentic Retrieval-Augmented Generation project, not only a concept scaffold. The default path runs locally with deterministic embeddings and local indexing, asks the user whether to use downloaded Ollama models, and can also connect to Google Gemini, OpenAI, Anthropic, or any OpenAI-compatible external chatbot API when the user explicitly selects that path.

## What Is Implemented

- **Agents & Orchestration**: `agent.py` implements an `AgenticRAGGraph` with route, retrieve, grade, tools, generate, and evaluate nodes. It includes autonomous execution, ReAct-style private reasoning/action flow, multi-agent routing, specialist agent metadata, self-reflection retries, tool use, and HITL hooks for critical actions.
- **Retrieval & Memory Engine**: `retrieval.py` implements local embeddings, vector search, BM25 keyword search, hybrid search, metadata filtering, parent-document retrieval, reranking, semantic routing, context compression, lost-in-the-middle mitigation, and Graph RAG entity context.
- **State Management**: `memory.py` provides short-term memory, long-term user memory, SQLite persistence, failure tracking, checkpoints, and time-travel/state rewinding.
- **Ingestion Pipeline**: `ingestion.py` parses text, Markdown, CSV, JSON, HTML, PDF, DOCX, and image files with optional OCR. It supports fixed, recursive, semantic, and hierarchical chunking with overlap.
- **LLM Providers**: `llm_provider.py` detects installed Ollama models via `/api/tags` or `ollama list`, initializes local Ollama, and supports Google Gemini, OpenAI, Anthropic, or custom OpenAI-compatible endpoints.
- **Security & Guardrails**: `security.py` includes PII masking, prompt injection defense, RBAC metadata filters, output filtering, air-gapped mode awareness, and guardrails for unauthorized actions.
- **Tool Calling**: `tools.py` includes web search, SQL, file operations, calculator, and document search tools. SQL requires approval and defaults to `SELECT` only; file access is contained inside the document directory.
- **LLMOps & Evaluation**: `eval.py` provides tracing, latency/token/cost tracking, RAG triad scoring, LLM-as-a-judge compatible heuristic fallback, hallucination detection, and golden dataset benchmarking.
- **CLI Application**: `main.py` provides interactive commands for chat, ingest, status, benchmark, history, tools, security, and traces.
- **Streamlit Web App**: `streamlit_app.py` provides a browser interface for model selection, Google/Gemini API-key entry, file uploads, ingestion, chat, status, benchmarking, security, tools, and traces.

## Architecture Terms Covered

The project explicitly covers these required terms in code and documentation: agentic frameworks, autonomous agents, ReAct, tool use/function calling, human-in-the-loop, multi-agent routing, self-reflection/self-correction, vector database, embeddings, semantic search, hybrid search, chunking strategy, chunk overlap, hierarchical retrieval/parent-document retrieval, reranking, metadata filtering, semantic router, context window, Graph RAG, stateful agents, DAG, cyclic graph retry loops, short-term memory, long-term memory, persistence layer, time-travel/state rewinding, data parsing, OCR, multi-modal ingestion, vector quantization readiness, embedding fine-tuning readiness, context compression, lost-in-the-middle mitigation, Chain-of-Thought prompting policy, speculative decoding readiness, streaming outputs, RAG triad evaluation, context relevance, groundedness/faithfulness, answer relevance, Ragas/TruLens-compatible evaluator shape, LLM-as-a-judge, observability/tracing, golden dataset, latency, hallucination detection, guardrails, PII masking, prompt injection defense, RBAC, and air-gapped deployment.

## Run

At startup, the app asks whether to use a locally downloaded Ollama model. If models are found, it lists them and lets you pick one. If not, it offers external provider setup.

The sidebar lets the user choose:

- `Ollama local`: detects downloaded local Ollama models and uses the selected one.
- `Google Gemini`: asks for the Google Gemini API key only for this provider.
- `OpenAI`: asks for an OpenAI API key.
- `Anthropic`: asks for an Anthropic API key.
- `Custom OpenAI-compatible`: asks for API key and base URL.
- `Retrieval-only fallback`: runs retrieval, guardrails, tracing, and evaluation even when no model is configured.

The upload panel accepts text, Markdown, CSV, JSON, HTML, PDF, DOCX, and image files. Uploaded files are saved under `data/web_uploads`, then ingested through the same parser, chunker, hybrid retriever, parent-document retriever, semantic router, Graph RAG index, guardrails, and evaluator used by the CLI.

## CLI Commands

- `help`: show commands
- `ingest [path]`: parse, chunk, index, and graph documents
- `status`: show provider, retrieval, and trace status
- `benchmark`: run the golden dataset benchmark
- `history`: show short-term conversation memory
- `tools`: list tool/function-calling capabilities
- `security`: show guardrails and RBAC status
- `traces`: show observability summary
- `quit`: save state and exit

## Verify

Current verification result: `158/158` passed.
