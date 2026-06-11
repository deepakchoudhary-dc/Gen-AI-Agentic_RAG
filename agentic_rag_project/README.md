#  Enterprise Agentic RAG Architecture

A **100% comprehensive, fully implemented** Agentic Retrieval-Augmented Generation (RAG) system covering every advanced concept in modern AI engineering — from multi-agent orchestration to enterprise-grade security.

> **Supports Ollama (Air-Gapped) and External APIs (OpenAI, Anthropic, Custom)**

---

##  Project Structure

```
agentic_rag_project/
├── main.py              # Entry point - Interactive application
├── config.py            # Master configuration (all architecture params)
├── agent.py             # Agents & Orchestration (DAG/Cyclic Graph, ReAct, Multi-Agent)
├── retrieval.py         # Retrieval Engine (Vector DB, Hybrid Search, Graph RAG)
├── ingestion.py         # Data Engineering (Parsing, OCR, Chunking, Hierarchical)
├── memory.py            # State Management (Short/Long-Term Memory, Persistence)
├── security.py          # Security & Guardrails (PII, Injection, RBAC)
├── eval.py              # LLMOps & Evaluation (Tracing, RAG Triad, Hallucination)
├── llm_provider.py      # LLM Provider (Ollama, OpenAI, Anthropic, Custom)
├── tools.py             # Agent Tools (Web Search, SQL, File Ops, Calculator)
├── requirements.txt     # Dependencies
├── README.md            # This file
└── data/                # Data directory (auto-created)
    ├── documents/       # Source documents for ingestion
    ├── vectordb/        # Vector database storage
    ├── traces/          # Execution traces
    └── golden_dataset.json  #  Benchmark test data
```

---

## Complete Architecture Coverage

### 1. Agents & Orchestration (`agent.py`)
| Concept | Implementation |
|---------|---------------|
| **Agentic Frameworks** | Full LangGraph-style orchestration with custom DAG |
| **Autonomous Agents** | Goal-driven multi-step reasoning pipeline |
| **Reason + Act (ReAct)** | Thought → Action loops in generation node |
| **Tool Use / Function Calling** | 5 tools: Web Search, SQL, File, Calculator, Doc Search |
| **Human-in-the-Loop (HITL)** | Callback-based approval for critical actions |
| **Multi-Agent Routing** | 4 specialist agents: Technical, Security, Data, General |
| **Self-Reflection / Self-Correction** | Grade node with cyclic retry loops (max 3) |

### 2. Advanced State Management & Graph Architecture (`memory.py`)
| Concept | Implementation |
|---------|---------------|
| **Stateful Agents** | `AgentMemoryManager` tracks state across turns |
| **Directed Acyclic Graph (DAG)** | Graph: route → retrieve → grade → generate → evaluate |
| **Cyclic Graphs** | Grade node loops back to retrieve on low scores |
| **Short-Term Memory** | `ShortTermMemory` class with conversation turns |
| **Long-Term Memory** | `LongTermMemory` with user profiles & interaction history |
| **Persistence Layer** | SQLite database for state checkpointing |
| **Time-Travel / State Rewinding** | `PersistenceLayer.time_travel()` method |

### 3. Deep Data Engineering & Ingestion (`ingestion.py`)
| Concept | Implementation |
|---------|---------------|
| **Data Parsing** | Multi-format parser (PDF, TXT, MD, CSV, JSON, HTML, DOCX) |
| **OCR** | `OCRProcessor` with pytesseract integration |
| **Multi-Modal Ingestion** | Text, tables, images pipeline |
| **Chunking Strategy** | 3 strategies: Fixed-Size, Recursive, Semantic |
| **Chunk Overlap** | Configurable overlap (default 200 chars) |
| **Vector Quantization** | Awareness flag in config |
| **Embedding Fine-Tuning** | Awareness flag in config |

### 4. The Retrieval & Memory Engine (`retrieval.py`)
| Concept | Implementation |
|---------|---------------|
| **Vector Database** | Custom `VectorStore` with persistence |
| **Embeddings** | `EmbeddingEngine` (sentence-transformers + TF-IDF fallback) |
| **Semantic Search** | Cosine similarity-based vector search |
| **Hybrid Search** | `HybridSearchEngine` with Reciprocal Rank Fusion |
| **Hierarchical Retrieval** | `HierarchicalChunker` with parent-child linking |
| **Re-ranking** | `ReRanker` with cross-encoder + keyword overlap fallback |
| **Metadata Filtering** | RBAC-based filter generation for vector queries |
| **Semantic Router** | `SemanticRouter` with embedding-based intent routing |
| **Context Window** | `ContextWindowManager` with token estimation |
| **Graph RAG** | `KnowledgeGraph` with entity extraction & BFS traversal |

### 5. Advanced Generation & Post-Retrieval (`agent.py`)
| Concept | Implementation |
|---------|---------------|
| **Context Compression** | `ContextWindowManager.compress_context()` |
| **Lost-in-the-Middle Mitigation** | Strategic reordering in compression |
| **Chain-of-Thought (CoT)** | CoT instructions in system prompt |
| **Speculative Decoding** | Config awareness flag |
| **Streaming Outputs** | `LLMProvider.stream()` with callback |

### 6. LLMOps & Observability (`eval.py`)
| Concept | Implementation |
|---------|---------------|
| **LLM-as-a-Judge** | `RAGTriadEvaluator` with LLM scoring |
| **Observability / Tracing** | `TracingEngine` with per-step metrics |
| **Golden Dataset** | `GoldenDataset` with benchmark runner |
| **Latency** | Per-step and total latency tracking |
| **Hallucination** | `HallucinationDetector` with grounding analysis |
| **Guardrails** | Full input/output filtering pipeline |

### 7. Security, Privacy & Enterprise Guardrails (`security.py`)
| Concept | Implementation |
|---------|---------------|
| **Data Anonymization / PII Masking** | `PIIMasker` with 6 PII patterns |
| **Prompt Injection Defense** | `PromptInjectionDefense` with 12 patterns |
| **Role-Based Access Control (RBAC)** | `RBACManager` with 5 clearance levels |
| **Air-Gapped Deployment** | Full Ollama local model support |

### 8. Advanced Evaluation (RAG Triad) (`eval.py`)
| Concept | Implementation |
|---------|---------------|
| **Context Relevance** | Keyword overlap + LLM-as-a-Judge |
| **Groundedness / Faithfulness** | Token grounding analysis |
| **Answer Relevance** | Query-answer overlap scoring |
| **Frameworks (Ragas/TruLens)** | Compatible interface design |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Setup Flow
The application will interactively guide you through:
1. **LLM Selection** — Choose Ollama (local) or External API
2. **Model Detection** — Auto-detects installed Ollama models
3. **System Initialization** — All 8 components initialized
4. **Document Ingestion** — Auto-creates sample document

### 4. Interactive Commands
| Command | Description |
|---------|-------------|
| `<query>` | Ask any question through the full pipeline |
| `ingest` | Ingest documents from `data/documents/` |
| `ingest <path>` | Ingest from specific path |
| `status` | System status dashboard |
| `benchmark` | Run golden dataset benchmark |
| `history` | Show conversation history |
| `tools` | List available tools |
| `security` | Show security config |
| `traces` | Show execution traces |
| `help` | Show help |
| `quit` | Exit (saves state) |

---

##  LLM Support

### Local Models (Air-Gapped / Ollama)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama pull mistral
ollama pull gemma2
```
The app auto-detects installed models when you select "local."

### External APIs
- **OpenAI**: gpt-4o, gpt-4, gpt-3.5-turbo
- **Anthropic**: claude-3-5-sonnet, claude-3-opus
- **Custom**: Any OpenAI-compatible endpoint (vLLM, LocalAI, etc.)

---

##  Security Features

- **PII Masking**: SSN, Credit Cards, Emails, Phone Numbers, IPs, DOBs
- **Prompt Injection Defense**: 12 detection patterns for common attacks
- **RBAC**: 5 clearance levels (public → top_secret)
- **Output Filtering**: Redacts secrets, API keys from responses
- **Air-Gapped**: Full local deployment with Ollama

---

##  Pipeline Flow

```
User Query
    │
    ▼
┌─────────┐
│  ROUTE  │ ← Semantic Router / Multi-Agent Selection
└────┬────┘
     │
     ▼
┌──────────┐
│ RETRIEVE │ ← Hybrid Search (BM25 + Semantic) + Graph RAG
└────┬─────┘
     │
     ▼
┌─────────┐     ┌──────────┐
│  GRADE  │────▶│ RETRIEVE │  ← Self-Correction Loop (Cyclic)
└────┬────┘     └──────────┘
     │
     ▼
┌─────────┐
│  TOOLS  │ ← Function Calling (with HITL approval)
└────┬────┘
     │
     ▼
┌──────────┐
│ GENERATE │ ← ReAct + CoT + Streaming
└────┬─────┘
     │
     ▼
┌──────────┐
│ EVALUATE │ ← RAG Triad + Hallucination Detection
└────┬─────┘
     │
     ▼
  Response
```

---

##  Observability

Every query execution produces a full trace with:
- Per-step latency (ms)
- Input/output token counts
- Estimated cost (USD)
- RAG Triad evaluation scores
- Hallucination detection results
- Execution step sequence

Traces are saved to `data/traces/` as JSON files.

---

##  Technical Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Custom DAG/Graph (LangGraph-compatible) |
| LLM | Ollama, OpenAI, Anthropic, Custom |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | Custom VectorStore with persistence |
| Keyword Search | BM25 (from scratch) |
| Re-ranking | Cross-encoder + keyword overlap |
| Persistence | SQLite |
| Security | Regex-based PII/injection, RBAC |
| UI | Rich (terminal) |
