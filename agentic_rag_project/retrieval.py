"""Retrieval and memory engine for the Agentic RAG project.

This module deliberately keeps a production-shaped API while using local,
deterministic implementations by default:
  - hashing embeddings for semantic search
  - BM25 keyword retrieval
  - hybrid score fusion
  - parent-document lookup
  - heuristic reranking
  - semantic routing
  - a small knowledge graph for Graph RAG context
  - context compression and lost-in-the-middle mitigation

The contracts can be backed by Chroma/Qdrant/Pinecone, cross-encoders, or
domain-tuned embeddings later without changing the agent graph.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ingestion import ChunkingEngine, DocumentChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class EmbeddingEngine:
    """Local deterministic embeddings.

    A hashing vectorizer is not a replacement for a tuned embedding model, but
    it gives the project a reliable offline semantic-search contract. Swap this
    class for Ollama embeddings, sentence-transformers, or a managed embedding
    API when accuracy matters more than air-gapped portability.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed_single(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            idx = hash(token) % self.dimension
            vector[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_single(text) for text in texts]


class VectorStore:
    """Small local vector database with metadata filtering and parent lookup."""

    def __init__(self, embedding_engine: EmbeddingEngine, persist_dir: str = "data/vectordb"):
        self.embedding_engine = embedding_engine
        self.persist_dir = persist_dir
        self.chunks: Dict[str, DocumentChunk] = {}
        self.parent_chunks: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, List[float]] = {}
        os.makedirs(self.persist_dir, exist_ok=True)

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        parent_chunks: Optional[Sequence[DocumentChunk]] = None,
    ) -> None:
        for parent in parent_chunks or []:
            self.parent_chunks[parent.chunk_id] = parent

        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = self.embedding_engine.embed_single(chunk.content)

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_vector = self.embedding_engine.embed_single(query)
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk_id, chunk in self.chunks.items():
            if not self._matches_filter(chunk.metadata, metadata_filter):
                continue
            score = _cosine(query_vector, self.vectors.get(chunk_id, []))
            if score > 0:
                scored.append((self.get_parent_chunk(chunk) or chunk, score))
        return _dedupe_and_sort(scored, top_k)

    def get_parent_chunk(self, chunk_or_id: Any) -> Optional[DocumentChunk]:
        if isinstance(chunk_or_id, DocumentChunk):
            parent_id = chunk_or_id.parent_chunk_id or chunk_or_id.metadata.get("parent_chunk_id")
        else:
            chunk = self.chunks.get(str(chunk_or_id))
            parent_id = chunk.parent_chunk_id if chunk else str(chunk_or_id)
        if not parent_id:
            return None
        return self.parent_chunks.get(parent_id)

    def persist(self) -> str:
        path = os.path.join(self.persist_dir, "vector_store.json")
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks.values()],
            "parents": [asdict(chunk) for chunk in self.parent_chunks.values()],
            "vectors": self.vectors,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def load(self) -> None:
        path = os.path.join(self.persist_dir, "vector_store.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.chunks = {
            item["chunk_id"]: DocumentChunk(**item) for item in payload.get("chunks", [])
        }
        self.parent_chunks = {
            item["chunk_id"]: DocumentChunk(**item) for item in payload.get("parents", [])
        }
        self.vectors = {key: list(value) for key, value in payload.get("vectors", {}).items()}

    def _matches_filter(self, metadata: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
        if not metadata_filter:
            return True
        for key, expected in metadata_filter.items():
            actual = metadata.get(key, 0 if key == "security_clearance" else None)
            if isinstance(expected, dict):
                if "$lte" in expected and not (actual <= expected["$lte"]):
                    return False
                if "$gte" in expected and not (actual >= expected["$gte"]):
                    return False
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$eq" in expected and actual != expected["$eq"]:
                    return False
            elif actual != expected:
                return False
        return True


class BM25Retriever:
    """BM25 keyword retriever with metadata filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.doc_tokens = [_tokens(chunk.content) for chunk in self.chunks]
        self.doc_freqs = Counter()
        for token_set in (set(tokens) for tokens in self.doc_tokens):
            self.doc_freqs.update(token_set)
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_terms = _tokens(query)
        if not query_terms or not self.chunks:
            return []

        results: List[Tuple[DocumentChunk, float]] = []
        total_docs = len(self.chunks)
        for chunk, tokens in zip(self.chunks, self.doc_tokens):
            if not VectorStore._matches_filter(self, chunk.metadata, metadata_filter):
                continue
            term_counts = Counter(tokens)
            score = 0.0
            doc_len = len(tokens) or 1
            for term in query_terms:
                if term_counts[term] == 0:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                numerator = term_counts[term] * (self.k1 + 1)
                denominator = term_counts[term] + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avg_doc_len or 1)
                )
                score += idf * numerator / denominator
            if score > 0:
                results.append((chunk, score))
        return _dedupe_and_sort(results, top_k)


class ReRanker:
    """Heuristic reranker that can be replaced by a cross-encoder later."""

    def rerank(
        self,
        query: str,
        results: Sequence[Tuple[DocumentChunk, float]],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_terms = set(_tokens(query))
        reranked: List[Tuple[DocumentChunk, float]] = []
        for chunk, score in results:
            content_terms = set(_tokens(chunk.content))
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            citation_boost = 0.03 if chunk.source_file else 0.0
            reranked.append((chunk, score + overlap + citation_boost))
        return _dedupe_and_sort(reranked, top_k)


class HybridSearchEngine:
    """Hybrid keyword + vector retrieval with score normalization."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        reranker: Optional[ReRanker] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker or ReRanker()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[DocumentChunk],
        parent_chunks: Optional[Sequence[DocumentChunk]] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        persist_dir: str = "data/vectordb",
    ) -> "HybridSearchEngine":
        embeddings = embedding_engine or EmbeddingEngine()
        vector_store = VectorStore(embeddings, persist_dir=persist_dir)
        vector_store.add_chunks(chunks, parent_chunks)
        bm25 = BM25Retriever()
        bm25.index(chunks)
        return cls(vector_store=vector_store, bm25=bm25)

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        vector_results = self.vector_store.search(query, top_k=top_k * 2, metadata_filter=metadata_filter)
        keyword_results = self.bm25.search(query, top_k=top_k * 2, metadata_filter=metadata_filter)

        fused: Dict[str, Tuple[DocumentChunk, float]] = {}
        for chunk, score in _normalize(vector_results):
            fused[chunk.chunk_id] = (chunk, fused.get(chunk.chunk_id, (chunk, 0.0))[1] + score * self.vector_weight)
        for chunk, score in _normalize(keyword_results):
            fused[chunk.chunk_id] = (chunk, fused.get(chunk.chunk_id, (chunk, 0.0))[1] + score * self.keyword_weight)

        results = list(fused.values())
        return self.reranker.rerank(query, results, top_k=top_k)


class SemanticRouter:
    """Intent router based on route exemplars and embedding similarity."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.routes: Dict[str, Dict[str, Any]] = {}

    def add_route(self, name: str, description: str, examples: Sequence[str]) -> None:
        route_text = " ".join([description, *examples])
        self.routes[name] = {
            "description": description,
            "examples": list(examples),
            "embedding": self.embedding_engine.embed_single(route_text),
        }

    def route(self, query: str) -> Tuple[str, float]:
        if not self.routes:
            return "general", 0.0
        query_embedding = self.embedding_engine.embed_single(query)
        best_name = "general"
        best_score = -1.0
        for name, route in self.routes.items():
            score = _cosine(query_embedding, route["embedding"])
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, max(best_score, 0.0)


class KnowledgeGraph:
    """Small entity-to-chunk graph for Graph RAG context."""

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Counter[str]] = defaultdict(Counter)
        self.chunk_text: Dict[str, str] = {}

    def extract_entities_from_text(self, text: str, chunk_id: str) -> List[str]:
        candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4}|[A-Z]{2,})\b", text)
        entities = []
        for candidate in candidates:
            entity = candidate.strip()
            if len(entity) < 3:
                continue
            entities.append(entity)
            record = self.entities.setdefault(entity, {"mentions": 0, "chunks": set()})
            record["mentions"] += 1
            record["chunks"].add(chunk_id)
        self.chunk_text[chunk_id] = text

        unique_entities = list(dict.fromkeys(entities))
        for left in unique_entities:
            for right in unique_entities:
                if left != right:
                    self.edges[left][right] += 1
        return unique_entities

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.extract_entities_from_text(chunk.content, chunk.chunk_id)

    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        query_terms = set(_tokens(query))
        matched_chunks: Counter[str] = Counter()
        for entity, record in self.entities.items():
            entity_terms = set(_tokens(entity))
            if query_terms & entity_terms:
                for chunk_id in record["chunks"]:
                    matched_chunks[chunk_id] += record["mentions"]
                for neighbor, weight in self.edges[entity].most_common(3):
                    for chunk_id in self.entities.get(neighbor, {}).get("chunks", []):
                        matched_chunks[chunk_id] += max(1, weight // 2)

        parts = []
        for chunk_id, _ in matched_chunks.most_common(top_k):
            text = self.chunk_text.get(chunk_id, "")
            if text:
                parts.append(f"[GraphRAG:{chunk_id}] {text[:700]}")
        return "\n".join(parts)


class ContextWindowManager:
    """Context compression and lost-in-the-middle mitigation."""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens

    def compress_context(
        self,
        results: Sequence[Tuple[DocumentChunk, float]],
        query: str,
        graph_context: str = "",
    ) -> str:
        budget_chars = max(1000, self.max_tokens * 4)
        ordered = self._lost_in_middle_order(list(results))
        sections = []
        if graph_context:
            sections.append(graph_context[: min(1500, budget_chars)])
        for idx, (chunk, score) in enumerate(ordered, start=1):
            source = chunk.source_file or chunk.metadata.get("file_path", "memory")
            sections.append(
                f"[{idx}] source={source} score={score:.4f} chunk={chunk.chunk_id}\n"
                f"{chunk.content.strip()}"
            )

        compressed = "\n\n".join(sections)
        if len(compressed) <= budget_chars:
            return compressed

        query_terms = set(_tokens(query))
        sentences = re.split(r"(?<=[.!?])\s+", compressed)
        kept = []
        used = 0
        for sentence in sentences:
            sentence_score = len(query_terms & set(_tokens(sentence)))
            if sentence_score == 0 and used > budget_chars * 0.6:
                continue
            next_len = len(sentence) + 1
            if used + next_len > budget_chars:
                break
            kept.append(sentence)
            used += next_len
        return " ".join(kept)

    def _lost_in_middle_order(
        self, results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        if len(results) <= 2:
            return results
        sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
        reordered: List[Tuple[DocumentChunk, float]] = []
        left = True
        for item in sorted_results:
            if left:
                reordered.insert(0, item)
            else:
                reordered.append(item)
            left = not left
        return reordered


def _normalize(results: Sequence[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
    if not results:
        return []
    max_score = max(score for _, score in results) or 1.0
    return [(chunk, score / max_score) for chunk, score in results]


def _dedupe_and_sort(
    results: Sequence[Tuple[DocumentChunk, float]],
    top_k: int,
) -> List[Tuple[DocumentChunk, float]]:
    best: Dict[str, Tuple[DocumentChunk, float]] = {}
    for chunk, score in results:
        current = best.get(chunk.chunk_id)
        if current is None or score > current[1]:
            best[chunk.chunk_id] = (chunk, score)
    return sorted(best.values(), key=lambda item: item[1], reverse=True)[:top_k]


# Compatibility helpers retained for the original scaffold API.
def process_documents(documents: Sequence[Any]) -> List[DocumentChunk]:
    """Chunk LangChain-style documents or plain strings into DocumentChunk objects."""

    chunker = ChunkingEngine(chunk_size=1000, chunk_overlap=200, strategy="recursive")
    chunks: List[DocumentChunk] = []
    for idx, document in enumerate(documents):
        content = getattr(document, "page_content", str(document))
        metadata = dict(getattr(document, "metadata", {}) or {})
        source = metadata.get("source", f"document_{idx}")
        chunks.extend(chunker.chunk(content, source_file=source, metadata=metadata))
    return chunks


def build_hybrid_retriever(chunks: Sequence[DocumentChunk], embeddings: Any = None) -> HybridSearchEngine:
    """Build the project-level hybrid search engine."""

    engine = embeddings if isinstance(embeddings, EmbeddingEngine) else EmbeddingEngine()
    return HybridSearchEngine.from_chunks(chunks, embedding_engine=engine)


def apply_metadata_filtering(query: str, user_clearance: int) -> Dict[str, Dict[str, int]]:
    """Return a vector-store-compatible RBAC metadata filter."""

    return {"security_clearance": {"$lte": user_clearance}}
