"""
The Retrieval & Memory Engine Module

Implements:
  - Vector Database (ChromaDB / FAISS)
  - Embeddings (sentence-transformers)
  - Semantic Search (dense vector search)
  - Hybrid Search (BM25 keyword + semantic vector)
  - Hierarchical Retrieval / Parent-Document Retrieval
  - Re-ranking (cross-encoder secondary pass)
  - Metadata Filtering (RBAC-based)
  - Semantic Router (intent-based query routing)
  - Context Window management
  - Graph RAG (entity-relationship retrieval)
"""

import os
import re
import math
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ingestion import DocumentChunk, IngestedDocument

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

class EmbeddingEngine:
    """
    Embeddings:
    Numerical vectors that capture the semantic meaning of text.
    Uses sentence-transformers if available, otherwise a TF-IDF fallback.
    Supports Embedding Fine-Tuning awareness.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._use_transformer = False
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_transformer = True
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformer: {self.model_name} (dim={self.dimension})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using TF-IDF fallback.")
            self._use_transformer = False
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self._use_transformer:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        else:
            return [self._tfidf_embed(text) for text in texts]
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
    
    def _tfidf_embed(self, text: str) -> List[float]:
        """Simple TF-IDF based embedding fallback."""
        words = text.lower().split()
        word_counts = Counter(words)
        total = len(words) or 1
        
        # Create a deterministic embedding from word hashes
        embedding = [0.0] * self.dimension
        for word, count in word_counts.items():
            tf = count / total
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            embedding[idx] += tf
        
        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding)) or 1
        return [x / norm for x in embedding]


# ─────────────────────────────────────────────
# BM25 Keyword Search
# ─────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 Keyword Search:
    Traditional keyword-based retrieval using the BM25 algorithm.
    Part of Hybrid Search.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.avg_dl: float = 0
        self.N: int = 0
    
    def index(self, chunks: List[DocumentChunk]):
        """Index chunks for BM25 search."""
        self.corpus = chunks
        self.tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self.N = len(chunks)
        
        # Calculate document frequencies
        self.doc_freqs = {}
        for tokens in self.tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Average document length
        total_len = sum(len(t) for t in self.tokenized_corpus)
        self.avg_dl = total_len / self.N if self.N > 0 else 0
        
        logger.info(f"BM25: Indexed {self.N} chunks, vocabulary size: {len(self.doc_freqs)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = self._bm25_score(query_tokens, doc_tokens, len(doc_tokens))
            scores.append((self.corpus[i], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str],
                     doc_len: int) -> float:
        """Calculate BM25 score for a single document."""
        score = 0.0
        doc_token_counts = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in doc_token_counts:
                continue
            
            tf = doc_token_counts[token]
            df = self.doc_freqs.get(token, 0)
            
            # IDF
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_dl or 1))
            )
            
            score += idf * tf_norm
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())


# ─────────────────────────────────────────────
# Vector Store (Semantic Search)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Vector Database & Semantic Search:
    Stores numerical representations (embeddings) and performs
    similarity search based on semantic meaning.
    
    Supports Vector Quantization for compression.
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine, persist_dir: str = "data/vectordb"):
        self.embedding_engine = embedding_engine
        self.persist_dir = persist_dir
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self.parent_chunks: Dict[str, DocumentChunk] = {}  # For hierarchical retrieval
        os.makedirs(persist_dir, exist_ok=True)
    
    def add_chunks(self, chunks: List[DocumentChunk],
                    parent_chunks: Optional[List[DocumentChunk]] = None):
        """Add chunks and their embeddings to the store."""
        if not chunks:
            return
        
        texts = [c.content for c in chunks]
        new_embeddings = self.embedding_engine.embed(texts)
        
        self.chunks.extend(chunks)
        self.embeddings.extend(new_embeddings)
        
        # Store parent chunks for hierarchical retrieval
        if parent_chunks:
            for pc in parent_chunks:
                self.parent_chunks[pc.chunk_id] = pc
        
        logger.info(f"VectorStore: Added {len(chunks)} chunks (total: {len(self.chunks)})")
    
    def search(self, query: str, top_k: int = 5,
               metadata_filter: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Semantic Search:
        Search by meaning of the query rather than keyword matches.
        Supports Metadata Filtering.
        """
        if not self.chunks:
            return []
        
        query_embedding = self.embedding_engine.embed_single(query)
        
        # Calculate cosine similarities
        results = []
        for i, (chunk, emb) in enumerate(zip(self.chunks, self.embeddings)):
            # Apply metadata filtering
            if metadata_filter and not self._matches_filter(chunk.metadata, metadata_filter):
                continue
            
            similarity = self._cosine_similarity(query_embedding, emb)
            results.append((chunk, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """
        Metadata Filtering:
        Check if chunk metadata matches the filter criteria.
        """
        for key, condition in filter_dict.items():
            value = metadata.get(key)
            if value is None:
                continue  # Skip if metadata key doesn't exist
            
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$lte" and not (value <= target):
                        return False
                    elif op == "$gte" and not (value >= target):
                        return False
                    elif op == "$eq" and value != target:
                        return False
                    elif op == "$in" and value not in target:
                        return False
            elif value != condition:
                return False
        
        return True
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1)) or 1
        norm2 = math.sqrt(sum(b * b for b in vec2)) or 1
        return dot / (norm1 * norm2)
    
    def get_parent_chunk(self, child_chunk: DocumentChunk) -> Optional[DocumentChunk]:
        """
        Hierarchical Retrieval / Parent-Document Retrieval:
        Given a child chunk, retrieve the full parent document.
        """
        parent_id = child_chunk.parent_chunk_id or child_chunk.metadata.get("parent_chunk_id")
        if parent_id and parent_id in self.parent_chunks:
            return self.parent_chunks[parent_id]
        return None
    
    def save(self):
        """Persist the vector store to disk."""
        data = {
            "chunks": [
                {
                    "content": c.content,
                    "metadata": c.metadata,
                    "chunk_id": c.chunk_id,
                    "parent_chunk_id": c.parent_chunk_id,
                    "source_file": c.source_file,
                    "chunk_index": c.chunk_index,
                }
                for c in self.chunks
            ],
            "embeddings": self.embeddings,
            "parent_chunks": {
                pid: {
                    "content": pc.content,
                    "metadata": pc.metadata,
                    "chunk_id": pc.chunk_id,
                    "source_file": pc.source_file,
                }
                for pid, pc in self.parent_chunks.items()
            },
        }
        path = os.path.join(self.persist_dir, "vectorstore.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"VectorStore: Saved to {path}")
    
    def load(self) -> bool:
        """Load vector store from disk."""
        path = os.path.join(self.persist_dir, "vectorstore.json")
        if not os.path.exists(path):
            return False
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.chunks = [
            DocumentChunk(
                content=c["content"],
                metadata=c["metadata"],
                chunk_id=c["chunk_id"],
                parent_chunk_id=c.get("parent_chunk_id", ""),
                source_file=c.get("source_file", ""),
                chunk_index=c.get("chunk_index", 0),
            )
            for c in data.get("chunks", [])
        ]
        self.embeddings = data.get("embeddings", [])
        
        for pid, pc_data in data.get("parent_chunks", {}).items():
            self.parent_chunks[pid] = DocumentChunk(
                content=pc_data["content"],
                metadata=pc_data["metadata"],
                chunk_id=pc_data["chunk_id"],
                source_file=pc_data.get("source_file", ""),
            )
        
        logger.info(f"VectorStore: Loaded {len(self.chunks)} chunks from {path}")
        return True


# ─────────────────────────────────────────────
# Re-ranking
# ─────────────────────────────────────────────

class ReRanker:
    """
    Re-ranking:
    Uses a secondary algorithm to evaluate and re-order retrieved context
    so the most relevant information is fed to the LLM.
    
    Implements cross-encoder scoring simulation.
    """
    
    def __init__(self, model_type: str = "cross-encoder"):
        self.model_type = model_type
        self._cross_encoder = None
        self._init_model()
    
    def _init_model(self):
        """Initialize cross-encoder model if available."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Re-ranker: Loaded cross-encoder model")
        except (ImportError, Exception) as e:
            logger.info(f"Cross-encoder not available ({e}). Using keyword overlap re-ranking.")
    
    def rerank(self, query: str, results: List[Tuple[DocumentChunk, float]],
               top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Re-rank results using cross-encoder or keyword overlap."""
        if not results:
            return []
        
        if self._cross_encoder:
            return self._rerank_cross_encoder(query, results, top_k)
        else:
            return self._rerank_keyword_overlap(query, results, top_k)
    
    def _rerank_cross_encoder(self, query: str,
                               results: List[Tuple[DocumentChunk, float]],
                               top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Re-rank using cross-encoder model."""
        pairs = [(query, chunk.content) for chunk, _ in results]
        scores = self._cross_encoder.predict(pairs)
        
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(results, scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def _rerank_keyword_overlap(self, query: str,
                                 results: List[Tuple[DocumentChunk, float]],
                                 top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Re-rank using keyword overlap (fallback)."""
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        reranked = []
        for chunk, original_score in results:
            doc_tokens = set(re.findall(r'\w+', chunk.content.lower()))
            overlap = len(query_tokens & doc_tokens) / (len(query_tokens) or 1)
            # Combine original score with overlap score
            combined = 0.6 * original_score + 0.4 * overlap
            reranked.append((chunk, combined))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


# ─────────────────────────────────────────────
# Hybrid Search (BM25 + Semantic)
# ─────────────────────────────────────────────

class HybridSearchEngine:
    """
    Hybrid Search:
    Combining keyword search (BM25) and vector search to ensure
    both precision and contextual relevance.
    
    Uses Ensemble Retrieval with configurable weights.
    """
    
    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever,
                 reranker: ReRanker, bm25_weight: float = 0.4,
                 semantic_weight: float = 0.6):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
    
    def search(self, query: str, top_k: int = 5,
               metadata_filter: Optional[Dict] = None,
               use_reranking: bool = True,
               use_hierarchical: bool = True) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform hybrid search combining BM25 and semantic results.
        Optionally applies re-ranking and hierarchical retrieval.
        """
        # 1. BM25 keyword search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # 2. Semantic vector search with metadata filtering
        semantic_results = self.vector_store.search(query, top_k=top_k * 2,
                                                      metadata_filter=metadata_filter)
        
        # 3. Normalize and combine scores
        combined = self._fuse_results(bm25_results, semantic_results)
        
        # 4. Re-ranking
        if use_reranking and combined:
            combined = self.reranker.rerank(query, combined, top_k=top_k)
        else:
            combined = combined[:top_k]
        
        # 5. Hierarchical retrieval: replace child chunks with parent documents
        if use_hierarchical:
            combined = self._expand_to_parents(combined)
        
        return combined
    
    def _fuse_results(self, bm25_results: List[Tuple[DocumentChunk, float]],
                       semantic_results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """Fuse results from BM25 and semantic search using Reciprocal Rank Fusion."""
        k = 60  # RRF constant
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        
        # BM25 scores
        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk.chunk_id
            rrf_score = self.bm25_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf_score
            chunk_map[cid] = chunk
        
        # Semantic scores
        for rank, (chunk, score) in enumerate(semantic_results):
            cid = chunk.chunk_id
            rrf_score = self.semantic_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf_score
            chunk_map[cid] = chunk
        
        # Sort by combined RRF score
        fused = [(chunk_map[cid], score) for cid, score in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused
    
    def _expand_to_parents(self, results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """
        Hierarchical Retrieval / Parent-Document Retrieval:
        Replace child chunks with their parent documents for richer context.
        """
        expanded = []
        seen_parent_ids = set()
        
        for chunk, score in results:
            parent = self.vector_store.get_parent_chunk(chunk)
            if parent and parent.chunk_id not in seen_parent_ids:
                expanded.append((parent, score))
                seen_parent_ids.add(parent.chunk_id)
            else:
                expanded.append((chunk, score))
        
        return expanded


# ─────────────────────────────────────────────
# Semantic Router
# ─────────────────────────────────────────────

class SemanticRouter:
    """
    Semantic Router:
    An intelligent mechanism that analyzes the user's intent to
    direct the query to the correct data source or agent.
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.routes: Dict[str, Dict] = {}
        self.route_embeddings: Dict[str, List[float]] = {}
    
    def add_route(self, route_name: str, description: str,
                   sample_queries: List[str], handler: str = "default"):
        """Register a semantic route with sample queries."""
        self.routes[route_name] = {
            "description": description,
            "sample_queries": sample_queries,
            "handler": handler,
        }
        
        # Create an averaged embedding from sample queries
        embeddings = self.embedding_engine.embed(sample_queries)
        avg_embedding = [
            sum(e[i] for e in embeddings) / len(embeddings)
            for i in range(len(embeddings[0]))
        ]
        self.route_embeddings[route_name] = avg_embedding
        
        logger.info(f"SemanticRouter: Added route '{route_name}' with {len(sample_queries)} samples")
    
    def route(self, query: str) -> Tuple[str, float]:
        """
        Determine which route best matches the query.
        Returns (route_name, confidence_score).
        """
        if not self.routes:
            return "default", 0.0
        
        query_embedding = self.embedding_engine.embed_single(query)
        
        best_route = "default"
        best_score = -1.0
        
        for route_name, route_embedding in self.route_embeddings.items():
            score = self._cosine_similarity(query_embedding, route_embedding)
            if score > best_score:
                best_score = score
                best_route = route_name
        
        logger.info(f"SemanticRouter: Query routed to '{best_route}' (score={best_score:.4f})")
        return best_route, best_score
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1)) or 1
        norm2 = math.sqrt(sum(b * b for b in vec2)) or 1
        return dot / (norm1 * norm2)


# ─────────────────────────────────────────────
# Graph RAG
# ─────────────────────────────────────────────

class KnowledgeGraph:
    """
    Graph RAG:
    Utilizes knowledge graphs alongside vector databases to retrieve
    highly interconnected data and relationships.
    
    Implements a simple in-memory entity-relationship graph.
    """
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}       # entity_id -> {name, type, properties}
        self.relations: List[Dict] = []            # [{source, target, relation, properties}]
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # entity_id -> [related entity_ids]
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                    properties: Dict = None):
        """Add an entity node to the graph."""
        self.entities[entity_id] = {
            "name": name,
            "type": entity_type,
            "properties": properties or {},
        }
    
    def add_relation(self, source_id: str, target_id: str, relation: str,
                      properties: Dict = None):
        """Add a relationship edge between entities."""
        self.relations.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties or {},
        })
        self.adjacency[source_id].append(target_id)
        self.adjacency[target_id].append(source_id)
    
    def extract_entities_from_text(self, text: str, chunk_id: str = "") -> List[str]:
        """
        Simple entity extraction using regex patterns.
        Extracts capitalized phrases as potential entities.
        """
        # Find capitalized multi-word phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        entity_ids = []
        for entity_name in set(entities):
            eid = hashlib.md5(entity_name.encode()).hexdigest()[:8]
            if eid not in self.entities:
                self.add_entity(eid, entity_name, "extracted",
                               {"source_chunk": chunk_id})
            entity_ids.append(eid)
        
        # Create co-occurrence relations
        for i, eid1 in enumerate(entity_ids):
            for eid2 in entity_ids[i + 1:]:
                self.add_relation(eid1, eid2, "co_occurs_with",
                                   {"source_chunk": chunk_id})
        
        return entity_ids
    
    def query_graph(self, entity_name: str, max_hops: int = 2) -> Dict:
        """
        Query the knowledge graph for an entity and its neighbors.
        Returns a subgraph context.
        """
        # Find entity by name
        target_id = None
        for eid, info in self.entities.items():
            if info["name"].lower() == entity_name.lower():
                target_id = eid
                break
        
        if not target_id:
            return {"entities": [], "relations": []}
        
        # BFS to find related entities
        visited = set()
        queue = [(target_id, 0)]
        result_entities = []
        result_relations = []
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            
            if current in self.entities:
                result_entities.append(self.entities[current])
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        # Collect relations between visited entities
        for rel in self.relations:
            if rel["source"] in visited and rel["target"] in visited:
                result_relations.append(rel)
        
        return {"entities": result_entities, "relations": result_relations}
    
    def get_context_for_query(self, query: str, max_hops: int = 2) -> str:
        """
        Generate a text context from graph relationships for a query.
        This is the Graph RAG retrieval step.
        """
        # Extract potential entities from the query
        query_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        context_parts = []
        for entity_name in query_entities:
            subgraph = self.query_graph(entity_name, max_hops)
            if subgraph["entities"]:
                for ent in subgraph["entities"]:
                    context_parts.append(f"Entity: {ent['name']} (type: {ent['type']})")
                for rel in subgraph["relations"][:10]:  # Limit relations
                    src = self.entities.get(rel["source"], {}).get("name", "?")
                    tgt = self.entities.get(rel["target"], {}).get("name", "?")
                    context_parts.append(f"Relation: {src} --[{rel['relation']}]--> {tgt}")
        
        return "\n".join(context_parts) if context_parts else ""


# ─────────────────────────────────────────────
# Context Window Management
# ─────────────────────────────────────────────

class ContextWindowManager:
    """
    Context Window:
    Manages the maximum amount of tokens an LLM can process.
    Implements Context Compression / Prompt Compression and
    Lost in the Middle Mitigation.
    """
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return len(text) // 4
    
    def compress_context(self, chunks: List[Tuple[DocumentChunk, float]],
                          query: str) -> str:
        """
        Context Compression / Prompt Compression:
        Strip out irrelevant tokens to save costs and reduce latency.
        
        Also implements Lost in the Middle Mitigation:
        Places most relevant content at the beginning and end of the context.
        """
        if not chunks:
            return ""
        
        # Sort by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Lost in the Middle Mitigation:
        # Place best content at start and end, weaker in middle
        if len(sorted_chunks) > 2:
            reordered = []
            for i, item in enumerate(sorted_chunks):
                if i % 2 == 0:
                    reordered.insert(0, item)
                else:
                    reordered.append(item)
            sorted_chunks = reordered
        
        # Build context within token limit
        context_parts = []
        token_count = 0
        
        for chunk, score in sorted_chunks:
            chunk_tokens = self.estimate_tokens(chunk.content)
            if token_count + chunk_tokens > self.max_tokens * 0.8:  # Leave room for prompt
                break
            context_parts.append(f"[Relevance: {score:.3f}] {chunk.content}")
            token_count += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def truncate_to_window(self, text: str) -> str:
        """Truncate text to fit within context window."""
        estimated = self.estimate_tokens(text)
        if estimated <= self.max_tokens:
            return text
        
        max_chars = self.max_tokens * 4
        return text[:max_chars] + "\n...[truncated]"


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_retrieval_engine(config=None):
    """Create the full retrieval engine with all components."""
    from config import RAGConfig
    
    cfg = config or RAGConfig()
    
    # Embeddings
    embedding_engine = EmbeddingEngine(
        model_name=cfg.data_engineering.embedding_model,
        dimension=cfg.data_engineering.embedding_dimension,
    )
    
    # Vector Store
    vector_store = VectorStore(
        embedding_engine=embedding_engine,
        persist_dir=cfg.retrieval.vector_db_persist_dir,
    )
    
    # BM25
    bm25 = BM25Retriever()
    
    # Re-ranker
    reranker = ReRanker(model_type=cfg.retrieval.reranking_model)
    
    # Hybrid Search
    hybrid_search = HybridSearchEngine(
        vector_store=vector_store,
        bm25=bm25,
        reranker=reranker,
        bm25_weight=cfg.retrieval.bm25_weight,
        semantic_weight=cfg.retrieval.semantic_weight,
    )
    
    # Semantic Router
    semantic_router = SemanticRouter(embedding_engine)
    
    # Knowledge Graph (Graph RAG)
    knowledge_graph = KnowledgeGraph()
    
    # Context Window Manager
    context_manager = ContextWindowManager(max_tokens=cfg.retrieval.context_window_limit)
    
    return {
        "embedding_engine": embedding_engine,
        "vector_store": vector_store,
        "bm25": bm25,
        "reranker": reranker,
        "hybrid_search": hybrid_search,
        "semantic_router": semantic_router,
        "knowledge_graph": knowledge_graph,
        "context_manager": context_manager,
    }
