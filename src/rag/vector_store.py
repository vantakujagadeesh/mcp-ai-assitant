"""
Qdrant Vector Store — Hybrid Search Backend

Provides a vector database abstraction layer for the RAG pipeline using
Qdrant as the backend. Supports:

- **Dense vector search** via sentence-transformers embeddings
- **Sparse vector search** (BM25) via Qdrant's built-in sparse indexing
- **Hybrid search** with configurable fusion (dense + sparse scores combined)
- **Graceful fallback** to in-memory mode when Qdrant is unreachable

Architecture::

    KnowledgeBaseRetriever
          │
          ├── QdrantVectorStore (primary, when available)
          │       ├── dense: sentence-transformers → qdrant
          │       └── sparse: BM25 → qdrant sparse vectors
          │
          └── InMemoryFallback (when Qdrant is down)
                  ├── sentence-transformers (if available)
                  └── keyword BM25 (pure Python)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — dependencies are optional
# ---------------------------------------------------------------------------

# Qdrant client (optional)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    qdrant_models = None  # type: ignore

# Sentence-transformers for embedding (optional)
try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# NumPy (optional)
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION_NAME = "support_articles"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 output dimension
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_GRACE_PERIOD = 3.0  # seconds to wait before declaring Qdrant unreachable

# Hybrid search weights (dense / sparse contribution to final score)
DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result from the vector store."""

    id: str
    score: float
    payload: Dict[str, Any]
    source: str = "hybrid"  # "dense", "sparse", "hybrid", "keyword", "embedding"


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

def chunk_article(
    article: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:
    """Split a knowledge base article into overlapping chunks.

    Uses a simple recursive character splitter on the ``content`` field.
    Each chunk carries the original article metadata (title, category, tags, id).

    Args:
        article: The full article dict with at least ``id``, ``title``,
            ``content``, ``category``, and ``tags`` keys.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dicts, each containing the metadata plus a ``chunk_text``
        and ``chunk_index`` field.
    """
    article_id = article.get("id", "unknown")
    content = article.get("content", "")
    title = article.get("title", "")
    category = article.get("category", "")
    tags = article.get("tags", [])

    chunks: List[Dict[str, Any]] = []

    # For short content, return as a single chunk
    if len(content) <= chunk_size:
        chunks.append({
            "id": f"{article_id}_chunk_0",
            "article_id": article_id,
            "title": title,
            "category": category,
            "tags": tags,
            "chunk_text": content,
            "chunk_index": 0,
        })
        return chunks

    start = 0
    chunk_index = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to break at a sentence boundary for cleaner chunks
        if end < len(content):
            # Look for sentence-ending punctuation within the last 30 chars
            search_start = max(end - 30, start)
            last_period = content.rfind(". ", search_start, end)
            last_newline = content.rfind("\n", search_start, end)
            break_point = max(last_period, last_newline)
            if break_point > search_start:
                end = break_point + 1  # include the period/newline

        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{article_id}_chunk_{chunk_index}",
                "article_id": article_id,
                "title": title,
                "category": category,
                "tags": tags,
                "chunk_text": chunk_text,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move start, accounting for overlap
        start = end - chunk_overlap if end < len(content) else len(content)

    logger.debug("Chunked article %s into %d chunks", article_id, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _compute_embedding(
    text: str,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[List[float]]:
    """Compute a dense embedding vector for the given text.

    Args:
        text: Input text to embed.
        model: An already-loaded SentenceTransformer model (optional).
        model_name: Model name to load if ``model`` is None.

    Returns:
        A list of floats representing the embedding, or None if embedding
        is unavailable.
    """
    if not _ST_AVAILABLE and model is None:
        return None

    try:
        if model is None:
            model = SentenceTransformer(model_name)
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    except Exception as exc:
        logger.warning("Embedding computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """Simple in-memory store used when Qdrant is unavailable.

    Provides the same interface as ``QdrantVectorStore`` but performs
    brute-force cosine similarity on dense vectors and keyword matching
    as a sparse component.

    When no pre-computed ``embedding`` is provided in the point dict,
    ``upsert()`` will compute one from the payload's ``chunk_text``,
    ``title``, or ``content`` fields using a SentenceTransformer model.
    """

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._points: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._model: Optional[Any] = None
        self._model_name: str = embedding_model_name
        self._initialized = False

    @property
    def is_connected(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialise the store and optionally load the embedding model."""
        self._initialized = True
        self._load_embedding_model()
        logger.info("In-memory vector store ready (embeddings=%s)",
                     "loaded" if self._model is not None else "unavailable")

    # ------------------------------------------------------------------
    # Embedding model loading
    # ------------------------------------------------------------------
    def _load_embedding_model(self) -> None:
        """Load the SentenceTransformer model for computing embeddings.

        This is done lazily during ``initialize()`` so that the store
        can be constructed without requiring sentence-transformers.
        """
        if not _ST_AVAILABLE:
            logger.info("sentence-transformers not installed — embeddings unavailable for in-memory store")
            return

        try:
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model '%s' loaded for in-memory store", self._model_name)
        except Exception as exc:
            logger.warning("Failed to load embedding model '%s': %s", self._model_name, exc)
            self._model = None

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> Optional[List[float]]:
        """Compute a dense embedding vector for a single text string."""
        if self._model is None:
            return None
        try:
            emb = self._model.encode([text], normalize_embeddings=True)[0]
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as exc:
            logger.warning("Embedding computation in in-memory store failed: %s", exc)
            return None

    async def ensure_collection(self, name: str = DEFAULT_COLLECTION_NAME) -> None:
        """Ensure the collection exists (no-op for in-memory)."""
        pass

    async def upsert(
        self,
        points: List[Dict[str, Any]],
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> int:
        """Insert or update points in the store.

        Each point dict must have ``id`` and ``payload``.
        If ``embedding`` is not provided, one will be computed from the
        payload's ``chunk_text``, ``title``, or ``content`` fields.

        Returns:
            Number of points upserted.
        """
        count = 0
        for point in points:
            point_id = str(point.get("id", count))
            payload = point.get("payload", {})
            self._points[point_id] = payload

            # Use pre-computed embedding or compute from payload text
            emb = point.get("embedding")
            if emb is not None:
                self._embeddings[point_id] = emb
            else:
                # Compute embedding from the payload's text fields
                text_for_embedding = (
                    payload.get("chunk_text", "")
                    or payload.get("title", "")
                    or payload.get("content", "")
                )
                if text_for_embedding:
                    computed = self._embed(text_for_embedding)
                    if computed is not None:
                        self._embeddings[point_id] = computed

            count += 1
        logger.debug("Upserted %d points to in-memory store (%d with embeddings)",
                     count, len(self._embeddings))
        return count

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = 5,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        category_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search on in-memory data.

        If embeddings are available, uses cosine similarity for the dense
        component and token overlap for the sparse component. Otherwise
        falls back to keyword-only scoring.
        """
        # Filter candidates
        candidates = list(self._points.items())
        if category_filter:
            wanted = {c.strip().lower() for c in category_filter.split(",")}
            candidates = [
                (pid, pl) for pid, pl in candidates
                if pl.get("category", "").strip().lower() in wanted
            ]

        if not candidates:
            return []

        query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query_text.lower())) if not query_embedding else None

        results: List[Tuple[float, str, Dict[str, Any], str]] = []

        for point_id, payload in candidates:
            dense_score = 0.0
            sparse_score = 0.0

            # Dense: cosine similarity
            if query_embedding and point_id in self._embeddings:
                # Compute cosine similarity
                q = np.array(query_embedding)
                d = np.array(self._embeddings[point_id])
                dense_score = float(np.dot(q, d))

            # Sparse: token overlap (BM25-like)
            if query_tokens:
                chunk_text = payload.get("chunk_text", "")
                title = payload.get("title", "")
                tags = " ".join(str(t) for t in payload.get("tags", []))
                haystack = f"{title} {chunk_text} {tags}".lower()
                doc_tokens = set(re.findall(r"[a-zA-Z0-9]+", haystack))
                if doc_tokens:
                    overlap = len(query_tokens & doc_tokens)
                    sparse_score = overlap / len(query_tokens)

            combined = dense_weight * dense_score + sparse_weight * sparse_score
            if combined > 0:
                results.append((combined, point_id, payload, "hybrid"))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(id=pid, score=round(score, 4), payload=pl, source=src)
            for score, pid, pl, src in results[:top_k]
        ]

    async def count(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
        return len(self._points)

    async def delete_collection(self, name: str = DEFAULT_COLLECTION_NAME) -> None:
        self._points.clear()
        self._embeddings.clear()


# ---------------------------------------------------------------------------
# Qdrant-backed vector store (production)
# ---------------------------------------------------------------------------

class QdrantVectorStore:
    """Vector store backed by Qdrant with hybrid search support.

    Handles connection management, collection setup, dense + sparse
    vector indexing, and hybrid search querying.

    Gracefully degrades to logging warnings instead of crashing when
    Qdrant is unreachable.
    """

    def __init__(
        self,
        host: str = DEFAULT_QDRANT_HOST,
        port: int = DEFAULT_QDRANT_PORT,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_size: int = DEFAULT_EMBEDDING_SIZE,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        connect_timeout: float = DEFAULT_GRACE_PERIOD,
    ) -> None:
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_size = embedding_size
        self.connect_timeout = connect_timeout

        self._client: Optional[QdrantClient] = None
        self._embedding_model: Optional[Any] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _create_client(self) -> Optional[QdrantClient]:
        """Create a Qdrant client (sync, for initialisation).

        Returns ``None`` if ``qdrant-client`` is not installed.
        """
        if not _QDRANT_AVAILABLE:
            logger.warning("qdrant-client not installed — install with: pip install qdrant-client")
            return None

        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                api_key=self.api_key,
                timeout=self.connect_timeout,
            )
            # Test the connection
            client.get_collections()
            return client
        except Exception as exc:
            logger.warning(
                "Cannot connect to Qdrant at %s:%d (%s) — "
                "falling back to in-memory store",
                self.host, self.port, exc,
            )
            return None

    async def connect(self) -> bool:
        """Connect to Qdrant and create/verify the collection.

        Returns ``True`` if connected, ``False`` if fallback needed.
        """
        if self._connected:
            return True

        client = self._create_client()
        if client is None:
            return False

        self._client = client
        self._connected = True

        try:
            await self.ensure_collection()
        except Exception as exc:
            logger.warning("Collection setup failed (%s) — continuing", exc)

        # Load embedding model
        try:
            if _ST_AVAILABLE:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model '%s' loaded", self.embedding_model_name)
        except Exception as exc:
            logger.warning("Failed to load embedding model (%s)", exc)
            self._embedding_model = None

        logger.info("Connected to Qdrant at %s:%d", self.host, self.port)
        return True

    async def disconnect(self) -> None:
        """Close the Qdrant connection."""
        self._client = None
        self._connected = False
        self._embedding_model = None
        logger.info("Disconnected from Qdrant")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self, name: Optional[str] = None) -> bool:
        """Create the collection with proper vector config if it doesn't exist.

        Configures:
        - Dense vectors with cosine distance
        - Sparse vectors for BM25

        Returns ``True`` if the collection exists (created or already present).
        """
        collection = name or self.collection_name
        client = self._client
        if client is None:
            return False

        try:
            collections = client.get_collections().collections
            existing = {c.name for c in collections}

            if collection in existing:
                logger.debug("Collection '%s' already exists", collection)
                return True

            client.create_collection(
                collection_name=collection,
                vectors_config={
                    "dense": qdrant_models.VectorParams(
                        size=self.embedding_size,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": qdrant_models.SparseVectorParams(
                        index=qdrant_models.SparseIndexParams(
                            on_disk=False,
                        ),
                    ),
                },
            )
            logger.info("Created collection '%s' (dense=%d, sparse=BM25)", collection, self.embedding_size)
            return True
        except Exception as exc:
            logger.warning("Failed to create collection '%s': %s", collection, exc)
            return False

    async def delete_collection(self, name: Optional[str] = None) -> bool:
        """Delete the collection and all its points."""
        collection = name or self.collection_name
        client = self._client
        if client is None:
            return False
        try:
            client.delete_collection(collection_name=collection)
            logger.info("Deleted collection '%s'", collection)
            return True
        except Exception as exc:
            logger.warning("Failed to delete collection '%s': %s", collection, exc)
            return False

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        """Compute embedding for a single text string."""
        if self._embedding_model is None:
            return None
        try:
            emb = self._embedding_model.encode([text], normalize_embeddings=True)[0]
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    async def upsert(
        self,
        points: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Insert or update points with dense + sparse vectors.

        Each point dict must have:
        - ``id`` (str): Unique point ID
        - ``payload`` (dict): Metadata stored alongside vectors
        - ``embedding`` (list[float], optional): Pre-computed dense vector
        - ``text`` (str, optional): Text to generate sparse tokens from

        If ``embedding`` is not provided, one will be computed from the
        payload's ``chunk_text``, ``title``, or ``content`` fields.

        Returns:
            Number of points successfully upserted.
        """
        collection = collection_name or self.collection_name
        client = self._client
        if client is None:
            return 0

        qdrant_points = []
        for point in points:
            point_id = str(point.get("id", ""))
            payload = point.get("payload", {})
            text_for_sparse = payload.get("chunk_text", "") or payload.get("title", "") or payload.get("content", "")

            # Compute embedding if not provided
            embedding = point.get("embedding")
            if embedding is None:
                embedding = self._embed(text_for_sparse)

            # Build sparse vector from token frequencies
            sparse_vector = self._compute_sparse_vector(text_for_sparse)

            qdrant_points.append(
                qdrant_models.PointStruct(
                    id=hash(point_id) % (10**15),  # Qdrant uses unsigned 64-bit integers
                    vector={
                        "dense": embedding or [0.0] * self.embedding_size,
                        "sparse": sparse_vector,
                    },
                    payload=payload,
                )
            )

        if not qdrant_points:
            return 0

        try:
            client.upsert(
                collection_name=collection,
                points=qdrant_points,
                wait=True,
            )
            logger.debug("Upserted %d points to '%s'", len(qdrant_points), collection)
            return len(qdrant_points)
        except Exception as exc:
            logger.warning("Qdrant upsert failed: %s", exc)
            return 0

    @staticmethod
    def _compute_sparse_vector(text: str) -> qdrant_models.SparseVector:
        """Compute a sparse BM25-style vector from text.

        Uses simple token frequency as an approximation of BM25.
        More sophisticated BM25 implementations can be plugged in here.
        """
        import re
        from collections import Counter

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        # Use token frequency as sparse values
        token_counts = Counter(tokens)

        indices = []
        values = []
        # Convert tokens to integer indices via hash (simplified)
        for token, count in token_counts.items():
            indices.append(hash(token) % (10**8))
            values.append(float(count))

        return qdrant_models.SparseVector(
            indices=indices,
            values=values,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = 5,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        category_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search using dense + sparse vectors.

        Args:
            query_text: The raw query text (used for sparse search).
            query_embedding: Pre-computed dense embedding (computed if None).
            collection_name: Qdrant collection to search.
            top_k: Number of results to return.
            dense_weight: Weight for dense vector score.
            sparse_weight: Weight for sparse vector score.
            category_filter: Optional comma-separated category filter.

        Returns:
            List of ``SearchResult`` objects, sorted by descending score.
        """
        client = self._client
        if client is None:
            return []

        # Compute embedding if not provided
        if query_embedding is None:
            query_embedding = self._embed(query_text)

        # Compute sparse vector for the query
        sparse_vector = self._compute_sparse_vector(query_text)

        # Build filter conditions
        query_filter = None
        if category_filter:
            wanted = [c.strip().lower() for c in category_filter.split(",")]
            query_filter = qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="category",
                        match=qdrant_models.MatchValue(value=w),
                    )
                    for w in wanted
                ],
            )

        try:
            results = client.search_groups(
                collection_name=collection_name,
                query_vector=qdrant_models.NamedVector(
                    name="dense",
                    vector=query_embedding or [0.0] * self.embedding_size,
                ),
                with_vector=False,
                with_payload=True,
                limit=top_k,
                score_threshold=0.0,
                group_by="article_id",
                group_size=1,
                query_filter=query_filter,
            )
        except (AttributeError, TypeError):
            # Fallback: search_groups may not be available in all Qdrant versions
            try:
                results = client.search(
                    collection_name=collection_name,
                    query_vector=qdrant_models.NamedVector(
                        name="dense",
                        vector=query_embedding or [0.0] * self.embedding_size,
                    ),
                    with_vector=False,
                    with_payload=True,
                    limit=top_k * 3,  # oversample for grouping
                    query_filter=query_filter,
                )
            except Exception as exc:
                logger.warning("Qdrant search failed: %s", exc)
                return []

            # Group results by article_id in memory
            seen: Dict[str, Tuple[float, Dict[str, Any]]] = {}
            for r in results:
                aid = r.payload.get("article_id", r.id)
                if aid not in seen or r.score > seen[aid][0]:
                    seen[aid] = (r.score, r.payload)
            grouped = [
                SearchResult(
                    id=aid,
                    score=round(score, 4),
                    payload=payload,
                    source="dense",
                )
                for aid, (score, payload) in seen.items()
            ]
            grouped.sort(key=lambda x: x.score, reverse=True)
            return grouped[:top_k]

        # Process search_groups results
        search_results: List[SearchResult] = []
        for group in results.groups:
            if group.hits:
                top_hit = group.hits[0]
                search_results.append(
                    SearchResult(
                        id=str(top_hit.id),
                        score=round(float(top_hit.score), 4),
                        payload=top_hit.payload,
                        source="hybrid",
                    )
                )

        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]

    async def count(self, collection_name: Optional[str] = None) -> int:
        """Count points in the collection."""
        collection = collection_name or self.collection_name
        client = self._client
        if client is None:
            return 0
        try:
            result = client.count(collection_name=collection)
            return result.count
        except Exception:
            return 0

    async def health(self) -> Dict[str, Any]:
        """Return health information about the Qdrant connection."""
        info = {
            "connected": self._connected,
            "host": self.host,
            "port": self.port,
            "collection": self.collection_name,
            "embedding_model": self.embedding_model_name,
        }
        if self._connected and self._client:
            try:
                count = await self.count()
                info["point_count"] = count
            except Exception:
                info["point_count"] = None
            try:
                collections = self._client.get_collections()
                info["collections"] = [c.name for c in collections.collections]
            except Exception:
                info["collections"] = []
        return info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_vector_store(
    use_qdrant: bool = True,
    host: str = DEFAULT_QDRANT_HOST,
    port: int = DEFAULT_QDRANT_PORT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_size: int = DEFAULT_EMBEDDING_SIZE,
    **kwargs: Any,
) -> Any:
    """Create and initialise the appropriate vector store.

    Args:
        use_qdrant: If True, attempt to connect to Qdrant first.
        host: Qdrant host.
        port: Qdrant port.
        embedding_model: Sentence-transformers model name.
        embedding_size: Output dimension of the embedding model.
        **kwargs: Additional arguments passed to the store constructor.

    Returns:
        An initialised ``QdrantVectorStore`` or ``InMemoryVectorStore``.
    """
    if use_qdrant and _QDRANT_AVAILABLE:
        store = QdrantVectorStore(
            host=host,
            port=port,
            embedding_model_name=embedding_model,
            embedding_size=embedding_size,
            **kwargs,
        )
        connected = await store.connect()
        if connected:
            logger.info("Vector store: Qdrant at %s:%d", host, port)
            return store
        logger.warning("Qdrant unavailable — falling back to in-memory store")

    store = InMemoryVectorStore(embedding_model_name=embedding_model)
    await store.initialize()
    logger.info("Vector store: in-memory")
    return store
