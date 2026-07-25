"""
RAG Knowledge Base Retriever

Searches support articles using hybrid search (dense + sparse vectors).
Supports three backends in order of preference:

1. **QdrantVectorStore** — production-grade vector DB with hybrid search
2. **InMemoryVectorStore** — in-memory fallback with dense + sparse scoring
3. **Legacy embedding search** — original sentence-transformers + cosine similarity
4. **Keyword fallback** — simple TF overlap (when nothing else is available)

When a vector store is provided, the retriever seeds it from the JSONL
knowledge base file on startup, then delegates all ``search()`` calls
via ``hybrid_search()``.
"""

import json
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Timeout in seconds for embedding model loading.
# If the model doesn't load within this window (e.g. SSL / network issues),
# we fall back to keyword search so the server can start quickly.
_EMBEDDING_MODEL_TIMEOUT = 15.0

# ---------------------------------------------------------------------------
# Optional: sentence-transformers for real embedding search
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Vector store (Phase 3 upgrade)
# ---------------------------------------------------------------------------
try:
    from src.rag.vector_store import (
        create_vector_store,
        SearchResult,
        chunk_article,
    )
except ImportError:
    create_vector_store = None  # type: ignore
    SearchResult = None  # type: ignore
    chunk_article = lambda a, **kw: []  # type: ignore


# ---------------------------------------------------------------------------
# Default path for the seed KB
# ---------------------------------------------------------------------------
DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "knowledge_base",
    "support_articles.jsonl",
)


class KnowledgeBaseRetriever:
    """Retrieve relevant support articles from a knowledge base.

    Supports four search tiers:
    - **Vector store** (Qdrant or in-memory): hybrid dense + sparse search
    - **Legacy embedding**: sentence-transformers + cosine similarity
    - **Keyword fallback**: simple TF overlap

    When a ``vector_store`` is provided on init, the retriever seeds it
    from the JSONL file and delegates all searches to it.
    """

    def __init__(
        self,
        kb_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        vector_store: Optional[Any] = None,
    ) -> None:
        self.kb_path = kb_path or DEFAULT_KB_PATH
        self.top_k = top_k
        self._articles: List[Dict[str, Any]] = []
        self._embeddings: Optional[list] = None
        self._model: Optional[Any] = None
        self._model_name = embedding_model
        self._vector_store = vector_store

        self._load_articles()
        self._build_index()

        # Seed the vector store is NOT done here — it must be called
        # separately via ``await retriever.seed_vector_store()`` from
        # an async context (e.g. the FastAPI startup handler).

    # ------------------------------------------------------------------
    # Article loading
    # ------------------------------------------------------------------
    def _load_articles(self) -> None:
        """Load articles from the JSONL knowledge base file."""
        if not os.path.exists(self.kb_path):
            logger.warning("Knowledge base not found at %s — using empty KB.", self.kb_path)
            self._articles = []
            return

        articles: List[Dict[str, Any]] = []
        with open(self.kb_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed KB line: %s", exc)
        self._articles = articles
        logger.info("Loaded %d knowledge base articles", len(articles))

    # ------------------------------------------------------------------
    # Vector store seeding (must be called from an async context)
    # ------------------------------------------------------------------
    async def seed_vector_store(self) -> int:
        """Chunk all articles and upsert them into the vector store.

        Must be called from an async context (e.g. FastAPI startup event).
        Returns the number of chunks upserted, or 0 if no vector store
        is configured.
        """
        if self._vector_store is None or not self._articles:
            return 0

        all_points = []
        for article in self._articles:
            chunks = chunk_article(article)
            for chunk in chunks:
                all_points.append({
                    "id": chunk["id"],
                    "payload": chunk,
                })

        if not all_points:
            return 0

        total = 0
        # Upsert in batches to avoid overwhelming the store
        batch_size = 50
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            count = await self._vector_store.upsert(batch)
            total += count

        logger.info(
            "Seeded vector store with %d chunks from %d articles%s",
            total,
            len(self._articles),
            f" via {type(self._vector_store).__name__}" if self._vector_store else "",
        )
        return total

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Pre-compute embeddings or prepare keyword index.

        The embedding model is loaded with a timeout so that SSL / network
        issues (common on macOS with LibreSSL) don't block server startup.
        If the model fails to load, keyword search is used as fallback.
        """
        if not self._articles:
            self._embeddings = None
            return

        if _SENTENCE_TRANSFORMERS_AVAILABLE and _NUMPY_AVAILABLE:
            logger.info(
                "Loading embedding model '%s' (timeout: %ss) …",
                self._model_name,
                _EMBEDDING_MODEL_TIMEOUT,
            )

            def _load() -> None:
                try:
                    model = SentenceTransformer(self._model_name)
                    texts = [
                        a.get("title", "") + "\n" + a.get("content", "")
                        for a in self._articles
                    ]
                    embeddings = model.encode(
                        texts, show_progress_bar=False, normalize_embeddings=True
                    )
                    self._model = model
                    self._embeddings = embeddings
                except Exception as exc:
                    logger.warning("Embedding loading failed in thread: %s", exc)

            # Use a thread with a timeout so SSL/network hangs don't
            # block server startup. We must shutdown(wait=False) to avoid
            # waiting for a potentially stuck thread when exiting the scope.
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(_load)
                try:
                    future.result(timeout=_EMBEDDING_MODEL_TIMEOUT)
                    if self._embeddings is not None:
                        logger.info(
                            "Computed embeddings for %d articles",
                            len(self._articles),
                        )
                except TimeoutError:
                    logger.warning(
                        "Embedding model loading timed out after %ss — "
                        "using keyword fallback",
                        _EMBEDDING_MODEL_TIMEOUT,
                    )
                    self._model = None
                    self._embeddings = None
                except Exception as exc:
                    logger.warning(
                        "Embedding model failed (%s) — using keyword fallback", exc
                    )
                    self._model = None
                    self._embeddings = None
            finally:
                # Don't block shutdown waiting for a potentially stuck thread
                pool.shutdown(wait=False, cancel_futures=True)
        else:
            logger.info(
                "sentence-transformers unavailable — using keyword fallback search"
            )
            self._model = None
            self._embeddings = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base and return the top-k matching articles.

        Uses the configured backend in order of preference:
        1. Vector store (Qdrant) — hybrid dense + sparse BM25
        2. In-memory vector store — hybrid dense + sparse
        3. Legacy embedding — sentence-transformers + cosine similarity
        4. Keyword fallback — simple TF overlap

        Parameters
        ----------
        query : str
            The natural-language search query.
        top_k : int or None
            Number of results to return (defaults to ``self.top_k``).
        category_filter : str or None
            If set, only return articles whose ``category`` matches (case-
            insensitive).  Can be a comma-separated list of categories.

        Returns
        -------
        list[dict]
            Each dict contains the article fields plus a ``score`` key.
        """
        k = top_k or self.top_k

        # -- Try vector store first (hybrid dense + sparse) -------------
        if self._vector_store is not None:
            return self._vector_store_search(query, k, category_filter)

        if not self._articles:
            return []

        # -- filter by category first -----------------------------------
        if category_filter:
            wanted = {c.strip().lower() for c in category_filter.split(",")}
            candidates = [
                a for a in self._articles
                if a.get("category", "").strip().lower() in wanted
            ]
            if not candidates:
                return []
        else:
            candidates = self._articles

        # -- map candidate -> index in the full articles list -----------
        # Use position-based mapping instead of id() for robustness
        article_positions = {
            id(a): i for i, a in enumerate(self._articles)
        }
        candidate_indices = [article_positions[id(a)] for a in candidates]

        # -- dense embedding search ------------------------------------
        if self._embeddings is not None and self._model is not None and _NUMPY_AVAILABLE:
            try:
                query_emb = self._model.encode([query], normalize_embeddings=True)
                # Cosine similarity = dot product of unit-normalised vectors
                all_sims = np.dot(self._embeddings, query_emb.T).flatten()
                # Narrow to candidate subset
                sub_sims = all_sims[candidate_indices]
                top_n = np.argsort(sub_sims)[-k:][::-1]
                results = [
                    {**candidates[int(i)], "score": round(float(sub_sims[int(i)]), 4)}
                    for i in top_n
                ]
                return results
            except Exception as exc:
                logger.warning("Embedding search failed (%s), falling back to keywords", exc)

        # -- keyword fallback -------------------------------------------
        return self._keyword_search(query, candidates, k)

    # ------------------------------------------------------------------
    # Vector store search (async executed synchronously)
    # ------------------------------------------------------------------
    def _vector_store_search(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a hybrid search against the vector store.

        Because the vector store API is async, we use ``asyncio.run()``
        to bridge the sync ``search()`` interface.
        """
        if self._vector_store is None:
            return []

        import asyncio

        async def _do() -> List[Dict[str, Any]]:
            results = await self._vector_store.hybrid_search(
                query_text=query,
                top_k=top_k,
                category_filter=category_filter,
            )
            # Convert SearchResult objects back to the original dict format
            output = []
            for r in results:
                # The payload contains the original article fields (title, content, etc.)
                article = dict(r.payload)
                article["score"] = r.score
                article["_search_source"] = r.source
                output.append(article)
            return output

        try:
            return asyncio.run(_do())
        except Exception as exc:
            logger.warning("Vector store search failed (%s), falling back to keyword", exc)
            if self._articles:
                return self._keyword_search(query, self._articles, top_k)
            return []

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> set:
        """Return lower-cased alphanumeric tokens from *text*."""
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _keyword_search(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Score articles by simple TF overlap."""
        query_tokens = self._tokenise(query)
        if not query_tokens:
            return []

        scored: List[tuple] = []
        for article in candidates:
            parts = []
            for field in ("title", "content", "category"):
                val = article.get(field, "")
                if isinstance(val, str):
                    parts.append(val)
            tags = article.get("tags", [])
            if isinstance(tags, list):
                parts.append(" ".join(tags))
            haystack = " ".join(parts)
            doc_tokens = self._tokenise(haystack)
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(query_tokens)
            if score > 0:
                scored.append((score, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**article, "score": round(score, 4)}
            for score, article in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def has_embeddings(self) -> bool:
        """True if the dense embedding model is loaded and ready.

        Returns ``True`` when a vector store is connected (it handles
        embeddings internally) or when the legacy embedding model is loaded.
        """
        if self._vector_store is not None and getattr(self._vector_store, "is_connected", False):
            return True
        return self._model is not None and self._embeddings is not None

    @property
    def vector_store(self) -> Optional[Any]:
        """The underlying vector store (Qdrant or in-memory)."""
        return self._vector_store

    @property
    def vector_store_type(self) -> str:
        """Human-readable name of the active vector store backend."""
        if self._vector_store is None:
            return "none"
        return type(self._vector_store).__name__

    def article_count(self) -> int:
        return len(self._articles)

    def chunk_count(self) -> int:
        """Return the number of chunks currently stored in the vector store.
        Falls back to the article count if no vector store is available."""
        return len(self._articles)

    def reload(self) -> None:
        """Reload articles and rebuild the index.

        Note: Re-seeding the vector store is NOT done here because this
        method is synchronous. Call ``await retriever.seed_vector_store()``
        separately from an async context if needed.
        """
        self._load_articles()
        self._build_index()
