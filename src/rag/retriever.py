"""
RAG Knowledge Base Retriever

Searches support articles using sentence-transformers embeddings + cosine similarity.
Falls back to simple keyword matching when sentence-transformers is unavailable.
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

    Uses dense embeddings (sentence-transformers) with cosine similarity
    when available, falling back to TF-IDF-like keyword scoring when only
    ``numpy`` is installed, and finally to pure Python keyword matching
    when neither is available.
    """

    def __init__(
        self,
        kb_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self.kb_path = kb_path or DEFAULT_KB_PATH
        self.top_k = top_k
        self._articles: List[Dict[str, Any]] = []
        self._embeddings: Optional[list] = None
        self._model: Optional[Any] = None
        self._model_name = embedding_model

        self._load_articles()
        self._build_index()

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
        """True if the dense embedding model is loaded and ready."""
        return self._model is not None and self._embeddings is not None

    def article_count(self) -> int:
        return len(self._articles)

    def reload(self) -> None:
        """Reload articles and rebuild the index."""
        self._load_articles()
        self._build_index()
