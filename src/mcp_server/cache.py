"""
Caching Layer

Provides caching for RAG search results, classification results, and
frequent queries. Uses Redis when available, falls back to in-memory
LRU cache.

This improves response times for repeated queries and reduces load on
the embedding model.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type variable for cached functions
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------
class LRUCache:
    """Simple in-memory LRU cache with TTL support.

    Thread-safe for read operations. Evicts the least recently used
    items when the cache exceeds ``max_size``.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache. Returns None if missing or expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]
        if time.time() > expiry:
            self._evict(key)
            self._misses += 1
            return None

        self._hits += 1
        self._cache.move_to_end(key)
        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache with optional TTL (seconds)."""
        ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
        self._cache.move_to_end(key)
        self._ensure_capacity()

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        return self._evict(key)

    def _evict(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def _ensure_capacity(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def keys(self) -> List[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------
def make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Create a deterministic cache key from arguments.

    Example::

        make_cache_key("rag", query="password reset", top_k=5)
        # -> "rag:hash_of_inputs"
    """
    raw = f"{repr(args)}:{repr(sorted(kwargs.items()))}"
    hash_digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{prefix}:{hash_digest}"


# ---------------------------------------------------------------------------
# Redis-backed cache
# ---------------------------------------------------------------------------
class RedisCache:
    """Cache backed by Redis for persistence across restarts.
    
    Handles serialization of complex types (dicts, lists) via JSON.
    """

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "cache:",
        default_ttl: int = 300,
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        try:
            data = await self._redis.get(self._full_key(key))
            if data is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(data)
        except Exception as exc:
            logger.debug("Redis cache get error: %s", exc)
            self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in Redis cache."""
        try:
            ttl = ttl if ttl is not None else self._default_ttl
            data = json.dumps(value, default=str)
            await self._redis.setex(self._full_key(key), ttl, data)
            return True
        except Exception as exc:
            logger.debug("Redis cache set error: %s", exc)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis cache."""
        try:
            result = await self._redis.delete(self._full_key(key))
            return result > 0
        except Exception as exc:
            logger.debug("Redis cache delete error: %s", exc)
            return False

    async def clear(self) -> bool:
        """Clear all cached entries with the configured prefix."""
        try:
            async for key in self._redis.scan_iter(match=f"{self._prefix}*"):
                await self._redis.delete(key)
            return True
        except Exception as exc:
            logger.debug("Redis cache clear error: %s", exc)
            return False

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics (best-effort)."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        count = 0
        try:
            async for _ in self._redis.scan_iter(match=f"{self._prefix}*"):
                count += 1
        except Exception:
            pass
        return {
            "size": count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }


# ---------------------------------------------------------------------------
# Cache usage patterns for this project
# ---------------------------------------------------------------------------

# Cache TTLs (seconds)
CACHE_TTL_RAG = 600         # 10 minutes for RAG results
CACHE_TTL_CLASSIFY = 300    # 5 minutes for classification
CACHE_TTL_GENERATE = 600    # 10 minutes for generated responses
CACHE_TTL_HEALTH = 30       # 30 seconds for health checks


def cache_key_rag(query: str, top_k: int, category_filter: Optional[str] = None) -> str:
    """Create a cache key for RAG search results."""
    return make_cache_key("rag", query=query.lower().strip(), top_k=top_k, category_filter=category_filter)


def cache_key_classify(ticket_text: str) -> str:
    """Create a cache key for ticket classification."""
    return make_cache_key("classify", ticket=ticket_text.lower().strip())


def cache_key_generate(ticket_text: str, category: str, urgency: str) -> str:
    """Create a cache key for response generation."""
    return make_cache_key(
        "generate",
        ticket=ticket_text.lower().strip(),
        category=category.lower(),
        urgency=urgency.lower(),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_cache(
    use_redis: bool = False,
    redis_client: Any = None,
    max_size: int = 1000,
    default_ttl: int = 300,
) -> Any:
    """Create the appropriate cache backend."""
    if use_redis and redis_client is not None:
        try:
            return RedisCache(redis_client, default_ttl=default_ttl)
        except Exception as exc:
            logger.warning("Redis cache init failed (%s) — using LRU cache", exc)
    return LRUCache(max_size=max_size, default_ttl=default_ttl)
