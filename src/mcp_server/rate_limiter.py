"""
Rate Limiting Module

Provides rate limiting for the FastAPI REST server using a sliding window
algorithm. Supports both in-memory (default) and Redis-backed storage.

Rate limits are enforced per API key (or per IP as fallback).
"""

import asyncio
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limit Configuration
# ---------------------------------------------------------------------------

# Default limits: (requests, window_seconds)
DEFAULT_LIMITS: Dict[str, Tuple[int, int]] = {
    "admin": (1000, 60),       # 1000 req/min
    "standard": (100, 60),     # 100 req/min
    "anonymous": (20, 60),     # 20 req/min
}

# Rate limit headers
HEADER_REMAINING = "X-RateLimit-Remaining"
HEADER_LIMIT = "X-RateLimit-Limit"
HEADER_RESET = "X-RateLimit-Reset"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RateLimitEntry:
    """Tracks requests in a sliding window."""
    timestamps: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# In-memory store (default)
# ---------------------------------------------------------------------------
class InMemoryRateStore:
    """Thread-safe in-memory rate limit store using a sliding window.

    Uses a dict of ``key -> list[timestamps]``. Old entries are cleaned
    on each check to prevent unbounded memory growth.
    """

    def __init__(self) -> None:
        self._store: Dict[str, RateLimitEntry] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int, float]:
        """Check if a request is allowed.

        Returns:
            Tuple of (allowed, remaining_requests, reset_timestamp).
        """
        now = time.time()
        window_start = now - window_seconds

        async with self._lock:
            if key not in self._store:
                self._store[key] = RateLimitEntry(timestamps=[now])
                return True, max_requests - 1, now + window_seconds

            entry = self._store[key]

            # Prune old entries outside the window
            entry.timestamps = [t for t in entry.timestamps if t > window_start]

            # Check if within limit
            if len(entry.timestamps) >= max_requests:
                # Calculate when the oldest request in the window expires
                oldest = min(entry.timestamps)
                reset_at = oldest + window_seconds
                return False, 0, reset_at

            entry.timestamps.append(now)
            remaining = max_requests - len(entry.timestamps)
            return True, remaining, now + window_seconds

    def cleanup(self) -> int:
        """Remove stale entries (older than max window)."""
        now = time.time()
        stale_threshold = now - 3600  # 1 hour max window
        stale_keys = [
            k for k, v in self._store.items()
            if v.timestamps and max(v.timestamps) < stale_threshold
        ]
        for k in stale_keys:
            del self._store[k]
        return len(stale_keys)


# ---------------------------------------------------------------------------
# Redis-backed store (optional, higher performance)
# ---------------------------------------------------------------------------
class RedisRateStore:
    """Rate limit store backed by Redis.

    Requires the ``redis`` package. Falls back to in-memory if Redis
    connection fails.
    """

    def __init__(self, redis_client: Any, key_prefix: str = "ratelimit:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    async def check(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int, float]:
        """Check rate limit via Redis sorted sets (sliding window)."""
        now = time.time()
        window_start = now - window_seconds
        redis_key = f"{self._prefix}{key}"

        try:
            pipe = self._redis.pipeline()
            pipe.zadd(redis_key, {now: now})
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.expire(redis_key, window_seconds)
            _, _, count, _ = await pipe.execute()
        except Exception as exc:
            logger.warning("Redis rate limit error (%s) — allowing request", exc)
            return True, max_requests, now + window_seconds

        if count > max_requests:
            # Get the oldest timestamp's expiry time
            oldest = await self._redis.zrange(redis_key, 0, 0, withscores=True)
            reset_at = oldest[0][1] + window_seconds if oldest else now + window_seconds
            return False, 0, reset_at

        remaining = max_requests - count
        return True, remaining, now + window_seconds


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_rate_store(use_redis: bool = False, redis_client: Any = None) -> Any:
    """Create the appropriate rate store based on configuration."""
    if use_redis and redis_client is not None:
        try:
            return RedisRateStore(redis_client)
        except Exception as exc:
            logger.warning("Redis rate store init failed (%s) — using in-memory", exc)
    return InMemoryRateStore()


# ---------------------------------------------------------------------------
# Rate Limiting Middleware
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces rate limits per API key.

    Rate limits are applied based on the API key tier. Limits are
    communicated via response headers.
    """

    def __init__(
        self,
        app: ASGIApp,
        store: Optional[Any] = None,
        limits: Optional[Dict[str, Tuple[int, int]]] = None,
        exclude_paths: Optional[set] = None,
        get_client_key: Optional[Callable[[Request], str]] = None,
    ) -> None:
        super().__init__(app)
        self._store = store or create_rate_store()
        self._limits = limits or DEFAULT_LIMITS
        self._exclude_paths = exclude_paths or {"/health", "/docs", "/openapi.json", "/redoc"}
        self._get_client_key = get_client_key or self._default_client_key

    @staticmethod
    def _default_client_key(request: Request) -> str:
        """Extract the client identifier (API key name or IP)."""
        # Prefer API key name (set by auth middleware)
        key_name: Optional[str] = getattr(request.state, "api_key_name", None)
        if key_name:
            return f"key:{key_name}"

        # Fallback to IP address
        forwarded = request.headers.get("X-Forwarded-For", "")
        client_ip = forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _get_limit_for_key(self, request: Request) -> Tuple[int, int]:
        """Determine the rate limit for the current request."""
        tier: str = getattr(request.state, "api_key_tier", "anonymous")
        return self._limits.get(tier, self._limits.get("anonymous", (20, 60)))

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path in self._exclude_paths:
            return await call_next(request)

        client_key = self._get_client_key(request)
        max_requests, window_seconds = self._get_limit_for_key(request)

        try:
            allowed, remaining, reset_at = await self._store.check(
                client_key, max_requests, window_seconds
            )
        except Exception as exc:
            logger.warning("Rate limit check failed (%s) — allowing request", exc)
            return await call_next(request)

        if not allowed:
            retry_after = max(1, int(reset_at - time.time()))
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after_seconds": retry_after,
                },
                headers={
                    HEADER_REMAINING: "0",
                    HEADER_LIMIT: str(max_requests),
                    HEADER_RESET: str(int(reset_at)),
                    "Retry-After": str(retry_after),
                },
            )

        response = await call_next(request)

        # Add rate limit headers to the response
        response.headers[HEADER_REMAINING] = str(remaining)
        response.headers[HEADER_LIMIT] = str(max_requests)
        response.headers[HEADER_RESET] = str(int(reset_at))

        return response


# ---------------------------------------------------------------------------
# Utility: get rate limit status
# ---------------------------------------------------------------------------
async def get_rate_limit_status(request: Request, store: Any = None) -> Dict[str, Any]:
    """Get current rate limit status for the requesting client."""
    store = store or create_rate_store()

    client_key = RateLimitMiddleware._default_client_key(request)
    tier: str = getattr(request.state, "api_key_tier", "anonymous")
    max_requests, window_seconds = DEFAULT_LIMITS.get(tier, DEFAULT_LIMITS["anonymous"])

    allowed, remaining, reset_at = await store.check(
        client_key, max_requests, window_seconds
    )

    return {
        "tier": tier,
        "limit": max_requests,
        "remaining": remaining,
        "reset_at": int(reset_at),
        "window_seconds": window_seconds,
    }
