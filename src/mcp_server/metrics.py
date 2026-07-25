"""
Prometheus Metrics for MCP AI Assistant

Central metric definitions used across the entire application.
Instrumented via ``prometheus-fastapi-instrumentator`` for automatic
HTTP metrics and custom counters/gauges/histograms for business logic.

Usage
-----
Import in ``rest_api.py``::

    from src.mcp_server.metrics import (
        setup_metrics,
        REQUEST_COUNT,
        REQUEST_LATENCY,
        ...
    )

Then ``.inc()`` / ``.observe()`` at the relevant points in the code.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load prometheus_client (optional dependency)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    from prometheus_fastapi_instrumentator import Instrumentator

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    # Stub classes so imports don't crash when prometheus is not installed
    class _StubMetric:
        """No-op stub used when prometheus_client is not installed.

        Accepts and ignores all constructor arguments so it can be used
        as a drop-in replacement for ``Counter``, ``Gauge``, ``Histogram``,
        and ``Info``.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: Any) -> None:
            pass

        def observe(self, amount: float) -> None:
            pass

        def info(self, info_dict: dict) -> None:
            pass

        def labels(self, **kwargs: Any) -> "_StubMetric":
            return self

        def clear(self) -> None:
            pass

    Counter = Gauge = Histogram = Info = _StubMetric  # type: ignore

    class _StubInstrumentator:
        """No-op stub for Instrumentator when prometheus is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def instrument(self, app: Any) -> "_StubInstrumentator":
            return self

        def expose(self, app: Any, **kwargs: Any) -> "_StubInstrumentator":
            return self

    Instrumentator = _StubInstrumentator  # type: ignore


# ===================================================================
# Metric Definitions
# ===================================================================

# -- HTTP Request Metrics (auto-instrumented by Instrumentator) --
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests by method, endpoint, and status",
    labelnames=["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds by endpoint",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

IN_FLIGHT_REQUESTS = Gauge(
    "http_requests_in_flight",
    "Number of HTTP requests currently being processed",
    labelnames=["method", "endpoint"],
)

# -- Rate Limiting --
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Total number of requests that were rate-limited (HTTP 429)",
    labelnames=["tier"],
)

RATE_LIMIT_REMAINING = Gauge(
    "rate_limit_remaining",
    "Number of remaining requests in the current rate-limit window",
    labelnames=["tier"],
)

# -- Cache --
CACHE_HITS = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    labelnames=["cache_type"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    labelnames=["cache_type"],
)

CACHE_SIZE = Gauge(
    "cache_size",
    "Current number of items in the cache",
    labelnames=["cache_type"],
)

# -- Classification --
CLASSIFICATIONS_TOTAL = Counter(
    "classifications_total",
    "Total number of ticket classifications by category and urgency",
    labelnames=["category", "urgency", "method"],
)

CLASSIFICATION_LATENCY = Histogram(
    "classification_duration_seconds",
    "Time taken to classify a ticket in seconds",
    labelnames=["method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# -- Conversation / Sessions --
ACTIVE_SESSIONS = Gauge(
    "conversation_active_sessions",
    "Number of active conversation sessions",
)

CONVERSATION_MESSAGES_TOTAL = Counter(
    "conversation_messages_total",
    "Total number of conversation messages processed",
    labelnames=["role"],
)

# -- Knowledge Base / RAG --
RAG_SEARCHES_TOTAL = Counter(
    "rag_searches_total",
    "Total number of RAG knowledge base searches",
    labelnames=["method"],
)

RAG_SEARCH_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "Time taken for RAG search in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RAG_RESULTS_RETURNED = Histogram(
    "rag_results_returned",
    "Number of results returned per RAG search",
    buckets=(1, 2, 3, 5, 10, 20, 50),
)

# -- Response Generation --
GENERATIONS_TOTAL = Counter(
    "generations_total",
    "Total number of response generations",
    labelnames=["method"],
)

GENERATION_LATENCY = Histogram(
    "generation_duration_seconds",
    "Time taken to generate a response in seconds",
    labelnames=["method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# -- Escalation (n8n) --
ESCALATION_EVENTS_TOTAL = Counter(
    "escalation_events_total",
    "Total number of escalation events triggered",
    labelnames=["urgency", "status"],
)

ESCALATION_LATENCY = Histogram(
    "escalation_duration_seconds",
    "Time taken to send escalation webhook in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# -- Feedback --
FEEDBACK_TOTAL = Counter(
    "feedback_total",
    "Total number of feedback submissions",
    labelnames=["rating"],
)

# -- Server Info (static metadata) --
SERVER_INFO = Info("mcp_server", "MCP AI Assistant server metadata")
SERVER_INFO.info({"version": "2.0.0", "name": "MCP AI Assistant"})


# ===================================================================
# Instrumentator setup
# ===================================================================
def setup_metrics(app: Any, metrics_enabled: bool = True) -> Optional[Any]:
    """Attach Prometheus instrumentator and expose ``/metrics``.

    Call this once when creating the FastAPI application.

    Args:
        app: The FastAPI application instance.
        metrics_enabled: If ``False``, skip instrumentation entirely.

    Returns:
        The ``Instrumentator`` instance, or ``None`` if disabled.
    """
    if not _PROMETHEUS_AVAILABLE or not metrics_enabled:
        if metrics_enabled:
            logger.warning(
                "prometheus-fastapi-instrumentator not installed. "
                "Install with: pip install prometheus-fastapi-instrumentator"
            )
        return None

    try:
        instrumentator = (
            Instrumentator()
            .instrument(app)
            .expose(app, endpoint="/metrics", include_in_schema=False)
        )
        logger.info("Prometheus metrics enabled at /metrics")
        return instrumentator
    except Exception as exc:
        logger.warning("Failed to set up Prometheus metrics: %s", exc)
        return None


# ===================================================================
# Convenience wrapper for endpoint instrumentation
# ===================================================================
# NOTE: Standard HTTP metrics (request count, latency, in-flight) are
# handled by the ``prometheus-fastapi-instrumentator`` via
# ``setup_metrics()``, which auto-instruments all FastAPI routes.
# Custom metrics below are for business-logic events only (classifications,
# cache hits, escalations, etc.) and are manually incremented in endpoint
# handlers in ``rest_api.py``. No custom ASGI middleware is needed.

# (The MetricsMiddleware has been removed because the Instrumentator
#  already tracks http_request_duration_seconds, http_requests_total,
#  etc. Adding a second middleware would cause duplicate metric
#  registration errors in Prometheus.)


# ===================================================================
# Health check for Prometheus endpoint
# ===================================================================
def metrics_health() -> Dict[str, Any]:
    """Return the status of the Prometheus metrics system."""
    return {
        "enabled": _PROMETHEUS_AVAILABLE,
        "library_available": _PROMETHEUS_AVAILABLE,
        "endpoint": "/metrics",
        "format": "Prometheus text/plain",
    }
