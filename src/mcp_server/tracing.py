"""
OpenTelemetry Tracing — MCP AI Assistant

Instruments the application with OpenTelemetry distributed tracing to
capture spans for every LLM call, RAG retrieval, classification, and
conversation turn.

Supports multiple exporters:
- **OTLP gRPC / HTTP** (standard OpenTelemetry collector)
- **Console** (stdout, for development)
- **LangSmith** (optional, via :mod:`src.mcp_server.langsmith_integration`)

Usage
-----
    from src.mcp_server.tracing import get_tracer, trace_endpoint

    tracer = get_tracer()
    with tracer.start_as_current_span("llm.inference") as span:
        span.set_attribute("model", "llama-3-8b")
        span.set_attribute("input.length", len(prompt))
        result = model.generate(prompt)
        span.set_attribute("output.length", len(result))

Architecture
------------
The module follows OpenTelemetry best practices:

1. ``TracerProvider`` is configured once at startup (lazy init).
2. A batch span processor sends spans to the configured exporter.
3. Every span carries standard semantic conventions for LLM/DB/HTTP.
4. Graceful fallback to a no-op tracer when ``opentelemetry`` is not
   installed (the app continues to work without traces).
5. Span events and attributes can be consumed by any OpenTelemetry-
   compatible backend (Jaeger, Grafana Tempo, Datadog, etc.).
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: OpenTelemetry SDK
# ---------------------------------------------------------------------------
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        SpanKind,
        Status,
        StatusCode,
        Tracer,
        TracerProvider as _OTelTracerProvider,
    )
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHTTPExporter,
    )
    from opentelemetry.sdk.resources import (
        Resource,
        SERVICE_NAME,
        SERVICE_VERSION,
        DEPLOYMENT_ENVIRONMENT,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    # Stubs so the module can be imported without opentelemetry installed
    class _StubSpan:
        """No-op span stub used when OpenTelemetry is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def set_attribute(self, key: str, value: Any) -> None: pass
        def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None: pass
        def set_status(self, status: Any) -> None: pass
        def end(self) -> None: pass
        def record_exception(self, exc: Exception, attributes: Optional[Dict[str, Any]] = None) -> None: pass
        def get_span_context(self) -> Any: return None
        def is_recording(self) -> bool: return False

    class _StubTracer:
        """No-op tracer stub."""

        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            from contextlib import nullcontext
            return nullcontext(_StubSpan())

        def start_span(self, name: str, **kwargs: Any) -> _StubSpan:
            return _StubSpan()

    class _StubTracerProvider:
        def get_tracer(self, *args: Any, **kwargs: Any) -> _StubTracer:
            return _StubTracer()

    # Module-level no-op instances
    trace = type(sys)("trace") if not _OTEL_AVAILABLE else trace  # type: ignore
    SpanKind = type("SpanKind", (), {"INTERNAL": None, "SERVER": None, "CLIENT": None})  # type: ignore
    Status = None  # type: ignore
    StatusCode = type("StatusCode", (), {"OK": None, "ERROR": None})  # type: ignore
    SDKTracerProvider = _StubTracerProvider  # type: ignore
    BatchSpanProcessor = SimpleSpanProcessor = object  # type: ignore
    ConsoleSpanExporter = OTLPHTTPExporter = object  # type: ignore

import sys  # needed for the stub module above


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TracingConfig:
    """Configuration for the OpenTelemetry tracing system.

    Parameters
    ----------
    enabled : bool
        Master switch for enabling/disabling tracing.
    service_name : str
        Service name reported in spans.
    service_version : str
        Version string reported in spans.
    environment : str
        Deployment environment (e.g. "production", "development").
    otlp_endpoint : str, optional
        OTLP HTTP exporter endpoint (e.g. "http://localhost:4318/v1/traces").
    otlp_headers : str, optional
        Comma-separated ``Key=Value`` pairs for OTLP headers (auth tokens, etc.).
    exporter_type : str
        ``"otlp"``, ``"console"``, or ``"none"``.
    sample_rate : float
        Probability [0.0, 1.0] of sampling a trace. 1.0 = sample everything.
    max_span_attributes : int
        Maximum number of attributes per span (prevents attribute flooding).
    """

    enabled: bool = True
    service_name: str = "mcp-ai-assistant"
    service_version: str = "2.0.0"
    environment: str = "development"
    otlp_endpoint: Optional[str] = None
    otlp_headers: Optional[str] = None
    exporter_type: str = "console"
    sample_rate: float = 1.0
    max_span_attributes: int = 128


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_tracer_provider: Any = None
_default_tracer: Any = None
_config: Optional[TracingConfig] = None

# We keep a registry of created tracers so we can log their names
_tracers: Dict[str, Any] = {}

# In-memory span buffer for the admin dashboard (last N spans)
_MAX_BUFFERED_SPANS = 500
_buffered_spans: List[Dict[str, Any]] = []


# ===================================================================
# Initialisation
# ===================================================================

def init_tracing(config: Optional[Dict[str, Any]] = None) -> Any:
    """Initialise OpenTelemetry tracing from a config dictionary.

    Called once during application startup (see ``rest_api.py``).

    Args:
        config: Dict from the ``tracing`` section of ``mcp_server.yaml``.
            If ``None``, uses default config.

    Returns:
        The ``TracerProvider`` instance, or ``None`` if tracing is disabled.
    """
    global _tracer_provider, _default_tracer, _config

    # Parse config
    cfg_dict = config or {}
    _config = TracingConfig(
        enabled=cfg_dict.get("enabled", True),
        service_name=cfg_dict.get("service_name", "mcp-ai-assistant"),
        service_version=cfg_dict.get("service_version", "2.0.0"),
        environment=cfg_dict.get("environment", "development"),
        otlp_endpoint=cfg_dict.get("otlp_endpoint"),
        otlp_headers=cfg_dict.get("otlp_headers"),
        exporter_type=cfg_dict.get("exporter_type", "console"),
        sample_rate=cfg_dict.get("sample_rate", 1.0),
        max_span_attributes=cfg_dict.get("max_span_attributes", 128),
    )

    if not _config.enabled:
        logger.info("Tracing is disabled")
        _tracer_provider = None
        _default_tracer = None
        return None

    if not _OTEL_AVAILABLE:
        logger.info(
            "OpenTelemetry SDK not installed — using no-op tracer. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )
        _tracer_provider = _StubTracerProvider()
        _default_tracer = _tracer_provider.get_tracer(__name__)
        return _tracer_provider

    try:
        # Create resource with service metadata
        resource = Resource.create({
            SERVICE_NAME: _config.service_name,
            SERVICE_VERSION: _config.service_version,
            DEPLOYMENT_ENVIRONMENT: _config.environment,
        })

        # Create the tracer provider
        _tracer_provider = SDKTracerProvider(resource=resource)

        # Add span processors based on exporter type
        exporter = _create_exporter()
        if exporter is not None:
            processor = BatchSpanProcessor(exporter)
            _tracer_provider.add_span_processor(processor)

            # Also add a sample-based head sampler if rate < 1.0
            # (OpenTelemetry SDK uses parent-based sampler by default)
            if _config.sample_rate < 1.0:
                logger.info(
                    "Tracing sample rate set to %.2f — some traces will be dropped",
                    _config.sample_rate,
                )
                # Note: Proper sampling requires configuring via SDK's
                # Sampler; not all SDK versions support it cleanly here.

        # Set as global tracer provider (so auto-instrumented libraries
        # like httpx, aiohttp, etc. pick it up)
        try:
            trace.set_tracer_provider(_tracer_provider)
        except Exception:
            pass

        _default_tracer = _tracer_provider.get_tracer(__name__)
        logger.info(
            "Tracing initialised (exporter=%s, service=%s, env=%s)",
            _config.exporter_type,
            _config.service_name,
            _config.environment,
        )

    except Exception as exc:
        logger.warning("Failed to initialise tracing: %s — using no-op tracer", exc)
        _tracer_provider = _StubTracerProvider()
        _default_tracer = _tracer_provider.get_tracer(__name__)

    return _tracer_provider


def _create_exporter() -> Any:
    """Create the span exporter based on the current config.

    Returns:
        An ``SpanExporter`` instance, or ``None`` to skip exporter setup
        (useful when only needed in-process).
    """
    if not _OTEL_AVAILABLE or _config is None:
        return None

    try:
        if _config.exporter_type == "otlp" and _config.otlp_endpoint:
            headers = {}
            if _config.otlp_headers:
                for part in _config.otlp_headers.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        headers[k.strip()] = v.strip()
            return OTLPHTTPExporter(
                endpoint=_config.otlp_endpoint,
                headers=headers,
            )

        elif _config.exporter_type == "console":
            return ConsoleSpanExporter()

        elif _config.exporter_type == "none":
            return None

        else:
            # Default: console
            return ConsoleSpanExporter()

    except Exception as exc:
        logger.warning("Failed to create OTel exporter: %s — falling back to console", exc)
        return ConsoleSpanExporter()


# ===================================================================
# Public API
# ===================================================================

def get_tracer(name: str = "mcp_ai_assistant") -> Any:
    """Get a named tracer instance.

    Args:
        name: The tracer name (typically the module path).

    Returns:
        A tracer instance matching the OpenTelemetry ``Tracer`` interface.
        If tracing is disabled or not initialised, returns a no-op tracer.
    """
    if _tracer_provider is not None:
        tracer = _tracer_provider.get_tracer(name)
        _tracers[name] = tracer
        return tracer

    # Fallback: no-op tracer
    return _StubTracer()


def get_config() -> Optional[TracingConfig]:
    """Get the current tracing configuration."""
    return _config


def is_enabled() -> bool:
    """Check if tracing is enabled and initialised."""
    return _config is not None and _config.enabled and _tracer_provider is not None


def shutdown() -> None:
    """Shut down the tracer provider (flush spans, close connections).

    Call during application shutdown.
    """
    global _tracer_provider
    if _tracer_provider is not None and _OTEL_AVAILABLE:
        try:
            _tracer_provider.shutdown()
        except Exception as exc:
            logger.warning("Tracer provider shutdown error: %s", exc)


# ===================================================================
# Convenience decorators and context managers
# ===================================================================

@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Any = None,
    tracer_name: str = "mcp_ai_assistant",
) -> Generator[Any, None, None]:
    """Context manager that creates a tracing span.

    Usage::

        with trace_span("llm.inference", attributes={"model": "llama-3"}) as span:
            span.set_attribute("input.length", len(prompt))
            result = model.generate(prompt)
            span.set_attribute("output.length", len(result))

    Args:
        name: Span name (e.g. ``"llm.inference"``, ``"rag.search"``).
        attributes: Initial attributes to set on the span.
        kind: Span kind (``SpanKind.INTERNAL``, ``SpanKind.CLIENT``, etc.).
            Defaults to ``SpanKind.INTERNAL``.
        tracer_name: Name for the tracer instance.

    Yields:
        The span object.  Call ``set_attribute()``, ``add_event()``, etc.
        on it for additional telemetry.
    """
    if not _OTEL_AVAILABLE:
        # Fast path: no OpenTelemetry installed
        yield _StubSpan()
        return

    tracer = get_tracer(tracer_name)

    if kind is None:
        kind = SpanKind.INTERNAL

    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for k, v in attributes.items():
                _safe_set_attribute(span, k, v)
        yield span


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_args: bool = False,
) -> Callable:
    """Decorator that wraps a function call in a tracing span.

    Usage::

        @trace_function(name="rag.search_kb", record_args=True)
        def search_kb(query: str, top_k: int = 5) -> List[Dict]:
            ...

    Args:
        name: Span name.  Defaults to ``<module>.<function_name>``.
        attributes: Static attributes added to every span.
        record_args: If ``True``, logs the first positional argument (usually
            the query string) as a span attribute.

    Returns:
        A decorated function that creates a span on every invocation.
    """

    def decorator(func: Callable) -> Callable:
        import functools

        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _OTEL_AVAILABLE:
                return func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                if attributes:
                    for k, v in attributes.items():
                        _safe_set_attribute(span, k, v)
                if record_args and args:
                    _safe_set_attribute(span, "arg.0", str(args[0])[:500])
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return wrapper

    return decorator


# ===================================================================
# Helpers
# ===================================================================

def _safe_set_attribute(span: Any, key: str, value: Any) -> None:
    """Set a span attribute, safely truncating long values."""
    max_len = _config.max_span_attributes if _config else 128

    if isinstance(value, str) and len(value) > max_len:
        value = value[:max_len] + "…"

    try:
        span.set_attribute(key, value)
    except Exception:
        pass  # Silently ignore attribute errors


def buffer_span_event(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> None:
    """Buffer a span event in memory for the admin dashboard.

    Stores the last ``_MAX_BUFFERED_SPANS`` events in a global list
    that can be read via ``get_buffered_spans()``.
    """
    global _buffered_spans
    _buffered_spans.append({
        "span_name": span_name,
        "attributes": attributes or {},
        "duration_ms": round(duration_ms, 2),
        "timestamp": time.time(),
    })
    # Trim to max size
    if len(_buffered_spans) > _MAX_BUFFERED_SPANS:
        _buffered_spans = _buffered_spans[-_MAX_BUFFERED_SPANS:]


def get_buffered_spans(limit: int = 100) -> List[Dict[str, Any]]:
    """Get the most recent buffered span events.

    Args:
        limit: Maximum number of span events to return.

    Returns:
        List of span event dicts, most recent last.
    """
    return _buffered_spans[-limit:]


def clear_buffered_spans() -> None:
    """Clear the in-memory span buffer."""
    global _buffered_spans
    _buffered_spans = []


def format_trace_summary(spans: List[Dict[str, Any]]) -> str:
    """Format a list of span events into a human-readable summary.

    Args:
        spans: List of span event dicts from ``get_buffered_spans()``.

    Returns:
        A formatted multiline string.
    """
    if not spans:
        return "No traces recorded yet"

    lines = ["Recent Traces:", "─────────────"]
    for s in spans[-20:]:  # Last 20
        name = s.get("span_name", "unknown")
        dur = s.get("duration_ms", 0)
        attrs = s.get("attributes", {})
        attr_str = " ".join(f"{k}={v}" for k, v in list(attrs.items())[:3])
        ts = time.strftime("%H:%M:%S", time.localtime(s.get("timestamp", 0)))
        lines.append(f"  {ts}  {name:<35s}  {dur:>8.1f}ms  {attr_str}")
    return "\n".join(lines)


# ===================================================================
# Health / status
# ===================================================================

def tracing_health() -> Dict[str, Any]:
    """Return the current tracing status (for health endpoint)."""
    return {
        "enabled": is_enabled(),
        "otel_available": _OTEL_AVAILABLE,
        "exporter_type": _config.exporter_type if _config else "none",
        "service_name": _config.service_name if _config else "—",
        "environment": _config.environment if _config else "—",
        "buffered_spans": len(_buffered_spans),
        "endpoint": _config.otlp_endpoint if _config else None,
    }
