"""
FastAPI REST wrapper for the MCP-powered AI assistant.

Provides plain HTTP endpoints (no SSE, no JSON-RPC) so n8n workflows
can call the assistant with standard POST requests.

Endpoints
---------
- GET  /health                              → health check
- POST /api/classify_ticket                 → classify ticket (category + urgency) [auto-escalates critical]
- POST /api/generate_response               → generate support response
- POST /api/search_kb                       → RAG search over knowledge base
- POST /api/resolve_ticket                  → resolve / summarise a ticket
- POST /api/feedback_log                    → log feedback for future re-tuning
- POST /api/conversation/message            → send message with conversation memory
- GET  /api/sessions                        → list active conversation sessions
- DELETE /api/sessions/{session_id}         → clear a conversation session
- GET  /api/conversation/{session_id}       → get conversation history
- GET  /api/rate_limit/status               → get current rate limit status
- GET  /api/cache/stats                     → get cache statistics (admin)
- POST /api/admin/keys                      → generate a new API key (admin)
- POST /api/escalate                        → auto-escalate a critical ticket via n8n [NEW]
- GET  /api/escalate/status                 → n8n escalation health check [NEW]
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# FastAPI / uvicorn / pydantic
# ---------------------------------------------------------------------------
try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.rag.retriever import KnowledgeBaseRetriever
from src.rag.vector_store import create_vector_store, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_SIZE

# ---------------------------------------------------------------------------
# New modules
# ---------------------------------------------------------------------------
from src.mcp_server.auth import (
    APIKeyMiddleware,
    require_admin_key,
    get_key_store,
    TIER_ADMIN,
    TIER_STANDARD,
)
from src.mcp_server.rate_limiter import (
    RateLimitMiddleware,
    create_rate_store,
)
from src.mcp_server.conversation import (
    Message,
    Conversation,
    InMemoryConversationStore,
    RedisConversationStore,
    build_conversation_context,
    create_conversation_store,
)
from src.mcp_server.cache import (
    create_cache,
    cache_key_rag,
    cache_key_classify,
    cache_key_generate,
    CACHE_TTL_RAG,
    CACHE_TTL_CLASSIFY,
    CACHE_TTL_GENERATE,
    CACHE_TTL_HEALTH,
)
from src.mcp_server.n8n_client import (
    N8nWebhookClient,
    get_n8n_client,
    reset_n8n_client,
)
from src.mcp_server.admin_dashboard import mount_dashboard
from src.mcp_server.metrics import (
    setup_metrics,
    CACHE_HITS,
    CACHE_MISSES,
    CLASSIFICATIONS_TOTAL,
    CLASSIFICATION_LATENCY,
    ACTIVE_SESSIONS,
    CONVERSATION_MESSAGES_TOTAL,
    RAG_SEARCHES_TOTAL,
    RAG_SEARCH_LATENCY,
    RAG_RESULTS_RETURNED,
    GENERATIONS_TOTAL,
    GENERATION_LATENCY,
    ESCALATION_EVENTS_TOTAL,
    ESCALATION_LATENCY,
    FEEDBACK_TOTAL,
)
from src.mcp_server.guardrails import (
    PIISanitizer,
    SafetyChecker,
    GuardrailAction,
    get_pii_sanitizer,
    get_safety_checker,
)
from src.mcp_server.dpo_worker import (
    try_schedule_pipeline,
    get_dpo_status,
)
from src.eval.ragas_eval import (  # RAG evaluation
    load_results as load_ragas_results,
    compare_results as compare_ragas_results,
    DEFAULT_RESULTS_PATH as RAGAS_RESULTS_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_model_assistant: Optional[Any] = None
_rag_retriever: Optional[KnowledgeBaseRetriever] = None
_vector_store: Optional[Any] = None
_conversation_store: Optional[Any] = None
_cache: Optional[Any] = None
_rate_store: Optional[Any] = None
_redis_client: Optional[Any] = None
_n8n: Optional[N8nWebhookClient] = None
_config: Dict[str, Any] = {}
_pii_sanitizer: Optional[PIISanitizer] = None
_safety_checker: Optional[SafetyChecker] = None

# ===================================================================
# Pydantic request / response models
# ===================================================================

# -- Classify ticket -------------------------------------------------
class ClassifyRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1, description="The raw ticket text")

class ClassifyResponse(BaseModel):
    category: str
    urgency: str
    raw: str
    cached: bool = False

# -- Generate response -----------------------------------------------
class GenerateRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1)
    category: str = ""
    urgency: str = ""

class GenerateResponse(BaseModel):
    response: str
    raw: str
    cached: bool = False

# -- Search KB -------------------------------------------------------
class SearchKBRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    category_filter: Optional[str] = None

class SearchKBResult(BaseModel):
    id: str
    title: str
    category: str
    tags: List[str]
    content: str
    score: float

class SearchKBResponse(BaseModel):
    query: str
    results: List[SearchKBResult]
    total_found: int
    method: str
    cached: bool = False

# -- Resolve ticket --------------------------------------------------
class ResolveRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1)
    category: str = ""
    urgency: str = ""
    resolution_notes: str = ""

class ResolveResponse(BaseModel):
    ticket_id: str
    status: str
    summary: str

# -- Feedback log ----------------------------------------------------
class FeedbackRequest(BaseModel):
    data: dict = Field(default_factory=dict)

class FeedbackResponse(BaseModel):
    status: str
    ticket_id: str
    rating: str

# -- Conversation ----------------------------------------------------
class ConversationMessageRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session ID (omit to start a new conversation)")
    message: str = Field(..., min_length=1, description="The user's message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

class ConversationMessageResponse(BaseModel):
    session_id: str
    reply: str
    history: List[Dict[str, Any]] = Field(default_factory=list)

class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    message_count: int

# -- Admin -----------------------------------------------------------
class GenerateKeyResponse(BaseModel):
    api_key: str
    name: str
    tier: str

# -- Escalation ------------------------------------------------------
class EscalateRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1, description="The raw ticket text")
    category: str = Field(default="", description="Ticket category")
    urgency: str = Field(default="critical", description="Ticket urgency")
    classification_raw: str = Field(default="", description="Raw classification output")
    ticket_id: Optional[str] = Field(default=None, description="Existing ticket ID (auto-generated if empty)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

class EscalateResponse(BaseModel):
    status: str
    ticket_id: str
    escalation: Dict[str, Any]
    n8n_configured: bool

# ===================================================================
# Redis client helper
# ===================================================================

def _create_redis_client() -> Optional[Any]:
    """Create a Redis async client from the global config.

    Reads ``redis`` section from ``_config`` and returns an async
    Redis connection, or ``None`` if ``redis`` is not installed.
    """
    redis_config = _config.get("redis", {})
    host = redis_config.get("host", "localhost")
    port = redis_config.get("port", 6379)
    db = redis_config.get("db", 0)
    password = redis_config.get("password") or None

    try:
        import redis.asyncio as aioredis

        client = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info("Redis client created (host=%s:%d, db=%d)", host, port, db)
        return client
    except ImportError:
        logger.warning(
            "redis package not installed — Redis features disabled. "
            "Install with: pip install redis[hiredis]"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to create Redis client (%s) — Redis features disabled", exc)
        return None


# ===================================================================
# FastAPI application
# ===================================================================

def create_app(custom_config: Optional[Dict[str, Any]] = None) -> "FastAPI":
    """Build and return a fully configured FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi and uvicorn are required.  Install them with:\n"
            "  pip install fastapi uvicorn pydantic"
        )

    # Load config from the central config
    global _config
    _config = custom_config or _load_server_config()

    app = FastAPI(
        title="MCP AI Assistant — REST API",
        version="2.0.0",
        description="HTTP endpoints for ticket classification, response generation, "
        "RAG-powered knowledge base search, conversation memory, and more.",
    )

    # -- CORS (allow n8n from any origin) --------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Authentication middleware (skip if disabled) --------------------
    auth_config = _config.get("auth", {})
    if auth_config.get("enabled", True):
        app.add_middleware(
            APIKeyMiddleware,
            exclude_paths=set(auth_config.get("public_paths", ["/health", "/docs", "/openapi.json", "/redoc"])),
        )

    # -- Prometheus metrics (skip if disabled) -------------------------
    metrics_config = _config.get("prometheus", {})
    metrics_enabled = metrics_config.get("enabled", True)
    if metrics_enabled:
        # Setup the Instrumentator to auto-collect HTTP metrics + expose /metrics
        setup_metrics(app, metrics_enabled=True)

    # -- Rate limiting middleware (skip if disabled) --------------------
    rate_limit_config = _config.get("rate_limiting", {})
    if rate_limit_config.get("enabled", True):
        global _rate_store, _redis_client
        # Create Redis client if rate limiting uses Redis
        if rate_limit_config.get("redis_enabled", False):
            _redis_client = _create_redis_client()
        _rate_store = create_rate_store(
            use_redis=rate_limit_config.get("redis_enabled", False),
            redis_client=_redis_client,
        )
        app.add_middleware(
            RateLimitMiddleware,
            store=_rate_store,
            limits=rate_limit_config.get("limits"),
        )

    # -- Startup: load RAG, model, conversation store, cache, n8n ------
    @app.on_event("startup")
    async def startup() -> None:
        global _rag_retriever, _vector_store, _model_assistant, _conversation_store, _cache, _redis_client, _n8n, _pii_sanitizer, _safety_checker

        # Set Prometheus server info
        try:
            from src.mcp_server.metrics import SERVER_INFO
            SERVER_INFO.info({
                "version": "2.0.0",
                "name": "MCP AI Assistant",
                "metrics": "/metrics",
            })
        except Exception:
            pass

        # ---- Initialise guardrails (Phase 4: PII masking & safety) -----------
        logger.info("Initialising guardrails …")
        try:
            _pii_sanitizer = get_pii_sanitizer(_config)
            _safety_checker = get_safety_checker(_config)
            logger.info(
                "Guardrails ready (PII=%s, Safety=%s)",
                _pii_sanitizer.status.get("engine", "unknown"),
                _safety_checker.status.get("engine", "unknown"),
            )
        except Exception as exc:
            logger.warning("Could not initialise guardrails: %s", exc)
            _pii_sanitizer = PIISanitizer(enabled=False)
            _safety_checker = SafetyChecker(enabled=False)

        # ---- Initialise vector database (Phase 3: Qdrant hybrid search) ----
        logger.info("Initialising vector database …")
        vector_db_config = _config.get("vector_db", {})
        vector_db_enabled = vector_db_config.get("enabled", True)
        try:
            _vector_store = await create_vector_store(
                use_qdrant=vector_db_enabled and vector_db_config.get("provider", "qdrant") == "qdrant",
                host=vector_db_config.get("host", "localhost"),
                port=vector_db_config.get("port", 6333),
                embedding_model=vector_db_config.get("embedding_model", "all-MiniLM-L6-v2"),
                embedding_size=vector_db_config.get("embedding_size", 384),
                grpc_port=vector_db_config.get("grpc_port", 6334),
                prefer_grpc=vector_db_config.get("prefer_grpc", False),
                api_key=vector_db_config.get("api_key"),
                collection_name=vector_db_config.get("collection_name", "support_articles"),
                connect_timeout=vector_db_config.get("connect_timeout", 3.0),
            )
            store_type = type(_vector_store).__name__ if _vector_store else "none"
            logger.info("Vector database ready (type=%s)", store_type)
        except Exception as exc:
            logger.warning("Could not initialise vector database: %s", exc)
            _vector_store = None

        # ---- Initialise RAG retriever with vector store backend -----------
        logger.info("Initialising knowledge base retriever …")
        try:
            rag_config = _config.get("rag", {})
            _rag_retriever = KnowledgeBaseRetriever(
                kb_path=rag_config.get("kb_path"),
                embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
                top_k=rag_config.get("top_k", 5),
                vector_store=_vector_store,
            )
            # Seed the vector store with KB articles (async, safe to await here)
            if _vector_store is not None:
                seeded = await _rag_retriever.seed_vector_store()
                logger.info("Vector store seeded with %d chunks", seeded)
            logger.info("RAG retriever ready (%d articles, backend=%s)",
                        _rag_retriever.article_count(),
                        _rag_retriever.vector_store_type)
        except Exception as exc:
            logger.warning("Could not initialise RAG retriever: %s", exc)
            _rag_retriever = None

        logger.info("Loading classification model …")
        try:
            from src.mcp_server.server import SupportAssistant, load_config

            cfg = load_config()
            _model_assistant = SupportAssistant(cfg)
            # Check if the underlying model actually loaded (not just the wrapper)
            if _model_assistant.model is None:
                logger.warning("Model wrapper created but underlying LLM not loaded — using fallback")
                _model_assistant = None
            else:
                logger.info("Classification model loaded")
        except Exception as exc:
            logger.warning("Could not load model (will use regex fallback): %s", exc)
            _model_assistant = None

        # Create shared Redis client if any component needs it
        if _redis_client is None:
            conv_redis = _config.get("conversation", {}).get("redis_enabled", False)
            cache_redis = _config.get("cache", {}).get("redis_enabled", False)
            if conv_redis or cache_redis:
                _redis_client = _create_redis_client()

        # Conversation store
        conv_config = _config.get("conversation", {})
        persistence_path = conv_config.get("persistence_path")
        if persistence_path:
            persistence_path = os.path.join(_project_root, persistence_path)
        _conversation_store = create_conversation_store(
            use_redis=conv_config.get("redis_enabled", False),
            redis_client=_redis_client,
            persistence_path=persistence_path,
        )
        if conv_config.get("redis_enabled", False) and _redis_client:
            logger.info("Conversation store ready (Redis-backed)")
        else:
            logger.info("Conversation store ready (in-memory)")

        # Cache
        cache_config = _config.get("cache", {})
        _cache = create_cache(
            use_redis=cache_config.get("redis_enabled", False),
            redis_client=_redis_client,
            max_size=cache_config.get("max_size", 1000),
            default_ttl=cache_config.get("default_ttl", 300),
        )
        cache_type = "Redis" if cache_config.get("redis_enabled", False) and _redis_client else "LRU (in-memory)"
        logger.info("Cache ready (type=%s, max_size=%d, ttl=%ds)",
                     cache_type,
                     cache_config.get("max_size", 1000),
                     cache_config.get("default_ttl", 300))

        # n8n webhook client
        _n8n = get_n8n_client(_config)
        if _n8n:
            logger.info("n8n escalation client ready")
        else:
            logger.info("n8n escalation not configured (set n8n.enabled=true in config)")

    # -- Shutdown: close Redis connection ------------------------------
    @app.on_event("shutdown")
    async def shutdown() -> None:
        global _redis_client, _vector_store
        if _redis_client is not None:
            try:
                await _redis_client.aclose()
                logger.info("Redis connection closed")
            except Exception as exc:
                logger.warning("Error closing Redis connection: %s", exc)
            _redis_client = None
        if _vector_store is not None:
            try:
                if hasattr(_vector_store, "disconnect"):
                    await _vector_store.disconnect()
                    logger.info("Vector store disconnected")
            except Exception as exc:
                logger.warning("Error disconnecting vector store: %s", exc)
            _vector_store = None

    # ==================================================================
    # GET /health
    # ==================================================================
    @app.get("/health")
    async def health():
        redis_ok = False
        if _redis_client is not None:
            try:
                await _redis_client.ping()
                redis_ok = True
            except Exception:
                pass

        # Check vector store status
        vector_db_connected = False
        vector_db_type = "none"
        vector_db_point_count = 0
        if _vector_store is not None:
            vector_db_connected = getattr(_vector_store, "is_connected", False)
            vector_db_type = type(_vector_store).__name__
            try:
                vector_db_point_count = await _vector_store.count()
            except Exception:
                pass

        # Guardrails status
        guardrails_status = {
            "pii": _pii_sanitizer.status if _pii_sanitizer else {"enabled": False, "engine": "none"},
            "safety": _safety_checker.status if _safety_checker else {"enabled": False, "engine": "none"},
        }

        # DPO training pipeline status
        try:
            dpo_status = get_dpo_status()
        except Exception:
            dpo_status = {"enabled": False, "error": "unavailable"}

        # RAG evaluation status (read latest result)
        try:
            _rag_eval_history = load_ragas_results(RAGAS_RESULTS_PATH)
            if _rag_eval_history:
                _last_eval = _rag_eval_history[-1]
                _rag_eval_available = True
                _rag_eval_engine = _last_eval.get("engine", "heuristic")
                _rag_eval_num_samples = _last_eval.get("num_samples", 0)
                _rag_eval_faithfulness = _last_eval.get("faithfulness", 0.0)
                _rag_eval_answer_relevancy = _last_eval.get("answer_relevancy", 0.0)
                _rag_eval_context_recall = _last_eval.get("context_recall", 0.0)
                _rag_eval_context_precision = _last_eval.get("context_precision", 0.0)
            else:
                _rag_eval_available = False
                _rag_eval_engine = None
                _rag_eval_num_samples = 0
                _rag_eval_faithfulness = 0.0
                _rag_eval_answer_relevancy = 0.0
                _rag_eval_context_recall = 0.0
                _rag_eval_context_precision = 0.0
        except Exception:
            _rag_eval_available = False
            _rag_eval_engine = None
            _rag_eval_num_samples = 0
            _rag_eval_faithfulness = 0.0
            _rag_eval_answer_relevancy = 0.0
            _rag_eval_context_recall = 0.0
            _rag_eval_context_precision = 0.0

        return {
            "status": "ok",
            "version": "2.0.0",
            "model_loaded": _model_assistant is not None and _model_assistant.model is not None,
            "rag_loaded": _rag_retriever is not None,
            "rag_articles": _rag_retriever.article_count() if _rag_retriever else 0,
            "vector_db": {
                "connected": vector_db_connected,
                "type": vector_db_type,
                "point_count": vector_db_point_count,
            },
            "guardrails": guardrails_status,
            "dpo_training": dpo_status,
            "rag_eval_available": _rag_eval_available,
            "rag_eval_engine": _rag_eval_engine,
            "rag_eval_num_samples": _rag_eval_num_samples,
            "rag_eval_faithfulness": _rag_eval_faithfulness,
            "rag_eval_answer_relevancy": _rag_eval_answer_relevancy,
            "rag_eval_context_recall": _rag_eval_context_recall,
            "rag_eval_context_precision": _rag_eval_context_precision,
            "conversation_sessions": _conversation_store.get_session_count() if _conversation_store else 0,
            "auth_enabled": _config.get("auth", {}).get("enabled", True),
            "rate_limiting_enabled": _config.get("rate_limiting", {}).get("enabled", True),
            "redis_connected": redis_ok,
            "n8n_configured": _n8n is not None,
            "n8n_base_url": _n8n.base_url if _n8n else None,
            "metrics_enabled": _config.get("prometheus", {}).get("enabled", True),
            "metrics_endpoint": "/metrics" if _config.get("prometheus", {}).get("enabled", True) else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # POST /api/classify_ticket
    # ==================================================================
    @app.post("/api/classify_ticket", response_model=ClassifyResponse)
    async def classify_ticket(req: ClassifyRequest, request: Request):
        text = req.ticket_text.strip()

        # ── Guardrails: PII masking + safety check ─────────────────────
        if _pii_sanitizer is not None:
            pii_result = _pii_sanitizer.sanitize(text)
            text = pii_result.sanitized_text
            if pii_result.entities_found:
                logger.info("PII sanitized in classify_ticket: %s",
                            [e["type"] for e in pii_result.entities_found])
        if _safety_checker is not None:
            safety_result = _safety_checker.check_input(text)
            if safety_result.blocked:
                logger.warning("Blocked classification request: %s", safety_result.reason)
                raise HTTPException(status_code=400, detail=f"Request blocked: {safety_result.reason}")
        # ───────────────────────────────────────────────────────────────

        cls_start = time.time()

        # Check cache
        if _cache is not None:
            cache_key = cache_key_classify(text)
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                CACHE_HITS.labels(cache_type="classification").inc()
                return ClassifyResponse(**cached_result, cached=True)
            CACHE_MISSES.labels(cache_type="classification").inc()

        if _model_assistant is not None:
            try:
                raw = _model_assistant.handle_classify_ticket(text)
                category, urgency = _parse_classification(raw)
                CLASSIFICATIONS_TOTAL.labels(category=category, urgency=urgency, method="model").inc()
                CLASSIFICATION_LATENCY.labels(method="model").observe(time.time() - cls_start)
                result = {"category": category, "urgency": urgency, "raw": raw}
                if _cache is not None:
                    _cache.set(cache_key_classify(text), result, CACHE_TTL_CLASSIFY)
                return _finalize_classify(text, category, urgency, raw, result, request)
            except Exception as exc:
                logger.error("Model classification failed: %s", exc)

        # Fallback: simple keyword-based classification
        category, urgency = _keyword_classify(text)
        CLASSIFICATIONS_TOTAL.labels(category=category, urgency=urgency, method="fallback").inc()
        raw = f"Category: {category}\nUrgency: {urgency}"
        result = {"category": category, "urgency": urgency, "raw": raw}
        if _cache is not None:
            _cache.set(cache_key_classify(text), result, CACHE_TTL_CLASSIFY)
        return _finalize_classify(text, category, urgency, raw, result, request)

    # ==================================================================
    # POST /api/generate_response
    # ==================================================================
    @app.post("/api/generate_response", response_model=GenerateResponse)
    async def generate_response(req: GenerateRequest):
        text = req.ticket_text.strip()

        # ── Guardrails: PII masking ───────────────────────────────────
        if _pii_sanitizer is not None:
            pii_result = _pii_sanitizer.sanitize(text)
            text = pii_result.sanitized_text
        # ───────────────────────────────────────────────────────────────

        # Check cache
        if _cache is not None:
            cache_key = cache_key_generate(text, req.category, req.urgency)
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                CACHE_HITS.labels(cache_type="generation").inc()
                return GenerateResponse(**cached_result, cached=True)
            CACHE_MISSES.labels(cache_type="generation").inc()

        if _model_assistant is not None:
            try:
                gen_start = time.time()
                raw = _model_assistant.handle_generate_response(
                    text, category=req.category, urgency=req.urgency
                )
                response = _extract_response(raw)
                GENERATIONS_TOTAL.labels(method="model").inc()
                GENERATION_LATENCY.labels(method="model").observe(time.time() - gen_start)
                result = {"response": response, "raw": raw}
                if _cache is not None:
                    _cache.set(cache_key, result, CACHE_TTL_GENERATE)
                return GenerateResponse(**result)
            except Exception as exc:
                logger.error("Model response generation failed: %s", exc)

        # Fallback: template-based response
        gen_start = time.time()
        response = _template_response(text, req.category, req.urgency)
        GENERATIONS_TOTAL.labels(method="fallback").inc()
        GENERATION_LATENCY.labels(method="fallback").observe(time.time() - gen_start)
        result = {"response": response, "raw": response}
        if _cache is not None:
            _cache.set(cache_key_generate(text, req.category, req.urgency), result, CACHE_TTL_GENERATE)
        return GenerateResponse(**result)

    # ==================================================================
    # POST /api/search_kb   <--- THE RAG ENDPOINT
    # ==================================================================
    @app.post("/api/search_kb", response_model=SearchKBResponse)
    async def search_kb(req: SearchKBRequest):
        if _rag_retriever is None:
            raise HTTPException(status_code=503, detail="Knowledge base not loaded")

        query = req.query

        # ── Guardrails: PII masking + safety check ─────────────────────
        if _pii_sanitizer is not None:
            pii_result = _pii_sanitizer.sanitize(query)
            query = pii_result.sanitized_text
        if _safety_checker is not None:
            safety_result = _safety_checker.check_input(query)
            if safety_result.blocked:
                logger.warning("Blocked search_kb query: %s", safety_result.reason)
                raise HTTPException(status_code=400, detail=f"Query blocked: {safety_result.reason}")
        # ───────────────────────────────────────────────────────────────

        # Check cache
        if _cache is not None:
            cache_key = cache_key_rag(query, req.top_k, req.category_filter)
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                CACHE_HITS.labels(cache_type="rag").inc()
                return SearchKBResponse(**cached_result, cached=True)
            CACHE_MISSES.labels(cache_type="rag").inc()

        rag_start = time.time()
        results = _rag_retriever.search(
            query=query,
            top_k=req.top_k,
            category_filter=req.category_filter,
        )
        RAG_SEARCH_LATENCY.observe(time.time() - rag_start)

        method = "embedding" if _rag_retriever.has_embeddings else "keyword"
        RAG_SEARCHES_TOTAL.labels(method=method).inc()
        RAG_RESULTS_RETURNED.observe(len(results))

        kb_results = [
            SearchKBResult(
                id=r.get("id", ""),
                title=r.get("title", ""),
                category=r.get("category", ""),
                tags=r.get("tags", []),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in results
        ]

        response_data = {
            "query": query,  # Use PII-sanitized query, not original
            "results": [r.model_dump() for r in kb_results],
            "total_found": len(kb_results),
            "method": method,
        }

        # Cache result as serializable dicts
        if _cache is not None:
            _cache.set(cache_key_rag(query, req.top_k, req.category_filter),
                       response_data, CACHE_TTL_RAG)

        # Reconstruct results as Pydantic models for the response
        return SearchKBResponse(
            query=query,  # Use PII-sanitized query, not original
            results=kb_results,
            total_found=len(kb_results),
            method=method,
        )

    # ==================================================================
    # POST /api/resolve_ticket
    # ==================================================================
    @app.post("/api/resolve_ticket", response_model=ResolveResponse)
    async def resolve_ticket(req: ResolveRequest):
        ticket_id = f"TKT-{int(time.time())}"
        summary = (
            f"Ticket classified as '{req.category}' (urgency: {req.urgency}). "
            f"Resolution notes: {req.resolution_notes or 'No notes provided.'}"
        )
        return ResolveResponse(
            ticket_id=ticket_id,
            status="resolved",
            summary=summary,
        )

    # ==================================================================
    # POST /api/feedback_log
    # ==================================================================
    @app.post("/api/feedback_log", response_model=FeedbackResponse)
    async def log_feedback(req: FeedbackRequest):
        fb = req.data
        ticket_id = fb.get("ticket_id", "unknown")
        rating = fb.get("rating", "unknown")

        feedback_dir = os.path.join(_project_root, "data", "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_path = os.path.join(feedback_dir, "feedback_log.jsonl")

        FEEDBACK_TOTAL.labels(rating=rating).inc()

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fb,
        }
        try:
            with open(feedback_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            logger.info("Feedback logged for ticket %s (rating: %s)", ticket_id, rating)
        except OSError as exc:
            logger.error("Failed to write feedback: %s", exc)

        # ── Trigger DPO pipeline (background) ─────────────────────────
        try:
            dpo_config = _config.get("dpo_training", {})
            if dpo_config.get("enabled", True):
                # Schedule the pipeline in background (non-blocking)
                dpo_result = try_schedule_pipeline()
                if dpo_result.get("scheduled"):
                    logger.debug("DPO pipeline scheduled after feedback: %s", dpo_result["method"])
        except Exception as exc:
            logger.warning("Failed to trigger DPO pipeline: %s", exc)
        # ───────────────────────────────────────────────────────────────

        return FeedbackResponse(status="logged", ticket_id=ticket_id, rating=rating)

    # ==================================================================
    # POST /api/conversation/message   <--- NEW: Conversation memory
    # ==================================================================
    @app.post("/api/conversation/message", response_model=ConversationMessageResponse)
    async def conversation_message(req: ConversationMessageRequest, request: Request):
        if _conversation_store is None:
            raise HTTPException(status_code=503, detail="Conversation store not initialised")

        message = req.message

        # ── Guardrails: PII masking + safety check ─────────────────────
        if _pii_sanitizer is not None:
            pii_result = _pii_sanitizer.sanitize(message)
            message = pii_result.sanitized_text
            if pii_result.entities_found:
                logger.info("PII sanitized in conversation: %s",
                            [e["type"] for e in pii_result.entities_found])
        if _safety_checker is not None:
            safety_result = _safety_checker.check_input(message)
            if safety_result.blocked:
                logger.warning("Blocked conversation message: %s", safety_result.reason)
                raise HTTPException(status_code=400, detail=f"Message blocked: {safety_result.reason}")
        # ───────────────────────────────────────────────────────────────

        # Get or create session
        session_id = req.session_id
        if not session_id:
            # Create a new session
            conv = _conversation_store.create_session(
                metadata={"source": "api", "ip": request.client.host if request.client else "unknown"}
            )
            session_id = conv.session_id
        else:
            conv = _conversation_store.get_or_create_session(session_id)

        CONVERSATION_MESSAGES_TOTAL.labels(role="user").inc()

        # Store sanitized user message (PII-removed version in history)
        _conversation_store.add_message(session_id, "user", message, metadata=req.context)

        # Build conversation context for the model
        history = _conversation_store.get_history(session_id, max_messages=10)
        context_str = build_conversation_context(history)

        # Generate response with context
        if _model_assistant is not None:
            try:
                full_prompt = f"{context_str}\nUser: {message}\nAssistant:"
                raw = _model_assistant.handle_generate_response(
                    full_prompt, category="", urgency=""
                )
                reply = _extract_response(raw)
            except Exception as exc:
                logger.error("Conversation generation failed: %s", exc)
                reply = _template_response(message, "general", "low")
        else:
            reply = _template_response(message, "general", "low")

        CONVERSATION_MESSAGES_TOTAL.labels(role="assistant").inc()

        # Store assistant response
        _conversation_store.add_message(session_id, "assistant", reply)

        # Get updated history
        updated_history = _conversation_store.get_history(session_id, max_messages=10)

        return ConversationMessageResponse(
            session_id=session_id,
            reply=reply,
            history=updated_history,
        )

    # ==================================================================
    # GET /api/sessions   <--- NEW: List active sessions
    # ==================================================================
    @app.get("/api/sessions")
    async def list_sessions(limit: int = 100, offset: int = 0):
        if _conversation_store is None:
            raise HTTPException(status_code=503, detail="Conversation store not initialised")
        sessions = _conversation_store.list_sessions(limit=limit, offset=offset)
        ACTIVE_SESSIONS.set(_conversation_store.get_session_count())
        return {
            "sessions": sessions,
            "total": _conversation_store.get_session_count(),
        }

    # ==================================================================
    # DELETE /api/sessions/{session_id}   <--- NEW: Clear a session
    # ==================================================================
    @app.delete("/api/sessions/{session_id}")
    async def clear_session(session_id: str):
        if _conversation_store is None:
            raise HTTPException(status_code=503, detail="Conversation store not initialised")
        deleted = _conversation_store.clear_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {"status": "deleted", "session_id": session_id}

    # ==================================================================
    # GET /api/conversation/{session_id}   <--- NEW: Get conversation history
    # ==================================================================
    @app.get("/api/conversation/{session_id}", response_model=ConversationHistoryResponse)
    async def get_conversation_history(session_id: str, max_messages: int = 50):
        if _conversation_store is None:
            raise HTTPException(status_code=503, detail="Conversation store not initialised")
        conv = _conversation_store.get_session(session_id)
        if conv is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        history = _conversation_store.get_history(session_id, max_messages=max_messages)
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=history,
            message_count=len(history),
        )

    # ==================================================================
    # GET /api/rate_limit/status   <--- NEW: Get rate limit info
    # ==================================================================
    @app.get("/api/rate_limit/status")
    async def rate_limit_status(request: Request):
        # Use the shared rate store (same instance the middleware uses)
        from src.mcp_server.rate_limiter import DEFAULT_LIMITS

        store = _rate_store
        if store is None:
            return {"tier": "anonymous", "limit": "disabled", "remaining": "unlimited", "reset_at": 0, "window_seconds": 0}

        tier: str = getattr(request.state, "api_key_tier", "anonymous")
        max_requests, window_seconds = DEFAULT_LIMITS.get(tier, DEFAULT_LIMITS["anonymous"])

        client_key = f"key:{getattr(request.state, 'api_key_name', 'unknown')}"
        if client_key == "key:unknown":
            forwarded = request.headers.get("X-Forwarded-For", "")
            client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
            client_key = f"ip:{client_ip}"

        allowed, remaining, reset_at = await store.check(client_key, max_requests, window_seconds)

        return {
            "tier": tier,
            "limit": max_requests,
            "remaining": remaining,
            "reset_at": int(reset_at),
            "window_seconds": window_seconds,
        }

    # ==================================================================
    # GET /api/cache/stats   <--- NEW: Cache statistics (admin)
    # ==================================================================
    @app.get("/api/cache/stats")
    async def cache_stats():
        if _cache is None:
            raise HTTPException(status_code=503, detail="Cache not initialised")
        stats = _cache.stats()
        return {
            "cache_type": "Redis" if hasattr(_cache, "_redis") else "LRU (In-Memory)",
            "stats": stats,
        }

    # ==================================================================
    # POST /api/admin/keys   <--- NEW: Generate API key (admin)
    # ==================================================================
    @app.post("/api/admin/keys", response_model=GenerateKeyResponse)
    async def generate_api_key(
        request: Request,
        name: str = "auto-generated",
        tier: str = TIER_STANDARD,
        admin_key: Dict[str, str] = Depends(require_admin_key),
    ):
        if tier not in (TIER_ADMIN, TIER_STANDARD):
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Use '{TIER_ADMIN}' or '{TIER_STANDARD}'.")
        key = get_key_store().__class__.generate_key()
        return GenerateKeyResponse(api_key=key, name=name, tier=tier)

    # ==================================================================
    # POST /api/escalate   <--- NEW: Auto-escalate a critical ticket
    # ==================================================================
    @app.post("/api/escalate", response_model=EscalateResponse)
    async def escalate_ticket(req: EscalateRequest):
        """Manually escalate a ticket via n8n webhook.

        Use this endpoint to trigger an escalation alert for critical
        or high-urgency tickets. The payload is sent to the configured
        n8n webhook, which can route to Slack, email, PagerDuty, etc.

        Works even when n8n is not configured (returns n8n_configured=false).
        """
        text = req.ticket_text.strip()

        # ── Guardrails: PII masking on escalation text ─────────────────
        if _pii_sanitizer is not None:
            pii_result = _pii_sanitizer.sanitize(text)
            text = pii_result.sanitized_text
        # ───────────────────────────────────────────────────────────────

        ticket_id = req.ticket_id or f"TKT-{int(time.time())}"

        if _n8n is None:
            logger.warning(
                "Escalation requested but n8n not configured. "
                "Set n8n.enabled=true in config and point to your n8n instance."
            )
            return EscalateResponse(
                status="n8n_not_configured",
                ticket_id=ticket_id,
                escalation={
                    "sent": False,
                    "reason": "n8n client not configured — set n8n.enabled=true in configs/mcp_server.yaml",
                },
                n8n_configured=False,
            )

        esc_start = time.time()
        result = await _n8n.escalate_ticket(
            ticket_text=text,
            category=req.category or "unclassified",
            urgency=req.urgency or "critical",
            classification_raw=req.classification_raw,
            ticket_id=ticket_id,
            metadata=req.metadata or {},
        )
        ESCALATION_LATENCY.observe(time.time() - esc_start)

        status = "escalated" if result.get("success") else "escalation_failed"
        ESCALATION_EVENTS_TOTAL.labels(urgency=req.urgency or "critical", status=status).inc()
        return EscalateResponse(
            status=status,
            ticket_id=ticket_id,
            escalation=result,
            n8n_configured=True,
        )

    # ==================================================================
    # GET /api/escalate/status   <--- NEW: n8n escalation health
    # ==================================================================
    @app.get("/api/escalate/status")
    async def escalate_status():
        """Check if n8n escalation is configured and reachable."""
        if _n8n is None:
            return {
                "configured": False,
                "message": "n8n not configured. Set n8n.enabled=true in configs/mcp_server.yaml",
                "base_url": None,
            }
        return {
            "configured": True,
            "base_url": _n8n.base_url,
            "webhook_path": _n8n.webhook_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # GET /api/dpo/status   <--- NEW: DPO training pipeline status
    # ==================================================================
    @app.get("/api/dpo/status")
    async def dpo_pipeline_status():
        """Get the current status of the DPO training pipeline.

        Returns information about feedback processed, preference pairs
        generated, and whether a training run is due.
        """
        try:
            status = get_dpo_status()
            return status
        except Exception as exc:
            return {
                "enabled": False,
                "error": str(exc),
                "message": "DPO pipeline status unavailable",
            }

    # ==================================================================
    # GET /api/eval/ragas   <--- NEW: RAG evaluation results
    # ==================================================================
    @app.get("/api/eval/ragas")
    async def ragas_evaluation_results():
        """Get the latest RAG evaluation results.

        Returns the most recent evaluation run with faithfulness,
        answer relevancy, context recall, and context precision scores.
        Also shows deltas from the previous run if available.
        """
        try:
            history = load_ragas_results(RAGAS_RESULTS_PATH)
            if not history:
                return {
                    "available": False,
                    "message": "No evaluation runs yet. Run: python src/eval/ragas_eval.py",
                }

            comparison = compare_ragas_results(history)
            current = history[-1]

            return {
                "available": True,
                "engine": current.get("engine", "heuristic"),
                "num_samples": current.get("num_samples", 0),
                "timestamp": current.get("timestamp", ""),
                "faithfulness": current.get("faithfulness"),
                "answer_relevancy": current.get("answer_relevancy"),
                "context_recall": current.get("context_recall"),
                "context_precision": current.get("context_precision"),
                "deltas": comparison.get("deltas"),
                "total_runs": len(history),
                "samples": current.get("samples", []),
            }
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
                "message": "Failed to load evaluation results",
            }

    # ==================================================================
    # Mount Admin Dashboard  <--- NEW: Web-based admin dashboard
    # ==================================================================
    mount_dashboard(app)

    return app


# ===================================================================
# Config loading
# ===================================================================

def _load_server_config() -> Dict[str, Any]:
    """Load server configuration from mcp_server.yaml."""
    config_path = os.path.join(_project_root, "configs", "mcp_server.yaml")
    default_config: Dict[str, Any] = {
        "auth": {
            "enabled": True,
            "public_paths": ["/health", "/docs", "/openapi.json", "/redoc", "/admin/dashboard", "/metrics"],
        },
        "rate_limiting": {
            "enabled": True,
            "redis_enabled": False,
        },
        "conversation": {
            "redis_enabled": False,
            "persistence_path": "data/conversations/conversations.json",
        },
        "cache": {
            "redis_enabled": False,
            "max_size": 1000,
            "default_ttl": 300,
        },
        "rag": {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 5,
        },
    }

    if not os.path.exists(config_path):
        logger.info("Config not found at %s — using defaults", config_path)
        return default_config

    try:
        import yaml
        with open(config_path, "r") as fh:
            loaded = yaml.safe_load(fh) or {}
        # Merge loaded config into defaults (deep merge for nested dicts)
        for section, values in loaded.items():
            if section in default_config and isinstance(values, dict):
                default_config[section].update(values)
            else:
                default_config[section] = values
        logger.info("Loaded server config from %s", config_path)
    except Exception as exc:
        logger.warning("Failed to load config (%s) — using defaults", exc)

    return default_config


# ===================================================================
# Fallback helpers (used when no model is loaded)
# ===================================================================

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "billing": ["payment", "invoice", "refund", "charge", "subscription", "price", "plan", "cancel", "bill"],
    "account": ["password", "login", "sign in", "account", "profile", "email", "verify", "register"],
    "technical": ["error", "bug", "crash", "not working", "failed", "broken", "slow", "timeout"],
    "security": ["hack", "breach", "unauthorized", "2fa", "authentication", "phishing", "suspicious"],
    "feature_request": ["feature", "suggestion", "request", "would like", "add", "improve"],
    "general": ["help", "question", "how to", "what is", "info", "guide"],
}

URGENCY_KEYWORDS: Dict[str, List[str]] = {
    "critical": ["down", "outage", "emergency", "urgent", "asap", "blocked", "cannot work", "immediately"],
    "high": ["frustrated", "angry", "serious", "important", "deadline", "urgent", "losing money"],
    "medium": ["annoying", "inconvenient", "please help", "issue", "problem"],
    "low": ["question", "how to", "curious", "wondering", "feature request", "suggestion"],
}


def _parse_classification(raw: str) -> Tuple[str, str]:
    """Extract ``category`` and ``urgency`` from a model-generated string."""
    cat_match = re.search(r"(?i)(?:category|Category)[:\s]+(\w+)", raw)
    urg_match = re.search(r"(?i)(?:urgency|Urgency)[:\s]+(\w+)", raw)
    category = cat_match.group(1).lower() if cat_match else "general"
    urgency = urg_match.group(1).lower() if urg_match else "low"
    return category, urgency


def _keyword_classify(text: str) -> Tuple[str, str]:
    """Simple keyword-based classification (fallback when no model loaded)."""
    text_lower = text.lower()

    best_cat = "general"
    best_cat_score = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_cat_score:
            best_cat_score = score
            best_cat = cat

    best_urg = "low"
    best_urg_score = 0
    for urg, keywords in URGENCY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_urg_score:
            best_urg_score = score
            best_urg = urg

    return best_cat, best_urg


def _extract_response(raw: str) -> str:
    """Pull the main response from a model output, stripping labels."""
    resp_match = re.search(r"(?i)(?:Response:|Answer:)\s*(.+)", raw, re.DOTALL)
    if resp_match:
        return resp_match.group(1).strip()
    return raw.strip()


def _template_response(text: str, category: str, urgency: str) -> str:
    """Generate a simple template-based response when no model is loaded."""
    templates = {
        "billing": "Thank you for reaching out regarding billing. We'll review your account and get back to you within 4 hours with a resolution.",
        "account": "Thanks for contacting us about your account. We'll look into this and respond shortly.",
        "technical": "We're sorry you're experiencing a technical issue. Our team will investigate and follow up with troubleshooting steps.",
        "security": "Security is our top priority. We've flagged this for immediate review by our security team.",
        "feature_request": "Thanks for your suggestion! We've logged this feature request for our product team to evaluate.",
        "general": "Thank you for your inquiry. We'll get back to you with more information shortly.",
    }
    return templates.get(category.lower(), templates["general"])


def _finalize_classify(
    text: str,
    category: str,
    urgency: str,
    raw: str,
    result: dict,
    request: Request,
) -> ClassifyResponse:
    """Shared logic after classification: cache + auto-escalate.

    Deduplicates the escalation trigger code that was previously
    duplicated between the model path and fallback path.
    """
    # 🔔 Auto-escalate if critical urgency and n8n is configured
    if urgency == "critical" and _n8n is not None:
        ticket_id = f"TKT-{int(time.time())}"
        client_ip = request.client.host if request.client else "unknown"
        _bg_escalate(text, category, urgency, raw, ticket_id, {
            "source": "classify_endpoint",
            "ip": client_ip,
        })

    return ClassifyResponse(**result)


def _bg_escalate(
    ticket_text: str,
    category: str,
    urgency: str,
    classification_raw: str,
    ticket_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire-and-forget escalation via n8n in a background task.

    Uses ``asyncio.create_task`` to schedule the webhook call without
    blocking the response. The caller doesn't wait for the result.
    """
    import asyncio

    if _n8n is None:
        return

    async def _do_escalate() -> None:
        try:
            result = await _n8n.escalate_ticket(
                ticket_text=ticket_text,
                category=category,
                urgency=urgency,
                classification_raw=classification_raw,
                ticket_id=ticket_id,
                metadata=metadata or {},
            )
            if result.get("success"):
                logger.info("Auto-escalated critical ticket %s to n8n", ticket_id)
            else:
                logger.warning("Auto-escalation failed for %s: %s", ticket_id, result.get("error"))
        except Exception as exc:
            logger.error("Auto-escalation error for %s: %s", ticket_id, exc)

    # Schedule the coroutine in the background; don't await it
    asyncio.create_task(_do_escalate())


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    """Run the REST API server via uvicorn."""
    parser = argparse.ArgumentParser(description="MCP AI Assistant — REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--log-level", default="info", help="Logging level")
    parser.add_argument("--disable-auth", action="store_true", help="Disable API key authentication")
    parser.add_argument("--disable-rate-limit", action="store_true", help="Disable rate limiting")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not _FASTAPI_AVAILABLE:
        print(
            "ERROR: Missing dependencies. Install with:\n"
            "  pip install fastapi uvicorn pydantic"
        )
        sys.exit(1)

    import uvicorn

    # Apply CLI overrides to config
    global _config
    _config = _load_server_config()
    if args.disable_auth:
        _config["auth"] = {"enabled": False}
        logger.info("API authentication disabled via CLI flag")
    if args.disable_rate_limit:
        _config["rate_limiting"] = {"enabled": False}
        logger.info("Rate limiting disabled via CLI flag")

    app = create_app(_config)
    print(f"\n{'='*60}")
    print(f"  🚀  MCP AI Assistant REST API v2.0.0")
    print(f"  {'='*60}")
    print(f"  Starting on  http://{args.host}:{args.port}")
    print(f"  {'─'*60}")
    print(f"  Endpoints:")
    print(f"    GET  /health                     Health check")
    print(f"    POST /api/classify_ticket        Classify ticket")
    print(f"    POST /api/generate_response      Generate response")
    print(f"    POST /api/search_kb              RAG knowledge base search")
    print(f"    POST /api/resolve_ticket         Resolve ticket")
    print(f"    POST /api/feedback_log           Log feedback")
    print(f"    POST /api/conversation/message   Conversational AI ✨")
    print(f"    GET  /api/sessions               List sessions ✨")
    print(f"    DELETE /api/sessions/{id}        Delete session ✨")
    print(f"    GET  /api/conversation/{id}      Get history ✨")
    print(f"    GET  /api/rate_limit/status      Rate limit info ✨")
    print(f"    GET  /api/cache/stats            Cache stats ✨")
    print(f"    POST /api/admin/keys             Generate API key ✨")
    print(f"    POST /api/escalate              Escalate critical ticket 🚨")
    print(f"    GET  /api/escalate/status        Escalation health 🚨")
    print(f"    GET  /api/eval/ragas             RAG evaluation results 📊")
    print(f"  {'─'*60}")
    n8n_status = "🚨 Active" if _n8n is not None else "🚨 Not configured"
    print(f"  Auth:       {'🔒 Enabled' if _config.get('auth', {}).get('enabled', True) else '🔓 Disabled'}")
    print(f"  Rate Limit: {'⏱️  Enabled' if _config.get('rate_limiting', {}).get('enabled', True) else '⏱️  Disabled'}")
    print(f"  Cache:      {'⚡ Active' if _config.get('cache', {}).get('redis_enabled', False) else '⚡ LRU (in-memory)'}")
    print(f"  Sessions:   {'💬 Active' if _config.get('conversation', {}).get('redis_enabled', False) else '💬 In-memory'}")
    print(f"  Escalation: {n8n_status}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
