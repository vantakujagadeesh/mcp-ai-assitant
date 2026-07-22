"""
FastAPI REST wrapper for the MCP-powered AI assistant.

Provides plain HTTP endpoints (no SSE, no JSON-RPC) so n8n workflows
can call the assistant with standard POST requests.

Endpoints
---------
- GET  /health                     → health check
- POST /api/classify_ticket        → classify ticket (category + urgency)
- POST /api/generate_response      → generate support response
- POST /api/search_kb              → RAG search over knowledge base
- POST /api/resolve_ticket         → resolve / summarise a ticket
- POST /api/feedback_log           → log feedback for future re-tuning
"""

import argparse
import json
import logging
import os
import sys
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# FastAPI / uvicorn / pydantic
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------
# Adjust sys.path so we can import from sibling modules when run directly.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.rag.retriever import KnowledgeBaseRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model assistant  (set during startup)
# ---------------------------------------------------------------------------
_model_assistant: Optional[Any] = None
_rag_retriever: Optional[KnowledgeBaseRetriever] = None

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

# -- Generate response -----------------------------------------------
class GenerateRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1)
    category: str = ""
    urgency: str = ""

class GenerateResponse(BaseModel):
    response: str
    raw: str

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


# ===================================================================
# FastAPI application
# ===================================================================

def create_app() -> "FastAPI":
    """Build and return a fully configured FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi and uvicorn are required.  Install them with:\n"
            "  pip install fastapi uvicorn pydantic"
        )

    app = FastAPI(
        title="MCP AI Assistant — REST API",
        version="1.0.0",
        description="HTTP endpoints for ticket classification, response generation, "
        "and RAG-powered knowledge base search.",
    )

    # -- CORS (allow n8n from any origin) --------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Startup: load RAG and model ------------------------------------
    @app.on_event("startup")
    async def startup() -> None:
        global _rag_retriever, _model_assistant

        logger.info("Initialising knowledge base retriever …")
        try:
            _rag_retriever = KnowledgeBaseRetriever()
            logger.info("RAG retriever ready (%d articles)", _rag_retriever.article_count())
        except Exception as exc:
            logger.warning("Could not initialise RAG retriever: %s", exc)
            _rag_retriever = None

        logger.info("Loading classification model …")
        try:
            from src.mcp_server.server import SupportAssistant, load_config

            cfg = load_config()
            _model_assistant = SupportAssistant(cfg)
            logger.info("Classification model loaded")
        except Exception as exc:
            logger.warning("Could not load model (will use regex fallback): %s", exc)
            _model_assistant = None

    # ==================================================================
    # GET /health
    # ==================================================================
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": _model_assistant is not None,
            "rag_loaded": _rag_retriever is not None,
            "rag_articles": _rag_retriever.article_count() if _rag_retriever else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # POST /api/classify_ticket
    # ==================================================================
    @app.post("/api/classify_ticket", response_model=ClassifyResponse)
    async def classify_ticket(req: ClassifyRequest):
        text = req.ticket_text.strip()

        if _model_assistant is not None:
            try:
                raw = _model_assistant.handle_classify_ticket(text)
                category, urgency = _parse_classification(raw)
                return ClassifyResponse(category=category, urgency=urgency, raw=raw)
            except Exception as exc:
                logger.error("Model classification failed: %s", exc)
                # fall through to regex fallback

        # Fallback: simple keyword-based classification
        category, urgency = _keyword_classify(text)
        raw = f"Category: {category}\nUrgency: {urgency}"
        return ClassifyResponse(category=category, urgency=urgency, raw=raw)

    # ==================================================================
    # POST /api/generate_response
    # ==================================================================
    @app.post("/api/generate_response", response_model=GenerateResponse)
    async def generate_response(req: GenerateRequest):
        text = req.ticket_text.strip()

        if _model_assistant is not None:
            try:
                raw = _model_assistant.handle_generate_response(
                    text, category=req.category, urgency=req.urgency
                )
                response = _extract_response(raw)
                return GenerateResponse(response=response, raw=raw)
            except Exception as exc:
                logger.error("Model response generation failed: %s", exc)

        # Fallback: template-based response
        response = _template_response(text, req.category, req.urgency)
        return GenerateResponse(response=response, raw=response)

    # ==================================================================
    # POST /api/search_kb   <--- THE RAG ENDPOINT
    # ==================================================================
    @app.post("/api/search_kb", response_model=SearchKBResponse)
    async def search_kb(req: SearchKBRequest):
        if _rag_retriever is None:
            raise HTTPException(status_code=503, detail="Knowledge base not loaded")

        results = _rag_retriever.search(
            query=req.query,
            top_k=req.top_k,
            category_filter=req.category_filter,
        )

        method = "embedding" if _rag_retriever.has_embeddings else "keyword"

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

        return SearchKBResponse(
            query=req.query,
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

        # Persist feedback to a JSONL file for future re-tuning
        feedback_dir = os.path.join(_project_root, "data", "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_path = os.path.join(feedback_dir, "feedback_log.jsonl")

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

        return FeedbackResponse(
            status="logged",
            ticket_id=ticket_id,
            rating=rating,
        )

    return app


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

    # Category
    best_cat = "general"
    best_cat_score = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_cat_score:
            best_cat_score = score
            best_cat = cat

    # Urgency
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


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    """Run the REST API server via uvicorn."""
    parser = argparse.ArgumentParser(description="MCP AI Assistant — REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--log-level", default="info", help="Logging level")
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

    app = create_app()
    print(f"\n🚀 MCP AI Assistant REST API starting on http://{args.host}:{args.port}")
    print(f"   Health:    http://{args.host}:{args.port}/health")
    print(f"   Classify:  POST http://{args.host}:{args.port}/api/classify_ticket")
    print(f"   Generate:  POST http://{args.host}:{args.port}/api/generate_response")
    print(f"   Search KB: POST http://{args.host}:{args.port}/api/search_kb")
    print(f"   Resolve:   POST http://{args.host}:{args.port}/api/resolve_ticket")
    print(f"   Feedback:  POST http://{args.host}:{args.port}/api/feedback_log\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
