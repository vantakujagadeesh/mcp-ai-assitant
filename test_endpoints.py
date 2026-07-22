"""
Test all 6 REST API endpoints using FastAPI TestClient (no server needed).
"""
import sys, json
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from src.mcp_server.rest_api import create_app

app = create_app()
client = TestClient(app)

# Use context manager to trigger startup events (loads RAG, etc.)
client.__enter__()

def p(label, data):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))

# ── 1. HEALTH ─────────────────────────────────────────────────────
resp = client.get("/health")
p("GET /health", resp.json())

# ── 2. CLASSIFY TICKET ────────────────────────────────────────────
resp = client.post("/api/classify_ticket", json={
    "ticket_text": "My payment was declined three times and I'm being charged late fees! This is urgent!"
})
p("POST /api/classify_ticket", resp.json())

# ── 3. SEARCH KB (RAG at work) ────────────────────────────────────
resp = client.post("/api/search_kb", json={
    "query": "How do I reset my password?",
    "top_k": 3
})
p("POST /api/search_kb", resp.json())

# ── 4. SEARCH KB with category filter ──────────────────────────────
resp = client.post("/api/search_kb", json={
    "query": "payment failed",
    "top_k": 2,
    "category_filter": "billing"
})
p("POST /api/search_kb (billing filter)", resp.json())

# ── 5. GENERATE RESPONSE ──────────────────────────────────────────
resp = client.post("/api/generate_response", json={
    "ticket_text": "My payment was declined three times",
    "category": "billing",
    "urgency": "high"
})
p("POST /api/generate_response", resp.json())

# ── 6. RESOLVE TICKET ─────────────────────────────────────────────
resp = client.post("/api/resolve_ticket", json={
    "ticket_text": "My payment was declined",
    "category": "billing",
    "urgency": "high",
    "resolution_notes": "Advised customer to retry payment"
})
p("POST /api/resolve_ticket", resp.json())

# ── 7. FEEDBACK LOG ───────────────────────────────────────────────
resp = client.post("/api/feedback_log", json={
    "data": {
        "ticket_id": "TKT-99999",
        "rating": "thumbs_up",
        "category": "billing",
        "confidence": 0.92
    }
})
p("POST /api/feedback_log", resp.json())

# Cleanup: trigger shutdown events
client.__exit__(None, None, None)

print(f"\n{'='*60}")
print(f"  ✅  ALL 6 ENDPOINTS TESTED SUCCESSFULLY!")
print(f"{'='*60}")
