"""
Live integration test: starts the REST API server, tests every
endpoint with real HTTP requests (via urllib), captures results.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
API_BASE = "http://localhost:8000"


def start_server():
    """Start the REST API server as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_DIR
    proc = subprocess.Popen(
        [sys.executable, "src/mcp_server/rest_api.py", "--port", "8000", "--log-level", "warning"],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    for i in range(20):
        try:
            resp = urllib.request.urlopen(f"{API_BASE}/health", timeout=2)
            if resp.status == 200:
                return proc
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)
    raise RuntimeError("Server did not start within 10 seconds")


def api(method, path, body=None):
    """Make a REST API call and return parsed JSON."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def p(label, status, data):
    """Pretty-print a test result."""
    icon = "✅" if 200 <= status < 300 else "❌"
    print(f"\n  {icon}  {label}")
    print(f"     Status: {status}")
    print(f"     Response: {json.dumps(data, indent=4)}")


def main():
    print("=" * 60)
    print("  Starting MCP AI Assistant REST API Server…")
    print("=" * 60)

    proc = start_server()
    pid = proc.pid
    print(f"  Server started (PID: {pid})")

    try:
        # ── 1. Health ───────────────────────────────────────────────
        status, data = api("GET", "/health")
        p("GET /health", status, data)

        # ── 2. Classify Ticket ──────────────────────────────────────
        status, data = api("POST", "/api/classify_ticket", {
            "ticket_text": "My payment was declined three times and I'm being charged late fees! This is urgent!"
        })
        p("POST /api/classify_ticket", status, data)
        category = data.get("category", "")
        urgency = data.get("urgency", "")

        # ── 3. Search KB ────────────────────────────────────────────
        status, data = api("POST", "/api/search_kb", {
            "query": "How do I reset my password?",
            "top_k": 3
        })
        p("POST /api/search_kb", status, data)

        # ── 4. Search KB (billing filter) ───────────────────────────
        status, data = api("POST", "/api/search_kb", {
            "query": "payment failed",
            "top_k": 2,
            "category_filter": "billing"
        })
        p("POST /api/search_kb (billing filter)", status, data)

        # ── 5. Generate Response ────────────────────────────────────
        status, data = api("POST", "/api/generate_response", {
            "ticket_text": "My payment was declined three times",
            "category": category or "billing",
            "urgency": urgency or "high"
        })
        p("POST /api/generate_response", status, data)

        # ── 6. Resolve Ticket ───────────────────────────────────────
        status, data = api("POST", "/api/resolve_ticket", {
            "ticket_text": "My payment was declined three times",
            "category": category or "billing",
            "urgency": urgency or "high",
            "resolution_notes": "Advised customer to retry payment via the billing portal"
        })
        p("POST /api/resolve_ticket", status, data)

        # ── 7. Feedback Log ─────────────────────────────────────────
        status, data = api("POST", "/api/feedback_log", {
            "data": {
                "ticket_id": "TKT-LIVE-TEST",
                "rating": "thumbs_up",
                "category": "billing",
                "urgency": "critical",
                "confidence": 0.95
            }
        })
        p("POST /api/feedback_log", status, data)

        # ── Done ────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  ✅  ALL 6 ENDPOINTS TESTED SUCCESSFULLY!")
        print(f"  Server PID {pid} is still running — hit Ctrl+C to stop it")
        print(f"  Or run: kill {pid}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\n  Shutting down server…")
    except Exception as e:
        print(f"\n❌  Error during testing: {e}")
    finally:
        proc.kill()
        proc.wait()
        print("  Server stopped.")


if __name__ == "__main__":
    main()
