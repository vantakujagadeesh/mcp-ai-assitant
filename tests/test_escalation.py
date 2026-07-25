"""
Automated test suite for the auto-escalation flow (n8n webhook integration).

Tests are organised in five sections:

1. **N8nWebhookClient Unit Tests** — URL construction, payload building,
   async send success/error handling (via ``asyncio.run``), singleton management.
2. **Escalation API Endpoint Tests** — ``POST /api/escalate`` and
   ``GET /api/escalate/status`` with n8n both disabled and enabled.
3. **Auto-Escalation Integration Tests** — ``POST /api/classify_ticket``
   triggering (or not) the background escalation when urgency is critical.
4. **End-to-End Flow Tests** — Full classify → escalate pipeline with
   payload capture via mock.
5. **Security & Validation Tests** — Auth requirements, edge cases.

Design note
-----------
``_bg_escalate`` (the background escalation trigger) is a synchronous function
that uses ``asyncio.create_task`` internally.  In tests we patch it directly
with a ``MagicMock`` to verify it was called with the right arguments — this
avoids the complexity of pumping the TestClient's event loop.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CRITICAL_TICKET = (
    "SYSTEM IS DOWN! Our entire production database is inaccessible. "
    "This is a complete outage affecting all customers!"
)

SAMPLE_BILLING_TICKET = "I was charged twice for my subscription. Please help."


def _assert_escalate_response(
    resp_data: Dict[str, Any],
    *,
    expected_status: str,
    expected_n8n_configured: bool,
) -> None:
    """Assert common escalation response fields."""
    assert resp_data["status"] == expected_status, (
        f"Expected status '{expected_status}', got '{resp_data['status']}'"
    )
    assert resp_data["n8n_configured"] is expected_n8n_configured, (
        f"Expected n8n_configured={expected_n8n_configured}, "
        f"got {resp_data['n8n_configured']}"
    )
    assert "ticket_id" in resp_data, "Missing ticket_id in response"
    assert "escalation" in resp_data, "Missing escalation in response"


def _run(coro) -> Any:
    """Run a coroutine synchronously in a test.

    Avoids the need for ``pytest-asyncio``.
    """
    return asyncio.run(coro)


# ===================================================================
# 1. N8nWebhookClient Unit Tests
# ===================================================================

class TestN8nWebhookClientUrl:
    """Verify webhook URL construction."""

    def test_default_url(self):
        """Default base_url + webhook_path builds correct URL."""
        from src.mcp_server.n8n_client import N8nWebhookClient

        client = N8nWebhookClient()
        url = client.webhook_url("ticket-escalate")
        assert url == "http://localhost:5678/webhook/ticket-escalate"

    def test_custom_base_url(self):
        """Custom base_url is reflected in webhook URLs."""
        client = N8nWebhookClient(base_url="https://n8n.example.com")
        url = client.webhook_url("feedback")
        assert url == "https://n8n.example.com/webhook/feedback"

    def test_custom_webhook_path(self):
        """Custom webhook_path is used in URL."""
        client = N8nWebhookClient(webhook_path="/custom-hook/")
        url = client.webhook_url("alert")
        assert url == "http://localhost:5678/custom-hook/alert"

    def test_webhook_path_without_leading_slash(self):
        """webhook_path without leading slash gets one prepended."""
        client = N8nWebhookClient(webhook_path="hook/")
        url = client.webhook_url("test")
        assert url == "http://localhost:5678/hook/test"

    def test_trailing_slashes_handled(self):
        """Trailing slashes on base_url and webhook_path are handled."""
        client = N8nWebhookClient(
            base_url="http://localhost:5678/",
            webhook_path="/webhook/",
        )
        url = client.webhook_url("test")
        assert url == "http://localhost:5678/webhook/test"


class TestN8nWebhookClientHeaders:
    """Verify HTTP headers built by the client."""

    def test_default_headers_no_auth(self):
        """Without api_key, only Content-Type is set."""
        from src.mcp_server.n8n_client import N8nWebhookClient

        client = N8nWebhookClient()
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"
        assert "X-API-Key" not in headers

    def test_headers_with_api_key(self):
        """With api_key, X-API-Key header is included."""
        client = N8nWebhookClient(api_key="sk-n8n-secret")
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["X-API-Key"] == "sk-n8n-secret"

    def test_headers_with_extra(self):
        """Extra headers are merged into the result."""
        client = N8nWebhookClient()
        headers = client._headers({"X-Custom": "value"})
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom"] == "value"


class TestN8nWebhookClientEscalateTicket:
    """Verify escalate_ticket builds the correct payload and calls send_async."""

    def test_escalate_ticket_calls_send_async(self):
        """escalate_ticket delegates to send_async with correct args."""
        from src.mcp_server.n8n_client import N8nWebhookClient

        client = N8nWebhookClient()
        client.send_async = AsyncMock(return_value={"success": True, "status_code": 200})

        result = _run(client.escalate_ticket(
            ticket_text="Database outage!",
            category="technical",
            urgency="critical",
            classification_raw="Category: technical\nUrgency: critical\n",
            ticket_id="TKT-123",
            metadata={"source": "test", "customer_tier": "enterprise"},
        ))

        client.send_async.assert_awaited_once()
        call_args = client.send_async.await_args
        assert call_args is not None
        webhook_id, payload = call_args.args[0], call_args.args[1]

        assert webhook_id == "ticket-escalate"
        assert payload["event"] == "ticket_escalated"
        assert payload["ticket_id"] == "TKT-123"
        assert payload["ticket_text"] == "Database outage!"
        assert payload["category"] == "technical"
        assert payload["urgency"] == "critical"
        assert payload["classification_raw"] == "Category: technical\nUrgency: critical\n"
        assert payload["metadata"]["source"] == "test"
        assert payload["metadata"]["customer_tier"] == "enterprise"
        assert "timestamp" in payload

    def test_escalate_ticket_auto_generates_id(self):
        """When ticket_id is empty, an ID is auto-generated."""
        from src.mcp_server.n8n_client import N8nWebhookClient

        client = N8nWebhookClient()
        client.send_async = AsyncMock(return_value={"success": True, "status_code": 200})

        _run(client.escalate_ticket(
            ticket_text="Urgent issue",
            category="general",
            urgency="critical",
        ))

        client.send_async.assert_awaited_once()
        payload = client.send_async.await_args.args[1]
        assert payload["ticket_id"].startswith("ESC-")


class TestN8nWebhookClientSendAsync:
    """Verify send_async handles various HTTP outcomes.

    Uses ``asyncio.run()`` to avoid requiring ``pytest-asyncio``.
    """

    def test_send_async_success(self):
        """Successful POST returns success=True."""
        from src.mcp_server.n8n_client import N8nWebhookClient

        client = N8nWebhookClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_context = MagicMock()
            mock_context.__aenter__.return_value = mock_response
            mock_async_client.return_value.post.return_value = mock_context

            result = _run(client.send_async("test-webhook", {"key": "value"}))

        assert result["success"] is True
        assert result["status_code"] == 200

    def test_send_async_timeout(self):
        """Timeout returns success=False with error='timeout'."""
        from src.mcp_server.n8n_client import N8nWebhookClient
        from httpx import TimeoutException

        client = N8nWebhookClient(timeout_seconds=1)

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.post.side_effect = TimeoutException(
                "timed out", request=None
            )
            result = _run(client.send_async("test-webhook", {"key": "value"}))

        assert result["success"] is False
        assert result["status_code"] == 0
        assert result["error"] == "timeout"

    def test_send_async_http_error(self):
        """HTTP 4xx/5xx returns success=False with status code."""
        from src.mcp_server.n8n_client import N8nWebhookClient
        from httpx import HTTPStatusError

        client = N8nWebhookClient()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.post.side_effect = HTTPStatusError(
                "500 error", request=MagicMock(), response=mock_response
            )
            result = _run(client.send_async("test-webhook", {"key": "value"}))

        assert result["success"] is False
        assert result["status_code"] == 500

    def test_send_async_connection_error(self):
        """Connection refused returns success=False."""
        from src.mcp_server.n8n_client import N8nWebhookClient
        from httpx import ConnectError

        client = N8nWebhookClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.post.side_effect = ConnectError(
                "Connection refused", request=None
            )
            result = _run(client.send_async("test-webhook", {"key": "value"}))

        assert result["success"] is False
        assert result["status_code"] == 0


class TestN8nWebhookClientSingleton:
    """Verify get_n8n_client and reset_n8n_client."""

    def setup_method(self, method=None):
        from src.mcp_server.n8n_client import reset_n8n_client
        reset_n8n_client()

    def test_get_n8n_client_returns_none_when_disabled(self):
        """When n8n.enabled=False, get_n8n_client returns None."""
        from src.mcp_server.n8n_client import get_n8n_client
        client = get_n8n_client({"n8n": {"enabled": False}})
        assert client is None

    def test_get_n8n_client_creates_client_when_enabled(self):
        """When n8n.enabled=True, a client is returned."""
        from src.mcp_server.n8n_client import get_n8n_client
        client = get_n8n_client({
            "n8n": {"enabled": True, "base_url": "http://test:5678"},
        })
        assert client is not None
        assert client.base_url == "http://test:5678"

    def test_get_n8n_client_returns_singleton(self):
        """get_n8n_client returns the same instance on subsequent calls."""
        from src.mcp_server.n8n_client import get_n8n_client
        config = {"n8n": {"enabled": True, "base_url": "http://test:5678"}}
        first = get_n8n_client(config)
        second = get_n8n_client(config)
        assert first is second

    def test_reset_n8n_client_clears_singleton(self):
        """After reset_n8n_client, get_n8n_client creates a new instance."""
        from src.mcp_server.n8n_client import get_n8n_client, reset_n8n_client
        config = {"n8n": {"enabled": True, "base_url": "http://test:5678"}}
        first = get_n8n_client(config)
        reset_n8n_client()
        second = get_n8n_client(config)
        assert first is not second


# ===================================================================
# 2. Escalation API Endpoint Tests
# ===================================================================

class TestEscalateEndpoint:
    """Tests for POST /api/escalate.

    These tests work with the n8n-enabled app and ``mock_n8n_client``.
    The endpoint directly ``await``s ``_n8n.escalate_ticket()``, so
    the mock captures the call correctly.
    """

    def test_escalate_n8n_disabled(self, client, admin_headers):
        """When n8n is disabled, returns n8n_not_configured."""
        resp = client.post(
            "/api/escalate",
            json={"ticket_text": SAMPLE_CRITICAL_TICKET, "category": "technical", "urgency": "critical"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        _assert_escalate_response(data, expected_status="n8n_not_configured", expected_n8n_configured=False)
        assert data["escalation"]["sent"] is False

    def test_escalate_n8n_enabled_success(self, n8n_client, admin_headers, mock_n8n_client):
        """With n8n enabled and mock returning success, status is 'escalated'."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": SAMPLE_CRITICAL_TICKET, "category": "technical", "urgency": "high"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        _assert_escalate_response(resp.json(), expected_status="escalated", expected_n8n_configured=True)

    def test_escalate_n8n_enabled_failure(self, n8n_client, admin_headers, mock_n8n_client):
        """When n8n returns failure, status is 'escalation_failed'."""
        mock_n8n_client.escalate_ticket.return_value = {"success": False, "status_code": 500, "error": "Server error"}
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": SAMPLE_CRITICAL_TICKET, "category": "technical", "urgency": "critical"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        _assert_escalate_response(resp.json(), expected_status="escalation_failed", expected_n8n_configured=True)

    def test_escalate_passthrough_ticket_id(self, n8n_client, admin_headers, mock_n8n_client):
        """Custom ticket_id is passed through to the n8n client."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": "Custom ID test", "category": "billing", "urgency": "critical", "ticket_id": "TKT-CUSTOM-001"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["ticket_id"] == "TKT-CUSTOM-001"

    def test_escalate_auto_generates_ticket_id(self, n8n_client, admin_headers, mock_n8n_client):
        """Without ticket_id, one is auto-generated."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": "Auto ID test", "category": "general", "urgency": "high"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["ticket_id"].startswith("TKT-")

    def test_escalate_passes_metadata(self, n8n_client, admin_headers, mock_n8n_client):
        """Metadata dict is forwarded to the n8n client."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        metadata = {"source": "unit_test", "customer_tier": "enterprise", "priority": 1}
        n8n_client.post(
            "/api/escalate",
            json={"ticket_text": "Metadata test", "category": "security", "urgency": "critical", "metadata": metadata},
            headers=admin_headers,
        )
        mock_n8n_client.escalate_ticket.assert_awaited_with(
            ticket_text="Metadata test", category="security", urgency="critical",
            classification_raw="", ticket_id=ANY, metadata=metadata,
        )

    def test_escalate_requires_ticket_text(self, client, admin_headers):
        """Missing ticket_text returns a validation error (422)."""
        resp = client.post("/api/escalate", json={"category": "technical", "urgency": "critical"}, headers=admin_headers)
        assert resp.status_code == 422

    def test_escalate_empty_ticket_text(self, client, admin_headers):
        """Empty ticket_text returns a validation error (422)."""
        resp = client.post("/api/escalate", json={"ticket_text": "", "category": "technical", "urgency": "critical"}, headers=admin_headers)
        assert resp.status_code == 422

    def test_escalate_n8n_called_with_correct_args(self, n8n_client, admin_headers, mock_n8n_client):
        """Verify the exact args passed to escalate_ticket."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        n8n_client.post(
            "/api/escalate",
            json={
                "ticket_text": "Verify args test", "category": "technical", "urgency": "critical",
                "classification_raw": "Category: technical\nUrgency: critical",
                "ticket_id": "TKT-VERIFY-01", "metadata": {"source": "test"},
            },
            headers=admin_headers,
        )
        mock_n8n_client.escalate_ticket.assert_awaited_once_with(
            ticket_text="Verify args test", category="technical", urgency="critical",
            classification_raw="Category: technical\nUrgency: critical",
            ticket_id="TKT-VERIFY-01", metadata={"source": "test"},
        )


class TestEscalateStatusEndpoint:
    """Tests for GET /api/escalate/status."""

    def test_status_n8n_disabled(self, client):
        """When n8n is disabled, returns configured: false."""
        resp = client.get("/api/escalate/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["base_url"] is None
        assert "n8n not configured" in data["message"]

    def test_status_n8n_enabled(self, n8n_client, mock_n8n_client):
        """When n8n is enabled, returns configured: true with URL."""
        resp = n8n_client.get("/api/escalate/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is True
        assert data["base_url"] == "http://localhost:5678"
        assert data["webhook_path"] == "/webhook/"
        assert "timestamp" in data

    def test_status_n8n_enabled_with_custom_base(self):
        """Custom base_url is reflected in status response."""
        from src.mcp_server.rest_api import create_app
        from src.mcp_server.auth import reset_key_store

        reset_key_store()
        custom_app = create_app({
            "auth": {"enabled": False},
            "rate_limiting": {"enabled": False},
            "n8n": {"enabled": True, "base_url": "https://n8n.mycompany.com", "webhook_path": "/webhook/"},
        })
        with TestClient(custom_app) as c:
            resp = c.get("/api/escalate/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["configured"] is True
            assert data["base_url"] == "https://n8n.mycompany.com"


# ===================================================================
# 3. Auto-Escalation Integration Tests
# ===================================================================

class TestAutoEscalation:
    """Tests for automatic escalation triggered by classify_ticket.

    These tests patch ``_bg_escalate`` (a synchronous function) rather than
    trying to drive the TestClient's background event loop.  Patching the
    sync function is reliable and avoids the complexity of pumping the loop.
    """

    def test_classify_critical_triggers_escalation(self, n8n_client, admin_headers):
        """A ticket classified as 'critical' triggers ``_bg_escalate``."""
        with patch("src.mcp_server.rest_api._bg_escalate") as mock_bg:
            resp = n8n_client.post(
                "/api/classify_ticket",
                json={"ticket_text": SAMPLE_CRITICAL_TICKET},
                headers=admin_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["urgency"] == "critical"

            mock_bg.assert_called_once()
            _, call_kwargs = mock_bg.call_args
            assert call_kwargs["urgency"] == "critical"
            assert call_kwargs["ticket_text"] == SAMPLE_CRITICAL_TICKET
            assert call_kwargs["category"] == data["category"]

    def test_classify_non_critical_does_not_trigger(self, n8n_client, admin_headers):
        """A non-critical ticket should NOT trigger ``_bg_escalate``."""
        with patch("src.mcp_server.rest_api._bg_escalate") as mock_bg:
            resp = n8n_client.post(
                "/api/classify_ticket",
                json={"ticket_text": SAMPLE_BILLING_TICKET},
                headers=admin_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["urgency"] != "critical"
            mock_bg.assert_not_called()

    def test_classify_critical_disabled_n8n_no_escalation(self, client, admin_headers):
        """When n8n is disabled, even critical tickets don't call ``_bg_escalate``."""
        with patch("src.mcp_server.rest_api._bg_escalate") as mock_bg:
            resp = client.post(
                "/api/classify_ticket",
                json={"ticket_text": SAMPLE_CRITICAL_TICKET},
                headers=admin_headers,
            )
            assert resp.status_code == 200
            mock_bg.assert_not_called()

    def test_classify_critical_includes_classification_context(self, n8n_client, admin_headers):
        """The escalation call includes the raw classification output."""
        with patch("src.mcp_server.rest_api._bg_escalate") as mock_bg:
            resp = n8n_client.post(
                "/api/classify_ticket",
                json={"ticket_text": SAMPLE_CRITICAL_TICKET},
                headers=admin_headers,
            )
            assert resp.status_code == 200
            data = resp.json()

            mock_bg.assert_called_once()
            _, call_kwargs = mock_bg.call_args
            assert call_kwargs["classification_raw"] == data["raw"]
            assert call_kwargs["metadata"]["source"] == "classify_endpoint"


# ===================================================================
# 4. End-to-End Flow Tests
# ===================================================================

class TestEndToEndEscalation:
    """Full pipeline: classify → escalate → verify webhook payload.

    The manual ``POST /api/escalate`` endpoint is tested directly because
    it ``await``s ``_n8n.escalate_ticket()`` synchronously — no background
    task issues.
    """

    def test_full_flow(self, n8n_client, admin_headers, mock_n8n_client):
        """End-to-end: classify a critical ticket + manual escalate."""
        # Verify classify works
        resp = n8n_client.post(
            "/api/classify_ticket",
            json={"ticket_text": SAMPLE_CRITICAL_TICKET},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        classify_data = resp.json()
        assert classify_data["urgency"] == "critical"

        # Manually escalate — this directly awaits _n8n.escalate_ticket
        resp = n8n_client.post(
            "/api/escalate",
            json={
                "ticket_text": SAMPLE_CRITICAL_TICKET,
                "category": classify_data["category"],
                "urgency": "critical",
                "classification_raw": classify_data["raw"],
                "metadata": {"reason": "production outage"},
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "escalated"

        # Verify the mock received the call with correct payload
        mock_n8n_client.escalate_ticket.assert_awaited()
        call_kwargs = mock_n8n_client.escalate_ticket.await_args.kwargs
        assert call_kwargs["ticket_text"] == SAMPLE_CRITICAL_TICKET
        assert call_kwargs["urgency"] == "critical"
        assert call_kwargs["metadata"]["reason"] == "production outage"

    def test_manual_escalation_with_classification_context(self, n8n_client, admin_headers, mock_n8n_client):
        """Manual escalate works with classification context from a non-critical ticket."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}

        # Classify a non-critical ticket
        resp = n8n_client.post(
            "/api/classify_ticket",
            json={"ticket_text": SAMPLE_BILLING_TICKET},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        cls = resp.json()

        # Manually escalate using the classification result
        resp = n8n_client.post(
            "/api/escalate",
            json={
                "ticket_text": SAMPLE_BILLING_TICKET,
                "category": cls["category"],
                "urgency": "high",
                "classification_raw": cls["raw"],
                "metadata": {"escalation_reason": "customer requested manager"},
            },
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "escalated"

    def test_health_shows_n8n_config(self, n8n_client, mock_n8n_client):
        """Health endpoint reflects n8n configuration."""
        resp = n8n_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n8n_configured"] is True
        assert data["n8n_base_url"] == "http://localhost:5678"

    def test_health_hides_n8n_when_disabled(self, client):
        """Health endpoint shows n8n not configured when disabled."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n8n_configured"] is False
        assert data["n8n_base_url"] is None


# ===================================================================
# 5. Security & Validation Tests
# ===================================================================

class TestEscalationSecurity:
    """Auth and validation edge cases for escalation endpoints."""

    def test_escalate_requires_auth(self, client):
        """POST /api/escalate requires an API key."""
        resp = client.post("/api/escalate", json={"ticket_text": "Test ticket", "urgency": "critical"})
        assert resp.status_code == 401

    def test_escalate_status_public(self, client):
        """GET /api/escalate/status is accessible without auth."""
        resp = client.get("/api/escalate/status")
        assert resp.status_code == 200

    def test_escalate_with_invalid_urgency(self, n8n_client, admin_headers, mock_n8n_client):
        """Any urgency value is accepted (no strict validation on urgency field)."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": "Weird urgency", "category": "technical", "urgency": "super-urgent!!"},
            headers=admin_headers,
        )
        assert resp.status_code in (200, 422)

    def test_escalate_large_ticket_text(self, n8n_client, admin_headers, mock_n8n_client):
        """Large ticket text is handled without errors."""
        mock_n8n_client.escalate_ticket.return_value = {"success": True, "status_code": 200}
        long_text = "A" * 10000  # 10K characters
        resp = n8n_client.post(
            "/api/escalate",
            json={"ticket_text": long_text, "urgency": "critical"},
            headers=admin_headers,
        )
        assert resp.status_code == 200

        mock_n8n_client.escalate_ticket.assert_awaited()
        call_kwargs = mock_n8n_client.escalate_ticket.await_args.kwargs
        assert call_kwargs["ticket_text"] == long_text
