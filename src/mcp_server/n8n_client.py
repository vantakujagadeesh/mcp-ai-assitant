"""
n8n Webhook Client

Sends payloads to n8n webhook endpoints for workflow automation.
Used for auto-escalation of critical tickets, feedback logging,
and other n8n-triggered actions.

Supports both sync (httpx/urllib) and async (httpx) modes.
Gracefully degrades when n8n is unreachable.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async HTTP client (preferred)
# ---------------------------------------------------------------------------
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fallback sync client
# ---------------------------------------------------------------------------
try:
    import urllib.request
    import urllib.error

    _URLLIB_AVAILABLE = True
except ImportError:
    _URLLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# n8n Client
# ---------------------------------------------------------------------------
class N8nWebhookClient:
    """Client for sending webhook payloads to n8n workflows.

    Args:
        base_url: Base URL of the n8n instance (e.g. ``http://localhost:5678``).
        webhook_path: Default webhook path (e.g. ``/webhook/``).
        api_key: Optional API key for authenticated webhooks.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        webhook_path: str = "/webhook/",
        api_key: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.webhook_path = webhook_path if webhook_path.startswith("/") else f"/{webhook_path}"
        self.api_key = api_key
        self.timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Build webhook URL
    # ------------------------------------------------------------------
    def webhook_url(self, webhook_id: str) -> str:
        """Build the full URL for an n8n webhook.

        Args:
            webhook_id: The webhook ID/path (e.g. ``ticket-escalate``).

        Returns:
            Full URL like ``http://localhost:5678/webhook/ticket-escalate``
        """
        path = urljoin(self.webhook_path, webhook_id.lstrip("/"))
        return urljoin(f"{self.base_url}/", path.lstrip("/"))

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if extra:
            headers.update(extra)
        return headers

    # ------------------------------------------------------------------
    # Send webhook (async via httpx)
    # ------------------------------------------------------------------
    async def send_async(
        self,
        webhook_id: str,
        payload: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a webhook payload to n8n asynchronously.

        Returns a dict with ``success``, ``status_code``, and optional ``error``.
        """
        url = self.webhook_url(webhook_id)
        timeout = timeout_seconds or self.timeout

        if _HTTPX_AVAILABLE:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
                    logger.info(
                        "n8n webhook '%s' sent successfully (status=%d)",
                        webhook_id,
                        resp.status_code,
                    )
                    return {"success": True, "status_code": resp.status_code}
            except httpx.TimeoutException:
                logger.warning("n8n webhook '%s' timed out after %ss", webhook_id, timeout)
                return {"success": False, "status_code": 0, "error": "timeout"}
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "n8n webhook '%s' returned %d: %s",
                    webhook_id,
                    exc.response.status_code,
                    exc.response.text[:200],
                )
                return {"success": False, "status_code": exc.response.status_code, "error": str(exc)}
            except httpx.RequestError as exc:
                logger.warning("n8n webhook '%s' request failed: %s", webhook_id, exc)
                return {"success": False, "status_code": 0, "error": str(exc)}
        else:
            # Fallback to sync urllib
            return await self._send_sync_fallback(url, payload, timeout)

    # ------------------------------------------------------------------
    # Send webhook (sync fallback)
    # ------------------------------------------------------------------
    async def _send_sync_fallback(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Fallback using urllib (runs in executor to avoid blocking)."""

        def _do_send() -> Dict[str, Any]:
            if not _URLLIB_AVAILABLE:
                logger.warning("No HTTP client available (httpx/urllib) — cannot send to n8n")
                return {"success": False, "status_code": 0, "error": "no_http_client"}

            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    logger.info(
                        "n8n webhook sent successfully (status=%d)",
                        resp.status,
                    )
                    return {"success": True, "status_code": resp.status}
            except urllib.error.HTTPError as exc:
                logger.warning("n8n webhook returned %d: %s", exc.code, exc.reason)
                return {"success": False, "status_code": exc.code, "error": exc.reason}
            except Exception as exc:
                logger.warning("n8n webhook failed: %s", exc)
                return {"success": False, "status_code": 0, "error": str(exc)}

        return await asyncio.get_event_loop().run_in_executor(None, _do_send)

    # ------------------------------------------------------------------
    # Convenience: escalate a critical ticket
    # ------------------------------------------------------------------
    async def escalate_ticket(
        self,
        ticket_text: str,
        category: str,
        urgency: str,
        classification_raw: str = "",
        ticket_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send an escalation alert to n8n for a critical/high-urgency ticket.

        This is called automatically when ``classify_ticket`` detects
        ``critical`` urgency, or manually via the ``/api/escalate`` endpoint.
        """
        from datetime import datetime, timezone

        payload = {
            "event": "ticket_escalated",
            "ticket_id": ticket_id or f"ESC-{int(time.time())}",
            "ticket_text": ticket_text,
            "category": category,
            "urgency": urgency,
            "classification_raw": classification_raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        return await self.send_async("ticket-escalate", payload)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------
_n8n_client: Optional[N8nWebhookClient] = None


def get_n8n_client(config: Optional[Dict[str, Any]] = None) -> Optional[N8nWebhookClient]:
    """Get or create the global n8n webhook client from config.

    Config format::

        n8n:
          base_url: "http://localhost:5678"
          webhook_path: "/webhook/"
          api_key: null
          timeout_seconds: 10
          enabled: false  # set to true to enable auto-escalation
    """
    global _n8n_client
    if _n8n_client is not None:
        return _n8n_client

    if config is None:
        return None

    n8n_config = config.get("n8n", {})
    if not n8n_config.get("enabled", False):
        return None

    try:
        _n8n_client = N8nWebhookClient(
            base_url=n8n_config.get("base_url", "http://localhost:5678"),
            webhook_path=n8n_config.get("webhook_path", "/webhook/"),
            api_key=n8n_config.get("api_key"),
            timeout_seconds=n8n_config.get("timeout_seconds", 10.0),
        )
        logger.info("n8n webhook client initialized (base: %s)", n8n_config.get("base_url"))
        return _n8n_client
    except Exception as exc:
        logger.warning("Failed to initialize n8n client: %s", exc)
        return None


def reset_n8n_client() -> None:
    """Reset the global n8n client (useful for testing)."""
    global _n8n_client
    _n8n_client = None
