"""
Shared pytest fixtures for the MCP AI Assistant test suite.

Provides:
- ``app`` — FastAPI TestClient application with configurable n8n/auth state
- ``client`` — FastAPI TestClient wrapping the app
- ``mock_n8n_client`` — mocked N8nWebhookClient that records calls
- ``admin_headers`` — pre-configured auth headers for admin access
"""

import os
import sys
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set up admin API key before importing app modules
os.environ["MCP_API_KEY_ADMIN"] = "sk-test-admin-key:admin"

from src.mcp_server.rest_api import create_app
from src.mcp_server.auth import reset_key_store


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

def _base_config(n8n_enabled: bool = False, **overrides: Any) -> Dict[str, Any]:
    """Return a base configuration dict with sensible test defaults.

    Args:
        n8n_enabled: Whether to enable the n8n webhook client.
        **overrides: Additional config sections to merge.

    Returns:
        A configuration dict suitable for ``create_app()``.
    """
    config: Dict[str, Any] = {
        "auth": {
            "enabled": True,
            "public_paths": [
                "/health", "/docs", "/openapi.json", "/redoc",
                "/admin/dashboard", "/metrics",
                "/api/escalate/status",
            ],
        },
        "rate_limiting": {"enabled": False},
        "vector_db": {"enabled": False},  # disable Qdrant for tests (speeds up + avoids timeout)
        "rag": {"kb_path": "/dev/null/non_existent_kb.jsonl"},  # skip sentence-transformers model loading
        "n8n": {
            "enabled": n8n_enabled,
            "base_url": "http://localhost:5678",
            "webhook_path": "/webhook/",
            "timeout_seconds": 5.0,
        },
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Public test constants
# ---------------------------------------------------------------------------

ADMIN_KEY = "sk-test-admin-key"
ADMIN_HEADERS = {"X-API-Key": ADMIN_KEY}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _setup_env() -> Generator[None, None, None]:
    """Ensure the admin API key env var is set for the entire session."""
    old_val = os.environ.get("MCP_API_KEY_ADMIN")
    os.environ["MCP_API_KEY_ADMIN"] = "sk-test-admin-key:admin"
    reset_key_store()
    yield
    if old_val is None:
        os.environ.pop("MCP_API_KEY_ADMIN", None)
    else:
        os.environ["MCP_API_KEY_ADMIN"] = old_val


@pytest.fixture
def app() -> Generator[Any, None, None]:
    """Create a fresh FastAPI app with n8n **disabled**."""
    reset_key_store()
    app_instance = create_app(_base_config(n8n_enabled=False))
    yield app_instance


@pytest.fixture
def n8n_enabled_app() -> Generator[Any, None, None]:
    """Create a fresh FastAPI app with n8n **enabled**."""
    reset_key_store()
    app_instance = create_app(_base_config(n8n_enabled=True))
    yield app_instance


@pytest.fixture
def client(app: Any) -> Generator[TestClient, None, None]:
    """Yield a TestClient wrapping the default (n8n-disabled) app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def n8n_client(n8n_enabled_app: Any) -> Generator[TestClient, None, None]:
    """Yield a TestClient wrapping the n8n-enabled app."""
    with TestClient(n8n_enabled_app) as c:
        yield c


@pytest.fixture
def admin_headers() -> Dict[str, str]:
    """Return HTTP headers carrying a valid admin-tier API key."""
    return dict(ADMIN_HEADERS)


@pytest.fixture
def mock_n8n_client() -> Generator[AsyncMock, None, None]:
    """Mock the global ``_n8n`` object inside ``rest_api``.

    Patches ``src.mcp_server.rest_api._n8n`` with an ``AsyncMock``
    whose ``escalate_ticket`` method returns a configurable result.

    After the test, restores the original value.

    Usage::

        def test_my_thing(mock_n8n_client):
            mock_n8n_client.escalate_ticket.return_value = {
                "success": True, "status_code": 200
            }
            # ... test code that triggers escalation ...
            mock_n8n_client.escalate_ticket.assert_awaited_once()
    """
    mock_client = AsyncMock(spec=[])
    mock_client.base_url = "http://localhost:5678"
    mock_client.webhook_path = "/webhook/"
    mock_client.escalate_ticket = AsyncMock(
        return_value={"success": True, "status_code": 200}
    )

    patcher = patch("src.mcp_server.rest_api._n8n", mock_client)
    patcher.start()
    yield mock_client
    patcher.stop()
