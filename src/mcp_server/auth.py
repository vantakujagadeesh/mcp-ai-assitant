"""
API Key Authentication Module

Provides API key validation middleware for the FastAPI REST server.
Supports multiple key tiers (admin, standard) with configurable keys
loaded from environment variables or config file.
"""

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_KEY_HEADER = "X-API-Key"
API_KEY_ENV_PREFIX = "MCP_API_KEY_"

# Key tiers
TIER_ADMIN = "admin"
TIER_STANDARD = "standard"

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS: Set[str] = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


# ---------------------------------------------------------------------------
# API Key Store
# ---------------------------------------------------------------------------
class APIKeyStore:
    """Manage API keys loaded from environment variables.

    Environment variables follow the pattern::

        MCP_API_KEY_<NAME>=<key>:<tier>

    Example::

        MCP_API_KEY_ADMIN=sk-admin-secret-key:admin
        MCP_API_KEY_SERVICE=sk-service-key-abc:standard

    A default key ``sk-default-key`` with ``standard`` tier is created
    automatically if no keys are configured.
    """

    def __init__(self, keys: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self._keys: Dict[str, Dict[str, str]] = {}
        self._key_hash_map: Dict[str, str] = {}  # key_hash -> name

        if keys:
            self._load_from_dict(keys)
        self._load_from_env()

        if not self._keys:
            self._add_key("default", "sk-default-key", TIER_STANDARD)
            logger.info("No API keys configured. Created default key: sk-default-key")

    def _load_from_dict(self, keys: Dict[str, Dict[str, str]]) -> None:
        """Load keys from a dictionary (used for testing)."""
        for name, info in keys.items():
            self._add_key(name, info.get("key", ""), info.get("tier", TIER_STANDARD))

    def _load_from_env(self) -> None:
        """Load keys from environment variables prefixed with MCP_API_KEY_."""
        for env_name, env_value in os.environ.items():
            if env_name.startswith(API_KEY_ENV_PREFIX):
                name = env_name[len(API_KEY_ENV_PREFIX):].lower()
                parts = env_value.split(":", 1)
                key = parts[0].strip()
                tier = parts[1].strip() if len(parts) > 1 else TIER_STANDARD
                if key:
                    self._add_key(name, key, tier)

    def _add_key(self, name: str, key: str, tier: str) -> None:
        """Add a key to the store."""
        key_hash = self._hash_key(key)
        self._keys[name] = {"key_hash": key_hash, "tier": tier, "name": name}
        self._key_hash_map[key_hash] = name

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage comparison."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate(self, api_key: str) -> Optional[Dict[str, str]]:
        """Validate an API key and return its metadata if valid."""
        key_hash = self._hash_key(api_key)
        name = self._key_hash_map.get(key_hash)
        if name and name in self._keys:
            return self._keys[name]
        return None

    def has_tier(self, api_key: str, required_tier: str) -> bool:
        """Check if the API key has at least the required tier."""
        info = self.validate(api_key)
        if not info:
            return False
        if required_tier == TIER_ADMIN and info["tier"] != TIER_ADMIN:
            return False
        return True

    def list_keys(self) -> List[Dict[str, str]]:
        """List all configured keys (excluding the actual key values)."""
        return [
            {"name": info["name"], "tier": info["tier"]}
            for info in self._keys.values()
        ]

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @classmethod
    def generate_key(cls) -> str:
        """Generate a new random API key."""
        return f"sk-{secrets.token_hex(24)}"


# ---------------------------------------------------------------------------
# Global store (lazy-loaded)
# ---------------------------------------------------------------------------
_key_store: Optional[APIKeyStore] = None


def get_key_store() -> APIKeyStore:
    """Get or create the global API key store."""
    global _key_store
    if _key_store is None:
        _key_store = APIKeyStore()
    return _key_store


def reset_key_store() -> None:
    """Reset the key store (useful for testing)."""
    global _key_store
    _key_store = None


# ---------------------------------------------------------------------------
# Authentication Middleware
# ---------------------------------------------------------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that validates API keys on every request
    (except public endpoints like /health).

    The client must include the API key in the ``X-API-Key`` header.

    Returns 401 if the key is missing or invalid.
    Returns 403 if the key doesn't have sufficient permissions.
    """

    def __init__(
        self,
        app: ASGIApp,
        key_store: Optional[APIKeyStore] = None,
        exclude_paths: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.key_store = key_store or get_key_store()
        self.exclude_paths = PUBLIC_ENDPOINTS | (exclude_paths or set())

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Skip auth for public endpoints
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Extract API key
        api_key = request.headers.get(API_KEY_HEADER, "")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": f"Missing API key. Provide it via the '{API_KEY_HEADER}' header.",
                },
                headers={"WWW-Authenticate": "APIKey"},
            )

        # Validate
        key_info = self.key_store.validate(api_key)
        if not key_info:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Invalid API key.",
                },
                headers={"WWW-Authenticate": "APIKey"},
            )

        # Attach key info to request state
        request.state.api_key_name = key_info["name"]
        request.state.api_key_tier = key_info["tier"]

        return await call_next(request)


# ---------------------------------------------------------------------------
# Dependency: require API key (for selective endpoint protection)
# ---------------------------------------------------------------------------
def require_api_key(request: Request) -> Dict[str, str]:
    """FastAPI dependency to require a valid API key on specific endpoints.

    Use this instead of the middleware when you want per-endpoint control.
    """
    api_key = request.headers.get(API_KEY_HEADER, "")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"Missing API key. Provide it via the '{API_KEY_HEADER}' header.",
        )

    store = get_key_store()
    key_info = store.validate(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    return key_info


def require_admin_key(request: Request) -> Dict[str, str]:
    """FastAPI dependency that requires an admin-level API key."""
    key_info = require_api_key(request)
    if key_info.get("tier") != TIER_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin-level API key required for this operation.",
        )
    return key_info
