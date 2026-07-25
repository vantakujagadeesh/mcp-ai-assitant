"""
Conversation Memory / Session Management

Provides persistent conversation context for multi-turn interactions
with the AI assistant. Supports in-memory (default, with optional
file persistence) and Redis-backed storage.

Each conversation is identified by a ``session_id`` and stores a list
of messages with metadata (timestamp, role, content).
"""

import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    """A conversation session containing multiple messages."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# In-memory conversation store (with optional file persistence)
# ---------------------------------------------------------------------------
MAX_CONVERSATIONS_IN_MEMORY = 10000
MAX_MESSAGES_PER_CONVERSATION = 200
FILE_PERSIST_INTERVAL = 30  # seconds between file saves


class InMemoryConversationStore:
    """Store conversations in memory with optional JSON file persistence.

    Automatically evicts the oldest conversations when the limit is reached.
    """

    def __init__(self, persistence_path: Optional[str] = None) -> None:
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._persistence_path = persistence_path
        self._last_save = 0.0

        # Load persisted conversations if available
        if persistence_path and os.path.exists(persistence_path):
            self._load_from_file()

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Create a new conversation session."""
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        conv = Conversation(
            session_id=sid,
            metadata=metadata or {},
        )
        self._conversations[sid] = conv
        self._ensure_capacity()
        self._maybe_persist()
        return conv

    def get_session(self, session_id: str) -> Optional[Conversation]:
        """Retrieve a conversation session by ID."""
        conv = self._conversations.get(session_id)
        if conv:
            # Move to end to mark as recently used (LRU)
            self._conversations.move_to_end(session_id)
        return conv

    def get_or_create_session(self, session_id: str) -> Conversation:
        """Get an existing session or create a new one."""
        conv = self.get_session(session_id)
        if conv is None:
            conv = self.create_session(session_id=session_id)
        return conv

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        conv = self.get_or_create_session(session_id)
        if len(conv.messages) >= MAX_MESSAGES_PER_CONVERSATION:
            logger.warning(
                "Conversation %s has reached max messages (%d). Truncating oldest.",
                session_id, MAX_MESSAGES_PER_CONVERSATION,
            )
            # Remove oldest messages (keep the most recent half)
            conv.messages = conv.messages[-(MAX_MESSAGES_PER_CONVERSATION // 2):]

        msg = Message(role=role, content=content, metadata=metadata or {})
        conv.messages.append(msg)
        conv.updated_at = time.time()
        self._maybe_persist()
        return msg

    def get_history(
        self,
        session_id: str,
        max_messages: int = 20,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get conversation history, most recent last."""
        conv = self.get_session(session_id)
        if conv is None:
            return []

        messages = conv.messages
        if since_timestamp is not None:
            messages = [m for m in messages if m.timestamp > since_timestamp]

        # Return the most recent messages up to max_messages
        return [m.to_dict() for m in messages[-max_messages:]]

    def clear_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            self._maybe_persist()
            return True
        return False

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all active sessions with summary info."""
        sessions = list(self._conversations.values())
        # Sort by most recent activity
        sessions.sort(key=lambda c: c.updated_at, reverse=True)
        return [
            {
                "session_id": c.session_id,
                "message_count": len(c.messages),
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "metadata": c.metadata,
            }
            for c in sessions[offset:offset + limit]
        ]

    def _ensure_capacity(self) -> None:
        """Evict oldest conversations if over capacity."""
        while len(self._conversations) > MAX_CONVERSATIONS_IN_MEMORY:
            self._conversations.popitem(last=False)

    def _maybe_persist(self) -> None:
        """Save to file if persistence is configured and enough time has passed."""
        if not self._persistence_path:
            return
        now = time.time()
        if now - self._last_save < FILE_PERSIST_INTERVAL:
            return
        self._persist_now()

    def _persist_now(self) -> None:
        """Immediately persist to file."""
        if not self._persistence_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persistence_path), exist_ok=True)
            data = {
                sid: conv.to_dict()
                for sid, conv in self._conversations.items()
            }
            with open(self._persistence_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self._last_save = time.time()
            logger.debug("Persisted %d conversations to %s", len(data), self._persistence_path)
        except Exception as exc:
            logger.warning("Failed to persist conversations: %s", exc)

    def _load_from_file(self) -> None:
        """Load conversations from a JSON file."""
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for sid, conv_data in data.items():
                messages = [
                    Message(**m) if isinstance(m, dict) else m
                    for m in conv_data.get("messages", [])
                ]
                conv = Conversation(
                    session_id=sid,
                    messages=messages,
                    created_at=conv_data.get("created_at", 0),
                    updated_at=conv_data.get("updated_at", 0),
                    metadata=conv_data.get("metadata", {}),
                )
                self._conversations[sid] = conv
            logger.info("Loaded %d conversations from %s", len(data), self._persistence_path)
        except Exception as exc:
            logger.warning("Failed to load conversations from %s: %s", self._persistence_path, exc)

    def get_session_count(self) -> int:
        return len(self._conversations)

    def get_total_messages(self) -> int:
        return sum(len(c.messages) for c in self._conversations.values())


# ---------------------------------------------------------------------------
# Conversation Store for building prompt context
# ---------------------------------------------------------------------------
def build_conversation_context(
    history: List[Dict[str, Any]],
    max_tokens: int = 2048,
) -> str:
    """Build a conversation context string from message history.

    This formats the history for inclusion in an LLM prompt so the model
    has context of previous turns.

    Args:
        history: List of message dicts with 'role' and 'content' keys.
        max_tokens: Rough token limit (approximated by characters / 4).

    Returns:
        Formatted conversation context string.
    """
    if not history:
        return ""

    parts = []
    char_limit = max_tokens * 4  # rough estimate
    char_count = 0

    # Process from most recent backwards to fit within limit
    for msg in reversed(history):
        role_label = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        entry = f"{role_label}: {content}\n"

        if char_count + len(entry) > char_limit:
            break
        parts.insert(0, entry)
        char_count += len(entry)

    if parts:
        context = "Previous conversation:\n" + "".join(parts)
        return context
    return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_conversation_store(
    use_redis: bool = False,
    redis_client: Any = None,
    persistence_path: Optional[str] = None,
) -> Any:
    """Create the appropriate conversation store."""
    if use_redis and redis_client is not None:
        try:
            return RedisConversationStore(redis_client)
        except Exception as exc:
            logger.warning("Redis conversation store failed (%s) — using in-memory", exc)
    return InMemoryConversationStore(persistence_path=persistence_path)


# ---------------------------------------------------------------------------
# Redis-backed store (optional)
# ---------------------------------------------------------------------------
class RedisConversationStore:
    """Conversation store backed by Redis for persistence across restarts."""

    def __init__(self, redis_client: Any, key_prefix: str = "conversation:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    def _conv_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        conv = Conversation(session_id=sid, metadata=metadata or {})
        await self._redis.set(
            self._conv_key(sid),
            json.dumps(conv.to_dict()),
            ex=86400 * 7,  # 7 day TTL
        )
        return conv

    async def get_session(self, session_id: str) -> Optional[Conversation]:
        data = await self._redis.get(self._conv_key(session_id))
        if data:
            conv_dict = json.loads(data)
            messages = [Message(**m) for m in conv_dict.get("messages", [])]
            return Conversation(
                session_id=conv_dict["session_id"],
                messages=messages,
                created_at=conv_dict.get("created_at", 0),
                updated_at=conv_dict.get("updated_at", 0),
                metadata=conv_dict.get("metadata", {}),
            )
        return None

    async def get_or_create_session(self, session_id: str) -> Conversation:
        conv = await self.get_session(session_id)
        if conv is None:
            conv = await self.create_session(session_id=session_id)
        return conv

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        conv = await self.get_or_create_session(session_id)
        msg = Message(role=role, content=content, metadata=metadata or {})
        conv.messages.append(msg)
        conv.updated_at = time.time()
        await self._redis.set(
            self._conv_key(session_id),
            json.dumps(conv.to_dict()),
            ex=86400 * 7,
        )
        return msg

    async def get_history(
        self,
        session_id: str,
        max_messages: int = 20,
        since_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        conv = await self.get_session(session_id)
        if conv is None:
            return []
        messages = conv.messages
        if since_timestamp is not None:
            messages = [m for m in messages if m.timestamp > since_timestamp]
        return [m.to_dict() for m in messages[-max_messages:]]

    async def clear_session(self, session_id: str) -> bool:
        result = await self._redis.delete(self._conv_key(session_id))
        return result > 0

    async def list_sessions(self) -> List[Dict[str, Any]]:
        # Redis doesn't support listing keys with pattern efficiently in production
        # This is a simplified version using SCAN
        sessions = []
        async for key in self._redis.scan_iter(match=f"{self._prefix}*"):
            data = await self._redis.get(key)
            if data:
                conv_dict = json.loads(data)
                sessions.append({
                    "session_id": conv_dict["session_id"],
                    "message_count": len(conv_dict.get("messages", [])),
                    "created_at": conv_dict.get("created_at", 0),
                    "updated_at": conv_dict.get("updated_at", 0),
                })
        sessions.sort(key=lambda c: c["updated_at"], reverse=True)
        return sessions
