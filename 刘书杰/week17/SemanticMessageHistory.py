"""
Semantic Message History
========================

Conversation history for LLM agents / chatbots, persisted in Redis
with optional **semantic search** over past messages.

Features
--------
* Session-scoped storage — isolate conversations by ``session_id``.
* Automatic trimming — keep only the last *N* messages.
* Semantic search — find past messages by meaning, not keywords.
* TTL support — auto-expire idle sessions.
* LangChain-compatible — can be used as a drop-in
  ``BaseChatMessageHistory`` backend (just wrap or subclass).

Usage
-----
.. code-block:: python

    import redis.asyncio as redis
    from vl_redis import SemanticMessageHistory

    r = redis.Redis()
    history = SemanticMessageHistory(r, session_id="user-42", max_messages=100)

    # Add messages
    await history.add_message({"role": "user", "content": "What is Redis?"})
    await history.add_message({"role": "assistant", "content": "Redis is ..."})

    # Retrieve full history
    messages = await history.get_messages()

    # Semantic search (requires embed_fn)
    results = await history.search("Tell me about caching", k=5)
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import redis.asyncio as redis

from vl_redis._types import EmbeddingFunc, MessageRecord
from vl_redis._utils import (
    _bytes_to_float32,
    _float_to_bytes,
    cosine_similarity,
    json_dumps,
    json_loads,
    make_session_key,
)


class SemanticMessageHistory:
    """Redis-backed conversation history with optional semantic search.

    Parameters
    ----------
    redis_client : redis.Redis
        Async Redis client.
    session_id : str
        Unique identifier for this conversation.
    max_messages : int | None
        Maximum number of messages to retain.  When exceeded the oldest
        messages are trimmed automatically on each ``add_message()``.
        ``None`` means unbounded.
    embed_fn : EmbeddingFunc | None
        Embedding function for semantic search.  If ``None``,
        ``search()`` and ``search_messages()`` will raise.
    ttl : int | None
        Session TTL in seconds (refreshed on writes).
    prefix : str
        Redis key prefix.  Default ``"msghist"``.
    """

    # Redis key suffixes
    _K_MESSAGES = "messages"
    _K_EMBEDDINGS = "embeddings"
    _K_META = "meta"

    def __init__(
        self,
        redis_client: redis.Redis,
        session_id: str,
        *,
        max_messages: Optional[int] = None,
        embed_fn: Optional[EmbeddingFunc] = None,
        ttl: Optional[int] = None,
        prefix: str = "msghist",
    ) -> None:
        self._redis = redis_client
        self._session_id = session_id
        self._max_messages = max_messages
        self._embed_fn = embed_fn
        self._ttl = ttl
        self._prefix = prefix

        # Track this session in the global registry
        self._registered = False

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def _base(self) -> str:
        return make_session_key(self._prefix, self._session_id)

    @property
    def _msg_key(self) -> str:
        return f"{self._base}:{self._K_MESSAGES}"

    @property
    def _emb_key(self) -> str:
        return f"{self._base}:{self._K_EMBEDDINGS}"

    @property
    def _meta_key(self) -> str:
        return f"{self._base}:{self._K_META}"

    @property
    def _sessions_key(self) -> str:
        return f"{self._prefix}:sessions"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _ensure_registered(self) -> None:
        """Add this session to the global session set (idempotent)."""
        if self._registered:
            return
        await self._redis.sadd(self._sessions_key, self._session_id)
        self._registered = True

    # ------------------------------------------------------------------
    # Add messages
    # ------------------------------------------------------------------

    async def add_message(
        self,
        message: MessageRecord | Dict[str, Any],
        *,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Append a single message to the history.

        Parameters
        ----------
        message : MessageRecord | dict
            A message with at least ``"role"`` and ``"content"`` keys.
            Also accepts LangChain ``BaseMessage``-like objects if they
            have a ``.dict()`` method.
        embedding : List[float] | None
            Pre-computed embedding of the message content for semantic
            search.  If omitted and *embed_fn* is configured the embedding
            is computed automatically.
        """
        await self._ensure_registered()
        msg_dict = _to_dict(message)

        # Optionally embed
        if embedding is None and self._embed_fn is not None:
            embedding = await _maybe_await(self._embed_fn(msg_dict.get("content", "")))

        idx = await self._redis.rpush(self._msg_key, json_dumps(msg_dict))

        if embedding is not None:
            await self._redis.hset(self._emb_key, str(idx - 1), _float_to_bytes(embedding))

        # TTL refresh
        if self._ttl is not None:
            async with self._redis.pipeline() as pipe:
                pipe.expire(self._msg_key, self._ttl)
                pipe.expire(self._emb_key, self._ttl)
                pipe.expire(self._meta_key, self._ttl)
                await pipe.execute()

        # Trim if needed
        if self._max_messages is not None:
            await self._trim()

        # Update session metadata
        await self._redis.hset(
            self._meta_key,
            mapping={
                "last_updated": str(time.time()),
                "message_count": str(await self.count()),
            },
        )

    async def add_messages(
        self,
        messages: Sequence[MessageRecord | Dict[str, Any]],
    ) -> None:
        """Append multiple messages in one pipeline."""
        for msg in messages:
            await self.add_message(msg)

    # ------------------------------------------------------------------
    # Retrieve messages
    # ------------------------------------------------------------------

    async def get_messages(
        self,
        start: int = 0,
        end: int = -1,
    ) -> List[Dict[str, Any]]:
        """Return messages in chronological order (oldest first).

        Parameters
        ----------
        start, end : int
            Python slice semantics on the underlying Redis list.
            ``(0, -1)`` returns everything.
        """
        raw = await self._redis.lrange(self._msg_key, start, end)
        return [json_loads(r) for r in raw]

    async def get_message(self, index: int) -> Optional[Dict[str, Any]]:
        """Return a single message by index (0 = oldest)."""
        raw = await self._redis.lindex(self._msg_key, index)
        if raw is None:
            return None
        return json_loads(raw)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        embedding: Optional[List[float]] = None,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search message history by semantic similarity.

        Parameters
        ----------
        query : str
            The search query.
        k : int
            Number of top results to return.
        embedding : List[float] | None
            Pre-computed embedding of *query*.
        threshold : float
            Minimum cosine similarity (0–1).  Results below this are
            filtered out.

        Returns
        -------
        List[Tuple[dict, float]]
            Sorted list of ``(message_dict, similarity)``, best first.
        """
        if self._embed_fn is None:
            raise RuntimeError(
                "SemanticMessageHistory.search() requires embed_fn to be configured"
            )

        if embedding is None:
            embedding = await _maybe_await(self._embed_fn(query))

        # Fetch all message embeddings
        emb_map = await self._redis.hgetall(self._emb_key)
        if not emb_map:
            return []

        # Score every embedded message
        scored: List[Tuple[int, float]] = []
        for idx_bytes, vec_bytes in emb_map.items():
            idx = int(idx_bytes)
            vec = _bytes_to_float32(vec_bytes)
            score = cosine_similarity(embedding, vec)
            if score >= threshold:
                scored.append((idx, score))

        # Sort by similarity descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        # Fetch the actual messages (pipelined)
        pipe = self._redis.pipeline()
        for idx, _ in top:
            pipe.lindex(self._msg_key, idx)
        results = await pipe.execute()

        return [
            (json_loads(msg_raw), score)
            for (_, score), msg_raw in zip(top, results)
            if msg_raw is not None
        ]

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the number of messages in this session."""
        return await self._redis.llen(self._msg_key)

    async def clear(self) -> None:
        """Remove all messages and embeddings for this session."""
        async with self._redis.pipeline() as pipe:
            pipe.delete(self._msg_key)
            pipe.delete(self._emb_key)
            pipe.delete(self._meta_key)
            pipe.srem(self._sessions_key, self._session_id)
            await pipe.execute()

    async def trim(self, max_messages: int) -> int:
        """Trim history to *max_messages*, dropping the oldest first.

        Returns the number of messages removed.
        """
        current = await self.count()
        if current <= max_messages:
            return 0

        to_remove = current - max_messages
        async with self._redis.pipeline() as pipe:
            pipe.ltrim(self._msg_key, to_remove, -1)
            # Remove old embedding entries and re-index the survivors
            for i in range(to_remove):
                pipe.hdel(self._emb_key, str(i))
            await pipe.execute()

        # Re-index embedding hash keys after trimming
        # (the indices shifted — rebuild the hash)
        emb_raw = await self._redis.hgetall(self._emb_key)
        if emb_raw:
            new_emb: Dict[str, bytes] = {}
            for old_idx_bytes, vec in emb_raw.items():
                old_idx = int(old_idx_bytes)
                new_idx = old_idx - to_remove
                if new_idx >= 0:
                    new_emb[str(new_idx)] = vec
            await self._redis.delete(self._emb_key)
            if new_emb:
                await self._redis.hset(self._emb_key, mapping=new_emb)

        return to_remove

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    async def get_metadata(self) -> Dict[str, Any]:
        """Return session-level metadata."""
        raw = await self._redis.hgetall(self._meta_key)
        return {_to_str(k): _to_str(v) for k, v in raw.items()}

    async def set_metadata(self, key: str, value: Any) -> None:
        """Set a single metadata field."""
        await self._redis.hset(self._meta_key, key, json_dumps(value))

    # ------------------------------------------------------------------
    # Class-level — list / manage sessions
    # ------------------------------------------------------------------

    @classmethod
    async def list_sessions(
        cls,
        redis_client: redis.Redis,
        prefix: str = "msghist",
    ) -> List[str]:
        """Return all known session IDs."""
        key = f"{prefix}:sessions"
        members = await redis_client.smembers(key)
        return sorted(m.decode("utf-8") if isinstance(m, bytes) else m for m in members)

    @classmethod
    async def delete_session(
        cls,
        redis_client: redis.Redis,
        session_id: str,
        prefix: str = "msghist",
    ) -> bool:
        """Delete an entire session (messages + embeddings + metadata).

        Returns ``True`` if the session existed.
        """
        base = make_session_key(prefix, session_id)
        keys = [f"{base}:{s}" for s in ("messages", "embeddings", "meta")]
        removed = await redis_client.delete(*keys)
        await redis_client.srem(f"{prefix}:sessions", session_id)
        return bool(removed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _trim(self) -> None:
        """Internal auto-trim triggered on add_message."""
        if self._max_messages is None:
            return
        current = await self.count()
        if current > self._max_messages:
            await self.trim(self._max_messages)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dict(message: MessageRecord | Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a message to a plain dict.

    Handles:
    - Plain dicts (pass-through)
    - ``MessageRecord`` dataclass instances
    - Objects with a ``.dict()`` method (LangChain BaseMessage, Pydantic models)
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "dict"):
        return message.dict()  # type: ignore[union-attr]
    if hasattr(message, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(message)
    # Fallback: try dict-style access
    return dict(message)  # type: ignore[call-overload]


async def _maybe_await(result: Any) -> Any:
    """Await if awaitable, else return directly."""
    if hasattr(result, "__await__") or isinstance(result, Awaitable):
        return await result
    return result


def _to_str(val: bytes | str) -> str:
    """Decode bytes to str; pass str through unchanged."""
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)
