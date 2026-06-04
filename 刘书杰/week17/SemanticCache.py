"""
Semantic Cache
==============

Cache LLM responses keyed by **semantic similarity** rather than exact
string match.  Two prompts that mean the same thing but are worded
differently will hit the same cached response — dramatically reducing
LLM API costs for common query patterns.

How it works
------------
1. Every stored (prompt, response) pair has its prompt embedded.
2. On lookup the query is embedded and compared (cosine similarity)
   against every cached prompt.
3. If the best similarity ≥ *similarity_threshold* the cached response
   is returned immediately — no LLM call needed.

Usage
-----
.. code-block:: python

    import redis.asyncio as redis
    from vl_redis import SemanticCache

    r = redis.Redis()
    cache = SemanticCache(r, embed_fn=my_embed_fn, similarity_threshold=0.92)

    # Before every LLM call:
    hit = await cache.lookup(user_prompt)
    if hit:
        response = hit.response       # cached — free!
    else:
        response = await call_llm(user_prompt)
        await cache.store(user_prompt, response)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import redis.asyncio as redis

from vl_redis._types import EmbeddingFunc
from vl_redis._utils import (
    _bytes_to_float32,
    _float_to_bytes,
    cosine_similarity,
    json_dumps,
    json_loads,
    make_cache_key,
)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class SemanticCacheHit:
    """A successful semantic cache lookup."""

    prompt: str
    """The *original* prompt that produced the cached response."""

    response: str
    """The cached LLM response."""

    similarity: float
    """Cosine similarity between the query and the matched prompt (0–1)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """User-supplied metadata stored with the entry."""

    created_at: float = 0.0
    """Unix timestamp of when the entry was stored."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Semantic LLM response cache backed by Redis.

    Parameters
    ----------
    redis_client : redis.Redis
        Async Redis client.
    embed_fn : EmbeddingFunc
        Function ``(str) -> List[float]`` used to embed prompts.
    similarity_threshold : float
        Minimum cosine similarity (0–1) to consider a cache hit.
        Higher = stricter (fewer hits).  0.90–0.95 is a good starting range.
    prefix : str
        Redis key namespace.  Default ``"semcache"``.
    default_ttl : int | None
        Default TTL in seconds for stored entries.
    max_entries : int | None
        Soft cap on stored entries.  When exceeded the oldest entries
        are evicted (best-effort, checked on ``store()``).
    """

    # Redis Hash field names
    _F_PROMPT = b"prompt"
    _F_RESPONSE = b"response"
    _F_VECTOR = b"vector"
    _F_META = b"metadata"
    _F_CREATED = b"created_at"

    def __init__(
        self,
        redis_client: redis.Redis,
        embed_fn: EmbeddingFunc,
        *,
        similarity_threshold: float = 0.92,
        prefix: str = "semcache",
        default_ttl: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

        self._redis = redis_client
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._max_entries = max_entries

        # Client-side stats
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def index_key(self) -> str:
        """Redis Set key that tracks all cache entry keys."""
        return f"{self._prefix}:index"

    async def lookup(
        self,
        query: str,
        *,
        embedding: Optional[List[float]] = None,
        top_k: int = 1,
    ) -> Optional[SemanticCacheHit]:
        """Search the cache for a semantically similar prompt.

        Parameters
        ----------
        query : str
            The incoming user prompt.
        embedding : List[float] | None
            Pre-computed embedding for *query*.  If omitted the cache
            computes it via ``embed_fn``.
        top_k : int
            Number of candidates to return (currently only ``1`` is used).

        Returns
        -------
        SemanticCacheHit | None
            Best match above *similarity_threshold*, or ``None``.
        """
        # 1. Embed the query
        if embedding is None:
            embedding = await _maybe_await(self._embed_fn(query))

        # 2. Collect candidate entry keys
        entry_keys = await self._get_all_entry_keys()
        if not entry_keys:
            self._misses += 1
            return None

        # 3. Fetch vectors for all candidates
        best = await self._find_best_match(embedding, entry_keys)

        if best is None or best.similarity < self._threshold:
            self._misses += 1
            return None

        self._hits += 1
        return best

    async def store(
        self,
        prompt: str,
        response: str,
        *,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a prompt→response pair in the cache.

        Parameters
        ----------
        prompt : str
            The user prompt.
        response : str
            The LLM response to cache.
        embedding : List[float] | None
            Pre-computed embedding of *prompt*.
        metadata : dict | None
            Arbitrary JSON-serialisable metadata.
        ttl : int | None
            TTL in seconds for this entry.

        Returns
        -------
        str
            The Redis key of the stored entry.
        """
        if embedding is None:
            embedding = await _maybe_await(self._embed_fn(prompt))

        key = make_cache_key(self._prefix, prompt)
        ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        mapping = {
            self._F_PROMPT: prompt,
            self._F_RESPONSE: response,
            self._F_VECTOR: _float_to_bytes(embedding),
            self._F_META: json_dumps(metadata or {}),
            self._F_CREATED: str(now),
        }

        async with self._redis.pipeline() as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.sadd(self.index_key, key)
            if ttl is not None:
                pipe.expire(key, ttl)
            await pipe.execute()

        # Index TTL — keep in sync with entry TTL (use a generous buffer)
        if ttl is not None:
            await self._redis.expire(self.index_key, ttl + 300)

        self._stores += 1

        # Eviction check
        if self._max_entries is not None:
            await self._evict_if_needed()

        return key

    async def delete(self, prompt: str) -> bool:
        """Remove a specific entry by its prompt text.

        Returns ``True`` if the entry existed.
        """
        key = make_cache_key(self._prefix, prompt)
        existed = await self._redis.exists(key)
        async with self._redis.pipeline() as pipe:
            pipe.delete(key)
            pipe.srem(self.index_key, key)
            await pipe.execute()
        return bool(existed)

    async def clear(self) -> int:
        """Remove **all** entries.  Returns the number of keys deleted."""
        keys = await self._get_all_entry_keys()
        if not keys:
            return 0
        async with self._redis.pipeline() as pipe:
            pipe.delete(self.index_key)
            for k in keys:
                pipe.delete(k)
            await pipe.execute()
        return len(keys)

    async def count(self) -> int:
        """Return the current number of cached entries."""
        return await self._redis.scard(self.index_key)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Client-side statistics (best-effort)."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Zero out client-side stats."""
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_all_entry_keys(self) -> List[bytes]:
        """Return all entry keys tracked in the index set."""
        members = await self._redis.smembers(self.index_key)
        return list(members)

    async def _find_best_match(
        self,
        query_vec: List[float],
        entry_keys: List[bytes],
    ) -> Optional[SemanticCacheHit]:
        """Brute-force cosine similarity across all cached entries.

        For caches with >10k entries consider adding a RediSearch vector
        index — but brute-force with pipelined MGET is surprisingly fast
        for moderate sizes.
        """
        # Fetch all vectors in one round-trip
        pipe = self._redis.pipeline()
        for k in entry_keys:
            pipe.hget(k, self._F_VECTOR)
        pipe.hgetall("__sentinel__")  # dummy to avoid empty pipeline issues
        results = await pipe.execute()
        raw_vectors = results[: len(entry_keys)]

        best_score = -1.0
        best_key: Optional[bytes] = None

        for key, raw in zip(entry_keys, raw_vectors):
            if raw is None:
                continue
            vec = _bytes_to_float32(raw)
            score = cosine_similarity(query_vec, vec)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            return None

        # Fetch remaining fields for the winner
        fields = await self._redis.hgetall(best_key)
        return SemanticCacheHit(
            prompt=_b2s(fields.get(self._F_PROMPT, b"")),
            response=_b2s(fields.get(self._F_RESPONSE, b"")),
            similarity=best_score,
            metadata=json_loads(fields.get(self._F_META, b"{}")),
            created_at=float(_b2s(fields.get(self._F_CREATED, b"0"))),
        )

    async def _evict_if_needed(self) -> None:
        """Evict the oldest entry if over *max_entries*."""
        current = await self.count()
        if current <= (self._max_entries or 0):
            return

        # Fetch all keys with their creation timestamps
        keys = await self._get_all_entry_keys()
        pipe = self._redis.pipeline()
        for k in keys:
            pipe.hget(k, self._F_CREATED)
        results = await pipe.execute()

        # Sort by created_at ascending, evict the oldest
        scored = []
        for key, raw_ts in zip(keys, results):
            ts = float(_b2s(raw_ts)) if raw_ts else 0.0
            scored.append((ts, key))
        scored.sort(key=lambda x: x[0])

        to_evict = current - self._max_entries
        victims = [k for _, k in scored[:to_evict]]

        async with self._redis.pipeline() as pipe:
            for k in victims:
                pipe.delete(k)
                pipe.srem(self.index_key, k)
            await pipe.execute()

        self._evictions += len(victims)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _maybe_await(result: Any) -> Any:
    """Await *result* if it's an awaitable, otherwise return it directly."""
    if hasattr(result, "__await__") or isinstance(result, Awaitable):
        return await result
    return result


def _b2s(b: bytes) -> str:
    """Decode bytes to str, tolerating None."""
    if b is None:
        return ""
    return b.decode("utf-8") if isinstance(b, bytes) else str(b)
