"""
Embeddings Cache
================

Cache embedding vectors in Redis to avoid recomputing the same text
across runs, providers, and models.

Key design
----------
Every cached embedding is keyed by ``{prefix}:{model}:{sha256_truncated}``.
This means:

* The same text embedded with *different* models never collides.
* Raw user text never appears in Redis key names (privacy by design).
* Lookups are O(1) deterministic — no vector search needed.

Usage
-----
.. code-block:: python

    import redis.asyncio as redis
    from vl_redis import EmbeddingsCache

    r = redis.Redis()
    cache = EmbeddingsCache(r)

    # Low-level get / set
    vec = await cache.get("hello world", model="text-embedding-3-small")
    if vec is None:
        vec = await openai_embed("hello world")
        await cache.set("hello world", vec, model="text-embedding-3-small")

    # High-level get-or-compute
    vec = await cache.get_or_compute(
        "hello world",
        embed_fn=openai_embed,
        model="text-embedding-3-small",
    )
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import redis.asyncio as redis

from vl_redis._types import EmbeddingFunc
from vl_redis._utils import _float_to_bytes, _bytes_to_float32, json_dumps, json_loads, make_cache_key


class EmbeddingsCache:
    """LRU-style embedding cache backed by Redis.

    Parameters
    ----------
    redis_client : redis.Redis
        An async Redis client (``redis.asyncio.Redis``).
    prefix : str
        Namespace prefix for Redis keys.  Default ``"emb"``.
    default_ttl : int | None
        Default TTL in seconds for cached embeddings.  ``None`` means no
        expiry.  Can be overridden per-``set()`` call.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        prefix: str = "emb",
        default_ttl: Optional[int] = None,
    ) -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._default_ttl = default_ttl

        # Lightweight client-side stats (best-effort; may drift under concurrency)
        self._hits = 0
        self._misses = 0
        self._sets = 0

    # ------------------------------------------------------------------
    # Public API — single key
    # ------------------------------------------------------------------

    async def get(
        self,
        text: str,
        *,
        model: str = "default",
    ) -> Optional[List[float]]:
        """Retrieve a cached embedding vector.

        Returns ``None`` when the key is absent or expired.

        Parameters
        ----------
        text : str
            The text whose embedding was cached.
        model : str
            Embedding model identifier.  Must match the value used at
            ``set()`` time.

        Returns
        -------
        List[float] | None
            The embedding vector, or ``None`` on cache miss.
        """
        key = make_cache_key(self._prefix, text, model)
        raw = await self._redis.get(key)

        if raw is None:
            self._misses += 1
            return None

        self._hits += 1
        return _bytes_to_float32(raw)

    async def set(
        self,
        text: str,
        vector: List[float],
        *,
        model: str = "default",
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an embedding vector in the cache.

        Parameters
        ----------
        text : str
            The original text.
        vector : List[float]
            The embedding vector to cache.
        model : str
            Embedding model identifier.
        ttl : int | None
            TTL in seconds.  Falls back to *default_ttl* then to no expiry.
        metadata : dict | None
            Optional JSON-serialisable metadata stored alongside the vector.
        """
        key = make_cache_key(self._prefix, text, model)
        ttl = ttl if ttl is not None else self._default_ttl

        packed = _float_to_bytes(vector)

        async with self._redis.pipeline() as pipe:
            pipe.set(key, packed)
            if metadata is not None:
                pipe.set(f"{key}:meta", json_dumps(metadata))
                if ttl is not None:
                    pipe.expire(f"{key}:meta", ttl)
            if ttl is not None:
                pipe.expire(key, ttl)
            await pipe.execute()

        self._sets += 1

    async def get_or_compute(
        self,
        text: str,
        embed_fn: EmbeddingFunc,
        *,
        model: str = "default",
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Return cached embedding or compute + cache on miss.

        This is the recommended high-level API — it eliminates the
        get-if-none-set dance.

        Parameters
        ----------
        text : str
            The text to embed.
        embed_fn : EmbeddingFunc
            Sync or async callable ``(str) -> List[float]``.
        model : str
            Embedding model identifier.
        ttl : int | None
            TTL for the cached entry (only used on cache miss).
        metadata : dict | None
            Arbitrary metadata to store alongside the vector.

        Returns
        -------
        List[float]
            The embedding vector (from cache or freshly computed).
        """
        cached = await self.get(text, model=model)
        if cached is not None:
            return cached

        # Compute — support both sync and async callables
        result = embed_fn(text)
        if _is_awaitable(result):
            vector = await result
        else:
            vector = result  # type: ignore[assignment]

        await self.set(text, vector, model=model, ttl=ttl, metadata=metadata)
        return vector

    async def get_metadata(
        self,
        text: str,
        *,
        model: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the metadata stored alongside an embedding."""
        key = make_cache_key(self._prefix, text, model)
        raw = await self._redis.get(f"{key}:meta")
        if raw is None:
            return None
        return json_loads(raw)

    async def delete(self, text: str, *, model: str = "default") -> bool:
        """Remove a cached embedding (and its metadata).

        Returns ``True`` if the key existed, ``False`` otherwise.
        """
        key = make_cache_key(self._prefix, text, model)
        removed = await self._redis.delete(key, f"{key}:meta")
        return bool(removed)

    async def exists(self, text: str, *, model: str = "default") -> bool:
        """Check whether an embedding is cached (and not expired)."""
        key = make_cache_key(self._prefix, text, model)
        return bool(await self._redis.exists(key))

    # ------------------------------------------------------------------
    # Public API — batch
    # ------------------------------------------------------------------

    async def get_many(
        self,
        texts: List[str],
        *,
        model: str = "default",
    ) -> Dict[str, Optional[List[float]]]:
        """Batched retrieval — pipelines GET for multiple texts.

        Returns a ``{text: vector | None}`` mapping.
        """
        keys = [make_cache_key(self._prefix, t, model) for t in texts]
        values: List[Optional[bytes]] = await self._redis.mget(keys)

        result: Dict[str, Optional[List[float]]] = {}
        for text, raw in zip(texts, values):
            if raw is None:
                result[text] = None
                self._misses += 1
            else:
                result[text] = _bytes_to_float32(raw)
                self._hits += 1
        return result

    async def set_many(
        self,
        mapping: Dict[str, List[float]],
        *,
        model: str = "default",
        ttl: Optional[int] = None,
    ) -> None:
        """Store many text→vector pairs in one pipeline."""
        ttl = ttl if ttl is not None else self._default_ttl

        async with self._redis.pipeline() as pipe:
            for text, vector in mapping.items():
                key = make_cache_key(self._prefix, text, model)
                pipe.set(key, _float_to_bytes(vector))
                if ttl is not None:
                    pipe.expire(key, ttl)
            await pipe.execute()

        self._sets += len(mapping)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def clear(self, *, model: Optional[str] = None) -> int:
        """Delete all cached embeddings, optionally scoped to *model*.

        Returns the number of keys removed.
        """
        pattern = f"{self._prefix}:*" if model is None else f"{self._prefix}:{model}:*"
        removed = 0

        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=500)
            if keys:
                removed += await self._redis.delete(*keys)
            if cursor == 0:
                break

        return removed

    async def count(self, *, model: Optional[str] = None) -> int:
        """Return the approximate number of cached embeddings."""
        pattern = f"{self._prefix}:*" if model is None else f"{self._prefix}:{model}:*"
        count = 0
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern, count=500)
            # Filter out metadata keys
            count += sum(1 for k in keys if not str(k, "utf-8").endswith(":meta"))
            if cursor == 0:
                break
        return count

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
            "sets": self._sets,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Zero out the client-side stat counters."""
        self._hits = 0
        self._misses = 0
        self._sets = 0


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _is_awaitable(obj: Any) -> bool:
    """Return ``True`` if *obj* is an awaitable."""
    return hasattr(obj, "__await__") or isinstance(obj, Awaitable)
