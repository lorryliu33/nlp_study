"""
Semantic Router
===============

Intelligent query classification / routing based on semantic similarity.
Define routes with example utterances, then match incoming queries to the
best route in milliseconds — no LLM call needed for the routing decision.

How it works
------------
1. Each **route** is defined with a name and a handful of example utterances
   (e.g. ``"billing" → ["pay my bill", "invoice", "charge on card"]``).
2. Utterance embeddings are averaged into a **centroid vector** per route.
3. At routing time the query is embedded and compared (cosine similarity)
   to every route centroid.
4. The route with the highest similarity ≥ *threshold* wins.

Usage
-----
.. code-block:: python

    import redis.asyncio as redis
    from vl_redis import SemanticRouter

    r = redis.Redis()
    router = SemanticRouter(r, embed_fn=my_embed_fn)

    # Define routes
    await router.add_route("billing", ["pay my bill", "invoice", "refund"])
    await router.add_route("support", ["help me", "broken", "not working"])
    await router.add_route("sales",   ["pricing", "upgrade", "demo"])

    # Route a query
    match = await router.route("I need to pay my invoice")
    print(match.name)   # "billing"
    print(match.score)  # e.g. 0.93

    # Top-k routing
    matches = await router.route("Can you help with billing?", top_k=3)
    for m in matches:
        print(f"{m.name}: {m.score:.3f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import redis.asyncio as redis

from vl_redis._types import EmbeddingFunc
from vl_redis._utils import (
    _bytes_to_float32,
    _float_to_bytes,
    cosine_similarity,
    json_dumps,
    json_loads,
    make_session_key,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class Route:
    """Definition of a semantic route (as returned by ``get_route()``)."""

    name: str
    utterances: List[str]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class RouteMatch:
    """Result of a successful semantic route lookup."""

    name: str
    """The matched route name."""

    score: float
    """Cosine similarity between the query and the route centroid (0–1)."""

    route: Route
    """The full route definition."""

    threshold_met: bool = True
    """Whether *score* ≥ the routing threshold."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class SemanticRouter:
    """Semantic query router backed by Redis.

    Parameters
    ----------
    redis_client : redis.Redis
        Async Redis client.
    embed_fn : EmbeddingFunc
        Embedding function ``(str) -> List[float]``.
    threshold : float
        Minimum cosine similarity for a route match.  Queries whose best
        score falls below this threshold return ``None`` from ``route()``
        (unless *top_k* > 1, which may still return sub-threshold results).
    prefix : str
        Redis key prefix.  Default ``"semroute"``.
    """

    # Redis Hash fields for route storage
    _F_NAME = "name"
    _F_UTTERANCES = "utterances"
    _F_CENTROID = "centroid"
    _F_DESC = "description"
    _F_META = "metadata"
    _F_CREATED = "created_at"
    _F_UPDATED = "updated_at"

    def __init__(
        self,
        redis_client: redis.Redis,
        embed_fn: EmbeddingFunc,
        *,
        threshold: float = 0.75,
        prefix: str = "semroute",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self._redis = redis_client
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._prefix = prefix

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def _routes_key(self) -> str:
        """Redis Set holding all route names."""
        return f"{self._prefix}:routes"

    def _route_key(self, name: str) -> str:
        """Redis Hash key for a single route's data."""
        return make_session_key(self._prefix, f"route:{name}")

    # ------------------------------------------------------------------
    # Add / update routes
    # ------------------------------------------------------------------

    async def add_route(
        self,
        name: str,
        utterances: List[str],
        *,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Route:
        """Register a new route (or overwrite an existing one).

        Parameters
        ----------
        name : str
            Unique route name (e.g. ``"billing"``, ``"support"``).
        utterances : List[str]
            Example phrases that should map to this route.  3–10 diverse
            examples produce good centroids.
        description : str
            Human-readable description of what this route handles.
        metadata : dict | None
            Arbitrary JSON-serialisable metadata.

        Returns
        -------
        Route
            The stored route definition.

        Raises
        ------
        ValueError
            If *utterances* is empty.
        """
        if not utterances:
            raise ValueError(f"Route '{name}': at least one utterance is required")

        # Compute centroid from all utterance embeddings
        embeddings: List[List[float]] = []
        for text in utterances:
            result = self._embed_fn(text)
            emb = await _maybe_await(result)
            embeddings.append(emb)

        centroid = _average_vector(embeddings)
        now = time.time()

        route_data = {
            self._F_NAME: name,
            self._F_UTTERANCES: json_dumps(utterances),
            self._F_CENTROID: _float_to_bytes(centroid),
            self._F_DESC: description,
            self._F_META: json_dumps(metadata or {}),
            self._F_CREATED: str(now),
            self._F_UPDATED: str(now),
        }

        key = self._route_key(name)

        # Check if this is an update (preserve created_at)
        existing_created = await self._redis.hget(key, self._F_CREATED)
        if existing_created is not None:
            route_data[self._F_CREATED] = (
                existing_created.decode("utf-8")
                if isinstance(existing_created, bytes)
                else str(existing_created)
            )

        async with self._redis.pipeline() as pipe:
            pipe.hset(key, mapping=route_data)
            pipe.sadd(self._routes_key, name)
            await pipe.execute()

        return Route(
            name=name,
            utterances=utterances,
            description=description,
            metadata=metadata or {},
            created_at=float(route_data[self._F_CREATED]),
            updated_at=now,
        )

    async def add_routes(
        self,
        routes: List[Dict[str, Any]],
    ) -> List[Route]:
        """Batch-register routes.

        Each dict should have: ``name`` (str), ``utterances`` (List[str]),
        and optionally ``description``, ``metadata``.
        """
        results = []
        for r in routes:
            route = await self.add_route(
                name=r["name"],
                utterances=r["utterances"],
                description=r.get("description", ""),
                metadata=r.get("metadata"),
            )
            results.append(route)
        return results

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(
        self,
        query: str,
        *,
        embedding: Optional[List[float]] = None,
        top_k: int = 1,
    ) -> Union[Optional[RouteMatch], List[RouteMatch]]:
        """Route a query to the best-matching route(s).

        Parameters
        ----------
        query : str
            The query text to classify.
        embedding : List[float] | None
            Pre-computed embedding of *query*.
        top_k : int
            Number of top matches to return.  When ``top_k=1`` (default)
            returns a single ``RouteMatch`` or ``None``.  When ``top_k>1``
            always returns a list (may be empty).

        Returns
        -------
        RouteMatch | None  (when top_k=1)
        List[RouteMatch]   (when top_k>1)
        """
        if embedding is None:
            embedding = await _maybe_await(self._embed_fn(query))

        # Get all route names
        route_names = await self._get_route_names()
        if not route_names:
            return None if top_k == 1 else []

        # Fetch all centroids
        pipe = self._redis.pipeline()
        for name in route_names:
            pipe.hget(self._route_key(name), self._F_CENTROID)
        centroid_results = await pipe.execute()

        # Score every route
        scored: List[tuple[str, float]] = []
        for name, raw in zip(route_names, centroid_results):
            if raw is None:
                continue
            centroid = _bytes_to_float32(raw)
            score = cosine_similarity(embedding, centroid)
            scored.append((name, score))

        if not scored:
            return None if top_k == 1 else []

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k == 1:
            best_name, best_score = scored[0]
            route_def = await self.get_route(best_name)
            if route_def is None:
                return None
            return RouteMatch(
                name=best_name,
                score=best_score,
                route=route_def,
                threshold_met=best_score >= self._threshold,
            )

        # top_k > 1 — build list
        matches: List[RouteMatch] = []
        for name, score in scored[:top_k]:
            route_def = await self.get_route(name)
            if route_def is not None:
                matches.append(
                    RouteMatch(
                        name=name,
                        score=score,
                        route=route_def,
                        threshold_met=score >= self._threshold,
                    )
                )
        return matches

    # ------------------------------------------------------------------
    # Route CRUD
    # ------------------------------------------------------------------

    async def get_route(self, name: str) -> Optional[Route]:
        """Retrieve a single route definition."""
        raw = await self._redis.hgetall(self._route_key(name))
        if not raw:
            return None
        return self._deserialise_route(raw)

    async def list_routes(self) -> List[Route]:
        """Return all registered routes."""
        names = await self._get_route_names()
        routes: List[Route] = []
        for name in names:
            raw = await self._redis.hgetall(self._route_key(name))
            if raw:
                routes.append(self._deserialise_route(raw))
        return routes

    async def remove_route(self, name: str) -> bool:
        """Delete a route.  Returns ``True`` if it existed."""
        key = self._route_key(name)
        existed = await self._redis.exists(key)
        async with self._redis.pipeline() as pipe:
            pipe.delete(key)
            pipe.srem(self._routes_key, name)
            await pipe.execute()
        return bool(existed)

    async def clear(self) -> int:
        """Remove all routes.  Returns the number of routes deleted."""
        names = await self._get_route_names()
        if not names:
            return 0
        async with self._redis.pipeline() as pipe:
            pipe.delete(self._routes_key)
            for name in names:
                pipe.delete(self._route_key(name))
            await pipe.execute()
        return len(names)

    async def count(self) -> int:
        """Return the number of registered routes."""
        return await self._redis.scard(self._routes_key)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_route_names(self) -> List[str]:
        members = await self._redis.smembers(self._routes_key)
        return sorted(
            m.decode("utf-8") if isinstance(m, bytes) else m for m in members
        )

    def _deserialise_route(self, raw: Dict[bytes, bytes]) -> Route:
        """Build a ``Route`` from raw Redis Hash bytes."""

        def _field(field: str, default: str = "") -> str:
            # hgetall returns bytes keys when decode_responses=False,
            # but str keys when decode_responses=True.  Try both.
            val = raw.get(field) or raw.get(field.encode("utf-8"))
            if val is None:
                return default
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)

        return Route(
            name=_field(self._F_NAME, ""),
            utterances=json_loads(_field(self._F_UTTERANCES, "[]")),
            description=_field(self._F_DESC, ""),
            metadata=json_loads(_field(self._F_META, "{}")),
            created_at=float(_field(self._F_CREATED, "0")),
            updated_at=float(_field(self._F_UPDATED, "0")),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _average_vector(vectors: List[List[float]]) -> List[float]:
    """Compute the element-wise mean of multiple equal-length vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


async def _maybe_await(result: Any) -> Any:
    """Await if awaitable, else return directly."""
    if hasattr(result, "__await__") or isinstance(result, Awaitable):
        return await result
    return result
