"""Shared causal helpers for serial-arrival Lifecycle-LRU."""

from __future__ import annotations

import math
from typing import Any


ONLINE_SOURCES = {"runtime_explicit", "controlled_protocol"}


def validate_online_source(source: str) -> None:
    if source not in ONLINE_SOURCES:
        raise ValueError(
            f"online lifecycle policy rejects source={source!r}; "
            f"allowed sources are {sorted(ONLINE_SOURCES)}"
        )


def update_session_metadata(
    metadata: dict[str, dict[str, float | int | None]],
    session_id: str,
    request_index: int,
    cached_tokens: int,
) -> None:
    previous = metadata.get(session_id)
    if previous is None:
        ema_interval = None
    else:
        observed = request_index - int(previous["last_access"])
        old_ema = previous["ema_interval"]
        ema_interval = (
            float(observed)
            if old_ema is None
            else 0.5 * float(old_ema) + 0.5 * observed
        )
    metadata[session_id] = {
        "last_access": request_index,
        "ema_interval": ema_interval,
        "cached_tokens": int(cached_tokens),
    }


def choose_serial_lifecycle_victim(
    cache_lru_order: list[str],
    metadata: dict[str, dict[str, Any]],
    request_index: int,
    session_count: int,
) -> tuple[str, dict[str, float]]:
    """Choose the lowest-value WAITING Session using causal history only."""
    if not cache_lru_order:
        raise ValueError("cannot choose a victim from an empty cache")
    max_cached = max(float(metadata[item]["cached_tokens"]) for item in cache_lru_order)
    scores: dict[str, float] = {}
    for resident in cache_lru_order:
        details = metadata[resident]
        idle = request_index - int(details["last_access"])
        ema = details["ema_interval"]
        interval_scale = float(ema) if ema is not None else session_count
        recency = math.exp(-idle / max(interval_scale, 1.0))
        cached_value = float(details["cached_tokens"]) / max_cached
        scores[resident] = recency + cached_value
    victim = min(
        cache_lru_order,
        key=lambda resident: (scores[resident], cache_lru_order.index(resident)),
    )
    return victim, scores
