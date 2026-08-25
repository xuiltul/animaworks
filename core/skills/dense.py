from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding-based (dense) scoring for the skill router.

Provides :func:`skill_dense_scores` which returns per-skill cosine similarity
between the embedding of *query* and each skill's stable embedding text
(name + description + routing hints).  Skill embeddings are cached in a
process-local dict and on disk under the shared data dir so that repeated
calls only recompute the query embedding.  Failures (missing model,
timeout, etc.) are swallowed so prompt construction is never blocked.
"""

import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Sequence

from core.memory.rag.singleton import generate_embeddings, get_embedding_model_name
from core.paths import get_shared_dir
from core.skills.models import SkillMetadata

logger = logging.getLogger(__name__)

#: Minimum cosine similarity for a skill to influence routing.  Lower scores
#: are treated as noise and zeroed so the router's exclusion rule
#: (deterministic==0 and dense==0 and lexical<3) is not silently bypassed.
DENSE_MIN_SIM = 0.45
#: Number of top-similarity skills that receive a dense bonus (rank-based).
DENSE_TOP_K = 5

#: Maximum number of distinct query embeddings kept in the process LRU.
_QUERY_LRU_SIZE = 256

_CACHE_FILENAME = "skill_embed_cache.json"

#: Process-local disk-cache overlay.  {sha256(text): [floats]}
_PROCESS_CACHE: dict[str, list[float]] = {}

#: Process-local LRU of completed query → embedding.  {query: [floats]}
_QUERY_LRU: OrderedDict[str, list[float]] = OrderedDict()

_EMBED_LOCK = threading.Lock()


def _embedding_text(meta: SkillMetadata) -> str:
    """Build the stable text embedded for *meta* (short, no full body)."""
    parts: list[str] = [meta.name, meta.description]
    for values in (
        meta.tags,
        meta.trigger_phrases,
        meta.use_when,
        meta.routing.trigger_phrases,
        meta.routing.use_when,
        meta.routing.domains,
        meta.domains,
        meta.routing.routing_examples,
        meta.routing_examples,
    ):
        for value in values:
            if value:
                parts.append(str(value))
    return "\n".join(part for part in parts if part)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_name() -> str:
    try:
        return get_embedding_model_name() or "default"
    except Exception:
        return "default"


def _load_disk_cache(model: str) -> dict[str, list[float]]:
    try:
        path = get_shared_dir() / _CACHE_FILENAME
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("model") == model:
            entries = raw.get("entries") or {}
            if isinstance(entries, dict):
                return {k: list(v) for k, v in entries.items() if v}
    except Exception:
        logger.debug("Failed to load skill embedding cache", exc_info=True)
    return {}


def _write_disk_cache(model: str, entries: dict[str, list[float]]) -> None:
    if not entries:
        return
    try:
        path = get_shared_dir() / _CACHE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"model": model, "entries": entries}, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write skill embed cache", exc_info=True)


def _query_embedding(query: str) -> list[float] | None:
    global _QUERY_LRU
    cached = _QUERY_LRU.get(query)
    if cached is not None:
        _QUERY_LRU.move_to_end(query)
        return cached
    with _EMBED_LOCK:
        # Re-check under lock to avoid duplicate work under contention.
        cached = _QUERY_LRU.get(query)
        if cached is not None:
            _QUERY_LRU.move_to_end(query)
            return cached
        results = generate_embeddings([query], purpose="query")
        if not results or not results[0]:
            return None
        vec = list(results[0])
        _QUERY_LRU[query] = vec
        while len(_QUERY_LRU) > _QUERY_LRU_SIZE:
            _QUERY_LRU.popitem(last=False)
        return vec


def _document_embeddings(
    model: str,
    texts: Sequence[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return (updated process cache, disk cache) for uncached *texts*."""
    disk = _load_disk_cache(model)
    missing: list[str] = []
    for text in texts:
        key = _text_key(text)
        if key not in _PROCESS_CACHE and key not in disk:
            missing.append(text)
    if not missing:
        return _PROCESS_CACHE, disk
    results = generate_embeddings(missing, purpose="document")
    for text, vec in zip(missing, results, strict=False):
        if not vec:
            continue
        key = _text_key(text)
        _PROCESS_CACHE[key] = list(vec)
        disk[key] = list(vec)
    return _PROCESS_CACHE, disk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _safe_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        return _cosine(a, b)
    except Exception:
        return 0.0


def skill_dense_scores(
    query: str,
    skills: Sequence[SkillMetadata],
    *,
    include_body: bool = True,
) -> dict[str, float]:
    """Return ``{str(meta.path) or name: score}`` of dense similarity.

    Scores below :data:`DENSE_MIN_SIM` are zeroed.  Any failure during
    embedding generation returns an empty dict so prompt construction is
    never blocked.
    """
    if not query or not skills:
        return {}
    try:
        query_vec = _query_embedding(query)
        if not query_vec:
            return {}
        model = _model_name()
        texts = [_embedding_text(meta) for meta in skills]
        with _EMBED_LOCK:
            process_cache, disk_cache = _document_embeddings(model, texts)
        sims: dict[str, float] = {}
        for meta in skills:
            key = _text_key(_embedding_text(meta))
            vec = process_cache.get(key) or disk_cache.get(key)
            if not vec:
                continue
            cos = max(0.0, _safe_cosine(query_vec, vec))
            if cos < DENSE_MIN_SIM:
                continue
            out_key = str(meta.path) if meta.path is not None else meta.name
            sims[out_key] = cos
        _write_disk_cache(model, disk_cache)
        # Rank-based output: ruri-v3 cosines are compressed (~0.80-0.88 for
        # everything), so the absolute value barely discriminates. Only the
        # top DENSE_TOP_K by similarity get a bonus fraction 1.0, 0.8, ... 0.2;
        # the router multiplies by dense_weight.
        top = sorted(sims.items(), key=lambda kv: -kv[1])[:DENSE_TOP_K]
        return {key: (DENSE_TOP_K - i) / DENSE_TOP_K for i, (key, _cos) in enumerate(top)}
    except Exception:
        logger.debug("skill_dense_scores failed", exc_info=True)
        return {}


__all__ = ["skill_dense_scores", "DENSE_MIN_SIM", "DENSE_TOP_K"]
