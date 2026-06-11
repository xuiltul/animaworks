from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""LLM classifier helper for legacy fact reconciliation."""

import json
import logging
from pathlib import Path
from typing import Any

from core.memory.fact_config import DEFAULT_FACT_EXTRACTION_TIMEOUT_SECONDS, _coerce_timeout_seconds
from core.memory.facts import FactRecord

logger = logging.getLogger("animaworks.memory.fact_invalidation_llm")


def classify_fact_relation(new_fact: FactRecord, candidates: list[Any], anima_dir: Path) -> str:
    import litellm

    model, llm_extra, timeout = _resolve_reconcile_llm_config(anima_dir)
    from core.memory._llm_utils import get_memory_llm_kwargs_for_model

    llm_kwargs = get_memory_llm_kwargs_for_model(model, llm_extra)
    resolved_model = llm_kwargs.pop("model", model)
    effective_timeout = llm_kwargs.pop("timeout", timeout)

    response = litellm.completion(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(new_fact, candidates)},
        ],
        temperature=0.0,
        max_tokens=16,
        timeout=effective_timeout,
        **llm_kwargs,
    )
    return response.choices[0].message.content or ""


def _user_prompt(new_fact: FactRecord, candidates: list[Any]) -> str:
    candidates_json = json.dumps(
        [
            {
                "fact_id": candidate.record.fact_id,
                "text": candidate.record.text,
                "source_entity": candidate.record.source_entity,
                "target_entity": candidate.record.target_entity,
                "edge_type": candidate.record.edge_type,
                "valid_at": candidate.record.valid_at,
                "recorded_at": candidate.record.recorded_at,
                "valid_until": candidate.record.valid_until,
                "similarity": round(candidate.score, 4),
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
    )
    return (
        "New fact JSON:\n"
        f"{json.dumps(new_fact.to_dict(), ensure_ascii=False)}\n\n"
        "Existing active candidate facts JSON:\n"
        f"{candidates_json}\n\n"
        "Definitions:\n"
        "- DUPLICATE: same meaning, no new durable information.\n"
        "- CONTRADICT: cannot both be true for the same time period.\n"
        "- COMPLEMENT: compatible additional detail should be merged into the existing fact.\n"
        "- ADD: distinct fact that should be appended.\n\n"
        "Label:"
    )


def _resolve_reconcile_llm_config(anima_dir: Path) -> tuple[str, dict[str, object], int]:
    timeout = DEFAULT_FACT_EXTRACTION_TIMEOUT_SECONDS
    llm_extra: dict[str, object] = {}
    try:
        from core.config import load_config

        cfg = load_config()
        timeout = _coerce_timeout_seconds(
            getattr(getattr(cfg, "rag", None), "fact_extraction_timeout_seconds", None),
            timeout,
        )
    except Exception:
        logger.debug("Failed to load fact reconciliation timeout from config", exc_info=True)

    try:
        status_path = anima_dir / "status.json"
        if status_path.is_file():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            if data.get("extraction_timeout"):
                timeout = _coerce_timeout_seconds(data["extraction_timeout"], timeout)
            if data.get("background_model"):
                return str(data["background_model"]), llm_extra, timeout
            if data.get("extraction_model"):
                return str(data["extraction_model"]), llm_extra, timeout
    except Exception:
        logger.debug("Failed to resolve reconcile LLM config from status.json", exc_info=True)

    try:
        from core.config.models import load_config

        cfg = load_config()
        return cfg.anima_defaults.background_model or cfg.anima_defaults.model, llm_extra, timeout
    except Exception:
        return "claude-sonnet-4-6", llm_extra, timeout


_SYSTEM_PROMPT = (
    "You classify whether a new memory fact should be reconciled with existing active facts. "
    "Return exactly one label and no other text: CONTRADICT, COMPLEMENT, DUPLICATE, or ADD."
)
