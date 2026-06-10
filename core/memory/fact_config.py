from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration resolution for legacy atomic fact extraction."""

import json
import logging
from pathlib import Path
from typing import Any

from core.memory.fact_observability import warn_rate_limited

logger = logging.getLogger("animaworks.memory.fact_extraction")


def _resolve_extraction_config(anima_dir: Path) -> tuple[str, dict[str, object], str, int, str]:
    """Resolve extraction model/credential without trusting endpoint fields in status.json."""
    llm_extra: dict[str, object] = {}
    timeout = 30
    status_data: dict[str, Any] = {}
    cfg: Any | None = None

    try:
        from core.config import load_config

        cfg = load_config()
    except Exception:
        cfg = None

    locale = str(getattr(cfg, "locale", "") or _resolve_locale())

    try:
        status_path = Path(anima_dir) / "status.json"
        if status_path.is_file():
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            if status_data.get("extraction_timeout"):
                timeout = int(status_data["extraction_timeout"])
    except Exception:
        warn_rate_limited(
            logger,
            "fact_extraction.status_config",
            "Failed to read status.json for fact extraction",
            exc_info=True,
        )

    consolidation_model = str(getattr(getattr(cfg, "consolidation", None), "llm_model", "") or "")
    consolidation_credential = str(getattr(getattr(cfg, "consolidation", None), "llm_credential", "") or "")
    consolidation_base = consolidation_model.split("/", 1)[-1]

    def credential_for(model: str, explicit: object = "") -> str:
        if isinstance(explicit, str) and explicit:
            return explicit
        model_base = model.split("/", 1)[-1]
        if consolidation_credential and model_base == consolidation_base:
            return consolidation_credential
        return ""

    if status_data.get("extraction_model"):
        model = str(status_data["extraction_model"])
        return (
            model,
            llm_extra,
            locale,
            timeout,
            credential_for(model, status_data.get("extraction_credential")),
        )

    if status_data.get("background_model"):
        model = str(status_data["background_model"])
        credential = credential_for(model, status_data.get("background_credential"))
        if credential:
            return model, llm_extra, locale, timeout, credential

    if consolidation_model:
        return consolidation_model, llm_extra, locale, timeout, consolidation_credential

    try:
        model = cfg.anima_defaults.background_model or cfg.anima_defaults.model
        credential = cfg.anima_defaults.background_credential or cfg.anima_defaults.credential or ""
        return model, llm_extra, locale, timeout, credential
    except Exception:
        return "claude-sonnet-4-6", llm_extra, "ja", timeout, ""


def _resolve_locale() -> str:
    try:
        from core.config.models import load_config

        return str(load_config().locale or "ja")
    except Exception:
        return "ja"
