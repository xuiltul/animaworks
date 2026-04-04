# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Helpers for local Ollama-backed model defaults and role presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.schemas import (
    DEFAULT_LOCAL_LLM_BASE_URL,
    DEFAULT_LOCAL_LLM_PRESETS,
    LocalLLMConfig,
)


def normalize_ollama_base_url(base_url: str | None) -> str:
    normalized = (base_url or DEFAULT_LOCAL_LLM_BASE_URL).strip() or DEFAULT_LOCAL_LLM_BASE_URL
    normalized = normalized.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


def normalize_ollama_model_name(model: str) -> str:
    normalized = model.strip()
    if not normalized:
        return ""
    if normalized.startswith("ollama/"):
        return normalized
    return f"ollama/{normalized}"


def resolve_local_llm_role_preset(local_llm: LocalLLMConfig, role: str | None) -> str:
    role_name = (role or "general").strip() or "general"
    return local_llm.role_presets.get(role_name, local_llm.role_presets.get("general", "general"))


def resolve_local_llm_role_model(local_llm: LocalLLMConfig, role: str | None) -> str:
    preset_name = resolve_local_llm_role_preset(local_llm, role)
    model = local_llm.presets.get(preset_name) or DEFAULT_LOCAL_LLM_PRESETS[preset_name]
    return normalize_ollama_model_name(model)


def is_local_llm_default(config: Any) -> bool:
    return getattr(config.anima_defaults, "credential", None) == "ollama"


def apply_local_llm_role_to_status(status_data: dict[str, Any], config: Any, role: str | None) -> bool:
    """Apply the configured local LLM model/credential for *role* to status data."""
    if not is_local_llm_default(config):
        return False
    status_data["model"] = resolve_local_llm_role_model(config.local_llm, role)
    status_data["credential"] = "ollama"
    return True


def apply_local_llm_presets_to_animas(animas_dir: Path, config: Any) -> list[str]:
    """Apply role-based local LLM model settings to all animas with status.json.

    Guarded by ``config.local_llm.auto_apply_presets`` (default False).
    """
    updated: list[str] = []
    if not getattr(config.local_llm, "auto_apply_presets", False):
        return updated
    if not animas_dir.exists():
        return updated

    for anima_dir in sorted(path for path in animas_dir.iterdir() if path.is_dir()):
        status_path = anima_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        role = status_data.get("role", "general")
        if apply_local_llm_role_to_status(status_data, config, role):
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(status_path)
            updated.append(anima_dir.name)

    return updated


__all__ = [
    "apply_local_llm_presets_to_animas",
    "apply_local_llm_role_to_status",
    "is_local_llm_default",
    "normalize_ollama_base_url",
    "normalize_ollama_model_name",
    "resolve_local_llm_role_model",
    "resolve_local_llm_role_preset",
]
