# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of AnimaWorks core/server, licensed under AGPL-3.0.
# See LICENSES/AGPL-3.0.txt for the full license text.

"""Central configuration module for AnimaWorks.

Defines Pydantic models for the unified config.json and provides
load / save / resolve helpers with a module-level singleton cache.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("animaworks.config")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GatewaySystemConfig(BaseModel):
    """Deprecated: Gateway configuration retained for config.json compatibility."""

    host: str = "0.0.0.0"
    port: int = 18500
    redis_url: str | None = None
    worker_heartbeat_timeout: int = 45


class WorkerSystemConfig(BaseModel):
    """Deprecated: Worker configuration retained for config.json compatibility."""

    gateway_url: str = "http://localhost:18500"
    redis_url: str | None = None
    listen_port: int = 18501
    heartbeat_interval: int = 15


class SystemConfig(BaseModel):
    mode: str = "server"
    log_level: str = "INFO"
    gateway: GatewaySystemConfig = GatewaySystemConfig()
    worker: WorkerSystemConfig = WorkerSystemConfig()


class CredentialConfig(BaseModel):
    api_key: str = ""
    base_url: str | None = None


class PersonModelConfig(BaseModel):
    """Per-person overrides.  All fields are optional (None = use default)."""

    model: str | None = None
    fallback_model: str | None = None
    max_tokens: int | None = None
    max_turns: int | None = None
    credential: str | None = None
    context_threshold: float | None = None
    max_chains: int | None = None
    conversation_history_threshold: float | None = None
    execution_mode: str | None = None  # "autonomous" or "assisted"
    supervisor: str | None = None  # name of supervisor Person
    speciality: str | None = None  # free-text specialisation


class PersonDefaults(BaseModel):
    """Concrete defaults applied when a per-person field is None."""

    model: str = "claude-sonnet-4-20250514"
    fallback_model: str | None = None
    max_tokens: int = 4096
    max_turns: int = 20
    credential: str = "anthropic"
    context_threshold: float = 0.50
    max_chains: int = 2
    conversation_history_threshold: float = 0.30
    execution_mode: str | None = None  # None = auto-detect from model
    supervisor: str | None = None
    speciality: str | None = None


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""

    enabled: bool = True
    embedding_model: str = "intfloat/multilingual-e5-small"
    use_gpu: bool = False
    enable_spreading_activation: bool = True
    max_graph_hops: int = 2
    enable_file_watcher: bool = True


class PrimingConfig(BaseModel):
    """Configuration for priming layer (automatic memory retrieval)."""

    dynamic_budget: bool = True
    budget_greeting: int = 500
    budget_question: int = 1500
    budget_request: int = 3000
    budget_heartbeat: int = 200


class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation processes."""

    daily_enabled: bool = True
    daily_time: str = "02:00"  # Format: HH:MM
    min_episodes_threshold: int = 1
    llm_model: str = "anthropic/claude-sonnet-4-20250514"
    weekly_enabled: bool = True  # Phase 3 implementation
    weekly_time: str = "sun:03:00"  # Format: day:HH:MM
    duplicate_threshold: float = 0.85  # Similarity threshold for duplicate detection
    episode_retention_days: int = 30  # Days to retain uncompressed episodes


class AnimaWorksConfig(BaseModel):
    version: int = 1
    system: SystemConfig = SystemConfig()
    credentials: dict[str, CredentialConfig] = {"anthropic": CredentialConfig()}
    model_modes: dict[str, str] = {}  # モデル名 → "A1"/"A2"/"B"
    person_defaults: PersonDefaults = PersonDefaults()
    persons: dict[str, PersonModelConfig] = {}
    consolidation: ConsolidationConfig = ConsolidationConfig()
    rag: RAGConfig = RAGConfig()
    priming: PrimingConfig = PrimingConfig()


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: AnimaWorksConfig | None = None
_config_path: Path | None = None


def invalidate_cache() -> None:
    """Reset the module-level singleton cache."""
    global _config, _config_path
    _config = None
    _config_path = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_config_path(data_dir: Path | None = None) -> Path:
    """Return the path to config.json inside *data_dir*.

    If *data_dir* is not given, it is resolved via ``core.paths.get_data_dir``
    (imported lazily to avoid circular imports).
    """
    if data_dir is None:
        from core.paths import get_data_dir

        data_dir = get_data_dir()
    return data_dir / "config.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AnimaWorksConfig:
    """Load configuration from disk, returning cached instance when possible.

    If *path* is ``None``, :func:`get_config_path` determines the location.
    When the file does not exist the default configuration is returned.
    """
    global _config, _config_path

    if path is None:
        path = get_config_path()

    # Return cached instance if same path was already loaded.
    if _config is not None and _config_path == path:
        return _config

    if path.is_file():
        logger.debug("Loading config from %s", path)
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw_text)
            config = AnimaWorksConfig.model_validate(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            raise
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", path, exc)
            raise
    else:
        logger.info("Config file not found at %s; using defaults", path)
        config = AnimaWorksConfig()

    _config = config
    _config_path = path
    return config


def save_config(config: AnimaWorksConfig, path: Path | None = None) -> None:
    """Persist *config* to disk as pretty-printed JSON (mode 0o600).

    Updates the module-level singleton cache so subsequent :func:`load_config`
    calls return the freshly saved config.
    """
    global _config, _config_path

    if path is None:
        path = get_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = config.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")

    # Restrict permissions — the file may contain API keys.
    os.chmod(path, 0o600)

    logger.debug("Config saved to %s", path)

    _config = config
    _config_path = path


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_person_config(
    config: AnimaWorksConfig,
    person_name: str,
) -> tuple[PersonDefaults, CredentialConfig]:
    """Merge per-person overrides with *person_defaults* and resolve the credential.

    For each field in :class:`PersonModelConfig`, the person's value is used
    when it is not ``None``; otherwise the corresponding default is used.

    Returns:
        A ``(resolved_defaults, credential)`` tuple.

    Raises:
        KeyError: If the resolved credential name is not in
            ``config.credentials``.
    """
    person = config.persons.get(person_name, PersonModelConfig())
    defaults = config.person_defaults

    # Build a dict with resolved values: person override wins when not None.
    resolved: dict[str, Any] = {}
    for field_name in PersonModelConfig.model_fields:
        person_value = getattr(person, field_name)
        resolved[field_name] = (
            person_value if person_value is not None else getattr(defaults, field_name)
        )

    resolved_defaults = PersonDefaults.model_validate(resolved)

    credential_name = resolved_defaults.credential
    if credential_name not in config.credentials:
        raise KeyError(
            f"Credential '{credential_name}' (for person '{person_name}') "
            f"not found in config.credentials"
        )

    credential = config.credentials[credential_name]
    return resolved_defaults, credential


# ---------------------------------------------------------------------------
# Model mode resolution
# ---------------------------------------------------------------------------

# Default model_modes (fallback when config.json has no entry)
DEFAULT_MODEL_MODES: dict[str, str] = {
    "claude-opus-4-20250514": "A1",
    "claude-sonnet-4-20250514": "A1",
    "claude-haiku-3.5-20241022": "A1",
    "openai/gpt-4o": "A2",
    "openai/gpt-4o-mini": "A2",
    "google/gemini-2.0-flash": "A2",
    "google/gemini-2.5-pro": "A2",
    "ollama/gemma3:27b": "B",
    "ollama/llama3.3:70b": "B",
    "ollama/qwen2.5-coder:32b": "B",
}


def resolve_execution_mode(
    config: AnimaWorksConfig,
    model_name: str,
    explicit_override: str | None = None,
) -> str:
    """Resolve execution mode from model name.

    Priority:
      1. Person's execution_mode explicit override (legacy)
      2. config.json model_modes table
      3. DEFAULT_MODEL_MODES (hard-coded fallback)
      4. Default "B" (safe side)
    """
    if explicit_override:
        mapping = {"autonomous": "A2", "assisted": "B"}
        return mapping.get(explicit_override, explicit_override.upper())

    table = config.model_modes or {}
    if model_name in table:
        return table[model_name].upper()
    if model_name in DEFAULT_MODEL_MODES:
        return DEFAULT_MODEL_MODES[model_name]
    return "B"  # unknown model → safe side