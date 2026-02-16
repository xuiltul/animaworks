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

import fnmatch
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
    type: str = "api_key"
    api_key: str = ""
    keys: dict[str, str] = {}
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
    graph_cache_enabled: bool = True
    implicit_link_threshold: float = 0.75
    spreading_memory_types: list[str] = ["knowledge", "episodes"]


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


class ImageGenConfig(BaseModel):
    """Configuration for image generation and style consistency."""

    style_reference: str | None = None  # Path to organization-wide style reference image
    style_prefix: str = ""  # Common style tags prepended to character prompt
    style_suffix: str = ""  # Common style tags appended to character prompt
    negative_prompt_extra: str = ""  # Extra tags added to negative prompt
    vibe_strength: float = 0.6  # Vibe Transfer strength (0.0-1.0)
    vibe_info_extracted: float = 0.8  # Vibe Transfer information extraction (0.0-1.0)


class NotificationChannelConfig(BaseModel):
    """Configuration for a single human notification channel."""

    type: str  # "slack", "line", "telegram", "chatwork", "ntfy"
    enabled: bool = True
    config: dict[str, Any] = {}


class HumanNotificationConfig(BaseModel):
    """Global configuration for human notification from top-level Persons."""

    enabled: bool = False
    channels: list[NotificationChannelConfig] = []


class ServerConfig(BaseModel):
    """Server runtime configuration."""

    ipc_stream_timeout: int = 300  # seconds (default raised from 120)
    stream_checkpoint_enabled: bool = True  # save tool results during streaming
    stream_retry_max: int = 3  # max automatic retries on stream disconnect
    stream_retry_delay_s: float = 5.0  # delay between retries (seconds)


class BackgroundToolConfig(BaseModel):
    """Per-tool background execution threshold."""

    threshold_s: int = 30


class BackgroundTaskConfig(BaseModel):
    """Configuration for background tool execution."""

    enabled: bool = True
    eligible_tools: dict[str, BackgroundToolConfig] = {
        "image_generation": BackgroundToolConfig(threshold_s=30),
        "local_llm": BackgroundToolConfig(threshold_s=60),
        "run_command": BackgroundToolConfig(threshold_s=60),
    }
    result_retention_hours: int = 24


class AnimaWorksConfig(BaseModel):
    version: int = 1
    setup_complete: bool = False
    locale: str = "ja"
    system: SystemConfig = SystemConfig()
    credentials: dict[str, CredentialConfig] = {"anthropic": CredentialConfig()}
    model_modes: dict[str, str] = {}  # モデル名 → "A1"/"A2"/"B"
    person_defaults: PersonDefaults = PersonDefaults()
    persons: dict[str, PersonModelConfig] = {}
    consolidation: ConsolidationConfig = ConsolidationConfig()
    rag: RAGConfig = RAGConfig()
    priming: PrimingConfig = PrimingConfig()
    image_gen: ImageGenConfig = ImageGenConfig()
    human_notification: HumanNotificationConfig = HumanNotificationConfig()
    server: ServerConfig = ServerConfig()
    background_task: BackgroundTaskConfig = BackgroundTaskConfig()


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

# Default model_modes with wildcard pattern support.
# Patterns use fnmatch syntax (*, ?, [seq]).
# Order matters for specificity — more specific patterns should appear first,
# but the resolver sorts by specificity automatically.
DEFAULT_MODEL_MODE_PATTERNS: dict[str, str] = {
    # ── A1: Claude Agent SDK ──────────────────────────────
    "claude-*": "A1",

    # ── A2: Cloud API providers (LiteLLM + tool_use) ──────
    "openai/*": "A2",
    "azure/*": "A2",
    "google/*": "A2",
    "mistral/*": "A2",
    "xai/*": "A2",
    "cohere/*": "A2",
    "zai/*": "A2",
    "minimax/*": "A2",
    "moonshot/*": "A2",
    "deepseek/deepseek-chat": "A2",

    # ── A2: Ollama models with reliable tool_use ──────────
    "ollama/qwen3:14b": "A2",
    "ollama/qwen3:30b": "A2",
    "ollama/qwen3:32b": "A2",
    "ollama/qwen3:235b": "A2",
    "ollama/qwen3-coder:*": "A2",
    "ollama/llama4:*": "A2",
    "ollama/mistral-small3.2:*": "A2",
    "ollama/devstral*": "A2",
    "ollama/glm-4.7*": "A2",
    "ollama/glm-5*": "A2",
    "ollama/minimax*": "A2",
    "ollama/kimi-k2*": "A2",
    "ollama/gpt-oss*": "A2",

    # ── B: No reliable tool_use ───────────────────────────
    "ollama/qwen3:0.6b": "B",
    "ollama/qwen3:1.7b": "B",
    "ollama/qwen3:4b": "B",
    "ollama/qwen3:8b": "B",
    "ollama/gemma3*": "B",
    "ollama/deepseek-r1*": "B",
    "ollama/deepseek-v3*": "B",
    "ollama/phi4*": "B",
    "ollama/*": "B",
}

# Backward-compatible alias
DEFAULT_MODEL_MODES = DEFAULT_MODEL_MODE_PATTERNS


def _pattern_specificity(pattern: str) -> tuple[int, int, int]:
    """Return a sort key so more-specific patterns match first.

    Ranking (lower = matched first):
      - Exact match (no wildcard chars): (0, 0, -len)
      - Wildcard pattern: (1, -prefix_len, -total_len)
        where prefix_len is the length before the first wildcard char.
    """
    if not any(c in pattern for c in ("*", "?", "[")):
        # Exact match — highest priority
        return (0, 0, -len(pattern))
    # Find prefix length before first wildcard character
    prefix_len = len(pattern)
    for i, ch in enumerate(pattern):
        if ch in ("*", "?", "["):
            prefix_len = i
            break
    return (1, -prefix_len, -len(pattern))


def _match_pattern_table(
    model_name: str,
    table: dict[str, str],
) -> str | None:
    """Match *model_name* against a pattern table.

    Phase 1: O(1) exact dict lookup.
    Phase 2: fnmatch scan in specificity-descending order.

    Returns the mode string (e.g. ``"A1"``) or ``None`` if no match.
    """
    # Phase 1: exact match
    if model_name in table:
        return table[model_name].upper()

    # Phase 2: wildcard patterns sorted by specificity
    wildcard_patterns = [
        p for p in table
        if any(c in p for c in ("*", "?", "["))
    ]
    wildcard_patterns.sort(key=_pattern_specificity)

    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(model_name, pattern):
            return table[pattern].upper()

    return None


def load_model_config(person_dir: Path) -> "ModelConfig":
    """Build a ModelConfig for *person_dir* from the unified config.json.

    This is a standalone version of ``MemoryManager.read_model_config()``
    for use in server routes that do not have a live DigitalPerson instance.
    """
    from core.schemas import ModelConfig

    config_path = get_config_path()
    if not config_path.exists():
        return ModelConfig()

    config = load_config(config_path)
    person_name = person_dir.name
    resolved, credential = resolve_person_config(config, person_name)

    cred_name = resolved.credential
    api_key_env = f"{cred_name.upper()}_API_KEY"
    mode = resolve_execution_mode(
        config, resolved.model, resolved.execution_mode,
    )
    return ModelConfig(
        model=resolved.model,
        fallback_model=resolved.fallback_model,
        max_tokens=resolved.max_tokens,
        max_turns=resolved.max_turns,
        api_key=credential.api_key or None,
        api_key_env=api_key_env,
        api_base_url=credential.base_url,
        context_threshold=resolved.context_threshold,
        max_chains=resolved.max_chains,
        conversation_history_threshold=resolved.conversation_history_threshold,
        execution_mode=resolved.execution_mode,
        supervisor=resolved.supervisor,
        speciality=resolved.speciality,
        resolved_mode=mode,
    )


def resolve_execution_mode(
    config: AnimaWorksConfig,
    model_name: str,
    explicit_override: str | None = None,
) -> str:
    """Resolve execution mode from model name with wildcard pattern support.

    Priority:
      1. Person's execution_mode explicit override (legacy)
      2. config.json model_modes table (exact + wildcard)
      3. DEFAULT_MODEL_MODE_PATTERNS (exact + wildcard)
      4. Default "B" (safe side)
    """
    if explicit_override:
        mapping = {"autonomous": "A2", "assisted": "B"}
        return mapping.get(explicit_override, explicit_override.upper())

    # config.json model_modes (user overrides)
    user_table = config.model_modes or {}
    if user_table:
        result = _match_pattern_table(model_name, user_table)
        if result is not None:
            return result

    # Code defaults
    result = _match_pattern_table(model_name, DEFAULT_MODEL_MODE_PATTERNS)
    if result is not None:
        return result

    return "B"  # unknown model → safe side