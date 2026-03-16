# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Central configuration module for AnimaWorks.

Defines Pydantic models for the unified config.json and provides
load / save / resolve helpers with a module-level singleton cache.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from core.schemas import ModelConfig

from pydantic import BaseModel, Field, model_validator

from core.exceptions import ConfigError  # noqa: F401

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
    timezone: str = ""  # IANA TZ name; empty = auto-detect from system
    gateway: GatewaySystemConfig = GatewaySystemConfig()
    worker: WorkerSystemConfig = WorkerSystemConfig()


class CredentialConfig(BaseModel):
    type: str = "api_key"
    api_key: str = ""
    keys: dict[str, str] = {}
    base_url: str | None = None


class AnimaModelConfig(BaseModel):
    """Per-anima config in config.json. Organization structure only."""

    supervisor: str | None = None
    speciality: str | None = None
    model: str | None = None


# ── Default model names (single source of truth) ─────────────────────────────
DEFAULT_ANIMA_MODEL: str = "claude-sonnet-4-6"
DEFAULT_CONSOLIDATION_MODEL: str = f"anthropic/{DEFAULT_ANIMA_MODEL}"


class AnimaDefaults(BaseModel):
    """Concrete defaults applied when a per-anima field is None."""

    model: str = DEFAULT_ANIMA_MODEL
    fallback_model: str | None = None
    background_model: str | None = None
    background_credential: str | None = None
    max_tokens: int = 8192
    max_turns: int = 20
    credential: str = "anthropic"
    context_threshold: float = 0.50
    max_chains: int = 2
    conversation_history_threshold: float = 0.30
    execution_mode: str | None = None  # None = auto-detect from model
    supervisor: str | None = None
    speciality: str | None = None
    thinking: bool | None = None  # Extended thinking (Bedrock: reasoning_effort, Ollama: think)
    thinking_effort: str | None = None  # "low"/"medium"/"high"/"max" (default: "high")
    llm_timeout: int = 600  # default LLM API timeout (seconds)
    mode_s_auth: str | None = None  # Mode S auth: "max"|"api"|"bedrock"|"vertex"|None(=max)
    max_outbound_per_hour: int | None = None
    max_outbound_per_day: int | None = None
    max_recipients_per_run: int | None = None
    default_workspace: str = ""


# ── Outbound budget defaults per role ─────────────────────────────────────────
ROLE_OUTBOUND_DEFAULTS: dict[str, dict[str, int]] = {
    "manager": {"max_outbound_per_hour": 60, "max_outbound_per_day": 300, "max_recipients_per_run": 10},
    "engineer": {"max_outbound_per_hour": 40, "max_outbound_per_day": 200, "max_recipients_per_run": 5},
    "writer": {"max_outbound_per_hour": 30, "max_outbound_per_day": 150, "max_recipients_per_run": 3},
    "researcher": {"max_outbound_per_hour": 30, "max_outbound_per_day": 150, "max_recipients_per_run": 3},
    "ops": {"max_outbound_per_hour": 20, "max_outbound_per_day": 80, "max_recipients_per_run": 2},
    "general": {"max_outbound_per_hour": 15, "max_outbound_per_day": 50, "max_recipients_per_run": 2},
}


def resolve_outbound_limits(
    anima_name: str,
    anima_dir: Path | None = None,
) -> dict[str, int]:
    """Resolve outbound limits for an Anima.

    Resolution order:
      1. status.json (per-Anima override)
      2. Role defaults from ROLE_OUTBOUND_DEFAULTS (based on status.json "role")
      3. "general" role as final fallback
    """
    _FIELDS = ("max_outbound_per_hour", "max_outbound_per_day", "max_recipients_per_run")
    fallback = ROLE_OUTBOUND_DEFAULTS["general"]

    if anima_dir is None:
        return dict(fallback)

    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return dict(fallback)

    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(fallback)

    role = data.get("role", "general")
    role_defaults = ROLE_OUTBOUND_DEFAULTS.get(role, fallback)

    result: dict[str, int] = {}
    for field in _FIELDS:
        val = data.get(field)
        if isinstance(val, int) and val > 0:
            result[field] = val
        else:
            result[field] = role_defaults.get(field, fallback[field])

    return result


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
    min_retrieval_score: float = 0.3
    skill_match_min_score: float = 0.75


class PromptConfig(BaseModel):
    """Configuration for system prompt building."""

    injection_size_warning_chars: int = 5000


class PrimingConfig(BaseModel):
    """Configuration for priming layer (automatic memory retrieval)."""

    dynamic_budget: bool = True
    budget_greeting: int = 500
    budget_question: int = 1500
    budget_request: int = 3000
    budget_heartbeat: int = 200  # fallback when context_window is unknown
    heartbeat_context_pct: float = 0.05  # fraction of context_window for HB budget


class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation processes."""

    daily_enabled: bool = True
    daily_time: str = "02:00"  # Format: HH:MM
    min_episodes_threshold: int = 1
    llm_model: str = DEFAULT_CONSOLIDATION_MODEL
    max_turns: int = 30  # Tool-call loop limit for consolidation tasks
    weekly_enabled: bool = True  # Phase 3 implementation
    weekly_time: str = "sun:03:00"  # Format: day:HH:MM
    duplicate_threshold: float = 0.85  # Similarity threshold for duplicate detection
    episode_retention_days: int = 30  # Days to retain uncompressed episodes
    monthly_enabled: bool = True  # Monthly forgetting toggle
    monthly_time: str = "1:04:00"  # Format: day:HH:MM (day of month)
    indexing_enabled: bool = True  # Daily RAG indexing toggle
    indexing_time: str = "04:00"  # Format: HH:MM


class ImageGenConfig(BaseModel):
    """Configuration for image generation and style consistency."""

    image_style: Literal["anime", "realistic"] = "realistic"
    style_reference: str | None = None  # Path to organization-wide style reference image
    style_prefix: str = ""  # Common style tags prepended to character prompt
    style_suffix: str = ""  # Common style tags appended to character prompt
    negative_prompt_extra: str = ""  # Extra tags added to negative prompt
    vibe_strength: float = 0.6  # Vibe Transfer strength (0.0-1.0)
    vibe_info_extracted: float = 0.8  # Vibe Transfer information extraction (0.0-1.0)
    enable_3d: bool = True  # Enable 3D model generation (Meshy API)


class NotificationChannelConfig(BaseModel):
    """Configuration for a single human notification channel."""

    type: str  # "slack", "line", "telegram", "chatwork", "ntfy"
    enabled: bool = True
    config: dict[str, Any] = {}


class HumanNotificationConfig(BaseModel):
    """Global configuration for human notification from top-level Animas."""

    enabled: bool = False
    channels: list[NotificationChannelConfig] = []


class UserAliasConfig(BaseModel):
    """External user contact information for outbound message routing."""

    slack_user_id: str = ""
    chatwork_room_id: str = ""


class ExternalMessagingChannelConfig(BaseModel):
    """Configuration for a single external messaging platform."""

    enabled: bool = False
    mode: str = "socket"  # "socket" | "webhook"
    anima_mapping: dict[str, str] = {}  # channel_id → anima_name
    default_anima: str = ""  # fallback anima for unmapped channels
    app_id_mapping: dict[str, str] = {}  # api_app_id → anima_name (per-Anima webhook routing)


class ExternalMessagingConfig(BaseModel):
    """Configuration for external messaging integration (inbound + outbound)."""

    preferred_channel: str = "slack"  # "slack" | "chatwork"
    user_aliases: dict[str, UserAliasConfig] = {}  # alias → contact info
    slack: ExternalMessagingChannelConfig = ExternalMessagingChannelConfig()
    chatwork: ExternalMessagingChannelConfig = ExternalMessagingChannelConfig()


class MediaProxyConfig(BaseModel):
    """Configuration for external image proxy hardening."""

    mode: Literal["allowlist", "open_with_scan"] = "open_with_scan"
    allowed_domains: list[str] = [
        "cdn.search.brave.com",
        "images.unsplash.com",
        "images.pexels.com",
        "upload.wikimedia.org",
    ]
    max_bytes: int = 5 * 1024 * 1024
    max_redirects: int = 3
    timeout_connect_s: float = 5.0
    timeout_read_s: float = 10.0
    rate_limit_requests: int = 30
    rate_limit_window_s: int = 60


class ServerConfig(BaseModel):
    """Server runtime configuration."""

    session_ttl_days: int | None = 7  # None = unlimited
    ipc_stream_timeout: int = 60  # per-chunk timeout in seconds
    keepalive_interval: int = 30  # keep-alive emission interval in seconds
    max_streaming_duration: int = 1800  # max streaming duration before hang (seconds)
    busy_hang_threshold: int = 900  # no-progress timeout for busy processes (seconds)
    stream_checkpoint_enabled: bool = True  # save tool results during streaming
    stream_retry_max: int = 3  # max automatic retries on stream disconnect
    stream_retry_delay_s: float = 5.0  # delay between retries (seconds)
    llm_num_retries: int = 3  # retries for LLM API calls (429/5xx/network)
    media_proxy: MediaProxyConfig = MediaProxyConfig()

    @model_validator(mode="after")
    def _validate_intervals(self) -> ServerConfig:
        if self.keepalive_interval >= self.ipc_stream_timeout:
            raise ValueError(
                f"keepalive_interval ({self.keepalive_interval}) must be "
                f"less than ipc_stream_timeout ({self.ipc_stream_timeout})"
            )
        return self


class BackgroundToolConfig(BaseModel):
    """Per-tool background execution threshold."""

    threshold_s: int = 30


class BackgroundTaskConfig(BaseModel):
    """Configuration for background tool execution."""

    enabled: bool = True
    eligible_tools: dict[str, BackgroundToolConfig] = {
        "generate_character_assets": BackgroundToolConfig(threshold_s=30),
        "generate_fullbody": BackgroundToolConfig(threshold_s=30),
        "generate_bustup": BackgroundToolConfig(threshold_s=30),
        "generate_chibi": BackgroundToolConfig(threshold_s=30),
        "generate_3d_model": BackgroundToolConfig(threshold_s=30),
        "generate_rigged_model": BackgroundToolConfig(threshold_s=30),
        "generate_animations": BackgroundToolConfig(threshold_s=30),
        "local_llm": BackgroundToolConfig(threshold_s=60),
        "run_command": BackgroundToolConfig(threshold_s=60),
    }
    result_retention_hours: int = 24
    max_parallel_llm_tasks: int = Field(default=3, ge=1, le=10)


class ActivityLogConfig(BaseModel):
    """Configuration for activity log rotation."""

    rotation_enabled: bool = True
    rotation_mode: Literal["size", "time", "both"] = "size"
    max_size_mb: int = Field(default=1024, ge=0)  # per-anima total, default 1GB
    max_age_days: int = Field(default=7, ge=0)  # mode="time"|"both" で使用
    rotation_time: str = "05:00"  # 実行時刻 (configured TZ)


class MachineConfig(BaseModel):
    """Configuration for machine tool (external agent CLI)."""

    engine_priority: list[str] = Field(
        default_factory=list,
        description="Engine priority order. First = recommended. Empty = use default.",
    )


class HousekeepingConfig(BaseModel):
    """Configuration for periodic disk cleanup."""

    enabled: bool = True
    run_time: str = "05:30"

    prompt_log_retention_days: int = 3
    daemon_log_max_size_mb: int = 100
    daemon_log_keep_generations: int = 5
    frontend_log_backup_count: int = 7
    dm_log_archive_retention_days: int = 30
    cron_log_retention_days: int = 30
    shortterm_retention_days: int = 7
    task_results_retention_days: int = 7
    pending_failed_retention_days: int = 14


class HeartbeatConfig(BaseModel):
    """Heartbeat scheduling and cascade prevention settings."""

    interval_minutes: int = Field(
        default=30, ge=1, le=1440
    )  # heartbeat interval (config-driven, not parsed from heartbeat.md)
    soft_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Seconds before injecting a wrap-up system-reminder into the HB session",
    )
    hard_timeout_seconds: int = Field(
        default=600,
        ge=60,
        le=7200,
        description="Seconds before forcefully terminating the HB session",
    )
    max_turns: int | None = Field(
        default=None,
        ge=3,
        le=200,
        description="HB-specific max_turns override (None = use per-anima model_config.max_turns)",
    )

    @model_validator(mode="after")
    def _validate_soft_lt_hard(self) -> HeartbeatConfig:
        if self.soft_timeout_seconds >= self.hard_timeout_seconds:
            raise ValueError(
                f"soft_timeout_seconds ({self.soft_timeout_seconds}) must be "
                f"less than hard_timeout_seconds ({self.hard_timeout_seconds})"
            )
        return self

    default_model: str | None = None  # global background model for heartbeat/cron (None = use main model)
    msg_heartbeat_cooldown_s: int = 300  # message-triggered heartbeat cooldown
    cascade_window_s: int = 1800  # sliding window for cascade detection
    cascade_threshold: int = 3  # max round-trips per pair within window
    depth_window_s: int = 600  # bilateral depth limiter window
    max_depth: int = 6  # max bilateral exchange depth
    actionable_intents: list[str] = ["report", "question"]
    enable_read_ack: bool = (
        False  # Send read-receipt ACK to message senders (disabled by default to prevent gratitude loops)
    )
    channel_post_cooldown_s: int = 300  # Min seconds between board posts per Anima (0 = no limit)
    max_messages_per_hour: int = 30  # Deprecated: use ROLE_OUTBOUND_DEFAULTS + status.json override
    max_messages_per_day: int = 100  # Deprecated: use ROLE_OUTBOUND_DEFAULTS + status.json override
    idle_compaction_minutes: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Minutes after last stream end to trigger idle auto-compaction",
    )


# ── Voice Chat Config ───────────────────────────────────────────────────────


class VoicevoxConfig(BaseModel):
    """VOICEVOX Engine connection settings."""

    base_url: str = "http://localhost:50021"


class ElevenLabsVoiceConfig(BaseModel):
    """ElevenLabs TTS API settings."""

    api_key_env: str = "ELEVENLABS_API_KEY"
    model_id: str = "eleven_flash_v2_5"


class StyleBertVits2Config(BaseModel):
    """Style-BERT-VITS2 / AivisSpeech connection settings."""

    base_url: str = "http://localhost:5000"


class VoiceConfig(BaseModel):
    """Voice chat configuration."""

    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "default"
    stt_language: str | None = None
    stt_refine_enabled: bool = False
    default_tts_provider: str = "voicevox"
    audio_format: str = "wav"
    voicevox: VoicevoxConfig = VoicevoxConfig()
    elevenlabs: ElevenLabsVoiceConfig = ElevenLabsVoiceConfig()
    style_bert_vits2: StyleBertVits2Config = StyleBertVits2Config()


# ── UI Config ────────────────────────────────────────────────────────────────


class UIConfig(BaseModel):
    """UI appearance and theme settings."""

    theme: str = "default"
    demo_mode: bool = False


# ── Activity Schedule ───────────────────────────────────────────────────────


class ActivityScheduleEntry(BaseModel):
    """A time-based activity level entry (e.g. daytime=100%, nighttime=30%)."""

    start: str = Field(description="Start time in HH:MM format")
    end: str = Field(description="End time in HH:MM format (may wrap past midnight)")
    level: int = Field(ge=10, le=400, description="Activity level percentage for this period")

    @model_validator(mode="after")
    def _validate_times(self) -> ActivityScheduleEntry:
        import re as _re

        _TIME_RE = _re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
        if not _TIME_RE.match(self.start):
            raise ValueError(f"Invalid start time: {self.start!r} (expected HH:MM)")
        if not _TIME_RE.match(self.end):
            raise ValueError(f"Invalid end time: {self.end!r} (expected HH:MM)")
        if self.start == self.end:
            raise ValueError(f"start and end must differ: {self.start}")
        return self


# ── Main Config ─────────────────────────────────────────────────────────────


class AnimaWorksConfig(BaseModel):
    version: int = 1
    setup_complete: bool = False
    locale: str = "ja"
    system: SystemConfig = SystemConfig()
    credentials: dict[str, CredentialConfig] = {"anthropic": CredentialConfig()}
    model_modes: dict[str, str] = {}  # モデル名 → "S"/"A"/"B" (legacy: "A1"/"A2" も可)
    model_context_windows: dict[str, int] = {}  # モデル名パターン → コンテキストウィンドウサイズ
    model_max_tokens: dict[str, int] = {}  # モデル名パターン → デフォルト max_tokens
    anima_defaults: AnimaDefaults = AnimaDefaults()
    animas: dict[str, AnimaModelConfig] = {}
    consolidation: ConsolidationConfig = ConsolidationConfig()
    rag: RAGConfig = RAGConfig()
    prompt: PromptConfig = PromptConfig()
    priming: PrimingConfig = PrimingConfig()
    image_gen: ImageGenConfig = ImageGenConfig()
    human_notification: HumanNotificationConfig = HumanNotificationConfig()
    server: ServerConfig = ServerConfig()
    external_messaging: ExternalMessagingConfig = ExternalMessagingConfig()
    background_task: BackgroundTaskConfig = BackgroundTaskConfig()
    activity_log: ActivityLogConfig = ActivityLogConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    voice: VoiceConfig = VoiceConfig()
    housekeeping: HousekeepingConfig = HousekeepingConfig()
    machine: MachineConfig = MachineConfig()
    workspaces: dict[str, str] = {}  # alias → absolute path
    activity_level: int = Field(
        default=100,
        ge=10,
        le=400,
        description="Global activity level (10-400%). Scales heartbeat interval and max_turns.",
    )
    activity_schedule: list[ActivityScheduleEntry] = Field(
        default_factory=list,
        description="Time-based activity level schedule. Empty = use fixed activity_level.",
    )
    ui: UIConfig = UIConfig()


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: AnimaWorksConfig | None = None
_config_path: Path | None = None
_config_mtime: float = 0.0


def invalidate_cache() -> None:
    """Reset the module-level singleton cache."""
    global _config, _config_path, _config_mtime
    _config = None
    _config_path = None
    _config_mtime = 0.0


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

    The cache is automatically invalidated when the file's mtime changes,
    so external edits (org_sync, manual changes) are picked up without
    requiring a server restart.
    """
    global _config, _config_path, _config_mtime

    if path is None:
        path = get_config_path()

    # Check whether the on-disk file has been modified since last load.
    if _config is not None and _config_path == path:
        try:
            disk_mtime = path.stat().st_mtime
        except OSError:
            disk_mtime = 0.0
        if disk_mtime == _config_mtime:
            return _config
        logger.debug("Config file changed on disk (mtime %.3f → %.3f); reloading", _config_mtime, disk_mtime)

    if path.is_file():
        logger.debug("Loading config from %s", path)
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw_text)
            config = AnimaWorksConfig.model_validate(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
        except ConfigError:
            raise
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", path, exc)
            raise ConfigError(f"Failed to load config from {path}: {exc}") from exc
    else:
        logger.info("Config file not found at %s; using defaults", path)
        config = AnimaWorksConfig()

    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0
    return config


def save_config(config: AnimaWorksConfig, path: Path | None = None) -> None:
    """Persist *config* to disk as pretty-printed JSON (mode 0o600).

    Updates the module-level singleton cache so subsequent :func:`load_config`
    calls return the freshly saved config.
    """
    global _config, _config_path, _config_mtime

    if path is None:
        path = get_config_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = config.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    # Atomic write: write to a PID-unique sibling temp file then rename so
    # that concurrent writers (multiple anima workers) never clobber each
    # other's temp file.  Each process writes to .config.json.<PID>.tmp,
    # then renames it to config.json atomically.
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.rename(path)

    logger.debug("Config saved to %s", path)

    _config = config
    _config_path = path
    try:
        _config_mtime = path.stat().st_mtime
    except OSError:
        _config_mtime = 0.0


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _load_status_json(anima_dir: Path) -> dict[str, Any]:
    """Load ModelConfig-relevant fields from anima's status.json.

    Returns a dict with field names matching AnimaDefaults fields.
    Missing or invalid files return an empty dict.
    """
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("Failed to read status.json from %s", anima_dir)
        return {}

    # Map status.json fields to AnimaModelConfig field names
    result: dict[str, Any] = {}
    field_mapping = {
        "model": "model",
        "background_model": "background_model",
        "background_credential": "background_credential",
        "context_threshold": "context_threshold",
        "max_turns": "max_turns",
        "max_chains": "max_chains",
        "conversation_history_threshold": "conversation_history_threshold",
        "credential": "credential",
        "execution_mode": "execution_mode",
        "supervisor": "supervisor",
        "max_tokens": "max_tokens",
        "fallback_model": "fallback_model",
        "thinking": "thinking",
        "thinking_effort": "thinking_effort",
        "llm_timeout": "llm_timeout",
        "mode_s_auth": "mode_s_auth",
        "max_outbound_per_hour": "max_outbound_per_hour",
        "max_outbound_per_day": "max_outbound_per_day",
        "max_recipients_per_run": "max_recipients_per_run",
        "default_workspace": "default_workspace",
    }
    # Fields where None is a valid explicit value (e.g. supervisor=null
    # means "top-level / no supervisor").  Empty string is still "not set".
    _nullable_fields = frozenset({"supervisor", "speciality"})
    for status_key, config_key in field_mapping.items():
        if status_key in data:
            value = data[status_key]
            if value is None and status_key in _nullable_fields or value not in (None, ""):
                result[config_key] = value
    return result


def resolve_anima_config(
    config: AnimaWorksConfig,
    anima_name: str,
    anima_dir: Path | None = None,
) -> tuple[AnimaDefaults, CredentialConfig]:
    """Merge status.json with *anima_defaults* (3-layer merge).

    Resolution uses a 3-layer priority (strongest first):

      1. ``status.json`` in *anima_dir* (SSoT for all fields including org)
      2. ``config.json`` per-anima (``config.animas``) — fallback for
         ``supervisor`` and ``speciality`` only
      3. ``config.json`` anima_defaults (global defaults)

    For ``supervisor`` and ``speciality``, explicit ``null`` in status.json
    is respected (means "top-level / no supervisor").

    When *anima_dir* is ``None``, layer 1 is skipped (no status.json).

    Returns:
        A ``(resolved_defaults, credential)`` tuple.

    Raises:
        KeyError: If the resolved credential name is not in
            ``config.credentials``.
    """
    anima_entry = config.animas.get(anima_name, AnimaModelConfig())
    defaults = config.anima_defaults

    status_values = _load_status_json(anima_dir) if anima_dir else {}

    # Merge priority (strongest first):
    #   1. status.json (including explicit null for supervisor/speciality)
    #   2. config.animas (fallback for supervisor/speciality only)
    #   3. anima_defaults (global defaults)
    resolved: dict[str, Any] = {}
    for field_name in AnimaDefaults.model_fields:
        if field_name in status_values:
            resolved[field_name] = status_values[field_name]
        elif field_name in ("supervisor", "speciality"):
            # Fallback to config.animas for org structure fields
            anima_value = getattr(anima_entry, field_name)
            if anima_value is not None:
                resolved[field_name] = anima_value
            else:
                resolved[field_name] = getattr(defaults, field_name)
        else:
            resolved[field_name] = getattr(defaults, field_name)

    resolved_defaults = AnimaDefaults.model_validate(resolved)

    credential_name = resolved_defaults.credential
    if credential_name not in config.credentials:
        raise KeyError(f"Credential '{credential_name}' (for anima '{anima_name}') not found in config.credentials")

    credential = config.credentials[credential_name]
    return resolved_defaults, credential


# ---------------------------------------------------------------------------
# Model mode resolution
# ---------------------------------------------------------------------------

# Default model_modes with wildcard pattern support.
# Patterns use fnmatch syntax (*, ?, [seq]).
# Order matters for specificity — more specific patterns should appear first,
# but the resolver sorts by specificity automatically.
#
# Mode values: S = SDK (Agent SDK / Claude Code), A = Autonomous (tool_use),
#              C = Codex (Codex CLI wrapper), B = Basic (no tool_use)
#
# IMPORTANT: When status.json omits "execution_mode", resolve_execution_mode()
# falls through to these patterns to determine the mode from the model name.
# For example, "claude-sonnet-4-6" matches "claude-*" → Mode S.
# An anima can override this by setting execution_mode explicitly in status.json
# (e.g. bedrock/* defaults to A, but mei uses execution_mode="S" to force Mode S).
DEFAULT_MODEL_MODE_PATTERNS: dict[str, str] = {
    # ── S: Claude Agent SDK ──────────────────────────────
    "claude-*": "S",
    # ── C: Codex SDK (Codex CLI wrapper) ─────────────────
    "codex/*": "C",
    # ── A: Cloud API providers (LiteLLM + tool_use) ──────
    "openai/*": "A",
    "azure/*": "A",
    "bedrock/*": "A",
    "google/*": "A",
    "vertex_ai/*": "A",
    "mistral/*": "A",
    "xai/*": "A",
    "cohere/*": "A",
    "zai/*": "A",
    "minimax/*": "A",
    "moonshot/*": "A",
    "deepseek/deepseek-chat": "A",
    # ── A: Ollama models with reliable tool_use ──────────
    "ollama/qwen3.5*": "A",
    "ollama/qwen3:14b": "A",
    "ollama/qwen3:30b": "A",
    "ollama/qwen3:32b": "A",
    "ollama/qwen3:235b": "A",
    "ollama/qwen3-coder:*": "A",
    "ollama/llama4:*": "A",
    "ollama/mistral-small3.2:*": "A",
    "ollama/devstral*": "A",
    "ollama/glm-4.7*": "A",
    "ollama/glm-5*": "A",
    "ollama/minimax*": "A",
    "ollama/kimi-k2*": "A",
    "ollama/gpt-oss*": "A",
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

# ── Known model catalog ──────────────────────────────────────────────────────
# Concrete model names for reference. Used by SUPERVISOR_TOOLS description.
# Mode is determined by DEFAULT_MODEL_MODE_PATTERNS at runtime; this list is
# informational and does NOT restrict which models can be used.
KNOWN_MODELS: list[dict[str, str]] = [
    # ── Claude / Anthropic (Mode S) ──────────────────────────────────────────
    {"name": "claude-opus-4-6", "mode": "S", "note": "最高性能・推奨"},
    {"name": "claude-sonnet-4-6", "mode": "S", "note": "バランス型・推奨"},
    {"name": "claude-haiku-4-5-20251001", "mode": "S", "note": "軽量・高速"},
    # Legacy (still available)
    {"name": "claude-opus-4-5-20251101", "mode": "S", "note": "旧フラッグシップ"},
    {"name": "claude-opus-4-1-20250805", "mode": "S", "note": "旧Opus"},
    {"name": "claude-sonnet-4-5-20250929", "mode": "S", "note": "旧Sonnet"},
    {"name": "claude-sonnet-4-20250514", "mode": "S", "note": "旧Sonnet4"},
    # ── OpenAI (Mode A) ──────────────────────────────────────────────────────
    {"name": "openai/gpt-4.1", "mode": "A", "note": "最新・コーディング強"},
    {"name": "openai/gpt-4.1-mini", "mode": "A", "note": "高速・低コスト"},
    {"name": "openai/gpt-4.1-nano", "mode": "A", "note": "最軽量"},
    {"name": "openai/gpt-4o", "mode": "A", "note": "音声対応・レガシー"},
    {"name": "openai/o3-2025-04-16", "mode": "A", "note": "推論特化"},
    {"name": "openai/o4-mini-2025-04-16", "mode": "A", "note": "推論・低コスト"},
    # ── Azure OpenAI (Mode A) ──────────────────────────────────────────────────
    {"name": "azure/gpt-4.1-mini", "mode": "A", "note": "Azure OpenAI 4.1-mini"},
    {"name": "azure/gpt-4.1", "mode": "A", "note": "Azure OpenAI 4.1"},
    # ── Google Gemini (Mode A) ────────────────────────────────────────────────
    {"name": "google/gemini-2.5-pro", "mode": "A", "note": "最高性能"},
    {"name": "google/gemini-2.5-flash", "mode": "A", "note": "高速バランス"},
    {"name": "google/gemini-2.5-flash-lite", "mode": "A", "note": "軽量・高スループット"},
    # ── Vertex AI (Mode A) ────────────────────────────────────────────────────
    {"name": "vertex_ai/gemini-2.5-flash", "mode": "A", "note": "Vertex AI Flash"},
    {"name": "vertex_ai/gemini-2.5-pro", "mode": "A", "note": "Vertex AI Pro"},
    # ── xAI Grok (Mode A) ─────────────────────────────────────────────────────
    {"name": "xai/grok-4", "mode": "A", "note": "最新Grok"},
    {"name": "xai/grok-3-beta", "mode": "A", "note": "安定版"},
    {"name": "xai/grok-3-mini-beta", "mode": "A", "note": "軽量Grok"},
    # ── Ollama Local (Mode A: tool_use 対応) ─────────────────────────────────
    {"name": "ollama/qwen3.5:9b", "mode": "A", "note": "GDN hybrid 9B・高効率"},
    {"name": "ollama/glm-4.7", "mode": "A", "note": "ローカル・tool_use対応"},
    {"name": "ollama/qwen3:14b", "mode": "A", "note": "ローカル中型"},
    {"name": "ollama/qwen3:32b", "mode": "A", "note": "ローカル大型"},
    # ── Codex (Mode C) ──────────────────────────────────────────────────────
    {"name": "codex/o4-mini", "mode": "C", "note": "Codex CLI経由・高速"},
    {"name": "codex/o3", "mode": "C", "note": "Codex CLI経由・推論"},
    {"name": "codex/gpt-4.1", "mode": "C", "note": "Codex CLI経由・コーディング"},
    # ── Ollama Local (Mode B: tool_use 非対応) ────────────────────────────────
    {"name": "ollama/gemma3:4b", "mode": "B", "note": "軽量ローカル"},
    {"name": "ollama/gemma3:12b", "mode": "B", "note": "中型ローカル"},
]

# ── Legacy mode value mapping ──────────────────────────────
# Maps legacy A1/A1F/A2 and text-based values to canonical S/A/B scheme.
_LEGACY_MODE_MAP: dict[str, str] = {
    "autonomous": "A",
    "assisted": "B",
    "a1": "S",
    "a1f": "A",
    "a1_fallback": "A",
    "a2": "A",
}

# ── models.json cache ─────────────────────────────────────
_models_json_cache: dict[str, dict] | None = None
_models_json_mtime: float = 0.0


def _load_models_json() -> dict[str, dict]:
    """Load the user-editable models.json from the runtime data directory.

    Reads ``~/.animaworks/models.json`` (resolved via
    ``core.paths.get_data_dir``).  The result is cached at module level and
    automatically reloaded when the file's mtime changes.

    Returns:
        A dict mapping model-name patterns to entry dicts containing
        ``"mode"`` and ``"context_window"`` keys.  Returns an empty dict
        if the file is missing or cannot be parsed.
    """
    global _models_json_cache, _models_json_mtime

    from core.paths import get_data_dir

    models_path = get_data_dir() / "models.json"

    # Fast path: return cache if mtime unchanged
    if _models_json_cache is not None:
        try:
            disk_mtime = models_path.stat().st_mtime
        except OSError:
            disk_mtime = 0.0
        if disk_mtime == _models_json_mtime:
            return _models_json_cache

    # Capture mtime before reading to avoid TOCTOU race
    try:
        file_mtime = models_path.stat().st_mtime
    except OSError:
        logger.debug("models.json not found at %s; skipping", models_path)
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load models.json from %s: %s", models_path, exc)
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    if not isinstance(raw, dict):
        logger.warning("models.json is not a JSON object; ignoring")
        _models_json_cache = {}
        _models_json_mtime = 0.0
        return _models_json_cache

    # Filter to entries that are dicts (skip comment keys, etc.)
    result: dict[str, dict] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            result[key] = value

    _models_json_cache = result
    _models_json_mtime = file_mtime

    logger.debug("Loaded models.json with %d entries", len(result))
    return _models_json_cache


def invalidate_models_json_cache() -> None:
    """Reset the models.json module-level cache."""
    global _models_json_cache, _models_json_mtime
    _models_json_cache = None
    _models_json_mtime = 0.0


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

    Returns the mode string (e.g. ``"S"``) or ``None`` if no match.
    """
    # Phase 1: exact match
    if model_name in table:
        return table[model_name].upper()

    # Phase 2: wildcard patterns sorted by specificity
    wildcard_patterns = [p for p in table if any(c in p for c in ("*", "?", "["))]
    wildcard_patterns.sort(key=_pattern_specificity)

    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(model_name, pattern):
            return table[pattern].upper()

    return None


def load_model_config(anima_dir: Path) -> ModelConfig:
    """Build a ModelConfig for *anima_dir* from the unified config.json.

    This is a standalone version of ``MemoryManager.read_model_config()``
    for use in server routes that do not have a live DigitalAnima instance.
    """
    from core.schemas import ModelConfig

    config_path = get_config_path()
    if not config_path.exists():
        return ModelConfig()

    config = load_config(config_path)
    anima_name = anima_dir.name
    resolved, credential = resolve_anima_config(config, anima_name, anima_dir=anima_dir)

    cred_name = resolved.credential
    api_key_env = f"{cred_name.upper()}_API_KEY"
    mode = resolve_execution_mode(
        config,
        resolved.model,
        resolved.execution_mode,
    )
    return ModelConfig(
        model=resolved.model,
        fallback_model=resolved.fallback_model,
        background_model=resolved.background_model,
        background_credential=resolved.background_credential,
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
        thinking=resolved.thinking,
        thinking_effort=resolved.thinking_effort,
        llm_timeout=resolved.llm_timeout,
        extra_keys=credential.keys or {},
        mode_s_auth=resolved.mode_s_auth,
    )


def _normalise_mode(raw: str) -> str:
    """Normalise a mode value to S/A/B, applying legacy mapping if needed.

    Accepts legacy values (``"A1"``, ``"A2"``, ``"autonomous"``, etc.) and
    canonical values (``"S"``, ``"A"``, ``"B"``).  Returns uppercase S/A/B.
    """
    lower = raw.strip().lower()
    mapped = _LEGACY_MODE_MAP.get(lower)
    if mapped:
        return mapped
    upper = raw.strip().upper()
    if upper in ("S", "A", "B", "C"):
        return upper
    # Unrecognised — return as-is (upper) for forward compat
    logger.warning("Unrecognised execution mode '%s'; passing through as '%s'", raw, upper)
    return upper


def _match_models_json(model_name: str) -> dict | None:
    """Match *model_name* against models.json entries.

    Returns the matched entry dict (with ``"mode"`` and ``"context_window"``
    keys) or ``None`` if no match.  Uses specificity-sorted pattern matching.
    """
    table = _load_models_json()
    if not table:
        return None

    # Phase 1: exact match
    if model_name in table:
        return table[model_name]

    # Phase 2: wildcard patterns sorted by specificity
    wildcard_patterns = [p for p in table if any(c in p for c in ("*", "?", "["))]
    wildcard_patterns.sort(key=_pattern_specificity)

    for pattern in wildcard_patterns:
        if fnmatch.fnmatch(model_name, pattern):
            return table[pattern]

    return None


def resolve_execution_mode(
    config: AnimaWorksConfig,
    model_name: str,
    explicit_override: str | None = None,
) -> str:
    """Resolve execution mode from model name with wildcard pattern support.

    When ``status.json`` omits ``execution_mode``, this function determines
    the mode automatically from the model name.  For example,
    ``claude-sonnet-4-6`` matches the ``"claude-*": "S"`` pattern and runs
    in Mode S without an explicit setting.

    Priority:
      1. Per-anima explicit override (``status.json`` ``execution_mode``)
      2. models.json user table (``~/.animaworks/models.json``)
      3. config.json model_modes (deprecated fallback, with legacy mapping)
      4. DEFAULT_MODEL_MODE_PATTERNS (code defaults, e.g. ``"claude-*"`` → S)
      5. Default ``"B"`` (safe side)

    Args:
        config: Global AnimaWorks configuration.
        model_name: Model identifier (e.g. ``"claude-sonnet-4-6"``,
            ``"bedrock/jp.anthropic.claude-sonnet-4-6"``).
        explicit_override: Per-anima ``execution_mode`` from ``status.json``.
            When set, takes highest priority.

    Returns:
        One of ``"S"`` (SDK), ``"C"`` (Codex), ``"A"`` (Autonomous),
        or ``"B"`` (Basic).
    """
    # 1. Per-anima explicit override
    if explicit_override:
        return _normalise_mode(explicit_override)

    # 2. models.json user table
    entry = _match_models_json(model_name)
    if entry is not None:
        mode_val = entry.get("mode")
        if mode_val:
            return _normalise_mode(str(mode_val))

    # 3. config.json model_modes (deprecated fallback)
    user_table = config.model_modes or {}
    if user_table:
        result = _match_pattern_table(model_name, user_table)
        if result is not None:
            return _normalise_mode(result)

    # 4. Code defaults
    result = _match_pattern_table(model_name, DEFAULT_MODEL_MODE_PATTERNS)
    if result is not None:
        return result  # Already S/A/B in the table

    return "B"  # unknown model → safe side


def resolve_context_window(
    model_name: str,
    config: AnimaWorksConfig | None = None,
) -> int | None:
    """Resolve context window size from model name.

    Priority:
      1. models.json (``~/.animaworks/models.json``, ``"context_window"`` field)
      2. config.json ``model_context_windows`` (deprecated fallback)
      3. ``None`` (caller should fall through to ``core.prompt.context``
         defaults)

    Args:
        model_name: The model name to resolve (e.g. ``"claude-sonnet-4-6"``).
        config: Optional config instance.  Loaded lazily if not provided.

    Returns:
        Context window size in tokens, or ``None`` if not found in any
        user-editable source (caller should use hardcoded defaults).
    """
    # 1. models.json
    entry = _match_models_json(model_name)
    if entry is not None:
        cw = entry.get("context_window")
        if cw is not None:
            try:
                return int(cw)
            except (ValueError, TypeError):
                pass

    # 2. config.json model_context_windows
    if config is None:
        try:
            config = load_config()
        except Exception:
            return None
    overrides = config.model_context_windows or {}
    if overrides:
        bare = model_name.split("/", 1)[-1] if "/" in model_name else model_name
        for pattern, size in overrides.items():
            if fnmatch.fnmatch(model_name, pattern) or fnmatch.fnmatch(bare, pattern):
                return size

    return None


def resolve_penalties(model_name: str) -> dict[str, float]:
    """Resolve frequency_penalty and presence_penalty from models.json.

    Returns a dict with only the keys that have non-None values.
    An empty dict means no penalties configured (backward-compatible).
    Values are clamped to [-2.0, 2.0] per the OpenAI API specification.
    """
    entry = _match_models_json(model_name)
    result: dict[str, float] = {}
    if entry is not None:
        for key in ("frequency_penalty", "presence_penalty"):
            val = entry.get(key)
            if val is not None:
                try:
                    result[key] = max(-2.0, min(2.0, float(val)))
                except (ValueError, TypeError):
                    pass
    return result


# ---------------------------------------------------------------------------
# max_tokens resolution
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: int = 8192
_THINKING_MIN_MAX_TOKENS: int = 16384


def _match_model_max_tokens(
    model_name: str,
    config: AnimaWorksConfig | None = None,
) -> int | None:
    """Match *model_name* against ``config.model_max_tokens`` pattern table."""
    if config is None:
        try:
            config = load_config()
        except Exception:
            return None
    table = config.model_max_tokens or {}
    if not table:
        return None
    bare = model_name.split("/", 1)[-1] if "/" in model_name else model_name
    for pattern, value in table.items():
        if fnmatch.fnmatch(model_name, pattern) or fnmatch.fnmatch(bare, pattern):
            return value
    return None


def resolve_max_tokens(
    model_name: str,
    explicit: int | None,
    thinking: bool | None,
    config: AnimaWorksConfig | None = None,
) -> int:
    """Resolve effective max_tokens.

    Priority:
      1. ``config.model_max_tokens`` pattern match (overrides all)
      2. Thinking minimum floor (16384 when thinking enabled, raises low values)
      3. Explicit value (status.json ``max_tokens``)
      4. DEFAULT_MAX_TOKENS (8192)
    """
    matched = _match_model_max_tokens(model_name, config)
    if matched is not None:
        return matched
    base = explicit if (explicit is not None and explicit != DEFAULT_MAX_TOKENS) else DEFAULT_MAX_TOKENS
    if thinking:
        return max(_THINKING_MIN_MAX_TOKENS, base)
    return base


# ---------------------------------------------------------------------------
# Anima registration helpers
# ---------------------------------------------------------------------------


_SENTINEL = object()

_NONE_SUPERVISOR_VALUES = frozenset({"なし", "(なし)", "（なし）", "-", "---", ""})

_PAREN_EN_NAME_RE = re.compile(r"[（(]([A-Za-z_][A-Za-z0-9_]*)[）)]")


def update_status_model(
    anima_dir: Path,
    *,
    model: str | None = None,
    credential: str | None = None,
    background_model: str | None | object = _SENTINEL,
    background_credential: str | None | object = _SENTINEL,
) -> None:
    """Update model/credential in an anima's status.json (atomic write).

    For background_model/background_credential, pass empty string ``""`` to
    clear (remove the field).  The default sentinel leaves the field unchanged.
    """
    status_path = anima_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"status.json not found: {status_path}")
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if model is not None:
        data["model"] = model
    if credential is not None:
        data["credential"] = credential
    if background_model is not _SENTINEL:
        if background_model:
            data["background_model"] = background_model
        else:
            data.pop("background_model", None)
    if background_credential is not _SENTINEL:
        if background_credential:
            data["background_credential"] = background_credential
        else:
            data.pop("background_credential", None)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(status_path)


def _resolve_supervisor_name(raw: str) -> str | None:
    """Resolve a raw supervisor value to an anima name.

    Handles formats like ``"琴葉（kotoha）"`` → ``"kotoha"``,
    ``"sakura"`` → ``"sakura"``, ``"(なし)"`` → ``None``.
    """
    raw = raw.strip()
    if not raw or raw in _NONE_SUPERVISOR_VALUES:
        return None

    # Extract English name from parentheses (full-width or half-width)
    m = _PAREN_EN_NAME_RE.search(raw)
    if m:
        return m.group(1).lower()

    # Plain ASCII name
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
        return raw.lower()

    # Non-ASCII without English equivalent — cannot resolve to an anima name
    logger.warning(
        "Supervisor value '%s' has no English name in parentheses; ignoring",
        raw,
    )
    return None


def read_anima_supervisor(anima_dir: Path) -> str | None:
    """Read the supervisor field for an anima from status.json or identity.md.

    Tries two sources in order:
      1. ``status.json`` — looks for ``"supervisor"`` key.
      2. ``identity.md`` — parses the Japanese table format ``| 上司 | name |``.

    The raw value is resolved to an English anima name (e.g.
    ``"琴葉（kotoha）"`` → ``"kotoha"``).

    Args:
        anima_dir: Path to the anima's runtime directory
            (e.g. ``~/.animaworks/animas/hinata``).

    Returns:
        The supervisor anima name if found, otherwise ``None``.
    """
    # Source 1: status.json
    status_path = anima_dir / "status.json"
    if status_path.is_file():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            raw = data.get("supervisor", "")
            if raw:
                resolved = _resolve_supervisor_name(raw)
                if resolved:
                    return resolved
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", status_path, exc)

    # Source 2: identity.md table row  | 上司 | name |
    identity_path = anima_dir / "identity.md"
    if identity_path.is_file():
        try:
            content = identity_path.read_text(encoding="utf-8")
            m = re.search(r"\|\s*上司\s*\|\s*(.+?)\s*\|", content)
            if m:
                resolved = _resolve_supervisor_name(m.group(1))
                if resolved:
                    return resolved
        except OSError as exc:
            logger.warning("Failed to read %s: %s", identity_path, exc)

    return None


def register_anima_in_config(
    data_dir: Path,
    anima_name: str,
) -> None:
    """Register a newly created anima in config.json with supervisor synced.

    Reads status.json / identity.md in the anima directory to extract the
    ``supervisor`` field and stores it in the :class:`AnimaModelConfig` entry
    inside config.json.  If the anima already exists in the config the call
    is a no-op.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        anima_name: Name of the anima to register.
    """
    config_path = data_dir / "config.json"
    if not config_path.exists():
        return
    config = load_config(config_path)
    if anima_name not in config.animas:
        anima_dir = data_dir / "animas" / anima_name
        supervisor = read_anima_supervisor(anima_dir) if anima_dir.exists() else None
        config.animas[anima_name] = AnimaModelConfig(supervisor=supervisor)
        save_config(config, config_path)
        logger.debug(
            "Registered anima '%s' in config (supervisor=%s)",
            anima_name,
            supervisor,
        )


def unregister_anima_from_config(
    data_dir: Path,
    anima_name: str,
) -> bool:
    """Remove an anima from config.json.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        anima_name: Name of the anima to unregister.

    Returns:
        True if the anima was found and removed, False if it was not present.
    """
    config_path = data_dir / "config.json"
    if not config_path.exists():
        return False
    config = load_config(config_path)
    if anima_name not in config.animas:
        return False
    del config.animas[anima_name]
    save_config(config, config_path)
    logger.debug("Unregistered anima '%s' from config", anima_name)
    return True


def rename_anima_in_config(
    data_dir: Path,
    old_name: str,
    new_name: str,
) -> int:
    """Rename an anima in config.json: key, supervisor refs, anima_mapping.

    Args:
        data_dir: The AnimaWorks data directory (e.g. ``~/.animaworks``).
        old_name: Current anima name.
        new_name: New anima name.

    Returns:
        Number of supervisor references updated.

    Raises:
        KeyError: If *old_name* is not found in ``config.animas``.
    """
    config_path = data_dir / "config.json"
    if not config_path.exists():
        raise KeyError(f"config.json not found in {data_dir}")
    config = load_config(config_path)
    if old_name not in config.animas:
        raise KeyError(f"Anima '{old_name}' not found in config.animas")

    # 1. Move animas entry
    entry = config.animas.pop(old_name)
    config.animas[new_name] = entry

    # 2. Update supervisor references across all animas
    supervisor_count = 0
    for _name, anima_cfg in config.animas.items():
        if anima_cfg.supervisor == old_name:
            anima_cfg.supervisor = new_name
            supervisor_count += 1

    # 3. Update external_messaging: anima_mapping, app_id_mapping, default_anima
    for channel_cfg in (config.external_messaging.slack, config.external_messaging.chatwork):
        for mapping_attr in ("anima_mapping", "app_id_mapping"):
            mapping = getattr(channel_cfg, mapping_attr, {})
            for key, mapped_name in list(mapping.items()):
                if mapped_name == old_name:
                    mapping[key] = new_name
        if channel_cfg.default_anima == old_name:
            channel_cfg.default_anima = new_name

    save_config(config, config_path)
    logger.debug(
        "Renamed anima '%s' → '%s' in config (%d supervisor refs)",
        old_name,
        new_name,
        supervisor_count,
    )
    return supervisor_count
