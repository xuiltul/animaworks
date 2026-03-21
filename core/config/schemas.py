# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Pydantic configuration schemas for AnimaWorks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    extra_mcp_servers: dict[str, dict] = Field(default_factory=dict)
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
    budget_question: int = 2000
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
        _TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
        if not _TIME_RE.match(self.start):
            raise ValueError(f"Invalid start time: {self.start!r} (expected HH:MM)")
        if not _TIME_RE.match(self.end):
            raise ValueError(f"Invalid end time: {self.end!r} (expected HH:MM)")
        if self.start == self.end:
            raise ValueError(f"start and end must differ: {self.start}")
        return self


# ── Global Permissions Config ─────────────────────────────────────────────────


class GlobalDenyPattern(BaseModel):
    """A single deny pattern with regex and human-readable reason."""

    pattern: str
    reason: str


class GlobalCommandsDeny(BaseModel):
    """Global command deny configuration."""

    deny: list[GlobalDenyPattern] = Field(default_factory=list)


class GlobalPermissionsConfig(BaseModel):
    """Global permissions loaded from permissions.global.json.

    Applies to ALL Animas.  Loaded once at server startup, cached in memory.
    Runtime modifications to the on-disk file are auto-reverted.
    """

    version: int = 1
    injection_patterns: list[GlobalDenyPattern] = Field(default_factory=list)
    commands: GlobalCommandsDeny = Field(default_factory=GlobalCommandsDeny)


# ── Per-Anima Permissions Config ──────────────────────────────────────────────


class CommandsPermission(BaseModel):
    """Permission rules for command execution."""

    allow_all: bool = True
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class ExternalToolsPermission(BaseModel):
    """Permission rules for external tool access."""

    allow_all: bool = True
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class ToolCreationPermission(BaseModel):
    """Permission rules for tool/skill creation."""

    personal: bool = True
    shared: bool = False


class PermissionsConfig(BaseModel):
    """Structured permissions for an Anima (replaces permissions.md).

    Default values implement 'Open by Default, Deny by Exception' policy.
    """

    version: int = 1
    file_roots: list[str] = Field(default_factory=lambda: ["/"])
    commands: CommandsPermission = Field(default_factory=CommandsPermission)
    external_tools: ExternalToolsPermission = Field(default_factory=ExternalToolsPermission)
    tool_creation: ToolCreationPermission = Field(default_factory=ToolCreationPermission)


def load_permissions(anima_dir: Path) -> PermissionsConfig:
    """Load permissions from permissions.json, with migration fallback.

    Resolution order:
      1. permissions.json exists -> load and validate
      2. permissions.md only -> auto-migrate, return config
      3. Neither exists -> return default (open)
      4. Invalid JSON -> warning + return default (open)
    """
    json_path = anima_dir / "permissions.json"
    md_path = anima_dir / "permissions.md"

    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            config = PermissionsConfig.model_validate(data)
            version = data.get("version")
            if version is not None and version != 1:
                logger.warning("permissions.json version %s is unknown; using known fields only", version)
            return config
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse permissions.json at %s: %s — using open defaults", json_path, exc)
            return PermissionsConfig()
        except Exception as exc:
            logger.warning("Invalid permissions.json at %s: %s — using open defaults", json_path, exc)
            return PermissionsConfig()

    if md_path.is_file():
        from core.config.migrate import migrate_permissions_md_to_json

        return migrate_permissions_md_to_json(anima_dir)

    return PermissionsConfig()


def _format_permissions_for_prompt(config: PermissionsConfig, anima_name: str) -> str:
    """Convert PermissionsConfig to a human/LLM-readable text block."""
    lines = [f"## Permissions: {anima_name}"]
    if config.file_roots == ["/"]:
        lines.append("- File access: unrestricted")
    elif not config.file_roots:
        lines.append("- File access: own directory and shared framework directories only")
    else:
        lines.append(f"- File access limited to: {', '.join(config.file_roots)}")
    if config.commands.allow_all:
        lines.append("- Commands: all allowed (global permission blocks still apply)")
    else:
        if config.commands.allow:
            lines.append(f"- Allowed commands: {', '.join(config.commands.allow)}")
        else:
            lines.append("- Commands: none allowed")
    if config.commands.deny:
        lines.append(f"- Additionally denied commands: {', '.join(config.commands.deny)}")
    if config.external_tools.allow_all:
        lines.append("- External tools: all allowed")
    else:
        if config.external_tools.allow:
            lines.append(f"- Allowed external tools: {', '.join(config.external_tools.allow)}")
        else:
            lines.append("- External tools: none allowed")
    if config.external_tools.deny:
        lines.append(f"- Denied external tools: {', '.join(config.external_tools.deny)}")
    tc = config.tool_creation
    lines.append(f"- Tool creation: personal={'yes' if tc.personal else 'no'}, shared={'yes' if tc.shared else 'no'}")
    return "\n".join(lines)


# ── Main Config ─────────────────────────────────────────────────────────────


class AnimaWorksConfig(BaseModel):
    version: int = 1
    setup_complete: bool = False
    locale: str = "ja"
    system: SystemConfig = SystemConfig()
    credentials: dict[str, CredentialConfig] = {"anthropic": CredentialConfig()}
    model_modes: dict[str, str] = {}  # モデル名 → "S"/"A"/"B" (legacy: "A1"/"A2" も可)
    model_context_windows: dict[str, int] = {}  # DEPRECATED: use models.json instead. Kept for backward compat only.
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


__all__ = [
    "ActivityLogConfig",
    "ActivityScheduleEntry",
    "AnimaDefaults",
    "AnimaModelConfig",
    "AnimaWorksConfig",
    "BackgroundTaskConfig",
    "BackgroundToolConfig",
    "ConsolidationConfig",
    "CredentialConfig",
    "DEFAULT_ANIMA_MODEL",
    "DEFAULT_CONSOLIDATION_MODEL",
    "ElevenLabsVoiceConfig",
    "ExternalMessagingChannelConfig",
    "ExternalMessagingConfig",
    "GatewaySystemConfig",
    "HeartbeatConfig",
    "HousekeepingConfig",
    "HumanNotificationConfig",
    "ImageGenConfig",
    "MachineConfig",
    "MediaProxyConfig",
    "NotificationChannelConfig",
    "PrimingConfig",
    "PromptConfig",
    "RAGConfig",
    "ROLE_OUTBOUND_DEFAULTS",
    "ServerConfig",
    "StyleBertVits2Config",
    "SystemConfig",
    "UIConfig",
    "UserAliasConfig",
    "VoiceConfig",
    "VoicevoxConfig",
    "WorkerSystemConfig",
    "resolve_outbound_limits",
]
