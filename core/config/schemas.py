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
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from core.enclave.config import EnclaveClientConfig, EnclaveConfig

logger = logging.getLogger("animaworks.config")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    mode: str = "server"
    timezone: str = ""  # IANA TZ name; empty = auto-detect from system


class CredentialConfig(BaseModel):
    type: str = "api_key"
    api_key: str = ""
    keys: dict[str, str] = {}
    base_url: str | None = None


class AnimaModelConfig(BaseModel):
    """Per-anima config in config.json. Organization structure only."""

    supervisor: str | None = None
    company: str | None = None
    speciality: str | None = None
    model: str | None = None
    heartbeat_enabled: bool | None = None
    background_review_enabled: bool | None = None
    token_budget_monthly: int | None = None
    aliases: list[str] = []
    """Alternative names (e.g. Japanese) that resolve to this anima's canonical name."""


# ── Default model names (single source of truth) ─────────────────────────────
DEFAULT_ANIMA_MODEL: str = "claude-sonnet-5-5"
DEFAULT_CONSOLIDATION_MODEL: str = DEFAULT_ANIMA_MODEL


class AnimaDefaults(BaseModel):
    """Concrete defaults applied when a per-anima field is None."""

    model: str = DEFAULT_ANIMA_MODEL
    fallback_model: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    background_model: str | None = None
    background_credential: str | None = None
    background_thinking_effort: str | None = None  # heartbeat/cron thinking effort override
    voice_thinking_effort: str | None = None  # voice chat thinking effort override
    max_tokens: int = 8192
    credential: str = "anthropic"
    context_threshold: float = 0.50
    context_absolute_ceiling: float = 0.75
    task_compaction_tokens: int = 0
    task_compaction_max: int = 6
    max_session_age_hours: float = 24.0
    conversation_history_threshold: float = 0.30
    execution_mode: str | None = None  # None = auto-detect from model
    supervisor: str | None = None
    speciality: str | None = None
    extra_mcp_servers: dict[str, dict] = Field(default_factory=dict)
    thinking: bool | None = None  # Extended thinking (Bedrock: reasoning_effort, Ollama: think)
    thinking_effort: str | None = None  # "low"/"medium"/"high"/"max" (default: "high")
    mode_s_auth: str | None = None  # Mode S auth: "max"|"api"|"bedrock"|"vertex"|None(=max)
    default_workspace: str = ""
    consolidation_enabled: bool = True
    heartbeat_enabled: bool = True  # 既定true。falseで定期heartbeatのみ無効化。メッセージ起因HB・cronは影響なし
    token_budget_monthly: int | None = None  # None = monthly token usage is unlimited


# ── Local LLM defaults ───────────────────────────────────────────────────────
DEFAULT_LOCAL_LLM_BASE_URL: str = "http://127.0.0.1:11434"
DEFAULT_LOCAL_LLM_MODEL: str = "ollama/qwen2.5-coder:14b"
DEFAULT_LOCAL_LLM_PRESETS: dict[str, str] = {
    "coding": "ollama/qwen2.5-coder:14b",
    "reasoning": "ollama/deepseek-r1:8b",
    "general": "ollama/glm4:9b",
}
DEFAULT_LOCAL_LLM_ROLE_PRESETS: dict[str, str] = {
    "engineer": "coding",
    "researcher": "reasoning",
    "manager": "reasoning",
    "writer": "general",
    "ops": "general",
    "general": "general",
}


class LocalLLMConfig(BaseModel):
    """User-facing defaults for locally hosted models.

    Historically this only described an Ollama endpoint. Deployments that put a
    LiteLLM/vLLM gateway in front of local GPUs serve OpenAI-protocol models
    instead, and those need a named credential (base URL + key) rather than the
    bare ``base_url`` an Ollama provider infers. Set ``credential`` to route
    through such a gateway; leave it empty for a plain Ollama endpoint.
    """

    base_url: str = DEFAULT_LOCAL_LLM_BASE_URL
    default_model: str = DEFAULT_LOCAL_LLM_MODEL
    credential: str = ""
    auto_apply_presets: bool = False
    presets: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_LOCAL_LLM_PRESETS))
    role_presets: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_LOCAL_LLM_ROLE_PRESETS))

    @model_validator(mode="after")
    def ensure_required_presets(self) -> LocalLLMConfig:
        merged_presets = dict(DEFAULT_LOCAL_LLM_PRESETS)
        for key, value in self.presets.items():
            if value:
                merged_presets[key] = value
        self.presets = merged_presets

        merged_role_presets = dict(DEFAULT_LOCAL_LLM_ROLE_PRESETS)
        for key, value in self.role_presets.items():
            if value in merged_presets:
                merged_role_presets[key] = value
        self.role_presets = merged_role_presets

        if not self.default_model:
            self.default_model = merged_presets["coding"]
        return self


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""

    enabled: bool = True
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_e5_prefix_enabled: bool = Field(
        default=False,
        description=(
            "When enabled, prefix query embeddings with embedding_query_prefix "
            "and indexed document embeddings with embedding_document_prefix. "
            "This is intended for E5-family ablations and requires re-indexing "
            "for document-prefix changes to take effect."
        ),
    )
    embedding_query_prefix: str = "query: "
    embedding_document_prefix: str = "passage: "
    embedding_max_seq_length: int = Field(
        default=2048,
        ge=0,
        description=(
            "Cap on the embedding model's max sequence length (tokens). "
            "Long-context models like ruri-v3 default to 8192, which blows up "
            "GPU activation memory during bulk encode. 0 = use model default."
        ),
    )
    use_gpu: bool = False
    min_retrieval_score: float = 0.3
    skill_match_min_score: float = 0.75
    repair_enabled: bool = True
    repair_error_threshold: int = 2
    repair_window_minutes: int = 5
    repair_cooldown_minutes: int = 60
    repair_max_consecutive_failures: int = 2
    repair_timeout_seconds: int = 1800
    repair_poll_interval_seconds: int = 5
    # Root staging rebuilds are CPU/IO-heavy; limit concurrent rebuilds.
    repair_max_concurrent: int = 1
    upsert_quarantine_failure_threshold: int = Field(default=3, ge=1)
    shared_check_ttl_seconds: float = Field(default=30.0, ge=0)
    shared_check_backoff_initial_seconds: float = Field(default=5.0, ge=0)
    shared_check_backoff_max_seconds: float = Field(default=300.0, ge=0)
    rerank_enabled: bool = True
    rerank_candidate_pool: int = 50
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    confidence_threshold: float = 0.35
    rrf_confidence_threshold: float = 0.02
    facts_extraction_enabled: bool = True
    fact_extraction_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description=(
            "Default timeout in seconds for LLM calls used by legacy atomic fact extraction; "
            "per-Anima status.json extraction_timeout overrides this value."
        ),
    )
    fact_extraction_max_tokens: int = Field(
        default=8192,
        ge=1024,
        description="Maximum output tokens for legacy atomic fact extraction LLM calls.",
    )
    facts_extraction_single_call: bool = Field(
        default=True,
        description="Extract legacy atomic facts and their entities in one LLM call when supported.",
    )
    facts_reconcile_enabled: bool = Field(
        default=True,
        description="Enable legacy atomic fact reconciliation before append; failures fall back to ADD.",
    )
    facts_reconcile_similarity_threshold: float = Field(
        default=0.82,
        description="Minimum facts vector similarity before strict LLM duplicate/contradiction/complement labeling.",
    )
    facts_reconcile_top_k: int = Field(
        default=5,
        description="Maximum similar active facts considered during legacy fact reconciliation.",
    )
    entity_registry_enabled: bool = True


class GPUConfig(BaseModel):
    """GPU device preferences for local ML components."""

    embedding_device: Literal["auto", "cuda", "cpu"] = "auto"
    reranker_device: Literal["auto", "cuda", "cpu"] = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_bulk_yield_batches: int = Field(
        default=5,
        ge=1,
        description="Maximum consecutive embedding batches bulk work may yield to waiting interactive work.",
    )


class FactEdgeTypeConfig(BaseModel):
    """Configurable semantic edge type for extracted facts."""

    name: str = Field(..., description="Upper snake case semantic edge type name")
    description: str = Field(..., description="Short explanation shown in extraction prompts")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip().upper()
        if not name:
            raise ValueError("Fact edge type name must not be empty")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError("Fact edge type name must be upper snake case")
        return name

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        description = value.strip()
        if not description:
            raise ValueError("Fact edge type description must not be empty")
        return description


class MemoryConfig(BaseModel):
    """Configuration for fact extraction."""

    fact_edge_types: list[FactEdgeTypeConfig] = Field(default_factory=list)


class PromptConfig(BaseModel):
    """Configuration for system prompt building."""

    injection_size_warning_chars: int = 2000
    identity_business_exclude_headings: list[str] = Field(
        default_factory=lambda: ["外見", "基本プロフィール", "Appearance", "Basic Profile"]
    )
    system_prompt_target_tokens: int = Field(default=6000, ge=2000)
    system_prompt_ceiling_pct: float = Field(default=0.35, gt=0.0, le=1.0)
    skill_catalog_router_enabled: bool = True
    skill_catalog_router_top_k: int = Field(default=5, ge=1)
    skill_catalog_router_min_score: float = Field(default=1.15, ge=0.0)
    skill_catalog_router_include_body: bool = True
    skill_catalog_router_dense_enabled: bool = True
    skill_catalog_router_dense_weight: float = Field(default=8.0, ge=0.0)
    skill_catalog_max_items: int = Field(default=3, ge=1)


class CompactBackgroundRecallConfig(BaseModel):
    """Per-trigger limits for compact-profile background memory recall."""

    related_knowledge_max_items: int = Field(default=3, ge=0)
    related_knowledge_max_tokens: int = Field(default=180, ge=0)
    episodes_max_items: int = Field(default=2, ge=0)
    episodes_max_tokens: int = Field(default=400, ge=0)
    recent_activity_max_items: int = Field(default=5, ge=0)
    recent_activity_max_tokens: int = Field(default=300, ge=0)


class PrimingConfig(BaseModel):
    """Configuration for priming layer (automatic memory retrieval)."""

    max_tokens: int = Field(default=2000, ge=200)
    channel_timeout_seconds: float = Field(default=60.0, ge=0.1)
    recent_facts_enabled: bool = True
    recent_facts_max_tokens: int = Field(default=500, ge=0)
    compact_background_recall_enabled: bool = True
    compact_background_recall: CompactBackgroundRecallConfig = Field(default_factory=CompactBackgroundRecallConfig)


class BackgroundReviewConfig(BaseModel):
    """Limits for asynchronous post-session memory review."""

    enabled: bool = True
    chat_every_user_turns: int = Field(default=10, ge=1)
    min_interval_minutes: int = Field(default=10, ge=0)
    max_per_day: int = Field(default=24, ge=1)
    max_input_bytes: int = Field(default=60 * 1024, ge=2048)
    max_writes: int = Field(default=3, ge=0)
    peer_profile_max_chars: int = Field(default=1500, ge=1)


class ConsolidationConfig(BaseModel):
    """Configuration for memory consolidation processes."""

    daily_enabled: bool = True
    weekly_distillation_enabled: bool = True
    synaptic_downscaling_enabled: bool = True
    skill_autolearn_enabled: bool = True
    curator_auto_apply_enabled: bool = False
    daily_time: str = "02:00"  # Format: HH:MM
    min_episodes_threshold: int = 1
    llm_model: str = DEFAULT_CONSOLIDATION_MODEL
    llm_credential: str = ""
    weekly_llm_model: str | None = None
    weekly_llm_credential: str | None = None
    llm_fallback_model: str | None = None
    llm_fallback_credential: str | None = None
    fact_reconcile_model: str | None = None
    fact_reconcile_credential: str | None = None
    episode_summary_max_input_bytes: int = Field(
        default=200 * 1024,
        ge=1024,
        description="Maximum UTF-8 prompt size for each daily episode-summary LLM call.",
    )
    episode_summary_backfill_days: int = Field(
        default=7,
        ge=1,
        description="Look back this many local days for unprocessed daily episode activity.",
    )
    episode_summary_backfill_max_days_per_run: int = Field(
        default=1,
        ge=0,
        description="Maximum older days to backfill during one daily consolidation (yesterday is separate).",
    )
    episode_summary_exclude_noop_cron: bool = Field(
        default=True,
        description="Exclude 'did nothing' cron executions from daily episode-summary input.",
    )
    ipc_timeout_base_seconds: int = Field(default=1800, ge=60)
    ipc_timeout_per_activity_entry_seconds: float = Field(default=4.0, ge=0.0)
    ipc_timeout_per_episode_seconds: float = Field(default=120.0, ge=0.0)
    ipc_timeout_max_seconds: int = Field(default=7200, ge=60)
    weekly_ipc_timeout_seconds: int = Field(default=3600, ge=60)
    max_concurrent_animas: int = Field(
        default=3,
        description="Maximum number of Anima daily/weekly consolidations to run concurrently.",
    )
    weekly_enabled: bool = False
    weekly_time: str = "sun:03:00"  # Format: day:HH:MM
    indexing_enabled: bool = True  # Daily RAG indexing toggle
    indexing_time: str = "04:00"  # Format: HH:MM
    knowledge_self_correction_enabled: bool = True
    knowledge_self_correction_max_reconsolidation_files: int = Field(default=5, ge=0)
    knowledge_self_correction_timeout_seconds: int = Field(default=300, ge=1)
    fact_extraction_chunk_chars: int = Field(
        default=12000,
        ge=0,
        description="Maximum characters per atomic-fact-extraction chunk during "
        "consolidation. 0 (or negative) disables splitting.",
    )
    post_processing_cooldown_seconds: int = Field(default=30, ge=0)
    inactivity_skip_enabled: bool = True
    inactivity_days: int = Field(default=7, ge=1)
    episode_summary_input_profile: Literal["full", "compact"] = "compact"
    episode_summary_cron_digest_min_runs: int = Field(default=6, ge=1)
    episode_summary_cron_digest_max_notable_runs: int = Field(default=5, ge=0)
    episode_summary_tool_use_max_bytes: int = Field(default=300, ge=0)
    # Bash tool_use bodies are cut further for dates summarised with compact_v2.
    episode_summary_bash_tool_use_max_bytes: int = Field(default=120, ge=0)
    episode_summary_error_tail_bytes: int = Field(default=300, ge=0)
    episode_summary_max_output_tokens: int = Field(default=4096, ge=1)
    live_fact_extraction_enabled: bool = True
    live_fact_model: str | None = None
    live_fact_credential: str | None = None
    live_fact_min_input_chars: int = Field(default=200, ge=0)
    live_fact_max_input_chars: int = Field(default=24000, ge=0)
    live_fact_debounce_seconds: int = Field(default=120, ge=0)


class HelperModelFallback(BaseModel):
    """A fallback model and its optional named credential for one helper role."""

    model: str = Field(..., min_length=1, description="Fallback model identifier.")
    credential: str | None = Field(default=None, description="Credential name for the fallback model.")


class HelperModelRole(BaseModel):
    """Model selection and one-shot policy for a single helper role."""

    model: str | None = Field(default=None, description="Primary helper model identifier.")
    credential: str | None = Field(default=None, description="Credential name for the primary model.")
    fallbacks: list[HelperModelFallback] = Field(default_factory=list, description="Ordered explicit fallback models.")
    allow_agent_sdk_fallback: bool = Field(
        default=False,
        description="Allow one-shot calls for this role to fall back to the Claude Agent SDK.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum output tokens for one-shot calls made by this role.",
    )


class HelperModelsConfig(BaseModel):
    """Named helper-model roles plus a shared fallback role configuration."""

    episode_summary: HelperModelRole | None = None
    fact_extraction: HelperModelRole | None = None
    fact_reconcile: HelperModelRole | None = None
    weekly_consolidation: HelperModelRole | None = None
    project_consolidation: HelperModelRole | None = None
    conversation_compression: HelperModelRole | None = None
    distillation: HelperModelRole | None = None
    reconsolidation: HelperModelRole | None = None
    asset_reconcile: HelperModelRole | None = None
    meeting_summary: HelperModelRole | None = None
    default: HelperModelRole = Field(default_factory=HelperModelRole)


class ImageGenConfig(BaseModel):
    """Configuration for image generation and style consistency."""

    backend: Literal["api", "diffusers", "atlascloud"] = "api"
    image_style: Literal["anime", "realistic"] = "realistic"
    prefer_codex: bool = True  # codex CLIがあれば画像生成に最優先で使う
    style_reference: str | None = None  # Path to organization-wide style reference image
    style_prefix: str = ""  # Common style tags prepended to character prompt
    style_suffix: str = ""  # Common style tags appended to character prompt
    negative_prompt_extra: str = ""  # Extra tags added to negative prompt
    vibe_strength: float = 0.6  # Vibe Transfer strength (0.0-1.0)
    vibe_info_extracted: float = 0.8  # Vibe Transfer information extraction (0.0-1.0)
    enable_3d: bool = True  # Enable 3D model generation (Meshy API)
    pixel_base_sheet: str | None = None  # Base pixel sheet; None uses the bundled sample_01.png
    pixel_concurrency: int = 4  # Parallel codex image_gen calls for pixel sheet rows
    diffusers_text2img_model: str = "auto"
    diffusers_img2img_model: str = "auto"
    diffusers_text2img_model_realistic: str = ""  # Override for realistic style
    diffusers_text2img_model_anime: str = ""  # Override for anime style
    diffusers_device: Literal["auto", "cuda", "cpu"] = "auto"
    diffusers_torch_dtype: Literal["auto", "float16", "float32", "bfloat16"] = "auto"
    diffusers_local_files_only: bool = True
    diffusers_num_inference_steps: int = 28
    diffusers_img2img_strength: float = 0.55
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_scale: float = 0.6  # IP-Adapter face reference blend weight (0.0-1.0)


class NotificationChannelConfig(BaseModel):
    """Configuration for a single human notification channel.

    For ``type="chatwork"``, when ``config.api_token_env`` is omitted the
    channel falls back to ``CHATWORK_API_TOKEN__kotoha`` (system notification
    default identity: kotoha). Set ``api_token_env`` explicitly to override.
    """

    type: str  # "slack", "line", "telegram", "chatwork", "ntfy"
    enabled: bool = True
    config: dict[str, Any] = {}


class HumanNotificationConfig(BaseModel):
    """Global configuration for human notification from top-level Animas."""

    enabled: bool = False
    channels: list[NotificationChannelConfig] = []
    web_ui: bool = True  # built-in channel: push call_human into the Web UI chat


class InteractionConfig(BaseModel):
    """Configuration for interactive call_human approvals."""

    default_approver_ids: list[str] = Field(
        default_factory=list,
        description="Default Slack user IDs merged with per-call call_human allowed_users.",
    )
    web_base_url: str = ""

    @field_validator("default_approver_ids", mode="before")
    @classmethod
    def _coerce_default_approver_ids(cls, v: object) -> list[str]:
        """Accept legacy dict[str, list[str]] from older config.json."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, dict):
            out: list[str] = []
            for ids in v.values():
                if isinstance(ids, list):
                    out.extend(str(x).strip() for x in ids if str(x).strip())
            return out
        return []


class UserAliasConfig(BaseModel):
    """External user contact information for outbound message routing."""

    slack_user_id: str = ""
    chatwork_room_id: str = ""
    discord_user_id: str = ""
    # Allow explicit sends (send_message etc.) to resolve this alias to an
    # external DM. Off by default: aliases also serve inbound trust elevation,
    # which must not silently reroute internal replies to external platforms.
    outbound_dm: bool = False


class ExternalMessagingChannelConfig(BaseModel):
    """Configuration for a single external messaging platform."""

    enabled: bool = False
    mode: str = "socket"  # "socket" | "webhook"
    anima_mapping: dict[str, str] = {}  # channel_id → anima_name ("" = ignore this channel)
    default_anima: str = ""  # fallback anima for unmapped channels
    app_id_mapping: dict[str, str] = {}  # api_app_id → anima_name (per-Anima webhook routing)
    auto_response: bool = False  # auto-post LLM responses back to originating platform
    board_mapping: dict[str, str] = {}  # channel_id → animaworks_board_name (auto-populated)
    board_outbound_sync: list[str] = []  # board names to sync outbound to this platform (whitelist)
    board_outbound_sync_all: bool = False  # when whitelist is empty, opt in to syncing all mapped boards
    guild_id: str = ""  # Discord guild snowflake ID (Discord only)
    channel_members: dict[str, list[str]] = {}  # channel_id → [anima_name, ...] (Discord only)
    default_channel_company: str = ""  # company for auto-created boards (empty = no attribution)

    def resolve_anima(self, channel_id: str) -> str:
        """Return the anima that handles *channel_id*, or ``""`` to ignore it.

        An explicit entry in ``anima_mapping`` always wins, including an empty
        value, which opts the channel out of routing entirely.  Only channels
        with no entry at all fall back to ``default_anima``.
        """
        if channel_id in self.anima_mapping:
            return self.anima_mapping[channel_id] or ""
        return self.default_anima


class ZoomRTMSConfig(BaseModel):
    """Configuration for Zoom RTMS (Real-Time Media Streams) ingestion."""

    enabled: bool = False
    default_anima: str = ""  # fallback anima for unmapped meetings
    meeting_mapping: dict[str, str] = {}  # meeting_id → anima_name
    chunk_interval_seconds: int = 300  # flush interval for buffered transcript
    chunk_max_chars: int = 4000  # max chars per chunk (flush on whichever comes first)


class GitHubWebhookConfig(BaseModel):
    """Configuration for GitHub webhook-driven PR dispatch.

    2026-09 teardown: the gateway only sends one notification per PR event to
    ``dispatcher_anima`` and no longer creates tasks or posts to GitHub.  The
    old reviewer/implementer routing and multi-model review-pass fields were
    dropped from this model; this class has no ``extra="forbid"``, so an
    existing ``config.json`` still carrying those keys loads fine (pydantic
    silently ignores unknown keys by default).
    """

    enabled: bool = False
    repos: list[str] = Field(default_factory=list)
    dispatcher_anima: str = "rin"
    bot_login: str = ""
    # Dedicated review-bot GitHub login (e.g. animaworks-reviewer).
    # Treated like bot_login for comment exclusion.
    reviewer_login: str = ""
    quiet_seconds: float = Field(default=180, ge=0)
    # Thin out auto-detected bot noise (logins ending in "[bot]"): drop
    # notifications whose body is empty or is only a "Review thread
    # resolved" auto-reply, while still delivering bots with a real body.
    # Independent of bot_login/reviewer_login (those are filtered outright).
    drop_bot_noise: bool = True


class EventExportConfig(BaseModel):
    """Best-effort export of runtime activity and token usage events."""

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    event_types: list[str] | None = None
    include_token_usage: bool = True
    max_retries: int = Field(default=8, ge=0)
    backoff_base_seconds: float = Field(default=2.0, ge=0)
    spool_max_mb: int = Field(default=64, ge=0)


class ExternalMessagingConfig(BaseModel):
    """Configuration for external messaging integration (inbound + outbound)."""

    preferred_channel: str = "slack"  # "slack" | "chatwork" | "discord"
    user_aliases: dict[str, UserAliasConfig] = {}  # alias → contact info
    # Redirect chat-UI replies to the sender's external DM (Slack etc.) when
    # the sender matches a user_alias. Off by default: aliases also serve
    # inbound trust elevation, which must not force outbound redirection.
    chat_dm_redirect: bool = False
    slack: ExternalMessagingChannelConfig = ExternalMessagingChannelConfig()
    chatwork: ExternalMessagingChannelConfig = ExternalMessagingChannelConfig()
    discord: ExternalMessagingChannelConfig = ExternalMessagingChannelConfig()
    zoom: ZoomRTMSConfig = ZoomRTMSConfig()


class ExternalTasksSourcesConfig(BaseModel):
    """Per-source enable flags for external tasks collection."""

    github: bool = True
    slack: bool = True
    chatwork: bool = True
    gmail: bool = True


class ExternalTasksConfig(BaseModel):
    """Configuration for the external tasks dashboard widget collector.

    Enabled by default; sources without credentials are simply reported as
    ``unavailable`` by the collector, so this is safe on fresh installs.
    """

    enabled: bool = True
    interval_minutes: int = 5
    sources: ExternalTasksSourcesConfig = Field(default_factory=ExternalTasksSourcesConfig)


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

    session_ttl_days: int | None = 90  # None = unlimited
    ipc_stream_timeout: int = 60  # per-chunk timeout in seconds
    keepalive_interval: int = 30  # keep-alive emission interval in seconds
    runner_liveness_timeout: int = Field(default=900, ge=1)
    chat_max_concurrency: int = Field(default=1, ge=1, le=10)
    anima_startup_ready_timeout: int = Field(default=120, ge=1)
    anima_stop_timeout: float = Field(default=60.0, gt=0)
    # Matches the 600-second TaskExec drain; 0 skips waiting for active work.
    restart_drain_timeout_seconds: int = Field(default=600, ge=0)
    health_check_warmup_seconds: int = Field(default=300, ge=0)
    runner_warmup_seconds: int = Field(default=180, ge=0)
    spawn_timeout: int = Field(default=300, ge=1)
    supervisor_respawn_max_retries: int = Field(
        default=3, ge=1
    )  # consecutive failures before FAILED display; auto-recovery continues after
    supervisor_respawn_retry_interval_seconds: float = Field(default=30.0, ge=0.0)  # base backoff interval (seconds)
    supervisor_respawn_backoff_max_seconds: float = Field(default=1800.0, ge=0.0)  # max backoff (seconds)
    stream_checkpoint_enabled: bool = True  # save tool results during streaming
    stream_retry_max: int = 3  # max automatic retries on stream disconnect
    stream_retry_delay_s: float = 5.0  # delay between retries (seconds)
    llm_num_retries: int = 3  # retries for LLM API calls (429/5xx/network)
    ollama_keep_alive: str = (
        ""  # Ollama model keep-alive after a request (e.g. "5m", "0", "1h"); empty = Ollama default
    )
    ollama_total_timeout: int = 0  # Hard upper bound (seconds) on a single Ollama generation call; 0 = unlimited
    media_proxy: MediaProxyConfig = MediaProxyConfig()
    base_path: str = ""  # Reverse proxy sub-path (e.g. "/app"); empty = root deploy
    internal_api_auth: Literal["off", "log", "enforce"] = "log"  # /api/internal/* caller verification

    @model_validator(mode="after")
    def _validate_intervals(self) -> ServerConfig:
        if self.keepalive_interval >= self.ipc_stream_timeout:
            raise ValueError(
                f"keepalive_interval ({self.keepalive_interval}) must be "
                f"less than ipc_stream_timeout ({self.ipc_stream_timeout})"
            )
        return self


class LlmRateGuardConfig(BaseModel):
    """Cross-process LLM rate guard settings (fleet-wide circuit breaker)."""

    enabled: bool = True
    default_block_seconds: int = Field(default=60, ge=0)
    max_block_seconds: int = Field(default=600, ge=0)
    quota_block_seconds: int = Field(default=1800, ge=0)
    max_quota_block_seconds: int = Field(default=14400, ge=0)


class MCPConfig(BaseModel):
    """Configuration for the Mode S aw MCP server tool exposure."""

    # Limit the advertised aw MCP tool set by trigger: interactive triggers
    # (chat / inbox / cron / task) skip the skill-management tools, while
    # heartbeat / consolidation still receive the full set.  Set False to
    # always expose every tool (previous behaviour).
    trigger_scoped_tools: bool = True


class BackgroundToolConfig(BaseModel):
    """Per-tool background execution threshold."""

    threshold_s: int = 30


class BackgroundTaskConfig(BaseModel):
    """Configuration for background tool execution."""

    enabled: bool = True
    shutdown_drain_seconds: float = Field(default=600.0, ge=0)
    eligible_tools: dict[str, BackgroundToolConfig] = {
        "generate_character_assets": BackgroundToolConfig(threshold_s=30),
        "generate_fullbody": BackgroundToolConfig(threshold_s=30),
        "generate_bustup": BackgroundToolConfig(threshold_s=30),
        "generate_icon": BackgroundToolConfig(threshold_s=30),
        "generate_chibi": BackgroundToolConfig(threshold_s=30),
        "generate_3d_model": BackgroundToolConfig(threshold_s=30),
        "generate_rigged_model": BackgroundToolConfig(threshold_s=30),
        "generate_animations": BackgroundToolConfig(threshold_s=30),
        "local_llm": BackgroundToolConfig(threshold_s=60),
        "run_command": BackgroundToolConfig(threshold_s=60),
    }
    result_memory_retention_minutes: int = Field(default=60, ge=0)  # in-process result cache
    max_completed_tasks_in_memory: int = Field(default=200, ge=0)
    worker_pool_size: int = Field(default=1, ge=1, le=10)
    # The task-control keys retired with the teardown are deliberately absent.
    # This model ignores unknown keys, so an older config.json that still
    # carries them loads without error.


def resolve_background_worker_pool_size(
    anima_dir: Path,
    default: int | None = None,
) -> int:
    """Resolve the TaskExec worker count, including a per-Anima override.

    ``status.json`` may set ``background_worker_pool_size`` to opt one Anima
    into a different pool size. Invalid or unreadable overrides are ignored so
    a damaged status file cannot prevent the Anima from starting.
    """
    if default is None:
        try:
            from core.config.io import load_config

            default = load_config().background_task.worker_pool_size
        except Exception:
            logger.debug(
                "Failed to load background worker pool default for %s",
                anima_dir.name,
                exc_info=True,
            )
            default = 1

    if isinstance(default, bool) or not isinstance(default, int) or not 1 <= default <= 10:
        default = 1

    status_path = anima_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

    if not isinstance(status, dict) or "background_worker_pool_size" not in status:
        return default

    override = status["background_worker_pool_size"]
    if isinstance(override, int) and not isinstance(override, bool) and 1 <= override <= 10:
        return override

    logger.warning(
        "Ignoring invalid background_worker_pool_size=%r for anima %s; using %d",
        override,
        anima_dir.name,
        default,
    )
    return default


def resolve_chat_max_concurrency(
    anima_dir: Path,
    default: int | None = None,
) -> int:
    """Resolve the isolated chat limit, including a per-Anima status override.

    ``status.json`` may set ``chat_max_concurrency`` to allow concurrent chat
    jobs for different threads. Invalid or unreadable overrides are ignored.
    """
    if default is None:
        try:
            from core.config.io import load_config

            default = load_config().server.chat_max_concurrency
        except Exception:
            logger.debug(
                "Failed to load chat max concurrency default for %s",
                anima_dir.name,
                exc_info=True,
            )
            default = 1

    if isinstance(default, bool) or not isinstance(default, int) or not 1 <= default <= 10:
        default = 1

    status_path = anima_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

    if not isinstance(status, dict) or "chat_max_concurrency" not in status:
        return default

    override = status["chat_max_concurrency"]
    if isinstance(override, int) and not isinstance(override, bool) and 1 <= override <= 10:
        return override

    logger.warning(
        "Ignoring invalid chat_max_concurrency=%r for anima %s; using %d",
        override,
        anima_dir.name,
        default,
    )
    return default


class ActivityLogConfig(BaseModel):
    """Configuration for activity log rotation."""

    rotation_enabled: bool = True
    rotation_mode: Literal["size", "time", "both"] = "size"
    max_size_mb: int = Field(default=1024, ge=0)  # per-anima total, default 1GB
    max_file_size_mb: int = Field(default=100, ge=0)  # per-file bloat trigger; 0 disables
    max_age_days: int = Field(default=7, ge=0)  # mode="time"|"both" で使用
    rotation_time: str = "05:00"  # 実行時刻 (configured TZ)


class LoggingConfig(BaseModel):
    """Configuration for the logging subsystem."""

    redaction_enabled: bool = True  # Mask secrets in log output; disable for raw-log debugging.


class HousekeepingConfig(BaseModel):
    """Configuration for periodic disk cleanup."""

    enabled: bool = True
    run_time: str = "05:30"

    prompt_log_retention_days: int = 3
    daemon_log_max_size_mb: int = 50
    daemon_log_keep_generations: int = 5
    anima_log_retention_days: int = Field(default=30, ge=1)
    anima_log_total_max_size_mb: int = Field(default=200, ge=1)
    frontend_log_backup_count: int = 7
    dm_log_archive_retention_days: int = 30
    cron_log_retention_days: int = 14
    shortterm_retention_days: int = 7
    shortterm_archive_retention_days: int = Field(default=30, ge=1)
    shortterm_thread_gc_days: int = Field(default=30, ge=1)
    facts_lock_stale_hours: int = Field(default=24, ge=1)
    curator_report_retention_days: int = Field(default=30, ge=1)
    task_results_retention_days: int = 7
    corrupt_vectordb_keep_generations: int = Field(default=2, ge=0)
    tmp_retention_days: int = Field(default=14, ge=1)
    backup_retention_days: int = Field(default=90, ge=1)
    codex_log_max_size_mb: int = Field(default=200, ge=1)
    codex_tmp_retention_hours: int = Field(default=12, ge=1)
    codex_sessions_retention_days: int = Field(default=7, ge=1)
    anima_scratch_retention_hours: int = Field(default=24, ge=1)
    shared_worktrees_idle_hours: int = Field(default=24, ge=1)
    workspace_sweep_enabled: bool = True
    anima_tmp_gitdirs_retention_days: int = Field(default=14, ge=1)
    anima_local_log_retention_days: int = Field(default=30, ge=1)
    pending_processing_stale_hours: int = Field(default=24, ge=1)
    background_running_stale_hours: int = Field(default=48, ge=1)
    current_state_stale_hours: int = Field(default=24, ge=1)
    suppressed_messages_max_size_mb: int = Field(default=10, ge=1)
    suppressed_messages_keep_generations: int = Field(default=5, ge=1)
    sdk_bash_injection_max_size_mb: int = Field(default=10, ge=1)
    archive_superseded_retention_days: int = Field(default=7, ge=1)
    archive_versions_keep_per_file: int = Field(default=5, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _preserve_legacy_sdk_log_limit(cls, values: Any) -> Any:
        """Carry forward old configs until they set the dedicated SDK limit."""
        if (
            isinstance(values, dict)
            and "sdk_bash_injection_max_size_mb" not in values
            and "suppressed_messages_max_size_mb" in values
        ):
            return {**values, "sdk_bash_injection_max_size_mb": values["suppressed_messages_max_size_mb"]}
        return values


class InboxConfig(BaseModel):
    """Configuration for shared inbox processing and stale-file hygiene."""

    inbox_quiet_seconds: float = Field(default=90.0, ge=0)
    inbox_quiet_max_wait_seconds: float = Field(default=300.0, ge=0)
    ttl_hours: float = Field(default=24.0, gt=0)
    expired_retention_days: int = Field(default=7, ge=1)
    processed_retention_days: int = Field(default=30, ge=1)
    quarantine_retention_days: int = Field(default=30, ge=1)


class HeartbeatConfig(BaseModel):
    """Heartbeat scheduling settings."""

    interval_minutes: int = Field(
        default=30, ge=1, le=1440
    )  # heartbeat interval (config-driven, not parsed from HEARTBEAT.md)
    current_state_max_chars: int = Field(
        default=8000,
        ge=0,
        description="Max chars for current_state.md before trim; 0 = disabled",
    )
    current_state_cleanup_chars: int = Field(
        default=2000,
        ge=0,
        description="Soft cleanup threshold for current_state.md; 0 = 80% of current_state_max_chars",
    )
    heartbeat_md_max_bytes: int = Field(
        default=8000,
        ge=0,
        description=(
            "Max bytes of HEARTBEAT.md before a compaction instruction is "
            "injected into the heartbeat prompt; 0 = disabled"
        ),
    )
    recent_dialogue_max_age_hours: int = Field(
        default=6,
        ge=0,
        description=(
            "Include recent chat dialogue in heartbeat context only when the "
            "last turn is younger than this many hours; 0 = always include"
        ),
    )
    soft_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Seconds before injecting a wrap-up system-reminder into the HB session",
    )
    hard_timeout_seconds: int = Field(
        default=0,
        ge=0,
        le=7200,
        description="Seconds before forcefully terminating the HB session; 0 = disabled",
    )
    cron_followup_repeat_suppress_hours: float = Field(
        default=6.0,
        ge=0,
        description=(
            "Suppress a command cron's LLM follow-up when its output is identical to the "
            "previous follow-up for this many hours; 0 disables suppression"
        ),
    )

    @model_validator(mode="after")
    def _validate_soft_lt_hard(self) -> HeartbeatConfig:
        if self.hard_timeout_seconds and self.soft_timeout_seconds >= self.hard_timeout_seconds:
            raise ValueError(
                f"soft_timeout_seconds ({self.soft_timeout_seconds}) must be "
                f"less than hard_timeout_seconds ({self.hard_timeout_seconds})"
            )
        return self

    default_model: str | None = None  # global background model for heartbeat/cron (None = use main model)
    enable_read_ack: bool = (
        False  # Send read-receipt ACK to message senders (disabled by default to prevent gratitude loops)
    )
    delegation_dm_enabled: bool = Field(
        default=True,
        description=(
            "delegate_task already writes the pending descriptor for the target; "
            "the DM only wakes an extra inbox run. Set false to skip it."
        ),
    )
    idle_compaction_minutes: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Minutes after last stream end to trigger idle auto-compaction",
    )
    resolved_interaction_reminder_hours: int = Field(
        default=48,
        ge=0,
        description=(
            "Hours to inject resolved-approval reminders into the system prompt; 0 disables the reminder section"
        ),
    )


# ── Voice Chat Config ───────────────────────────────────────────────────────


class PhoneConfig(BaseModel):
    """Twilio-backed phone channel for one Anima."""

    enabled: bool = False
    anima: str = "aoi"
    public_base_url: str = "https://zoomhook.kk-a.jp"
    from_number: str = "+16073257655"
    owner_numbers: list[str] = Field(default_factory=lambda: ["+819060943763"])
    from_person: str = "human"
    thread_id: str = "phone"
    pin_vault_key: str = "AOI_PHONE_PIN"
    account_sid_vault_key: str = "TWILIO_ACCOUNT_SID"
    auth_token_vault_key: str = "TWILIO_AUTH_TOKEN"
    alert_max_attempts: int = Field(default=3, ge=1)
    alert_retry_interval_sec: float = Field(default=120, ge=0)
    turn_timeout_sec: float = Field(
        default=300,
        gt=0,
        description="Legacy Gather/poll timeout; unused by the Media Streams phone conversation.",
    )
    turn_end_silence_ms: int = Field(default=1200, ge=300, le=5000)


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


class IrodoriConfig(BaseModel):
    """Irodori-TTS HTTP API connection settings."""

    base_url: str = "http://localhost:7861"


class GeminiTTSVoiceConfig(BaseModel):
    """Gemini API TTS settings (key: env var, then vault ``shared`` section)."""

    model: str = "gemini-3.8-flash-tts"
    api_key_env: str = "GEMINI_API_KEY"
    vault_key: str = "GEMINI_API_KEY"
    chunk_seconds: float = 1.0


class VoiceConfig(BaseModel):
    """Voice chat configuration."""

    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "default"
    stt_language: str | None = None
    stt_refine_enabled: bool = False
    default_tts_provider: str = "voicevox"
    front_model: str | None = None
    """voice front lane model (e.g. ``openai/qwen3.6-35b-a3b``). None = legacy path."""
    front_api_base: str | None = None
    """OpenAI-compatible base URL for the voice front lane. None = legacy path."""
    proactive_enabled: bool = True
    """Proactively speak up after sustained silence (requires front lane). Opt-out."""
    proactive_initial_delay_sec: float = 10.0
    """Silence seconds between proactive utterances."""
    proactive_lead_sec: float = 5.0
    """Start the next monologue this many seconds before the current playback ends
    (so speech is continuous but never more than one utterance is queued)."""
    notify_delegations_on_web_disconnect: bool = False
    """Send pending voice-front delegation results to humans when a Web voice session closes."""
    voicevox: VoicevoxConfig = VoicevoxConfig()
    elevenlabs: ElevenLabsVoiceConfig = ElevenLabsVoiceConfig()
    style_bert_vits2: StyleBertVits2Config = StyleBertVits2Config()
    irodori: IrodoriConfig = IrodoriConfig()
    gemini: GeminiTTSVoiceConfig = GeminiTTSVoiceConfig()


# ── UI Config ────────────────────────────────────────────────────────────────


class UIConfig(BaseModel):
    """UI appearance and theme settings."""

    theme: str = "default"
    demo_mode: bool = False


class CLIConfig(BaseModel):
    """Command-line client settings."""

    # Anima opened by a bare ``animaworks`` / ``animaworks chat``.
    # Empty = the single enabled top-level anima, if there is exactly one.
    default_anima: str = ""


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

    name: str = ""
    pattern: str
    reason: str


class GlobalCommandsDeny(BaseModel):
    """Global command deny configuration."""

    deny: list[GlobalDenyPattern] = Field(default_factory=list)


class SdkBashInjectionConfig(BaseModel):
    """Mode S Bash injection rollout configuration."""

    mode: Literal["off", "log", "enforce"] = "log"


class GlobalPermissionsConfig(BaseModel):
    """Global permissions loaded from permissions.global.json.

    Applies to ALL Animas.  Loaded once at server startup, cached in memory.
    Runtime modifications to the on-disk file are auto-reverted.
    """

    version: int = 1
    injection_patterns: list[GlobalDenyPattern] = Field(default_factory=list)
    sdk_bash_injection: SdkBashInjectionConfig = Field(default_factory=SdkBashInjectionConfig)
    commands: GlobalCommandsDeny = Field(default_factory=GlobalCommandsDeny)


# ── Per-Anima Permissions Config ──────────────────────────────────────────────


def command_deny_matches(denied: str, segment: str, cmd_base: str) -> bool:
    """Match one per-anima ``commands.deny`` entry against a command segment.

    Plain entries are substrings (``"gh pr merge"``).  An entry prefixed with
    ``re:`` is a regex, so a rule can cover every spelling of a flag
    (``re:\\brm\\s+(-\\w*[rR]|--recursive)`` catches ``rm -r``/``-fr``/``-R``,
    where the substring ``"rm -rf"`` was bypassed by ``rm -r``).
    """
    if denied.startswith("re:"):
        try:
            return re.search(denied[3:], segment) is not None
        except re.error:
            logger.warning("Invalid regex in commands.deny: %r", denied)
            return False
    return denied in cmd_base or denied in segment


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
    file_roots_readonly: list[str] = Field(default_factory=list)
    file_roots_denied: list[str] = Field(default_factory=list)
    commands: CommandsPermission = Field(default_factory=CommandsPermission)
    external_tools: ExternalToolsPermission = Field(default_factory=ExternalToolsPermission)
    tool_creation: ToolCreationPermission = Field(default_factory=ToolCreationPermission)

    @field_validator("file_roots_denied")
    @classmethod
    def _validate_file_roots_denied(cls, roots: list[str]) -> list[str]:
        """Require unambiguous absolute deny roots and store canonical paths."""
        normalized: list[str] = []
        for root in roots:
            if any(char in root for char in "*?["):
                raise ValueError(f"file_roots_denied does not support glob patterns: {root!r}")
            path = Path(root)
            if not path.is_absolute():
                raise ValueError(f"file_roots_denied entries must be absolute paths: {root!r}")
            try:
                normalized.append(str(path.resolve()))
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"file_roots_denied entry cannot be resolved: {root!r}") from exc
        return normalized


def load_permissions(anima_dir: Path, *, read_only: bool = False) -> PermissionsConfig:
    """Load permissions from permissions.json, with a read-only legacy fallback.

    Resolution order:
      1. permissions.json exists -> load and validate
      2. permissions.md only -> parse without writes for read-only/worker calls;
         root and offline CLI calls migrate and return config
      3. Neither exists -> return default (open)
      4. Existing but unreadable/invalid permissions.json -> raise (fail closed)

    Once ``permissions.json`` exists, the whole document is a security
    boundary.  Read, JSON parsing, and schema validation failures therefore
    propagate instead of silently replacing the configured policy with open
    defaults.  Only a genuinely absent file retains the legacy open default.
    """
    json_path = anima_dir / "permissions.json"
    md_path = anima_dir / "permissions.md"

    try:
        json_path.lstat()
    except FileNotFoundError:
        json_exists = False
    except OSError:
        logger.error("Failed to stat permissions.json at %s — refusing fail-open fallback", json_path, exc_info=True)
        raise
    else:
        json_exists = True

    if json_exists:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            config = PermissionsConfig.model_validate(data)
            version = data.get("version")
            if version is not None and version != 1:
                logger.warning("permissions.json version %s is unknown; using known fields only", version)
            return config
        except Exception as exc:
            logger.error(
                "Failed to load permissions.json at %s — refusing fail-open fallback: %s",
                json_path,
                exc,
            )
            raise

    if md_path.is_file():
        from core.config.migrate import migrate_permissions_md_to_json, parse_permissions_md
        from core.platform.pid import is_server_running
        from core.platform.process_role import get_process_role

        role = get_process_role()
        if (
            read_only
            or role in {"anima", "task_runner", "mcp"}
            or (role == "cli" and is_server_running(anima_dir.parent.parent))
        ):
            return parse_permissions_md(anima_dir)
        return migrate_permissions_md_to_json(anima_dir)

    return PermissionsConfig()


def _format_permissions_for_prompt(config: PermissionsConfig, anima_name: str) -> str:
    """Render only permission constraints that differ from open defaults."""
    lines: list[str] = []
    if sys.platform == "win32":
        lines.extend(
            [
                "- Runtime: native Windows environment (not a Linux container and not WSL unless explicitly stated)",
                "- File reading is available via read_file for absolute paths and read_memory_file for files inside your own memory directory",
                "- Command execution runs through a PowerShell-compatible shell via execute_command; do not assume Bash-only behavior unless a command actually fails",
            ]
        )
    if config.file_roots != ["/"]:
        if not config.file_roots and not config.file_roots_readonly:
            lines.append("- File access: own directory and shared framework directories only")
        else:
            if config.file_roots:
                lines.append(f"- Read/write access: {', '.join(config.file_roots)}")
            if config.file_roots_readonly:
                lines.append(f"- Read-only access: {', '.join(config.file_roots_readonly)}")
    if config.file_roots_denied:
        lines.append(f"- Denied file access (read/write; overrides all grants): {', '.join(config.file_roots_denied)}")
    if not config.commands.allow_all:
        if config.commands.allow:
            lines.append(f"- Allowed commands: {', '.join(config.commands.allow)}")
        else:
            lines.append("- Commands: none allowed")
    if config.commands.deny:
        lines.append(f"- Additionally denied commands: {', '.join(config.commands.deny)}")
    if not config.external_tools.allow_all:
        if config.external_tools.allow:
            lines.append(f"- Allowed external tools: {', '.join(config.external_tools.allow)}")
        else:
            lines.append("- External tools: none allowed")
    if config.external_tools.deny:
        lines.append(f"- Denied external tools: {', '.join(config.external_tools.deny)}")
    if not config.tool_creation.personal:
        lines.append("- Personal tool creation: not allowed")
    if config.tool_creation.shared:
        lines.append("- Shared tool creation: allowed")
    if not lines:
        return ""
    return f"## Permissions: {anima_name}\n" + "\n".join(lines)


# ── Main Config ─────────────────────────────────────────────────────────────


class SkillPromotionConfig(BaseModel):
    success_count_threshold: int = Field(default=3, ge=1)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    failure_count_max: int = Field(default=1, ge=0)
    last_used_within_days: int = Field(default=180, ge=1)


class SkillCronConfig(BaseModel):
    max_skill_chars: int = Field(default=6000, ge=0)
    max_total_chars: int = Field(default=12000, ge=0)
    allow_warn_caution: bool = False
    allow_destructive: bool = False
    allow_external_send: bool = False


class ExternalSkillRoot(BaseModel):
    """A read-only, engine-owned skill root scanned directly by SkillIndex.

    External roots are scanned in place (no copy/sync) and are read-only
    from animaworks' perspective. List ordering equals the precedence when
    same-name skills collide.
    """

    path: str  # may contain ``~``; expanded before use
    engine: str  # matches ^[a-z][a-z0-9-]*$
    trust_level: str = "trusted"
    enabled: bool = True

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("engine must be a string")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", value):
            raise ValueError(f"engine {value!r} must match ^[a-z][a-z0-9-]*$")
        return value


class SkillsConfig(BaseModel):
    promotion: SkillPromotionConfig = SkillPromotionConfig()
    cron: SkillCronConfig = SkillCronConfig()
    external_roots: list[ExternalSkillRoot] = Field(
        default_factory=lambda: [
            ExternalSkillRoot(path="~/.claude/skills", engine="claude"),
            ExternalSkillRoot(path="~/.codex/skills", engine="codex"),
            ExternalSkillRoot(path="~/.grok/skills", engine="grok"),
            ExternalSkillRoot(path="~/.agents/skills", engine="agents"),
        ]
    )


class ChatworkToolConfig(BaseModel):
    grants: dict[str, dict[str, str]] = {}


class AnimaWorksConfig(BaseModel):
    version: int = 1
    setup_complete: bool = False
    locale: str = "ja"
    system: SystemConfig = SystemConfig()
    credentials: dict[str, CredentialConfig] = {"anthropic": CredentialConfig()}
    model_modes: dict[str, str] = {}  # Model-name pattern to canonical mode (legacy: "A1"/"A2" also accepted).
    model_context_windows: dict[str, int] = {}  # DEPRECATED: use models.json instead. Kept for backward compat only.
    model_max_tokens: dict[str, int] = {}  # モデル名パターン → デフォルト max_tokens
    anima_defaults: AnimaDefaults = AnimaDefaults()
    animas: dict[str, AnimaModelConfig] = {}
    consolidation: ConsolidationConfig = ConsolidationConfig()
    helper_models: HelperModelsConfig = Field(
        default_factory=HelperModelsConfig,
        description="Resolved model, credential, fallbacks, and one-shot policy for each helper role.",
    )
    background_review: BackgroundReviewConfig = BackgroundReviewConfig()
    rag: RAGConfig = RAGConfig()
    gpu: GPUConfig = GPUConfig()
    memory: MemoryConfig = MemoryConfig()
    skills: SkillsConfig = SkillsConfig()
    chatwork_tool: ChatworkToolConfig = ChatworkToolConfig()
    prompt: PromptConfig = PromptConfig()
    priming: PrimingConfig = PrimingConfig()
    image_gen: ImageGenConfig = ImageGenConfig()
    human_notification: HumanNotificationConfig = HumanNotificationConfig()
    interaction: InteractionConfig = InteractionConfig()
    server: ServerConfig = ServerConfig()
    llm_rate_guard: LlmRateGuardConfig = LlmRateGuardConfig()
    mcp: MCPConfig = MCPConfig()
    external_messaging: ExternalMessagingConfig = ExternalMessagingConfig()
    external_tasks: ExternalTasksConfig = Field(default_factory=ExternalTasksConfig)
    github_webhook: GitHubWebhookConfig = GitHubWebhookConfig()
    event_export: EventExportConfig = EventExportConfig()
    background_task: BackgroundTaskConfig = BackgroundTaskConfig()
    activity_log: ActivityLogConfig = ActivityLogConfig()
    logging: LoggingConfig = LoggingConfig()
    heartbeat: HeartbeatConfig = HeartbeatConfig()
    voice: VoiceConfig = VoiceConfig()
    phone: PhoneConfig = PhoneConfig()
    housekeeping: HousekeepingConfig = HousekeepingConfig()
    inbox: InboxConfig = InboxConfig()
    local_llm: LocalLLMConfig = LocalLLMConfig()
    workspaces: dict[str, str] = {}  # alias → absolute path
    # company slug → GitHub account name (e.g. {"fs": "animaworks-dev-team"})
    # Used by executors to inject GH_TOKEN and pin push identity.
    github_identities: dict[str, str] = Field(default_factory=dict)
    activity_level: int = Field(
        default=100,
        ge=10,
        le=400,
        description="Global activity level (10-400%). Scales heartbeat interval.",
    )
    activity_schedule: list[ActivityScheduleEntry] = Field(
        default_factory=list,
        description="Time-based activity level schedule. Empty = use fixed activity_level.",
    )
    icon_url_template: str = ""
    ui: UIConfig = UIConfig()
    cli: CLIConfig = CLIConfig()
    enclave: EnclaveConfig = EnclaveConfig()
    enclaves: dict[str, EnclaveClientConfig] = Field(default_factory=dict)


__all__ = [
    "ActivityLogConfig",
    "ActivityScheduleEntry",
    "AnimaDefaults",
    "AnimaModelConfig",
    "AnimaWorksConfig",
    "BackgroundReviewConfig",
    "BackgroundTaskConfig",
    "BackgroundToolConfig",
    "CLIConfig",
    "ChatworkToolConfig",
    "ConsolidationConfig",
    "CompactBackgroundRecallConfig",
    "CredentialConfig",
    "DEFAULT_ANIMA_MODEL",
    "DEFAULT_CONSOLIDATION_MODEL",
    "DEFAULT_LOCAL_LLM_BASE_URL",
    "DEFAULT_LOCAL_LLM_MODEL",
    "DEFAULT_LOCAL_LLM_PRESETS",
    "DEFAULT_LOCAL_LLM_ROLE_PRESETS",
    "ElevenLabsVoiceConfig",
    "EventExportConfig",
    "ExternalMessagingChannelConfig",
    "ExternalMessagingConfig",
    "ExternalTasksConfig",
    "ExternalTasksSourcesConfig",
    "GitHubWebhookConfig",
    "GPUConfig",
    "HeartbeatConfig",
    "HelperModelFallback",
    "HelperModelRole",
    "HelperModelsConfig",
    "HousekeepingConfig",
    "HumanNotificationConfig",
    "ImageGenConfig",
    "InboxConfig",
    "InteractionConfig",
    "IrodoriConfig",
    "GeminiTTSVoiceConfig",
    "LlmRateGuardConfig",
    "LocalLLMConfig",
    "LoggingConfig",
    "MCPConfig",
    "MediaProxyConfig",
    "MemoryConfig",
    "FactEdgeTypeConfig",
    "NotificationChannelConfig",
    "PhoneConfig",
    "PrimingConfig",
    "PromptConfig",
    "RAGConfig",
    "ServerConfig",
    "SkillPromotionConfig",
    "SkillsConfig",
    "StyleBertVits2Config",
    "SystemConfig",
    "UIConfig",
    "UserAliasConfig",
    "VoiceConfig",
    "VoicevoxConfig",
]
