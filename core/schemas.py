from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from core.config.models import DEFAULT_ANIMA_MODEL
from core.time_utils import now_local

# ── Skill Metadata ────────────────────────────────────────


@dataclass
class SkillMeta:
    """Metadata extracted from a skill file's YAML frontmatter."""

    name: str  # frontmatter の name（なければファイル名 stem）
    description: str  # frontmatter の description（なければ空文字列）
    path: Path  # ファイルパス
    is_common: bool  # common_skills/ に配置されているか
    allowed_tools: list[str] = field(default_factory=list)  # frontmatter の allowed_tools


# ── Emotion Constants ─────────────────────────────────────

VALID_EMOTIONS: frozenset[str] = frozenset(
    {
        "neutral",
        "smile",
        "laugh",
        "troubled",
        "surprised",
        "thinking",
        "embarrassed",
    }
)


class CronTask(BaseModel):
    """Cron task definition supporting both LLM and command types.

    LLM type:
        - type: "llm" (default)
        - description: Task instruction for LLM

    Command type:
        - type: "command"
        - command: Bash command to execute
        - OR tool + args: Internal tool to invoke
    """

    name: str
    schedule: str
    type: str = "llm"  # "llm" or "command"
    description: str = ""  # LLM型の指示文
    command: str | None = None  # Command型のbashコマンド
    tool: str | None = None  # Command型の内部ツール名
    args: dict[str, Any] | None = None  # toolの引数
    skip_pattern: str | None = None  # stdoutがマッチしたらheartbeatをスキップ
    trigger_heartbeat: bool = True  # Falseならcron出力時のHBトリガーを抑制


class ModelConfig(BaseModel):
    """Per-anima API key, model, and endpoint configuration."""

    model: str = DEFAULT_ANIMA_MODEL
    fallback_model: str | None = None
    max_tokens: int = 8192
    max_turns: int = 20
    api_key: str | None = None  # direct API key (resolved from config.json)
    api_key_env: str = "ANTHROPIC_API_KEY"  # fallback: env var name
    api_base_url: str | None = None  # e.g. http://localhost:11434/v1
    context_threshold: float = 0.50  # short-term memory externalization threshold
    max_chains: int = 2  # max auto-continuation sessions
    conversation_history_threshold: float = 0.30  # conversation compression trigger
    execution_mode: str | None = None  # Canonical mode letter (S/C/D/G/A/B) or None for auto-resolve
    supervisor: str | None = None  # supervisor Anima name
    speciality: str | None = None  # free-text specialisation
    resolved_mode: str | None = None  # "S"/"C"/"D"/"G"/"A"/"B" — resolved from config
    thinking: bool | None = None  # Extended thinking (Bedrock: reasoning_effort, Ollama: think param)
    thinking_effort: str | None = None  # "low"/"medium"/"high"/"max" (default: "high")
    llm_timeout: int | None = None  # LLM API呼び出しタイムアウト（秒）
    background_model: str | None = None  # heartbeat/cron override model
    background_credential: str | None = None  # credential for background_model
    extra_keys: dict[str, str] = {}  # provider-specific credential keys (e.g. api_version, vertex_project)
    mode_s_auth: str | None = None  # Mode S auth: "max"|"api"|"bedrock"|"vertex"|None(=max)
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    extra_mcp_servers: dict[str, dict] = Field(default_factory=dict)


class AnimaConfig(BaseModel):
    name: str
    base_dir: Path
    identity: str = ""
    injection: str = ""
    permissions: str = ""
    heartbeat_interval: int = 30  # minutes
    active_hours: tuple[int, int] | None = (9, 22)  # None = 24h, e.g. (9, 22) for daytime only
    cron_tasks: list[CronTask] = []
    model_config_data: ModelConfig = Field(default_factory=ModelConfig)


EXTERNAL_PLATFORM_SOURCES: frozenset[str] = frozenset({"slack", "chatwork", "googlechat"})
"""Message ``source`` values representing external platforms (Slack, Chatwork, etc.)."""


class Message(BaseModel):
    id: str = Field(default_factory=lambda: now_local().strftime("%Y%m%d_%H%M%S_%f"))
    thread_id: str = ""  # conversation thread (empty = new thread)
    reply_to: str = ""  # id of message being replied to
    from_person: str
    to_person: str
    type: str = "message"
    content: str
    attachments: list[str] = []
    intent: str = ""  # sender-declared intent: "delegation" | "report" | "question" | ""
    timestamp: datetime = Field(default_factory=now_local)

    # External messaging integration
    source: str = "anima"  # "anima" | "human" | "slack" | "chatwork" | "googlechat"
    source_message_id: str = ""  # message ID on external platform
    external_user_id: str = ""  # user ID on external platform
    external_channel_id: str = ""  # channel/room ID on external platform
    external_thread_ts: str = ""  # thread parent ts on external platform (Slack thread_ts)

    # Provenance tracking (Phase 2)
    origin_chain: list[str] = Field(default_factory=list)

    # Structured metadata (e.g. task_id for delegation DMs)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_keys(cls, data: Any) -> Any:
        """Accept legacy ``from``/``to`` keys as aliases for from_person/to_person.

        Early versions of the message schema used ``from`` and ``to`` as field
        names.  If a JSON file still uses the old names, transparently map them
        to the current ``from_person`` / ``to_person`` fields so that the
        message can be parsed without error.
        """
        if isinstance(data, dict):
            if "from" in data and "from_person" not in data:
                data["from_person"] = data.pop("from")
            if "to" in data and "to_person" not in data:
                data["to_person"] = data.pop("to")
            # Legacy files may use ``ts`` instead of ``timestamp``; map it
            # to the canonical field so that the original value is preserved.
            if "ts" in data and "timestamp" not in data:
                data["timestamp"] = data.pop("ts")
        return data


# ── Shared TypedDicts ────────────────────────────────────────


class ImageData(TypedDict):
    """Base64-encoded image payload used across core and server layers."""

    data: str  # Base64 encoded (no data: prefix)
    media_type: str  # "image/jpeg", "image/png", etc.


class ToolCallRecordDict(TypedDict):
    """Serialised form of :class:`core.execution.base.ToolCallRecord`."""

    tool_name: str
    tool_id: str
    input_summary: str
    result_summary: str
    is_error: bool


class CycleResult(BaseModel):
    trigger: str
    action: str
    summary: str = ""
    thinking_text: str = ""
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=now_local)
    context_usage_ratio: float = 0.0
    context_window: int = 0
    context_threshold: float = 0.0
    session_chained: bool = False
    total_turns: int = 0
    tool_call_records: list[ToolCallRecordDict] = Field(default_factory=list)
    images: list[dict[str, str]] = Field(default_factory=list)
    usage: dict[str, int] | None = None


class AnimaStatus(BaseModel):
    name: str
    status: str = "idle"
    active_label: str = ""
    last_heartbeat: datetime | None = None
    last_activity: datetime | None = None
    pending_messages: int = 0


class TaskEntry(BaseModel):
    """永続タスクキューのエントリ."""

    task_id: str  # ULID or UUID
    ts: str  # ISO8601 作成日時
    source: Literal["human", "anima"]
    original_instruction: str  # 原文（委任時は引用を含む）
    assignee: str  # 担当Anima名
    status: str  # "pending" | "in_progress" | "done" | "cancelled" | "blocked" | "delegated"
    summary: str  # 1行要約
    deadline: str | None = None  # ISO8601 期限（任意）
    relay_chain: list[str] = Field(default_factory=list)  # 委任経路
    updated_at: str  # ISO8601 最終更新日時
    meta: dict[str, Any] = Field(default_factory=dict)  # 追加メタデータ（委譲追跡等）
