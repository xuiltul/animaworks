from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.time_utils import now_jst


# ── Skill Metadata ────────────────────────────────────────

@dataclass
class SkillMeta:
    """Metadata extracted from a skill file's YAML frontmatter."""

    name: str           # frontmatter の name（なければファイル名 stem）
    description: str    # frontmatter の description（なければ空文字列）
    path: Path          # ファイルパス
    is_common: bool     # common_skills/ に配置されているか


# ── Emotion Constants ─────────────────────────────────────

VALID_EMOTIONS: frozenset[str] = frozenset({
    "neutral", "smile", "laugh", "troubled",
    "surprised", "thinking", "embarrassed",
})


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


class ModelConfig(BaseModel):
    """Per-anima API key, model, and endpoint configuration."""

    model: str = "claude-sonnet-4-20250514"
    fallback_model: str | None = None
    max_tokens: int = 4096
    max_turns: int = 20
    api_key: str | None = None  # direct API key (resolved from config.json)
    api_key_env: str = "ANTHROPIC_API_KEY"  # fallback: env var name
    api_base_url: str | None = None  # e.g. http://localhost:11434/v1
    context_threshold: float = 0.50  # short-term memory externalization threshold
    max_chains: int = 2  # max auto-continuation sessions
    conversation_history_threshold: float = 0.30  # conversation compression trigger
    execution_mode: str | None = None  # "autonomous" or "assisted"; None = auto
    supervisor: str | None = None  # supervisor Anima name
    speciality: str | None = None  # free-text specialisation
    resolved_mode: str | None = None  # "A1"/"A2"/"B" — resolved from config
    thinking: bool | None = None  # Ollama think param (None = auto: off for ollama/)
    llm_timeout: int | None = None  # LLM API呼び出しタイムアウト（秒）


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


class Message(BaseModel):
    id: str = Field(
        default_factory=lambda: now_jst().strftime("%Y%m%d_%H%M%S_%f")
    )
    thread_id: str = ""  # conversation thread (empty = new thread)
    reply_to: str = ""  # id of message being replied to
    from_person: str
    to_person: str
    type: str = "message"
    content: str
    attachments: list[str] = []
    intent: str = ""  # sender-declared intent: "delegation" | "report" | "question" | ""
    timestamp: datetime = Field(default_factory=now_jst)

    # External messaging integration
    source: str = "anima"  # "anima" | "human" | "slack" | "chatwork"
    source_message_id: str = ""  # message ID on external platform
    external_user_id: str = ""  # user ID on external platform
    external_channel_id: str = ""  # channel/room ID on external platform


class CycleResult(BaseModel):
    trigger: str
    action: str
    summary: str = ""
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=now_jst)
    context_usage_ratio: float = 0.0
    session_chained: bool = False
    total_turns: int = 0
    tool_call_records: list[dict] = Field(default_factory=list)


class AnimaStatus(BaseModel):
    name: str
    status: str = "idle"
    current_task: str = ""
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
    status: str  # "pending" | "in_progress" | "done" | "cancelled" | "blocked"
    summary: str  # 1行要約
    deadline: str | None = None  # ISO8601 期限（任意）
    relay_chain: list[str] = Field(default_factory=list)  # 委任経路
    updated_at: str  # ISO8601 最終更新日時