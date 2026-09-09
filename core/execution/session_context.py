from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime session context shared by agent and tool execution paths."""

import contextvars
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeSessionContext:
    """Per-execution identity used to keep runtime state isolated."""

    request_id: str
    session_type: str
    thread_id: str
    trigger: str
    tool_session_id: str
    origin: str = ""
    origin_chain: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.session_type, self.thread_id, self.request_id)

    @classmethod
    def create(
        cls,
        *,
        session_type: str,
        thread_id: str,
        trigger: str,
        origin: str = "",
        origin_chain: tuple[str, ...] = (),
    ) -> RuntimeSessionContext:
        return cls(
            request_id=uuid.uuid4().hex,
            session_type=session_type or "unknown",
            thread_id=thread_id or "default",
            trigger=trigger or "",
            tool_session_id=uuid.uuid4().hex[:12],
            origin=origin,
            origin_chain=origin_chain,
        )

    @classmethod
    def from_env(cls) -> RuntimeSessionContext | None:
        request_id = os.environ.get("ANIMAWORKS_REQUEST_ID", "").strip()
        session_type = os.environ.get("ANIMAWORKS_SESSION_TYPE", "").strip()
        thread_id = os.environ.get("ANIMAWORKS_THREAD_ID", "").strip()
        trigger = os.environ.get("ANIMAWORKS_TRIGGER", "").strip()
        tool_session_id = os.environ.get("ANIMAWORKS_TOOL_SESSION_ID", "").strip()
        if not any((request_id, session_type, thread_id, trigger, tool_session_id)):
            return None
        return cls(
            request_id=request_id or uuid.uuid4().hex,
            session_type=session_type or "unknown",
            thread_id=thread_id or "default",
            trigger=trigger,
            tool_session_id=tool_session_id or uuid.uuid4().hex[:12],
        )

    def to_env(self) -> dict[str, str]:
        import json

        from core.taskboard.tasks import current_attempt_identity

        identity = current_attempt_identity()
        return {
            "ANIMAWORKS_REQUEST_ID": self.request_id,
            "ANIMAWORKS_SESSION_TYPE": self.session_type,
            "ANIMAWORKS_THREAD_ID": self.thread_id,
            "ANIMAWORKS_TRIGGER": self.trigger,
            "ANIMAWORKS_TOOL_SESSION_ID": self.tool_session_id,
            "ANIMAWORKS_TASK_IDENTITY": json.dumps(identity) if identity else "",
        }


active_runtime_session: contextvars.ContextVar[RuntimeSessionContext | None] = contextvars.ContextVar(
    "active_runtime_session",
    default=None,
)


def current_runtime_session() -> RuntimeSessionContext | None:
    return active_runtime_session.get(None)


@contextmanager
def runtime_session_scope(ctx: RuntimeSessionContext) -> Iterator[RuntimeSessionContext]:
    token = active_runtime_session.set(ctx)
    try:
        yield ctx
    finally:
        active_runtime_session.reset(token)
