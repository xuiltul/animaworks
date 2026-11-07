"""Tests for run triggers exported to Anima workspaces."""

from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from core.execution.session.board_reply import (
    RUN_TRIGGERS_ENV,
    current_run_triggers,
    run_triggers,
    run_triggers_env,
)
from core.execution.session.session_context import RuntimeSessionContext
from core.execution.workspace_env import build_workspace_env


@pytest.fixture
def triggers() -> tuple[dict[str, str], ...]:
    return (
        {"id": "msg-1", "from": "bob", "type": "message", "thread_id": "thread-1"},
        {
            "id": "mention-1",
            "from": "carol",
            "type": "board_mention",
            "board_channel": "general",
            "board_post_id": "post-9",
        },
    )


def test_run_triggers_env_roundtrip(triggers: tuple[dict[str, str], ...], monkeypatch) -> None:
    token = run_triggers.set(triggers)
    try:
        env = run_triggers_env()
        assert json.loads(env[RUN_TRIGGERS_ENV]) == list(triggers)

        monkeypatch.setenv(RUN_TRIGGERS_ENV, env[RUN_TRIGGERS_ENV])
        assert current_run_triggers() == triggers
    finally:
        run_triggers.reset(token)


def test_current_run_triggers_defaults_to_empty(monkeypatch) -> None:
    monkeypatch.delenv(RUN_TRIGGERS_ENV, raising=False)
    assert current_run_triggers() == ()


def test_current_run_triggers_ignores_malformed_env(monkeypatch) -> None:
    monkeypatch.setenv(RUN_TRIGGERS_ENV, "{not json")
    assert current_run_triggers() == ()


def test_session_context_exports_run_triggers(triggers: tuple[dict[str, str], ...]) -> None:
    token = run_triggers.set(triggers)
    try:
        env = RuntimeSessionContext.create(
            session_type="inbox",
            thread_id="thread-1",
            trigger="message",
        ).to_env()
    finally:
        run_triggers.reset(token)
    assert json.loads(env[RUN_TRIGGERS_ENV]) == list(triggers)


def test_build_workspace_env_exports_run_triggers(
    tmp_path: Path,
    triggers: tuple[dict[str, str], ...],
) -> None:
    token = run_triggers.set(triggers)
    try:
        env = build_workspace_env(tmp_path / "animas" / "alice")
    finally:
        run_triggers.reset(token)
    assert json.loads(env[RUN_TRIGGERS_ENV]) == list(triggers)
