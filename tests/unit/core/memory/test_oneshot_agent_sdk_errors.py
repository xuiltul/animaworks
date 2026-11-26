# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""The Agent SDK one-shot must not return the CLI's error text as an answer."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from core.llm import oneshot


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> None:
    class FakeClient:
        def __init__(self, options: Any) -> None:
            self.options = options

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            for message in messages:
                yield message

    module = ModuleType("claude_agent_sdk")
    module.ClaudeSDKClient = FakeClient
    module.ClaudeAgentOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setattr("core.execution.engines.claude.apply_sdk_transport_patch", lambda: None)
    monkeypatch.setattr("core.execution.engines.claude.resolve_sdk_cli_path", lambda: None)
    monkeypatch.setattr(oneshot, "_build_sdk_env", lambda: {})


def _text(text: str, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)], **extra)


async def _run() -> str | None:
    return await oneshot._try_agent_sdk("p", system_prompt="", model="claude-opus-5-5", max_tokens=100)


@pytest.mark.asyncio
async def test_not_logged_in_text_is_not_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(
        monkeypatch,
        [
            _text("Not logged in · Please run /login", error="authentication_failed"),
            SimpleNamespace(subtype="success", is_error=True, result="Not logged in · Please run /login"),
        ],
    )
    assert await _run() is None


@pytest.mark.asyncio
async def test_error_result_subtype_is_not_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(
        monkeypatch,
        [_text("partial"), SimpleNamespace(subtype="error_max_turns", is_error=False, result=None)],
    )
    assert await _run() is None


@pytest.mark.asyncio
async def test_normal_answer_mentioning_errors_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(
        monkeypatch,
        [
            _text("Summary: the login error was fixed.", error=None),
            SimpleNamespace(subtype="success", is_error=False, result="Summary: the login error was fixed."),
        ],
    )
    assert await _run() == "Summary: the login error was fixed."
