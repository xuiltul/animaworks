from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mode Y Antigravity CLI (agy) executor.

Subprocesses and binary discovery are mocked; these tests never invoke agy.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.execution.engines.agy.executor import (
    _AGY_BUILTIN_TOOLS,
    AgyCLIExecutor,
    _resolve_agy_model,
)
from core.execution.session.session_ids import EngineSessionIds
from core.i18n import t
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig

_AGY_SESSION_IDS = EngineSessionIds("agy")


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "test-agy"
    path.mkdir(parents=True)
    (path / "IDENTITY.md").write_text("# Test Agy Anima", encoding="utf-8")
    return path


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="agy/gemini-4-argon",
        max_tokens=4096,
        credential="gemini",
        context_threshold=0.5,
    )


@pytest.fixture
def executor(model_config: ModelConfig, anima_dir: Path) -> AgyCLIExecutor:
    return AgyCLIExecutor(model_config, anima_dir, tool_registry=["web_search"])


def _make_ndjson_lines(events: list[dict | str | bytes]) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        if isinstance(event, bytes):
            line = event
        elif isinstance(event, str):
            line = event.encode()
        else:
            line = json.dumps(event).encode()
        lines.append(line if line.endswith(b"\n") else line + b"\n")
    lines.append(b"")
    return lines


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeStream:
    def __init__(self, lines: list[bytes] | None = None, data: bytes = b"") -> None:
        self.lines = list(lines or [b""])
        self.data = data

    async def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    async def read(self) -> bytes:
        return self.data


class _FakeProc:
    def __init__(
        self,
        events: list[dict | str | bytes],
        *,
        returncode: int | None = 0,
        stderr: bytes = b"",
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(_make_ndjson_lines(events))
        self.stderr = _FakeStream(data=stderr)
        self.returncode = returncode
        self.pid = 1234
        self.kill_calls = 0

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, _sig: object) -> None:
        return None

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _success_events(
    *,
    conversation_id: str = "conv-new",
    response: str = "Hello world!",
    streamed: tuple[str, ...] = ("Hello ", "world"),
    with_tool: bool = False,
) -> list[dict]:
    events: list[dict] = [
        {"event": "init", "init": {"conversation_id": conversation_id}},
    ]
    for delta in streamed:
        events.append(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "ACTIVE",
                    "step_index": 1,
                    "text_delta": delta,
                },
            }
        )
    if with_tool:
        events.extend(
            [
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "step_index": 2,
                        "tool_info": {
                            "name": "call_mcp_tool",
                            "parameters": {
                                "ToolName": "web_search",
                                "Arguments": {"query": "anima"},
                            },
                        },
                    },
                },
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "DONE",
                        "step_index": 2,
                        "tool_info": {
                            "name": "call_mcp_tool",
                            "parameters": {
                                "ToolName": "web_search",
                                "Arguments": {"query": "anima"},
                            },
                            "output": "search-result",
                        },
                    },
                },
            ]
        )
    events.append(
        {
            "event": "step_update",
            "step_update": {
                "step_type": "agent_response",
                "state": "DONE",
                "step_index": 3,
                "usage": {"input_tokens": 100, "output_tokens": 5, "cache_read_tokens": 20},
            },
        }
    )
    events.append(
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": response,
                "conversation_id": conversation_id,
                "usage": {"input_tokens": 100, "output_tokens": 5, "cache_read_tokens": 20},
            },
        }
    )
    return events


async def _stream(
    executor: AgyCLIExecutor,
    procs: list[_FakeProc],
    *,
    trigger: str = "heartbeat",
    thread_id: str = "default",
    tracker: ContextTracker | MagicMock | None = None,
) -> tuple[list[dict], list[list[str]], list[str]]:
    tracker = tracker or ContextTracker(model=executor._model_config.model)
    commands: list[list[str]] = []
    agent_bodies: list[str] = []
    proc_iter = iter(procs)

    def _start(*args: str, **_kwargs: object) -> _FakeProc:
        command = list(args)
        commands.append(command)
        agent_name = command[command.index("--agent") + 1]
        agent_path = executor._agents_dir / f"{agent_name}.md"
        agent_bodies.append(agent_path.read_text(encoding="utf-8"))
        return next(proc_iter)

    with (
        patch("core.execution.engines.agy.executor._find_agy_binary", return_value="/usr/bin/agy"),
        patch("asyncio.create_subprocess_exec", side_effect=_start),
        # The fake pid may belong to a real process on the host; never resolve
        # (and signal) its process group, so signals reach the fake instead.
        patch("core.execution.process_runner.os.getpgid", side_effect=ProcessLookupError),
    ):
        events = [
            event
            async for event in executor.execute_streaming(
                "You are Agy.",
                "hello",
                tracker,
                trigger=trigger,
                thread_id=thread_id,
            )
        ]
    return events, commands, agent_bodies


def _event(events: list[dict], event_type: str) -> dict:
    return next(event for event in events if event.get("type") == event_type)


def _agent_body(raw: str) -> str:
    return raw.split("\n---\n", 1)[1].strip()


class TestCompatibilitySurface:
    def test_mode_and_session_engine(self, executor: AgyCLIExecutor) -> None:
        assert executor.engine_mode == "Y"
        assert executor.session_engine == "agy"
        assert executor.supports_streaming is True

    def test_model_prefix(self) -> None:
        assert _resolve_agy_model("agy/gemini-4-argon") == "gemini-4-argon"
        assert _resolve_agy_model("gemini-4-argon") == "gemini-4-argon"

    def test_builtin_tools_match_documented_set(self) -> None:
        assert "run_command" in _AGY_BUILTIN_TOOLS
        assert "view_file" in _AGY_BUILTIN_TOOLS
        assert "write_to_file" in _AGY_BUILTIN_TOOLS
        # agy rejects the whole agent definition when one name is unknown to its
        # custom-agent registry ("sed_file" is listed by init but not accepted).
        assert "sed_file" not in _AGY_BUILTIN_TOOLS

    def test_build_env_injects_api_key_only_when_configured(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            assert not executor._build_env().get("GEMINI_API_KEY")

        config = ModelConfig(
            model="agy/gemini-4-argon",
            max_tokens=4096,
            credential="gemini",
            api_key="test-key",
        )
        configured = AgyCLIExecutor(config, anima_dir)
        assert configured._build_env()["GEMINI_API_KEY"] == "test-key"

    def test_build_env_ignores_non_google_key_env(self, anima_dir: Path) -> None:
        # An Anima switched from Mode S keeps api_key_env=ANTHROPIC_API_KEY.
        config = ModelConfig(model="agy/gemini-4-argon", api_key_env="ANTHROPIC_API_KEY")
        switched = AgyCLIExecutor(config, anima_dir)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test", "GEMINI_API_KEY": ""}, clear=False):
            assert not switched._build_env().get("GEMINI_API_KEY")

        google = AgyCLIExecutor(ModelConfig(model="agy/gemini-4-argon", api_key_env="GOOGLE_API_KEY"), anima_dir)
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "g-test"}, clear=False):
            assert google._build_env()["GEMINI_API_KEY"] == "g-test"


class TestCommandConstruction:
    def test_command_has_effort_and_terminal_prompt_flag(self, executor: AgyCLIExecutor) -> None:
        executor._model_config.extra_keys = {"agy_effort": "high"}

        with patch("core.execution.engines.agy.executor._find_agy_binary", return_value="/usr/bin/agy"):
            cmd = executor._build_command(agent_name="aw-test")

        assert cmd[-1] == "-p="
        assert cmd[cmd.index("--agent") + 1] == "aw-test"
        assert cmd[cmd.index("--model") + 1] == "gemini-4-argon"
        assert cmd[cmd.index("--effort") + 1] == "high"
        settings = json.loads(cmd[cmd.index("--settings") + 1])
        assert settings["artifactReviewPolicy"] == "always-proceed"
        assert settings["allowNonWorkspaceAccess"] is True

    def test_model_suffix_suppresses_effort_flag(self, anima_dir: Path) -> None:
        config = ModelConfig(
            model="agy/gemini-4-argon-low",
            max_tokens=4096,
            credential="gemini",
            extra_keys={"agy_effort": "max"},
        )
        executor = AgyCLIExecutor(config, anima_dir)

        with patch("core.execution.engines.agy.executor._find_agy_binary", return_value="/usr/bin/agy"):
            cmd = executor._build_command(agent_name="aw-test")

        assert "--effort" not in cmd

    def test_invalid_effort_is_ignored(self, executor: AgyCLIExecutor) -> None:
        executor._model_config.extra_keys = {"agy_effort": "turbo"}

        assert executor._resolve_effort() is None

    def test_resume_adds_conversation_before_prompt_flag(self, executor: AgyCLIExecutor) -> None:
        with patch("core.execution.engines.agy.executor._find_agy_binary", return_value="/usr/bin/agy"):
            cmd = executor._build_command(agent_name="aw-test", resume_id="conv-old")

        assert cmd[cmd.index("--conversation") + 1] == "conv-old"
        assert cmd[-1] == "-p="
        assert cmd.index("--conversation") < cmd.index("-p=")


class TestStreaming:
    @pytest.mark.asyncio
    async def test_success_streams_text_tools_usage_and_persists_session(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        tracker = MagicMock(spec=ContextTracker)
        proc = _FakeProc(_success_events(with_tool=True))
        events, commands, agent_bodies = await _stream(executor, [proc], trigger="chat", tracker=tracker)

        text = "".join(event["text"] for event in events if event["type"] == "text_delta")
        assert text == "Hello world"
        assert _event(events, "tool_start")["tool_name"] == "web_search"
        assert _event(events, "tool_start")["input"] == {"query": "anima"}
        assert _event(events, "tool_end")["result"] == "search-result"

        done = _event(events, "done")
        assert done["full_text"] == "Hello world!"
        assert done["stop_kind"] == "normal"
        assert done["usage"] == {
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_tokens": 20,
            "cache_write_tokens": 0,
        }
        assert done["tool_call_records"][0]["tool_name"] == "web_search"
        assert done["tool_call_records"][0]["input_summary"].find("anima") >= 0
        assert done["tool_call_records"][0]["is_error"] is False

        tracker.update_from_message_start.assert_called_once_with({"input_tokens": 100, "cache_read_input_tokens": 20})

        payload = json.loads(proc.stdin.written.decode("utf-8"))
        assert payload["event"] == "user"
        assert payload["message"]["role"] == "user"
        assert payload["message"]["content"][0]["type"] == "text"
        assert payload["message"]["content"][0]["text"].startswith("[")
        assert payload["message"]["content"][0]["text"].endswith("hello")
        assert proc.stdin.closed is True

        record = _AGY_SESSION_IDS.load(anima_dir, "chat")
        assert record is not None
        assert record.session_id == "conv-new"
        assert record.turn_count == 1
        assert _AGY_SESSION_IDS.path_for(anima_dir, "chat").read_text(encoding="utf-8") == "conv-new\n1"
        assert not list((anima_dir / ".agy-workspace" / ".agents" / "agents").glob("aw-*.md"))
        assert len(commands) == 1
        assert agent_bodies[0]

    @pytest.mark.asyncio
    async def test_agent_definition_frontmatter_and_body(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        proc = _FakeProc(_success_events())
        events, _commands, agent_bodies = await _stream(executor, [proc])

        assert _event(events, "done")["full_text"] == "Hello world!"
        raw = agent_bodies[0]
        header, body = raw.split("\n---\n", 1)
        frontmatter = yaml.safe_load(header.removeprefix("---\n"))
        assert frontmatter["name"].startswith("aw-")
        assert frontmatter["mainAgent"] is True
        assert frontmatter["subagent"] is False
        assert frontmatter["tools"] == list(_AGY_BUILTIN_TOOLS)
        server = frontmatter["mcpServers"][0]
        assert server["name"] == "aw-test-agy"
        assert server["command"] == sys.executable
        assert server["args"]
        assert server["env"]
        assert body.strip() == "You are Agy."

    @pytest.mark.asyncio
    async def test_missing_binary_reports_not_installed(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        with patch("core.execution.engines.agy.executor._find_agy_binary", return_value=None):
            events = [
                event
                async for event in executor.execute_streaming(
                    "You are Agy.",
                    "hello",
                    ContextTracker(model="agy/gemini-4-argon"),
                    trigger="heartbeat",
                )
            ]

        assert any(event["type"] == "text_delta" and event["text"] == t("agy_cli.not_installed") for event in events)
        done = _event(events, "done")
        assert done["error"] is True
        assert done["reason"] == "unknown"
        assert _event(events, "error")["terminal"] is True

    @pytest.mark.asyncio
    async def test_auth_error_reports_login_message(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        proc = _FakeProc(
            [
                {
                    "event": "result",
                    "result": {
                        "status": "ERROR",
                        "response": "",
                        "error": "authentication required",
                        "conversation_id": "conv-auth",
                    },
                }
            ]
        )
        events, _commands, _bodies = await _stream(executor, [proc])

        done = _event(events, "done")
        assert done["error"] is True
        assert done["reason"] == "auth"
        assert done["full_text"] == t("agy_cli.not_authenticated")
        assert _event(events, "error")["reason"] == "auth"

    @pytest.mark.asyncio
    async def test_quota_error_uses_grok_reason_taxonomy(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        proc = _FakeProc(
            [
                {
                    "event": "result",
                    "result": {
                        "status": "ERROR",
                        "response": "",
                        "error": "quota exceeded for this project",
                    },
                }
            ]
        )
        with patch.object(executor, "_error_metadata", return_value={"terminal": True, "reason": "unknown"}):
            events, _commands, _bodies = await _stream(executor, [proc])

        done = _event(events, "done")
        assert done["error"] is True
        assert done["reason"] == "quota_exhausted"

    @pytest.mark.asyncio
    async def test_failed_tool_step_records_error(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        proc = _FakeProc(
            [
                {"event": "init", "init": {"conversation_id": "conv-tool"}},
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "step_index": 2,
                        "tool_name": "run_command",
                        "tool_info": {"name": "run_command", "parameters": {"CommandLine": "false"}},
                    },
                },
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "FAILED",
                        "step_index": 2,
                        "tool_name": "run_command",
                        "tool_info": {"name": "run_command", "parameters": {"CommandLine": "false"}, "output": "boom"},
                    },
                },
                {
                    "event": "result",
                    "result": {"status": "SUCCESS", "response": "done", "conversation_id": "conv-tool"},
                },
            ]
        )
        events, _commands, _bodies = await _stream(executor, [proc])

        end = _event(events, "tool_end")
        assert end["is_error"] is True
        assert _event(events, "done")["tool_call_records"][0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_resume_failure_retries_once_with_fresh_session(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        _AGY_SESSION_IDS.save(anima_dir, "conv-old", "chat", turn_count=2)
        failed = _FakeProc(
            [
                {
                    "event": "result",
                    "result": {
                        "status": "ERROR",
                        "response": "",
                        "error": "conversation not found",
                        "conversation_id": "conv-old",
                    },
                }
            ]
        )
        success = _FakeProc(_success_events(conversation_id="conv-new", streamed=("fresh",)))

        events, commands, agent_bodies = await _stream(executor, [failed, success], trigger="chat")

        assert len(commands) == 2
        assert "--conversation" in commands[0]
        assert "--conversation" not in commands[1]
        assert _agent_body(agent_bodies[0]) == "# AnimaWorks Anima\n\nContinue the existing conversation."
        assert "You are Agy." in _agent_body(agent_bodies[1])

        done = _event(events, "done")
        assert "error" not in done
        assert done["full_text"] == "Hello world!"
        record = _AGY_SESSION_IDS.load(anima_dir, "chat")
        assert record is not None
        assert record.session_id == "conv-new"
        assert record.turn_count == 1

    @pytest.mark.asyncio
    async def test_failed_fresh_turn_does_not_persist_conversation(
        self,
        executor: AgyCLIExecutor,
        anima_dir: Path,
    ) -> None:
        failed = _FakeProc(
            [
                {"event": "init", "conversation_id": "conv-broken", "init": {"model": "gemini-4-argon"}},
                {
                    "event": "result",
                    "result": {
                        "status": "ERROR",
                        "response": "",
                        "error": "failed to construct executor",
                        "conversation_id": "conv-broken",
                    },
                },
            ]
        )

        events, commands, _bodies = await _stream(executor, [failed], trigger="chat")

        assert len(commands) == 1
        assert _event(events, "done")["error"] is True
        assert _AGY_SESSION_IDS.load(anima_dir, "chat") is None

    @pytest.mark.asyncio
    async def test_pre_interrupted_turn_does_not_spawn(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
    ) -> None:
        interrupt_event = asyncio.Event()
        interrupt_event.set()
        executor = AgyCLIExecutor(model_config, anima_dir, interrupt_event=interrupt_event)

        with patch("asyncio.create_subprocess_exec") as spawn:
            events = [
                event
                async for event in executor.execute_streaming(
                    "You are Agy.",
                    "hello",
                    ContextTracker(model="agy/gemini-4-argon"),
                    trigger="heartbeat",
                )
            ]

        spawn.assert_not_called()
        done = _event(events, "done")
        assert done["stop_kind"] == "interrupted"
        assert done["full_text"] == "[Session interrupted by user]"
        assert "error" not in done
