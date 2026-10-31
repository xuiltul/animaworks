from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mode X Grok ACP executor.

Every subprocess and binary lookup is mocked; these tests never invoke Grok.
"""

import asyncio
import json
import logging
import os
import signal
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from core.execution.base import ExecutionResult, TokenUsage, ToolCallRecord
from core.execution.engine_base import engine_error_metadata
from core.execution.engines.grok.executor import (
    _MAX_RESUME_TURNS,
    GrokCLIExecutor,
    _find_grok_binary,
    _resolve_grok_model,
    _resolve_real_error,
    _resolve_session_type,
    is_grok_cli_available,
)
from core.execution.session.session_ids import EngineSessionIds
from core.prompt.context import ContextTracker
from core.schemas import ModelConfig

_GROK_SESSION_IDS = EngineSessionIds("grok")


def _load_grok_session_id(anima_dir: Path, session_type: str, thread_id: str = "default") -> tuple[str | None, int]:
    record = _GROK_SESSION_IDS.load(anima_dir, session_type, thread_id)
    return (record.session_id, record.turn_count) if record is not None else (None, 0)


_REAL_GROK_QUOTA_ERROR = "API error (status 402 Payment Required): Grok Build usage balance exhausted"


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    path = tmp_path / "animas" / "test-grok"
    path.mkdir(parents=True)
    (path / "IDENTITY.md").write_text("# Test Grok Anima", encoding="utf-8")
    return path


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model="grok/grok-4.5",
        max_tokens=4096,
        credential="grok",
        context_threshold=0.5,
    )


@pytest.fixture
def executor(model_config: ModelConfig, anima_dir: Path) -> GrokCLIExecutor:
    return GrokCLIExecutor(model_config, anima_dir, tool_registry=["web_search"])


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

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None


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
        self.sent_signals: list[signal.Signals] = []
        self.kill_calls = 0

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def send_signal(self, sig: signal.Signals) -> None:
        self.sent_signals.append(sig)

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _success_events(
    *,
    session_id: str = "session-new",
    updates: list[dict] | None = None,
    usage: dict | None = None,
    meta_top_level: dict | None = None,
    load: bool = False,
) -> list[dict]:
    events: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "result": {"agentCapabilities": {"loadSession": True}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": session_id} if not load else {}},
    ]
    events.extend(updates or [])
    meta: dict = {
        "sessionId": session_id,
        "usage": usage or {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
    }
    if meta_top_level:
        # Top-level fields (e.g. inputTokens for last-call context size) sit
        # alongside nested usage; merge without clobbering sessionId/usage.
        meta = {**meta_top_level, **meta}
        if "usage" in meta_top_level:
            meta["usage"] = meta_top_level["usage"]
        if "sessionId" in meta_top_level:
            meta["sessionId"] = meta_top_level["sessionId"]
    events.append(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "stopReason": "end_turn",
                "_meta": meta,
            },
        }
    )
    return events


def _requests(proc: _FakeProc) -> list[dict]:
    return [json.loads(line) for line in proc.stdin.written.splitlines()]


async def _stream(
    executor: GrokCLIExecutor,
    proc: _FakeProc,
    *,
    trigger: str = "heartbeat",
    thread_id: str = "default",
    tracker: ContextTracker | MagicMock | None = None,
) -> list[dict]:
    tracker = tracker or ContextTracker(model="grok/grok-4.5")
    with (
        patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
        # The fake pid may belong to a real process on the host; never resolve
        # (and signal) its process group, so signals reach the fake instead.
        patch("core.execution.process_runner.os.getpgid", side_effect=ProcessLookupError),
    ):
        return [
            event
            async for event in executor.execute_streaming(
                "You are Grok.",
                "hello",
                tracker,
                trigger=trigger,
                thread_id=thread_id,
            )
        ]


class TestDiscoveryAndHelpers:
    def test_binary_discovery_and_availability(self):
        with patch("core.platform.grok.shutil.which", return_value="/opt/grok"):
            assert _find_grok_binary() == "/opt/grok"
        with patch("core.platform.grok.get_grok_executable", return_value="/opt/grok"):
            assert is_grok_cli_available() is True
        with patch("core.platform.grok.get_grok_executable", return_value=None):
            assert is_grok_cli_available() is False

    @pytest.mark.parametrize(
        ("model", "expected"),
        [("grok/grok-4.5", "grok-4.5"), ("grok-4.5", "grok-4.5")],
    )
    def test_model_prefix(self, model: str, expected: str):
        assert _resolve_grok_model(model) == expected


class TestResolveRealError:
    _SINCE = datetime(2026, 8, 6, 6, 0, tzinfo=UTC).timestamp()

    @staticmethod
    def _entry(
        *,
        ts: str,
        pid: int = 1234,
        sid: str = "session-new",
        message: str = _REAL_GROK_QUOTA_ERROR,
    ) -> dict:
        return {
            "ts": ts,
            "lvl": "error",
            "pid": pid,
            "sid": sid,
            "msg": "shell.turn.inference_failed",
            "ctx": {"status_code": 402, "message": message},
        }

    @staticmethod
    def _write(path: Path, *entries: dict | str) -> None:
        lines = [entry if isinstance(entry, str) else json.dumps(entry) for entry in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_pid_match_returns_newest_context(self, tmp_path: Path) -> None:
        path = tmp_path / "unified.jsonl"
        self._write(
            path,
            self._entry(ts="2026-08-06T06:00:01Z", message="older"),
            self._entry(ts="2026-08-06T06:00:03Z", message="newest"),
        )

        assert _resolve_real_error(1234, "other-session", self._SINCE, path) == {
            "status_code": 402,
            "message": "newest",
        }

    def test_sid_match_accepts_different_pid(self, tmp_path: Path) -> None:
        path = tmp_path / "unified.jsonl"
        self._write(path, self._entry(ts="2026-08-06T06:00:01Z", pid=9999))

        assert _resolve_real_error(1234, "session-new", self._SINCE, path) == {
            "status_code": 402,
            "message": _REAL_GROK_QUOTA_ERROR,
        }

    def test_entry_outside_time_window_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "unified.jsonl"
        self._write(path, self._entry(ts="2026-08-06T05:59:50Z"))

        assert _resolve_real_error(1234, "session-new", self._SINCE, path) is None

    def test_broken_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "unified.jsonl"
        self._write(
            path,
            "{broken json",
            self._entry(ts="2026-08-06T06:00:01Z"),
        )

        assert _resolve_real_error(1234, "", self._SINCE, path) is not None

    def test_missing_file_fails_open(self, tmp_path: Path) -> None:
        assert (
            _resolve_real_error(
                1234,
                "session-new",
                self._SINCE,
                tmp_path / "missing.jsonl",
            )
            is None
        )

    def test_only_tail_256kb_is_considered(self, tmp_path: Path) -> None:
        path = tmp_path / "unified.jsonl"
        early = json.dumps(self._entry(ts="2026-08-06T06:00:01Z")).encode()
        path.write_bytes(early + b"\n" + b"x" * (300 * 1024) + b"\n")

        assert _resolve_real_error(1234, "session-new", self._SINCE, path) is None


class TestGrokFailureMetadata:
    def test_quota_error_records_real_guard_block(self, tmp_path: Path) -> None:
        from core.config.schemas import LlmRateGuardConfig
        from core.llm.guard.rate_guard import LlmRateGuard

        guard_path = tmp_path / "llm_rate_guard.json"
        guard = LlmRateGuard(
            config=LlmRateGuardConfig(quota_block_seconds=1800),
            path=guard_path,
        )

        with patch("core.execution.engine_base.get_rate_guard", return_value=guard):
            metadata = engine_error_metadata(
                _REAL_GROK_QUOTA_ERROR,
                mode="X",
                model="grok/grok-4.5",
                always_terminal=True,
            )

        state = json.loads(guard_path.read_text(encoding="utf-8"))
        assert metadata == {"terminal": True, "reason": "quota_exhausted"}
        assert state["grok:grok"]["reason"] == "quota_exhausted"
        assert guard.blocked_remaining("grok:grok") > 1700


class TestDiscoveryAndHelperEnvironment:
    def test_mcp_env_propagates_runtime_trigger(self, executor: GrokCLIExecutor):
        from core.execution.session.session_context import RuntimeSessionContext, runtime_session_scope

        ctx = RuntimeSessionContext.create(session_type="cron", thread_id="t-1", trigger="cron:daily")
        with runtime_session_scope(ctx):
            servers = executor._mcp_servers()
        env_map = {item["name"]: item["value"] for item in servers[0]["env"]}
        assert env_map["ANIMAWORKS_TRIGGER"] == "cron:daily"
        assert env_map["ANIMAWORKS_SERVER_URL"].startswith("http")

    def test_mcp_env_passes_embed_and_vector_urls(self, executor: GrokCLIExecutor):
        """MCP server env must include embed/vector/rerank URLs so it delegates
        over HTTP instead of loading models in-process
        (2026-07-17 OOM / 2026-08-07 CrossEncoder load storm)."""
        with patch.dict(
            os.environ,
            {
                "ANIMAWORKS_EMBED_URL": "http://127.0.0.1:18500/api/internal/embed",
                "ANIMAWORKS_VECTOR_URL": "http://127.0.0.1:18500/api/internal/vector",
                "ANIMAWORKS_RERANK_URL": "http://127.0.0.1:18500/api/internal/rerank",
            },
        ):
            servers = executor._mcp_servers()
        env_map = {item["name"]: item["value"] for item in servers[0]["env"]}
        assert env_map["ANIMAWORKS_EMBED_URL"] == "http://127.0.0.1:18500/api/internal/embed"
        assert env_map["ANIMAWORKS_VECTOR_URL"] == "http://127.0.0.1:18500/api/internal/vector"
        assert env_map["ANIMAWORKS_RERANK_URL"] == "http://127.0.0.1:18500/api/internal/rerank"

    def test_mcp_env_omits_urls_when_unset(self, executor: GrokCLIExecutor):
        env_copy = os.environ.copy()
        env_copy.pop("ANIMAWORKS_EMBED_URL", None)
        env_copy.pop("ANIMAWORKS_VECTOR_URL", None)
        env_copy.pop("ANIMAWORKS_RERANK_URL", None)
        with patch.dict(os.environ, env_copy, clear=True):
            servers = executor._mcp_servers()
        env_names = {item["name"] for item in servers[0]["env"]}
        assert "ANIMAWORKS_EMBED_URL" not in env_names
        assert "ANIMAWORKS_VECTOR_URL" not in env_names
        assert "ANIMAWORKS_RERANK_URL" not in env_names

    def test_session_paths_and_trigger_normalization(self, anima_dir: Path):
        assert _resolve_session_type("message:user") == "chat"
        assert _resolve_session_type("chat:web") == "chat"
        assert _resolve_session_type("heartbeat:timer") == "heartbeat"
        expected = anima_dir / "shortterm/chat/thread-a/grok_session_id.txt"
        assert _GROK_SESSION_IDS.path_for(anima_dir, "chat", "thread-a") == expected
        _GROK_SESSION_IDS.save(anima_dir, "sid-a", "chat", "thread-a", 4)
        assert _load_grok_session_id(anima_dir, "chat", "thread-a") == ("sid-a", 4)
        assert _load_grok_session_id(anima_dir, "chat", "thread-b") == (None, 0)


class TestSandboxConfiguration:
    @staticmethod
    def _write_permissions(
        anima_dir: Path,
        *,
        file_roots: list[str],
        denied_roots: list[str] | None = None,
    ) -> None:
        (anima_dir / "permissions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "file_roots": file_roots,
                    "file_roots_denied": denied_roots or [],
                }
            ),
            encoding="utf-8",
        )

    def test_denied_roots_generate_expected_profile(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        allowed = tmp_path.parent / f"{tmp_path.name}-allowed"
        denied = tmp_path / "denied"
        allowed.mkdir()
        denied.mkdir()
        self._write_permissions(
            anima_dir,
            file_roots=[str(allowed)],
            denied_roots=[str(denied)],
        )

        assert executor._write_sandbox_config() is True

        parsed = tomllib.loads((executor._workspace / ".grok" / "sandbox.toml").read_text(encoding="utf-8"))
        profile = parsed["profiles"]["animaworks"]
        assert profile == {
            "extends": "workspace",
            "read_write": [
                str(anima_dir.resolve()),
                str(allowed.resolve()),
                str((tmp_path / "cache").resolve()),
            ],
            "deny": [str(denied.resolve())],
        }
        assert "read_only" not in profile

    def test_full_access_without_denied_roots_disables_sandbox(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
    ) -> None:
        self._write_permissions(anima_dir, file_roots=["/"])

        assert executor._write_sandbox_config() is False
        assert not (executor._workspace / ".grok" / "sandbox.toml").exists()

    def test_company_shared_is_created_before_full_access_disables_sandbox(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        own_company = tmp_path / "companies" / "fs"
        own_company.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({"company": "fs"}), encoding="utf-8")
        self._write_permissions(anima_dir, file_roots=["/"])

        assert executor._write_sandbox_config() is False

        assert (own_company / "shared").is_dir()
        assert not (executor._workspace / ".grok" / "sandbox.toml").exists()

    def test_normal_roots_without_denied_roots_generate_profile(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        allowed = tmp_path.parent / f"{tmp_path.name}-normal-root"
        allowed.mkdir()
        task_cwd = tmp_path.parent / f"{tmp_path.name}-task"
        executor.set_task_cwd(task_cwd)
        (tmp_path / "companies" / "fs" / "shared").mkdir(parents=True)
        self._write_permissions(anima_dir, file_roots=[str(allowed)])

        assert executor._write_sandbox_config() is True
        profile = tomllib.loads((executor._workspace / ".grok" / "sandbox.toml").read_text(encoding="utf-8"))[
            "profiles"
        ]["animaworks"]
        assert profile["read_write"] == [
            str(anima_dir.resolve()),
            str(allowed.resolve()),
            str(task_cwd.resolve()),
            str((tmp_path / "cache").resolve()),
        ]
        assert profile["deny"] == []

    def test_company_shared_is_created_and_added_independent_of_file_roots(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        own_company = tmp_path / "companies" / "fs"
        foreign_company = tmp_path / "companies" / "other"
        own_company.mkdir(parents=True)
        foreign_company.mkdir(parents=True)
        for name in ("knowledge", "skills", "credentials"):
            (own_company / name).mkdir()
        for name in ("vision.md", "company.json"):
            (own_company / name).write_text("{}", encoding="utf-8")
        (anima_dir / "status.json").write_text(json.dumps({"company": "fs"}), encoding="utf-8")
        self._write_permissions(anima_dir, file_roots=[])

        assert executor._write_sandbox_config() is True

        own_shared = own_company / "shared"
        assert own_shared.is_dir()
        profile = tomllib.loads((executor._workspace / ".grok" / "sandbox.toml").read_text(encoding="utf-8"))[
            "profiles"
        ]["animaworks"]
        assert profile["read_write"] == [
            str(anima_dir.resolve()),
            str(own_shared.resolve()),
            str((tmp_path / "cache").resolve()),
        ]
        assert profile["deny"] == [str(foreign_company.resolve())]
        for name in ("knowledge", "skills", "vision.md", "company.json", "credentials"):
            assert str((own_company / name).resolve()) not in profile["read_write"]

    def test_model_escape_hatch_disables_sandbox(
        self,
        model_config: ModelConfig,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        denied = tmp_path / "denied"
        denied.mkdir()
        self._write_permissions(
            anima_dir,
            file_roots=[str(anima_dir)],
            denied_roots=[str(denied)],
        )
        config = model_config.model_copy(update={"extra_keys": {"grok_sandbox": "off"}})
        executor = GrokCLIExecutor(config, anima_dir)

        assert executor._write_sandbox_config() is False
        assert not (executor._workspace / ".grok" / "sandbox.toml").exists()

    def test_sandbox_toml_escapes_quotes_and_backslashes(
        self,
        executor: GrokCLIExecutor,
        anima_dir: Path,
        tmp_path: Path,
    ) -> None:
        unusual_root = tmp_path.parent / f'{tmp_path.name}-quote"and\\backslash'
        unusual_root.mkdir()
        self._write_permissions(anima_dir, file_roots=[str(unusual_root)])

        assert executor._write_sandbox_config() is True
        profile = tomllib.loads((executor._workspace / ".grok" / "sandbox.toml").read_text(encoding="utf-8"))[
            "profiles"
        ]["animaworks"]
        assert str(unusual_root.resolve()) in profile["read_write"]

    def test_sandbox_event_logs_enforced_profile(
        self,
        executor: GrokCLIExecutor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        grok_home = tmp_path / ".grok"
        grok_home.mkdir()
        (grok_home / "sandbox-events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "ProfileApplied",
                    "workspace": str(executor._workspace.resolve()),
                    "enforced": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        caplog.set_level(logging.DEBUG, logger="animaworks.execution.grok_cli")

        with patch("core.execution.engines.grok.executor.Path.home", return_value=tmp_path):
            executor._log_sandbox_status()

        assert "Grok sandbox enforced" in caplog.text

    def test_sandbox_event_warns_when_workspace_event_is_missing(
        self,
        executor: GrokCLIExecutor,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        grok_home = tmp_path / ".grok"
        grok_home.mkdir()
        (grok_home / "sandbox-events.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "ProfileApplied",
                    "workspace": "/different/workspace",
                    "enforced": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with patch("core.execution.engines.grok.executor.Path.home", return_value=tmp_path):
            executor._log_sandbox_status()

        assert "Grok sandbox not enforced (Landlock)" in caplog.text


class TestACPProtocol:
    @pytest.mark.asyncio
    async def test_handshake_command_and_stdin(self, executor: GrokCLIExecutor):
        proc = _FakeProc(_success_events())
        events = await _stream(executor, proc)

        requests = _requests(proc)
        assert [request.get("method") for request in requests] == [
            "initialize",
            "session/new",
            "session/prompt",
        ]
        initialize = requests[0]["params"]
        assert initialize["protocolVersion"] == 1
        assert isinstance(initialize["protocolVersion"], int)
        new = requests[1]["params"]
        assert Path(new["cwd"]).is_absolute()
        assert new["_meta"]["systemPromptOverride"] == "You are Grok."
        assert isinstance(new["mcpServers"], list)
        aw = new["mcpServers"][0]
        assert aw["name"] == "aw"
        assert aw["args"] == ["-m", "core.mcp.server"]
        # ACP requires EnvVariable objects ([{name, value}]), not a mapping
        assert {entry["name"] for entry in aw["env"]} == {
            "ANIMAWORKS_ANIMA_DIR",
            "ANIMAWORKS_PROJECT_DIR",
            "ANIMAWORKS_SERVER_URL",
            "PYTHONPATH",
            "PATH",
        }
        assert all(isinstance(entry["value"], str) for entry in aw["env"])
        assert requests[2]["params"] == {
            "sessionId": "session-new",
            "prompt": [{"type": "text", "text": "hello"}],
        }
        done = [event for event in events if event["type"] == "done"]
        assert len(done) == 1

    def test_build_command_order(self, executor: GrokCLIExecutor):
        with patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"):
            assert executor._build_command() == [
                "/usr/bin/grok",
                "agent",
                "--always-approve",
                "--no-leader",
                "-m",
                "grok-4.5",
                "stdio",
            ]

    @pytest.mark.asyncio
    async def test_sandbox_spawn_injects_env_and_controlling_pty(
        self,
        executor: GrokCLIExecutor,
    ) -> None:
        proc = _FakeProc(_success_events())
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch.object(executor, "_write_sandbox_config", return_value=True),
            patch.object(executor, "_log_sandbox_status"),
            patch("pty.openpty", return_value=(101, 102)),
            patch("core.execution.engines.grok.executor.os.close") as close_fd,
            patch("asyncio.create_subprocess_exec", return_value=proc) as create,
        ):
            events = [
                event
                async for event in executor.execute_streaming(
                    "system",
                    "hello",
                    ContextTracker(model="grok/grok-4.5"),
                    trigger="heartbeat",
                )
            ]

        assert events[-1]["type"] == "done"
        kwargs = create.call_args.kwargs
        assert kwargs["env"]["GROK_SANDBOX"] == "animaworks"
        assert kwargs["pass_fds"] == (102,)
        assert callable(kwargs["preexec_fn"])
        assert close_fd.call_args_list == [call(102), call(101)]

    @pytest.mark.asyncio
    async def test_disabled_sandbox_removes_inherited_env(
        self,
        executor: GrokCLIExecutor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GROK_SANDBOX", "foreign-profile")
        proc = _FakeProc(_success_events())
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch.object(executor, "_write_sandbox_config", return_value=False),
            patch("asyncio.create_subprocess_exec", return_value=proc) as create,
        ):
            await executor.execute("hello", "system", trigger="heartbeat")

        kwargs = create.call_args.kwargs
        assert "GROK_SANDBOX" not in kwargs["env"]
        assert "pass_fds" not in kwargs

    @pytest.mark.asyncio
    async def test_skills_enabled_flag_set_when_sandbox_enabled(
        self,
        executor: GrokCLIExecutor,
    ) -> None:
        """Env must set GROK_CLAUDE_SKILLS_ENABLED=false to stop Grok loading native skills."""
        proc = _FakeProc(_success_events())
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch.object(executor, "_write_sandbox_config", return_value=True),
            patch("core.execution.engines.grok.executor.os.close"),
            patch("pty.openpty", return_value=(101, 102)),
            patch("asyncio.create_subprocess_exec", return_value=proc) as create,
        ):
            await executor.execute("hello", "system", trigger="heartbeat")

        kwargs = create.call_args.kwargs
        assert kwargs["env"]["GROK_CLAUDE_SKILLS_ENABLED"] == "false"

    @pytest.mark.asyncio
    async def test_skills_enabled_flag_set_when_sandbox_disabled(
        self,
        executor: GrokCLIExecutor,
    ) -> None:
        """GROK_CLAUDE_SKILLS_ENABLED=false must be set even when sandbox is off."""
        proc = _FakeProc(_success_events())
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch.object(executor, "_write_sandbox_config", return_value=False),
            patch("asyncio.create_subprocess_exec", return_value=proc) as create,
        ):
            await executor.execute("hello", "system", trigger="heartbeat")

        kwargs = create.call_args.kwargs
        assert kwargs["env"]["GROK_CLAUDE_SKILLS_ENABLED"] == "false"

        events = _success_events()
        events.insert(
            0,
            {
                "jsonrpc": "2.0",
                "id": 77,
                "method": "session/request_permission",
                "params": {
                    "options": [
                        {"optionId": "deny", "name": "Deny"},
                        {"optionId": "allow", "name": "Allow once"},
                    ]
                },
            },
        )
        proc = _FakeProc(events)
        await _stream(executor, proc)
        response = next(item for item in _requests(proc) if item.get("id") == 77)
        assert response["result"]["outcome"]["optionId"] == "allow"

    @pytest.mark.asyncio
    async def test_non_json_empty_and_unknown_updates_are_ignored(self, executor: GrokCLIExecutor):
        updates = [
            {"method": "session/update", "params": {"update": {"sessionUpdate": "user_message_chunk"}}},
            {"method": "session/update", "params": {"update": {"sessionUpdate": "future_update"}}},
        ]
        raw_events: list[dict | str | bytes] = ["not-json", b"\n"]
        raw_events.extend(_success_events(updates=updates))
        proc = _FakeProc(raw_events)
        events = await _stream(executor, proc)
        assert [event["type"] for event in events] == ["done"]


class TestEventConversion:
    @pytest.mark.asyncio
    async def test_text_thinking_tools_usage_and_done_contract(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "think"}}
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hello "}}
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tool-1",
                        "title": "fallback-name",
                        "rawInput": {"query": "hello"},
                        "_meta": {"x.ai/tool": {"name": "mcp__aw__search_memory"}},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "tool-1",
                        "status": "completed",
                        "rawOutput": {"answer": "found"},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "world"}}
                },
            },
        ]
        tracker = MagicMock()
        proc = _FakeProc(
            _success_events(
                updates=updates,
                usage={"inputTokens": 101, "outputTokens": 9, "cachedReadTokens": 80},
            )
        )
        events = await _stream(executor, proc, tracker=tracker)

        assert [event["type"] for event in events] == [
            "thinking_delta",
            "text_delta",
            "tool_start",
            "tool_end",
            "text_delta",
            "done",
        ]
        done = events[-1]
        assert done["full_text"] == "Hello world"
        assert isinstance(done["result_message"].num_turns, int)
        assert done["result_message"].session_id == "session-new"
        assert done["tool_call_records"][0]["tool_name"] == "search_memory"
        assert isinstance(done["tool_call_records"][0], dict)
        assert done["usage"] == {
            "input_tokens": 101,
            "output_tokens": 9,
            "cache_read_tokens": 80,
            "cache_write_tokens": 0,
        }
        tracker.update_from_usage.assert_called_once_with(done["usage"])

    @pytest.mark.asyncio
    async def test_usage_update_camel_case_sets_tracker_from_last_response(self, executor: GrokCLIExecutor):
        """Last usage_update (camelCase) drives context ratio, not cumulative usage."""
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "usage_update",
                        "usage": {
                            "inputTokens": 14588,
                            "outputTokens": 100,
                            "cachedReadTokens": 5376,
                            "cachedWriteTokens": 0,
                        },
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "hi"},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "usage_update",
                        "usage": {
                            "inputTokens": 272,
                            "outputTokens": 50,
                            "thoughtTokens": 10,
                            "cachedReadTokens": 19840,
                            "cachedWriteTokens": 10,
                            "totalTokens": 20172,
                        },
                    }
                },
            },
        ]
        tracker = ContextTracker(model="grok/grok-4.5")
        # Cumulative PromptUsage is intentionally much larger than last response.
        proc = _FakeProc(
            _success_events(
                updates=updates,
                usage={"inputTokens": 40211, "outputTokens": 211, "cachedReadTokens": 0},
            )
        )
        events = await _stream(executor, proc, tracker=tracker)

        # Cost log usage remains cumulative.
        assert events[-1]["usage"]["input_tokens"] == 40211
        # Context ring uses last usage_update: input + cachedRead + cachedWrite.
        expected_tokens = 272 + 19840 + 10
        assert tracker.usage_ratio == pytest.approx(expected_tokens / tracker.context_window)

    @pytest.mark.asyncio
    async def test_usage_update_snake_case_fields(self, executor: GrokCLIExecutor):
        """usage_update accepts snake_case field names (nested or flat)."""
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "usage_update",
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 200,
                        "cache_creation_input_tokens": 50,
                    }
                },
            },
        ]
        tracker = ContextTracker(model="grok/grok-4.5")
        proc = _FakeProc(_success_events(updates=updates))
        await _stream(executor, proc, tracker=tracker)
        expected_tokens = 100 + 200 + 50
        assert tracker.usage_ratio == pytest.approx(expected_tokens / tracker.context_window)

    @pytest.mark.asyncio
    async def test_meta_top_level_input_tokens_sets_context_without_cache_add(self, executor: GrokCLIExecutor):
        """Grok CLI 0.2.x: _meta top-level inputTokens = last-call full context.

        Real wire shape (2 model calls): top-level inputTokens is cache-inclusive
        current context; nested usage.inputTokens is cumulative spend.  No
        usage_update is emitted.  Tracker must use top-level only (not add
        cachedReadTokens again).
        """
        tracker = ContextTracker(model="grok/grok-4.5")
        proc = _FakeProc(
            _success_events(
                usage={
                    "inputTokens": 40202,
                    "outputTokens": 87,
                    "totalTokens": 40289,
                    "cachedReadTokens": 25472,
                },
                meta_top_level={
                    "totalTokens": 20195,
                    "modelId": "grok-4.5",
                    "inputTokens": 20153,
                    "outputTokens": 42,
                    "cachedReadTokens": 19968,
                    "reasoningTokens": 23,
                },
            )
        )
        events = await _stream(executor, proc, tracker=tracker)

        # Cost log remains cumulative nested usage.
        assert events[-1]["usage"]["input_tokens"] == 40202
        # Context ring = top-level inputTokens only (already cache-inclusive).
        assert tracker.usage_ratio == pytest.approx(20153 / tracker.context_window)

    @pytest.mark.asyncio
    async def test_no_usage_update_falls_back_to_cumulative_usage(self, executor: GrokCLIExecutor):
        """Without usage_update or _meta top-level inputTokens, use cumulative."""
        tracker = MagicMock()
        proc = _FakeProc(
            _success_events(
                usage={"inputTokens": 101, "outputTokens": 9, "cachedReadTokens": 80},
            )
        )
        events = await _stream(executor, proc, tracker=tracker)
        done = events[-1]
        tracker.update_from_usage.assert_called_once_with(done["usage"])
        tracker.update_from_message_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_tool_update(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "t2",
                        "title": "mcp_aw_bash",
                        "rawInput": "x" * 700,
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "t2",
                        "status": "failed",
                        "rawOutput": "bad",
                    }
                },
            },
        ]
        events = await _stream(executor, _FakeProc(_success_events(updates=updates)))
        end = next(event for event in events if event["type"] == "tool_end")
        assert end["tool_name"] == "bash"
        assert end["is_error"] is True
        record = events[-1]["tool_call_records"][0]
        assert record["is_error"] is True
        assert len(record["input_summary"]) <= 503

    @pytest.mark.asyncio
    async def test_blocking_result_uses_dataclasses(self, executor: GrokCLIExecutor):
        updates = [
            {
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "answer"}}},
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "t",
                        "title": "read_file",
                        "rawInput": {"path": "x"},
                    }
                },
            },
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "t",
                        "status": "completed",
                        "rawOutput": "ok",
                    }
                },
            },
        ]
        proc = _FakeProc(_success_events(updates=updates))
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch("asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await executor.execute("hello", "system", trigger="heartbeat")
        assert isinstance(result, ExecutionResult)
        assert result.text == "answer"
        assert isinstance(result.tool_call_records[0], ToolCallRecord)
        assert result.usage is not None and result.usage.input_tokens == 12


class TestSessions:
    @pytest.mark.asyncio
    async def test_chat_persists_then_resumes_without_system_override(self, executor: GrokCLIExecutor, anima_dir: Path):
        first = _FakeProc(_success_events(session_id="saved"))
        await _stream(executor, first, trigger="message:user", thread_id="thread-a")
        assert _load_grok_session_id(anima_dir, "chat", "thread-a") == ("saved", 1)

        second = _FakeProc(_success_events(session_id="saved", load=True))
        await _stream(executor, second, trigger="chat", thread_id="thread-a")
        requests = _requests(second)
        assert requests[1]["method"] == "session/load"
        assert requests[1]["params"]["sessionId"] == "saved"
        assert "_meta" not in requests[1]["params"]
        assert _load_grok_session_id(anima_dir, "chat", "thread-a") == ("saved", 2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trigger", ["heartbeat", "cron:daily", "task:run"])
    async def test_non_chat_triggers_are_fresh(self, executor: GrokCLIExecutor, trigger: str):
        proc = _FakeProc(_success_events())
        await _stream(executor, proc, trigger=trigger)
        assert _requests(proc)[1]["method"] == "session/new"

    @pytest.mark.asyncio
    async def test_resume_failure_retries_once_fresh(self, executor: GrokCLIExecutor, anima_dir: Path):
        _GROK_SESSION_IDS.save(anima_dir, "stale", "chat", "default", 3)
        failed = _FakeProc(
            [
                {"id": 1, "result": {}},
                {"id": 2, "error": {"code": -32000, "message": "session not found"}},
            ]
        )
        fresh = _FakeProc(_success_events(session_id="recovered"))
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
            patch("asyncio.create_subprocess_exec", side_effect=[failed, fresh]) as create,
        ):
            _events = [
                event
                async for event in executor.execute_streaming(
                    "system", "hello", ContextTracker(model="grok/grok-4.5"), trigger="chat"
                )
            ]
        assert create.call_count == 2
        assert _requests(failed)[1]["method"] == "session/load"
        assert _requests(fresh)[1]["method"] == "session/new"
        assert _load_grok_session_id(anima_dir, "chat") == ("recovered", 1)

    @pytest.mark.asyncio
    async def test_turn_rotation_and_pending(self, executor: GrokCLIExecutor, anima_dir: Path):
        _GROK_SESSION_IDS.save(anima_dir, "old", "chat", "default", _MAX_RESUME_TURNS)
        rotated = _FakeProc(_success_events(session_id="fresh"))
        rotated_events = await _stream(executor, rotated, trigger="chat")
        assert _requests(rotated)[1]["method"] == "session/new"
        assert rotated_events[-1]["session_rotation_pending"] is False

        _GROK_SESSION_IDS.save(anima_dir, "almost", "chat", "default", _MAX_RESUME_TURNS - 1)
        pending = _FakeProc(_success_events(session_id="almost", load=True))
        pending_events = await _stream(executor, pending, trigger="chat")
        assert pending_events[-1]["session_rotation_pending"] is True
        assert _load_grok_session_id(anima_dir, "chat") == ("almost", _MAX_RESUME_TURNS)


class TestTerminalPaths:
    @pytest.mark.asyncio
    async def test_blocking_execute_preserves_terminal_error_metadata(
        self,
        executor: GrokCLIExecutor,
    ) -> None:
        async def stream(*args, **kwargs):
            yield {
                "type": "error",
                "message": "quota exhausted",
                "terminal": True,
                "reason": "quota_exhausted",
            }
            yield executor._done_event("", [], TokenUsage(), "", 0)

        executor.execute_streaming = stream  # type: ignore[method-assign]
        result = await executor.execute("hello", "system")

        assert result.error is True
        assert result.reason == "quota_exhausted"
        assert result.text == "quota exhausted"

    @pytest.mark.asyncio
    async def test_real_quota_error_precedes_legacy_text_and_done(
        self,
        executor: GrokCLIExecutor,
    ) -> None:
        proc = _FakeProc(
            [
                {"id": 1, "result": {}},
                {"id": 2, "result": {"sessionId": "session-new"}},
                {"id": 3, "error": {"code": -32603, "message": "Internal error"}},
            ]
        )
        guard = MagicMock()
        guard.config.default_block_seconds = 60
        guard.config.quota_block_seconds = 1800
        with (
            patch(
                "core.execution.engines.grok.executor._resolve_real_error",
                return_value={
                    "status_code": 402,
                    "message": _REAL_GROK_QUOTA_ERROR,
                },
            ),
            patch("core.execution.engine_base.get_rate_guard", return_value=guard),
        ):
            events = await _stream(executor, proc)

        error = next(event for event in events if event["type"] == "error")
        assert not any(event["type"] == "text_delta" for event in events)
        assert error["message"] == f"[Grok CLI Error: {_REAL_GROK_QUOTA_ERROR}]"
        assert error["terminal"] is True
        assert error["reason"] == "quota_exhausted"
        assert events[-1]["type"] == "done"
        assert events[-1]["full_text"] == ""
        guard.report_block.assert_called_once_with(
            "grok:grok",
            1800,
            "quota_exhausted",
            reset_in_s=None,
        )

    @pytest.mark.asyncio
    async def test_not_installed_done_once(self, executor: GrokCLIExecutor):
        with patch("core.execution.engines.grok.executor._find_grok_binary", return_value=None):
            events = [
                event
                async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
            ]
        assert len([event for event in events if event["type"] == "done"]) == 1
        done = events[-1]
        assert isinstance(done["full_text"], str)
        assert done["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_preexisting_interrupt_done_once(self, executor: GrokCLIExecutor):
        executor._interrupt_event = asyncio.Event()
        executor._interrupt_event.set()
        events = [
            event
            async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
        ]
        assert [event["type"] for event in events] == ["done"]
        assert "interrupted" in events[0]["full_text"].lower()

    @pytest.mark.asyncio
    async def test_file_not_found_done_once(self, executor: GrokCLIExecutor):
        with (
            patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/missing/grok"),
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
        ):
            events = [
                event
                async for event in executor.execute_streaming("system", "hello", ContextTracker(model="grok/grok-4.5"))
            ]
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "grok" in next(event for event in events if event["type"] == "error")["message"].lower()

    @pytest.mark.asyncio
    async def test_auth_error_from_stderr(self, executor: GrokCLIExecutor):
        proc = _FakeProc([], returncode=1, stderr=b"Unauthenticated: run grok login")
        events = await _stream(executor, proc)
        assert len([event for event in events if event["type"] == "done"]) == 1
        message = next(event for event in events if event["type"] == "error")["message"]
        assert "login" in message.lower() or "認証" in message

    @pytest.mark.asyncio
    async def test_idle_timeout_kills_stalled_stream(self, executor: GrokCLIExecutor):
        # A child that emits the handshake but then goes silent must be killed
        # once the idle gap between events exceeds the progress timeout, and
        # surface a single terminal done event with the timeout message.
        proc = _FakeProc(_success_events(), returncode=None)
        served = 0

        async def stall_after_session() -> bytes:
            # Serve only the two handshake responses (initialize + session/new),
            # then hang so the event-loop readline trips the idle timeout while
            # awaiting the prompt result — not EOF.
            nonlocal served
            if served < 2:
                served += 1
                return proc.stdout.lines.pop(0)
            await asyncio.sleep(3600)
            return b""

        proc.stdout.readline = stall_after_session  # type: ignore[method-assign]
        from core.execution.watchdog import Watchdog

        with (
            patch("core.execution.events.Watchdog", return_value=Watchdog(0.01)),
            pytest.raises(TimeoutError, match="idle"),
        ):
            await _stream(executor, proc)
        assert signal.SIGTERM in proc.sent_signals

    @pytest.mark.asyncio
    async def test_inflight_interrupt_sends_cancel(self, executor: GrokCLIExecutor):
        executor._interrupt_event = asyncio.Event()
        proc = _FakeProc(_success_events(), returncode=None)
        original_readline = proc.stdout.readline
        reads = 0

        async def interrupt_after_session() -> bytes:
            nonlocal reads
            line = await original_readline()
            reads += 1
            if reads == 2:
                executor._interrupt_event.set()
            return line

        proc.stdout.readline = interrupt_after_session  # type: ignore[method-assign]
        events = await _stream(executor, proc)
        methods = [request.get("method") for request in _requests(proc)]
        assert "session/cancel" in methods
        assert len([event for event in events if event["type"] == "done"]) == 1
        assert "interrupted" in events[-1]["full_text"].lower()

    @pytest.mark.asyncio
    async def test_kill_escalates_after_grace_period(self, executor: GrokCLIExecutor):
        proc = MagicMock()
        proc.returncode = None
        proc.wait = AsyncMock(side_effect=[TimeoutError, -9])
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        await executor._kill_process(proc, timeout=0)
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_sweeps_the_child_process_group(self, executor: GrokCLIExecutor):
        """The CLI exiting must not leave its bash/python descendants running.

        Regression: on 2026-08-04 a timed-out Grok CLI left `python3 -` shells
        spinning at 100% CPU for 10+ hours, reparented to systemd --user.
        """
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 4321
        proc.wait = AsyncMock(return_value=0)
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        with (
            patch("core.execution.engines.grok.executor.os.getpgid", return_value=9999),
            patch("core.execution.engines.grok.executor.os.getpgrp", return_value=1111),
            patch("core.execution.engines.grok.executor.os.killpg") as killpg,
        ):
            await executor._kill_process(proc, timeout=0)
        assert call(9999, signal.SIGKILL) in killpg.call_args_list

    @pytest.mark.asyncio
    async def test_kill_never_signals_its_own_process_group(self, executor: GrokCLIExecutor):
        """If setsid() failed, the child shares our group — killing it would
        take down the AnimaWorks daemon itself."""
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 4321
        proc.wait = AsyncMock(return_value=0)
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        with (
            patch("core.execution.engines.grok.executor.os.getpgid", return_value=1111),
            patch("core.execution.engines.grok.executor.os.getpgrp", return_value=1111),
            patch("core.execution.engines.grok.executor.os.killpg") as killpg,
        ):
            await executor._kill_process(proc, timeout=0)
        killpg.assert_not_called()
