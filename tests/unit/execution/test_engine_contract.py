"""Engine contract golden tests (E0).

For each engine (S/C/D/G/X/A) we drive the **current** implementation with a
scripted fake input (fake CLI lines / fake SDK messages / fake LiteLLM chunks)
and record the emitted event sequence / ExecutionResult as well as the
activity-log entries the engine writes.  The recorded output is committed as a
golden file so that later engine middle-layer refactors (E1–E6) can be
checked for regressions.

For every engine we fix **both** execution paths that the current code
provides where applicable:

* ``execute()`` — captured as the serialised ``ExecutionResult`` (text,
  tool_call_records, usage, session_id, num_turns, error, reason …).
* ``execute_streaming()`` — for engines where ``supports_streaming`` is true
  (S, C, D, G, X, A), captured as the emitted event sequence plus the final
  result carried on the terminal ``done`` event.
* ``cursor`` (D) reports ``supports_streaming == True`` and its stream path
  is captured alongside the other CLI engines.

Activity (what the executor writes to the anima's ``activity_log``) is
captured per path so E5 can detect double-logging or missed logging.

P0C additionally captures Mode A's blocking, token-level streaming, and
Ollama iteration-level streaming paths for text-only, one-tool, tool-error,
and runaway-limit scenarios. Mode S blocking has the corresponding text,
tool, tool-error, and interruption scenarios. The files record current
behavior—including path-specific event ordering and result differences—rather
than imposing parity.

Each engine also has one abnormal scenario:

* CLI engines (D, G, X): the CLI exits non-zero / the process emits an error.
* SDK engines (S, C): the SDK raises an error.
* A (LiteLLM): the LLM call raises a non-retryable exception.

Golden data lives in ``tests/unit/execution/golden/``:

* ``<id>.input.jsonl`` — scripted fake input (one JSON payload per line).
* ``<id>.expected.json`` — captured output + activity (committed).

``<id>`` is ``<engine>__<path>-<scenario>`` — e.g. ``codex__execute-happy``,
``codex__stream-happy``, ``codex__execute-error``.

Regenerate the golden files after an intentional behaviour change::

    UPDATE_GOLDEN=1 uv run --no-sync pytest tests/unit/execution/test_engine_contract.py -q

The golden captures the *actual* current output — never an idealised
expectation.  The ``ts`` field of activity entries (wall-clock timestamp) is
excluded from the comparison because it is inherently volatile.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.prompt.context import ContextTracker
from core.schemas import ModelConfig

pytestmark = pytest.mark.asyncio

GOLDEN_DIR = Path(__file__).parent / "golden"
REGEN = os.environ.get("UPDATE_GOLDEN") == "1"

# (engine, path, scenario) — the full set of contract scenarios.
_SCENARIOS: list[tuple[str, str, str]] = [
    ("agent_sdk", "execute", "happy"),
    ("agent_sdk", "stream", "happy"),
    ("agent_sdk", "execute", "error"),
    ("agent_sdk", "execute", "text"),
    ("agent_sdk", "execute", "tool"),
    ("agent_sdk", "execute", "tool-error"),
    ("agent_sdk", "execute", "limit"),
    ("codex", "execute", "happy"),
    ("codex", "stream", "happy"),
    ("codex", "execute", "error"),
    ("cursor", "execute", "happy"),
    ("cursor", "execute", "error"),
    ("cursor", "stream", "happy"),
    ("gemini", "execute", "happy"),
    ("gemini", "stream", "happy"),
    ("gemini", "execute", "error"),
    ("grok", "execute", "happy"),
    ("grok", "stream", "happy"),
    ("grok", "execute", "error"),
    ("litellm", "execute", "happy"),
    ("litellm", "stream", "happy"),
    ("litellm", "execute", "error"),
    ("litellm", "execute", "text"),
    ("litellm", "execute", "tool"),
    ("litellm", "execute", "tool-error"),
    ("litellm", "execute", "limit"),
    ("litellm", "stream", "text"),
    ("litellm", "stream", "tool"),
    ("litellm", "stream", "tool-error"),
    ("litellm", "stream", "limit"),
    ("litellm", "iteration", "text"),
    ("litellm", "iteration", "tool"),
    ("litellm", "iteration", "tool-error"),
    ("litellm", "iteration", "limit"),
]


# ── Normalisation helpers ────────────────────────────────────


def _norm(value: Any) -> Any:
    """Convert a captured value into a stable, JSON-serialisable form."""
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, set):
        return sorted(_norm(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _norm(to_dict())
        except Exception:
            pass
    d = getattr(value, "__dict__", None)
    if d:
        return _norm(d)
    return str(value)


def _read_activity(anima_dir: Path) -> list[dict]:
    """Read all JSONL entries written to the anima's activity_log dir.

    The ``ts`` field (wall-clock) is dropped because it is volatile.
    """
    entries: list[dict] = []
    log_dir = anima_dir / "activity_log"
    if not log_dir.exists():
        return entries
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entry.pop("ts", None)
                entries.append(_norm(entry))
    return entries


def _serialise_result(result) -> dict:
    """Serialise an ExecutionResult into a stable JSON dict."""
    usage = None
    if result.usage is not None:
        if hasattr(result.usage, "to_dict"):
            _u = result.usage.to_dict()
        else:
            _u = dict(result.usage)
        usage = {k: (v if isinstance(v, int) else 0) for k, v in _u.items()}
    return {
        "text": result.text,
        "tool_call_records": _norm(result.tool_call_records),
        "usage": usage,
        "truncated": result.truncated,
        "error": result.error,
        "reason": result.reason,
        "session_rotation_pending": result.session_rotation_pending,
        "force_chain": result.force_chain,
        "task_compact_requested": result.task_compact_requested,
        "session_id": (
            getattr(result.result_message, "session_id", None) if result.result_message is not None else None
        ),
        "num_turns": (getattr(result.result_message, "num_turns", None) if result.result_message is not None else None),
    }


def _load_input(golden_id: str) -> list[dict]:
    """Load the scripted input from ``golden/<id>.input.jsonl``."""
    path = GOLDEN_DIR / f"{golden_id}.input.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing golden input file: {path}")
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _compare_or_write(golden_id: str, actual: dict) -> None:
    """Write the golden file when regenerating, else assert equality."""
    path = GOLDEN_DIR / f"{golden_id}.expected.json"
    if REGEN:
        path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(f"golden file missing (run with UPDATE_GOLDEN=1): {path}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        import difflib

        a = json.dumps(actual, indent=2, ensure_ascii=False).splitlines()
        b = json.dumps(expected, indent=2, ensure_ascii=False).splitlines()
        diff = "\n".join(difflib.unified_diff(b, a, "expected", "actual"))
        raise AssertionError(f"engine contract mismatch for {golden_id}:\n{diff}")


def _anima_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / "animas" / name
    d.mkdir(parents=True)
    (d / "IDENTITY.md").write_text("# Test Anima\n", encoding="utf-8")
    (d / "shortterm" / "chat").mkdir(parents=True)
    (d / "shortterm" / "heartbeat").mkdir(parents=True)
    (d / "state").mkdir()
    (d / "state" / "current_state.md").write_text("status: idle\n", encoding="utf-8")
    (d / "permissions.md").write_text("", encoding="utf-8")
    return d


# ── Gemelli (G) driver ──────────────────────────────────────


def _make_ndjson_bytes(events: list[dict]) -> bytes:
    return b"".join(json.dumps(e).encode() + b"\n" for e in events)


def _gemini_proc(events: list[dict], returncode: int = 0, stderr: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.wait = AsyncMock()
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr)
    proc.stdout = AsyncMock()
    lines = iter(_make_ndjson_bytes(events).split(b"\n"))
    proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))
    return proc


def _gemini_executor(anima_dir: Path):
    from core.execution.engines.gemini.executor import GeminiCLIExecutor

    mc = ModelConfig(
        model="gemini/2.5-pro",
        max_tokens=4096,
        credential="gemini",
        context_threshold=0.5,
    )
    return GeminiCLIExecutor(
        model_config=mc,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


async def _drive_gemini_execute(anima_dir: Path, golden_id: str, path: str | None = None) -> dict:
    lines = _load_input(golden_id)
    executor = _gemini_executor(anima_dir)
    head = lines[0] if lines else {}
    if head.get("scenario") == "cli_error":
        proc = _gemini_proc([], returncode=head.get("returncode", 1), stderr=head.get("stderr", b"").encode())
    else:
        proc = _gemini_proc(lines)
    tracker = ContextTracker(model="gemini/2.5-pro", threshold=0.5)
    with (
        patch("core.execution.engines.gemini.executor._find_gemini_binary", return_value="/usr/bin/gemini"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        result = await executor.execute(system_prompt="You are helpful", prompt="hello", tracker=tracker)
    return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}


async def _drive_gemini_stream(anima_dir: Path, golden_id: str, path: str | None = None) -> dict:
    lines = _load_input(golden_id)
    executor = _gemini_executor(anima_dir)
    proc = _gemini_proc(lines)
    tracker = ContextTracker(model="gemini/2.5-pro", threshold=0.5)
    with (
        patch("core.execution.engines.gemini.executor._find_gemini_binary", return_value="/usr/bin/gemini"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        events = [
            e
            async for e in executor.execute_streaming(
                system_prompt="You are helpful",
                prompt="hello",
                tracker=tracker,
            )
        ]
    return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}


# ── Cursor (D) driver ───────────────────────────────────────


def _cursor_proc(events: list[dict], returncode: int = 0, stderr: bytes = b"") -> AsyncMock:
    proc = AsyncMock()
    proc.stdout = AsyncMock()
    proc.stderr = AsyncMock()
    proc.returncode = returncode
    proc.wait = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr)
    lines = iter(_make_ndjson_bytes(events).split(b"\n"))
    proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines, b""))
    return proc


def _cursor_executor(anima_dir: Path):
    from core.execution.engines.cursor.executor import CursorAgentExecutor

    mc = ModelConfig(
        model="cursor/claude-4-sonnet",
        max_tokens=4096,
        credential="cursor",
        context_threshold=0.5,
    )
    return CursorAgentExecutor(
        model_config=mc,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


async def _drive_cursor_execute(anima_dir: Path, golden_id: str, path: str | None = None) -> dict:
    lines = _load_input(golden_id)
    executor = _cursor_executor(anima_dir)
    head = lines[0] if lines else {}
    if head.get("scenario") == "cli_error":
        proc = _cursor_proc([], returncode=head.get("returncode", 1), stderr=head.get("stderr", b"").encode())
    else:
        proc = _cursor_proc(lines)
    with (
        patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
        patch.object(executor, "_ensure_workspace"),
        patch.object(executor, "_write_mcp_config"),
        patch.object(executor, "_write_cursor_rules"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        result = await executor.execute(prompt="use a tool", system_prompt="You are helpful")
    return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}


async def _drive_cursor_stream(anima_dir: Path, golden_id: str, path: str | None = None) -> dict:
    lines = _load_input(golden_id)
    executor = _cursor_executor(anima_dir)
    proc = _cursor_proc(lines)
    tracker = ContextTracker(model="cursor/claude-4-sonnet", threshold=0.5)
    with (
        patch.object(executor, "_find_binary", return_value="/usr/bin/agent"),
        patch.object(executor, "_ensure_workspace"),
        patch.object(executor, "_write_mcp_config"),
        patch.object(executor, "_write_cursor_rules"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        events = [
            event
            async for event in executor.execute_streaming(
                system_prompt="You are helpful",
                prompt="use a tool",
                tracker=tracker,
            )
        ]
    return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}


# ── Grok (X) driver ────────────────────────────────────────


class _FakeStdin:
    def __init__(self) -> None:
        self.written = b""

    def write(self, data: bytes) -> None:
        self.written += data

    async def drain(self) -> None:
        return None


class _FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)

    async def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""

    async def read(self) -> bytes:
        return b""


class _FakeXProc:
    def __init__(self, out_lines: list[bytes], returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(out_lines)
        self.stderr = _FakeStream([])
        self.returncode = returncode
        self.pid = 1234

    async def wait(self) -> int:
        return self.returncode

    def send_signal(self, _sig):
        pass

    def kill(self) -> None:
        self.returncode = -9


def _grok_wire_lines(updates: list[dict], *, session_id: str = "session-new") -> list[bytes]:
    """Build the full ACP wire output covering init, session, updates, result."""
    events: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "result": {"agentCapabilities": {"loadSession": True}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": session_id}},
    ]
    events.extend(updates)
    meta = {
        "sessionId": session_id,
        "usage": {"inputTokens": 12, "outputTokens": 4, "cachedReadTokens": 3},
    }
    events.append({"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "end_turn", "_meta": meta}})
    return [json.dumps(e).encode() + b"\n" for e in events]


def _grok_error_wire(error: dict) -> list[bytes]:
    """Build an ACP wire where session/prompt returns an error."""
    events: list[dict] = [
        {"jsonrpc": "2.0", "id": 1, "result": {"agentCapabilities": {"loadSession": True}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "session-new"}},
        {"jsonrpc": "2.0", "id": 3, "error": error},
    ]
    return [json.dumps(e).encode() + b"\n" for e in events]


def _grok_executor(anima_dir: Path):
    from core.execution.engines.grok.executor import GrokCLIExecutor

    mc = ModelConfig(
        model="grok/grok-4.5",
        max_tokens=4096,
        credential="grok",
        context_threshold=0.5,
    )
    return GrokCLIExecutor(mc, anima_dir, tool_registry=["web_search"])


async def _drive_grok(anima_dir: Path, golden_id: str, path: str) -> dict:
    lines = _load_input(golden_id)
    executor = _grok_executor(anima_dir)
    tracker = ContextTracker(model="grok/grok-4.5", threshold=0.5)
    head = lines[0] if lines else {}
    if head.get("scenario") == "acp_error":
        wire = _grok_error_wire({"code": head.get("code", -32000), "message": head.get("message", "boom failure")})
        proc = _FakeXProc(wire)
    else:
        proc = _FakeXProc(_grok_wire_lines(lines))
    with (
        patch("core.execution.engines.grok.executor._find_grok_binary", return_value="/usr/bin/grok"),
        patch("asyncio.create_subprocess_exec", return_value=proc),
    ):
        if path == "stream":
            events = [
                e
                async for e in executor.execute_streaming(
                    "You are Grok.",
                    "hello",
                    tracker,
                    trigger="heartbeat",
                    thread_id="default",
                )
            ]
            return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}
        result = await executor.execute(
            "hello",
            "You are Grok.",
            tracker,
            trigger="heartbeat",
            thread_id="default",
        )
    return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}


# ── Agent SDK (S) driver ────────────────────────────────────


def _sdk_messages_from_input(lines: list[dict]) -> list[Any]:
    """Translate scripted input lines into SDK message objects."""
    from tests.helpers.mocks import (
        MockAssistantMessage,
        MockResultMessage,
        MockStreamEvent,
        MockSystemMessage,
        MockTextBlock,
        MockToolResultBlock,
        MockToolUseBlock,
        MockUserMessage,
    )

    messages: list[Any] = []
    for line in lines:
        kind = line["kind"]
        if kind == "stream_event":
            messages.append(MockStreamEvent(line["event"], session_id=line.get("session_id", "mock-session")))
        elif kind == "assistant":
            blocks = []
            for b in line["blocks"]:
                if b["type"] == "text":
                    blocks.append(MockTextBlock(b["text"]))
                elif b["type"] == "tool_use":
                    blocks.append(MockToolUseBlock(b["name"], b.get("input", {}), id=b.get("id", "tu_1")))
            messages.append(MockAssistantMessage(blocks))
        elif kind == "tool_result":
            messages.append(
                MockUserMessage(
                    [
                        MockToolResultBlock(
                            line["tool_use_id"], content=line.get("content", ""), is_error=line.get("is_error")
                        )
                    ]
                )
            )
        elif kind == "user":
            messages.append(MockUserMessage([]))
        elif kind == "system":
            messages.append(MockSystemMessage(line.get("subtype", ""), line.get("data", {})))
        elif kind == "result":
            messages.append(
                MockResultMessage(
                    usage=line.get("usage"),
                    num_turns=line.get("num_turns", 1),
                    session_id=line.get("session_id", "test-session-001"),
                )
            )
    return messages


class _SDKHappyClient:
    def __init__(self, messages: list, **kwargs):
        self._messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def query(self, prompt):
        return None

    async def receive_response(self):
        for m in self._messages:
            yield m

    async def receive_messages(self):
        for m in self._messages:
            yield m


class _SDKErrorClient:
    def __init__(self, error_msg: str, error_type: type = Exception, **kwargs):
        self._error_msg = error_msg
        self._error_type = error_type

    async def __aenter__(self):
        # Model an SDK connection error: raise as soon as the client connects.
        raise self._error_type(self._error_msg)

    async def __aexit__(self, *a):
        return False

    async def query(self, prompt):
        return None

    async def receive_response(self):
        return
        yield  # pragma: no cover

    async def receive_messages(self):
        return
        yield  # pragma: no cover


def _install_sdk_module(_Client) -> dict:
    from tests.helpers.mocks import (
        MockAssistantMessage,
        MockResultMessage,
        MockStreamEvent,
        MockSystemMessage,
        MockTextBlock,
        MockToolResultBlock,
        MockToolUseBlock,
        MockUserMessage,
    )

    mock_module = MagicMock()
    mock_module.ClaudeSDKClient = _Client
    mock_module.AssistantMessage = MockAssistantMessage
    mock_module.ResultMessage = MockResultMessage
    mock_module.TextBlock = MockTextBlock
    mock_module.ToolUseBlock = MockToolUseBlock
    mock_module.ToolResultBlock = MockToolResultBlock
    mock_module.UserMessage = MockUserMessage
    mock_module.SystemMessage = MockSystemMessage
    mock_module.ClaudeAgentOptions = MagicMock
    mock_module.HookMatcher = MagicMock
    mock_module.ClaudeSDKError = Exception
    mock_module.ProcessError = Exception
    mock_types = MagicMock()
    mock_types.StreamEvent = MockStreamEvent
    mock_module.types = mock_types

    saved: dict = {}
    for key in ("claude_agent_sdk", "claude_agent_sdk.types"):
        saved[key] = sys.modules.get(key)
        sys.modules[key] = mock_types if key == "claude_agent_sdk.types" else mock_module
    return saved


async def _drive_agent_sdk(anima_dir: Path, golden_id: str, path: str) -> dict:
    from core.execution.engines.claude.executor import AgentSDKExecutor

    lines = _load_input(golden_id)
    head = lines[0] if lines else {}
    if head.get("scenario") == "sdk_error":

        def _error_client(**kw):
            return _SDKErrorClient(head.get("message", "SDK exploded"))

        _Client = _error_client
    else:
        # execute() reads via receive_response (blocking SDK), streaming via receive_messages.
        content_lines = [ln for ln in lines if ln.get("kind") != "stream_event"] if path == "execute" else lines
        messages = _sdk_messages_from_input(content_lines)

        def _happy_client(**kw):
            return _SDKHappyClient(messages)

        _Client = _happy_client

    saved = _install_sdk_module(_Client)
    mc = ModelConfig(model="claude-sonnet-4-6", api_key="sk-test", context_threshold=0.5)
    interrupt_event = asyncio.Event() if head.get("scenario") == "interrupt" else None
    if interrupt_event is not None:
        interrupt_event.set()
    executor = AgentSDKExecutor(model_config=mc, anima_dir=anima_dir, interrupt_event=interrupt_event)
    tracker = ContextTracker(model="claude-sonnet-4-6", threshold=0.5)
    try:
        if path == "stream":
            events = [e async for e in executor.execute_streaming(system_prompt="sys", prompt="test", tracker=tracker)]
            return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}
        result = await executor.execute(system_prompt="sys", prompt="test", tracker=tracker)
        return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


# ── Codex (C) driver ───────────────────────────────────────


def _mock_codex_result_thread(thread_id: str, lines: list[dict]):
    """Build a mock codex result thread / turn from scripted input lines."""
    items = []
    final_response = None
    usage = None
    for line in lines:
        kind = line["kind"]
        if kind == "text":
            item = SimpleNamespace(type="agent_message", text=line["text"])
            items.append(item)
        elif kind == "command":
            item = SimpleNamespace(
                type="command_execution",
                id=line.get("id", "cmd-1"),
                command=line.get("command", ""),
                aggregated_output=line.get("output", ""),
                exit_code=line.get("exit_code", 0),
            )
            items.append(item)
        elif kind == "file_change":
            item = SimpleNamespace(
                type="file_change",
                id=line.get("id", "fc-1"),
                path=line.get("path", ""),
                status=line.get("status", "changed"),
            )
            items.append(item)
        elif kind == "final_response":
            final_response = line["text"]
        elif kind == "usage":
            usage = SimpleNamespace(
                input_tokens=line.get("input_tokens", 0),
                output_tokens=line.get("output_tokens", 0),
            )

    if final_response and not any(getattr(it, "text", None) == final_response for it in items):
        items.append(SimpleNamespace(type="agent_message", text=final_response))

    async def fake_events():
        for item in items:
            yield SimpleNamespace(type="item.completed", item=item)
        yield SimpleNamespace(type="turn.completed", usage=usage)

    turn = MagicMock()
    turn.final_response = final_response
    turn.items = items
    turn.usage = usage
    turn.stream.return_value = fake_events()
    thread = MagicMock()
    thread.turn = AsyncMock(return_value=turn)
    thread.id = thread_id
    return thread


def _codex_executor(anima_dir: Path):
    from core.execution.engines.codex.executor import CodexSDKExecutor

    mc = ModelConfig(
        model="codex/o4-mini",
        max_tokens=4096,
        credential="openai",
        api_key="test-key-123",
        context_threshold=0.50,
    )
    return CodexSDKExecutor(
        model_config=mc,
        anima_dir=anima_dir,
        tool_registry=["web_search"],
        personal_tools={},
    )


async def _drive_codex(anima_dir: Path, golden_id: str, path: str) -> dict:
    from core.execution.engines.codex.executor import CodexSDKExecutor

    lines = _load_input(golden_id)
    executor: CodexSDKExecutor = _codex_executor(anima_dir)
    head = lines[0] if lines else {}

    if head.get("scenario") == "sdk_error":
        with patch.object(
            executor,
            "_create_codex_client",
            side_effect=RuntimeError(head.get("message", "SDK exploded")),
        ):
            result = await executor.execute(prompt="Hello", system_prompt="You are a test assistant")
        return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}

    success_lines = list(lines)
    thread = _mock_codex_result_thread("thread-contract", success_lines)
    mock_codex = MagicMock()
    mock_codex.thread_start = AsyncMock(return_value=thread)
    mock_codex.close = AsyncMock()
    with patch.object(executor, "_create_codex_client", return_value=mock_codex):
        if path == "stream":
            tracker = ContextTracker(model="codex/o4-mini", threshold=0.50)
            events = [
                e
                async for e in executor.execute_streaming(
                    "You are a test assistant",
                    "Hello",
                    tracker,
                )
            ]
            return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}
        result = await executor.execute(prompt="Hello", system_prompt="You are a test assistant")
        return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}


# ── LiteLLM (A) driver ─────────────────────────────────────


def _litellm_responses_from_input(lines: list[dict]):
    from tests.helpers.mocks import make_litellm_response, make_tool_call

    responses = []
    for line in lines:
        tool_calls = None
        if line.get("tool_calls"):
            tool_calls = [
                make_tool_call(tc["name"], json.loads(tc.get("arguments", "{}")), call_id=tc["id"])
                for tc in line["tool_calls"]
            ]
        resp = make_litellm_response(
            content=line.get("content", ""),
            tool_calls=tool_calls,
            prompt_tokens=line.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=line.get("usage", {}).get("completion_tokens", 0),
        )
        _u = {
            "prompt_tokens": line.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": line.get("usage", {}).get("completion_tokens", 0),
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "prompt_tokens_details": None,
        }
        resp.usage.get = lambda key, default=0, _u=_u: _u.get(key, default)
        resp.usage.cache_read_input_tokens = 0
        resp.usage.cache_creation_input_tokens = 0
        resp.usage.prompt_tokens_details = None
        responses.append(resp)
    return responses


def _litellm_executor(anima_dir: Path, model: str = "openai/gpt-4o"):
    from core.tooling.handler import ToolHandler

    mc = ModelConfig(
        model=model,
        api_key="sk-test",
        max_tokens=1024,
        context_threshold=0.50,
    )
    memory = MagicMock()
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    memory.anima_dir = anima_dir
    tool_handler = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
    from core.execution.engines.litellm.executor import LiteLLMExecutor

    return LiteLLMExecutor(
        model_config=mc,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


async def _drive_litellm(anima_dir: Path, golden_id: str, path: str) -> dict:
    lines = _load_input(golden_id)
    head = lines[0] if lines else {}
    _, scenario = golden_id.split("__", 1)
    _, scenario = scenario.split("-", 1)
    model = "ollama/qwen3:8b" if path == "iteration" else "openai/gpt-4o"
    executor = _litellm_executor(anima_dir, model=model)
    tool_error_patch = (
        patch.object(executor, "_execute_tool_call", side_effect=RuntimeError("simulated tool failure"))
        if scenario == "tool-error"
        else nullcontext()
    )

    sys_mod = sys.modules.get("litellm")
    if sys_mod is None or not isinstance(sys_mod, MagicMock):
        sys_mod = MagicMock()
        sys.modules["litellm"] = sys_mod
    sys_mod.token_counter = MagicMock(return_value=500)

    if head.get("scenario") == "api_error":

        def _raise_call(**kwargs):
            raise RuntimeError(f"litellm.AuthenticationError: {head.get('message', 'invalid api key')}")

        with patch("litellm.acompletion", side_effect=_raise_call):
            try:
                if path == "stream":
                    events = [
                        e
                        async for e in executor.execute_streaming(
                            "sys", "test prompt", ContextTracker(model="openai/gpt-4o", threshold=0.5)
                        )
                    ]
                    return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}
                result = await executor.execute("test prompt", system_prompt="sys")
                return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}
            except Exception as exc:  # noqa: BLE001
                return {
                    "events": None,
                    "result": None,
                    "raised": {"type": type(exc).__name__, "message": str(exc)},
                    "activity": _read_activity(anima_dir),
                }

    if path in {"stream", "iteration"}:
        chunks = _build_stream_chunks(lines) if path == "stream" else None
        call_count = {"n": 0}

        async def mock_acompletion(**kwargs):
            i = call_count["n"]
            call_count["n"] += 1
            if path == "iteration":
                return responses[min(i, len(responses) - 1)]
            return _fake_async_stream(chunks[min(i, len(chunks) - 1)])

        responses = _litellm_responses_from_input(lines) if path == "iteration" else []
        with (
            patch("litellm.acompletion", side_effect=mock_acompletion),
            patch.object(executor, "_preflight_clamp_with_compaction", AsyncMock(return_value={})),
            tool_error_patch,
        ):
            events = [
                e
                async for e in executor.execute_streaming(
                    "sys", "test prompt", ContextTracker(model=model, threshold=0.5)
                )
            ]
        return {"events": _norm(events), "result": None, "activity": _read_activity(anima_dir)}

    responses = _litellm_responses_from_input(lines)
    mock_fn = AsyncMock(side_effect=responses)
    with patch("litellm.acompletion", mock_fn), tool_error_patch:
        result = await executor.execute("test prompt", system_prompt="sys")
    return {"events": None, "result": _serialise_result(result), "activity": _read_activity(anima_dir)}


class FakeStreamChunk:
    def __init__(self, text=None, tool_calls=None, finish_reason=None, usage=None):
        delta = MagicMock()
        delta.content = text
        delta.tool_calls = tool_calls
        delta.reasoning_content = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        self.choices = [choice]
        self.usage = usage


class FakeUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeDelta:
    def __init__(self, index, id=None, name=None, arguments=""):
        self.index = index
        self.id = id
        func = MagicMock()
        func.name = name
        func.arguments = arguments
        self.function = func


async def _fake_async_stream(chunks):
    for chunk in chunks:
        yield chunk


def _build_stream_chunks(lines: list[dict]) -> list[list]:
    """Translate scripted stream input into per-call fake chunk lists."""
    call_chunks: list[list] = []
    for line in lines:
        if line.get("kind") != "stream":
            continue
        chunks: list = []
        for c in line.get("chunks", []):
            if "text" in c:
                chunks.append(FakeStreamChunk(text=c["text"]))
            if "tool_call" in c:
                tc = c["tool_call"]
                chunks.append(
                    FakeStreamChunk(
                        tool_calls=[
                            FakeDelta(
                                index=0,
                                id=tc.get("id", "call_1"),
                                name=tc.get("name", "search_memory"),
                                arguments=tc.get("arguments", "{}"),
                            )
                        ]
                    )
                )
            if "usage" in c:
                chunks.append(
                    FakeStreamChunk(
                        finish_reason=c.get("finish", "stop"),
                        usage=FakeUsage(
                            prompt_tokens=c["usage"].get("prompt_tokens", 100),
                            completion_tokens=c["usage"].get("completion_tokens", 50),
                        ),
                    )
                )
            elif "finish" in c:
                chunks.append(FakeStreamChunk(finish_reason=c["finish"]))
        call_chunks.append(chunks)
    return call_chunks


# ── The contract ───────────────────────────────────────────

_ENGINE_PATHS = {
    "agent_sdk": {"execute": _drive_agent_sdk, "stream": _drive_agent_sdk},
    "codex": {"execute": _drive_codex, "stream": _drive_codex},
    "cursor": {"execute": _drive_cursor_execute, "stream": _drive_cursor_stream},
    "gemini": {"execute": _drive_gemini_execute, "stream": _drive_gemini_stream},
    "grok": {"execute": _drive_grok, "stream": _drive_grok},
    "litellm": {"execute": _drive_litellm, "stream": _drive_litellm, "iteration": _drive_litellm},
}

_SUPPORTS_STREAMING = {
    "agent_sdk": True,
    "codex": True,
    "cursor": True,
    "gemini": True,
    "grok": True,
    "litellm": True,
}


async def _capture(golden_id: str, tmp_path: Path) -> dict:
    engine, rest = golden_id.split("__", 1)
    path, scenario = rest.split("-", 1)
    anima_dir = _anima_dir(tmp_path, f"{engine}-{path}-{scenario}")
    driver = _ENGINE_PATHS[engine][path]
    out = await driver(anima_dir, golden_id, path)
    return {
        "engine": engine,
        "supports_streaming": _SUPPORTS_STREAMING[engine],
        "path": path,
        "scenario": scenario,
        "events": out.get("events"),
        "result": out.get("result"),
        "raised": out.get("raised"),
        "activity": out.get("activity", []),
    }


def _scenario_ids() -> list[str]:
    return [f"{e}__{p}-{s}" for e, p, s in _SCENARIOS]


@pytest.mark.parametrize("golden_id", _scenario_ids())
async def test_engine_contract(golden_id: str, tmp_path: Path) -> None:
    """The engine's actual output matches its committed golden file."""
    actual = await _capture(golden_id, tmp_path)
    _compare_or_write(golden_id, actual)
