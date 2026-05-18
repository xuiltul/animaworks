from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _make_supervisor(tmp_path: Path):
    from core.supervisor.manager import ProcessSupervisor

    animas_dir = tmp_path / "animas"
    animas_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return ProcessSupervisor(
        animas_dir=animas_dir,
        shared_dir=shared_dir,
        run_dir=run_dir,
    )


def _create_anima(sup, name: str = "sora") -> Path:
    anima_dir = sup.animas_dir / name
    (anima_dir / "state").mkdir(parents=True, exist_ok=True)
    (anima_dir / "vectordb").mkdir(exist_ok=True)
    return anima_dir


def _create_enabled_anima(sup, name: str = "sora") -> Path:
    anima_dir = _create_anima(sup, name)
    (anima_dir / "identity.md").write_text(f"# {name}\n", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return anima_dir


def _read_state(anima_dir: Path) -> dict:
    return json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_supervised_rag_repair_stops_repairs_and_restarts(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    calls: list[tuple[str, str]] = []
    sup.processes["sora"] = object()

    async def stop_anima(name: str) -> None:
        calls.append(("stop", name))
        sup.processes.pop(name, None)

    async def start_anima(name: str) -> None:
        calls.append(("start", name))

    async def repair_cli(name: str, *, reason: str, include_shared: bool) -> dict[str, object]:
        calls.append(("repair", f"{name}:{reason}:{include_shared}"))
        return {"ok": True, "status": "success"}

    sup.stop_anima = stop_anima
    sup.start_anima = start_anima
    sup._run_rag_repair_cli_process = repair_cli

    await sup._run_supervised_rag_repair(
        "sora",
        {"status": "requested", "reason": "sqlite_malformed", "include_shared": True},
    )

    assert calls == [
        ("stop", "sora"),
        ("repair", "sora:sqlite_malformed:True"),
        ("start", "sora"),
    ]
    state = _read_state(anima_dir)
    assert state["status"] == "healthy"
    assert state["stage"] == "complete"
    assert state["pid"] is None


@pytest.mark.asyncio
async def test_supervised_rag_repair_stop_failure_does_not_run_repair(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    sup.processes["sora"] = object()
    repair_called = False

    async def stop_anima(name: str) -> None:
        raise RuntimeError("stop failed")

    async def repair_cli(name: str, *, reason: str, include_shared: bool) -> dict[str, object]:
        nonlocal repair_called
        repair_called = True
        return {"ok": True, "status": "success"}

    sup.stop_anima = stop_anima
    sup._run_rag_repair_cli_process = repair_cli

    await sup._run_supervised_rag_repair("sora", {"status": "requested", "reason": "sqlite_malformed"})

    assert repair_called is False
    state = _read_state(anima_dir)
    assert state["status"] == "failed"
    assert state["stage"] == "stop_anima"
    assert state["last_error"] == "stop failed"


@pytest.mark.asyncio
async def test_supervised_rag_repair_failure_restarts_when_active_db_exists(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    started: list[str] = []

    async def repair_cli(name: str, *, reason: str, include_shared: bool) -> dict[str, object]:
        return {"ok": False, "status": "timeout", "error": "timed out"}

    async def start_anima(name: str) -> None:
        started.append(name)

    sup._run_rag_repair_cli_process = repair_cli
    sup.start_anima = start_anima

    await sup._run_supervised_rag_repair("sora", {"status": "requested", "reason": "sqlite_malformed"})

    assert started == ["sora"]
    state = _read_state(anima_dir)
    assert state["status"] == "failed"
    assert state["stage"] == "failed"
    assert state["last_error"] == "timed out"


@pytest.mark.asyncio
async def test_supervised_rag_repair_restart_failure_records_state(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)

    async def repair_cli(name: str, *, reason: str, include_shared: bool) -> dict[str, object]:
        return {"ok": True, "status": "success"}

    async def start_anima(name: str) -> None:
        raise RuntimeError("restart failed")

    sup._run_rag_repair_cli_process = repair_cli
    sup.start_anima = start_anima

    await sup._run_supervised_rag_repair("sora", {"status": "requested", "reason": "sqlite_malformed"})

    state = _read_state(anima_dir)
    assert state["status"] == "repair_success_restart_failed"
    assert state["stage"] == "restart_anima"
    assert state["last_error"] == "restart failed"


@pytest.mark.asyncio
async def test_poll_requested_rag_repairs_starts_one_supervised_task(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps({"status": "requested", "reason": "sqlite_malformed"}),
        encoding="utf-8",
    )
    sup._rag_repair_poll_interval_seconds = lambda: 0.0
    started = asyncio.Event()
    calls: list[tuple[str, str]] = []

    async def run_repair(name: str, state: dict[str, object]) -> None:
        calls.append((name, str(state["reason"])))
        started.set()

    sup._run_supervised_rag_repair = run_repair

    await sup._poll_requested_rag_repairs()
    await asyncio.wait_for(started.wait(), timeout=1)

    assert calls == [("sora", "sqlite_malformed")]


@pytest.mark.asyncio
async def test_reconcile_does_not_start_anima_during_rag_repair(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    _create_enabled_anima(sup)
    sup._rag_repairs_in_progress.add("sora")
    sup.start_anima = AsyncMock()

    with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
        await sup._reconcile()

    sup.start_anima.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_defers_restart_requested_during_rag_repair(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_enabled_anima(sup)
    (anima_dir / "status.json").write_text(
        json.dumps({"enabled": True, "restart_requested": True}),
        encoding="utf-8",
    )
    sup._rag_repairs_in_progress.add("sora")
    sup.restart_anima = AsyncMock()

    with patch.object(sup, "_reconcile_assets", new_callable=AsyncMock):
        await sup._reconcile()

    sup.restart_anima.assert_not_called()
    status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
    assert status["restart_requested"] is True


@pytest.mark.asyncio
async def test_repair_cli_process_passes_reason_to_subprocess(tmp_path: Path, monkeypatch) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 12345
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_create_subprocess_exec(*cmd, cwd, stdout, stderr):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return FakeProc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    sup._rag_repair_timeout_seconds = lambda: 10

    result = await sup._run_rag_repair_cli_process(
        "sora",
        reason="sqlite_malformed",
        include_shared=True,
    )

    assert result["ok"] is True
    cmd = captured["cmd"]
    assert isinstance(cmd, tuple)
    assert "--reason" in cmd
    assert cmd[cmd.index("--reason") + 1] == "sqlite_malformed"
    assert "--shared" in cmd
    state = _read_state(anima_dir)
    assert state["pid"] == 12345


@pytest.mark.asyncio
async def test_repair_cli_process_timeout_kills_subprocess(tmp_path: Path, monkeypatch) -> None:
    sup = _make_supervisor(tmp_path)
    _create_anima(sup)

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        async def communicate(self):
            if self.killed:
                return b"", b""
            await asyncio.sleep(10)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    proc = FakeProc()

    async def fake_create_subprocess_exec(*cmd, cwd, stdout, stderr):
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    sup._rag_repair_timeout_seconds = lambda: 0.01

    result = await sup._run_rag_repair_cli_process(
        "sora",
        reason="sqlite_malformed",
        include_shared=False,
    )

    assert result["ok"] is False
    assert result["status"] == "timeout"
    assert proc.killed is True
