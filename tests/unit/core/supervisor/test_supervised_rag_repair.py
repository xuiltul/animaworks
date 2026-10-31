from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _make_supervisor(tmp_path: Path):
    from server.supervisor.manager import ProcessSupervisor

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
    (anima_dir / "status.json").write_text('{"enabled": true}', encoding="utf-8")
    return anima_dir


def _create_enabled_anima(sup, name: str = "sora") -> Path:
    anima_dir = _create_anima(sup, name)
    (anima_dir / "IDENTITY.md").write_text(f"# {name}\n", encoding="utf-8")
    (anima_dir / "status.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    return anima_dir


def _read_state(anima_dir: Path) -> dict:
    return json.loads((anima_dir / "state" / "rag_repair.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_supervised_rag_repair_repairs_without_stopping_by_default(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    calls: list[tuple[str, str]] = []
    stages: list[str] = []
    sup.processes["sora"] = object()
    write_state = sup._write_rag_repair_state

    def track_state(name: str, updates: dict[str, object]) -> None:
        if stage := updates.get("stage"):
            stages.append(str(stage))
        write_state(name, updates)

    async def repair_request(
        name: str,
        command: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        assert command == "repair_memory"
        assert timeout == 1830.0
        calls.append(("repair", f"{name}:{payload['reason']}:{payload['include_shared']}"))
        return {"ok": True, "status": "success"}

    sup._write_rag_repair_state = track_state
    sup.stop_anima = AsyncMock()
    sup.start_anima = AsyncMock()
    sup.send_request = repair_request

    await sup._run_supervised_rag_repair(
        "sora",
        {"status": "requested", "reason": "sqlite_malformed", "include_shared": True},
    )

    assert calls == [("repair", "sora:sqlite_malformed:True")]
    sup.stop_anima.assert_not_awaited()
    sup.start_anima.assert_not_awaited()
    assert "sora" in sup.processes
    assert stages == ["fence_access", "repair", "unfence"]
    state = _read_state(anima_dir)
    assert state["status"] == "healthy"
    assert state["stage"] == "unfence"
    assert state["pid"] is None


@pytest.mark.asyncio
async def test_requested_repair_waits_until_anima_process_is_running(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_enabled_anima(sup)
    from core.memory.rag.repair import state as repair_state

    repair_state.write_repair_request_state(
        "sora",
        reason="cli_repair",
        collection=None,
        source="cli",
        include_shared=True,
        animas_dir=sup.animas_dir,
    )
    sup._last_rag_repair_poll_at = 0.0
    sup._read_rag_repair_state = lambda _name: repair_state.read_state("sora", animas_dir=sup.animas_dir)

    await sup._poll_requested_rag_repairs()

    assert repair_state.read_state("sora", animas_dir=sup.animas_dir)["status"] == "requested"
    assert sup._rag_repairs_in_progress == set()
    assert anima_dir.is_dir()


@pytest.mark.asyncio
async def test_supervised_rag_repair_failure_unfences_without_restart(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    started: list[str] = []

    sup.send_request = AsyncMock(return_value={"ok": False, "status": "timeout", "error": "timed out"})
    sup.start_anima = AsyncMock(side_effect=lambda name: started.append(name))

    await sup._run_supervised_rag_repair("sora", {"status": "requested", "reason": "sqlite_malformed"})

    assert started == []
    state = _read_state(anima_dir)
    assert state["status"] == "failed"
    assert state["stage"] == "unfence"
    assert state["last_error"] == "timed out"


@pytest.mark.asyncio
async def test_poll_requested_rag_repairs_starts_one_supervised_task(tmp_path: Path) -> None:
    sup = _make_supervisor(tmp_path)
    anima_dir = _create_anima(sup)
    (anima_dir / "state" / "rag_repair.json").write_text(
        json.dumps({"status": "requested", "reason": "sqlite_malformed"}),
        encoding="utf-8",
    )
    sup._rag_repair_poll_interval_seconds = lambda: 0.0
    sup.processes["sora"] = SimpleNamespace(is_alive=lambda: True)
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
async def test_poll_requested_rag_repairs_caps_concurrency(tmp_path: Path) -> None:
    """Only ``repair_max_concurrent`` CPU/IO-heavy rebuilds may start per poll."""
    sup = _make_supervisor(tmp_path)
    for name in ("aoi", "rin", "sora"):
        anima_dir = _create_anima(sup, name)
        (anima_dir / "state" / "rag_repair.json").write_text(
            json.dumps({"status": "requested", "reason": "chroma_corruption"}),
            encoding="utf-8",
        )
    sup._rag_repair_poll_interval_seconds = lambda: 0.0
    sup._rag_repair_max_concurrent = lambda: 1
    for name in ("aoi", "rin", "sora"):
        sup.processes[name] = SimpleNamespace(is_alive=lambda: True)
    started: list[str] = []

    async def run_repair(name: str, state: dict[str, object]) -> None:
        # Simulate an in-flight repair that does not release its slot yet.
        started.append(name)
        await asyncio.sleep(0.05)

    sup._run_supervised_rag_repair = run_repair

    await sup._poll_requested_rag_repairs()
    await asyncio.sleep(0)  # let the scheduled task start

    assert len(started) == 1
    assert sup._rag_repairs_in_progress == {started[0]}


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
