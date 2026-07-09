from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_chroma_signal_to_supervised_repair_lifecycle(data_dir: Path) -> None:
    """Corruption detection records a request that supervisor repairs out-of-process."""
    from core.memory.rag.repair_service import RAGRepairService, _reset_for_testing
    from core.supervisor.manager import ProcessSupervisor

    _reset_for_testing()
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "vectordb").mkdir()

    service = RAGRepairService(enabled=True, threshold=1, window_minutes=5, cooldown_minutes=60)
    assert service.record_chroma_error(
        anima_name="sora",
        collection="sora_knowledge",
        error="database disk image is malformed",
        source="query",
    )

    state_path = anima_dir / "state" / "rag_repair.json"
    requested = json.loads(state_path.read_text(encoding="utf-8"))
    assert requested["status"] == "requested"
    assert requested["stage"] == "detect"

    sup = ProcessSupervisor(
        animas_dir=data_dir / "animas",
        shared_dir=data_dir / "shared",
        run_dir=data_dir / "run",
    )
    calls: list[str] = []
    sup.processes["sora"] = object()

    async def stop_anima(name: str, **_: object) -> None:
        calls.append(f"stop:{name}")
        sup.processes.pop(name, None)

    async def start_anima(name: str) -> None:
        calls.append(f"start:{name}")

    async def repair_cli(name: str, *, reason: str, include_shared: bool) -> dict[str, object]:
        calls.append(f"repair:{name}:{reason}:{include_shared}")
        return {"ok": True, "status": "success"}

    sup.stop_anima = stop_anima
    sup.start_anima = start_anima
    sup._run_rag_repair_cli_process = repair_cli

    await sup._run_supervised_rag_repair("sora", requested)

    assert calls == [
        "stop:sora",
        "repair:sora:sqlite_malformed:True",
        "start:sora",
    ]
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["status"] == "healthy"
    assert final_state["stage"] == "complete"
    assert final_state["last_error"] is None
