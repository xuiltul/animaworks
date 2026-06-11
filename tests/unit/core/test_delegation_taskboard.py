from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.messenger import Messenger
from core.taskboard.models import BoardColumn
from core.taskboard.store import TaskBoardStore
from core.tooling.handler_delegation import DelegationMixin


class _DelegationHarness(DelegationMixin):
    def __init__(self, anima_dir: Path, messenger: Messenger) -> None:
        self._anima_dir = anima_dir
        self._anima_name = anima_dir.name
        self._activity = MagicMock()
        self._messenger = messenger
        self._session_origin = "test"
        self._session_origin_chain = []


def _write_status(anima_dir: Path, *, enabled: bool = True, supervisor: str | None = None) -> None:
    payload = {"enabled": enabled}
    if supervisor:
        payload["supervisor"] = supervisor
    (anima_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_delegate_task_records_taskboard_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    animas_dir = tmp_path / "animas"
    boss_dir = animas_dir / "boss"
    worker_dir = animas_dir / "worker"
    (boss_dir / "state").mkdir(parents=True)
    (worker_dir / "state").mkdir(parents=True)
    _write_status(boss_dir)
    _write_status(worker_dir, supervisor="boss")
    shared_dir = tmp_path / "shared"
    messenger = Messenger(shared_dir, "boss")
    harness = _DelegationHarness(boss_dir, messenger)

    config = SimpleNamespace(
        animas={
            "boss": SimpleNamespace(supervisor=None, aliases=[]),
            "worker": SimpleNamespace(supervisor="boss", aliases=[]),
        },
        heartbeat=SimpleNamespace(depth_window_s=600, max_depth=6),
    )
    with (
        patch("core.config.models.load_config", return_value=config),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.paths.get_data_dir", return_value=tmp_path),
    ):
        result = harness._handle_delegate_task(
            {
                "name": "worker",
                "instruction": "Prepare the incident summary",
                "summary": "Incident summary",
                "deadline": "2h",
            }
        )

    assert "worker" in result
    assert "DM" in result
    store = TaskBoardStore(tmp_path / "shared" / "taskboard.sqlite3")
    metadata = store.list_metadata()
    by_anima = {row.anima_name: row for row in metadata}

    assert set(by_anima) == {"boss", "worker"}
    assert by_anima["worker"].column == BoardColumn.TODO
    assert by_anima["worker"].source_ref == f"task_queue:worker:{by_anima['worker'].task_id}"
    assert by_anima["boss"].column == BoardColumn.WAITING
    assert by_anima["boss"].source_ref == f"task_queue:boss:{by_anima['boss'].task_id}"

    worker_queue = (worker_dir / "state" / "task_queue.jsonl").read_text(encoding="utf-8")
    boss_queue = (boss_dir / "state" / "task_queue.jsonl").read_text(encoding="utf-8")
    assert by_anima["worker"].task_id in worker_queue
    assert by_anima["boss"].task_id in boss_queue


def test_delegate_task_aborts_when_taskboard_write_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    animas_dir = tmp_path / "animas"
    boss_dir = animas_dir / "boss"
    worker_dir = animas_dir / "worker"
    (boss_dir / "state").mkdir(parents=True)
    (worker_dir / "state").mkdir(parents=True)
    _write_status(boss_dir)
    _write_status(worker_dir, supervisor="boss")
    messenger = Messenger(tmp_path / "shared", "boss")
    harness = _DelegationHarness(boss_dir, messenger)

    config = SimpleNamespace(
        animas={
            "boss": SimpleNamespace(supervisor=None, aliases=[]),
            "worker": SimpleNamespace(supervisor="boss", aliases=[]),
        },
        heartbeat=SimpleNamespace(depth_window_s=600, max_depth=6),
    )
    with (
        patch("core.config.models.load_config", return_value=config),
        patch("core.paths.get_animas_dir", return_value=animas_dir),
        patch("core.paths.get_data_dir", return_value=tmp_path),
        patch("core.taskboard.store.TaskBoardStore.upsert_metadata", side_effect=RuntimeError("sqlite down")),
    ):
        result = harness._handle_delegate_task(
            {
                "name": "worker",
                "instruction": "Prepare the incident summary",
                "summary": "Incident summary",
                "deadline": "2h",
            }
        )

    assert "PersistenceFailed" in result
    assert not (worker_dir / "state" / "task_queue.jsonl").exists()
    assert not (boss_dir / "state" / "task_queue.jsonl").exists()
