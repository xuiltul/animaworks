from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.memory.task_queue import TaskQueueManager
from core.taskboard.readiness import require_task_store_ready
from core.taskboard.tasks import TaskStore, task_database_path


def _anima(tmp_path: Path) -> Path:
    anima = tmp_path / "runtime" / "animas" / "fixture"
    (anima / "state").mkdir(parents=True)
    return anima


def test_readiness_on_fresh_runtime_is_read_only(tmp_path: Path):
    anima = _anima(tmp_path)
    require_task_store_ready(anima)
    assert not task_database_path(anima).exists()
    assert not TaskQueueManager(anima).load_active_tasks()


@pytest.mark.parametrize("filename", ["task_queue.jsonl", "task_queue_archive.jsonl", "pending/job.json"])
def test_arbitrary_read_cannot_import_populated_legacy_state(tmp_path: Path, filename: str):
    anima = _anima(tmp_path)
    legacy = anima / "state" / filename
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"task_id":"legacy"}\n')
    before = legacy.read_bytes()
    with pytest.raises(RuntimeError, match="task-store migrate"):
        TaskQueueManager(anima).load_active_tasks()
    assert legacy.read_bytes() == before
    assert not task_database_path(anima).exists()


def test_explicit_import_marker_allows_canonical_reads_without_replay(tmp_path: Path):
    anima = _anima(tmp_path)
    store = TaskStore(task_database_path(anima))
    store.import_legacy(anima)
    legacy = anima / "state/task_queue.jsonl"
    legacy.write_text("late obsolete write must not be revived\n")
    before = store.db_path.read_bytes()
    require_task_store_ready(anima)
    assert store.db_path.read_bytes() == before
    assert not TaskQueueManager(anima).load_active_tasks()


@pytest.mark.asyncio
async def test_server_preflight_blocks_worker_and_infrastructure_start(tmp_path: Path):
    from server.app import _startup_animas_background

    anima = _anima(tmp_path)
    (anima / "state/task_queue.jsonl").write_text('{"task_id":"old"}\n')
    supervisor = SimpleNamespace(start_all=AsyncMock())
    app = SimpleNamespace(
        state=SimpleNamespace(
            anima_names=[anima.name],
            animas_dir=anima.parent,
            supervisor=supervisor,
        )
    )
    with (
        patch("core.memory.frontmatter.FrontmatterService"),
        patch("core.infra.ensure_infra_services", new=AsyncMock()) as start_infra,
        pytest.raises(RuntimeError, match="task-store migrate"),
    ):
        await _startup_animas_background(app, suppress_errors=False)
    supervisor.start_all.assert_not_called()
    start_infra.assert_not_called()
    assert not task_database_path(anima).exists()
