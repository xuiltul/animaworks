from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.memory.rag.sqlite_health import SQLiteHealthResult
from core.supervisor._mgr_scheduler import SchedulerMixin


class _DailyIndexingHarness(SchedulerMixin):
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _get_data_dir(self) -> Path:
        return self._data_dir

    def _iter_consolidation_targets(self):
        yield "sora", self._data_dir / "animas" / "sora"

    async def _broadcast_event(self, *_args, **_kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_daily_indexing_skips_anima_when_quick_check_finds_corruption(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "knowledge").mkdir(parents=True)
    (data_dir / "shared" / "common_knowledge").mkdir(parents=True)
    get_vector_store = MagicMock()

    monkeypatch.setattr(
        "core.config.load_config", lambda: SimpleNamespace(rag=SimpleNamespace(quick_check_timeout_seconds=2.0))
    )
    monkeypatch.setattr("core.memory.rag.singleton.get_embedding_model_name", lambda: "model")
    monkeypatch.setattr("core.memory.rag.singleton.get_vector_store", get_vector_store)
    monkeypatch.setattr("core.memory.rag.repair.is_repair_locked", lambda _anima_name: False)
    monkeypatch.setattr(
        "core.memory.rag.sqlite_health.check_anima_vectordb_health_via_worker_or_direct",
        lambda anima_name, **_kwargs: SQLiteHealthResult(
            db_path=data_dir / "animas" / anima_name / "vectordb" / "chroma.sqlite3",
            ok=False,
            status="corrupt",
            error="database disk image is malformed",
        ),
    )

    await _DailyIndexingHarness(data_dir)._run_daily_indexing()

    get_vector_store.assert_not_called()
