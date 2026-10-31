from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.anima.factory import TEMPLATES_DIR, create_from_template
from core.memory.maintenance.consolidation import ConsolidationEngine, list_project_archives
from core.runtime.runner import AnimaRunner
from core.runtime.task_runner import execute_background_contract
from core.schemas import CycleResult


def test_project_engine_paths_and_default_compatibility(tmp_path: Path) -> None:
    project_engine = ConsolidationEngine(tmp_path, "librarian", project="foo")
    default_engine = ConsolidationEngine(tmp_path, "normal")

    assert project_engine.episodes_dir == tmp_path / "episodes" / "projects" / "foo"
    assert project_engine.knowledge_dir == tmp_path / "knowledge" / "projects" / "foo"
    assert default_engine.episodes_dir == tmp_path / "episodes"
    assert default_engine.knowledge_dir == tmp_path / "knowledge"
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("project", ["../outside", 123])
def test_project_engine_rejects_invalid_project(tmp_path: Path, project: object) -> None:
    with pytest.raises(ValueError, match="project"):
        ConsolidationEngine(tmp_path, "librarian", project=project)  # type: ignore[arg-type]


def test_list_project_archives_returns_only_sorted_directories(tmp_path: Path) -> None:
    assert list_project_archives(tmp_path) == []
    projects_dir = tmp_path / "episodes" / "projects"
    (projects_dir / "zeta").mkdir(parents=True)
    (projects_dir / "alpha").mkdir()
    (projects_dir / "ignored.txt").write_text("not a directory", encoding="utf-8")

    assert list_project_archives(tmp_path) == ["alpha", "zeta"]


@pytest.mark.parametrize("locale", ["ja", "en"])
def test_librarian_template_configuration(tmp_path: Path, locale: str) -> None:
    animas_dir = tmp_path / "animas"
    animas_dir.mkdir()
    with (
        patch("core.anima.factory.ANIMA_TEMPLATES_DIR", TEMPLATES_DIR / locale / "anima_templates"),
        patch("core.anima.factory.BOOTSTRAP_TEMPLATE", tmp_path / "no"),
    ):
        anima_dir = create_from_template(animas_dir, "librarian")

    status = json.loads((anima_dir / "status.json").read_text(encoding="utf-8"))
    assert status == {
        "enabled": True,
        "heartbeat_enabled": False,
        "consolidation_enabled": True,
        "role": "general",
        "speciality": "librarian",
    }
    assert list((anima_dir / "skills").iterdir()) == []
    assert "{name}" not in (anima_dir / "IDENTITY.md").read_text(encoding="utf-8")
    assert "schedule:" not in (anima_dir / "cron.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_propagates_project_to_anima() -> None:
    runner = AnimaRunner.__new__(AnimaRunner)
    runner.anima = SimpleNamespace(
        run_consolidation=AsyncMock(return_value=CycleResult(trigger="consolidation:daily", action="completed"))
    )
    supervisor = SimpleNamespace(run_background=AsyncMock(return_value={"summary": "consolidated", "duration_ms": 12}))
    runner._scheduler_mgr = SimpleNamespace(_task_runner_supervisor=supervisor)

    result = await runner._handle_run_consolidation({"consolidation_type": "daily", "project": "foo"})

    supervisor.run_background.assert_awaited_once_with(
        kind="consolidation",
        payload={"consolidation_type": "daily", "project": "foo"},
        display_lane="background",
    )
    runner.anima.run_consolidation.assert_not_awaited()
    assert result["status"] == "completed"
    assert result["summary"] == "consolidated"


@pytest.mark.asyncio
async def test_isolated_background_contract_propagates_project() -> None:
    anima = SimpleNamespace(
        run_consolidation=AsyncMock(return_value=CycleResult(trigger="consolidation:weekly", action="completed"))
    )

    await execute_background_contract(
        anima,
        kind="consolidation",
        payload={"consolidation_type": "weekly", "project": "foo"},
    )

    anima.run_consolidation.assert_awaited_once_with(
        consolidation_type="weekly",
        project="foo",
    )
