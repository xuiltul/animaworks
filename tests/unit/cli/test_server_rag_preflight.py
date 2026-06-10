from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.memory.rag.repair import RepairResult


def test_rag_startup_preflight_repairs_suspects() -> None:
    from cli.commands.server import _run_rag_startup_preflight

    config = SimpleNamespace(
        setup_complete=True,
        rag=SimpleNamespace(
            repair_enabled=True,
            startup_repair_preflight_enabled=True,
            startup_repair_window_minutes=30,
            quick_check_timeout_seconds=4.0,
        ),
    )
    service = MagicMock()
    service.discover_suspect_animas.return_value = ["sora"]
    service.repair_animas_if_allowed.return_value = {
        "sora": RepairResult(status="success", anima_name="sora", reason="startup_chroma_crash_preflight")
    }

    with (
        patch("core.config.load_config", return_value=config),
        patch("core.memory.rag.repair.get_repair_service", return_value=service),
    ):
        _run_rag_startup_preflight()

    service.discover_suspect_animas.assert_called_once_with(
        window_minutes=30,
        quick_check_timeout_seconds=4.0,
        quick_check_source="startup_quick_check",
    )
    service.repair_animas_if_allowed.assert_called_once_with(
        ["sora"],
        reason="startup_chroma_crash_preflight",
        source="startup_preflight",
        include_shared=True,
    )


def test_rag_startup_preflight_repairs_vectordbs_after_unclean_exit(data_dir: Path) -> None:
    from cli.commands.server import _run_rag_startup_preflight

    anima_dir = data_dir / "animas" / "sora"
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "identity.md").write_text("# sora", encoding="utf-8")
    (anima_dir / "vectordb").mkdir()
    config = SimpleNamespace(
        setup_complete=True,
        rag=SimpleNamespace(
            repair_enabled=True,
            startup_repair_preflight_enabled=True,
            startup_repair_window_minutes=30,
            quick_check_timeout_seconds=4.0,
        ),
    )
    service = MagicMock()
    service.discover_suspect_animas.return_value = []
    service.list_repairable_animas.return_value = ["sora"]
    service.repair_animas_if_allowed.return_value = {
        "sora": RepairResult(status="success", anima_name="sora", reason="startup_unclean_exit_preflight")
    }

    with (
        patch("core.config.load_config", return_value=config),
        patch("core.memory.rag.repair.get_repair_service", return_value=service),
    ):
        _run_rag_startup_preflight(force_all_vectordb=True)

    service.repair_animas_if_allowed.assert_called_once_with(
        ["sora"],
        reason="startup_unclean_exit_preflight",
        source="startup_preflight",
        include_shared=True,
    )
