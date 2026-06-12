from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for RAG startup repair preflight gating."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _mark_setup_complete(data_dir: Path) -> None:
    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["setup_complete"] = True
    rag = config.setdefault("rag", {})
    rag["repair_enabled"] = True
    rag["startup_repair_preflight_enabled"] = True
    rag["startup_repair_window_minutes"] = 60
    rag["quick_check_timeout_seconds"] = 1.0
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    from core.config import invalidate_cache

    invalidate_cache()


def test_startup_preflight_ignores_unclean_exit_for_healthy_vectordb(
    data_dir: Path,
    make_anima,
) -> None:
    """A stale PID/orphan-runner startup must not rebuild every existing vectordb."""
    from cli.commands.server import _run_rag_startup_preflight
    from core.memory.rag.repair import _reset_for_testing

    _reset_for_testing()
    try:
        _mark_setup_complete(data_dir)
        anima_dir = make_anima("sora")
        vectordb_dir = anima_dir / "vectordb"
        vectordb_dir.mkdir()
        sentinel = vectordb_dir / "healthy-placeholder.txt"
        sentinel.write_text("existing non-corrupt vector store payload", encoding="utf-8")

        _run_rag_startup_preflight(force_all_vectordb=True)

        assert sentinel.is_file()
        assert not (anima_dir / "state" / "rag_repair.json").exists()
        assert not list((anima_dir / "archive").glob("vectordb-corrupt-*"))
    finally:
        _reset_for_testing()
