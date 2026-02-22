# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for consolidation targeting all initialized animas.

Verifies that ``_iter_consolidation_targets()`` scans ``self.animas_dir``
on disk rather than relying on ``self.processes`` (live process dict),
so that stopped / crashed animas still receive memory consolidation.

Note: Tests for ``_run_daily_consolidation()`` and ``_run_weekly_integration()``
were removed because those methods call ``daily_consolidate``/``weekly_integrate``
which were removed from ConsolidationEngine in the consolidation refactor.

Issue: docs/issues/20260217_consolidation-run-for-all-animas.md
"""
from __future__ import annotations

import json
from pathlib import Path

from core.supervisor.manager import ProcessSupervisor


# ── Helpers ──────────────────────────────────────────────────────────


def _make_supervisor(tmp_path: Path) -> ProcessSupervisor:
    """Create a minimal ProcessSupervisor rooted under *tmp_path*."""
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


def _create_anima_dir(
    animas_dir: Path,
    name: str,
    *,
    has_identity: bool = True,
    has_status: bool = True,
    enabled: bool = True,
) -> Path:
    """Create a mock anima directory on disk with optional files."""
    d = animas_dir / name
    d.mkdir(parents=True, exist_ok=True)
    if has_identity:
        (d / "identity.md").write_text(f"# {name}", encoding="utf-8")
    if has_status:
        (d / "status.json").write_text(
            json.dumps({"enabled": enabled}), encoding="utf-8"
        )
    return d


# ── _iter_consolidation_targets ──────────────────────────────────────


class TestIterConsolidationTargets:
    """Tests for the helper that enumerates consolidation-eligible animas."""

    def test_returns_fully_initialized_anima(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "sakura")

        targets = sup._iter_consolidation_targets()

        assert len(targets) == 1
        assert targets[0][0] == "sakura"
        assert targets[0][1] == sup.animas_dir / "sakura"

    def test_skips_directory_without_identity(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "incomplete", has_identity=False)

        targets = sup._iter_consolidation_targets()

        assert targets == []

    def test_skips_directory_without_status_json(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "no_status", has_status=False)

        targets = sup._iter_consolidation_targets()

        assert targets == []

    def test_skips_disabled_anima(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "disabled_anima", enabled=False)

        targets = sup._iter_consolidation_targets()

        assert targets == []

    def test_skips_regular_files(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        (sup.animas_dir / "not_a_dir.txt").write_text("file", encoding="utf-8")

        targets = sup._iter_consolidation_targets()

        assert targets == []

    def test_returns_empty_when_animas_dir_missing(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        sup.animas_dir = tmp_path / "nonexistent"

        targets = sup._iter_consolidation_targets()

        assert targets == []

    def test_includes_multiple_animas_sorted(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "zeta")
        _create_anima_dir(sup.animas_dir, "alpha")
        _create_anima_dir(sup.animas_dir, "mu")

        targets = sup._iter_consolidation_targets()
        names = [t[0] for t in targets]

        assert names == ["alpha", "mu", "zeta"]

    def test_mixed_valid_and_invalid(self, tmp_path: Path) -> None:
        sup = _make_supervisor(tmp_path)
        _create_anima_dir(sup.animas_dir, "good")
        _create_anima_dir(sup.animas_dir, "no_id", has_identity=False)
        _create_anima_dir(sup.animas_dir, "no_status", has_status=False)
        _create_anima_dir(sup.animas_dir, "off", enabled=False)
        _create_anima_dir(sup.animas_dir, "also_good")

        targets = sup._iter_consolidation_targets()
        names = [t[0] for t in targets]

        assert names == ["also_good", "good"]


