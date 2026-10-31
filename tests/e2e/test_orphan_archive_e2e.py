# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for orphan archive and cleanup functionality.

Tests the full pipeline of orphan detection, archiving, and cleanup
using real filesystem operations without mocking core functionality.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.config.models import (
    AnimaModelConfig,
    AnimaWorksConfig,
    invalidate_cache,
    load_config,
    save_config,
)
from core.org.org_sync import (
    cleanup_orphan_archives,
    detect_orphan_animas,
)

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_cache():
    invalidate_cache()
    yield
    invalidate_cache()


def _create_config(
    config_path: Path,
    *,
    animas: dict[str, AnimaModelConfig] | None = None,
) -> AnimaWorksConfig:
    cfg = AnimaWorksConfig(setup_complete=True, animas=animas or {})
    save_config(cfg, config_path)
    invalidate_cache()
    return cfg


def _make_anima(
    animas_dir: Path,
    name: str,
    identity_content: str = "# Identity\nValid identity.",
) -> Path:
    anima_dir = animas_dir / name
    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "IDENTITY.md").write_text(identity_content, encoding="utf-8")
    return anima_dir


def _make_orphan(
    animas_dir: Path,
    name: str,
    *,
    subdirs: list[str] | None = None,
    files: dict[str, str] | None = None,
) -> Path:
    """Create an orphan directory (no valid IDENTITY.md)."""
    orphan_dir = animas_dir / name
    orphan_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs or []:
        (orphan_dir / subdir).mkdir(parents=True, exist_ok=True)
    for fpath, content in (files or {}).items():
        fobj = orphan_dir / fpath
        fobj.parent.mkdir(parents=True, exist_ok=True)
        fobj.write_text(content, encoding="utf-8")
    return orphan_dir


# ── Test: Full archive pipeline ──────────────────────────────────


class TestOrphanArchiveFullPipeline:
    """End-to-end test of orphan detection → archive → cleanup."""

    def test_nontrivial_orphan_full_lifecycle(self, tmp_path: Path) -> None:
        """Full lifecycle: detect → archive → verify data → age-out → purge.

        1. Create a valid anima and a non-trivial orphan
        2. detect_orphan_animas archives the orphan
        3. Verify archive contains all original data
        4. Simulate aging past 30 days
        5. cleanup_orphan_archives purges the expired archive
        """
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={"sakura": AnimaModelConfig(supervisor=None)},
        )
        _make_anima(animas_dir, "sakura")

        _make_orphan(
            animas_dir,
            "bob",
            subdirs=["episodes", "knowledge", "state"],
            files={
                "episodes/2026-02-20.jsonl": '{"event":"test"}\n',
                "knowledge/notes.md": "# Notes\nImportant info.",
                "state/current_state.md": "status: idle\n",
            },
        )

        # Step 1: Detect and archive
        results = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)

        assert len(results) == 1
        assert results[0]["name"] == "bob"
        assert results[0]["action"] == "archived"
        assert not (animas_dir / "bob").exists()

        # Step 2: Verify archive content
        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        archive = archives[0]
        assert archive.name.startswith("bob_")

        assert (archive / "episodes" / "2026-02-20.jsonl").read_text(encoding="utf-8") == '{"event":"test"}\n'
        assert "Important info" in (archive / "knowledge" / "notes.md").read_text(encoding="utf-8")
        assert (archive / "state" / "current_state.md").exists()

        # Step 3: Archive is preserved when young
        removed = cleanup_orphan_archives(data_dir, max_age_days=30)
        assert removed == 0
        assert archive.exists()

        # Step 4: Simulate aging past 30 days
        old_time = time.time() - (31 * 86400)
        os.utime(archive, (old_time, old_time))

        # Step 5: Cleanup purges expired archive
        removed = cleanup_orphan_archives(data_dir, max_age_days=30)
        assert removed == 1
        assert not archive.exists()

    def test_mixed_trivial_and_nontrivial_orphans(self, tmp_path: Path) -> None:
        """Both trivial and non-trivial orphans are handled correctly."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={"sakura": AnimaModelConfig(supervisor=None)},
        )
        _make_anima(animas_dir, "sakura")

        # Trivial orphan: only vectordb
        trivial_dir = animas_dir / "trivial_test"
        trivial_dir.mkdir()
        (trivial_dir / "vectordb").mkdir()

        # Non-trivial orphan: has episodes
        _make_orphan(
            animas_dir,
            "nontrivial_test",
            subdirs=["episodes"],
            files={"episodes/log.jsonl": "data\n"},
        )

        results = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)

        actions = {r["name"]: r["action"] for r in results}
        assert actions["trivial_test"] == "auto_removed"
        assert actions["nontrivial_test"] == "archived"

        assert not (animas_dir / "trivial_test").exists()
        assert not (animas_dir / "nontrivial_test").exists()

        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        assert archives[0].name.startswith("nontrivial_test_")

    def test_config_entry_removed_after_archive(self, tmp_path: Path) -> None:
        """Config.json entry for archived orphan is removed."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={
                "sakura": AnimaModelConfig(supervisor=None),
                "orphan_bob": AnimaModelConfig(),
            },
        )
        _make_anima(animas_dir, "sakura")
        _make_orphan(
            animas_dir,
            "orphan_bob",
            subdirs=["state"],
        )

        detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)

        invalidate_cache()
        cfg = load_config(config_path)
        assert "orphan_bob" not in cfg.animas
        assert "sakura" in cfg.animas

    def test_valid_animas_never_touched(self, tmp_path: Path) -> None:
        """Valid animas with proper IDENTITY.md are never archived."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={
                "sakura": AnimaModelConfig(supervisor=None),
                "rin": AnimaModelConfig(supervisor="sakura"),
            },
        )
        _make_anima(animas_dir, "sakura", "# Sakura\nThe manager.")
        _make_anima(animas_dir, "rin", "# Rin\nReports to sakura.")

        results = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)
        assert results == []
        assert (animas_dir / "sakura").exists()
        assert (animas_dir / "rin").exists()

    def test_young_directory_skipped(self, tmp_path: Path) -> None:
        """Directories younger than age_threshold_s are not archived."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(config_path)
        _make_orphan(animas_dir, "just_created", subdirs=["state"])

        results = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=9999)
        assert results == []
        assert (animas_dir / "just_created").exists()

    def test_repeated_detection_idempotent(self, tmp_path: Path) -> None:
        """Running detection twice produces consistent results."""
        data_dir = tmp_path / "animaworks"
        data_dir.mkdir()
        animas_dir = data_dir / "animas"
        animas_dir.mkdir()
        shared_dir = data_dir / "shared"
        shared_dir.mkdir()
        config_path = data_dir / "config.json"

        _create_config(
            config_path,
            animas={"sakura": AnimaModelConfig(supervisor=None)},
        )
        _make_anima(animas_dir, "sakura")
        _make_orphan(animas_dir, "bob", subdirs=["episodes"])

        # First run
        results1 = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)
        assert len(results1) == 1
        assert results1[0]["action"] == "archived"

        # Second run — orphan already gone
        results2 = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)
        assert results2 == []
