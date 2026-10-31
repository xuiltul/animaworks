# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for orphan archive and cleanup functionality.

Covers:
- _archive_and_remove_orphan: archive non-trivial orphans then delete
- cleanup_orphan_archives: purge expired archives
- detect_orphan_animas integration with archive flow
- Backward compatibility with existing trivial orphan auto-removal
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch


class TestArchiveAndRemoveOrphan:
    """Tests for _archive_and_remove_orphan."""

    def test_archives_and_removes_nontrivial_orphan(self, data_dir, make_anima):
        """Non-trivial orphan is copied to archive/ then removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "bob"
        orphan_dir.mkdir()
        (orphan_dir / "episodes").mkdir()
        (orphan_dir / "episodes" / "2026-02-25.jsonl").write_text('{"ts":"2026-02-25T10:00:00"}\n', encoding="utf-8")
        (orphan_dir / "knowledge").mkdir()
        (orphan_dir / "knowledge" / "learned.md").write_text("# Learned\nSomething useful.", encoding="utf-8")

        from core.org.org_sync import _archive_and_remove_orphan

        result = _archive_and_remove_orphan(orphan_dir)

        assert result is True
        assert not orphan_dir.exists()

        archive_root = data_dir / "archive" / "orphans"
        assert archive_root.is_dir()
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        archive = archives[0]
        assert archive.name.startswith("bob_")
        assert (archive / "episodes" / "2026-02-25.jsonl").exists()
        assert (archive / "knowledge" / "learned.md").exists()

    def test_archive_preserves_all_files(self, data_dir, make_anima):
        """All files in the orphan directory are preserved in the archive."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "carol"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()
        (orphan_dir / "state" / "current_state.md").write_text("busy", encoding="utf-8")
        (orphan_dir / "procedures").mkdir()
        (orphan_dir / "procedures" / "howto.md").write_text("steps", encoding="utf-8")
        (orphan_dir / "status.json").write_text('{"enabled":true}', encoding="utf-8")

        from core.org.org_sync import _archive_and_remove_orphan

        _archive_and_remove_orphan(orphan_dir)

        archive_root = data_dir / "archive" / "orphans"
        archive = next(archive_root.iterdir())
        assert (archive / "state" / "current_state.md").read_text(encoding="utf-8") == "busy"
        assert (archive / "procedures" / "howto.md").read_text(encoding="utf-8") == "steps"
        assert (archive / "status.json").read_text(encoding="utf-8") == '{"enabled":true}'

    def test_archive_unregisters_config_entry(self, data_dir, make_anima):
        """Archived orphan is also unregistered from config.json."""
        make_anima("sakura")

        from core.config.models import (
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )

        config = load_config(data_dir / "config.json")
        config.animas["bob"] = AnimaModelConfig()
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        orphan_dir = data_dir / "animas" / "bob"
        orphan_dir.mkdir()
        (orphan_dir / "episodes").mkdir()

        from core.org.org_sync import _archive_and_remove_orphan

        _archive_and_remove_orphan(orphan_dir)

        invalidate_cache()
        updated = load_config(data_dir / "config.json")
        assert "bob" not in updated.animas

    def test_archive_failure_returns_false(self, data_dir, make_anima):
        """If shutil.copytree fails, returns False and directory is untouched."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "bob"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()

        from core.org.org_sync import _archive_and_remove_orphan

        with patch("core.org.org_sync.shutil.copytree", side_effect=OSError("disk full")):
            result = _archive_and_remove_orphan(orphan_dir)

        assert result is False
        assert orphan_dir.exists()

    def test_archive_creates_parent_dirs(self, data_dir, make_anima):
        """Archive directory parents are created automatically."""
        make_anima("sakura")

        archive_root = data_dir / "archive" / "orphans"
        assert not archive_root.exists()

        orphan_dir = data_dir / "animas" / "test_new"
        orphan_dir.mkdir()
        (orphan_dir / "knowledge").mkdir()

        from core.org.org_sync import _archive_and_remove_orphan

        _archive_and_remove_orphan(orphan_dir)
        assert archive_root.is_dir()


class TestCleanupOrphanArchives:
    """Tests for cleanup_orphan_archives."""

    def test_removes_expired_archives(self, data_dir):
        """Archives older than max_age_days are removed."""
        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)

        old_archive = archive_root / "bob_20260101_000000"
        old_archive.mkdir()
        (old_archive / "episodes").mkdir()

        old_time = time.time() - (31 * 86400)
        os.utime(old_archive, (old_time, old_time))

        from core.org.org_sync import cleanup_orphan_archives

        removed = cleanup_orphan_archives(data_dir, max_age_days=30)
        assert removed == 1
        assert not old_archive.exists()

    def test_preserves_recent_archives(self, data_dir):
        """Archives younger than max_age_days are preserved."""
        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)

        recent_archive = archive_root / "carol_20260224_120000"
        recent_archive.mkdir()
        (recent_archive / "state").mkdir()

        from core.org.org_sync import cleanup_orphan_archives

        removed = cleanup_orphan_archives(data_dir, max_age_days=30)
        assert removed == 0
        assert recent_archive.exists()

    def test_mixed_old_and_new_archives(self, data_dir):
        """Only expired archives are removed; recent ones survive."""
        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)

        old = archive_root / "old_20260101_000000"
        old.mkdir()
        old_time = time.time() - (40 * 86400)
        os.utime(old, (old_time, old_time))

        new = archive_root / "new_20260224_120000"
        new.mkdir()

        from core.org.org_sync import cleanup_orphan_archives

        removed = cleanup_orphan_archives(data_dir, max_age_days=30)
        assert removed == 1
        assert not old.exists()
        assert new.exists()

    def test_noop_when_no_archive_dir(self, data_dir):
        """Returns 0 when archive directory does not exist."""
        from core.org.org_sync import cleanup_orphan_archives

        removed = cleanup_orphan_archives(data_dir)
        assert removed == 0

    def test_noop_when_empty_archive_dir(self, data_dir):
        """Returns 0 when archive directory is empty."""
        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)

        from core.org.org_sync import cleanup_orphan_archives

        removed = cleanup_orphan_archives(data_dir)
        assert removed == 0

    def test_custom_max_age(self, data_dir):
        """Custom max_age_days parameter is respected."""
        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)

        archive = archive_root / "test_20260220_000000"
        archive.mkdir()
        age_time = time.time() - (3 * 86400)
        os.utime(archive, (age_time, age_time))

        from core.org.org_sync import cleanup_orphan_archives

        removed_keep = cleanup_orphan_archives(data_dir, max_age_days=7)
        assert removed_keep == 0

        removed_purge = cleanup_orphan_archives(data_dir, max_age_days=2)
        assert removed_purge == 1


class TestDetectOrphanAnimasArchiveIntegration:
    """Tests for detect_orphan_animas with archive behavior."""

    def test_nontrivial_orphan_archived(self, data_dir, make_anima):
        """Non-trivial orphans are archived and removed via detect_orphan_animas."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()
        (orphan_dir / "episodes").mkdir()
        (orphan_dir / "episodes" / "log.jsonl").write_text("data\n", encoding="utf-8")

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
        assert orphans[0]["action"] == "archived"
        assert not orphan_dir.exists()

        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        assert (archives[0] / "episodes" / "log.jsonl").exists()

    def test_nontrivial_orphan_no_marker_file(self, data_dir, make_anima):
        """Archived orphans do not leave .orphan_notified markers."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "bob"
        orphan_dir.mkdir()
        (orphan_dir / "knowledge").mkdir()

        from core.org.org_sync import detect_orphan_animas

        detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        assert not orphan_dir.exists()

    def test_trivial_orphan_still_auto_removed(self, data_dir, make_anima):
        """Trivial orphans are still auto-removed (no archiving)."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "trivial"
        orphan_dir.mkdir()
        (orphan_dir / "vectordb").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

        archive_root = data_dir / "archive" / "orphans"
        if archive_root.exists():
            assert len(list(archive_root.iterdir())) == 0

    def test_age_threshold_respected(self, data_dir, make_anima):
        """Orphans younger than age_threshold_s are not archived."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "young"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
        )
        assert orphans == []
        assert orphan_dir.exists()

    def test_cleanup_archives_runs_during_detection(self, data_dir, make_anima):
        """detect_orphan_animas triggers cleanup_orphan_archives."""
        make_anima("sakura")

        archive_root = data_dir / "archive" / "orphans"
        archive_root.mkdir(parents=True)
        old = archive_root / "old_20260101_000000"
        old.mkdir()
        old_time = time.time() - (31 * 86400)
        os.utime(old, (old_time, old_time))

        from core.org.org_sync import detect_orphan_animas

        detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        assert not old.exists()

    def test_multiple_nontrivial_orphans(self, data_dir, make_anima):
        """Multiple non-trivial orphans are all archived."""
        make_anima("sakura")

        for name in ("bob", "carol", "test"):
            orphan_dir = data_dir / "animas" / name
            orphan_dir.mkdir()
            (orphan_dir / "episodes").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        archived = [o for o in orphans if o["action"] == "archived"]
        assert len(archived) == 3
        assert {o["name"] for o in archived} == {"bob", "carol", "test"}

        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 3

    def test_existing_valid_animas_untouched(self, data_dir, make_anima):
        """Valid animas with IDENTITY.md are never archived."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas",
            data_dir / "shared",
            age_threshold_s=0,
        )
        assert orphans == []
        assert (data_dir / "animas" / "sakura").exists()
        assert (data_dir / "animas" / "rin").exists()
