from __future__ import annotations

import time

from core.memory.housekeeping import _rotate_archive_superseded


class TestRotateArchiveSuperseded:
    def test_no_animas_dir(self, tmp_path):
        result = _rotate_archive_superseded(tmp_path / "nonexistent", 7)
        assert result == {"skipped": True}

    def test_no_archive_dirs(self, tmp_path):
        animas = tmp_path / "animas"
        (animas / "rin" / "state").mkdir(parents=True)
        result = _rotate_archive_superseded(animas, 7)
        assert result["deleted_files"] == 0

    def test_deletes_old_files(self, tmp_path):
        animas = tmp_path / "animas"
        archive = animas / "rin" / "archive" / "superseded"
        archive.mkdir(parents=True)

        old_file = archive / "old.md"
        old_file.write_text("old content")
        eight_days_ago = time.time() - (8 * 86400)
        import os

        os.utime(old_file, (eight_days_ago, eight_days_ago))

        new_file = archive / "new.md"
        new_file.write_text("new content")

        result = _rotate_archive_superseded(animas, 7)
        assert result["deleted_files"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_keeps_recent_files(self, tmp_path):
        animas = tmp_path / "animas"
        archive = animas / "rin" / "archive" / "superseded"
        archive.mkdir(parents=True)

        recent = archive / "recent.md"
        recent.write_text("recent content")

        result = _rotate_archive_superseded(animas, 7)
        assert result["deleted_files"] == 0
        assert recent.exists()

    def test_multiple_animas(self, tmp_path):
        animas = tmp_path / "animas"
        for name in ("rin", "sakura", "natsume"):
            archive = animas / name / "archive" / "superseded"
            archive.mkdir(parents=True)
            old = archive / "old.md"
            old.write_text("old")
            import os

            eight_days_ago = time.time() - (8 * 86400)
            os.utime(old, (eight_days_ago, eight_days_ago))

        result = _rotate_archive_superseded(animas, 7)
        assert result["deleted_files"] == 3
