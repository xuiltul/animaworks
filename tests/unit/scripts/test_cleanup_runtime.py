from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scripts.cleanup_runtime import collect_cleanup_targets, execute_cleanup


def _old(path: Path, days: int = 120) -> None:
    old_ts = time.time() - (days * 86400)
    os.utime(path, (old_ts, old_ts))


def test_collect_cleanup_targets_lists_residue_and_estimated_reclaim(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    bloated = data_dir / "animas" / "sanae" / "activity_log" / "2026-03-20.jsonl.bloated.bak"
    bloated.parent.mkdir(parents=True)
    bloated.write_bytes(b"x" * 10)

    corrupt = data_dir / "animas" / "rin" / "archive" / "vectordb-corrupt-20260101-010101"
    corrupt.mkdir(parents=True)
    (corrupt / "chroma.sqlite3").write_bytes(b"y" * 20)
    # Directory name timestamp, not mtime, determines age. This simulates a
    # copied old archive with a fresh mtime.

    config_bak = data_dir / "config.json.bak.1"
    config_bak.parent.mkdir(parents=True, exist_ok=True)
    config_bak.write_bytes(b"z" * 5)
    recent_named_corrupt = data_dir / "animas" / "rin" / "archive" / "vectordb-corrupt-20990101-010101"
    recent_named_corrupt.mkdir()
    (recent_named_corrupt / "chroma.sqlite3").write_bytes(b"recent")
    _old(recent_named_corrupt)
    memory = data_dir / "animas" / "sanae" / "knowledge" / "keep.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("do not delete", encoding="utf-8")

    targets = collect_cleanup_targets(data_dir)

    assert {target.reason for target in targets} == {
        "bloated activity log backup",
        "corrupt vectordb archive older than 90d",
        "root config backup",
    }
    assert sum(target.size_bytes for target in targets) == 35

    result = execute_cleanup(data_dir, execute=False)
    assert result.dry_run is True
    assert result.target_count == 3
    assert result.estimated_reclaim_bytes == 35
    assert bloated.exists()
    assert corrupt.exists()
    assert config_bak.exists()
    assert recent_named_corrupt.exists()
    assert memory.exists()


def test_execute_archives_and_deletes_targets_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    bloated = data_dir / "animas" / "sanae" / "activity_log" / "2026-03-20.jsonl.bloated.bak"
    bloated.parent.mkdir(parents=True)
    bloated.write_text("bloated", encoding="utf-8")

    root_dump = data_dir / "sakura.jsonl"
    root_dump.parent.mkdir(parents=True, exist_ok=True)
    root_dump.write_text("debug dump", encoding="utf-8")

    accident = data_dir / "Directories created"
    accident.write_text("oops", encoding="utf-8")

    memory = data_dir / "animas" / "sanae" / "episodes" / "2026-06-10.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("real memory", encoding="utf-8")

    result = execute_cleanup(data_dir, execute=True, archive_name="cleanup-test")

    assert result.dry_run is False
    assert result.deleted_count == 3
    assert result.archive_path is not None
    assert result.archive_path.exists()
    assert result.archive_path.suffixes[-2:] == [".tar", ".zst"]
    assert not bloated.exists()
    assert not root_dump.exists()
    assert not accident.exists()
    assert memory.exists()


def test_execute_does_not_delete_when_archive_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    bloated = data_dir / "animas" / "sanae" / "activity_log" / "2026-03-20.jsonl.bloated.bak"
    bloated.parent.mkdir(parents=True)
    bloated.write_text("bloated", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        if name == "tar":
            return "/bin/false"
        return None

    monkeypatch.setattr("scripts.cleanup_runtime.shutil.which", fake_which)

    result = execute_cleanup(data_dir, execute=True, archive_name="cleanup-test")

    assert result.deleted_count == 0
    assert result.errors
    assert "archive failed" in result.errors[0]
    assert bloated.exists()
