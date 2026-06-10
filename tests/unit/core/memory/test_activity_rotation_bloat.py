from __future__ import annotations

import logging

from core.memory.activity import ActivityLogger


def test_activity_log_over_max_file_size_rotates_to_bloated_backup(tmp_path, caplog) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "2026-03-20.jsonl"
    log_file.write_bytes(b"x" * (2 * 1024 * 1024))

    logger = ActivityLogger(anima_dir)

    with caplog.at_level(logging.WARNING, logger="animaworks.activity"):
        result = logger.rotate(max_file_size_mb=1)

    assert result["bloated_rotated_files"] == 1
    assert result["deleted_files"] == 0
    assert not log_file.exists()
    assert (log_dir / "2026-03-20.jsonl.bloated.bak").exists()
    assert "exceeded 1 MB" in caplog.text


def test_activity_rotate_all_reports_bloated_rotation(tmp_path) -> None:
    anima_dir = tmp_path / "animas" / "sakura"
    log_dir = anima_dir / "activity_log"
    log_dir.mkdir(parents=True)
    (log_dir / "2026-03-20.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))

    result = ActivityLogger.rotate_all(tmp_path / "animas", max_file_size_mb=1)

    assert result["sakura"]["bloated_rotated_files"] == 1
