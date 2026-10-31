"""Unit tests for activity.log() safe parameter — double-fault prevention."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.activity.logger import ActivityLogger
from core.exceptions import MemoryWriteError


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    d = tmp_path / "animas" / "test-anima"
    (d / "activity_log").mkdir(parents=True)
    return d


@pytest.fixture
def activity_logger(anima_dir: Path) -> ActivityLogger:
    return ActivityLogger(anima_dir)


# ── safe=False (default): raises MemoryWriteError ─────────


class TestSafeFalseRaises:
    def test_oserror_raises_memory_write_error(self, activity_logger: ActivityLogger) -> None:
        with patch("core.activity.logger.os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(MemoryWriteError, match="disk full"):
                activity_logger.log("error", summary="test error")

    def test_type_error_raises_memory_write_error(self, activity_logger: ActivityLogger) -> None:
        with (
            patch(
                "core.activity.logger.json.dumps",
                side_effect=TypeError("not serializable"),
            ),
            pytest.raises(MemoryWriteError, match="not serializable"),
        ):
            activity_logger.log("error", summary="test error")


# ── safe=True: suppresses exceptions ─────────────────────


class TestSafeTrueSuppresses:
    def test_oserror_suppressed_with_safe(self, activity_logger: ActivityLogger) -> None:
        with patch("core.activity.logger.os.fsync", side_effect=OSError("disk full")):
            entry = activity_logger.log("error", summary="test error", safe=True)
            assert entry.type == "error"
            assert entry.summary == "test error"

    def test_type_error_suppressed_with_safe(self, activity_logger: ActivityLogger) -> None:
        with patch(
            "core.activity.logger.json.dumps",
            side_effect=TypeError("not serializable"),
        ):
            entry = activity_logger.log("error", summary="test error", safe=True)
            assert entry.type == "error"

    def test_value_error_suppressed_with_safe(self, activity_logger: ActivityLogger) -> None:
        with patch(
            "core.activity.logger.json.dumps",
            side_effect=ValueError("bad value"),
        ):
            entry = activity_logger.log("error", summary="test", safe=True)
            assert entry.type == "error"

    def test_safe_false_is_default(self, activity_logger: ActivityLogger) -> None:
        with patch("core.activity.logger.os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(MemoryWriteError):
                activity_logger.log("error", summary="test error")


# ── Normal operation unaffected ──────────────────────────


class TestNormalOperation:
    def test_log_writes_to_disk(self, activity_logger: ActivityLogger, anima_dir: Path) -> None:
        entry = activity_logger.log("heartbeat_end", summary="OK")
        assert entry.type == "heartbeat_end"
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1

    def test_log_safe_true_writes_normally(self, activity_logger: ActivityLogger, anima_dir: Path) -> None:
        entry = activity_logger.log("error", summary="normal write", safe=True)
        assert entry.type == "error"
        log_files = list((anima_dir / "activity_log").glob("*.jsonl"))
        assert len(log_files) == 1


# ── Double-fault prevention scenario ─────────────────────


class TestDoubleFaultPrevention:
    """Simulate the exact double-fault scenario from the issue:
    an error handler calls _activity.log() but the log itself fails.
    """

    def test_recovery_code_executes_after_safe_log_failure(
        self, activity_logger: ActivityLogger, tmp_path: Path
    ) -> None:
        recovery_marker = tmp_path / "recovery_executed"

        with patch("core.activity.logger.os.fsync", side_effect=OSError("disk full")):
            try:
                raise RuntimeError("original error")
            except RuntimeError:
                activity_logger.log("error", summary="logging the error", safe=True)
                recovery_marker.write_text("recovered")

        assert recovery_marker.exists()
        assert recovery_marker.read_text() == "recovered"

    def test_without_safe_recovery_code_skipped(self, activity_logger: ActivityLogger, tmp_path: Path) -> None:
        recovery_marker = tmp_path / "recovery_executed"

        with patch("core.activity.logger.os.fsync", side_effect=OSError("disk full")):
            try:
                raise RuntimeError("original error")
            except RuntimeError:
                with pytest.raises(MemoryWriteError):
                    activity_logger.log("error", summary="logging the error", safe=False)
                    recovery_marker.write_text("recovered")

        assert not recovery_marker.exists()
