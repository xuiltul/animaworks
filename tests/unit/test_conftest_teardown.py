"""Unit tests for conftest _kill_orphan_runners helper."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestKillOrphanRunners:
    """Verify _kill_orphan_runners identifies and kills matching processes."""

    def test_kills_matching_process(self, tmp_path: Path):
        """Should send SIGTERM to processes matching data_dir in cmdline."""
        from tests.conftest import _kill_orphan_runners

        data_dir = str(tmp_path / ".animaworks")
        fake_pid = "99999"
        fake_cmdline = f"python\x00-m\x00core.supervisor.runner\x00--animas-dir\x00{data_dir}/animas"

        with (
            patch("tests.conftest.subprocess.run", return_value=MagicMock(stdout=fake_pid)),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=fake_cmdline),
            patch("tests.conftest.os.kill") as mock_kill,
        ):
            _kill_orphan_runners(data_dir)

        mock_kill.assert_called_once_with(int(fake_pid), signal.SIGTERM)

    def test_does_not_kill_unrelated_processes(self, tmp_path: Path):
        """Should not touch processes that don't reference our data dir."""
        from tests.conftest import _kill_orphan_runners

        with patch("tests.conftest.os.kill") as mock_kill:
            _kill_orphan_runners(str(tmp_path / "unique-test-dir-12345"))
            mock_kill.assert_not_called()

    def test_does_not_kill_process_without_readable_cmdline(self, tmp_path: Path):
        from tests.conftest import _kill_orphan_runners

        with (
            patch("tests.conftest.subprocess.run", return_value=MagicMock(stdout="99999")),
            patch.object(Path, "read_text", side_effect=FileNotFoundError),
            patch("tests.conftest.os.kill") as kill,
        ):
            _kill_orphan_runners(str(tmp_path))
        kill.assert_not_called()

    def test_process_exiting_after_verification_is_harmless(self, tmp_path: Path):
        from tests.conftest import _kill_orphan_runners

        with (
            patch("tests.conftest.subprocess.run", return_value=MagicMock(stdout="99999")),
            patch.object(Path, "read_text", return_value=f"core.supervisor.runner {tmp_path}"),
            patch("tests.conftest.os.kill", side_effect=ProcessLookupError) as kill,
        ):
            _kill_orphan_runners(str(tmp_path))
        kill.assert_called_once_with(99999, signal.SIGTERM)
