"""Unit tests for ProcessSupervisor log_dir configuration in server/app.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestAppLogDir:
    """Verify that create_app passes log_dir to ProcessSupervisor."""

    def test_supervisor_receives_log_dir(self, tmp_path: Path):
        """create_app should pass a log_dir to ProcessSupervisor."""
        from core.config.models import AnimaWorksConfig

        config = AnimaWorksConfig(setup_complete=False)
        animas_dir = tmp_path / "animas"
        animas_dir.mkdir()
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with (
            patch("server.app.load_config", return_value=config),
            patch("core.paths.get_data_dir", return_value=data_dir),
            patch("server.app.ProcessSupervisor") as mock_sup_cls,
        ):
            mock_sup_cls.return_value = MagicMock()

            from server.app import create_app
            app = create_app(animas_dir, shared_dir)

            # Verify log_dir was passed
            mock_sup_cls.assert_called_once()
            kwargs = mock_sup_cls.call_args
            assert "log_dir" in kwargs.kwargs, \
                "log_dir should be passed to ProcessSupervisor"

            log_dir = kwargs.kwargs["log_dir"]
            assert log_dir == data_dir / "logs"
