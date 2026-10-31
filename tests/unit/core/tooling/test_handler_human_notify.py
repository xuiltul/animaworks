"""Unit tests for call_human activity log content limit (200→1000)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.exceptions import ConfigError


class TestCallHumanActivityLogContentLimit:
    def _make_handler(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test_anima"
        anima_dir.mkdir(parents=True)
        (anima_dir / "activity_log").mkdir()

        memory = MagicMock()
        memory.anima_dir = anima_dir

        with (
            patch("core.tooling.handler.ExternalToolDispatcher"),
            patch("core.config.models.load_config", side_effect=ConfigError("skip")),
        ):
            from core.tooling.handler import ToolHandler

            handler = ToolHandler(anima_dir=anima_dir, memory=memory)
        return handler

    def test_logs_up_to_1000_chars(self, tmp_path: Path):
        """call_human should log up to 1000 characters of the body."""
        handler = self._make_handler(tmp_path)
        body = "A" * 1200
        logged_content = None
        original_log = handler._activity.log

        def capture_log(event_type, **kwargs):
            nonlocal logged_content
            if event_type == "human_notify":
                logged_content = kwargs.get("content", "")
            return original_log(event_type, **kwargs)

        handler._activity.log = capture_log
        handler._log_tool_activity("call_human", {"body": body})

        assert logged_content is not None
        assert len(logged_content) == 1000

    def test_short_body_not_truncated(self, tmp_path: Path):
        """Bodies shorter than 1000 chars are preserved."""
        handler = self._make_handler(tmp_path)
        body = "Short notification"
        logged_content = None
        original_log = handler._activity.log

        def capture_log(event_type, **kwargs):
            nonlocal logged_content
            if event_type == "human_notify":
                logged_content = kwargs.get("content", "")
            return original_log(event_type, **kwargs)

        handler._activity.log = capture_log
        handler._log_tool_activity("call_human", {"body": body})

        assert logged_content == body
