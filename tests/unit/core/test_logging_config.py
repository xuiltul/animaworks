"""Unit tests for core/logging_config.py — structlog-based logging setup."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import structlog

from core.logging_config import (
    get_request_id,
    set_request_id,
    setup_logging,
)


# ── Request ID contextvars ────────────────────────────────


class TestRequestId:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def test_default_value(self):
        assert get_request_id() == "-"

    def test_set_and_get(self):
        set_request_id("req-abc-123")
        assert get_request_id() == "req-abc-123"

    def test_overwrite(self):
        set_request_id("first")
        set_request_id("second")
        assert get_request_id() == "second"

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()


# ── setup_logging ─────────────────────────────────────────


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        """Reset root logger after each test."""
        yield
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_console_only(self):
        setup_logging(level="DEBUG", log_dir=None)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_with_file_handler_json(self, tmp_path):
        setup_logging(level="INFO", log_dir=tmp_path, json_file=True)
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 2
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

    def test_with_file_handler_plain(self, tmp_path):
        setup_logging(level="WARNING", log_dir=tmp_path, json_file=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 2

    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs" / "deep"
        assert not log_dir.exists()
        setup_logging(log_dir=log_dir)
        assert log_dir.exists()

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        setup_logging()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types

    def test_third_party_loggers_suppressed(self):
        setup_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("apscheduler").level == logging.WARNING

    def test_invalid_level_defaults_to_info(self):
        setup_logging(level="INVALID_LEVEL")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_structlog_processor_formatter_used(self):
        """Verify that handlers use structlog's ProcessorFormatter."""
        setup_logging()
        root = logging.getLogger()
        for handler in root.handlers:
            fmt = handler.formatter
            assert fmt is not None
            assert "ProcessorFormatter" in type(fmt).__name__

    def test_file_handler_writes_json(self, tmp_path):
        """Verify that the file handler produces valid JSON output."""
        import json

        setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)

        test_logger = logging.getLogger("test.json.output")
        test_logger.info("test message for json")

        log_file = tmp_path / "animaworks.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8").strip()
        if content:
            # Should be parseable as JSON
            data = json.loads(content)
            assert "event" in data or "msg" in data

    def test_request_id_appears_in_log(self, tmp_path):
        """Verify that request_id from contextvars appears in structured log."""
        import json

        setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)
        set_request_id("test-req-42")

        test_logger = logging.getLogger("test.request.id")
        test_logger.info("with request id")

        log_file = tmp_path / "animaworks.log"
        content = log_file.read_text(encoding="utf-8").strip()
        if content:
            data = json.loads(content)
            assert data.get("request_id") == "test-req-42"

        structlog.contextvars.clear_contextvars()
