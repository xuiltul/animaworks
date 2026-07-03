"""Unit tests for core/logging_config.py — structlog-based logging setup."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
import structlog

from core.logging_config import (
    CycleContextFilter,
    SecretRedactionFilter,
    _AnimaDailyFileHandler,
    _redact_secrets,
    _redaction_processor,
    attach_standard_log_filters,
    bind_cycle_context,
    clear_cycle_context,
    get_request_id,
    set_request_id,
    setup_anima_logging,
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
        # console + animaworks.log + errors.log
        assert len(root.handlers) == 3
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

    def test_with_file_handler_plain(self, tmp_path):
        setup_logging(level="WARNING", log_dir=tmp_path, json_file=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        # console + animaworks.log + errors.log
        assert len(root.handlers) == 3

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


# ── Cycle correlation ─────────────────────────────────────


class TestCycleContext:
    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def _make_record(self) -> logging.LogRecord:
        return logging.getLogger("test.cycle").makeRecord(
            "test.cycle", logging.INFO, __file__, 1, "hello", (), None
        )

    def test_unbound_cycle_id_is_dash_plain(self):
        record = self._make_record()
        CycleContextFilter().filter(record)
        assert record.cycle_id == "-"

    def test_bound_cycle_id_on_record_plain(self):
        bind_cycle_context("abc12345", "heartbeat")
        record = self._make_record()
        CycleContextFilter().filter(record)
        assert record.cycle_id == "abc12345"

    def test_plain_format_includes_cycle_id(self):
        bind_cycle_context("deadbeef", "cron")
        record = self._make_record()
        CycleContextFilter().filter(record)
        rendered = logging.Formatter(
            "%(levelname)s [%(cycle_id)s] %(message)s"
        ).format(record)
        assert "[deadbeef]" in rendered

    def test_cycle_id_and_trigger_in_json_log(self, tmp_path):
        import json

        setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)
        bind_cycle_context("c0ffee00", "heartbeat")

        logging.getLogger("test.cycle.json").info("in a cycle")

        content = (tmp_path / "animaworks.log").read_text(encoding="utf-8").strip()
        data = json.loads(content.splitlines()[-1])
        assert data.get("cycle_id") == "c0ffee00"
        assert data.get("trigger") == "heartbeat"

    def test_nested_cycle_restores_outer(self):
        outer = bind_cycle_context("outer001", "cron")
        assert structlog.contextvars.get_contextvars()["cycle_id"] == "outer001"

        inner = bind_cycle_context("inner002", "chat")
        assert structlog.contextvars.get_contextvars()["cycle_id"] == "inner002"

        clear_cycle_context(inner)
        # Outer context restored, not wiped.
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["cycle_id"] == "outer001"
        assert ctx["trigger"] == "cron"

        clear_cycle_context(outer)
        assert "cycle_id" not in structlog.contextvars.get_contextvars()

    def test_clear_without_tokens_unbinds(self):
        bind_cycle_context("x1", "manual")
        clear_cycle_context()
        assert "cycle_id" not in structlog.contextvars.get_contextvars()


# ── Secret redaction ──────────────────────────────────────


class TestSecretRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-api03-abcDEF123456_789",
            "sk-abcdefghij0123456789klmnop",
            "xoxb-123456789012-abcdefABCDEF",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "AKIAIOSFODNN7EXAMPLE",
            "api_key=supersecretvalue123",
        ],
    )
    def test_each_pattern_is_masked(self, secret):
        text = f"leaked here: {secret} <- gone"
        out = _redact_secrets(text)
        assert "***REDACTED***" in out
        # No raw secret body remains.
        assert secret not in out

    def test_all_seven_keyword_forms(self):
        for key in ("api_key", "token", "secret", "password"):
            out = _redact_secrets(f"{key}=abcdefgh12345")
            assert out == "***REDACTED***"

    def test_precheck_passthrough_returns_same_object(self):
        benign = "just a normal log line with no credentials"
        out = _redact_secrets(benign)
        # Fast path returns the input unchanged (identity, no regex work).
        assert out is benign

    def test_filter_masks_record_msg(self):
        record = logging.getLogger("t").makeRecord(
            "t", logging.WARNING, __file__, 1, "key sk-ant-abcDEF123456789", (), None
        )
        SecretRedactionFilter().filter(record)
        assert "***REDACTED***" in record.getMessage()
        assert "sk-ant-abcDEF123456789" not in record.getMessage()

    def test_filter_folds_args(self):
        record = logging.getLogger("t").makeRecord(
            "t", logging.WARNING, __file__, 1, "token=%s here", ("abcdefgh12345",), None
        )
        SecretRedactionFilter().filter(record)
        assert record.args is None
        assert "abcdefgh12345" not in record.getMessage()
        assert "***REDACTED***" in record.getMessage()

    def test_filter_leaves_non_string_msg(self):
        # structlog-native records carry a dict in msg; the filter must not touch it.
        payload = {"event": "sk-ant-shouldstayhere123456", "level": "info"}
        record = logging.getLogger("t").makeRecord(
            "t", logging.INFO, __file__, 1, payload, (), None
        )
        SecretRedactionFilter().filter(record)
        assert record.msg is payload

    def test_filter_passes_through_on_exception(self, monkeypatch):
        monkeypatch.setattr(
            "core.logging_config._redact_secrets",
            lambda _t: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        record = logging.getLogger("t").makeRecord(
            "t", logging.WARNING, __file__, 1, "sk-ant-keepme123456789", (), None
        )
        # Must not raise, and the original message survives (no dropped line).
        assert SecretRedactionFilter().filter(record) is True
        assert record.msg == "sk-ant-keepme123456789"

    # ── R1: exception / stack masking on the plain path ──────────

    def _record_with_exc(self, *, stack: str | None = None) -> logging.LogRecord:
        try:
            raise ValueError("boom password=hunter2secretvalue")
        except ValueError:
            exc_info = __import__("sys").exc_info()
        record = logging.getLogger("t").makeRecord(
            "t", logging.ERROR, __file__, 1, "op failed sk-ant-msgsecret123456", (), exc_info
        )
        if stack is not None:
            record.stack_info = stack
        return record

    def test_filter_masks_exception_into_exc_text(self):
        record = self._record_with_exc()
        SecretRedactionFilter().filter(record)
        # exc_text is now materialised and masked; a plain Formatter uses it verbatim.
        assert record.exc_text is not None
        assert "hunter2secretvalue" not in record.exc_text
        assert "***REDACTED***" in record.exc_text
        rendered = logging.Formatter("%(message)s").format(record)
        assert "hunter2secretvalue" not in rendered
        assert "sk-ant-msgsecret123456" not in rendered

    def test_filter_masks_stack_info(self):
        record = self._record_with_exc(
            stack="Stack:\n  call token=stacksecret12345 here"
        )
        SecretRedactionFilter().filter(record)
        assert "stacksecret12345" not in record.stack_info
        assert "***REDACTED***" in record.stack_info

    def test_filter_exc_masking_is_idempotent_across_handlers(self):
        record = self._record_with_exc()
        f = SecretRedactionFilter()
        f.filter(record)
        first = record.exc_text
        f.filter(record)  # second handler visiting the same record
        assert record.exc_text == first
        assert "hunter2secretvalue" not in record.exc_text

    # ── R1: redaction processor for the structlog rendered line ──

    def test_redaction_processor_masks_rendered_string(self):
        out = _redaction_processor(None, "error", 'msg with token=procsecret12345 x')
        assert "procsecret12345" not in out
        assert "***REDACTED***" in out

    def test_redaction_processor_passes_non_string(self):
        obj = {"not": "a string"}
        assert _redaction_processor(None, "error", obj) is obj

    # ── R6: keyword prechecks gated to assignment context ────────

    def test_keyword_without_assignment_is_passthrough(self):
        # Generic words / metric fields must not trigger the full regex.
        for benign in ("max_tokens=100", "input_tokens: 512", "the secret meeting",
                       "password strength is high"):
            assert _redact_secrets(benign) is benign

    def test_keyword_with_assignment_is_masked(self):
        for text in ("token=realsecretvalue1", "token: realsecretvalue1",
                     '"token":"realsecretvalue1"'):
            assert "***REDACTED***" in _redact_secrets(text)


class TestStandardLogFilters:
    def test_attaches_cycle_and_redaction(self):
        handler = logging.NullHandler()
        attach_standard_log_filters(handler)
        assert any(isinstance(f, CycleContextFilter) for f in handler.filters)
        assert any(isinstance(f, SecretRedactionFilter) for f in handler.filters)

    def test_disabled_attaches_cycle_only(self):
        handler = logging.NullHandler()
        attach_standard_log_filters(handler, redaction_enabled=False)
        assert any(isinstance(f, CycleContextFilter) for f in handler.filters)
        assert not any(isinstance(f, SecretRedactionFilter) for f in handler.filters)


class TestRedactionToggle:
    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        yield
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_disabled_attaches_no_filter(self, tmp_path):
        setup_logging(level="INFO", log_dir=tmp_path, redaction_enabled=False)
        root = logging.getLogger()
        for handler in root.handlers:
            assert not any(
                isinstance(f, SecretRedactionFilter) for f in handler.filters
            )

    def test_enabled_attaches_filter_to_all_handlers(self, tmp_path):
        setup_logging(level="INFO", log_dir=tmp_path, redaction_enabled=True)
        root = logging.getLogger()
        for handler in root.handlers:
            assert any(
                isinstance(f, SecretRedactionFilter) for f in handler.filters
            )

    def test_disabled_writes_raw_secret(self, tmp_path):
        setup_logging(level="INFO", log_dir=tmp_path, redaction_enabled=False)
        logging.getLogger("test.raw").warning("token is sk-ant-rawvalue123456")
        content = (tmp_path / "animaworks.log").read_text(encoding="utf-8")
        assert "sk-ant-rawvalue123456" in content
        assert "***REDACTED***" not in content


class TestAnimaErrorsLog:
    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        from core.time_utils import configure_timezone

        configure_timezone("Asia/Tokyo")
        yield
        root = logging.getLogger()
        for handler in list(root.handlers):
            handler.close()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_per_anima_errors_log_warning_only(self, tmp_path):
        setup_anima_logging(
            anima_name="aoi", log_dir=tmp_path, level="INFO", also_to_console=False
        )
        logging.getLogger("test.anima.errsplit").info("routine info line")
        logging.getLogger("test.anima.errsplit").warning("a warning worth triage")
        for handler in logging.getLogger().handlers:
            handler.flush()

        errors_path = tmp_path / "animas" / "aoi" / "errors.log"
        assert errors_path.exists()
        content = errors_path.read_text(encoding="utf-8")
        assert "a warning worth triage" in content
        assert "routine info line" not in content


def test_anima_daily_file_handler_switches_to_new_date_without_renaming_old_log(tmp_path):
    tz = ZoneInfo("Asia/Tokyo")
    day1 = datetime(2026, 6, 11, 23, 59, tzinfo=tz)
    day2 = datetime(2026, 6, 12, 0, 1, tzinfo=tz)

    with patch("core.logging_config.now_local", return_value=day1):
        handler = _AnimaDailyFileHandler(tmp_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.anima.daily")

    try:
        with patch("core.logging_config.now_local", return_value=day1):
            handler.emit(logger.makeRecord(logger.name, logging.INFO, __file__, 1, "before midnight", (), None))
        with patch("core.logging_config.now_local", return_value=day2):
            handler.emit(logger.makeRecord(logger.name, logging.INFO, __file__, 1, "after midnight", (), None))
    finally:
        handler.close()

    assert (tmp_path / "20260611.log").read_text(encoding="utf-8").strip() == "before midnight"
    assert (tmp_path / "20260612.log").read_text(encoding="utf-8").strip() == "after midnight"
    assert not (tmp_path / "20260611.log.20260612.log").exists()
