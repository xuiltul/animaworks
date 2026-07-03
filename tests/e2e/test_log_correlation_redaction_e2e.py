from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""E2E tests for log cycle correlation, errors.log separation, and redaction.

These exercise the REAL logging handlers with real on-disk file I/O (no mocks):
  - ``errors.log`` receives WARNING+ only; INFO stays out of it.
  - Secrets written through any handler are masked on disk.
  - Per-anima plain logs carry ``[cycle_id]`` while a cycle is bound.
"""

import logging

import pytest
import structlog

from core.logging_config import (
    bind_cycle_context,
    clear_cycle_context,
    setup_anima_logging,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Restore a clean root logger after each test (setup_* mutate the root)."""
    yield
    structlog.contextvars.clear_contextvars()
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


def _flush_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_errors_log_receives_warning_and_above_only(tmp_path):
    setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)

    logger = logging.getLogger("test.e2e.errors")
    logger.info("an informational line about routine work")
    logger.warning("a warning that should be triaged")
    logger.error("an error that should be triaged")
    _flush_handlers()

    errors_content = (tmp_path / "errors.log").read_text(encoding="utf-8")
    main_content = (tmp_path / "animaworks.log").read_text(encoding="utf-8")

    # WARNING+ present in errors.log, INFO absent.
    assert "a warning that should be triaged" in errors_content
    assert "an error that should be triaged" in errors_content
    assert "an informational line about routine work" not in errors_content

    # The main log keeps everything.
    assert "an informational line about routine work" in main_content
    assert "a warning that should be triaged" in main_content


def test_secret_is_masked_on_disk_across_handlers(tmp_path):
    setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)

    logging.getLogger("test.e2e.redact").warning(
        "outbound call failed with key sk-ant-api03-SUPERSECRETVALUE123456"
    )
    _flush_handlers()

    errors_content = (tmp_path / "errors.log").read_text(encoding="utf-8")
    main_content = (tmp_path / "animaworks.log").read_text(encoding="utf-8")

    for content in (errors_content, main_content):
        assert "***REDACTED***" in content
        assert "sk-ant-api03-SUPERSECRETVALUE123456" not in content


def test_anima_plain_log_carries_cycle_id_and_redacts(tmp_path):
    from core.time_utils import configure_timezone

    # The per-anima daily handler resolves the local date on every emit; without
    # a configured timezone that path debug-logs and would recurse. The real
    # runner configures the timezone at startup, so mirror that here.
    configure_timezone("Asia/Tokyo")
    setup_anima_logging(
        anima_name="aoi",
        log_dir=tmp_path,
        level="INFO",
        also_to_console=False,
    )

    bind_cycle_context("feedface", "heartbeat")
    try:
        logging.getLogger("test.e2e.anima").warning(
            "leaking token=plaintextsecret9999 during cycle"
        )
    finally:
        clear_cycle_context()
    _flush_handlers()

    anima_dir = tmp_path / "animas" / "aoi"
    # The dated daily log is named YYYYMMDD.log (not current.log / errors.log).
    dated = [p for p in anima_dir.glob("*.log") if p.stem.isdigit()]
    content = dated[0].read_text(encoding="utf-8")

    assert "[feedface]" in content
    assert "***REDACTED***" in content
    assert "plaintextsecret9999" not in content


def test_exception_traceback_and_stack_masked_all_sinks(tmp_path):
    """R1: secrets in the exception message, traceback, and stack are masked in
    every structlog sink (animaworks.log JSON + global errors.log)."""
    setup_logging(level="DEBUG", log_dir=tmp_path, json_file=True)

    try:
        raise ValueError("inner failure password=hunter2secretvalue and api_key=abcd12345678")
    except ValueError:
        logging.getLogger("test.e2e.exc").exception(
            "outer op failed sk-ant-MSGSECRET1234567", stack_info=True
        )
    _flush_handlers()

    main_content = (tmp_path / "animaworks.log").read_text(encoding="utf-8")
    errors_content = (tmp_path / "errors.log").read_text(encoding="utf-8")

    leaked = ("hunter2secretvalue", "abcd12345678", "sk-ant-MSGSECRET1234567")
    for content in (main_content, errors_content):
        assert "***REDACTED***" in content
        for secret in leaked:
            assert secret not in content
        # Traceback still present (masked, not dropped).
        assert "ValueError" in content


def test_per_anima_errors_log_masks_exception(tmp_path):
    """R1+R3: per-anima errors.log exists, is WARNING+ only, and masks the
    traceback of a logged exception on the plain-formatter path."""
    from core.time_utils import configure_timezone

    configure_timezone("Asia/Tokyo")
    setup_anima_logging(
        anima_name="yuki", log_dir=tmp_path, level="INFO", also_to_console=False
    )

    logging.getLogger("test.e2e.anima2").info("routine startup line")
    try:
        raise RuntimeError("anima crash token=plaintextsecret9999")
    except RuntimeError:
        logging.getLogger("test.e2e.anima2").exception("cycle blew up", stack_info=True)
    _flush_handlers()

    errors_path = tmp_path / "animas" / "yuki" / "errors.log"
    assert errors_path.exists()
    content = errors_path.read_text(encoding="utf-8")

    assert "routine startup line" not in content  # WARNING+ only
    assert "plaintextsecret9999" not in content
    assert "***REDACTED***" in content
    assert "RuntimeError" in content  # traceback preserved
