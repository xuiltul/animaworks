# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Centralized logging configuration for AnimaWorks.

Uses structlog in stdlib-compatible mode so that existing
``logging.getLogger()`` calls continue to work while gaining
structured logging capabilities (context binding, JSON output, etc.).

Provides:
- setup_logging(): structlog + stdlib unified setup (console + file)
- setup_anima_logging(): per-anima daily log rotation
- set_request_id() / get_request_id(): backward-compatible request ID helpers
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

import structlog

# Re-export for backward compatibility with existing imports
# (e.g. ``from core.logging_config import set_request_id``)


def set_request_id(request_id: str) -> None:
    """Set the current request ID via structlog contextvars."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def get_request_id() -> str:
    """Get the current request ID from structlog contextvars."""
    ctx = structlog.contextvars.get_contextvars()
    return ctx.get("request_id", "-")


# ── Shared Processors ──────────────────────────────────────────


def _build_shared_processors() -> list:
    """Build the shared processor chain used by both structlog and stdlib."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


# ── Main Setup ─────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    json_file: bool = True,
) -> None:
    """Configure logging for the entire AnimaWorks process.

    Uses structlog's stdlib integration so that existing
    ``logging.getLogger("animaworks.xxx")`` calls are routed through
    structlog's processor pipeline automatically.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, etc.).
        log_dir: Directory for log files. If None, file logging is disabled.
        json_file: Whether to use JSON format for the file handler.
    """
    shared_processors = _build_shared_processors()

    # ── Configure structlog itself ──────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── Configure stdlib root logger ────────────────────────
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # foreign_pre_chain: processes stdlib LogRecords through structlog pipeline
    # so that contextvars (request_id etc.) and timestamps are merged in.
    foreign_pre_chain = list(shared_processors)

    # Console handler: human-readable colored output
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=foreign_pre_chain,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(console_formatter)
    root.addHandler(console)

    # File handler: rotated, JSON (via orjson for performance)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "animaworks.log"

        if json_file:
            try:
                import orjson

                def _orjson_serializer(obj: object, **_kw) -> str:  # noqa: ANN001
                    return orjson.dumps(obj).decode("utf-8")

                file_formatter = structlog.stdlib.ProcessorFormatter(
                    processors=[
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(
                            serializer=_orjson_serializer,
                        ),
                    ],
                    foreign_pre_chain=foreign_pre_chain,
                )
            except ImportError:
                # Fallback if orjson is unavailable
                file_formatter = structlog.stdlib.ProcessorFormatter(
                    processors=[
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                    foreign_pre_chain=foreign_pre_chain,
                )
        else:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
                foreign_pre_chain=foreign_pre_chain,
            )

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


# ── Anima-specific Logging ────────────────────────────────────────────


class AnimaNameFilter(logging.Filter):
    """Inject anima name into log records."""

    def __init__(self, anima_name: str):
        super().__init__()
        self.anima_name = anima_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.anima_name = self.anima_name  # type: ignore[attr-defined]
        return True


def setup_anima_logging(
    anima_name: str,
    log_dir: Path,
    level: str = "INFO",
    also_to_console: bool = True,
) -> None:
    """Configure anima-specific logging with daily rotation.

    Creates a dedicated log directory for the anima with:
    - Daily log rotation (YYYYMMDD.log format)
    - 30-day retention
    - current.log symlink to the current log file
    - Optional console output

    Args:
        anima_name: Name of the anima (used for log directory and prefix)
        log_dir: Base log directory (e.g., ~/.animaworks/logs)
        level: Log level (DEBUG, INFO, WARNING, etc.)
        also_to_console: Whether to also log to console

    Directory structure created:
        {log_dir}/animas/{anima_name}/
        |-- current.log -> 20260214.log
        |-- 20260214.log
        |-- 20260213.log
        +-- ...
    """
    # Create anima log directory
    anima_log_dir = log_dir / "animas" / anima_name
    anima_log_dir.mkdir(parents=True, exist_ok=True)

    # Main log file with daily rotation
    log_file = anima_log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"

    # Setup root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Create anima name filter
    anima_filter = AnimaNameFilter(anima_name)

    # File handler with timed rotation
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler.addFilter(anima_filter)
    file_handler.suffix = "%Y%m%d.log"  # Match filename format
    root.addHandler(file_handler)

    # Create/update current.log symlink
    current_link = anima_log_dir / "current.log"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    try:
        current_link.symlink_to(log_file.name)
    except OSError:
        # On Windows, symlinks may require admin privileges
        # Fall back to copying the path as a text file reference
        current_link.write_text(str(log_file.name))

    # Optional console handler
    if also_to_console:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(
            logging.Formatter(
                fmt=f"[{anima_name}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        console.addFilter(anima_filter)
        root.addHandler(console)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Anima logging configured: {anima_name} -> {log_file}")
