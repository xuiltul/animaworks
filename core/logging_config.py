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
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from core.time_utils import now_local

# Re-export for backward compatibility with existing imports
# (e.g. ``from core.logging_config import set_request_id``)


def set_request_id(request_id: str) -> None:
    """Set the current request ID via structlog contextvars."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def get_request_id() -> str:
    """Get the current request ID from structlog contextvars."""
    ctx = structlog.contextvars.get_contextvars()
    return ctx.get("request_id", "-")


# ── Cycle correlation context ──────────────────────────────────


def bind_cycle_context(cycle_id: str, trigger: str) -> dict:
    """Bind ``cycle_id`` / ``trigger`` for the current agent cycle.

    Returns the reset tokens from ``bind_contextvars`` so a nested cycle can
    restore the outer cycle's context on exit (see ``clear_cycle_context``).
    ``cycle_id`` lands in JSON logs automatically via ``merge_contextvars``.
    """
    return structlog.contextvars.bind_contextvars(cycle_id=cycle_id, trigger=trigger)


def clear_cycle_context(tokens: dict | None = None) -> None:
    """Clear cycle context bound by ``bind_cycle_context``.

    When ``tokens`` (the return value of ``bind_cycle_context``) is given, the
    previous context is restored -- this preserves an outer cycle's binding
    when cycles nest. Without tokens the keys are simply unbound.
    """
    if tokens is not None:
        structlog.contextvars.reset_contextvars(**tokens)
    else:
        structlog.contextvars.unbind_contextvars("cycle_id", "trigger")


class CycleContextFilter(logging.Filter):
    """Inject the current ``cycle_id`` into log records for plain formatters.

    JSON output already carries ``cycle_id`` via ``merge_contextvars``; this
    filter exposes it as a ``%(cycle_id)s`` attribute for the per-anima plain
    text logs. Records emitted outside a cycle get ``-``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = structlog.contextvars.get_contextvars()
        record.cycle_id = ctx.get("cycle_id", "-")  # type: ignore[attr-defined]
        return True


# ── Secret redaction ───────────────────────────────────────────

# Combined pattern for known credential shapes. Kept case-sensitive to match
# the documented seed patterns and minimise false positives.
_REDACTION_PATTERN = re.compile(
    "|".join(
        (
            r"sk-ant-[\w-]+",
            r"sk-[\w-]{20,}",
            r"xox[abps]-[\w-]+",
            r"ghp_[\w]+",
            r"Bearer [\w.\-+/=]+",
            r"AKIA[A-Z0-9]{16}",
            r"(?:api_key|token|secret|password)[\"']?\s*[:=]\s*[\"']?[\w.\-+/]{8,}",
        )
    )
)

# Cheap pre-check. Prefix triggers are distinctive enough to gate on as bare
# substrings. The keyword triggers (token/secret/password/api_key) are common
# English words that appear constantly in benign LLM logs (``max_tokens`` etc.),
# so they gate only in an assignment context (``token=`` / ``token:`` / ``token"``)
# to avoid running the full regex on every ordinary line (R6).
_REDACTION_PREFIX_TRIGGERS = (
    "sk-",
    "xox",
    "ghp_",
    "Bearer",
    "AKIA",
)
_REDACTION_KEYWORD_PRECHECK = re.compile(r"(?:api_key|token|secret|password)[\"']?\s*[:=]")

_REDACTION_PLACEHOLDER = "***REDACTED***"


def _redact_secrets(text: str) -> str:
    """Mask known credential shapes in ``text`` behind a fast pre-check gate."""
    if not (
        any(trigger in text for trigger in _REDACTION_PREFIX_TRIGGERS)
        or _REDACTION_KEYWORD_PRECHECK.search(text)
    ):
        return text
    return _REDACTION_PATTERN.sub(_REDACTION_PLACEHOLDER, text)


def _redaction_processor(logger: object, name: str, event_string: object) -> object:
    """Final ``ProcessorFormatter`` step: mask secrets in the rendered line.

    Runs after the renderer so it also covers the traceback and stack that
    structlog renders from the record's ``exc_info`` / ``stack_info`` -- those
    are produced inside the formatter pipeline and cannot be reached by the
    pre-format ``SecretRedactionFilter`` on the structlog (JSON/console) path.
    """
    if isinstance(event_string, str):
        return _redact_secrets(event_string)
    return event_string


# Rendering-only formatter used to materialise a record's traceback for masking.
_EXC_FORMATTER = logging.Formatter()


class SecretRedactionFilter(logging.Filter):
    """Mask secrets on log records before they are formatted to a handler.

    Covers, for plain (stdlib ``logging.Formatter``) handlers:
    - ``record.msg`` / ``record.args`` (the message),
    - the exception traceback (rendered once into ``record.exc_text``), and
    - the ``stack_info`` string.

    The structlog (JSON/console) path renders exception/stack from the raw
    ``exc_info`` inside the formatter, so those handlers additionally carry
    ``_redaction_processor`` to mask the final rendered string. Any failure is
    swallowed and the record passes through unchanged -- a missed mask is
    preferable to a dropped log line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.msg
            # structlog-native records carry an event dict in ``msg``; leave
            # them to the ProcessorFormatter and only touch string messages.
            if isinstance(msg, str):
                if record.args:
                    # Fold args in first so secrets spanning the msg/args
                    # boundary are caught, then clear args (already applied).
                    record.msg = _redact_secrets(record.getMessage())
                    record.args = None
                else:
                    record.msg = _redact_secrets(msg)
            # Exception traceback: render once, mask, and cache in exc_text so
            # plain Formatters emit the masked text without re-rendering from
            # exc_info. Idempotent across the multiple handlers a record visits.
            if record.exc_text:
                record.exc_text = _redact_secrets(record.exc_text)
            elif record.exc_info:
                record.exc_text = _redact_secrets(
                    _EXC_FORMATTER.formatException(record.exc_info)
                )
            # stack_info (stack_info=True) is emitted verbatim by plain
            # Formatters; mask it in place.
            if record.stack_info:
                record.stack_info = _redact_secrets(record.stack_info)
        except Exception:
            logging.getLogger(__name__).debug(
                "SecretRedactionFilter failed; passing record through", exc_info=True
            )
        return True


def attach_standard_log_filters(
    handler: logging.Handler, *, redaction_enabled: bool = True
) -> None:
    """Attach the standard cross-cutting log filters to *handler*.

    - ``CycleContextFilter`` exposes ``cycle_id`` for ``%(cycle_id)s`` formats.
    - ``SecretRedactionFilter`` masks secrets in message/args/traceback/stack.

    Use for any handler that bypasses the root pipeline -- e.g. a logger with
    ``propagate=False`` and its own handler (``animaworks.frontend``) -- so it
    still gets cycle correlation and redaction.
    """
    handler.addFilter(CycleContextFilter())
    if redaction_enabled:
        handler.addFilter(SecretRedactionFilter())


class _AnimaDailyFileHandler(logging.FileHandler):
    """Write each local day to ``YYYYMMDD.log`` without renaming old files."""

    def __init__(self, log_dir: Path, *, encoding: str = "utf-8") -> None:
        self.log_dir = log_dir
        self.current_day = now_local().strftime("%Y%m%d")
        super().__init__(self.log_dir / f"{self.current_day}.log", encoding=encoding)

    def _update_current_link(self) -> None:
        current_link = self.log_dir / "current.log"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        try:
            current_link.symlink_to(f"{self.current_day}.log")
        except OSError:
            current_link.write_text(f"{self.current_day}.log", encoding="utf-8")

    def doRollover(self) -> None:  # noqa: N802 - stdlib handler API
        new_day = now_local().strftime("%Y%m%d")
        if new_day == self.current_day:
            return
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_day = new_day
        self.baseFilename = str(self.log_dir / f"{self.current_day}.log")
        self.stream = self._open()
        self._update_current_link()

    def emit(self, record: logging.LogRecord) -> None:
        self.doRollover()
        super().emit(record)


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


def _build_processor_formatter(
    renderer: object,
    foreign_pre_chain: list,
    *,
    redaction_enabled: bool = True,
) -> structlog.stdlib.ProcessorFormatter:
    """Build a ``ProcessorFormatter`` for *renderer*, masking the rendered line.

    When ``redaction_enabled``, ``_redaction_processor`` is appended so the
    final string (including any traceback/stack structlog renders here) is
    masked.
    """
    processors: list = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        renderer,
    ]
    if redaction_enabled:
        processors.append(_redaction_processor)
    return structlog.stdlib.ProcessorFormatter(
        processors=processors,
        foreign_pre_chain=foreign_pre_chain,
    )


def _build_json_renderer() -> object:
    """Return a JSON renderer (orjson when available, stdlib json otherwise)."""
    try:
        import orjson

        def _orjson_serializer(obj: object, **_kw) -> str:  # noqa: ANN001
            return orjson.dumps(obj).decode("utf-8")

        return structlog.processors.JSONRenderer(serializer=_orjson_serializer)
    except ImportError:
        return structlog.processors.JSONRenderer()


# ── Main Setup ─────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    json_file: bool = True,
    redaction_enabled: bool = True,
) -> None:
    """Configure logging for the entire AnimaWorks process.

    Uses structlog's stdlib integration so that existing
    ``logging.getLogger("animaworks.xxx")`` calls are routed through
    structlog's processor pipeline automatically.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, etc.).
        log_dir: Directory for log files. If None, file logging is disabled.
        json_file: Whether to use JSON format for the file handler.
        redaction_enabled: When True, attach a secret-masking filter to every
            handler. Set False for raw-log debugging.
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
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(
        _build_processor_formatter(
            structlog.dev.ConsoleRenderer(),
            foreign_pre_chain,
            redaction_enabled=redaction_enabled,
        )
    )
    attach_standard_log_filters(console, redaction_enabled=redaction_enabled)
    root.addHandler(console)

    # File handler: rotated, JSON (via orjson for performance)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "animaworks.log"

        renderer = (
            _build_json_renderer() if json_file else structlog.dev.ConsoleRenderer(colors=False)
        )
        file_formatter = _build_processor_formatter(
            renderer, foreign_pre_chain, redaction_enabled=redaction_enabled
        )

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        attach_standard_log_filters(file_handler, redaction_enabled=redaction_enabled)
        root.addHandler(file_handler)

        # Severity-separated triage log: WARNING+ only, always JSON.
        errors_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        errors_handler.setLevel(logging.WARNING)
        errors_handler.setFormatter(
            _build_processor_formatter(
                _build_json_renderer(), foreign_pre_chain, redaction_enabled=redaction_enabled
            )
        )
        attach_standard_log_filters(errors_handler, redaction_enabled=redaction_enabled)
        root.addHandler(errors_handler)

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
    redaction_enabled: bool = True,
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
        redaction_enabled: When True, attach a secret-masking filter to every
            handler. Set False for raw-log debugging.

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
    log_file = anima_log_dir / f"{now_local().strftime('%Y%m%d')}.log"

    # Setup root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Anima name is anima-specific; cycle correlation + redaction come from the
    # shared standard-filter helper.
    anima_filter = AnimaNameFilter(anima_name)

    def _attach_filters(handler: logging.Handler) -> None:
        handler.addFilter(anima_filter)
        attach_standard_log_filters(handler, redaction_enabled=redaction_enabled)

    _plain_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(cycle_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with date-name rotation. Retention is handled by housekeeping.
    file_handler = _AnimaDailyFileHandler(anima_log_dir, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_plain_fmt)
    _attach_filters(file_handler)
    root.addHandler(file_handler)

    # Per-anima severity-separated triage log (WARNING+). Anima child processes
    # are the real incident source, so their WARNING+ needs a dedicated file;
    # one process owns this file, so a size-based RotatingFileHandler is safe.
    errors_handler = RotatingFileHandler(
        anima_log_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    errors_handler.setLevel(logging.WARNING)
    errors_handler.setFormatter(_plain_fmt)
    _attach_filters(errors_handler)
    root.addHandler(errors_handler)

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
                fmt=f"[{anima_name}] %(asctime)s [%(levelname)s] [%(cycle_id)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        _attach_filters(console)
        root.addHandler(console)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Anima logging configured: {anima_name} -> {log_file}")
