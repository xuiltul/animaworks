"""Activity-log integration for ephemeral runtime model fallback."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from core.config.model_config import fallback_event_meta, resolve_effective_model_config
from core.execution.error_classifier import (
    FailoverReason,
    classify_llm_error,
    classify_llm_error_message,
)
from core.schemas import ModelConfig

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger

_T = TypeVar("_T")

_logger = logging.getLogger("animaworks.execution.fallback")


def log_model_fallback(
    activity: ActivityLogger,
    primary_config: ModelConfig,
    effective_config: ModelConfig,
    *,
    channel: str,
    phase: str,
) -> dict[str, Any] | None:
    """Record a ``model_fallback`` event and return its metadata.

    Callers pass their existing :class:`ActivityLogger`; this helper only
    standardizes the event shape shared by chat and background paths.
    """
    raw_meta = fallback_event_meta(primary_config, effective_config)
    if raw_meta is None:
        return None
    meta: dict[str, Any] = {**raw_meta, "phase": phase}
    activity.log(
        "model_fallback",
        summary=(
            f"Model fallback: {meta.get('primary', primary_config.model)}"
            f" -> {meta.get('fallback', effective_config.model)}"
        ),
        channel=channel,
        meta=meta,
        safe=True,
    )
    return meta


def preflight_fallback_config(
    anima_dir: Any,
    base_config: ModelConfig,
    *,
    channel: str,
) -> ModelConfig:
    """Resolve the rate-guard fallback for *base_config* and log the swap.

    Shared by every ``run_cycle`` route so a blocked primary is never dialled
    again.  Fail-open: any resolution problem returns *base_config* unchanged.
    """
    if not isinstance(base_config, ModelConfig) or not base_config.fallback_models:
        return base_config
    try:
        effective = resolve_effective_model_config(base_config)
    except Exception:  # pragma: no cover - defensive, fallback must never break a cycle
        _logger.debug("Fallback preflight failed; using primary", exc_info=True)
        return base_config
    if effective is base_config:
        return base_config
    try:
        from core.memory.activity import ActivityLogger

        log_model_fallback(
            ActivityLogger(anima_dir),
            base_config,
            effective,
            channel=channel,
            phase="preflight",
        )
    except Exception:  # pragma: no cover - activity logging is best-effort
        _logger.debug("Fallback preflight logging failed", exc_info=True)
    return effective


_CAPACITY_REASONS = frozenset(
    {
        FailoverReason.RATE_LIMIT,
        FailoverReason.OVERLOADED,
        FailoverReason.QUOTA_EXHAUSTED,
        FailoverReason.AUTH,
        FailoverReason.BILLING,
    }
)


def report_capacity_block(
    active_config: ModelConfig,
    reason: FailoverReason,
    hint: Any,
) -> None:
    """Register a fleet-wide block for the engine that just refused capacity.

    Backstop for executors with no ``report_block`` wiring of their own (Agent
    SDK, Cursor, Gemini): without a block the preflight has nothing to route
    around.  Executors that already reported leave the key blocked, so this is
    a no-op for them and the escalation counter is not double-counted.
    """
    if reason not in _CAPACITY_REASONS:
        return
    try:
        from core.config.io import load_config
        from core.config.model_config import _guard_key_for_model_config
        from core.execution.rate_guard import get_rate_guard

        guard = get_rate_guard()
        key = _guard_key_for_model_config(active_config, load_config())
        if guard.blocked_remaining(key) > 0:
            return
        long_lived = reason in (FailoverReason.QUOTA_EXHAUSTED, FailoverReason.AUTH, FailoverReason.BILLING)
        seconds = (
            guard.config.quota_block_seconds
            if long_lived
            else (getattr(hint, "backoff_s", None) or guard.config.default_block_seconds)
        )
        guard.report_block(key, seconds, reason.value)
        _logger.warning("Registered %s block for %s (%.0fs)", reason.value, key, seconds)
    except Exception:  # pragma: no cover - the guard is fail-open by design
        _logger.debug("Capacity block registration failed", exc_info=True)


def runtime_fallback_config(
    anima_dir: Any,
    primary_config: ModelConfig,
    active_config: ModelConfig,
    *,
    error_text: str,
    reason: str = "",
    channel: str,
) -> ModelConfig | None:
    """Return a different config to retry with after a fallback-safe failure.

    ``None`` means "do not retry": the error is not fallback-eligible, or the
    re-resolved config is the one that just failed.
    """
    if not isinstance(primary_config, ModelConfig) or not isinstance(active_config, ModelConfig):
        return None
    if not primary_config.fallback_models:
        return None
    try:
        classified, hint = classify_llm_error_message(f"{reason.replace('_', ' ')} {error_text}".strip())
        if not hint.fallback_ok:
            return None
        report_capacity_block(active_config, classified, hint)
        retry_config = resolve_effective_model_config(primary_config)
    except Exception:  # pragma: no cover - defensive
        _logger.debug("Runtime fallback resolution failed", exc_info=True)
        return None
    if all(
        getattr(retry_config, field, None) == getattr(active_config, field, None)
        for field in ("model", "execution_mode", "resolved_mode", "credential")
    ):
        return None
    try:
        from core.memory.activity import ActivityLogger

        log_model_fallback(
            ActivityLogger(anima_dir),
            primary_config,
            retry_config,
            channel=channel,
            phase="runtime_retry",
        )
    except Exception:  # pragma: no cover - activity logging is best-effort
        _logger.debug("Runtime fallback logging failed", exc_info=True)
    return retry_config


async def run_with_model_fallback(
    run: Callable[[ModelConfig], Awaitable[_T]],
    *,
    activity: ActivityLogger,
    primary_config: ModelConfig,
    active_config: ModelConfig,
    channel: str,
) -> _T:
    """Walk the configured priority list until one model succeeds.

    Some CLI executors return provider failures as ordinary response text
    (not an exception or ``action=error``).  Classify every result summary so
    those responses cannot escape as a successful chat reply.
    """
    current_config = active_config
    seen: set[tuple[Any, ...]] = set()
    last_result: _T | None = None
    last_failure: Exception | None = None

    while True:
        key = tuple(
            getattr(current_config, field, None) for field in ("model", "execution_mode", "resolved_mode", "credential")
        )
        if key in seen:
            if last_failure is not None:
                raise last_failure
            assert last_result is not None
            return last_result
        seen.add(key)

        last_failure = None
        try:
            result = await run(current_config)
        except Exception as exc:
            last_failure = exc
            reason, hint = classify_llm_error(exc)
            if not hint.fallback_ok:
                raise
        else:
            last_result = result
            data = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
            if not isinstance(data, dict):
                return result
            error_text = str(data.get("summary") or "")
            reason, hint = classify_llm_error_message(f"{data.get('reason') or ''} {error_text}".strip())
            explicit_error = data.get("action") == "error" or bool(data.get("reason"))
            if reason is FailoverReason.UNKNOWN and not explicit_error:
                return result
            if not hint.fallback_ok:
                return result

        report_capacity_block(current_config, reason, hint)
        retry_config = resolve_effective_model_config(primary_config)
        retry_key = tuple(
            getattr(retry_config, field, None) for field in ("model", "execution_mode", "resolved_mode", "credential")
        )
        if retry_key in seen:
            if last_failure is not None:
                raise last_failure
            assert last_result is not None
            return last_result

        log_model_fallback(
            activity,
            primary_config,
            retry_config,
            channel=channel,
            phase="runtime_retry",
        )
        current_config = retry_config


__all__ = [
    "log_model_fallback",
    "preflight_fallback_config",
    "run_with_model_fallback",
    "runtime_fallback_config",
]
