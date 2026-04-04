from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Usage Governor — automatic throttling based on provider usage data.

Periodically fetches usage from the internal usage module, evaluates a
configurable rule-set, and suspends cloud-provider Anima processes to
stay within budget.  Local-LLM (ollama) animas are never affected.

Policy is stored in ``usage_policy.json`` next to ``config.json``.
Governor runtime state (suspended animas) is kept in
``usage_governor_state.json`` so recovery works across restarts.

Credential-to-provider mapping:
  - ``anthropic`` → ``claude`` rules
  - ``openai``    → ``openai`` rules
  - ``ollama``    → exempt (local)

Rule mode:
  - **time_proportional** — ``usage_remaining_%`` must stay above
    ``time_remaining_%``; deficit (in percentage points) determines
    throttle severity.  Applied uniformly to all windows (5 h and 7 d).
  - **threshold** (list format, legacy) — fixed remaining-% cut-offs.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.i18n import t

logger = logging.getLogger("animaworks.usage_governor")

# Map anima credential values to policy provider keys
_CREDENTIAL_TO_PROVIDER: dict[str, str] = {
    "anthropic": "claude",
    "openai": "openai",
    "nanogpt": "nanogpt",
}

# ── Policy schema ────────────────────────────────────────────────────────────

DEFAULT_POLICY: dict[str, Any] = {
    "enabled": True,
    "check_interval_seconds": 120,
    "hard_floor_pct": 15,  # absolute minimum remaining % for any window
    "providers": {
        "claude": {
            # All windows use time-proportional mode:
            # usage_remaining_% must stay above time_remaining_% of the window.
            "five_hour": {
                "mode": "time_proportional",
                "deficit_rules": [
                    {"deficit_above": 0, "activity_level": 60},
                    {"deficit_above": 10, "activity_level": 30},
                    {"deficit_above": 20, "activity_level": 10},
                ],
            },
            "seven_day": {
                "mode": "time_proportional",
                "deficit_rules": [
                    {"deficit_above": 0, "activity_level": 60},
                    {"deficit_above": 10, "activity_level": 30},
                    {"deficit_above": 20, "activity_level": 10},
                ],
            },
        },
        "openai": {
            "5h": {
                "mode": "time_proportional",
                "deficit_rules": [
                    {"deficit_above": 0, "activity_level": 60},
                    {"deficit_above": 10, "activity_level": 30},
                    {"deficit_above": 20, "activity_level": 10},
                ],
            },
            "Week": {
                "mode": "time_proportional",
                "deficit_rules": [
                    {"deficit_above": 0, "activity_level": 60},
                    {"deficit_above": 10, "activity_level": 30},
                    {"deficit_above": 20, "activity_level": 10},
                ],
            },
        },
        "nanogpt": {
            "Week": {
                "mode": "time_proportional",
                "deficit_rules": [
                    {"deficit_above": 0, "activity_level": 60},
                    {"deficit_above": 10, "activity_level": 30},
                    {"deficit_above": 20, "activity_level": 10},
                ],
            },
        },
    },
    "suspend_thresholds": {
        "non_essential_below": 30,
        "all_except_coo_below": 15,
    },
    "coo_anima": "sakura",
    "essential_animas": ["sakura"],
}


# ── Governor state ───────────────────────────────────────────────────────────


_RELOGIN_MAX_PER_HOUR = 3
_RELOGIN_COOLDOWN_SECONDS = 600  # 10 min after consecutive failures


class GovernorState:
    """Mutable runtime state persisted to ``usage_governor_state.json``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.suspended_animas: list[str] = []
        self.reason: str = ""
        self.since: str = ""
        self.last_check: float = 0.0
        self.last_usage: dict[str, Any] = {}
        self._relogin_timestamps: dict[str, list[float]] = {}
        self._relogin_cooldown_until: dict[str, float] = {}
        self._load()

    def can_relogin(self, provider: str) -> bool:
        """Return True if a relogin attempt is allowed for *provider*."""
        now = time.time()
        if self._relogin_cooldown_until.get(provider, 0) > now:
            return False
        recent = [t for t in self._relogin_timestamps.get(provider, []) if now - t < 3600]
        self._relogin_timestamps[provider] = recent
        return len(recent) < _RELOGIN_MAX_PER_HOUR

    def record_relogin(self, provider: str, *, success: bool) -> None:
        """Record a relogin attempt.  On failure, activate cooldown."""
        now = time.time()
        self._relogin_timestamps.setdefault(provider, []).append(now)
        if not success:
            self._relogin_cooldown_until[provider] = now + _RELOGIN_COOLDOWN_SECONDS

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
            self.suspended_animas = data.get("suspended_animas", [])
            self.reason = data.get("reason", "")
            self.since = data.get("since", "")
        except Exception:
            logger.debug("Failed to load governor state", exc_info=True)

    def save(self) -> None:
        data = {
            "suspended_animas": self.suspended_animas,
            "reason": self.reason,
            "since": self.since,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to save governor state", exc_info=True)

    @property
    def is_governing(self) -> bool:
        return bool(self.suspended_animas) or bool(self.reason)


# ── Policy I/O ───────────────────────────────────────────────────────────────


def _policy_path(data_dir: Path) -> Path:
    return data_dir / "usage_policy.json"


def load_policy(data_dir: Path) -> dict[str, Any]:
    path = _policy_path(data_dir)
    if path.is_file():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            logger.warning("Failed to load usage policy, using defaults")
    return dict(DEFAULT_POLICY)


def save_policy(data_dir: Path, policy: dict[str, Any]) -> None:
    path = _policy_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_policy_file(data_dir: Path) -> None:
    """Create default policy file if it doesn't exist."""
    path = _policy_path(data_dir)
    if not path.is_file():
        save_policy(data_dir, DEFAULT_POLICY)
        logger.info("Created default usage policy at %s", path)


# ── Credential resolution ───────────────────────────────────────────────────


def _read_anima_credential(animas_dir: Path, name: str) -> str:
    """Read the ``credential`` field from an anima's status.json."""
    status = animas_dir / name / "status.json"
    if not status.is_file():
        return ""
    try:
        data = json.loads(status.read_text("utf-8"))
        return data.get("credential", "")
    except Exception:
        return ""


def _classify_animas(
    animas_dir: Path,
    anima_names: list[str],
) -> dict[str, list[str]]:
    """Group anima names by their policy provider key.

    Returns e.g. ``{"claude": ["alice"], "openai": ["bob"], "local": ["eve"]}``.
    """
    groups: dict[str, list[str]] = {}
    for name in anima_names:
        cred = _read_anima_credential(animas_dir, name)
        provider = _CREDENTIAL_TO_PROVIDER.get(cred, "local")
        groups.setdefault(provider, []).append(name)
    return groups


# ── Timestamp helpers ────────────────────────────────────────────────────────


def _parse_resets_at(value: Any) -> float | None:
    """Convert ``resets_at`` (ISO string or unix seconds/ms) → epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # unix seconds (< 1e12) or ms
        return float(value) if value < 1e12 else float(value) / 1000.0
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except (ValueError, TypeError):
            return None
    return None


def _time_remaining_pct(resets_at_ts: float, window_seconds: int) -> float:
    """Calculate what % of the window period remains.

    Returns 0–100.  Clamps to [0, 100].
    """
    remaining_secs = resets_at_ts - time.time()
    if remaining_secs <= 0:
        return 0.0
    pct = (remaining_secs / window_seconds) * 100.0
    return min(pct, 100.0)


# ── Rule evaluation ──────────────────────────────────────────────────────────


def _evaluate_threshold(
    remaining: float,
    thresholds: list[dict[str, Any]],
    provider_key: str,
    window_key: str,
) -> tuple[int | None, str]:
    """Evaluate fixed-threshold rules.  Returns (activity_level, reason)."""
    sorted_thresholds = sorted(thresholds, key=lambda t: t["remaining_below"])
    for rule in sorted_thresholds:
        if remaining < rule["remaining_below"]:
            level = rule["activity_level"]
            reason = (
                f"{provider_key}.{window_key} remaining {remaining:.0f}% "
                f"< {rule['remaining_below']}% → activity {level}%"
            )
            return level, reason
    return None, ""


def _evaluate_time_proportional(
    remaining: float,
    window_data: dict[str, Any],
    config: dict[str, Any],
    provider_key: str,
    window_key: str,
) -> tuple[int | None, str]:
    """Evaluate time-proportional rules.  Returns (activity_level, reason).

    Compares ``usage_remaining_%`` vs ``time_remaining_%`` of the window.
    When usage remaining is below time remaining, the deficit (in percentage
    points) is compared against ``deficit_rules`` to determine the throttle.
    """
    resets_at = window_data.get("resets_at")
    window_seconds = window_data.get("window_seconds")

    resets_ts = _parse_resets_at(resets_at)
    if resets_ts is None or not window_seconds:
        # Cannot calculate — fall back to no-op
        return None, ""

    time_pct = _time_remaining_pct(resets_ts, window_seconds)
    deficit = time_pct - remaining  # positive means over-consuming

    if deficit <= 0:
        # Usage pace is sustainable — no throttle needed
        return None, ""

    # Match deficit against rules (sorted descending so highest matches first)
    deficit_rules = config.get("deficit_rules", [])
    sorted_rules = sorted(deficit_rules, key=lambda r: r["deficit_above"], reverse=True)
    for rule in sorted_rules:
        if deficit >= rule["deficit_above"]:
            level = rule["activity_level"]
            reason = (
                f"{provider_key}.{window_key} remaining {remaining:.0f}% "
                f"< time {time_pct:.0f}% (deficit {deficit:.0f}pt) → activity {level}%"
            )
            return level, reason

    return None, ""


def _evaluate_hard_floor(
    remaining: float,
    hard_floor: float,
    provider_key: str,
    window_key: str,
) -> tuple[int | None, str]:
    """Absolute safety net: if remaining drops below hard floor, emergency throttle."""
    if remaining < hard_floor:
        reason = f"{provider_key}.{window_key} remaining {remaining:.0f}% < hard floor {hard_floor:.0f}% → activity 10%"
        return 10, reason
    return None, ""


def _evaluate_provider_remaining(
    usage_data: dict[str, Any],
    policy: dict[str, Any],
    provider_key: str,
) -> tuple[float, int | None, str]:
    """Evaluate rules for a single provider.

    Handles both threshold (list) and time_proportional (dict) window configs.

    Returns (worst_remaining_pct, target_activity_level_or_None, reason).
    """
    providers_rules = policy.get("providers", {})
    windows_rules = providers_rules.get(provider_key, {})
    provider_data = usage_data.get(provider_key, {})
    hard_floor = policy.get("hard_floor_pct", 15)

    if provider_data.get("error"):
        return 100.0, None, ""

    worst_remaining: float = 100.0
    worst_level: int | None = None
    worst_reason = ""

    def _update_worst(level: int | None, reason: str) -> None:
        nonlocal worst_level, worst_reason
        if level is not None and (worst_level is None or level < worst_level):
            worst_level = level
            worst_reason = reason

    for window_key, window_config in windows_rules.items():
        window = provider_data.get(window_key)
        if not window or not isinstance(window, dict):
            continue

        remaining = window.get("remaining")
        if remaining is None:
            continue

        worst_remaining = min(worst_remaining, remaining)

        # Determine mode
        if isinstance(window_config, list):
            # Threshold mode (backward-compatible list format)
            level, reason = _evaluate_threshold(
                remaining,
                window_config,
                provider_key,
                window_key,
            )
            _update_worst(level, reason)
        elif isinstance(window_config, dict):
            mode = window_config.get("mode", "threshold")
            if mode == "time_proportional":
                level, reason = _evaluate_time_proportional(
                    remaining,
                    window,
                    window_config,
                    provider_key,
                    window_key,
                )
                _update_worst(level, reason)
            else:
                # Dict with "rules" key treated as threshold
                rules = window_config.get("rules", [])
                level, reason = _evaluate_threshold(
                    remaining,
                    rules,
                    provider_key,
                    window_key,
                )
                _update_worst(level, reason)

        # Hard floor — always checked regardless of mode
        level, reason = _evaluate_hard_floor(
            remaining,
            hard_floor,
            provider_key,
            window_key,
        )
        _update_worst(level, reason)

    return worst_remaining, worst_level, worst_reason


def _animas_to_suspend(
    worst_remaining: float,
    policy: dict[str, Any],
    provider_animas: list[str],
) -> list[str]:
    """Determine which animas of a given provider should be suspended."""
    thresholds = policy.get("suspend_thresholds", {})
    essential = set(policy.get("essential_animas", []))
    coo = policy.get("coo_anima", "sakura")

    all_except_coo_below = thresholds.get("all_except_coo_below", 15)
    non_essential_below = thresholds.get("non_essential_below", 30)

    to_suspend: list[str] = []
    if worst_remaining < all_except_coo_below:
        for name in provider_animas:
            if name != coo:
                to_suspend.append(name)
    elif worst_remaining < non_essential_below:
        for name in provider_animas:
            if name not in essential:
                to_suspend.append(name)

    return to_suspend


def _provider_usage_fetch_failed(usage_data: dict[str, Any], provider_key: str) -> bool:
    """Return True when provider usage could not be fetched this cycle."""
    provider_data = usage_data.get(provider_key, {})
    return bool(provider_data.get("error"))


# ── Governor loop ────────────────────────────────────────────────────────────


class UsageGovernor:
    """Background task that monitors usage and suspends cloud-provider animas."""

    def __init__(self, app: Any, data_dir: Path, animas_dir: Path) -> None:
        self._app = app
        self._data_dir = data_dir
        self._animas_dir = animas_dir
        self._state = GovernorState(data_dir / "usage_governor_state.json")
        self._task: asyncio.Task | None = None

    @property
    def state(self) -> GovernorState:
        return self._state

    async def start(self) -> None:
        ensure_policy_file(self._data_dir)
        if self._state.suspended_animas:
            logger.info(
                "Governor restoring state: %d animas suspended",
                len(self._state.suspended_animas),
            )
        self._task = asyncio.create_task(self._loop())
        logger.info("Usage Governor started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Usage Governor stopped")

    async def _loop(self) -> None:
        await asyncio.sleep(10)

        while True:
            try:
                policy = load_policy(self._data_dir)
                if not policy.get("enabled", True):
                    await asyncio.sleep(60)
                    continue

                interval = policy.get("check_interval_seconds", 120)
                await self._tick(policy)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Usage Governor tick failed")
                await asyncio.sleep(60)

    async def _tick(self, policy: dict[str, Any]) -> None:
        """Single governor check cycle."""
        from server.routes.usage_routes import (
            _fetch_claude_usage,
            _fetch_nanogpt_usage,
            _fetch_openai_usage,
            _relogin_claude,
        )

        usage_data = {
            "claude": _fetch_claude_usage(),
            "openai": _fetch_openai_usage(),
            "nanogpt": _fetch_nanogpt_usage(),
        }

        # Auto-recovery: if Claude fetch failed with recoverable error,
        # try token refresh / relogin then retry (rate-limited to avoid
        # excessive OAuth requests).
        claude_error = usage_data.get("claude", {}).get("error", "")
        if claude_error in ("rate_limited", "unauthorized", "no_credentials"):
            if self._state.can_relogin("claude"):
                logger.info("Governor: Claude usage fetch failed (%s), attempting relogin", claude_error)
                relogin_result, _status = _relogin_claude()
                success = bool(relogin_result.get("success"))
                self._state.record_relogin("claude", success=success)
                if success:
                    logger.info("Governor: relogin succeeded, retrying usage fetch")
                    usage_data["claude"] = _fetch_claude_usage(skip_cache=True)
                else:
                    logger.warning(
                        "Governor: relogin failed — %s",
                        relogin_result.get("message", "unknown"),
                    )
            else:
                logger.info("Governor: skipping relogin for claude (rate-limited / cooldown)")

        self._state.last_check = time.time()
        self._state.last_usage = usage_data

        # Classify running animas by provider credential
        all_names = self._get_all_anima_names()
        groups = _classify_animas(self._animas_dir, all_names)
        currently_suspended = set(self._state.suspended_animas)

        all_suspend: list[str] = []
        reasons: list[str] = []

        for provider_key in ("claude", "openai", "nanogpt"):
            provider_animas = groups.get(provider_key, [])
            if not provider_animas:
                continue  # No animas using this provider — skip

            if _provider_usage_fetch_failed(usage_data, provider_key):
                retained = sorted(currently_suspended.intersection(provider_animas))
                all_suspend.extend(retained)
                error_code = usage_data.get(provider_key, {}).get("error", "unknown")
                if retained:
                    reasons.append(
                        f"{provider_key} usage unavailable ({error_code}) → keeping {', '.join(retained)} suspended",
                    )
                else:
                    reasons.append(f"{provider_key} usage unavailable ({error_code})")
                continue

            remaining, _level, reason = _evaluate_provider_remaining(
                usage_data,
                policy,
                provider_key,
            )

            to_suspend = _animas_to_suspend(remaining, policy, provider_animas)
            all_suspend.extend(to_suspend)
            if reason:
                reasons.append(reason)

        # Bail out before mutating process state if the governor is shutting down.
        task = asyncio.current_task()
        if task is not None and task.cancelled():
            return

        # Apply suspensions (only cloud-provider animas; local ones untouched)
        await self._apply_suspensions(set(all_suspend))

        self._state.reason = " | ".join(reasons) if reasons else ""
        if all_suspend and not self._state.since:
            self._state.since = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif not all_suspend:
            self._state.since = ""

        self._state.save()

    def _get_all_anima_names(self) -> list[str]:
        """Get all registered anima names (running + governor-suspended)."""
        supervisor = getattr(self._app.state, "supervisor", None)
        names = set()
        if supervisor and hasattr(supervisor, "processes"):
            names.update(supervisor.processes.keys())
        # Include animas we suspended (they won't be in processes)
        names.update(self._state.suspended_animas)
        return sorted(names)

    async def _apply_suspensions(self, target_suspended: set[str]) -> None:
        supervisor = getattr(self._app.state, "supervisor", None)
        if not supervisor:
            return

        currently_suspended = set(self._state.suspended_animas)

        # Resume animas that are no longer in the suspend list
        to_resume = currently_suspended - target_suspended
        for name in to_resume:
            try:
                if name not in supervisor.processes:
                    await supervisor.start_anima(name)
                    logger.info("Governor: resumed anima %s", name)
            except Exception:
                logger.warning("Governor: failed to resume %s", name, exc_info=True)

        # Suspend animas that should be stopped.
        newly_suspended = target_suspended - currently_suspended
        for name in target_suspended:
            try:
                if name in supervisor.processes:
                    await supervisor.stop_anima(name)
                    logger.info("Governor: suspended anima %s", name)
                    if name in newly_suspended:
                        await self._notify_supervisor(name, self._state.reason)
            except Exception:
                logger.warning("Governor: failed to suspend %s", name, exc_info=True)

        self._state.suspended_animas = sorted(target_suspended)

    async def _notify_supervisor(self, anima_name: str, reason: str) -> None:
        """Notify the suspended anima's supervisor (or human if top-level)."""
        try:
            import json as _json

            status_path = self._animas_dir / anima_name / "status.json"
            if not status_path.is_file():
                return
            status = _json.loads(status_path.read_text("utf-8"))
            supervisor_name = status.get("supervisor")

            if supervisor_name:
                sup_status_path = self._animas_dir / supervisor_name / "status.json"
                sup_enabled = False
                if sup_status_path.is_file():
                    sup_data = _json.loads(sup_status_path.read_text("utf-8"))
                    sup_enabled = sup_data.get("enabled", False)

                if sup_enabled:
                    from core.messenger import Messenger
                    from core.paths import get_shared_dir

                    messenger = Messenger(get_shared_dir(), "system")
                    messenger.send(
                        to=supervisor_name,
                        content=t(
                            "governor.supervisor_notify",
                            anima=anima_name,
                            reason=reason,
                        ),
                        intent="report",
                    )
                    logger.info("Governor: notified supervisor %s about %s suspension", supervisor_name, anima_name)
                    return

            from core.config.models import load_config as _lc_hn
            from core.notification.notifier import HumanNotifier

            cfg = _lc_hn()
            notifier = HumanNotifier.from_config(cfg.human_notification)
            if notifier.channel_count > 0:
                msg = t("governor.human_notify", anima=anima_name, reason=reason)
                await notifier.notify(
                    subject=t("governor.human_notify_subject"),
                    body=msg,
                    anima_name=anima_name,
                )
            logger.info("Governor: notified human about %s suspension (no active supervisor)", anima_name)
        except Exception:
            logger.warning("Failed to notify supervisor for %s", anima_name, exc_info=True)
