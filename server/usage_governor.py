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
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("animaworks.usage_governor")

# Map anima credential values to policy provider keys
_CREDENTIAL_TO_PROVIDER: dict[str, str] = {
    "anthropic": "claude",
    "openai": "openai",
}

# ── Policy schema ────────────────────────────────────────────────────────────

DEFAULT_POLICY: dict[str, Any] = {
    "enabled": True,
    "check_interval_seconds": 120,
    "providers": {
        "claude": {
            "five_hour": [
                {"remaining_below": 50, "activity_level": 60},
                {"remaining_below": 30, "activity_level": 30},
                {"remaining_below": 15, "activity_level": 10},
            ],
            "seven_day": [
                {"remaining_below": 30, "activity_level": 50},
                {"remaining_below": 15, "activity_level": 10},
            ],
        },
        "openai": {
            "5h": [
                {"remaining_below": 50, "activity_level": 60},
                {"remaining_below": 30, "activity_level": 30},
                {"remaining_below": 15, "activity_level": 10},
            ],
            "Week": [
                {"remaining_below": 30, "activity_level": 50},
                {"remaining_below": 15, "activity_level": 10},
            ],
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


class GovernorState:
    """Mutable runtime state persisted to ``usage_governor_state.json``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.suspended_animas: list[str] = []
        self.reason: str = ""
        self.since: str = ""
        self.last_check: float = 0.0
        self.last_usage: dict[str, Any] = {}
        self._load()

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


# ── Rule evaluation ──────────────────────────────────────────────────────────


def _evaluate_provider_remaining(
    usage_data: dict[str, Any],
    policy: dict[str, Any],
    provider_key: str,
) -> tuple[float, int | None, str]:
    """Evaluate rules for a single provider.

    Returns (worst_remaining_pct, target_activity_level_or_None, reason).
    """
    providers_rules = policy.get("providers", {})
    windows_rules = providers_rules.get(provider_key, {})
    provider_data = usage_data.get(provider_key, {})

    if provider_data.get("error"):
        return 100.0, None, ""

    worst_remaining: float = 100.0
    worst_level: int | None = None
    worst_reason = ""

    for window_key, thresholds in windows_rules.items():
        window = provider_data.get(window_key)
        if not window or not isinstance(window, dict):
            continue

        remaining = window.get("remaining")
        if remaining is None:
            continue

        worst_remaining = min(worst_remaining, remaining)

        sorted_thresholds = sorted(thresholds, key=lambda t: t["remaining_below"])
        for rule in sorted_thresholds:
            if remaining < rule["remaining_below"]:
                level = rule["activity_level"]
                if worst_level is None or level < worst_level:
                    worst_level = level
                    worst_reason = (
                        f"{provider_key}.{window_key} remaining {remaining:.0f}% "
                        f"< {rule['remaining_below']}% → activity {level}%"
                    )
                break

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
        from server.routes.usage_routes import _fetch_claude_usage, _fetch_openai_usage

        usage_data = {
            "claude": _fetch_claude_usage(),
            "openai": _fetch_openai_usage(),
        }
        self._state.last_check = time.time()
        self._state.last_usage = usage_data

        # Classify running animas by provider credential
        all_names = self._get_all_anima_names()
        groups = _classify_animas(self._animas_dir, all_names)

        all_suspend: list[str] = []
        reasons: list[str] = []

        for provider_key in ("claude", "openai"):
            provider_animas = groups.get(provider_key, [])
            if not provider_animas:
                continue  # No animas using this provider — skip

            remaining, _level, reason = _evaluate_provider_remaining(
                usage_data, policy, provider_key,
            )

            to_suspend = _animas_to_suspend(remaining, policy, provider_animas)
            all_suspend.extend(to_suspend)
            if reason:
                reasons.append(reason)

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

        # Suspend new animas
        to_stop = target_suspended - currently_suspended
        for name in to_stop:
            try:
                if name in supervisor.processes:
                    await supervisor.stop_anima(name)
                    logger.info("Governor: suspended anima %s", name)
            except Exception:
                logger.warning("Governor: failed to suspend %s", name, exc_info=True)

        self._state.suspended_animas = sorted(target_suspended)
