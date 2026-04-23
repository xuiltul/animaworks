from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""SubordinateControlMixin — disable, enable, model, restart, ping, read_state."""

import json as _json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t
from core.tooling.handler_base import _error_result
from core.tooling.org_helpers import OrgHelpersMixin, resolve_anima_name

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger

logger = logging.getLogger("animaworks.tool_handler")


class SubordinateControlMixin(OrgHelpersMixin):
    """Mixin for subordinate control tools: disable, enable, model, restart, ping, read_state."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str
    _activity: ActivityLogger
    _process_supervisor: Any

    def _handle_disable_subordinate(self, args: dict[str, Any]) -> str:
        """Disable a subordinate anima (set enabled=false in status.json)."""
        target_name = resolve_anima_name(args.get("name", ""))
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        from core.paths import get_animas_dir

        target_dir = get_animas_dir() / target_name
        status_file = target_dir / "status.json"

        existing: dict[str, Any] = {}
        if status_file.exists():
            try:
                existing = _json.loads(status_file.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError):
                pass

        if not existing.get("enabled", True):
            return t("handler.already_disabled", target_name=target_name)

        existing["enabled"] = False
        status_file.write_text(
            _json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        log_summary = t("handler.disable_log_summary", target_name=target_name)
        if reason:
            log_summary += t("handler.disable_reason", reason=reason)
        self._activity.log(
            "tool_use",
            tool="disable_subordinate",
            summary=log_summary,
            meta={"target": target_name, "reason": reason},
        )

        logger.info(
            "disable_subordinate: %s disabled %s (reason=%s)",
            self._anima_name,
            target_name,
            reason or "(none)",
        )

        result = t("handler.disabled_success", target_name=target_name)
        if reason:
            result += "\n" + t("handler.reason_prefix", reason=reason)
        return result

    def _handle_enable_subordinate(self, args: dict[str, Any]) -> str:
        """Enable a subordinate anima (set enabled=true in status.json)."""
        target_name = resolve_anima_name(args.get("name", ""))

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        from core.paths import get_animas_dir

        target_dir = get_animas_dir() / target_name
        status_file = target_dir / "status.json"

        existing: dict[str, Any] = {}
        if status_file.exists():
            try:
                existing = _json.loads(status_file.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError):
                pass

        if existing.get("enabled", True):
            return t("handler.already_enabled", target_name=target_name)

        existing["enabled"] = True
        status_file.write_text(
            _json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        self._activity.log(
            "tool_use",
            tool="enable_subordinate",
            summary=t("handler.enable_log_summary", target_name=target_name),
            meta={"target": target_name},
        )

        logger.info(
            "enable_subordinate: %s enabled %s",
            self._anima_name,
            target_name,
        )

        return t("handler.enabled_success", target_name=target_name)

    def _handle_set_subordinate_model(self, args: dict[str, Any]) -> str:
        """Change a subordinate anima's LLM model (updates status.json)."""
        from core.config.model_config import smart_update_model
        from core.config.models import KNOWN_MODELS
        from core.paths import get_data_dir

        target_name = resolve_anima_name(args.get("name", ""))
        model = args.get("model", "").strip()
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")
        if not model:
            return _error_result("InvalidArguments", "model is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        known_names = {m["name"] for m in KNOWN_MODELS}
        warn_msg = ""
        if model not in known_names:
            logger.warning(
                "set_subordinate_model: unknown model '%s' for '%s'. Not in KNOWN_MODELS — proceeding anyway.",
                model,
                target_name,
            )
            warn_msg = "\n" + t("handler.model_warning", model=model)

        target_dir = get_data_dir() / "animas" / target_name
        update_result = smart_update_model(target_dir, model=model)

        log_summary = t("handler.model_change_log", target_name=target_name, model=model)
        if reason:
            log_summary += t("handler.disable_reason", reason=reason)
        self._activity.log(
            "tool_use",
            tool="set_subordinate_model",
            summary=log_summary,
            meta={"target": target_name, "model": model, "reason": reason},
        )

        logger.info(
            "set_subordinate_model: %s changed %s model to %s (reason=%s)",
            self._anima_name,
            target_name,
            model,
            reason or "(none)",
        )

        result_msg = t("handler.model_changed", target_name=target_name, model=model)
        if update_result.get("family_changed"):
            result_msg += f"\n  credential: {update_result['credential']}, mode: {update_result['execution_mode']}"
        if reason:
            result_msg += "\n" + t("handler.reason_prefix", reason=reason)
        return result_msg + warn_msg

    def _handle_set_subordinate_background_model(self, args: dict[str, Any]) -> str:
        """Change a subordinate's background model (heartbeat/cron)."""
        from core.config.models import update_status_model
        from core.paths import get_data_dir

        target_name = args.get("name", "")
        model = args.get("model", "")
        credential = args.get("credential")
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        target_dir = get_data_dir() / "animas" / target_name
        update_status_model(
            target_dir,
            background_model=model if model else "",
            background_credential=credential if credential else "",
        )

        log_summary = t(
            "handler.bg_model_change_log",
            target_name=target_name,
            model=model or t("handler.none_value"),
        )
        if reason:
            log_summary += t("handler.reason_prefix", reason=reason)
        self._activity.log(
            "tool_use",
            tool="set_subordinate_background_model",
            summary=log_summary,
            meta={"target": target_name, "model": model, "reason": reason},
        )

        logger.info(
            "set_subordinate_background_model: %s changed %s background_model to %s (reason=%s)",
            self._anima_name,
            target_name,
            model or "(clear)",
            reason or "(none)",
        )

        if model:
            return t("handler.bg_model_changed", target_name=target_name, model=model)
        return t("handler.bg_model_cleared", target_name=target_name)

    def _handle_restart_subordinate(self, args: dict[str, Any]) -> str:
        """Request restart of a subordinate anima via sentinel flag in status.json."""
        from core.paths import get_animas_dir

        target_name = args.get("name", "")
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        target_dir = get_animas_dir() / target_name
        status_file = target_dir / "status.json"

        existing: dict[str, Any] = {}
        if status_file.exists():
            try:
                existing = _json.loads(status_file.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError):
                pass

        existing["restart_requested"] = True
        status_file.write_text(
            _json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        log_summary = t("handler.restart_log", target_name=target_name)
        if reason:
            log_summary += t("handler.disable_reason", reason=reason)
        self._activity.log(
            "tool_use",
            tool="restart_subordinate",
            summary=log_summary,
            meta={"target": target_name, "reason": reason},
        )

        logger.info(
            "restart_subordinate: %s requested restart of %s (reason=%s)",
            self._anima_name,
            target_name,
            reason or "(none)",
        )

        result = t("handler.restart_success", target_name=target_name)
        if reason:
            result += "\n" + t("handler.reason_prefix", reason=reason)
        return result

    def _handle_ping_subordinate(self, args: dict[str, Any]) -> str:
        """Ping subordinate(s) for liveness check."""
        from datetime import datetime

        from core.time_utils import ensure_aware, now_local

        target_name = args.get("name")

        if target_name:
            err = self._check_descendant(target_name)
            if err:
                return err
            targets = [target_name]
        else:
            targets = self._get_all_descendants()
            if not targets:
                return t("handler.no_subordinates")

        from core.paths import get_animas_dir

        animas_dir = get_animas_dir()
        results: list[dict[str, Any]] = []

        for name in targets:
            desc_dir = animas_dir / name
            result: dict[str, Any] = {
                "name": name,
                "alive": False,
                "process_status": "unknown",
                "last_activity": t("handler.last_activity_unknown"),
                "since": "",
            }

            if self._process_supervisor:
                try:
                    ps = self._process_supervisor.get_process_status(name)
                    if isinstance(ps, dict):
                        result["process_status"] = ps.get("status", "unknown")
                        result["alive"] = ps.get("status") == "running"
                    else:
                        result["process_status"] = str(ps)
                        result["alive"] = "running" in str(ps).lower()
                except Exception:
                    result["process_status"] = "not_found"
            else:
                from core.paths import get_data_dir

                sock = get_data_dir() / "run" / "sockets" / f"{name}.sock"
                if sock.exists():
                    result["alive"] = True
                    result["process_status"] = "running (socket exists)"
                else:
                    status_file = desc_dir / "status.json"
                    if status_file.exists():
                        try:
                            sdata = _json.loads(status_file.read_text(encoding="utf-8"))
                            result["process_status"] = "enabled" if sdata.get("enabled", True) else "disabled"
                        except Exception:
                            logger.debug("Failed to read status.json for %s", name, exc_info=True)

            try:
                recent = self._read_recent_activity(desc_dir, limit=1)
                if recent:
                    result["last_activity"] = recent[-1].ts
                    ts = ensure_aware(datetime.fromisoformat(recent[-1].ts))
                    elapsed = (now_local() - ts).total_seconds()
                    minutes = int(elapsed / 60)
                    if minutes < 60:
                        result["since"] = t("handler.since_minutes", minutes=minutes)
                    else:
                        hours = minutes // 60
                        result["since"] = t("handler.since_hours", hours=hours, minutes=minutes % 60)
                else:
                    result["last_activity"] = t("handler.last_activity_none")
            except Exception:
                logger.debug("Failed to read activity for %s", name, exc_info=True)

            results.append(result)

        self._activity.log(
            "tool_use",
            tool="ping_subordinate",
            summary=t("handler.ping_summary", target=t("handler.all_descendants") if not target_name else target_name),  # noqa: SIM212
        )

        return _json.dumps(results, ensure_ascii=False, indent=2)

    def _handle_read_subordinate_state(self, args: dict[str, Any]) -> str:
        """Read a descendant's current task state."""
        target_name = args.get("name", "")
        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        from core.paths import get_animas_dir

        desc_dir = get_animas_dir() / target_name

        parts: list[str] = [t("handler.state_title", target_name=target_name), ""]

        task_file = desc_dir / "state" / "current_state.md"
        if task_file.exists():
            try:
                content = task_file.read_text(encoding="utf-8").strip()
                parts.append(t("handler.state_current_state"))
                parts.append(content if content else t("handler.state_none"))
            except Exception:
                parts.append(t("handler.state_current_state"))
                parts.append(t("handler.state_unreadable"))
        else:
            parts.append(t("handler.state_current_state"))
            parts.append(t("handler.state_none"))

        parts.append("")

        self._activity.log(
            "tool_use",
            tool="read_subordinate_state",
            summary=t("handler.state_read_summary", target_name=target_name),
        )

        return "\n".join(parts)
