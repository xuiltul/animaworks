from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""OrgToolsMixin — anima creation, supervisor tools, org dashboard, delegation."""

import json as _json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t

from core.tooling.handler_base import _error_result, build_outgoing_origin_chain

if TYPE_CHECKING:
    from core.memory.activity import ActivityLogger
    from core.messenger import Messenger

logger = logging.getLogger("animaworks.tool_handler")


class OrgToolsMixin:
    """Organization management: anima creation, supervisor tools, dashboard, delegation."""

    # Declared for type-checker visibility
    _anima_dir: Path
    _anima_name: str
    _activity: ActivityLogger
    _process_supervisor: Any
    _messenger: Messenger | None
    _session_origin: str
    _session_origin_chain: list[str]

    # ── Anima creation ────────────────────────────────────────

    def _handle_create_anima(self, args: dict[str, Any]) -> str:
        """Create a new anima from a character sheet via anima_factory."""
        from core.anima_factory import create_from_md
        from core.paths import get_data_dir, get_animas_dir

        content = args.get("character_sheet_content")
        sheet_path_raw = args.get("character_sheet_path")
        name = args.get("name")
        explicit_supervisor = args.get("supervisor")

        if content:
            md_path = None
        elif sheet_path_raw:
            md_path = Path(sheet_path_raw).expanduser()
            if not md_path.is_absolute():
                md_path = (self._anima_dir / md_path).resolve()
                if not md_path.is_relative_to(self._anima_dir.resolve()):
                    return _error_result(
                        "PermissionDenied",
                        "character_sheet_path must be within anima directory.",
                    )
            else:
                # Absolute paths are intentionally allowed without directory
                # restriction — the CLI and human operators specify full paths.
                # create_from_md validates the content as a character sheet,
                # so passing an arbitrary file (e.g. /etc/passwd) will fail
                # schema validation rather than leaking data.
                md_path = md_path.resolve()
            if not md_path.exists():
                return _error_result(
                    "FileNotFound",
                    f"Character sheet not found: {md_path}",
                    suggestion=(
                        "Use character_sheet_content to pass content directly, "
                        "or ensure the file exists"
                    ),
                )
        else:
            return _error_result(
                "MissingParameter",
                "Either character_sheet_content or character_sheet_path is required",
            )

        try:
            anima_dir = create_from_md(
                get_animas_dir(),
                md_path,
                name=name,
                content=content,
                supervisor=explicit_supervisor,
            )
        except FileExistsError as e:
            return _error_result(
                "AnimaExists",
                str(e),
                suggestion="Choose a different name",
            )
        except ValueError as e:
            return _error_result("InvalidCharacterSheet", str(e))

        status_path = anima_dir / "status.json"
        if status_path.exists() and self._anima_name:
            try:
                status_data = _json.loads(status_path.read_text(encoding="utf-8"))
                if not status_data.get("supervisor"):
                    status_data["supervisor"] = self._anima_name
                    status_path.write_text(
                        _json.dumps(status_data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    logger.debug(
                        "Set fallback supervisor '%s' for '%s'",
                        self._anima_name,
                        anima_dir.name,
                    )
            except (OSError, _json.JSONDecodeError):
                logger.warning("Failed to set fallback supervisor", exc_info=True)

        try:
            from cli.commands.init_cmd import _register_anima_in_config
            _register_anima_in_config(get_data_dir(), anima_dir.name)
        except Exception:
            logger.warning("Failed to register anima in config.json", exc_info=True)

        logger.info("create_anima: created '%s' at %s", anima_dir.name, anima_dir)
        return f"Anima '{anima_dir.name}' created successfully at {anima_dir}. Reload the server to activate."

    # ── Supervisor helpers ────────────────────────────────────

    def _check_subordinate(self, target_name: str) -> str | None:
        """Verify that *target_name* is a direct subordinate of this anima."""
        from core.config.models import load_config

        if target_name == self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.self_operation_denied"),
            )

        try:
            config = load_config()
        except Exception as e:
            return _error_result("ConfigError", t("handler.config_load_failed", e=e))

        target_cfg = config.animas.get(target_name)
        if target_cfg is None:
            return _error_result(
                "AnimaNotFound",
                t("handler.anima_not_found", target_name=target_name),
            )

        if target_cfg.supervisor != self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.not_direct_subordinate", target_name=target_name),
                context={"supervisor": target_cfg.supervisor or t("handler.none_value")},
            )

        return None

    def _get_all_descendants(self, root_name: str | None = None) -> list[str]:
        """Get all descendant Anima names recursively via supervisor chain."""
        from core.config.models import load_config

        config = load_config()
        root = root_name or self._anima_name
        descendants: list[str] = []
        visited: set[str] = {root}
        queue = [
            name for name, cfg in config.animas.items()
            if cfg.supervisor == root
        ]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            queue.extend(
                name for name, cfg in config.animas.items()
                if cfg.supervisor == current
            )
        return descendants

    @staticmethod
    def _read_recent_activity(anima_dir: Path, *, limit: int = 1) -> list:
        """Read recent activity entries from another anima's directory."""
        from core.memory.activity import ActivityLogger
        al = ActivityLogger(anima_dir)
        return al.recent(days=1, limit=limit)

    def _check_descendant(self, target_name: str) -> str | None:
        """Verify that target_name is a descendant (any depth) of this anima."""
        if target_name == self._anima_name:
            return _error_result(
                "PermissionDenied",
                t("handler.self_operation_denied"),
            )
        descendants = self._get_all_descendants()
        if target_name not in descendants:
            return _error_result(
                "PermissionDenied",
                t("handler.not_descendant", target_name=target_name),
            )
        return None

    # ── Supervisor tool handlers ──────────────────────────────

    def _handle_disable_subordinate(self, args: dict[str, Any]) -> str:
        """Disable a subordinate anima (set enabled=false in status.json)."""
        target_name = args.get("name", "")
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_subordinate(target_name)
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
            self._anima_name, target_name, reason or "(none)",
        )

        result = t("handler.disabled_success", target_name=target_name)
        if reason:
            result += "\n" + t("handler.reason_prefix", reason=reason)
        return result

    def _handle_enable_subordinate(self, args: dict[str, Any]) -> str:
        """Enable a subordinate anima (set enabled=true in status.json)."""
        target_name = args.get("name", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_subordinate(target_name)
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
            self._anima_name, target_name,
        )

        return t("handler.enabled_success", target_name=target_name)

    def _handle_set_subordinate_model(self, args: dict[str, Any]) -> str:
        """Change a subordinate anima's LLM model (updates status.json)."""
        from core.config.models import KNOWN_MODELS, update_status_model
        from core.paths import get_data_dir

        target_name = args.get("name", "")
        model = args.get("model", "").strip()
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")
        if not model:
            return _error_result("InvalidArguments", "model is required")

        err = self._check_subordinate(target_name)
        if err:
            return err

        known_names = {m["name"] for m in KNOWN_MODELS}
        warn_msg = ""
        if model not in known_names:
            logger.warning(
                "set_subordinate_model: unknown model '%s' for '%s'. "
                "Not in KNOWN_MODELS — proceeding anyway.",
                model, target_name,
            )
            warn_msg = "\n" + t("handler.model_warning", model=model)

        target_dir = get_data_dir() / "animas" / target_name
        update_status_model(target_dir, model=model)

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
            self._anima_name, target_name, model, reason or "(none)",
        )

        result = t("handler.model_changed", target_name=target_name, model=model)
        if reason:
            result += "\n" + t("handler.reason_prefix", reason=reason)
        return result + warn_msg

    def _handle_restart_subordinate(self, args: dict[str, Any]) -> str:
        """Request restart of a subordinate anima via sentinel flag in status.json."""
        from core.paths import get_animas_dir

        target_name = args.get("name", "")
        reason = args.get("reason", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_subordinate(target_name)
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
            self._anima_name, target_name, reason or "(none)",
        )

        result = t("handler.restart_success", target_name=target_name)
        if reason:
            result += "\n" + t("handler.reason_prefix", reason=reason)
        return result

    # ── Org dashboard ─────────────────────────────────────────

    def _handle_org_dashboard(self, args: dict[str, Any]) -> str:
        """Show organization dashboard with all descendants' status."""
        descendants = self._get_all_descendants()
        if not descendants:
            return t("handler.no_subordinates")

        from core.config.models import load_config
        from core.paths import get_animas_dir

        animas_dir = get_animas_dir()
        config = load_config()

        entries: list[dict[str, Any]] = []
        for name in descendants:
            desc_dir = animas_dir / name
            entry: dict[str, Any] = {"name": name, "supervisor": ""}

            cfg = config.animas.get(name)
            if cfg:
                entry["supervisor"] = cfg.supervisor or ""

            if self._process_supervisor:
                try:
                    ps = self._process_supervisor.get_process_status(name)
                    entry["process_status"] = ps.get("status", "unknown") if isinstance(ps, dict) else str(ps)
                except Exception:
                    entry["process_status"] = "unknown"
            else:
                status_file = desc_dir / "status.json"
                if status_file.exists():
                    try:
                        status_data = _json.loads(status_file.read_text(encoding="utf-8"))
                        entry["process_status"] = "enabled" if status_data.get("enabled", True) else "disabled"
                    except Exception:
                        entry["process_status"] = "unknown"
                else:
                    entry["process_status"] = "unknown"

            try:
                recent = self._read_recent_activity(desc_dir, limit=1)
                if recent:
                    entry["last_activity"] = recent[-1].ts
                else:
                    entry["last_activity"] = t("handler.last_activity_none")
            except Exception:
                entry["last_activity"] = t("handler.last_activity_unknown")

            task_file = desc_dir / "state" / "current_task.md"
            if task_file.exists():
                try:
                    task_text = task_file.read_text(encoding="utf-8").strip()
                    entry["current_task"] = task_text[:100] if task_text else t("handler.current_task_none")
                except Exception:
                    entry["current_task"] = t("handler.current_task_unreadable")
            else:
                entry["current_task"] = t("handler.current_task_none")

            try:
                from core.memory.task_queue import TaskQueueManager

                tqm = TaskQueueManager(desc_dir)
                active = tqm.get_all_active()
                entry["active_tasks"] = len(active)
            except Exception:
                entry["active_tasks"] = 0

            entries.append(entry)

        lines: list[str] = [t("handler.org_dashboard_title"), ""]
        by_supervisor: dict[str, list[dict[str, Any]]] = {}
        for e in entries:
            sup = e.get("supervisor", "")
            by_supervisor.setdefault(sup, []).append(e)

        def _render_tree(parent: str, indent: int = 0) -> None:
            children = by_supervisor.get(parent, [])
            for child in children:
                prefix = "  " * indent + "├─ " if indent > 0 else ""
                status_icon = "🟢" if child["process_status"] in ("running", "enabled") else "🔴" if child["process_status"] == "disabled" else "⚪"
                line = f"{prefix}{status_icon} **{child['name']}** [{child['process_status']}]"
                line += " | " + t("handler.dashboard_last", activity=child["last_activity"])
                line += " | " + t("handler.dashboard_tasks", count=child["active_tasks"])
                none_str = t("handler.current_task_none")
                if child["current_task"] != none_str:
                    line += "\n" + "  " * (indent + 1) + "└ " + t("handler.dashboard_working_on", task=child["current_task"])
                lines.append(line)
                _render_tree(child["name"], indent + 1)

        _render_tree(self._anima_name)

        rendered = set()
        for e in entries:
            rendered.add(e["name"])

        self._activity.log(
            "tool_use",
            tool="org_dashboard",
            summary=t("handler.dashboard_summary", count=len(descendants)),
        )

        return "\n".join(lines)

    # ── Ping ──────────────────────────────────────────────────

    def _handle_ping_subordinate(self, args: dict[str, Any]) -> str:
        """Ping subordinate(s) for liveness check."""
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
                    from core.time_utils import ensure_aware, now_jst

                    ts = ensure_aware(datetime.fromisoformat(recent[-1].ts))
                    elapsed = (now_jst() - ts).total_seconds()
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
            summary=t("handler.ping_summary", target=t("handler.all_descendants") if not target_name else target_name),
        )

        return _json.dumps(results, ensure_ascii=False, indent=2)

    # ── State reading ─────────────────────────────────────────

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

        task_file = desc_dir / "state" / "current_task.md"
        if task_file.exists():
            try:
                content = task_file.read_text(encoding="utf-8").strip()
                parts.append(t("handler.state_current_task"))
                parts.append(content if content else t("handler.state_none"))
            except Exception:
                parts.append(t("handler.state_current_task"))
                parts.append(t("handler.state_unreadable"))
        else:
            parts.append(t("handler.state_current_task"))
            parts.append(t("handler.state_none"))

        parts.append("")

        pending_file = desc_dir / "state" / "pending.md"
        if pending_file.exists():
            try:
                content = pending_file.read_text(encoding="utf-8").strip()
                parts.append(t("handler.state_pending"))
                parts.append(content if content else t("handler.state_none"))
            except Exception:
                parts.append(t("handler.state_pending"))
                parts.append(t("handler.state_unreadable"))
        else:
            parts.append(t("handler.state_pending"))
            parts.append(t("handler.state_none"))

        self._activity.log(
            "tool_use",
            tool="read_subordinate_state",
            summary=t("handler.state_read_summary", target_name=target_name),
        )

        return "\n".join(parts)

    # ── Delegation ────────────────────────────────────────────

    def _handle_delegate_task(self, args: dict[str, Any]) -> str:
        """Delegate a task to a direct subordinate."""
        target_name = args.get("name", "")
        instruction = args.get("instruction", "")
        summary = args.get("summary", "") or instruction[:100]
        deadline = args.get("deadline", "")

        if not target_name:
            return _error_result("InvalidArguments", "name is required")
        if not instruction:
            return _error_result("InvalidArguments", "instruction is required")
        if not deadline:
            return _error_result(
                "InvalidArguments",
                "deadline is required. Use relative format ('30m', '2h', '1d') or ISO8601.",
            )

        err = self._check_subordinate(target_name)
        if err:
            return err

        from core.memory.task_queue import TaskQueueManager
        from core.paths import get_animas_dir

        target_dir = get_animas_dir() / target_name

        sub_tqm = TaskQueueManager(target_dir)
        try:
            sub_entry = sub_tqm.add_task(
                source="anima",
                original_instruction=instruction,
                assignee=target_name,
                summary=summary,
                deadline=deadline,
                relay_chain=[self._anima_name],
            )
        except ValueError as e:
            return _error_result("InvalidArguments", str(e))

        # Build outgoing origin_chain (provenance Phase 3)
        outgoing_chain = build_outgoing_origin_chain(
            self._session_origin, self._session_origin_chain,
        )

        dm_result = ""
        if self._messenger:
            try:
                self._messenger.send(
                    to=target_name,
                    content=t(
                        "handler.delegation_dm_content",
                        instruction=instruction,
                        deadline=deadline,
                        task_id=sub_entry.task_id,
                    ),
                    intent="delegation",
                    origin_chain=outgoing_chain,
                )
                dm_result = t("handler.dm_sent")
            except Exception as e:
                dm_result = t("handler.dm_send_failed", e=e)
                logger.warning("delegate_task DM failed: %s -> %s: %s", self._anima_name, target_name, e)
        else:
            dm_result = t("handler.messenger_not_set")

        process_warning = ""
        try:
            from core.paths import get_data_dir
            sock = get_data_dir() / "run" / "sockets" / f"{target_name}.sock"
            if not sock.exists():
                status_file = target_dir / "status.json"
                if status_file.exists():
                    sdata = _json.loads(status_file.read_text(encoding="utf-8"))
                    if not sdata.get("enabled", True):
                        process_warning = t("handler.subordinate_disabled_warning", target_name=target_name)
        except Exception:
            logger.debug("Failed to check subordinate process status for %s", target_name, exc_info=True)

        own_tqm = TaskQueueManager(self._anima_dir)
        own_entry = own_tqm.add_delegated_task(
            original_instruction=instruction,
            assignee=target_name,
            summary=t("handler.delegation_summary", summary=summary),
            deadline=deadline,
            relay_chain=[self._anima_name, target_name],
            meta={
                "delegated_to": target_name,
                "delegated_task_id": sub_entry.task_id,
            },
        )

        self._activity.log(
            "tool_use",
            tool="delegate_task",
            summary=t("handler.delegate_log", target_name=target_name, summary=summary[:80]),
            meta={
                "target": target_name,
                "own_task_id": own_entry.task_id,
                "sub_task_id": sub_entry.task_id,
            },
        )

        result = t(
            "handler.delegated_success",
            target_name=target_name,
            sub_id=sub_entry.task_id,
            own_id=own_entry.task_id,
            dm_result=dm_result,
        )
        return result + process_warning

    # ── Audit ─────────────────────────────────────────────────

    def _handle_audit_subordinate(self, args: dict[str, Any]) -> str:
        """Generate a comprehensive audit report for a descendant Anima."""
        target_name = args.get("name", "")
        days = max(1, min(args.get("days", 1), 30))

        if not target_name:
            return _error_result("InvalidArguments", "name is required")

        err = self._check_descendant(target_name)
        if err:
            return err

        from collections import Counter

        from core.memory.activity import ActivityLogger
        from core.paths import get_animas_dir

        animas_dir = get_animas_dir()
        desc_dir = animas_dir / target_name

        lines: list[str] = [
            t("handler.audit_title", target_name=target_name),
            "",
            t("handler.audit_period", days=days),
        ]

        # ── Overview: process status & model ──
        status_file = desc_dir / "status.json"
        process_status = "unknown"
        model_name = "unknown"
        if status_file.exists():
            try:
                sdata = _json.loads(status_file.read_text(encoding="utf-8"))
                process_status = "enabled" if sdata.get("enabled", True) else "disabled"
                model_name = sdata.get("model", "unknown")
            except (_json.JSONDecodeError, OSError):
                pass

        if self._process_supervisor:
            try:
                ps = self._process_supervisor.get_process_status(target_name)
                if isinstance(ps, dict):
                    process_status = ps.get("status", process_status)
            except Exception:
                pass

        lines.append(t("handler.audit_process_status", status=process_status))
        lines.append(t("handler.audit_model", model=model_name))
        lines.append("")

        # ── Activity summary ──
        al = ActivityLogger(desc_dir)
        _AUDIT_ENTRY_LIMIT = 10_000
        entries = al.recent(days=days, limit=_AUDIT_ENTRY_LIMIT)

        lines.append(t("handler.audit_activity_header"))
        lines.append("")

        if entries:
            type_counts: Counter[str] = Counter()
            for e in entries:
                type_counts[e.type] += 1

            lines.append(t("handler.audit_total_events", count=len(entries)))
            for etype, count in type_counts.most_common():
                lines.append(t("handler.audit_event_type", event_type=etype, count=count))
        else:
            lines.append(t("handler.audit_no_activity"))

        lines.append("")

        # ── Task status ──
        lines.append(t("handler.audit_tasks_header"))
        lines.append("")

        task_file = desc_dir / "state" / "current_task.md"
        if task_file.exists():
            try:
                task_text = task_file.read_text(encoding="utf-8").strip()
                lines.append(t(
                    "handler.audit_current_task",
                    task=task_text[:150] if task_text else t("handler.state_none"),
                ))
            except Exception:
                lines.append(t("handler.audit_current_task", task=t("handler.state_unreadable")))
        else:
            lines.append(t("handler.audit_current_task", task=t("handler.state_none")))

        try:
            from core.memory.task_queue import TaskQueueManager

            tqm = TaskQueueManager(desc_dir)
            active = tqm.get_all_active()
            lines.append(t("handler.audit_active_tasks", count=len(active)))

            done_tasks = tqm.list_tasks(status="done")
            done_count = len(done_tasks)
            lines.append(t("handler.audit_completed_tasks", count=done_count))
        except Exception:
            lines.append(t("handler.audit_active_tasks", count=0))
            lines.append(t("handler.audit_completed_tasks", count=0))

        lines.append("")

        # ── Error summary ──
        lines.append(t("handler.audit_errors_header"))
        lines.append("")

        error_entries = [e for e in entries if e.type == "error"]
        lines.append(t("handler.audit_error_count", count=len(error_entries)))

        if error_entries:
            for e in error_entries[-5:]:
                summary = e.summary or e.content[:100]
                lines.append(t("handler.audit_error_entry", ts=e.ts[:16], summary=summary))
        else:
            lines.append(t("handler.audit_no_errors"))

        lines.append("")

        # ── Tool usage statistics ──
        lines.append(t("handler.audit_tools_header"))
        lines.append("")

        tool_entries = [e for e in entries if e.type == "tool_use" and e.tool]
        if tool_entries:
            tool_counts: Counter[str] = Counter()
            for e in tool_entries:
                tool_counts[e.tool] += 1
            for tool, count in tool_counts.most_common(10):
                lines.append(t("handler.audit_tool_entry", tool=tool, count=count))
        else:
            lines.append(t("handler.audit_no_tool_use"))

        lines.append("")

        # ── Communication patterns ──
        lines.append(t("handler.audit_comms_header"))
        lines.append("")

        sent = [e for e in entries if e.type in ("message_sent", "dm_sent")]
        received = [e for e in entries if e.type in ("message_received", "dm_received")]

        if sent or received:
            lines.append(t("handler.audit_msgs_sent", count=len(sent)))
            lines.append(t("handler.audit_msgs_received", count=len(received)))

            peer_sent: Counter[str] = Counter()
            peer_recv: Counter[str] = Counter()
            for e in sent:
                peer = e.to_person or "unknown"
                peer_sent[peer] += 1
            for e in received:
                peer = e.from_person or "unknown"
                peer_recv[peer] += 1

            all_peers = sorted(set(peer_sent) | set(peer_recv))
            for peer in all_peers:
                lines.append(t(
                    "handler.audit_comms_peer",
                    peer=peer, sent=peer_sent.get(peer, 0), received=peer_recv.get(peer, 0),
                ))
        else:
            lines.append(t("handler.audit_no_comms"))

        self._activity.log(
            "tool_use",
            tool="audit_subordinate",
            summary=t("handler.audit_log_summary", target_name=target_name, days=days),
            meta={"target": target_name, "days": days},
        )

        return "\n".join(lines)

    # ── Task tracking ────────────────────────────────────────

    def _handle_task_tracker(self, args: dict[str, Any]) -> str:
        """Track progress of delegated tasks."""
        status_filter = args.get("status", "active")

        from core.memory.task_queue import TaskQueueManager
        from core.paths import get_animas_dir

        own_tqm = TaskQueueManager(self._anima_dir)
        delegated = own_tqm.get_delegated_tasks()

        if not delegated:
            return t("handler.no_delegated_tasks")

        animas_dir = get_animas_dir()
        results: list[dict[str, Any]] = []

        for task in delegated:
            meta = task.meta or {}
            delegated_to = meta.get("delegated_to", "")
            delegated_task_id = meta.get("delegated_task_id", "")

            entry: dict[str, Any] = {
                "my_task_id": task.task_id,
                "delegated_to": delegated_to,
                "summary": task.summary,
                "delegated_at": task.ts,
                "deadline": task.deadline or "",
                "subordinate_status": "unknown",
                "last_updated": "",
            }

            if delegated_to and delegated_task_id:
                target_dir = animas_dir / delegated_to
                try:
                    sub_tqm = TaskQueueManager(target_dir)
                    sub_task = sub_tqm.get_task_by_id(delegated_task_id)
                    if sub_task:
                        entry["subordinate_status"] = sub_task.status
                        entry["last_updated"] = sub_task.updated_at
                except Exception:
                    entry["subordinate_status"] = "unknown"

            sub_status = entry["subordinate_status"]
            if status_filter == "active" and sub_status in ("done", "cancelled"):
                continue
            if status_filter == "completed" and sub_status not in ("done", "cancelled"):
                continue

            results.append(entry)

        self._activity.log(
            "tool_use",
            tool="task_tracker",
            summary=t("handler.task_tracker_log", status=status_filter, count=len(results)),
        )

        if not results:
            return t("handler.no_matching_delegated", status=status_filter)

        return _json.dumps(results, ensure_ascii=False, indent=2)
