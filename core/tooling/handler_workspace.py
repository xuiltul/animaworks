from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Workspace access grant handlers."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.execution._sanitize import ORIGIN_HUMAN
from core.tooling.handler_base import _error_result
from core.tooling.org_helpers import resolve_anima_name

logger = logging.getLogger("animaworks.tool_handler")

_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HUMAN_TRIGGER_PREFIX = "message:"
_HUMAN_TRIGGER_EXACT = {"chat", "manual", "greet:user"}
_SYSTEM_ROOTS = (
    Path("/etc"),
    Path("/proc"),
    Path("/dev"),
    Path("/sys"),
    Path("/run"),
    Path("/boot"),
    Path("/root"),
)


class WorkspaceToolsMixin:
    """Workspace registration and permission grant tools."""

    _anima_dir: Path
    _anima_name: str
    _session_origin: str
    _session_origin_chain: list[str]
    _trigger: str

    def _handle_grant_workspace_access(self, args: dict[str, Any]) -> str:
        alias = str(args.get("alias", "")).strip()
        raw_path = str(args.get("path", "")).strip()
        target_raw = str(args.get("target_anima", "") or "").strip()
        make_default = self._workspace_grant_bool(args.get("make_default", True))

        if not alias or not _ALIAS_RE.match(alias):
            return _error_result(
                "InvalidArguments",
                "alias must be 1-64 characters and contain only letters, numbers, dot, underscore, or dash.",
            )
        if not raw_path:
            return _error_result("InvalidArguments", "path is required")

        if not self._workspace_grant_has_human_origin():
            return _error_result(
                "PermissionDenied",
                "grant_workspace_access requires an explicit human-origin instruction.",
                context={"origin": self._session_origin, "trigger": self._trigger},
            )

        try:
            from core.config.models import load_config, save_config
            from core.paths import get_animas_dir
            from core.workspace import qualified_alias

            config = load_config()
        except Exception as exc:
            logger.warning("workspace grant config load failed: %s", exc)
            return _error_result("ConfigError", f"Failed to load global config: {exc}")

        caller_cfg = config.animas.get(self._anima_name)
        if caller_cfg is None or caller_cfg.supervisor is not None:
            return _error_result(
                "PermissionDenied",
                "Only top-level Animas can grant workspace access.",
                context={"caller": self._anima_name},
            )

        target_name = resolve_anima_name(target_raw) if target_raw else self._anima_name
        target_cfg = config.animas.get(target_name)
        if target_cfg is None:
            return _error_result("AnimaNotFound", f"Target Anima not found: {target_name}")

        if target_name != self._anima_name and not self._workspace_grant_is_descendant(config, target_name):
            return _error_result(
                "PermissionDenied",
                "Top-level Animas can grant workspace access only to themselves or descendants.",
                context={"caller": self._anima_name, "target_anima": target_name},
            )

        try:
            workspace_path = Path(raw_path).expanduser().resolve()
        except OSError as exc:
            return _error_result("InvalidArguments", f"Failed to resolve path: {exc}")

        safety_error = self._workspace_grant_path_error(workspace_path)
        if safety_error:
            return safety_error

        animas_dir = get_animas_dir()
        target_dir = (animas_dir / target_name).resolve()
        if not target_dir.is_dir():
            return _error_result("AnimaNotFound", f"Target Anima directory not found: {target_dir}")

        config.workspaces[alias] = str(workspace_path)
        try:
            save_config(config)
        except Exception as exc:
            logger.warning("workspace grant config save failed: %s", exc)
            return _error_result("ConfigError", f"Failed to save global config: {exc}")

        qualified = qualified_alias(alias, str(workspace_path))
        try:
            permissions_changed, permissions_unrestricted = self._workspace_grant_update_permissions(
                target_dir,
                workspace_path,
            )
            status_changed = False
            if make_default:
                status_changed = self._workspace_grant_update_status(target_dir, qualified)
        except OSError as exc:
            logger.warning("workspace grant target update failed: %s", exc)
            return _error_result("FileWriteError", f"Failed to update target Anima files: {exc}")

        return json.dumps(
            {
                "status": "ok",
                "qualified_alias": qualified,
                "alias": alias,
                "path": str(workspace_path),
                "target_anima": target_name,
                "global_workspace_registered": True,
                "permissions_changed": permissions_changed,
                "permissions_unrestricted": permissions_unrestricted,
                "default_workspace_changed": status_changed,
                "effective_next_codex_run": True,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _workspace_grant_has_human_origin(self) -> bool:
        if self._session_origin == ORIGIN_HUMAN:
            return True
        if ORIGIN_HUMAN in self._session_origin_chain:
            return True
        trigger = (self._trigger or "").strip()
        return trigger.startswith(_HUMAN_TRIGGER_PREFIX) or trigger in _HUMAN_TRIGGER_EXACT

    @staticmethod
    def _workspace_grant_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _workspace_grant_is_descendant(self, config: Any, target_name: str) -> bool:
        current = target_name
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            cfg = config.animas.get(current)
            if cfg is None:
                return False
            supervisor = cfg.supervisor
            if supervisor == self._anima_name:
                return True
            if not supervisor:
                return False
            current = supervisor
        return False

    def _workspace_grant_path_error(self, workspace_path: Path) -> str | None:
        if not workspace_path.is_dir():
            return _error_result("InvalidArguments", f"Workspace path is not an existing directory: {workspace_path}")
        if workspace_path == Path("/"):
            return _error_result("PermissionDenied", "grant_workspace_access cannot grant filesystem root access")

        try:
            home = Path.home().resolve()
            if workspace_path == home:
                return _error_result("PermissionDenied", "grant_workspace_access cannot grant the home directory root")
        except OSError:
            pass

        for root in _SYSTEM_ROOTS:
            try:
                root_resolved = root.resolve()
            except OSError:
                root_resolved = root
            if workspace_path == root_resolved or workspace_path.is_relative_to(root_resolved):
                return _error_result(
                    "PermissionDenied",
                    f"grant_workspace_access cannot grant protected system directory: {workspace_path}",
                )

        try:
            from core.paths import get_animas_dir

            animas_dir = get_animas_dir().resolve()
            if workspace_path == animas_dir or workspace_path.is_relative_to(animas_dir):
                return _error_result(
                    "PermissionDenied",
                    "grant_workspace_access cannot grant Anima home directories as workspaces.",
                )
        except OSError:
            pass
        return None

    def _workspace_grant_update_permissions(self, target_dir: Path, workspace_path: Path) -> tuple[bool, bool]:
        from core.config.models import load_permissions

        permissions = load_permissions(target_dir)
        if permissions.file_roots == ["/"]:
            return False, True

        for root in permissions.file_roots:
            try:
                allowed = Path(root).expanduser().resolve()
            except OSError:
                continue
            if workspace_path == allowed or workspace_path.is_relative_to(allowed):
                return False, False

        permissions.file_roots.append(str(workspace_path))
        payload = permissions.model_dump(mode="json")
        (target_dir / "permissions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return True, False

    def _workspace_grant_update_status(self, target_dir: Path, qualified: str) -> bool:
        status_path = target_dir / "status.json"
        data: dict[str, Any] = {}
        if status_path.is_file():
            try:
                parsed = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    data = parsed
            except (json.JSONDecodeError, OSError):
                data = {}
        if data.get("default_workspace") == qualified:
            return False
        data["default_workspace"] = qualified
        status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True
