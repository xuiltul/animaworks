from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""PermissionsMixin — file/command permission checks and check_permissions handler."""

import json as _json
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.config.models import PermissionsConfig, load_permissions
from core.i18n import t
from core.tooling.handler_base import (
    _error_result,
    _get_blocked_patterns,
    _get_injection_re,
    _is_global_permissions_write_blocked,
    _is_protected_write,
)

if TYPE_CHECKING:
    from core.memory import MemoryManager
    from core.tooling.dispatch import ExternalToolDispatcher

logger = logging.getLogger("animaworks.tool_handler")


class PermissionsMixin:
    """Permission checking for file access, command execution, and tool creation."""

    # Declared for type-checker visibility; actual values live on ToolHandler
    _memory: MemoryManager
    _anima_dir: Path
    _anima_name: str
    _superuser: bool
    _subordinate_activity_dirs: list[Path]
    _subordinate_management_files: list[Path]
    _subordinate_root_dirs: list[Path]
    _descendant_activity_dirs: list[Path]
    _descendant_state_files: list[Path]
    _descendant_state_dirs: list[Path]
    _peer_activity_dirs: list[Path]
    _dispatch: dict[str, Any]
    _external: ExternalToolDispatcher

    # ── check_permissions handler ────────────────────────────

    def _handle_check_permissions(self, args: dict[str, Any]) -> str:
        """Return a summary of what tools, external tools, and file access this anima has."""
        internal_tools = sorted(self._dispatch.keys())

        external_enabled: list[str] = []
        external_available: list[str] = []
        try:
            from core.tools import TOOL_MODULES

            all_categories = sorted(TOOL_MODULES.keys())
            for cat in all_categories:
                if cat in (self._external.registry if self._external else []):
                    external_enabled.append(cat)
                else:
                    external_available.append(cat)
        except Exception:
            logger.debug("Failed to enumerate external tools", exc_info=True)

        config = self._load_permissions_config()

        file_read: list[str] = [t("handler.file_read_own"), t("handler.file_read_shared")]
        file_write: list[str] = [t("handler.file_write_own")]
        if self._subordinate_management_files:
            file_read.append(t("handler.subordinate_management"))
            file_write.append(t("handler.subordinate_management"))
        if self._subordinate_root_dirs:
            file_read.append(t("handler.subordinate_dir_list"))
        if self._descendant_activity_dirs:
            file_read.append(t("handler.descendant_activity"))
        if self._descendant_state_files:
            file_read.append(t("handler.descendant_state"))
        if self._descendant_state_dirs:
            file_read.append(t("handler.descendant_pending"))
        if self._peer_activity_dirs:
            file_read.append(t("handler.peer_activity"))

        if config.file_roots and config.file_roots != ["/"]:
            for root in config.file_roots:
                if root.startswith("/"):
                    file_read.append(root)
                    file_write.append(root)

        restrictions: list[str] = []
        for cmd in config.commands.deny:
            restrictions.append(t("handler.cmd_denied", cmd=cmd))

        result = {
            "internal_tools": internal_tools,
            "external_tools": {
                "enabled": external_enabled,
                "available_but_not_enabled": external_available,
            },
            "file_access": {
                "read": file_read,
                "write": file_write,
            },
            "restrictions": restrictions,
        }

        return _json.dumps(result, ensure_ascii=False, indent=2)

    # ── Tool creation permission ─────────────────────────────

    def _check_tool_creation_permission(self, kind: str) -> bool:
        """Check if tool creation is permitted via permissions config."""
        if self._memory is None:
            return False
        config = self._load_permissions_config()
        kind_lower = kind.lower()
        if "personal" in kind_lower or "個人" in kind:
            return config.tool_creation.personal
        if "shared" in kind_lower or "共有" in kind:
            return config.tool_creation.shared
        return False

    # ── Permission helpers ───────────────────────────────────

    def _load_permissions_config(self) -> PermissionsConfig:
        """Load PermissionsConfig from permissions.json (with migration fallback)."""
        return load_permissions(self._anima_dir)

    def _check_file_permission(self, path: str, *, write: bool = False) -> str | None:
        """Check if the file path is allowed by permissions config.

        Returns ``None`` if allowed, or an error message string if denied.
        """
        if self._superuser:
            return None
        resolved = Path(path).resolve()

        if write:
            gp_err = _is_global_permissions_write_blocked(resolved)
            if gp_err:
                logger.warning(
                    "permission_denied anima=%s path=%s reason=global_permissions_protected", self._anima_name, path
                )
                return gp_err

        # Own anima_dir
        if resolved.is_relative_to(self._anima_dir.resolve()):
            if write:
                err = _is_protected_write(self._anima_dir, resolved)
                if err:
                    logger.warning("permission_denied anima=%s path=%s reason=protected_file", self._anima_name, path)
                    return err
            return None

        # Supervisor can read direct subordinate's activity_log (work records)
        if not write:
            for sub_activity in self._subordinate_activity_dirs:
                if resolved.is_relative_to(sub_activity):
                    return None

        # Supervisor can read any descendant's activity_log
        if not write:
            for desc_activity in self._descendant_activity_dirs:
                if resolved.is_relative_to(desc_activity):
                    return None

        # Peers (same supervisor) can read each other's activity_log
        if not write:
            for peer_activity in self._peer_activity_dirs:
                if resolved.is_relative_to(peer_activity):
                    return None

        # Supervisor can read any descendant's state files
        if not write:
            for desc_state in self._descendant_state_files:
                if resolved == desc_state:
                    return None

        # Supervisor can read any descendant's state/pending/ directory
        if not write:
            for desc_state_dir in self._descendant_state_dirs:
                if resolved.is_relative_to(desc_state_dir):
                    return None

        # Supervisor can read/write subordinate's management files
        for mgmt_file in self._subordinate_management_files:
            if resolved == mgmt_file:
                return None

        # Supervisor can list direct subordinate's root directory
        if not write:
            for sub_root in self._subordinate_root_dirs:
                if resolved == sub_root:
                    return None

        # Framework shared directories — read-only for all Animas
        if not write:
            from core.paths import (
                get_common_knowledge_dir,
                get_common_skills_dir,
                get_company_dir,
                get_reference_dir,
                get_shared_dir,
            )

            for shared_dir in (
                get_shared_dir(),
                get_common_knowledge_dir(),
                get_common_skills_dir(),
                get_reference_dir(),
                get_company_dir(),
            ):
                if shared_dir.exists() and resolved.is_relative_to(shared_dir.resolve()):
                    return None

        # Inter-anima boundary: block access to other anima's directories
        # that were not already allowed by subordinate/descendant/peer rules.
        animas_root = self._anima_dir.resolve().parent
        if resolved.is_relative_to(animas_root) and not resolved.is_relative_to(self._anima_dir.resolve()):
            logger.warning("permission_denied anima=%s path=%s reason=other_anima_dir", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"Access to other anima's directory is not allowed: {path}",
            )

        config = self._load_permissions_config()

        # file_roots == ["/"]: allow all (after protected file check)
        if config.file_roots == ["/"]:
            return None

        # file_roots == []: only anima_dir + framework shared dirs (already handled above)
        if not config.file_roots:
            logger.warning("permission_denied anima=%s path=%s reason=outside_allowed_dirs", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"'{path}' is not under any allowed directory",
                context={"allowed_dirs": []},
            )

        # Otherwise: check if path is under any of the file_roots
        allowed_dirs = [Path(r).resolve() for r in config.file_roots if r.startswith("/")]
        for allowed in allowed_dirs:
            if resolved.is_relative_to(allowed):
                return None

        logger.warning("permission_denied anima=%s path=%s reason=outside_allowed_dirs", self._anima_name, path)
        return _error_result(
            "PermissionDenied",
            f"'{path}' is not under any allowed directory",
            context={"allowed_dirs": [str(d) for d in allowed_dirs]},
        )

    def _check_command_permission(self, command: str) -> str | None:
        """Check if the command is allowed by permissions config and security rules.

        Returns ``None`` if allowed, or an error message string if denied.
        """
        if self._superuser:
            return None
        if not command or not command.strip():
            logger.warning("permission_denied anima=%s command=<empty>", self._anima_name)
            return _error_result("PermissionDenied", "Empty command")

        # Layer 1: Reject injection vectors
        inj_re = _get_injection_re()
        if inj_re and inj_re.search(command):
            logger.warning(
                "permission_denied anima=%s command=%s reason=injection_pattern",
                self._anima_name,
                command[:80],
            )
            return _error_result(
                "PermissionDenied",
                "Command contains injection patterns (;  \\n  `  $()  $VAR)",
                suggestion="Use pipes (|) or logical operators (&&) instead of semicolons. Avoid variable expansion and newlines.",
            )

        # Layer 2: Dangerous command patterns
        for pattern, reason in _get_blocked_patterns():
            if pattern.search(command):
                logger.warning(
                    "permission_denied anima=%s command=%s reason=blocked_pattern(%s)",
                    self._anima_name,
                    command[:80],
                    reason,
                )
                return _error_result("PermissionDenied", reason)

        # Layer 2.5: Per-anima denied commands from permissions config
        config = self._load_permissions_config()
        denied_items = config.commands.deny
        if denied_items:
            segments = [s.strip() for s in re.split(r"\|(?!\|)|\&\&|\|\|", command) if s.strip()]
            for segment in segments:
                try:
                    seg_argv = shlex.split(segment)
                except ValueError:
                    continue
                if not seg_argv:
                    continue
                cmd_base = seg_argv[0]
                for denied in denied_items:
                    if denied in cmd_base or denied in segment:
                        logger.warning(
                            "permission_denied anima=%s command=%s reason=denied_list(%s)",
                            self._anima_name,
                            command[:80],
                            denied,
                        )
                        return _error_result(
                            "PermissionDenied",
                            f"Command '{cmd_base}' is in denied list ('{denied}')",
                        )

        # Layer 3: If commands.allow_all is False, check commands.allow whitelist
        if not config.commands.allow_all:
            allowed = config.commands.allow
            if not allowed:
                logger.warning(
                    "permission_denied anima=%s command=%s reason=cmd_not_enabled", self._anima_name, command[:80]
                )
                return _error_result("PermissionDenied", "Command execution not enabled in permissions")
            segments = [s.strip() for s in re.split(r"\|(?!\|)|\&\&|\|\|", command) if s.strip()]
            for segment in segments:
                try:
                    seg_argv = shlex.split(segment)
                except ValueError as e:
                    return _error_result("PermissionDenied", f"Invalid command syntax: {e}")
                if not seg_argv:
                    continue
                cmd_base = seg_argv[0]
                if cmd_base not in allowed:
                    logger.warning(
                        "permission_denied anima=%s command=%s reason=not_in_allowed_list cmd=%s",
                        self._anima_name,
                        command[:80],
                        cmd_base,
                    )
                    return _error_result(
                        "PermissionDenied",
                        f"Command '{cmd_base}' not in allowed list",
                        context={"allowed_commands": allowed},
                    )
        else:
            segments = [s.strip() for s in re.split(r"\|(?!\|)|\&\&|\|\|", command) if s.strip()] or [command]

        # Layer 5: Path traversal check on all segments
        for segment in segments:
            try:
                seg_argv = shlex.split(segment)
            except ValueError:
                continue
            for arg in seg_argv[1:]:
                if ".." in arg:
                    try:
                        resolved = (self._anima_dir / arg).resolve()
                        if not resolved.is_relative_to(self._anima_dir.resolve()):
                            return _error_result(
                                "PermissionDenied",
                                "Command argument resolves outside anima directory",
                            )
                    except (ValueError, OSError):
                        pass

        return None
