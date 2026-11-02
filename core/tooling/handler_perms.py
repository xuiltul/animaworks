from __future__ import annotations

from core.tooling._handler_protocols import (
    _PermissionsHost,
)

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""PermissionsMixin — file/command permission checks and check_permissions handler."""

import json as _json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.config.file_access_policy import (
    FileAccessContext,
    FileAccessDecision,
    company_read_roots,
    effective_write_roots,
    evaluate_file_access,
    find_denied_root,
    resolve_effective_denied_roots,
)
from core.config.models import PermissionsConfig, load_permissions
from core.i18n import t
from core.tooling.handler_base import _error_result

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

    def _handle_check_permissions(self: _PermissionsHost, args: dict[str, Any]) -> str:
        """Return permission details or check one external tool/action."""
        if "tool_name" in args:
            tool_name = args.get("tool_name")
            action = args.get("action")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return _error_result("InvalidArguments", "tool_name is required")
            if action is not None and not isinstance(action, str):
                return _error_result("InvalidArguments", "action must be a string")
            tool_name = tool_name.strip()
            action = action.strip() if action else ""

            from core.integrations import TOOL_MODULES, discover_common_tools, discover_personal_tools
            from core.tooling.permissions import check_tool_access

            tool_file = None
            if tool_name in TOOL_MODULES:
                origin = "core"
            else:
                common = discover_common_tools()
                personal = discover_personal_tools(self._anima_dir)
                if tool_name in personal:
                    origin = "personal"
                    tool_file = Path(personal[tool_name])
                elif tool_name in common:
                    origin = "common"
                    tool_file = Path(common[tool_name])
                else:
                    return _json.dumps(
                        {
                            "tool": tool_name,
                            "action": action or None,
                            "permitted": False,
                            "reason": "unknown_tool",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            decision = check_tool_access(
                self._anima_dir,
                tool_name,
                action or None,
                origin=origin,
                tool_file=tool_file,
            )
            return _json.dumps(
                {
                    "tool": tool_name,
                    "action": action or None,
                    "permitted": decision.allowed,
                    "reason": decision.reason,
                },
                ensure_ascii=False,
                indent=2,
            )

        if args.get("action"):
            return _error_result("InvalidArguments", "tool_name is required when action is provided")

        internal_tools = sorted(self._dispatch.keys())

        external_enabled: list[str] = []
        external_available: list[str] = []
        try:
            from core.tooling.policy.registry import get_tool_modules

            all_categories = sorted(get_tool_modules().keys())
            for cat in all_categories:
                if cat in (self._external.registry if self._external else []):
                    external_enabled.append(cat)
                else:
                    external_available.append(cat)
        except Exception:
            logger.debug("Failed to enumerate external tools", exc_info=True)

        config = self._load_permissions_config()
        denied_roots = self._resolved_file_deny_roots(config)

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
        if self._peer_activity_dirs:
            file_read.append(t("handler.peer_activity"))

        if config.file_roots and config.file_roots != ["/"]:
            for root in config.file_roots:
                if Path(root).is_absolute():
                    file_read.append(root)
                    file_write.append(root)

        restrictions: list[str] = []
        for cmd in config.commands.deny:
            restrictions.append(t("handler.cmd_denied", cmd=cmd))

        # Command execution permissions
        commands_info: dict[str, Any] = {}
        if config.commands.allow_all:
            commands_info["policy"] = "allow_all"
            commands_info["note"] = "All commands are allowed (except those in deny list and built-in security blocks)"
        else:
            commands_info["policy"] = "allowlist"
            commands_info["allowed"] = config.commands.allow
        if config.commands.deny:
            commands_info["denied"] = config.commands.deny

        result = {
            "internal_tools": internal_tools,
            "external_tools": {
                "enabled": external_enabled,
                "available_but_not_enabled": external_available,
            },
            "file_access": {
                "read": file_read,
                "write": file_write,
                "denied": [str(root) for root in denied_roots],
            },
            "commands": commands_info,
            "restrictions": restrictions,
        }

        return _json.dumps(result, ensure_ascii=False, indent=2)

    # ── Tool creation permission ─────────────────────────────

    def _check_tool_creation_permission(self: _PermissionsHost, kind: str) -> bool:
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

    def _load_permissions_config(self: _PermissionsHost) -> PermissionsConfig:
        """Load PermissionsConfig from permissions.json (with migration fallback)."""
        return load_permissions(self._anima_dir)

    def _resolved_file_deny_roots(self: _PermissionsHost, config: PermissionsConfig | None = None) -> tuple[Path, ...]:
        """Return canonical configured and company-derived deny roots."""
        if self._superuser:
            return ()
        effective_config = config or self._load_permissions_config()
        return resolve_effective_denied_roots(self._anima_dir, effective_config.file_roots_denied)

    def _find_denied_file_root(
        self: _PermissionsHost,
        path: str | Path,
        denied_roots: tuple[Path, ...] | None = None,
    ) -> Path | None:
        """Return the deny root containing *path*, resolving symlinks first."""
        if self._superuser:
            return None
        roots = denied_roots if denied_roots is not None else self._resolved_file_deny_roots()
        return find_denied_root(path, roots)

    def _check_file_permission(
        self: _PermissionsHost,
        path: str,
        *,
        write: bool = False,
        trusted_internal_cache_write: bool = False,
        config: PermissionsConfig | None = None,
        denied_roots: tuple[Path, ...] | None = None,
    ) -> str | None:
        """Check if the file path is allowed by permissions config.

        Returns ``None`` if allowed, or an error message string if denied.
        """
        if self._superuser:
            context = FileAccessContext(
                anima_dir=self._anima_dir,
                data_dir=self._anima_dir.resolve().parent.parent,
                superuser=True,
            )
        else:
            effective_config = config or self._load_permissions_config()
            effective_denied_roots = (
                denied_roots if denied_roots is not None else self._resolved_file_deny_roots(effective_config)
            )

            from core.paths import get_data_dir

            data_dir = get_data_dir().resolve()
            write_roots = effective_write_roots(self._anima_dir, effective_config.file_roots) if write else ()
            additional_read_dirs: list[Path] = []
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
                    if shared_dir.exists():
                        additional_read_dirs.append(shared_dir.resolve())
                additional_read_dirs.extend(company_read_roots(self._anima_dir))

                # External skill roots (host ~/.claude/skills etc.) are
                # read-only and surfaced via external/<engine>/<name>/SKILL.md.
                try:
                    from core.config.models import load_config

                    for root in load_config().skills.external_roots:
                        if not getattr(root, "enabled", True):
                            continue
                        root_dir = Path(root.path).expanduser().resolve()
                        if root_dir.exists():
                            additional_read_dirs.append(root_dir)
                except Exception:
                    logger.debug("external_roots check skipped", exc_info=True)

            context = FileAccessContext(
                anima_dir=self._anima_dir.resolve(),
                data_dir=data_dir,
                denied_roots=effective_denied_roots,
                file_roots=tuple(effective_config.file_roots),
                file_roots_readonly=tuple(effective_config.file_roots_readonly),
                write_roots=tuple(write_roots),
                restrict_reads_to_roots=True,
                additional_read_dirs=tuple(additional_read_dirs),
                subordinate_activity_dirs=tuple(path.resolve() for path in self._subordinate_activity_dirs),
                descendant_activity_dirs=tuple(path.resolve() for path in self._descendant_activity_dirs),
                peer_activity_dirs=tuple(path.resolve() for path in self._peer_activity_dirs),
                subordinate_management_files=tuple(path.resolve() for path in self._subordinate_management_files),
                subordinate_root_dirs=tuple(path.resolve() for path in self._subordinate_root_dirs),
                descendant_read_files=tuple(path.resolve() for path in self._descendant_state_files),
                descendant_read_dirs=tuple(path.resolve() for path in self._descendant_state_dirs),
                trusted_internal_cache_write=trusted_internal_cache_write,
            )

        decision = evaluate_file_access(path, context, write=write)
        return self._file_access_error(path, decision)

    def _file_access_error(self: _PermissionsHost, path: str, decision: FileAccessDecision) -> str | None:
        """Convert the shared policy decision to ToolHandler's JSON error format."""
        if decision.allowed:
            return None

        if decision.reason == "internal_cache":
            logger.warning(
                "permission_denied anima=%s path=%s reason=internal_cache root=%s",
                self._anima_name,
                path,
                decision.internal_cache_root,
            )
            return _error_result(
                "PermissionDenied",
                f"Direct access to internal runtime cache is not allowed: '{path}'",
                context={"system_denied_root": str(decision.internal_cache_root)},
            )
        if decision.reason == "denied_root":
            logger.warning(
                "permission_denied anima=%s path=%s reason=file_root_denied root=%s",
                self._anima_name,
                path,
                decision.denied_root,
            )
            return _error_result(
                "PermissionDenied",
                f"'{path}' is under an explicitly denied directory",
                context={"denied_root": str(decision.denied_root)},
            )
        if decision.reason == "global_permissions":
            logger.warning(
                "permission_denied anima=%s path=%s reason=global_permissions_protected",
                self._anima_name,
                path,
            )
            return _error_result(
                "PermissionDenied",
                "permissions.global.json is a protected system file and cannot be modified by the anima itself",
            )
        if decision.reason == "protected_file":
            logger.warning("permission_denied anima=%s path=%s reason=protected_file", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"'{decision.protected_path}' is a protected file and cannot be modified by the anima itself",
            )
        if decision.reason == "protected_directory":
            logger.warning("permission_denied anima=%s path=%s reason=protected_directory", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"'{decision.protected_path}/' is a protected directory and cannot be modified by the anima itself",
            )
        if decision.reason == "other_anima":
            logger.warning("permission_denied anima=%s path=%s reason=other_anima_dir", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"Access to other anima's directory is not allowed: {path}",
            )
        if decision.reason == "readonly_dir":
            logger.warning("permission_denied anima=%s path=%s reason=readonly_dir", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"'{path}' is in a read-only directory (write not allowed)",
                context={"readonly_dir": str(decision.readonly_root)},
            )
        if decision.reason == "outside_allowed_dirs":
            logger.warning("permission_denied anima=%s path=%s reason=outside_allowed_dirs", self._anima_name, path)
            return _error_result(
                "PermissionDenied",
                f"'{path}' is not under any allowed directory",
                context={"allowed_dirs": [str(root) for root in decision.allowed_dirs]},
            )
        return _error_result("PermissionDenied", f"Access to file is not allowed: {path}")

    def _check_command_permission(self: _PermissionsHost, command: str) -> str | None:
        """Check if the command is allowed by permissions config and security rules.

        Returns ``None`` if allowed, or an error message string if denied.
        """
        if self._superuser:
            return None
        if not command or not command.strip():
            logger.warning("permission_denied anima=%s command=<empty>", self._anima_name)
            return _error_result("PermissionDenied", "Empty command")

        # Single shared command-policy decision function (all layers, same order
        # as Mode S and the Codex hook).  Loader failures are fail-closed.
        from core.tooling.policy.command_policy import (
            evaluate_command,
            load_command_policy_context,
            record_injection_hit,
        )

        try:
            ctx = load_command_policy_context(
                self._anima_dir,
                cwd=self._task_cwd or self._anima_dir,
                superuser=self._superuser,
                permissions=self._load_permissions_config(),
            )
        except Exception as exc:
            logger.error("command_policy load failed anima=%s: %s", self._anima_name, exc, exc_info=True)
            return _error_result(
                "PermissionDenied",
                t("tooling.command_policy_check_failed", error=type(exc).__name__),
            )

        decision = evaluate_command(command, ctx)
        if decision.injection_hit:
            record_injection_hit(
                command,
                ctx,
                pattern_name=decision.injection_hit,
                trigger=getattr(self, "_trigger", ""),
            )
        if decision.allowed:
            return None

        logger.warning(
            "permission_denied anima=%s command=%s reason=%s",
            self._anima_name,
            command[:80],
            decision.layer,
        )
        if decision.layer == "injection":
            return _error_result(
                "PermissionDenied",
                decision.reason,
                suggestion="Use pipes (|) or logical operators (&&) instead of semicolons. Avoid embedded newlines.",
            )
        if decision.layer == "allowlist":
            return _error_result(
                "PermissionDenied",
                decision.reason,
                context={"allowed_commands": ctx.permissions.commands.allow},
            )
        return _error_result("PermissionDenied", decision.reason)
