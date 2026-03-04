from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""MemoryToolsMixin — memory file search, read, write, and archive handlers."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.i18n import t

from core.tooling.handler_base import (
    _error_result,
    _extract_first_heading,
    _is_protected_write,
    _validate_episode_path,
    _validate_procedure_format,
    _validate_skill_format,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from core.memory import MemoryManager
    from core.memory.activity import ActivityLogger

logger = logging.getLogger("animaworks.tool_handler")


class MemoryToolsMixin:
    """Memory file search, read, write, and archive tool handlers."""

    # Declared for type-checker visibility; actual values live on ToolHandler
    _anima_dir: Path
    _anima_name: str
    _superuser: bool
    _memory: MemoryManager
    _activity: ActivityLogger
    _subordinate_activity_dirs: list[Path]
    _subordinate_management_files: list[Path]
    _descendant_activity_dirs: list[Path]
    _descendant_state_files: list[Path]
    _descendant_state_dirs: list[Path]
    _peer_activity_dirs: list[Path]
    _state_file_lock: threading.Lock | None
    _on_schedule_changed: Callable[[str], Any] | None
    _min_trust_seen: int

    def _handle_search_memory(self, args: dict[str, Any]) -> str:
        scope = args.get("scope", "all")
        query = args.get("query", "")
        results = self._memory.search_memory_text(query, scope=scope)
        logger.debug(
            "search_memory query=%s scope=%s results=%d",
            query, scope, len(results),
        )
        if not results:
            return f"No results for '{query}'"
        return "\n".join(f"- {fname}: {line}" for fname, line in results[:10])

    def _handle_read_memory_file(self, args: dict[str, Any]) -> str:
        rel = args["path"]
        # Support common_knowledge/ prefix — resolve to shared dir
        if rel.startswith("common_knowledge/"):
            from core.paths import get_common_knowledge_dir
            suffix = rel[len("common_knowledge/"):]
            ck_dir = get_common_knowledge_dir()
            path = (ck_dir / suffix).resolve()
            if not path.is_relative_to(ck_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        else:
            path = self._anima_dir / rel
            resolved = path.resolve()
            # Allow if within own anima_dir
            if not self._superuser and not resolved.is_relative_to(self._anima_dir.resolve()):
                allowed = False
                for sub_activity in self._subordinate_activity_dirs:
                    if resolved.is_relative_to(sub_activity):
                        allowed = True
                        break
                if not allowed:
                    for mgmt_file in self._subordinate_management_files:
                        if resolved == mgmt_file:
                            allowed = True
                            break
                if not allowed:
                    for desc_activity in self._descendant_activity_dirs:
                        if resolved.is_relative_to(desc_activity):
                            allowed = True
                            break
                if not allowed:
                    for desc_state in self._descendant_state_files:
                        if resolved == desc_state:
                            allowed = True
                            break
                if not allowed:
                    for desc_state_dir in self._descendant_state_dirs:
                        if resolved.is_relative_to(desc_state_dir):
                            allowed = True
                            break
                if not allowed:
                    return _error_result(
                        "PermissionDenied",
                        "Path resolves outside anima directory",
                    )
        if path.exists() and path.is_file():
            logger.debug("read_memory_file path=%s", rel)
            return path.read_text(encoding="utf-8")
        logger.debug("read_memory_file NOT FOUND path=%s", rel)
        parent = path.parent
        hint = ""
        if parent.exists() and parent.is_dir():
            siblings = sorted(f.name for f in parent.iterdir() if f.is_file())[:20]
            if siblings:
                hint = f"\nAvailable files in {parent.name}/:\n" + "\n".join(
                    f"  - {s}" for s in siblings
                )
        return f"File not found: {rel}{hint}"

    def _handle_write_memory_file(self, args: dict[str, Any]) -> str:
        rel = args["path"]

        # Support common_knowledge/ prefix — resolve to shared dir
        if rel.startswith("common_knowledge/"):
            from core.paths import get_common_knowledge_dir

            suffix = rel[len("common_knowledge/"):]
            ck_dir = get_common_knowledge_dir()
            path = (ck_dir / suffix).resolve()
            if not path.is_relative_to(ck_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        else:
            path = self._anima_dir / rel

        # Security check: block protected files and path traversal
        if not self._superuser and not rel.startswith("common_knowledge/"):
            err = _is_protected_write(self._anima_dir, path)
            if err:
                resolved = path.resolve()
                subordinate_allowed = False
                for mgmt_file in self._subordinate_management_files:
                    if resolved == mgmt_file:
                        subordinate_allowed = True
                        break
                if not subordinate_allowed:
                    return err

        # Tool creation permission check
        if rel.startswith("tools/") and rel.endswith(".py"):
            if not self._check_tool_creation_permission("個人ツール"):
                return _error_result(
                    "PermissionDenied",
                    t("handler.tool_creation_denied"),
                )

        content = args["content"]
        mode = args.get("mode", "overwrite")

        path.parent.mkdir(parents=True, exist_ok=True)

        lock = self._state_file_lock if self._state_file_lock and self._is_state_file(path) else None
        if lock:
            lock.acquire()
        try:
            # Auto-add YAML frontmatter for procedure overwrite writes
            auto_frontmatter_applied = False
            if (rel.startswith("procedures/") and rel.endswith(".md")
                    and mode == "overwrite"
                    and not content.lstrip().startswith("---")):
                desc = _extract_first_heading(content)
                metadata = {
                    "description": desc,
                    "success_count": 0,
                    "failure_count": 0,
                    "confidence": 0.5,
                }
                self._memory.write_procedure_with_meta(path, content, metadata)
                auto_frontmatter_applied = True
            elif (rel.startswith("knowledge/") and rel.endswith(".md")
                    and mode == "overwrite"
                    and not content.lstrip().startswith("---")):
                import yaml as _yaml_km
                from core.schemas import now_jst
                from core.memory.frontmatter import strip_content_frontmatter
                ts = now_jst().isoformat()
                metadata: dict[str, Any] = {
                    "confidence": 0.5,
                    "created_at": ts,
                    "updated_at": ts,
                    "source_episodes": 0,
                    "auto_consolidated": False,
                    "version": 1,
                }
                _trust_rank_map_pre = {0: "external_web", 1: "mixed"}
                _min_trust_pre = getattr(self, "_min_trust_seen", 2)
                if _min_trust_pre >= 2:
                    _trust_file_pre = self._anima_dir / "run" / "min_trust_seen"
                    try:
                        if _trust_file_pre.exists():
                            _min_trust_pre = min(
                                _min_trust_pre,
                                int(_trust_file_pre.read_text(encoding="utf-8").strip()),
                            )
                    except (ValueError, OSError):
                        pass
                _origin_pre = _trust_rank_map_pre.get(_min_trust_pre, "")
                if _origin_pre:
                    metadata["origin"] = _origin_pre
                _clean = strip_content_frontmatter(content)
                _fm = _yaml_km.dump(metadata, default_flow_style=False, allow_unicode=True)
                path.write_text(f"---\n{_fm}---\n\n{_clean}", encoding="utf-8")
                auto_frontmatter_applied = True
            elif mode == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")
        finally:
            if lock:
                lock.release()
        logger.info(
            "write_memory_file path=%s mode=%s",
            args["path"], args.get("mode", "overwrite"),
        )

        # Activity log: memory write
        self._activity.log(
            "memory_write",
            summary=f"{rel} ({args.get('mode', 'overwrite')})",
            meta={"path": rel, "mode": args.get("mode", "overwrite")},
        )

        # Trigger schedule reload if heartbeat or cron config changed
        if args["path"] in ("heartbeat.md", "cron.md") and self._on_schedule_changed:
            try:
                self._on_schedule_changed(self._anima_name)
                logger.info("Schedule reload triggered for '%s'", self._anima_name)
            except Exception:
                logger.exception("Schedule reload failed for '%s'", self._anima_name)

        result = f"Written to {args['path']}"

        # Warn (but don't block) if episode filename is non-standard
        episode_warning = _validate_episode_path(args["path"])
        if episode_warning:
            logger.warning("Non-standard episode path: %s", args["path"])
            result = f"{result}\n\n{episode_warning}"

        # Validate skill file format (soft validation: warn but don't block)
        if (rel.startswith("skills/") or rel.startswith("common_skills/")) and rel.endswith(".md"):
            validation_msg = _validate_skill_format(args["content"])
            if validation_msg:
                result = f"{result}\n\n{t('handler.skill_format_validation', msg=validation_msg)}"

        # Validate procedure file format (soft validation: warn but don't block)
        if rel.startswith("procedures/") and rel.endswith(".md") and not auto_frontmatter_applied:
            validation_msg = _validate_procedure_format(args["content"])
            if validation_msg:
                result = f"{result}\n\n{t('handler.procedure_format_validation', msg=validation_msg)}"

        # Auto-update RAG index for skill/procedure writes
        if rel.startswith(("skills/", "procedures/")) and rel.endswith(".md"):
            indexer = self._memory._get_indexer()
            if indexer:
                memory_type = "skills" if rel.startswith("skills/") else "procedures"
                try:
                    indexer.index_file(path, memory_type=memory_type, force=True)
                except Exception as e:
                    logger.warning("Failed to update RAG index for %s: %s", rel, e)

        # Auto-update RAG index for knowledge writes + origin frontmatter
        # (skip origin injection when auto-frontmatter already handled it)
        if rel.startswith("knowledge/") and rel.endswith(".md"):
            _trust_rank_map = {0: "external_web", 1: "mixed"}
            min_trust = getattr(self, "_min_trust_seen", 2)

            # Also check file-based trust (Mode S writes via MCP subprocess)
            if min_trust >= 2:
                _trust_file = self._anima_dir / "run" / "min_trust_seen"
                try:
                    if _trust_file.exists():
                        file_val = int(_trust_file.read_text(encoding="utf-8").strip())
                        min_trust = min(min_trust, file_val)
                except (ValueError, OSError):
                    pass

            origin = _trust_rank_map.get(min_trust, "")

            if origin and mode != "append" and not auto_frontmatter_applied:
                current = path.read_text(encoding="utf-8")
                if not current.startswith("---\norigin:"):
                    path.write_text(
                        f"---\norigin: {origin}\n---\n\n{current}",
                        encoding="utf-8",
                    )

            indexer = self._memory._get_indexer()
            if indexer:
                try:
                    indexer.index_file(
                        path,
                        memory_type="knowledge",
                        force=True,
                        origin=origin or None,
                    )
                except Exception as e:
                    logger.warning("Failed to update RAG index for %s: %s", rel, e)

        return result

    def _handle_archive_memory_file(self, args: dict[str, Any]) -> str:
        """Archive a memory file by moving it to archive/superseded/."""
        import shutil

        rel = args.get("path", "")
        reason = args.get("reason", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")
        if not reason:
            return _error_result("InvalidArguments", "reason is required")

        if not (rel.startswith("knowledge/") or rel.startswith("procedures/")):
            return _error_result(
                "PermissionDenied",
                "Only files under knowledge/ and procedures/ can be archived",
                suggestion="Specify a path like 'knowledge/old-info.md' or 'procedures/old-proc.md'",
            )

        target = self._anima_dir / rel

        err = _is_protected_write(self._anima_dir, target)
        if err:
            return err

        if not target.exists():
            return _error_result(
                "FileNotFound",
                f"File not found: {rel}",
                suggestion="Check the path with list_directory or search_memory",
            )

        if not target.is_file():
            return _error_result(
                "InvalidArguments",
                f"Not a file: {rel}",
            )

        archive_dir = self._anima_dir / "archive" / "superseded"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / target.name

        if dest.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 1
            while dest.exists():
                dest = archive_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.move(str(target), str(dest))

        logger.info("archive_memory_file: %s -> %s (reason: %s)", rel, dest.name, reason)

        self._activity.log(
            "memory_write",
            summary=f"archived {rel}: {reason}",
            meta={"path": rel, "reason": reason, "action": "archive"},
        )

        return f"Archived {rel} -> archive/superseded/{dest.name} (reason: {reason})"
