from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""MemoryToolsMixin — memory file search, read, write, and archive handlers."""

import logging
import re
from dataclasses import dataclass
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

_SEARCH_MAX_TOKENS = 25_000
_SEARCH_MAX_LINES = 2_000
_SEARCH_CONTEXT_BASE = 128_000
_SEARCH_MIN_RESULTS = 3


@dataclass(frozen=True, slots=True)
class _PathNormResult:
    """Result of memory path normalization."""

    rel: str
    channel_redirect: str | None = None


def _normalize_memory_path(raw: str, anima_dir: Path) -> _PathNormResult:
    """Normalize a memory file path to a canonical relative form.

    Resolves absolute paths, collapses slashes, and maps shared dirs
    (common_knowledge, reference, common_skills, shared/channels) to
    canonical prefixes. For shared/channels, returns channel_redirect
    so the caller can delegate to read_channel/post_channel.
    """
    original = raw
    raw = raw.strip()
    raw = re.sub(r"/+", "/", raw)
    if raw.endswith("/") and raw != "/":
        raw = raw[:-1]
    while raw.startswith("./"):
        raw = raw[2:]

    # Fast path: no absolute path and no .. in path
    if not raw.startswith("/") and ".." not in raw:
        result = _PathNormResult(rel=raw)
        if result.rel != original:
            logger.info("memory path normalized: %r → %r", original, result.rel)
        return result

    # Prefix-qualified paths with .. must NOT be normalized here — they must
    # go through the downstream prefix-specific traversal checks unchanged.
    _PREFIX_DIRS = ("common_knowledge/", "reference/", "common_skills/")
    if any(raw.startswith(p) for p in _PREFIX_DIRS) and ".." in raw:
        return _PathNormResult(rel=raw)

    # Resolve to absolute
    if raw.startswith("/"):
        resolved = Path(raw).resolve()
    else:
        resolved = (anima_dir / raw).resolve()

    anima_resolved = anima_dir.resolve()
    animas_dir = anima_resolved.parent

    # a. Under anima_dir
    try:
        rel = str(resolved.relative_to(anima_resolved))
        result = _PathNormResult(rel=rel)
        if result.rel != original:
            logger.info("memory path normalized: %r → %r", original, result.rel)
        return result
    except ValueError:
        pass

    # b. Under animas dir (sibling anima)
    if resolved.is_relative_to(animas_dir):
        try:
            rel = str(resolved.relative_to(animas_dir))
            result = _PathNormResult(rel=f"../{rel}")
            if result.rel != original:
                logger.info("memory path normalized: %r → %r", original, result.rel)
            return result
        except ValueError:
            pass

    # c–f. Shared dirs (lazy import to avoid circular deps)
    from core.paths import get_common_knowledge_dir, get_common_skills_dir, get_data_dir, get_reference_dir

    ck_dir = get_common_knowledge_dir().resolve()
    if resolved.is_relative_to(ck_dir):
        try:
            rel = str(resolved.relative_to(ck_dir))
            result = _PathNormResult(rel=f"common_knowledge/{rel}")
            if result.rel != original:
                logger.info("memory path normalized: %r → %r", original, result.rel)
            return result
        except ValueError:
            pass

    ref_dir = get_reference_dir().resolve()
    if resolved.is_relative_to(ref_dir):
        try:
            rel = str(resolved.relative_to(ref_dir))
            result = _PathNormResult(rel=f"reference/{rel}")
            if result.rel != original:
                logger.info("memory path normalized: %r → %r", original, result.rel)
            return result
        except ValueError:
            pass

    cs_dir = get_common_skills_dir().resolve()
    if resolved.is_relative_to(cs_dir):
        try:
            rel = str(resolved.relative_to(cs_dir))
            result = _PathNormResult(rel=f"common_skills/{rel}")
            if result.rel != original:
                logger.info("memory path normalized: %r → %r", original, result.rel)
            return result
        except ValueError:
            pass

    channels_dir = (get_data_dir() / "shared" / "channels").resolve()
    if resolved.is_relative_to(channels_dir):
        return _PathNormResult(rel=raw, channel_redirect=resolved.stem)

    # g. Fallback: strip leading /
    result = _PathNormResult(rel=raw.lstrip("/") if raw.startswith("/") else raw)
    if result.rel != original:
        logger.info("memory path normalized: %r → %r", original, result.rel)
    return result


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
    _on_schedule_changed: Callable[[str], None] | None
    _min_trust_seen: int
    _read_paths: set[str]

    def _handle_search_memory(self, args: dict[str, Any]) -> str:
        scope = args.get("scope", "all")
        query = args.get("query", "")
        offset = int(args.get("offset", 0))

        results = self._memory.search_memory_text(
            query,
            scope=scope,
            offset=offset,
            context_window=getattr(self, "_context_window", _SEARCH_CONTEXT_BASE),
        )
        logger.debug(
            "search_memory query=%s scope=%s offset=%d results=%d",
            query,
            scope,
            offset,
            len(results),
        )
        if not results:
            if offset > 0:
                return f"No more results for '{query}' at offset={offset}."
            return f"No results for '{query}'"

        scale = min(1.0, getattr(self, "_context_window", _SEARCH_CONTEXT_BASE) / _SEARCH_CONTEXT_BASE)
        max_tokens = int(_SEARCH_MAX_TOKENS * scale)
        max_lines = int(_SEARCH_MAX_LINES * scale)

        search_method = results[0].get("search_method", "vector") if results else "vector"
        header = f'Search results for "{query}" ({search_method}, {scope}, {offset + 1}-{offset + len(results)}):\n'

        output_parts: list[str] = [header]
        total_tokens = len(header) // 4
        total_lines = header.count("\n") + 1

        for shown_count, r in enumerate(results):
            source = r.get("source_file", "unknown")
            score = r.get("score", 0.0)
            chunk_idx = r.get("chunk_index", 0)
            total_chunks = r.get("total_chunks", 1)
            content = r.get("content", "")

            entry_header = (
                f"[{offset + shown_count + 1}] score={score:.2f} | {source} | chunk {chunk_idx + 1}/{total_chunks}"
            )
            entry = f"\n{entry_header}\n{content}\n"

            entry_tokens = len(entry) // 4
            entry_lines = entry.count("\n") + 1

            if shown_count >= _SEARCH_MIN_RESULTS and (
                total_tokens + entry_tokens > max_tokens or total_lines + entry_lines > max_lines
            ):
                output_parts.append("\n(truncated — output limit reached)")
                break

            output_parts.append(entry)
            total_tokens += entry_tokens
            total_lines += entry_lines

        if len(results) >= 10:
            output_parts.append(f"\nUse offset={offset + len(results)} to see next page.")

        return "".join(output_parts)

    def _handle_read_memory_file(self, args: dict[str, Any]) -> str:
        norm = _normalize_memory_path(args["path"], self._anima_dir)
        if norm.channel_redirect:
            return self._handle_read_channel({"channel": norm.channel_redirect})
        rel = args["path"] = norm.rel

        if rel == "state/current_task.md":
            rel = args["path"] = "state/current_state.md"

        # Support common_knowledge/ prefix — resolve to shared dir
        if rel.startswith("common_knowledge/"):
            from core.paths import get_common_knowledge_dir

            suffix = rel[len("common_knowledge/") :]
            ck_dir = get_common_knowledge_dir()
            path = (ck_dir / suffix).resolve()
            if not path.is_relative_to(ck_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        elif rel.startswith("reference/"):
            from core.paths import get_reference_dir

            suffix = rel[len("reference/") :]
            ref_dir = get_reference_dir()
            path = (ref_dir / suffix).resolve()
            if not path.is_relative_to(ref_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        elif rel.startswith("common_skills/"):
            from core.paths import get_common_skills_dir

            suffix = rel[len("common_skills/") :]
            cs_dir = get_common_skills_dir()
            path = (cs_dir / suffix).resolve()
            if not path.is_relative_to(cs_dir.resolve()):
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
                        f"Path '{rel}' resolves outside your directory. "
                        "Use relative paths like 'knowledge/foo.md'. "
                        "For shared channels use read_channel/post_channel.",
                    )
        if path.exists() and path.is_file():
            logger.debug("read_memory_file path=%s", rel)
            self._read_paths.add(rel)
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)
            MAX_LINES = 2000
            if len(lines) > MAX_LINES:
                truncated = "".join(lines[:MAX_LINES])
                return (
                    truncated
                    + f"\n[Truncated: showing {MAX_LINES} of {len(lines)} lines. Use read_file tool or narrow your search to access more.]"
                )
            return content
        logger.debug("read_memory_file NOT FOUND path=%s", rel)
        parent = path.parent
        hint = ""
        if parent.exists() and parent.is_dir():
            siblings = sorted(f.name for f in parent.iterdir() if f.is_file())[:20]
            if siblings:
                hint = f"\nAvailable files in {parent.name}/:\n" + "\n".join(f"  - {s}" for s in siblings)
        return f"File not found: {rel}{hint}"

    def _handle_write_memory_file(self, args: dict[str, Any]) -> str:
        norm = _normalize_memory_path(args["path"], self._anima_dir)
        if norm.channel_redirect:
            return self._handle_post_channel(
                {
                    "channel": norm.channel_redirect,
                    "content": args.get("content", ""),
                }
            )
        rel = args["path"] = norm.rel

        if rel == "state/current_task.md":
            rel = args["path"] = "state/current_state.md"

        if rel.startswith("reference/"):
            return _error_result(
                "PermissionDenied",
                "reference/ is read-only. Use common_knowledge/ for shared writable documents.",
            )

        # Support common_knowledge/ prefix — resolve to shared dir
        if rel.startswith("common_knowledge/"):
            from core.paths import get_common_knowledge_dir

            suffix = rel[len("common_knowledge/") :]
            ck_dir = get_common_knowledge_dir()
            path = (ck_dir / suffix).resolve()
            if not path.is_relative_to(ck_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        elif rel.startswith("common_skills/"):
            from core.paths import get_common_skills_dir

            suffix = rel[len("common_skills/") :]
            cs_dir = get_common_skills_dir()
            path = (cs_dir / suffix).resolve()
            if not path.is_relative_to(cs_dir.resolve()):
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
        else:
            path = self._anima_dir / rel

        # Security check: block protected files and path traversal
        if not self._superuser and not rel.startswith(("common_knowledge/", "common_skills/")):
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

        _was_existing = path.exists()
        mode = args.get("mode", "overwrite")

        # ── Read-before-write guard ──
        _rbw_skip = mode == "append" or not _was_existing or rel.startswith(("episodes/", "state/", "shortterm/"))
        if not _rbw_skip and rel not in self._read_paths:
            try:
                _existing = path.read_text(encoding="utf-8")[:2000]
            except OSError:
                _existing = "(could not read existing content)"
            return _error_result(
                "ReadBeforeWrite",
                t("handler.read_before_write", path=rel, existing=_existing),
            )

        content = args["content"]

        path.parent.mkdir(parents=True, exist_ok=True)

        lock = self._state_file_lock if self._state_file_lock and self._is_state_file(path) else None
        if lock:
            lock.acquire()
        try:
            # Auto-add YAML frontmatter for procedure overwrite writes
            auto_frontmatter_applied = False
            if (
                rel.startswith("procedures/")
                and rel.endswith(".md")
                and mode == "overwrite"
                and not content.lstrip().startswith("---")
            ):
                desc = _extract_first_heading(content)
                metadata = {
                    "description": desc,
                    "success_count": 0,
                    "failure_count": 0,
                    "confidence": 0.5,
                }
                self._memory.write_procedure_with_meta(path, content, metadata)
                auto_frontmatter_applied = True
            elif (
                rel.startswith("knowledge/")
                and rel.endswith(".md")
                and mode == "overwrite"
                and content.lstrip().startswith("---")
            ):
                # LLM wrote frontmatter — parse, validate, and complete
                import yaml as _yaml_km_fm

                from core.memory.frontmatter import (
                    parse_frontmatter as _parse_fm_hw,
                )
                from core.memory.frontmatter import (
                    validate_and_complete_frontmatter as _validate_fm_hw,
                )
                from core.time_utils import now_local as _now_local_hw

                _meta_hw, _body_hw = _parse_fm_hw(content.lstrip())
                if _meta_hw:
                    # Preserve original created_at on overwrite; update updated_at
                    if path.exists():
                        try:
                            _existing_text = path.read_text(encoding="utf-8")
                            _existing_meta, _ = _parse_fm_hw(_existing_text)
                            if _existing_meta.get("created_at"):
                                _meta_hw.setdefault("created_at", _existing_meta["created_at"])
                        except OSError:
                            pass
                    _validate_fm_hw(_meta_hw, path)
                    _meta_hw["updated_at"] = _now_local_hw().isoformat()
                    _fm_hw = _yaml_km_fm.dump(_meta_hw, default_flow_style=False, allow_unicode=True)
                    path.write_text(f"---\n{_fm_hw}---\n\n{_body_hw.lstrip()}", encoding="utf-8")
                    auto_frontmatter_applied = True
                else:
                    # Parse failed — strip broken FM, apply framework-generated metadata
                    from core.memory.frontmatter import strip_content_frontmatter as _strip_fm_hw

                    _clean_body_hw = _strip_fm_hw(content.lstrip())
                    _ts_fb = _now_local_hw().isoformat()
                    _fallback_meta: dict[str, Any] = {
                        "confidence": 0.5,
                        "created_at": _ts_fb,
                        "updated_at": _ts_fb,
                        "source_episodes": 0,
                        "auto_consolidated": False,
                        "version": 1,
                    }
                    if path.exists():
                        try:
                            _existing_text_fb = path.read_text(encoding="utf-8")
                            _existing_meta_fb, _ = _parse_fm_hw(_existing_text_fb)
                            if _existing_meta_fb.get("created_at"):
                                _fallback_meta["created_at"] = _existing_meta_fb["created_at"]
                        except OSError:
                            pass
                    _fm_fb = _yaml_km_fm.dump(
                        _fallback_meta,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                    path.write_text(
                        f"---\n{_fm_fb}---\n\n{_clean_body_hw.lstrip()}",
                        encoding="utf-8",
                    )
                    auto_frontmatter_applied = True
                    logger.info(
                        "Frontmatter parse failed for %s — applied fallback metadata",
                        rel,
                    )
            elif (
                rel.startswith("knowledge/")
                and rel.endswith(".md")
                and mode == "overwrite"
                and not content.lstrip().startswith("---")
            ):
                import yaml as _yaml_km

                from core.memory.frontmatter import strip_content_frontmatter
                from core.time_utils import now_local

                # Preserve original created_at on overwrite
                _original_created_at = None
                if path.exists():
                    try:
                        from core.memory.frontmatter import parse_frontmatter as _parse_fm_ow

                        _existing_text_ow = path.read_text(encoding="utf-8")
                        _existing_meta_ow, _ = _parse_fm_ow(_existing_text_ow)
                        _original_created_at = _existing_meta_ow.get("created_at")
                    except OSError:
                        pass
                ts = now_local().isoformat()
                metadata: dict[str, Any] = {
                    "confidence": 0.5,
                    "created_at": _original_created_at or ts,
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
            args["path"],
            args.get("mode", "overwrite"),
        )

        # Activity log: memory write
        self._activity.log(
            "memory_write",
            summary=f"{rel} ({args.get('mode', 'overwrite')})",
            meta={"path": rel, "mode": args.get("mode", "overwrite")},
        )

        # ── Filename token hint for new knowledge files ──
        _similar_hint = ""
        if rel.startswith("knowledge/") and mode == "overwrite" and not _was_existing:
            _new_stem = Path(rel).stem.replace("-", "_")
            _new_tokens = set(_new_stem.split("_"))
            _knowledge_dir = self._anima_dir / "knowledge"
            if _knowledge_dir.is_dir():
                _existing_names = [
                    f.name for f in _knowledge_dir.iterdir() if f.suffix == ".md" and f.name != Path(rel).name
                ]
                _similar = []
                for _ef in _existing_names:
                    _ef_tokens = set(_ef.replace("-", "_").replace(".md", "").split("_"))
                    if len(_new_tokens & _ef_tokens) >= 2:
                        _similar.append(_ef)
                if _similar:
                    _similar.sort()
                    _lines = "\n".join(f"  - {s}" for s in _similar[:10])
                    _similar_hint = t("handler.similar_knowledge_hint", files=_lines)

        # Trigger schedule reload if heartbeat or cron config changed
        if args["path"] in ("heartbeat.md", "cron.md") and self._on_schedule_changed:
            try:
                self._on_schedule_changed(self._anima_name)
                logger.info("Schedule reload triggered for '%s'", self._anima_name)
            except Exception:
                logger.exception("Schedule reload failed for '%s'", self._anima_name)

        result = f"Written to {args['path']}"
        if _similar_hint:
            result = f"{result}\n\n{_similar_hint}"

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
        norm = _normalize_memory_path(rel, self._anima_dir)
        rel = args["path"] = norm.rel
        reason = args.get("reason", "")

        if not rel:
            return _error_result("InvalidArguments", "path is required")
        if not reason:
            return _error_result("InvalidArguments", "reason is required")

        _ARCHIVABLE_PREFIXES = ("knowledge/", "procedures/", "state/overflow_inbox/")
        if not any(rel.startswith(p) for p in _ARCHIVABLE_PREFIXES):
            return _error_result(
                "PermissionDenied",
                "Only files under knowledge/, procedures/, or state/overflow_inbox/ can be archived",
                suggestion="Specify a path like 'knowledge/old-info.md', 'procedures/old-proc.md', or 'state/overflow_inbox/msg.md'",
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
