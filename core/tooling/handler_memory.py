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

from core.anima.workspace_files import (
    AGENTS_FILE,
    HEARTBEAT_FILE,
    IDENTITY_FILE,
    LEGACY_IDENTITY,
    LEGACY_RENAMES,
    SOUL_FILE,
)
from core.config.file_access_policy import load_denied_roots, memory_source_is_allowed
from core.i18n import t
from core.memory.io import archive_episode_before_write
from core.memory.retrieval.search_metadata import format_result_metadata_line
from core.tooling._handler_protocols import (
    _MemoryToolsHost,
)
from core.tooling.handler_base import (
    _error_result,
    _extract_first_heading,
    _is_protected_write,
    _validate_episode_path,
    _validate_procedure_format,
    _validate_skill_format,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.activity.logger import ActivityLogger
    from core.memory import MemoryManager
    from core.memory.state_lock import StateFileLock

logger = logging.getLogger("animaworks.tool_handler")

_SEARCH_MAX_TOKENS = 8_000
_SEARCH_MAX_LINES = 600
_SEARCH_CONTEXT_BASE = 128_000
_SEARCH_MIN_RESULTS = 3
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CASE_RECORD_PR_RE = re.compile(r"(?<![\w])#\s*\d{2,}\b")
# Hex runs that mix digits and letters, so plain numbers (amounts, IDs) don't count.
_CASE_RECORD_SHA_RE = re.compile(
    r"(?<![0-9a-f])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![0-9a-f])", re.IGNORECASE
)
_CASE_RECORD_DATE_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:-\d{1,2}-\d{1,2}|/\d{1,2}/\d{1,2}|年\d{1,2}月\d{1,2}日)(?!\d)"
)

_PROMPT_SETTING_FILES = {
    "identity": IDENTITY_FILE,
    "soul": SOUL_FILE,
    "agents": AGENTS_FILE,
    "injection": AGENTS_FILE,
}
_PROMPT_SETTINGS_BY_FILE = {
    IDENTITY_FILE: "identity",
    SOUL_FILE: "soul",
    AGENTS_FILE: "agents",
}


def _legacy_workspace_name(rel: str, legacy_names: set[str] | frozenset[str]) -> str | None:
    """Return the legacy basename when *rel* points at an anima workspace root file."""
    parts = rel.split("/")
    if len(parts) == 1 and parts[0] in legacy_names:
        return parts[0]
    if len(parts) == 3 and parts[0] == ".." and parts[2] in legacy_names:
        return parts[2]
    return None


def _replace_workspace_basename(rel: str, new_name: str) -> str:
    parent, sep, _old = rel.rpartition("/")
    return f"{parent}/{new_name}" if sep else new_name


def _looks_like_case_record(content: str) -> bool:
    """Heuristically identify project-specific notes better stored as episodes."""
    if _CASE_RECORD_PR_RE.search(content) or _CASE_RECORD_SHA_RE.search(content):
        return True
    return len(set(_CASE_RECORD_DATE_RE.findall(content))) >= 2


def _source_is_in_project(source: str, project: str) -> bool:
    """Return whether a search source belongs to exactly one project archive."""
    normalized = source.replace("\\", "/")
    if ".." in normalized.split("/"):
        return False
    return (
        re.match(
            rf"^(?:episodes|knowledge|procedures|facts)/projects/{re.escape(project)}/",
            normalized,
        )
        is not None
    )


@dataclass(frozen=True, slots=True)
class _PathNormResult:
    """Result of memory path normalization."""

    rel: str
    channel_redirect: str | None = None


@dataclass(frozen=True, slots=True)
class _MemoryWriteRequest:
    """Normalized inputs shared by scope-specific memory writers."""

    rel: str
    path: Path
    content: str
    mode: str
    was_existing: bool
    write_origin: str
    skill_capture: Any = None


@dataclass(frozen=True, slots=True)
class _MemoryWriteOutcome:
    """Scope writer metadata needed by the common post-write path."""

    auto_frontmatter_applied: bool = False
    error: str | None = None


_MEMORY_WRITE_SCOPE_PREFIXES = (
    ("common_knowledge/", "common_knowledge"),
    ("common_skills/", "common_skills"),
    ("knowledge/", "knowledge"),
    ("procedures/", "procedures"),
    ("episodes/", "episodes"),
    ("facts/", "facts"),
    ("state/", "state"),
    ("skills/", "skills"),
    ("shortterm/", "shortterm"),
    ("tools/", "tools"),
)
_MEMORY_WRITE_HANDLERS = {
    "knowledge": "_write_knowledge_memory_file",
    "procedures": "_write_procedure_memory_file",
    "episodes": "_write_episode_memory_file",
    "facts": "_write_plain_memory_file",
    "state": "_write_plain_memory_file",
    "skills": "_write_plain_memory_file",
    "common_knowledge": "_write_plain_memory_file",
    "common_skills": "_write_plain_memory_file",
    "shortterm": "_write_plain_memory_file",
    "tools": "_write_plain_memory_file",
    "default": "_write_plain_memory_file",
}


def _knowledge_frontmatter_text(path: Path, rel: str, content: str, write_origin: str) -> str:
    """Build completed YAML metadata for a knowledge overwrite."""
    import yaml

    from core.memory.frontmatter import parse_frontmatter, strip_content_frontmatter
    from core.time_utils import now_local

    if content.lstrip().startswith("---"):
        meta, body = parse_frontmatter(content.lstrip())
        if meta:
            if path.exists():
                try:
                    existing_text = path.read_text(encoding="utf-8")
                    existing_meta, _ = parse_frontmatter(existing_text)
                    if existing_meta.get("created_at"):
                        meta.setdefault("created_at", existing_meta["created_at"])
                except OSError:
                    pass
            if write_origin:
                meta["origin"] = write_origin
            from core.memory.frontmatter import validate_and_complete_frontmatter

            validate_and_complete_frontmatter(meta, path)
            meta["updated_at"] = now_local().isoformat()
            frontmatter = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
            return f"---\n{frontmatter}---\n\n{body.lstrip()}"

        clean_body = strip_content_frontmatter(content.lstrip())
        timestamp = now_local().isoformat()
        metadata: dict[str, Any] = {
            "confidence": 0.5,
            "created_at": timestamp,
            "updated_at": timestamp,
            "source_episodes": 0,
            "auto_consolidated": False,
            "version": 1,
        }
        if path.exists():
            try:
                existing_text = path.read_text(encoding="utf-8")
                existing_meta, _ = parse_frontmatter(existing_text)
                if existing_meta.get("created_at"):
                    metadata["created_at"] = existing_meta["created_at"]
            except OSError:
                pass
        if write_origin:
            metadata["origin"] = write_origin
        frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        logger.info("Frontmatter parse failed for %s — applied fallback metadata", rel)
        return f"---\n{frontmatter}---\n\n{clean_body.lstrip()}"

    original_created_at = None
    if path.exists():
        try:
            existing_text = path.read_text(encoding="utf-8")
            existing_meta, _ = parse_frontmatter(existing_text)
            original_created_at = existing_meta.get("created_at")
        except OSError:
            pass
    timestamp = now_local().isoformat()
    metadata = {
        "confidence": 0.5,
        "created_at": original_created_at or timestamp,
        "updated_at": timestamp,
        "source_episodes": 0,
        "auto_consolidated": False,
        "version": 1,
    }
    if write_origin:
        metadata["origin"] = write_origin
    clean_body = strip_content_frontmatter(content)
    frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
    return f"---\n{frontmatter}---\n\n{clean_body}"


def _append_memory_content(path: Path, content: str) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(content)


def _append_knowledge_content(path: Path, content: str, write_origin: str) -> None:
    """Append knowledge content and retain the most conservative source origin."""
    _append_memory_content(path, content)
    if not write_origin or not path.is_file():
        return

    from core.memory.frontmatter import parse_frontmatter
    from core.trust import ORIGIN_TRUST_MAP, TRUST_RANK

    current_text = path.read_text(encoding="utf-8")
    if not current_text.startswith("---"):
        return
    current_meta, current_body = parse_frontmatter(current_text)
    if not current_meta:
        return

    current_origin = current_meta.get("origin")
    current_trust = ORIGIN_TRUST_MAP.get(str(current_origin), "untrusted")
    next_trust = ORIGIN_TRUST_MAP.get(write_origin, "untrusted")
    current_rank = TRUST_RANK.get(current_trust, 0)
    next_rank = TRUST_RANK.get(next_trust, 0)
    # Both mixed and external_web currently map to untrusted; prefer the
    # more specific external_web label on a tie.
    should_downgrade = (
        not current_origin or current_rank > next_rank or (current_origin == "mixed" and write_origin == "external_web")
    )
    if should_downgrade:
        import yaml

        current_meta["origin"] = write_origin
        frontmatter = yaml.dump(current_meta, default_flow_style=False, allow_unicode=True)
        path.write_text(f"---\n{frontmatter}---\n\n{current_body.lstrip()}", encoding="utf-8")


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
    _PREFIX_DIRS = ("common_knowledge/", "reference/", "common_skills/", "companies/", "external/")
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

    from core.org.company_resources import company_resource_pointer, get_company_resources

    company_resources = get_company_resources(anima_dir)
    if company_resources is not None and resolved.is_relative_to(company_resources.root):
        pointer = company_resource_pointer(resolved)
        if pointer is not None:
            return _PathNormResult(rel=pointer)

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
    _state_file_lock: StateFileLock | None
    _on_schedule_changed: Callable[[str], None] | None
    _min_trust_seen: int
    _read_paths: set[str]

    _USED_COLLECTION_PREFIXES: dict[str, str] = {
        "knowledge/": "{anima}_knowledge",
        "episodes/": "{anima}_episodes",
        "procedures/": "{anima}_procedures",
        "skills/": "{anima}_skills",
        "common_knowledge/": "shared_common_knowledge",
        "common_skills/": "shared_common_skills",
    }

    def _record_memory_file_used(self: _MemoryToolsHost, rel: str) -> None:
        """Best-effort explicit-use accounting for indexed memory files."""
        collection = self._collection_for_memory_file(rel)
        if collection is None:
            return
        try:
            indexer = self._memory._get_indexer()
            if indexer is None:
                return
            vector_store = indexer.vector_store
            hits = vector_store.get_by_metadata(collection, {"source_file": rel}, limit=10_000)
            if not hits:
                return

            from core.memory.rag.retriever import MemoryRetriever, RetrievalResult

            rag_results = [
                RetrievalResult(
                    doc_id=hit.document.id,
                    content=hit.document.content,
                    score=hit.score,
                    metadata=dict(hit.document.metadata),
                    source_scores={},
                )
                for hit in hits
            ]
            retriever = MemoryRetriever(vector_store, indexer, self._anima_dir / "knowledge")
            retriever.record_access(rag_results, self._current_anima_name(), kind="used")
        except Exception:
            logger.debug("Failed to record explicit memory use for %s", rel, exc_info=True)

    def _collection_for_memory_file(self: _MemoryToolsHost, rel: str) -> str | None:
        for prefix, template in self._USED_COLLECTION_PREFIXES.items():
            if rel.startswith(prefix):
                return template.format(anima=self._current_anima_name())
        return None

    def _current_anima_name(self: _MemoryToolsHost) -> str:
        return str(getattr(self, "_anima_name", "") or self._anima_dir.name)

    def _memory_source_is_denied(self: _MemoryToolsHost, source: str, denied_roots: tuple[Path, ...]) -> bool:
        """Return whether a persisted search hit originated below an explicit deny root."""
        return not memory_source_is_allowed(self._anima_dir, source, denied_roots)

    def _update_longterm_bm25_source(self: _MemoryToolsHost, rel: str) -> None:
        if not rel.startswith(("knowledge/", "episodes/", "procedures/")):
            return
        try:
            from core.memory.retrieval.bm25 import update_longterm_bm25_source

            update_longterm_bm25_source(self._anima_dir, rel)
        except Exception:
            logger.debug("Failed to update long-term BM25 index after memory write: %s", rel, exc_info=True)

    def _anima_search_hint(self: _MemoryToolsHost, query: str) -> str | None:
        """If query looks like a search for a registered Anima, return a redirect hint.

        Checks all anima directories and config aliases so that queries like
        'kanna', '環奈', 'kanna is alive?' all produce a helpful redirect to
        ping_subordinate instead of silently returning empty results.
        """
        try:
            animas_dir = self._anima_dir.parent
            query_lower = query.lower()
            matched_name: str | None = None

            # Pass 1: check anima directory names
            for d in sorted(animas_dir.iterdir()):
                if not d.is_dir() or d.name == self._anima_name:
                    continue
                if not (d / IDENTITY_FILE).exists():
                    continue
                if d.name.lower() in query_lower:
                    matched_name = d.name
                    break

            # Pass 2: check aliases from config
            if matched_name is None:
                try:
                    from core.config import load_config

                    config = load_config()
                    for name, cfg in config.animas.items():
                        if name == self._anima_name:
                            continue
                        for alias in cfg.aliases or []:
                            if alias and alias.lower() in query_lower:
                                matched_name = name
                                break
                        if matched_name:
                            break
                except Exception:
                    logger.debug("handler_memory operation failed", exc_info=True)

            if matched_name:
                return (
                    f"[重要] '{matched_name}' はメモリファイルではなく、"
                    f"登録済みの Anima（AIエージェント）です。"
                    f"メモリを検索しても '{matched_name}' の情報は見つかりません。\n"
                    f"稼働状態を確認するには ping_subordinate(name='{matched_name}') を呼び出してください。"
                )
        except Exception:
            logger.debug("handler_memory read failed", exc_info=True)
        return None

    def _handle_search_memory(self: _MemoryToolsHost, args: dict[str, Any]) -> str:
        scope = args.get("scope", "all")
        query = args.get("query", "")
        offset = int(args.get("offset", 0))
        project = args["project"] if "project" in args else getattr(self, "_default_project", None)
        if project is not None and (not isinstance(project, str) or not _PROJECT_NAME_RE.fullmatch(project)):
            return _error_result(
                "InvalidArguments",
                "project must contain only letters, numbers, underscores, or hyphens",
            )
        if scope == "code":
            if not project:
                return t("handler.code_search_requires_project")
            from core.memory.retrieval.code_index import search_code

            code_results = search_code(self._anima_dir, project, query, limit=offset + 10)
            if isinstance(code_results, str):
                return code_results
            results = code_results[offset:]
            return self._format_search_results(query, scope, offset, results)
        time_range = args.get("time_range") or {}
        if not isinstance(time_range, dict):
            time_range = {}
        time_start = time_range.get("after") if time_range else None
        time_end = time_range.get("before") if time_range else None
        if getattr(self, "_superuser", False):
            denied_roots = ()
        elif hasattr(self, "_load_permissions_config") and hasattr(self, "_resolved_file_deny_roots"):
            config = self._load_permissions_config()
            denied_roots = self._resolved_file_deny_roots(config)
        else:
            # Keep MemoryToolsMixin usable in isolation for compatibility.
            anima_dir = getattr(self, "_anima_dir", None)
            denied_roots = load_denied_roots(Path(anima_dir)) if anima_dir is not None else ()
        if time_start is not None and not isinstance(time_start, str):
            time_start = str(time_start)
        if time_end is not None and not isinstance(time_end, str):
            time_end = str(time_end)

        # If the query seems to be about a registered Anima, redirect immediately.
        anima_hint = self._anima_search_hint(query)

        legacy_time_range: dict[str, str] = {}
        if time_start is not None:
            legacy_time_range["time_start"] = time_start
        if time_end is not None:
            legacy_time_range["time_end"] = time_end
        results = self._memory.search_memory_text(
            query,
            scope=scope,
            offset=offset,
            context_window=getattr(self, "_context_window", _SEARCH_CONTEXT_BASE),
            **legacy_time_range,
        )
        if denied_roots:
            results = [
                result
                for result in results
                if not self._memory_source_is_denied(str(result.get("source_file", "")), denied_roots)
            ]
        if project:
            results = [
                result for result in results if _source_is_in_project(str(result.get("source_file", "")), project)
            ]
        return self._format_search_results(query, scope, offset, results, anima_hint=anima_hint)

    def _format_search_results(
        self: _MemoryToolsHost,
        query: str,
        scope: str,
        offset: int,
        results: list[dict[str, Any]],
        *,
        anima_hint: str | None = None,
    ) -> str:
        logger.debug(
            "search_memory query=%s scope=%s offset=%d results=%d",
            query,
            scope,
            offset,
            len(results),
        )
        if not results:
            if offset > 0:
                base = f"No more results for '{query}' at offset={offset}."
            else:
                base = f"No results for '{query}'"
            if anima_hint:
                return f"{base}\n\n{anima_hint}"
            return base

        scale = min(1.0, getattr(self, "_context_window", _SEARCH_CONTEXT_BASE) / _SEARCH_CONTEXT_BASE)
        max_tokens = int(_SEARCH_MAX_TOKENS * scale)
        max_lines = int(_SEARCH_MAX_LINES * scale)

        search_method = results[0].get("search_method", "vector") if results else "vector"
        header = f'Search results for "{query}" ({search_method}, {scope}, {offset + 1}-{offset + len(results)}):\n'
        meta = getattr(self._memory, "last_search_meta", {}) or {}
        if meta.get("low_confidence"):
            header = header.strip() + " [low-confidence]\n"

        output_parts: list[str] = [header]
        total_tokens = len(header) // 4
        total_lines = header.count("\n") + 1

        for shown_count, r in enumerate(results):
            source = r.get("source_file", "unknown")
            score = r.get("score", 0.0)
            chunk_idx = r.get("chunk_index", 0)
            total_chunks = r.get("total_chunks", 1)
            content = r.get("content", "")
            metadata_line = format_result_metadata_line(r)
            if r.get("last_scan"):
                metadata_line = f"indexed: {r['last_scan']}"

            entry_header = (
                f"[{offset + shown_count + 1}] score={score:.2f} | {source} | chunk {chunk_idx + 1}/{total_chunks}"
            )
            if metadata_line:
                entry = f"\n{entry_header}\n{metadata_line}\n{content}\n"
            else:
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

        result = "".join(output_parts)
        if anima_hint:
            result = f"{anima_hint}\n\n{result}"
        return result

    @staticmethod
    def _is_skill_path(rel: str) -> bool:
        """Return True if *rel* points to a skill or procedure file."""
        if MemoryToolsMixin._is_flat_personal_skill_path(rel):
            return True
        if rel.startswith("skills/") and "SKILL.md" in rel:
            return True
        if rel.startswith("common_skills/") and "SKILL.md" in rel:
            return True
        if rel.startswith("companies/") and "/skills/" in rel and rel.endswith("/SKILL.md"):
            return True
        if rel.startswith("external/") and "SKILL.md" in rel:
            return True
        return rel.startswith("procedures/") and rel.endswith(".md")

    @staticmethod
    def _is_flat_personal_skill_path(rel: str) -> bool:
        parts = Path(rel).parts
        return len(parts) == 2 and parts[0] == "skills" and parts[1].endswith(".md")

    @staticmethod
    def _resolve_external(rel: str):
        """Resolve ``external/<engine>/<name>/<rest...>`` to a real file.

        Returns ``(D, real_path)`` where *D* is the resolved real skill dir and
        *real_path* is the resolved file within *D* (guaranteed to be a
        descendant of *D*). Returns the string ``"traversal"`` when the path
        escapes *D*. Returns ``None`` when the engine is unknown, the named
        skill dir doesn't exist, the name is a dot-entry, or there is no
        trailing path.
        """
        parts = rel.split("/")
        if len(parts) < 4:
            return None
        engine, name = parts[1], parts[2]
        rest = "/".join(parts[3:])
        if not name or name.startswith(".") or not rest:
            return None
        try:
            from core.config.models import load_config

            roots = list(load_config().skills.external_roots)
        except Exception:
            return None
        D: Path | None = None
        for root in roots:
            if not getattr(root, "enabled", True):
                continue
            if getattr(root, "engine", "") != engine:
                continue
            try:
                rdir = Path(root.path).expanduser().resolve()
            except OSError:
                continue
            D = (rdir / name).resolve()
            break
        if D is None or not D.is_dir():
            return None
        real = (D / rest).resolve()
        if not real.is_relative_to(D):
            return "traversal"
        return (D, real)

    def _record_skill_view_if_applicable(self: _MemoryToolsHost, rel: str) -> None:
        """Record a 'view' event if the path looks like a skill or procedure."""
        is_flat_personal_skill = self._is_flat_personal_skill_path(rel)
        is_skill = is_flat_personal_skill or (rel.startswith("skills/") and "SKILL.md" in rel)
        is_common_skill = rel.startswith("common_skills/") and "SKILL.md" in rel
        is_external_skill = rel.startswith("external/") and "SKILL.md" in rel
        is_procedure = rel.startswith("procedures/") and rel.endswith(".md")

        if not (is_skill or is_common_skill or is_procedure or is_external_skill):
            return

        try:
            from core.skills.models import SkillUsageEventType
            from core.skills.usage import SkillUsageTracker

            tracker = SkillUsageTracker(self._anima_dir)
            is_common = is_common_skill

            if is_flat_personal_skill:
                skill_name = Path(rel).stem
            elif is_skill:
                skill_name = Path(rel).parent.name
            elif is_common_skill:
                parts = rel.split("/")
                skill_name = parts[-2] if len(parts) >= 3 else parts[-1].replace(".md", "")
            elif is_external_skill:
                parts = rel.split("/")
                skill_name = parts[-2] if len(parts) >= 4 else parts[-1].replace(".md", "")
            else:
                skill_name = Path(rel).stem

            tracker.record(
                skill_name,
                SkillUsageEventType.view,
                is_common=is_common,
                is_procedure=is_procedure,
                ref=rel,
            )
        except Exception:
            logger.debug("Failed to record skill view event for %s", rel, exc_info=True)

    def _handle_read_memory_file(self: _MemoryToolsHost, args: dict[str, Any]) -> str:
        raw_path = args["path"]
        norm = _normalize_memory_path(raw_path, self._anima_dir)
        if norm.channel_redirect:
            err = self._check_file_permission(raw_path)
            if err:
                return err
            return self._handle_read_channel({"channel": norm.channel_redirect})
        rel = args["path"] = norm.rel

        redirect_note = ""
        legacy_identity_read = _legacy_workspace_name(rel, {LEGACY_IDENTITY}) is not None
        if legacy_identity_read:
            rel = args["path"] = _replace_workspace_basename(rel, IDENTITY_FILE)
        else:
            legacy_name = _legacy_workspace_name(rel, frozenset(LEGACY_RENAMES))
            if legacy_name is not None:
                redirect_note = t(
                    "handler.memory_file_renamed",
                    old=legacy_name,
                    new=LEGACY_RENAMES[legacy_name],
                )
                rel = args["path"] = _replace_workspace_basename(rel, LEGACY_RENAMES[legacy_name])

        if rel == "state/current_task.md":
            rel = args["path"] = "state/current_state.md"

        # Support common_knowledge/ prefix — resolve to shared dir
        ext_origin: Path | None = None
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
        elif rel.startswith("companies/"):
            from core.org.company_resources import get_company_resources

            resources = get_company_resources(self._anima_dir)
            path = (resources.root.parent.parent / rel).resolve() if resources is not None else None
            if path is None or not path.is_relative_to(resources.root):
                return _error_result(
                    "PermissionDenied",
                    "Company resource access is limited to your assigned company.",
                )
        elif rel.startswith("external/"):
            ext = self._resolve_external(rel)
            if ext == "traversal":
                return _error_result(
                    "PermissionDenied",
                    "Path traversal detected — access denied.",
                )
            if ext is None:
                path = self._anima_dir / rel
                resolved = path.resolve()
            else:
                path = ext[1]
                resolved = path
                ext_origin = ext[0]
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
        err = self._check_file_permission(str(path))
        if err:
            return err
        soul_path = path.with_name(SOUL_FILE) if legacy_identity_read else None
        if (path.exists() and path.is_file()) or (soul_path is not None and soul_path.is_file()):
            # Block loading of skills with blocked trust_level or dangerous scan verdict
            if self._is_skill_path(rel):
                try:
                    from core.skills.loader import load_skill_metadata, skill_access_decision

                    skill_meta = load_skill_metadata(path)
                    allowed, reason = skill_access_decision(skill_meta, anima_dir=self._anima_dir)
                    if not allowed:
                        return _error_result(
                            "SkillBlocked",
                            f"Skill '{skill_meta.name}' is blocked "
                            f"(reason={reason}). "
                            "Contact your supervisor if you believe this is an error.",
                        )
                except ImportError:
                    pass
                except Exception:
                    logger.debug("Failed to check skill block status for %s", rel, exc_info=True)

            logger.debug("read_memory_file path=%s", rel)
            self._read_paths.add(rel)

            # Record skill view event
            self._record_skill_view_if_applicable(rel)
            self._record_memory_file_used(rel)

            content = path.read_text(encoding="utf-8") if path.is_file() else ""
            if soul_path is not None:
                soul_content = soul_path.read_text(encoding="utf-8") if soul_path.is_file() else ""
                content = "\n\n".join(part for part in (content, soul_content) if part)
                self._read_paths.add(rel)
                self._read_paths.add(_replace_workspace_basename(rel, SOUL_FILE))
            if ext_origin is not None and rel.endswith("SKILL.md"):
                content = f"> Real directory: {ext_origin}\n{content}"
            if redirect_note:
                content = f"{content}\n\n{redirect_note}" if content else redirect_note
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

    def _resolve_write_origin(self: _MemoryToolsHost) -> str:
        """Return the conservative origin for knowledge written this session."""
        from core.trust import read_session_trust

        min_trust = getattr(self, "_min_trust_seen", 2)
        runtime_context = getattr(self, "_runtime_session_context", None)
        tool_session_id = getattr(runtime_context, "tool_session_id", "")
        if tool_session_id:
            min_trust = min(min_trust, read_session_trust(self._anima_dir, tool_session_id))
        return {0: "external_web", 1: "mixed"}.get(min_trust, "")

    def _write_plain_memory_file(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> _MemoryWriteOutcome:
        if request.mode == "append":
            _append_memory_content(request.path, request.content)
        else:
            request.path.write_text(request.content, encoding="utf-8")
        return _MemoryWriteOutcome()

    def _write_knowledge_memory_file(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> _MemoryWriteOutcome:
        if request.mode == "overwrite" and request.rel.endswith(".md"):
            request.path.write_text(
                _knowledge_frontmatter_text(request.path, request.rel, request.content, request.write_origin),
                encoding="utf-8",
            )
            return _MemoryWriteOutcome(auto_frontmatter_applied=True)
        if request.mode == "append":
            _append_knowledge_content(request.path, request.content, request.write_origin)
            return _MemoryWriteOutcome()
        return self._write_plain_memory_file(request)

    def _write_procedure_memory_file(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> _MemoryWriteOutcome:
        if (
            request.mode == "overwrite"
            and request.rel.endswith(".md")
            and not request.content.lstrip().startswith("---")
        ):
            metadata = {
                "description": _extract_first_heading(request.content),
                "success_count": 0,
                "failure_count": 0,
                "confidence": 0.5,
            }
            self._memory.write_procedure_with_meta(request.path, request.content, metadata)
            return _MemoryWriteOutcome(auto_frontmatter_applied=True)
        return self._write_plain_memory_file(request)

    def _write_episode_memory_file(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> _MemoryWriteOutcome:
        from core.platform.locks import locked_path

        lock_path = request.path.with_name(f"{request.path.name}.lock")
        with locked_path(lock_path, exclusive=True, thread_lock=True):
            if request.mode == "overwrite" and request.path.exists():
                try:
                    archive_episode_before_write(self._anima_dir, request.path)
                except OSError as exc:
                    logger.warning("Failed to archive episode before overwrite: %s", request.path, exc_info=True)
                    return _MemoryWriteOutcome(
                        error=_error_result(
                            "WriteError",
                            f"Failed to archive existing episode before overwrite: {exc}",
                        )
                    )
            if request.mode == "overwrite":
                from core.platform.atomic_io import atomic_write_text

                atomic_write_text(request.path, request.content)
                return _MemoryWriteOutcome()
            return self._write_plain_memory_file(request)

    def _write_root_prompt_setting(
        self: _MemoryToolsHost,
        target_name: str,
        setting: str,
        content: str,
        mode: str,
    ) -> str:
        """Persist bootstrap/supervisor prompt settings through their root owner."""
        if not isinstance(mode, str) or mode not in {"overwrite", "append"}:
            return _error_result("InvalidArguments", "Root-owned settings support overwrite or append only")
        target_dir = self._anima_dir.parent / target_name
        filename = _PROMPT_SETTING_FILES.get(setting)
        if filename is None:
            return _error_result("InvalidArguments", f"Unknown prompt setting: {setting}")
        from core.platform.process_role import get_process_role

        role = get_process_role()
        from core.anima.settings_store import settings_server_running

        if role == "root" or (role == "cli" and not settings_server_running()):
            try:
                if mode == "append":
                    path = target_dir / filename
                    content = (path.read_text(encoding="utf-8") if path.is_file() else "") + content
                from core.anima.settings_store import write_agents, write_identity, write_soul

                writer = {
                    "identity": write_identity,
                    "soul": write_soul,
                    "agents": write_agents,
                    "injection": write_agents,
                }[setting]
                writer(target_dir, content)
            except Exception as exc:
                return _error_result("WriteError", f"Failed to update {filename}: {exc}")
            self._activity.log("memory_write", summary=f"../{target_name}/{filename} (offline root settings)")
            return f"Written to ../{target_name}/{filename} through the offline root settings store"

        from urllib.parse import quote

        from core.host_api import response_detail
        from core.internal_api import host_api

        try:
            response = host_api.post(
                f"/api/internal/animas/{quote(target_name, safe='')}/prompt-settings",
                json={"setting": setting, "content": content, "mode": mode},
                timeout=30.0,
            )
        except Exception as exc:
            logger.warning("Root prompt-settings API failed for %s/%s", target_name, setting, exc_info=True)
            return _error_result("HostAPIError", f"Root settings API unavailable: {exc}")
        if response.status_code >= 400:
            error_type = {
                400: "InvalidArguments",
                401: "PermissionDenied",
                403: "PermissionDenied",
                404: "FileNotFound",
            }.get(
                response.status_code,
                "HostAPIError",
            )
            return _error_result(error_type, response_detail(response))
        self._activity.log("memory_write", summary=f"../{target_name}/{filename} (root API)")
        return f"Written to ../{target_name}/{filename} through the root settings API"

    def _write_memory_scope(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> _MemoryWriteOutcome:
        scope = next(
            (scope for prefix, scope in _MEMORY_WRITE_SCOPE_PREFIXES if request.rel.startswith(prefix)),
            "default",
        )
        writer_name = _MEMORY_WRITE_HANDLERS[scope]
        return getattr(self, writer_name)(request)

    def _handle_write_memory_file(self: _MemoryToolsHost, args: dict[str, Any]) -> str:
        raw_path = args["path"]
        norm = _normalize_memory_path(raw_path, self._anima_dir)
        if norm.channel_redirect:
            err = self._check_file_permission(raw_path, write=True)
            if err:
                return err
            return self._handle_post_channel(
                {
                    "channel": norm.channel_redirect,
                    "content": args.get("content", ""),
                }
            )
        rel = args["path"] = norm.rel

        redirect_note = ""
        if _legacy_workspace_name(rel, {LEGACY_IDENTITY}) is not None:
            return _error_result(
                "InvalidArguments",
                t("handler.identity_write_rejected", old=LEGACY_IDENTITY),
            )
        legacy_name = _legacy_workspace_name(rel, frozenset(LEGACY_RENAMES))
        if legacy_name is not None:
            redirect_note = t(
                "handler.memory_file_renamed",
                old=legacy_name,
                new=LEGACY_RENAMES[legacy_name],
            )
            rel = args["path"] = _replace_workspace_basename(rel, LEGACY_RENAMES[legacy_name])

        if rel == "state/current_task.md":
            rel = args["path"] = "state/current_state.md"

        if rel.startswith("reference/"):
            return _error_result(
                "PermissionDenied",
                "reference/ is read-only. Use common_knowledge/ for shared writable documents.",
            )
        if rel.startswith("companies/"):
            return _error_result(
                "PermissionDenied",
                "Company knowledge and skills are read-only.",
            )
        if rel.startswith("external/"):
            return _error_result(
                "PermissionDenied",
                "external/ is read-only. Create new skills under skills/ or common_skills/.",
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

        mode = args.get("mode", "overwrite")
        content = args.get("content", "")
        if not rel.startswith(("common_knowledge/", "common_skills/")):
            try:
                from core.paths import get_animas_dir

                anima_root = get_animas_dir().resolve()
                target_path = path.resolve()
                target_relative = target_path.relative_to(anima_root)
            except (OSError, RuntimeError, ValueError):
                target_relative = None
            if target_relative is not None and len(target_relative.parts) == 2:
                target_name, filename = target_relative.parts
                if filename in _PROMPT_SETTINGS_BY_FILE:
                    if "content" not in args or not isinstance(content, str):
                        return _error_result("InvalidArguments", "content must be a string")
                    setting = _PROMPT_SETTINGS_BY_FILE[filename]
                    result = self._write_root_prompt_setting(target_name, setting, content, mode)
                    if redirect_note:
                        result = f"{result}\n\n{redirect_note}"
                    return result

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

        err = self._check_file_permission(str(path), write=True)
        if err:
            return err

        # Tool creation permission check
        if rel.startswith("tools/") and rel.endswith(".py"):
            if not self._check_tool_creation_permission("個人ツール"):
                return _error_result(
                    "PermissionDenied",
                    t("handler.tool_creation_denied"),
                )

        _was_existing = path.exists()

        from core.skills.ledger import capture_skill_document, is_skill_document_path

        _is_skill_document = is_skill_document_path(path, self._anima_dir)

        # ── Read-before-write guard ──
        _rbw_skip = mode == "append" or not _was_existing or rel.startswith(("episodes/", "state/", "shortterm/"))
        if not _rbw_skip and rel not in self._read_paths:
            try:
                _existing = path.read_text(encoding="utf-8")[:2000]
            except OSError:
                _existing = "(could not read existing content)"
            message_key = "handler.skill_read_before_write" if _is_skill_document else "handler.read_before_write"
            return _error_result(
                "ReadBeforeWrite",
                t(message_key, path=rel, existing=_existing),
            )

        _skill_capture = capture_skill_document(path, self._anima_dir)
        content = args["content"]
        write_origin = self._resolve_write_origin() if rel.startswith("knowledge/") and rel.endswith(".md") else ""

        path.parent.mkdir(parents=True, exist_ok=True)

        request = _MemoryWriteRequest(
            rel=rel,
            path=path,
            content=content,
            mode=mode,
            was_existing=_was_existing,
            write_origin=write_origin,
            skill_capture=_skill_capture,
        )

        lock = self._state_file_lock if self._state_file_lock and self._is_state_file(path) else None
        if lock:
            lock.acquire()
        try:
            outcome = self._write_memory_scope(request)
            if outcome.error:
                return outcome.error
        finally:
            if lock:
                lock.release()
        result = self._complete_memory_write(request, auto_frontmatter_applied=outcome.auto_frontmatter_applied)
        if redirect_note:
            result = f"{result}\n\n{redirect_note}"
        return result

    def _complete_memory_write(
        self: _MemoryToolsHost,
        request: _MemoryWriteRequest,
        *,
        auto_frontmatter_applied: bool,
    ) -> str:
        self._record_memory_file_change(request)
        similar_hint = self._memory_write_similarity_hint(request)
        self._reload_schedule_if_needed(request.rel)
        result = self._format_memory_write_result(
            request,
            auto_frontmatter_applied=auto_frontmatter_applied,
            similar_hint=similar_hint,
        )
        self._update_memory_write_indexes(request)
        return result

    def _record_memory_file_change(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> None:
        if request.skill_capture is not None:
            from core.skills.ledger import record_skill_change

            record_skill_change(
                request.path,
                request.skill_capture,
                anima_dir=self._anima_dir,
                after_text=request.path.read_text(encoding="utf-8") if request.path.is_file() else None,
                after_exists=request.path.is_file(),
                actor=self._anima_name,
                route="write_memory_file",
                reason=f"mode={request.mode}",
            )

        logger.info("write_memory_file path=%s mode=%s", request.rel, request.mode)
        self._activity.log(
            "memory_write",
            summary=f"{request.rel} ({request.mode})",
            meta={"path": request.rel, "mode": request.mode},
        )

    def _memory_write_similarity_hint(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> str:
        if not (request.rel.startswith("knowledge/") and request.mode == "overwrite" and not request.was_existing):
            return ""

        knowledge_dir = self._anima_dir / "knowledge"
        if not knowledge_dir.is_dir():
            return ""
        new_tokens = set(Path(request.rel).stem.replace("-", "_").split("_"))
        existing_names = [
            file.name
            for file in knowledge_dir.iterdir()
            if file.suffix == ".md" and file.name != Path(request.rel).name
        ]
        similar = [
            name
            for name in existing_names
            if len(new_tokens & set(name.replace("-", "_").replace(".md", "").split("_"))) >= 2
        ]
        if not similar:
            return ""
        similar.sort()
        lines = "\n".join(f"  - {name}" for name in similar[:10])
        return t("handler.similar_knowledge_hint", files=lines)

    def _reload_schedule_if_needed(self: _MemoryToolsHost, rel: str) -> None:
        if rel not in (HEARTBEAT_FILE, "cron.md") or not self._on_schedule_changed:
            return
        try:
            self._on_schedule_changed(self._anima_name)
            logger.info("Schedule reload triggered for '%s'", self._anima_name)
        except Exception:
            logger.exception("Schedule reload failed for '%s'", self._anima_name)

    def _format_memory_write_result(
        self: _MemoryToolsHost,
        request: _MemoryWriteRequest,
        *,
        auto_frontmatter_applied: bool,
        similar_hint: str,
    ) -> str:
        rel = request.rel
        result = f"Written to {rel}"
        if similar_hint:
            result = f"{result}\n\n{similar_hint}"

        if rel.startswith("knowledge/") and _looks_like_case_record(request.content):
            result = f"{result}\n\n{t('handler.case_record_episodes_hint')}"
            logger.info("Knowledge write resembles a case record; suggested episodes destination: %s", rel)

        episode_warning = _validate_episode_path(rel)
        if episode_warning:
            logger.warning("Non-standard episode path: %s", rel)
            result = f"{result}\n\n{episode_warning}"

        if (rel.startswith("skills/") or rel.startswith("common_skills/")) and rel.endswith(".md"):
            validation_msg = _validate_skill_format(request.content)
            if validation_msg:
                result = f"{result}\n\n{t('handler.skill_format_validation', msg=validation_msg)}"

        if rel.startswith("procedures/") and rel.endswith(".md") and not auto_frontmatter_applied:
            validation_msg = _validate_procedure_format(request.content)
            if validation_msg:
                result = f"{result}\n\n{t('handler.procedure_format_validation', msg=validation_msg)}"
        return result

    def _update_memory_write_indexes(self: _MemoryToolsHost, request: _MemoryWriteRequest) -> None:
        rel = request.rel
        path = request.path
        if rel.startswith(("skills/", "procedures/")) and rel.endswith(".md"):
            indexer = self._memory._get_indexer()
            if indexer:
                memory_type = "skills" if rel.startswith("skills/") else "procedures"
                try:
                    indexer.index_file(path, memory_type=memory_type, force=True)
                except Exception as exc:
                    logger.warning("Failed to update RAG index for %s: %s", rel, exc)
            self._update_longterm_bm25_source(rel)

        if rel.startswith("knowledge/") and rel.endswith(".md"):
            origin = request.write_origin
            if not origin:
                try:
                    from core.memory.frontmatter import parse_frontmatter

                    current_meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
                    origin = str(current_meta.get("origin", ""))
                except OSError:
                    origin = ""

            indexer = self._memory._get_indexer()
            if indexer:
                try:
                    indexer.index_file(
                        path,
                        memory_type="knowledge",
                        force=True,
                        origin=origin or None,
                    )
                except Exception as exc:
                    logger.warning("Failed to update RAG index for %s: %s", rel, exc)
            self._update_longterm_bm25_source(rel)

        if rel.startswith("episodes/") and rel.endswith(".md"):
            self._update_longterm_bm25_source(rel)

    def _handle_archive_memory_file(self: _MemoryToolsHost, args: dict[str, Any]) -> str:
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

        err = self._check_file_permission(str(target), write=True)
        if err:
            return err

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
        expected_archive_root = self._anima_dir.resolve() / "archive"
        if not archive_dir.resolve().is_relative_to(expected_archive_root):
            return _error_result(
                "PermissionDenied",
                "Archive destination resolves outside the protected archive directory",
            )
        err = self._check_file_permission(
            str(archive_dir),
            write=True,
            trusted_internal_cache_write=True,
        )
        if err:
            return err
        from core.skills.ledger import capture_skill_document

        skill_capture = capture_skill_document(target, self._anima_dir)
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
        if skill_capture is not None:
            from core.skills.ledger import record_skill_change

            record_skill_change(
                target,
                skill_capture,
                anima_dir=self._anima_dir,
                after_text=None,
                after_exists=False,
                actor=self._anima_name,
                route="archive_memory_file",
                reason=reason,
            )
        self._update_longterm_bm25_source(rel)

        logger.info("archive_memory_file: %s -> %s (reason: %s)", rel, dest.name, reason)

        self._activity.log(
            "memory_write",
            summary=f"archived {rel}: {reason}",
            meta={"path": rel, "reason": reason, "action": "archive"},
        )

        return f"Archived {rel} -> archive/superseded/{dest.name} (reason: {reason})"
