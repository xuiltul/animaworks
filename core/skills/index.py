from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Metadata index over personal skills, common skills, and procedures."""

import logging
from datetime import datetime
from pathlib import Path

from core.skills.loader import load_skill_metadata
from core.skills.models import SkillMetadata, SkillScanVerdict, SkillTrustLevel

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

_EXCLUDED_TRUST_LEVELS: frozenset[SkillTrustLevel] = frozenset({SkillTrustLevel.blocked, SkillTrustLevel.quarantine})


# ── SkillIndex ────────────────────────────────────────────


class SkillIndex:
    """Scan skill and procedure directories and query metadata."""

    def __init__(
        self,
        skills_dir: Path,
        common_skills_dir: Path,
        procedures_dir: Path | None = None,
        *,
        anima_dir: Path | None = None,
    ) -> None:
        """Initialize index roots.

        Args:
            skills_dir: Directory containing per-Anima skill folders with ``SKILL.md``.
            common_skills_dir: Directory containing shared skill folders (flat or nested).
            procedures_dir: Optional directory of procedure ``*.md`` files; ``None`` skips.
            anima_dir: Optional anima directory for usage stats integration.
        """
        self._skills_dir = skills_dir
        self._common_skills_dir = common_skills_dir
        self._procedures_dir = procedures_dir
        self._anima_dir = anima_dir
        self._cached_index: list[SkillMetadata] | None = None
        self._cached_all_entries: list[SkillMetadata] | None = None
        self._curator_state_marker: tuple[int, int] | None = None

    # ── Cache ───────────────────────────────────────────────

    def invalidate(self) -> None:
        """Drop cached scan results so the next access rebuilds from disk."""
        self._cached_index = None
        self._cached_all_entries = None
        self._curator_state_marker = None

    @property
    def all_skills(self) -> list[SkillMetadata]:
        """All indexed skills after trust filtering (same as :meth:`build_index`)."""
        self._invalidate_if_curator_state_changed()
        if self._cached_index is None:
            self.build_index()
        assert self._cached_index is not None
        return self._cached_index

    # ── Index build ───────────────────────────────────────────

    def build_index(self) -> list[SkillMetadata]:
        """Scan configured directories and return trusted skill metadata.

        Skips files that fail to parse. Omits ``blocked`` and ``quarantine`` trust levels.

        Returns:
            Sorted list: personal skills, then common, then procedures.
        """
        entries: list[SkillMetadata] = []
        seen_paths: set[Path] = set()
        curator_state_marker = self._read_curator_state_marker()

        def _add_metadata(meta: SkillMetadata) -> None:
            p = meta.path
            if p is None:
                return
            resolved = p.resolve()
            if resolved in seen_paths:
                return
            seen_paths.add(resolved)
            entries.append(meta)

        if self._skills_dir.exists():
            personal_skill_paths = [
                *sorted(self._skills_dir.glob("*.md")),
                *sorted(self._skills_dir.glob("*/SKILL.md")),
            ]
            for skill_path in personal_skill_paths:
                try:
                    meta = load_skill_metadata(skill_path)
                    meta = meta.model_copy(update={"is_common": False, "is_procedure": False})
                    _add_metadata(meta)
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill metadata from %s: %s",
                        skill_path,
                        exc,
                    )

        if self._common_skills_dir.exists():
            for skill_path in sorted(self._common_skills_dir.glob("*/SKILL.md")):
                try:
                    meta = load_skill_metadata(skill_path)
                    meta = meta.model_copy(update={"is_common": True, "is_procedure": False})
                    _add_metadata(meta)
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill metadata from %s: %s",
                        skill_path,
                        exc,
                    )
            for skill_path in sorted(self._common_skills_dir.glob("*/*/SKILL.md")):
                try:
                    meta = load_skill_metadata(skill_path)
                    meta = meta.model_copy(update={"is_common": True, "is_procedure": False})
                    _add_metadata(meta)
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill metadata from %s: %s",
                        skill_path,
                        exc,
                    )

        if self._procedures_dir is not None and self._procedures_dir.exists():
            for proc_path in sorted(self._procedures_dir.glob("*.md")):
                try:
                    meta = load_skill_metadata(proc_path)
                    meta = meta.model_copy(update={"is_procedure": True, "is_common": False})
                    _add_metadata(meta)
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill metadata from %s: %s",
                        proc_path,
                        exc,
                    )

        sorted_all = sorted(entries, key=self._sort_key)

        # Merge usage stats from SkillUsageTracker if anima_dir is available.
        #
        # Usage frequency policy: ``usage_count`` = view_count + use_count.
        # Currently only ``view`` events are emitted (on read_memory_file).
        # The ``use`` event type is reserved for future Skill-backed Cron
        # (Issue 7) where a cron job explicitly invokes a skill.  Until then,
        # ``view_count + success_count + failure_count`` serves as the
        # effective "how often is this skill actively used?" metric for
        # promotion decisions (Issue 4).
        if self._anima_dir is not None:
            try:
                from core.skills.usage import SkillUsageTracker

                tracker = SkillUsageTracker(self._anima_dir)
                all_stats = tracker.get_all_stats()
                for i, meta in enumerate(sorted_all):
                    stats = all_stats.get(meta.name)
                    if stats:
                        sorted_all[i] = meta.model_copy(
                            update={
                                "usage_count": stats.view_count + stats.use_count,
                                "success_count": stats.success_count,
                                "failure_count": stats.failure_count,
                                "patch_count": stats.patch_count,
                                "last_used_at": (
                                    datetime.fromisoformat(stats.last_used_at) if stats.last_used_at else None
                                ),
                            }
                        )
            except Exception:
                logger.debug("Failed to merge usage stats into index", exc_info=True)

        if self._anima_dir is not None:
            try:
                from core.skills.curator import apply_curator_state, replay_curator_state

                replay = replay_curator_state(self._anima_dir)
                sorted_all = [apply_curator_state(meta, replay) for meta in sorted_all]
            except Exception:
                logger.debug("Failed to merge curator state into index", exc_info=True)

        self._cached_all_entries = sorted_all
        self._curator_state_marker = curator_state_marker
        filtered = [m for m in sorted_all if self._is_catalog_visible(m)]
        self._cached_index = filtered
        return list(filtered)

    def _read_curator_state_marker(self) -> tuple[int, int] | None:
        if self._anima_dir is None:
            return None
        state_path = self._anima_dir / "state" / "skill_curator.jsonl"
        try:
            stat = state_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _invalidate_if_curator_state_changed(self) -> None:
        if self._anima_dir is None or self._cached_all_entries is None:
            return
        if self._read_curator_state_marker() != self._curator_state_marker:
            self.invalidate()

    @staticmethod
    def _is_catalog_visible(meta: SkillMetadata) -> bool:
        if meta.trust_level in _EXCLUDED_TRUST_LEVELS:
            return False
        if meta.security.verdict == SkillScanVerdict.dangerous:
            return False
        try:
            from core.skills.curator import is_unloadable_lifecycle_state

            return not is_unloadable_lifecycle_state(meta.lifecycle_state)
        except Exception:
            return True

    @staticmethod
    def _sort_key(meta: SkillMetadata) -> tuple[int, str, str]:
        """Personal (0), common (1), procedures (2); then name and path."""
        if meta.is_procedure:
            tier = 2
        elif meta.is_common:
            tier = 1
        else:
            tier = 0
        path_s = str(meta.path) if meta.path is not None else ""
        return (tier, meta.name.casefold(), path_s)

    # ── Reference Resolution ───────────────────────────────────

    def resolve_skill_reference(self, ref: str) -> SkillMetadata | None:
        """Resolve a cron skill reference to metadata.

        Supported refs are exact skill/procedure names and safe ``SKILL.md``
        pointers under ``skills/`` or ``common_skills/``.  Name matches use the
        index order, so personal skills win over common skills, which win over
        procedures.
        """
        value = str(ref).strip()
        if not self._is_safe_reference(value):
            return None

        entries = self.search("", include_blocked=True)
        pointer_path = self._path_from_pointer(value)
        if pointer_path is not None:
            existing = self._entry_for_path(entries, pointer_path)
            if existing is not None:
                return existing
            if not pointer_path.is_file():
                return None
            try:
                meta = load_skill_metadata(pointer_path)
            except Exception as exc:
                logger.warning("Failed to load skill reference metadata from %s: %s", pointer_path, exc)
                return None
            return meta.model_copy(
                update={
                    "is_common": self._is_under(pointer_path, self._common_skills_dir),
                    "is_procedure": False,
                }
            )

        matches = [
            meta for meta in entries if meta.name == value or (meta.path is not None and meta.path.parent.name == value)
        ]
        if len(matches) > 1:
            logger.warning(
                "Multiple skill references matched %r; using %s by deterministic priority",
                value,
                matches[0].path,
            )
        return matches[0] if matches else None

    @staticmethod
    def _is_safe_reference(ref: str) -> bool:
        if not ref or "\\" in ref:
            return False
        path = Path(ref)
        if path.is_absolute() or ".." in path.parts:
            return False
        if ref.startswith(("skills/", "common_skills/")):
            return len(path.parts) >= 3 and path.name == "SKILL.md"
        return "/" not in ref

    def _path_from_pointer(self, ref: str) -> Path | None:
        path = Path(ref)
        if ref.startswith("skills/"):
            return self._safe_child_path(self._skills_dir.parent, path)
        if ref.startswith("common_skills/"):
            return self._safe_child_path(self._common_skills_dir.parent, path)
        return None

    @staticmethod
    def _safe_child_path(root: Path, relative: Path) -> Path | None:
        if relative.name != "SKILL.md" or ".." in relative.parts:
            return None
        try:
            root_resolved = root.resolve(strict=False)
            candidate = (root / relative).resolve(strict=False)
            candidate.relative_to(root_resolved)
            return candidate
        except (OSError, ValueError):
            return None

    @staticmethod
    def _entry_for_path(entries: list[SkillMetadata], path: Path) -> SkillMetadata | None:
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            return None
        for meta in entries:
            if meta.path is None:
                continue
            try:
                if meta.path.resolve(strict=False) == resolved:
                    return meta
            except OSError:
                continue
        return None

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(root.resolve(strict=False))
            return True
        except (OSError, ValueError):
            return False

    # ── Search ────────────────────────────────────────────────

    def search(self, query: str, *, include_blocked: bool = False) -> list[SkillMetadata]:
        """Return metadata entries matching *query* as a case-insensitive substring.

        Matches against ``name``, ``description``, and ``category`` (when set).

        Args:
            query: Substring to match.
            include_blocked: When ``False``, exclude ``blocked`` / ``quarantine`` entries.

        Returns:
            Filtered list in personal → common → procedure order.
        """
        self._invalidate_if_curator_state_changed()
        if self._cached_all_entries is None:
            self.build_index()
        assert self._cached_all_entries is not None
        base = self._cached_all_entries if include_blocked else self.all_skills
        if not query:
            return list(base)
        q = query.casefold()

        def _matches(meta: SkillMetadata) -> bool:
            cat = meta.category
            return (
                q in meta.name.casefold()
                or q in meta.description.casefold()
                or (cat is not None and q in cat.casefold())
            )

        return [m for m in base if _matches(m)]
