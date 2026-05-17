from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic skill lifecycle curator."""

import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.skills.models import (
    SkillCuratorEvent,
    SkillCuratorEventType,
    SkillLifecycleState,
    SkillMetadata,
    SkillScanVerdict,
    SkillTrustLevel,
)
from core.time_utils import now_iso

logger = logging.getLogger(__name__)

CURATOR_STATE_FILE = "skill_curator.jsonl"
UNLOADABLE_LIFECYCLE_STATES = frozenset(
    {
        SkillLifecycleState.archived,
        SkillLifecycleState.blocked,
        SkillLifecycleState.deleted,
    }
)


@dataclass(slots=True)
class CuratorReplay:
    """Replay result from append-only curator state."""

    states: dict[str, SkillLifecycleState] = field(default_factory=dict)
    absorbed_into: dict[str, str | None] = field(default_factory=dict)
    events: list[SkillCuratorEvent] = field(default_factory=list)

    def state_for(self, skill_name: str) -> SkillLifecycleState:
        return self.states.get(skill_name, SkillLifecycleState.active)


@dataclass(slots=True)
class LifecycleSuggestion:
    skill_name: str
    suggested_state: SkillLifecycleState
    reason: str
    metric: float | int | str | None = None


@dataclass(slots=True)
class DuplicateCandidate:
    skill_name: str
    related_skill: str
    score: float
    signals: list[str] = field(default_factory=list)


def is_unloadable_lifecycle_state(state: SkillLifecycleState | str | None) -> bool:
    if state is None:
        return False
    try:
        resolved = state if isinstance(state, SkillLifecycleState) else SkillLifecycleState(str(state))
    except ValueError:
        return False
    return resolved in UNLOADABLE_LIFECYCLE_STATES


def replay_curator_state(anima_dir: Path) -> CuratorReplay:
    """Replay ``state/skill_curator.jsonl`` without constructing a curator."""
    return SkillCurator(anima_dir).replay_state()


def apply_curator_state(meta: SkillMetadata, replay: CuratorReplay) -> SkillMetadata:
    state = replay.state_for(meta.name)
    return meta.model_copy(
        update={
            "lifecycle_state": state,
            "absorbed_into": replay.absorbed_into.get(meta.name),
        }
    )


def curator_allows_access(
    meta: SkillMetadata,
    *,
    anima_dir: Path | None = None,
    replay: CuratorReplay | None = None,
) -> tuple[bool, str]:
    """Return whether a skill can be loaded, plus a machine-readable reason."""
    if meta.trust_level == SkillTrustLevel.blocked:
        return False, "trust_level_blocked"
    if meta.trust_level == SkillTrustLevel.quarantine:
        return False, "trust_level_quarantine"
    if meta.security.verdict == SkillScanVerdict.dangerous:
        return False, "security_dangerous"
    state = meta.lifecycle_state
    if replay is not None:
        state = replay.state_for(meta.name)
    elif anima_dir is not None:
        try:
            state = replay_curator_state(anima_dir).state_for(meta.name)
        except Exception:
            logger.debug("Failed to replay curator state for %s", meta.name, exc_info=True)
    if is_unloadable_lifecycle_state(state):
        state_value = state.value if isinstance(state, SkillLifecycleState) else str(state)
        return False, f"curator_{state_value}"
    return True, "allowed"


class SkillCurator:
    """Manage skill lifecycle state through append-only events."""

    def __init__(
        self,
        anima_dir: Path,
        *,
        common_skills_dir: Path | None = None,
    ) -> None:
        self.anima_dir = anima_dir
        self.skills_dir = anima_dir / "skills"
        self.common_skills_dir = common_skills_dir
        self.state_path = anima_dir / "state" / CURATOR_STATE_FILE
        self.proposal_dir = anima_dir / "state" / "skill_curator" / "proposals"

    def replay_state(self) -> CuratorReplay:
        replay = CuratorReplay()
        if not self.state_path.exists():
            return replay
        for raw in self.state_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = SkillCuratorEvent.model_validate(json.loads(raw))
            except Exception:
                logger.warning("Skipping malformed skill curator event: %s", raw[:120])
                continue
            replay.events.append(event)
            if event.event_type == SkillCuratorEventType.state_changed and event.to_state is not None:
                replay.states[event.skill_name] = event.to_state
                replay.absorbed_into[event.skill_name] = event.absorbed_into
        return replay

    def append_event(self, event: SkillCuratorEvent) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def change_state(
        self,
        skill_name: str,
        to_state: SkillLifecycleState | str,
        *,
        reason: str,
        actor: str = "curator",
        absorbed_into: str | None = None,
        create_reference_proposal: bool = True,
    ) -> SkillCuratorEvent:
        to_state = to_state if isinstance(to_state, SkillLifecycleState) else SkillLifecycleState(to_state)
        replay = self.replay_state()
        from_state = replay.state_for(skill_name)
        if to_state == SkillLifecycleState.active:
            self._assert_can_restore(skill_name)

        proposal_path = None
        if create_reference_proposal and to_state in UNLOADABLE_LIFECYCLE_STATES:
            proposal_path = self._create_reference_rewrite_proposal(
                skill_name,
                absorbed_into=absorbed_into,
                actor=actor,
            )

        event = SkillCuratorEvent(
            ts=now_iso(),
            event_type=SkillCuratorEventType.state_changed,
            skill_name=skill_name,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            actor=actor,
            absorbed_into=absorbed_into,
            proposal_path=proposal_path,
        )
        self.append_event(event)
        self._invalidate_rag_cache(skill_name)
        self._purge_personal_skill_vectors(skill_name)
        return event

    def archive_skill(
        self,
        skill_name: str,
        *,
        reason: str,
        actor: str = "curator",
        absorbed_into: str | None = None,
    ) -> SkillCuratorEvent:
        return self.change_state(
            skill_name,
            SkillLifecycleState.archived,
            reason=reason,
            actor=actor,
            absorbed_into=absorbed_into,
        )

    def restore_skill(self, skill_name: str, *, reason: str, actor: str = "curator") -> SkillCuratorEvent:
        return self.change_state(skill_name, SkillLifecycleState.active, reason=reason, actor=actor)

    def block_skill(self, skill_name: str, *, reason: str, actor: str = "curator") -> SkillCuratorEvent:
        return self.change_state(skill_name, SkillLifecycleState.blocked, reason=reason, actor=actor)

    def unblock_skill(self, skill_name: str, *, reason: str, actor: str = "curator") -> SkillCuratorEvent:
        return self.restore_skill(skill_name, reason=reason, actor=actor)

    def delete_skill(self, skill_name: str, *, reason: str, actor: str = "curator") -> SkillCuratorEvent:
        return self.change_state(skill_name, SkillLifecycleState.deleted, reason=reason, actor=actor)

    def suggest_lifecycle_transitions(
        self,
        skills: list[SkillMetadata],
        *,
        stale_days: int = 90,
        archive_days: int = 180,
    ) -> list[LifecycleSuggestion]:
        from core.skills.usage import SkillUsageTracker

        stats_by_name = SkillUsageTracker(self.anima_dir).get_all_stats()
        now = datetime.now(UTC)
        suggestions: list[LifecycleSuggestion] = []
        for meta in skills:
            if meta.is_procedure or is_unloadable_lifecycle_state(meta.lifecycle_state):
                continue
            stats = stats_by_name.get(meta.name)
            success = stats.success_count if stats else meta.success_count
            failure = stats.failure_count if stats else meta.failure_count
            patch_count = stats.patch_count if stats else meta.patch_count
            total = success + failure
            if total and failure / total >= 0.7:
                suggestions.append(
                    LifecycleSuggestion(meta.name, SkillLifecycleState.review, "failure_rate_high", failure / total)
                )
                continue
            if patch_count >= 5:
                suggestions.append(
                    LifecycleSuggestion(meta.name, SkillLifecycleState.review, "patch_count_consolidation", patch_count)
                )
                continue
            if meta.pinned or meta.protected:
                continue
            last_used = _parse_time(stats.last_used_at if stats else None) or meta.last_used_at
            if last_used is None:
                continue
            age_days = (now - last_used.astimezone(UTC)).days
            if age_days >= archive_days:
                suggestions.append(
                    LifecycleSuggestion(meta.name, SkillLifecycleState.archived, "unused_180d", age_days)
                )
            elif age_days >= stale_days:
                suggestions.append(LifecycleSuggestion(meta.name, SkillLifecycleState.stale, "unused_90d", age_days))
        return suggestions

    def metadata_gaps(self, skills: list[SkillMetadata]) -> dict[str, list[str]]:
        from core.skills.router import SkillRouter

        return SkillRouter().metadata_gaps(skills)

    def detect_duplicates(self, skills: list[SkillMetadata], *, min_score: float = 0.58) -> list[DuplicateCandidate]:
        candidates: list[DuplicateCandidate] = []
        active = [m for m in skills if not m.is_procedure and not is_unloadable_lifecycle_state(m.lifecycle_state)]
        for i, left in enumerate(active):
            for right in active[i + 1 :]:
                signals: list[str] = []
                name_score = difflib.SequenceMatcher(None, left.name.casefold(), right.name.casefold()).ratio()
                if name_score >= 0.72:
                    signals.append("name_path_similarity")
                routing_score = _jaccard(_routing_terms(left), _routing_terms(right))
                if routing_score >= 0.35:
                    signals.append("routing_metadata_overlap")
                desc_score = _jaccard(_tokens(left.description), _tokens(right.description))
                if desc_score >= 0.35:
                    signals.append("description_lexical_overlap")
                score = round((name_score * 0.35) + (routing_score * 0.4) + (desc_score * 0.25), 4)
                if signals and score >= min_score:
                    candidates.append(DuplicateCandidate(left.name, right.name, score, signals))
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def propose_merge(
        self,
        skill_name: str,
        related_skill: str,
        *,
        actor: str = "curator",
        reason: str = "duplicate_candidate",
    ) -> Path:
        self.proposal_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_stamp()}_{_safe_name(skill_name)}__{_safe_name(related_skill)}.md"
        proposal_path = self.proposal_dir / filename
        body = (
            f"# Skill Merge Proposal: {skill_name} -> {related_skill}\n\n"
            f"- source_skill: {skill_name}\n"
            f"- target_skill: {related_skill}\n"
            f"- reason: {reason}\n"
            f"- generated_at: {now_iso()}\n\n"
            "This is a proposal only. Do not overwrite SKILL.md without human approval.\n"
        )
        proposal_path.write_text(body, encoding="utf-8")
        self.append_event(
            SkillCuratorEvent(
                ts=now_iso(),
                event_type=SkillCuratorEventType.merge_proposed,
                skill_name=skill_name,
                related_skill=related_skill,
                proposal_path=str(proposal_path.relative_to(self.anima_dir)),
                reason=reason,
                actor=actor,
            )
        )
        return proposal_path

    def generate_report(self, skills: list[SkillMetadata]) -> dict[str, Any]:
        replay = self.replay_state()
        return {
            "states": {name: state.value for name, state in replay.states.items()},
            "suggestions": [
                asdict(s) | {"suggested_state": s.suggested_state.value}
                for s in self.suggest_lifecycle_transitions(skills)
            ],
            "metadata_gaps": self.metadata_gaps(skills),
            "duplicates": [asdict(d) for d in self.detect_duplicates(skills)],
        }

    def _assert_can_restore(self, skill_name: str) -> None:
        meta = self._load_metadata_by_name(skill_name)
        if meta is None:
            return
        if meta.trust_level == SkillTrustLevel.blocked or meta.security.verdict == SkillScanVerdict.dangerous:
            raise ValueError("Blocked or dangerous skills cannot be restored to active")

    def _load_metadata_by_name(self, skill_name: str) -> SkillMetadata | None:
        from core.skills.loader import load_skill_metadata

        paths = list(self.skills_dir.glob("*/SKILL.md"))
        if self.common_skills_dir is not None:
            paths.extend(self.common_skills_dir.glob("*/SKILL.md"))
            paths.extend(self.common_skills_dir.glob("*/*/SKILL.md"))
        for path in paths:
            try:
                meta = load_skill_metadata(path)
            except Exception:
                continue
            if meta.name == skill_name:
                return meta
        return None

    def _create_reference_rewrite_proposal(
        self, skill_name: str, *, absorbed_into: str | None, actor: str
    ) -> str | None:
        from core.skills.reference_rewriter import generate_reference_rewrite_proposal

        proposal = generate_reference_rewrite_proposal(
            self.anima_dir,
            skill_name,
            absorbed_into=absorbed_into,
            actor=actor,
        )
        if proposal is None:
            return None
        return str(proposal.relative_to(self.anima_dir))

    def _invalidate_rag_cache(self, skill_name: str) -> None:
        meta_path = self.anima_dir / "index_meta.json"
        if not meta_path.exists():
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            for key in (f"skills/{skill_name}/SKILL.md", f"skills/quarantine/{skill_name}/SKILL.md"):
                data.pop(key, None)
            meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to invalidate skill index metadata for %s", skill_name, exc_info=True)

    def _purge_personal_skill_vectors(self, skill_name: str) -> None:
        """Best-effort deletion of already indexed personal skill chunks."""
        try:
            from core.memory.rag.singleton import get_vector_store

            vector_store = get_vector_store(self.anima_dir.name)
            if vector_store is None:
                return
            collection = f"{self.anima_dir.name}_skills"
            for source_file in (
                f"skills/{skill_name}/SKILL.md",
                f"skills/quarantine/{skill_name}/SKILL.md",
            ):
                results = vector_store.get_by_metadata(collection, {"source_file": source_file}, limit=10_000)
                ids = [result.document.id for result in results]
                if ids:
                    vector_store.delete_documents(collection, ids)
        except Exception:
            logger.debug("Failed to purge skill vectors for %s", skill_name, exc_info=True)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9ぁ-んァ-ヶー一-龯々]{2,}", text.casefold()) if w not in {"skill", "tool"}}


def _routing_terms(meta: SkillMetadata) -> set[str]:
    values = [
        *meta.use_when,
        *meta.trigger_phrases,
        *meta.domains,
        *meta.tags,
        *meta.routing.use_when,
        *meta.routing.trigger_phrases,
        *meta.routing.domains,
    ]
    return set().union(*(_tokens(v) for v in values)) if values else set()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _safe_stamp() -> str:
    return now_iso().replace(":", "").replace("+", "_").replace("-", "").replace(".", "")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-._")
    return safe or "skill"
