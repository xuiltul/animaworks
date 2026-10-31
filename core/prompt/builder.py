from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""System prompt builder — facade module.

The heavy lifting is delegated to submodules:
  - :mod:`core.prompt.sections`   — template loading helpers
  - :mod:`core.prompt.org_context` — organisation tree/context
  - :mod:`core.prompt.messaging`  — messaging & notification sections
  - :mod:`core.prompt.assembler`  — budget allocation & XML assembly

This module keeps shared prompt-building entry points and patch targets
while delegating implementation details to the submodules above.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.anima.workspace_files import IDENTITY_FILE, SOUL_FILE
from core.i18n import t
from core.memory import MemoryManager
from core.memory.conversation.shortterm import ShortTermMemory
from core.paths import get_data_dir, load_prompt, load_prompt_text
from core.prompt.assembler import (
    _REFERENCE_WINDOW,
    SectionEntry,  # noqa: F401
    _allocate_sections,
    _assemble_with_tags,
    _compute_system_budget,
    _split_content_items,
)
from core.prompt.messaging import (
    _build_human_notification_guidance,  # noqa: F401
    _build_messaging_section,
)
from core.prompt.org_context import (
    _build_org_context,
    _discover_other_animas,
    _is_mcp_mode,
)
from core.prompt.sections import (
    _load_fallback_strings,
    _load_section_strings,
)
from core.time_utils import now_local
from core.tooling.policy.tool_content import load_guide
from core.voice.emotion_style import EmotionStyle, VoiceChannel, front_prompt_emotion_instruction

logger = logging.getLogger("animaworks.prompt_builder")

# Re-exported constants
_MCP_MODES = frozenset({"s", "c", "d", "g", "x", "y"})
_CURRENT_STATE_MAX_CHARS = 3000
_IDENTITY_H2_RE = re.compile(r"^ {0,3}##(?!#)[ \t]+(?P<heading>.+?)\s*#*\s*$")
_PRIMING_TAG_RE = re.compile(r"</?priming\b[^>]*>", re.IGNORECASE)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)

# ── Prompt tier constants ─────────────────────────────────────
TIER_FULL = "full"
TIER_STANDARD = "standard"
TIER_LIGHT = "light"
TIER_MINIMAL = "minimal"
TIER_MICRO = "micro"


def resolve_prompt_tier(context_window: int) -> str:
    """Determine prompt tier from context window size."""
    if context_window >= 128_000:
        return TIER_FULL
    if context_window >= 32_000:
        return TIER_STANDARD
    if context_window >= 16_000:
        return TIER_LIGHT
    if context_window > 8_192:
        return TIER_MINIMAL
    return TIER_MICRO


def _build_emotion_instruction() -> str:
    """Build EMOTION_INSTRUCTION with the canonical emotion list."""
    from core.schemas import VALID_EMOTIONS

    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    return load_prompt("builder/emotion_instruction", emotion_list=emotion_list).strip()


EMOTION_INSTRUCTION = _build_emotion_instruction()


def _read_default_workspace(anima_dir: Path) -> str:
    """Read default_workspace from status.json and resolve via workspace registry."""
    from core.org.workspace import resolve_default_workspace

    resolved, alias = resolve_default_workspace(anima_dir)
    if not alias:
        return ""
    if resolved:
        return t("builder.default_workspace", path=str(resolved), alias=alias)
    return t("builder.default_workspace_unresolved", alias=alias)


@dataclass
class BuildResult:
    """Result of system prompt building."""

    system_prompt: str

    def __str__(self) -> str:
        return self.system_prompt

    def __len__(self) -> int:
        return len(self.system_prompt)

    def encode(self, encoding: str = "utf-8") -> bytes:
        return self.system_prompt.encode(encoding)

    def __contains__(self, item: str) -> bool:
        return item in self.system_prompt

    def __add__(self, other: str) -> str:
        return self.system_prompt + other

    def __radd__(self, other: str) -> str:
        return other + self.system_prompt

    def index(self, sub: str, *args: int) -> int:
        return self.system_prompt.index(sub, *args)

    def count(self, sub: str, *args: int) -> int:
        return self.system_prompt.count(sub, *args)


@dataclass(frozen=True)
class _SkillCatalogRouterSettings:
    enabled: bool = True
    top_k: int = 5
    min_score: float = 1.15
    include_body: bool = True
    dense_enabled: bool = True
    dense_weight: float = 8.0
    max_items: int = 3


# ── Per-group section builders ────────────────────────────────
# Private helpers — kept in builder.py so they share the same
# ``load_prompt`` binding that tests patch.


def _filter_identity_business_sections(identity: str, excluded_headings: list[str]) -> str:
    """Remove configured H2 sections from identity text for non-chat triggers."""
    exclusions = [heading.casefold() for heading in excluded_headings if heading.strip()]
    if not identity or not exclusions:
        return identity

    sections: list[tuple[bool, str]] = []
    current: list[str] = []
    current_excluded = False
    found_heading = False
    in_fence = False

    for line in identity.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        match = _IDENTITY_H2_RE.match(line.rstrip("\r\n")) if not in_fence else None
        if match:
            found_heading = True
            if current:
                sections.append((current_excluded, "".join(current)))
            current = [line]
            heading = match.group("heading").casefold()
            current_excluded = any(exclusion in heading for exclusion in exclusions)
        else:
            current.append(line)

    if current:
        sections.append((current_excluded, "".join(current)))
    if not found_heading:
        return identity

    filtered = "".join(content for excluded, content in sections if not excluded)
    return filtered if filtered.strip() else identity


def _is_under(path: Path, root: Path) -> bool:
    """Return whether *path* resolves inside *root*."""
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _priming_has_content(content: str) -> bool:
    """Return whether priming contains data beyond headings and empty wrappers."""
    without_tags = _PRIMING_TAG_RE.sub("", content)
    intro = t("priming.section_intro").strip()
    if intro:
        without_tags = without_tags.replace(intro, "")
    without_headings = _MARKDOWN_HEADING_RE.sub("", without_tags)
    return bool(without_headings.strip())


def _build_group1(
    pd: Path,
    data_dir: Path,
    memory: MemoryManager,
    is_task: bool,
    _ss: dict[str, str],
    *,
    tier: str = TIER_FULL,
    is_heartbeat: bool = False,
    is_chat: bool = False,
    is_consolidation: bool = False,
) -> list[SectionEntry]:
    """Group 1: Environment, identity, injection, and behaviour rules."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group1_header", "# 1. Environment and Action Rules"), "group1_header", 1)

    if tier == TIER_MICRO:
        _add(f"Anima: {pd.name} | Data: {data_dir}", "environment", 1)
    else:
        dw = _read_default_workspace(pd)
        if dw:
            _add(dw, "default_workspace", 2)
            _add(
                load_prompt("builder/repo_work_rules", data_dir=data_dir),
                "repo_work_rules",
                2,
            )

        _env = load_prompt("environment", data_dir=data_dir, anima_name=pd.name)
        if _env:
            _add(_env, "environment", 2)

    identity = memory.read_identity()
    soul_reader = getattr(memory, "read_soul", None)
    if callable(soul_reader):
        soul = soul_reader()
        if isinstance(soul, str) and soul.strip():
            identity = "\n\n".join(part for part in (identity, soul) if part.strip())
    if identity:
        if not is_chat:
            try:
                from core.config import load_config

                excluded = load_config().prompt.identity_business_exclude_headings
                identity = _filter_identity_business_sections(identity, excluded)
            except Exception:
                logger.debug("Could not load identity heading exclusions", exc_info=True)
        _add(identity, "identity", 1)

    injection = memory.read_injection()
    if injection:
        _add(injection, "injection", 1)
        try:
            from core.config import load_config

            config = load_config()
            threshold = config.prompt.injection_size_warning_chars
            if is_consolidation and len(injection) > threshold:
                _add(
                    t("builder.injection_size_warning", size=len(injection), threshold=threshold),
                    "injection_size_warning",
                    1,
                )
                logger.warning("AGENTS.md oversized: %d chars (threshold=%d)", len(injection), threshold)
        except Exception:
            logger.debug("Best-effort operation failed", exc_info=True)

    if tier != TIER_MICRO:
        _br = load_prompt_text("behavior_rules")
        if _br:
            _add(_br, "behavior_rules", 2)

    return out


_VISION_PLACEHOLDERS = frozenset(
    {
        "要記入",
        "未記入",
        "todo",
        "tbd",
        "to be determined",
        "not entered",
        "not filled in",
        "작성 필요",
        "미작성",
        "추후 작성",
    }
)


def _is_placeholder_vision(text: str) -> bool:
    """Return whether a vision has no substantive body worth injecting."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).strip()
    if len(body) < 40:
        return True
    normalized = re.sub(r"^[\s\[\](){}<>（）：:._*-]+|[\s\[\](){}<>（）：:._*-]+$", "", body).casefold()
    return normalized in _VISION_PLACEHOLDERS


def _build_group2(
    memory: MemoryManager,
    permissions: str,
    is_background_auto: bool,
    is_task: bool,
    _ss: dict[str, str],
) -> list[SectionEntry]:
    """Group 2: Bootstrap, vision, specialty, permissions."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group2_header", "# 2. Yourself"), "group2_header", 1)
    b = memory.read_bootstrap()
    if b:
        _add(b, "bootstrap", 3)
    summary_info = memory.read_company_vision_summary()
    if isinstance(summary_info, tuple) and len(summary_info) == 2:
        summary, reference_path = summary_info
        summary = summary.strip()
        if summary:
            reference = t("memory.company_vision_summary_reference", path=reference_path)
            _add(f"{summary}\n{reference}", "vision", 3)
    else:
        v = memory.read_company_vision()
        if v and not _is_placeholder_vision(v):
            _add(v, "vision", 3)
    if not is_background_auto:
        sp = memory.read_specialty_prompt()
        if sp:
            _add(sp, "specialty", 3)
    if permissions:
        _add(permissions, "permissions", 1)
    return out


def _format_resolved_at_local(resolved_at: Any) -> str:
    """Format a resolved_at datetime in the app timezone (e.g. ``2026-07-27 16:59 JST``)."""
    from datetime import datetime as _dt

    from core.time_utils import get_app_timezone

    if not isinstance(resolved_at, _dt):
        try:
            resolved_at = _dt.fromisoformat(str(resolved_at))
        except (TypeError, ValueError):
            return str(resolved_at)
    tz = get_app_timezone()
    if resolved_at.tzinfo is None:
        local_dt = resolved_at.replace(tzinfo=tz)
    else:
        local_dt = resolved_at.astimezone(tz)
    tz_label = local_dt.tzname() or tz.key
    return f"{local_dt.strftime('%Y-%m-%d %H:%M')} {tz_label}"


def _build_resolved_approvals_section(anima_name: str, _ss: dict[str, str]) -> str:
    """Build the resolved-approvals reminder block, or empty string if none / disabled."""
    from core.config import load_config
    from core.notification.interactive import (
        _is_approval_category,
        list_resolved_for_anima_resilient,
    )

    try:
        hours = int(load_config().heartbeat.resolved_interaction_reminder_hours)
    except Exception:
        logger.debug("Failed to read resolved_interaction_reminder_hours; using 48", exc_info=True)
        hours = 48
    if hours <= 0:
        return ""

    pairs = list_resolved_for_anima_resilient(anima_name, within_hours=hours)
    approval_pairs = [(req, res) for req, res in pairs if _is_approval_category(req.category)]
    if not approval_pairs:
        return ""

    header_tmpl = _ss.get(
        "resolved_approvals_header",
        "## Resolved Approvals (last {hours}h)",
    )
    intro = _ss.get(
        "resolved_approvals_intro",
        "The following approval requests have already been answered by a human. "
        "If current_state.md still has pending-approval blockers for these, close them. "
        "Do not re-notify (call_human) about already-answered matters.",
    )
    item_tmpl = _ss.get(
        "resolved_approvals_item",
        '- callback_id: {callback_id} — "{decision}" by {actor} ({source}, {resolved_at})',
    )
    try:
        header = header_tmpl.format(hours=hours)
    except (KeyError, ValueError):
        header = header_tmpl

    lines = [header, "", intro, ""]
    for _req, res in approval_pairs:
        try:
            item = item_tmpl.format(
                callback_id=res.callback_id,
                decision=res.decision,
                actor=res.actor,
                source=res.source,
                resolved_at=_format_resolved_at_local(res.resolved_at),
            )
        except (KeyError, ValueError):
            item = (
                f'- callback_id: {res.callback_id} — "{res.decision}" '
                f"by {res.actor} ({res.source}, {_format_resolved_at_local(res.resolved_at)})"
            )
        lines.append(item)
    return "\n".join(lines)


_DATED_STATE_HEADING_RE = re.compile(
    r"^##\s+(?P<title>.+?)（(?P<date>[^）]*\d{1,4}[-/]\d{1,2}[^）]*)）\s*$",
    re.MULTILINE,
)


def _collapse_superseded_notes(state: str) -> str:
    """Collapse bodies of older dated notes that share the same case key."""
    matches = list(_DATED_STATE_HEADING_RE.finditer(state))
    by_key: dict[str, list[int]] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        if " " not in title:
            continue
        case_key = title.rsplit(" ", 1)[0].strip()
        if case_key:
            by_key.setdefault(case_key, []).append(index)

    superseded: dict[int, int] = {}
    for indices in by_key.values():
        if len(indices) > 1:
            latest = indices[-1]
            superseded.update({old: latest for old in indices[:-1]})
    if not superseded:
        return state

    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        if index not in superseded:
            continue
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(state)
        parts.append(state[cursor : match.start()])
        latest_match = matches[superseded[index]]
        latest_heading = latest_match.group(0).removeprefix("## ").strip()
        old_heading = match.group(0).rstrip()
        parts.append(f"{old_heading}{t('builder.superseded_note', latest=latest_heading)}\n\n")
        dropped_body = state[match.end() : next_start]
        logger.debug("Collapsed superseded current_state note body: %s", dropped_body.strip())
        cursor = next_start
    parts.append(state[cursor:])
    logger.info("Collapsed %d superseded current_state note(s)", len(superseded))
    return "".join(parts)


def _related_resolution_resolvers(anima_dir: Path) -> set[str]:
    """Return this Anima, its supervisor, and its direct subordinates."""
    related = {anima_dir.name}
    try:
        from core.config.file_access_policy import load_denied_roots
        from core.prompt.org_context import _filter_company_visible_animas, _scan_all_animas

        denied_roots = load_denied_roots(anima_dir)
        all_animas = _scan_all_animas(anima_dir.parent, denied_roots)
        all_animas = _filter_company_visible_animas(anima_dir.name, all_animas, anima_dir.parent)
        current = all_animas.get(anima_dir.name)
        if current is None:
            return related
        if current.supervisor:
            related.add(current.supervisor)
        related.update(name for name, config in all_animas.items() if config.supervisor == anima_dir.name)
    except Exception:
        logger.debug("Failed to determine related resolution owners", exc_info=True)
    return related


def _filter_relevant_resolutions(anima_dir: Path, resolutions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only recent resolutions owned by this Anima or its direct reports."""
    related = _related_resolution_resolvers(anima_dir)
    return [resolution for resolution in resolutions if resolution.get("resolver", "") in related]


def _build_resolution_registry_section(anima_dir: Path, memory: MemoryManager) -> str:
    """Build the recent resolution registry for this Anima's direct org scope."""
    try:
        resolutions = memory.read_resolutions(days=7)
        if not resolutions:
            return ""
        resolutions = memory.filter_resolutions_by_company(resolutions)
        resolutions = _filter_relevant_resolutions(anima_dir, resolutions)
        if not resolutions:
            return ""
        seen: dict[str, dict[str, str]] = {}
        for resolution in resolutions:
            seen[resolution.get("issue", "")] = resolution
        deduped = sorted(seen.values(), key=lambda item: item.get("ts", ""))
        lines = [
            f"- [{resolution.get('ts', '')[:16]}] {resolution.get('resolver', 'unknown')}: "
            f"{resolution.get('issue', '')}"
            for resolution in deduped[-10:]
        ]
        return load_prompt("builder/resolution_registry", res_lines="\n".join(lines))
    except Exception:
        logger.debug("Failed to inject resolution registry", exc_info=True)
        return ""


def _build_group3(
    pd: Path,
    memory: MemoryManager,
    scale: float,
    priming_section: str,
    pending_human_notifications: str,
    execution_mode: str,
    is_heartbeat: bool,
    is_chat: bool,
    is_task: bool,
    _ss: dict[str, str],
    _fs: dict[str, str],
    *,
    shortterm_text: str = "",
) -> list[SectionEntry]:
    """Group 3: Current state, resolutions, priming, notifications, recent tools."""
    out: list[SectionEntry] = []

    def _add(
        c: str,
        sid: str,
        pri: int = 2,
        kind: str = "rigid",
        *,
        trim_from: Literal["head", "tail"] = "tail",
        budget_group: Literal["framework", "recall"] = "framework",
    ) -> None:
        if c and c.strip():
            out.append(
                SectionEntry(
                    id=sid,
                    priority=pri,
                    kind=kind,
                    content=c,
                    trim_from=trim_from,
                    budget_group=budget_group,
                )
            )

    _add(_ss.get("group3_header", "# 6. Current Situation"), "group3_header", 1)

    current_time = now_local().strftime("%Y-%m-%d %H:%M (%Z)")
    _add(f"{_ss.get('current_time_label', '**Current time**:')} {current_time}", "current_time", 1)

    _state_max = max(int(_CURRENT_STATE_MAX_CHARS * scale), 500)
    state = memory.read_current_state()
    state_content = ""
    if state and state.strip() != "status: idle":
        state = _collapse_superseded_notes(state)
        if len(state) > _state_max:
            truncated = state[-_state_max:]
            first_nl = truncated.find("\n")
            if first_nl != -1 and first_nl < _state_max * 0.2:
                truncated = truncated[first_nl + 1 :]
            state = f"{_fs.get('truncated', '(earlier portion omitted)')}\n\n{truncated}"
        state_content = load_prompt("builder/task_in_progress", state=state)
    elif state:
        state_content = f"{_ss.get('current_state_header', '## Current State')}\n\n{state}"
    if state_content:
        _add(state_content, "current_state", 2, "elastic", trim_from="head")

    # Deterministic safety net: remind anima to close resolved approval blockers
    try:
        resolved_block = _build_resolved_approvals_section(pd.name, _ss)
        if resolved_block:
            _add(resolved_block, "resolved_approvals", 1, "rigid")
    except Exception:
        logger.debug("Failed to inject resolved approvals section", exc_info=True)

    resolution_block = _build_resolution_registry_section(pd, memory)
    if resolution_block:
        _add(resolution_block, "resolution_registry", 2, "rigid")

    if priming_section and _priming_has_content(priming_section):
        # Explicit source contracts survive recall trimming. Never infer safety
        # importance from arbitrary memory prose or split a trust-boundary block.
        protected, recall = [], []
        priming_items = _split_content_items(priming_section)
        has_pending_task_block = any(
            re.match(r'<priming\b[^>]*\bsource="pending_tasks"', item) for item in priming_items
        )
        pending_tasks_header = t("priming.pending_tasks_header").strip()
        for item in priming_items:
            if item.strip() == pending_tasks_header:
                # The formatter emits this heading immediately before the
                # protected payload. Keep them together so allocation cannot
                # leave a heading in the optional recall section by itself.
                if has_pending_task_block:
                    protected.append(item)
                continue
            if re.match(
                r'<priming\b[^>]*\bsource="(?:resident_knowledge|pending_tasks|recent_outbound|action_rule)"', item
            ):
                protected.append(item)
            else:
                recall.append(item)
        _add("\n\n".join(protected), "priming_required_context", 1, "rigid", budget_group="recall")
        _add("\n\n".join(recall), "priming", 2, "elastic", budget_group="recall")
    # Keep aligned with priming.outbound.HUMAN_NOTIFICATION_CHANNELS.
    if pending_human_notifications and (is_chat or is_heartbeat):
        _add(pending_human_notifications, "pending_human_notifications", 1, "rigid")
    if shortterm_text:
        _add(shortterm_text, "shortterm", 3, "elastic", trim_from="head")
    return out


def _format_trust_tag(meta: Any) -> str:
    """Format a trust_level bracket tag for the skill catalog.

    Returns empty string for the default ``trusted`` level to keep the
    catalog compact; shows ``[level]`` only when non-default.

    Accepts both ``SkillMetadata`` (with ``trust_level`` as enum) and
    legacy ``SkillMeta`` (which lacks the field).
    """
    from core.skills.models import SkillTrustLevel

    trust = getattr(meta, "trust_level", None)
    if trust is None:
        return ""
    if isinstance(trust, SkillTrustLevel):
        if trust == SkillTrustLevel.trusted:
            return ""
        return f" [{trust.value}]"
    level_str = str(trust)
    if level_str == "trusted":
        return ""
    return f" [{level_str}]"


def _load_skill_catalog_router_settings() -> _SkillCatalogRouterSettings:
    try:
        from core.config import load_config

        prompt_cfg = load_config().prompt
    except Exception:
        logger.debug("Failed to load skill catalog router settings", exc_info=True)
        from core.config.schemas import PromptConfig

        prompt_cfg = PromptConfig()

    return _SkillCatalogRouterSettings(
        enabled=bool(prompt_cfg.skill_catalog_router_enabled),
        top_k=max(1, int(prompt_cfg.skill_catalog_router_top_k)),
        min_score=max(0.0, float(prompt_cfg.skill_catalog_router_min_score)),
        include_body=bool(prompt_cfg.skill_catalog_router_include_body),
        dense_enabled=bool(prompt_cfg.skill_catalog_router_dense_enabled),
        dense_weight=max(0.0, float(prompt_cfg.skill_catalog_router_dense_weight)),
        max_items=max(1, int(prompt_cfg.skill_catalog_max_items)),
    )


def _skill_catalog_pointer(meta: Any) -> str:
    """Catalog pointer for *meta* — single source of truth is the router."""
    from core.skills.router import _pointer_path

    return _pointer_path(meta)


def _dedupe_skill_catalog_metas(metas: list[Any]) -> list[Any]:
    """Deduplicate catalog entries by description, preferring local skills."""
    source_priority = {
        "skills/": 0,
        "procedures/": 1,
        "common_skills/": 2,
        "external/": 3,
    }
    ordered = sorted(
        enumerate(metas),
        key=lambda item: (
            next(
                (
                    priority
                    for prefix, priority in source_priority.items()
                    if _skill_catalog_pointer(item[1]).startswith(prefix)
                ),
                len(source_priority),
            ),
            item[0],
        ),
    )
    deduped: list[Any] = []
    seen_descriptions: set[str] = set()
    for _, meta in ordered:
        description = str(getattr(meta, "description", "") or "").strip()[:60]
        if description:
            if description in seen_descriptions:
                continue
            seen_descriptions.add(description)
        deduped.append(meta)
    return deduped


def _limit_skill_catalog_entries(entries: list[str], max_items: int) -> list[str]:
    """Apply the configured catalog ceiling without changing router order."""
    return entries[: max(1, max_items)]


def _format_skill_catalog_line(
    meta: Any,
    *,
    path: str,
    common_label: str,
    procedure_label: str,
    desc_limit: int,
    match_confidence: str | None = None,
) -> str:
    description = getattr(meta, "description", "")
    desc = (description[:desc_limit] + "…") if len(description) > desc_limit else description
    labels: list[str] = []
    if bool(getattr(meta, "is_procedure", False)):
        labels.append(procedure_label)
    elif bool(getattr(meta, "is_common", False)):
        labels.append(common_label)
    risk = getattr(meta, "risk", None)
    if bool(getattr(risk, "requires_human_approval", False)):
        labels.append("human-approval")
    if match_confidence:
        labels.append(f"match={match_confidence}")
    label_text = f" ({', '.join(labels)})" if labels else ""
    ext_tag = ""
    if bool(getattr(meta, "is_external", False)):
        engine = ""
        source = getattr(meta, "source", None)
        if source is not None:
            engine = getattr(source, "engine", None) or ""
        ext_tag = f" [ext:{engine}]"
    return f"- {path}{label_text}{_format_trust_tag(meta)}{ext_tag}: {desc}"


def _skill_catalog_sections(entries: list[str]) -> list[SectionEntry]:
    """Render the skill catalog as an elastic prompt section."""
    content = "\n".join(
        [
            t("builder.skill_catalog_header"),
            t("builder.skill_catalog_instruction"),
            "",
            "<available_skills>",
            *entries,
            "</available_skills>",
        ]
    )
    return [SectionEntry("skill_catalog", 2, "elastic", content)]


def _requires_human_approval(meta: Any) -> bool:
    risk = getattr(meta, "risk", None)
    if isinstance(risk, dict):
        return bool(risk.get("requires_human_approval", False))
    return bool(getattr(risk, "requires_human_approval", False))


def _skill_visible_in_prompt_context(meta: Any, *, is_background_auto: bool) -> bool:
    return not (is_background_auto and _requires_human_approval(meta))


def _host_tool_line(execution_mode: str) -> str:
    """Return the host built-in tool line for the given execution mode.

    d (Cursor) and g (Gemini) are unused, so they reuse the 's' wording.
    y (Antigravity CLI) has its own wording because agy exposes a distinct
    built-in tool set and calls AnimaWorks tools through ``call_mcp_tool``.
    Unknown modes return an empty string (no injection).
    """
    mode = (execution_mode or "").lower()
    if mode in ("d", "g"):
        mode = "s"
    if mode not in ("s", "c", "x", "y", "a"):
        return ""
    return t(f"tool_guide.host_tools.{mode}")


def _build_group4(
    pd: Path,
    data_dir: Path,
    memory: MemoryManager,
    scale: float,
    execution_mode: str,
    skill_index: Any,
    is_heartbeat: bool,
    is_background_auto: bool,
    is_chat: bool,
    is_task: bool,
    tool_registry: list[str] | None,
    personal_tools: dict[str, str] | None,
    _ss: dict[str, str],
    _fs: dict[str, str],
    message: str = "",
    thread_id: str = "default",
) -> list[SectionEntry]:
    """Group 4: Memory guide, common knowledge, tool guides."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group4_header", "# 3. Memory and Capabilities"), "group4_header", 1)

    mg = load_prompt("memory_guide", anima_dir=pd.resolve()).strip()
    if mg:
        _add(mg, "memory_guide", 3)

    # ── Tool guides ───
    if is_heartbeat:
        try:
            hb_tool = load_prompt("builder/heartbeat_tool_instruction")
        except FileNotFoundError:
            hb_tool = t("builder.heartbeat_tool_fallback")
        _add(hb_tool, "tool_guides", 2)
    elif _is_mcp_mode(execution_mode):
        sb = load_guide("s_builtin").strip()
        sm = load_guide("s_mcp").strip()
        g = "\n\n".join(p for p in (sb, sm) if p)
        if g:
            host_line = _host_tool_line(execution_mode)
            if host_line:
                g = host_line + "\n\n" + g
            _add(g, "tool_guides", 2)
    else:
        ns = load_guide("non_s").strip()
        if ns:
            host_line = _host_tool_line(execution_mode)
            if host_line:
                ns = host_line + "\n\n" + ns
            _add(ns, "tool_guides", 2)

    if not is_heartbeat and (tool_registry or personal_tools):
        own_tools_dir = (pd / "tools").resolve()
        own_tools = {name for name, path in (personal_tools or {}).items() if _is_under(Path(str(path)), own_tools_dir)}
        cats = sorted(set((tool_registry or []) + [n for n in (personal_tools or {}) if n not in own_tools]))
        if cats or own_tools:
            et = t("builder.external_tools", categories=", ".join(cats)) if cats else ""
            if own_tools:
                own_line = t("builder.external_tools.personal_count", count=len(own_tools))
                et = f"{et}\n{own_line}" if et else own_line
            if _is_mcp_mode(execution_mode) or execution_mode == "a":
                et += "\n" + t("builder.external_tools.direct")
            _add(et, "external_tools", 2)

    if is_chat:
        try:
            from core.skills.activation import build_active_skill_context

            active_context = build_active_skill_context(pd, thread_id=thread_id)
            rendered_active = active_context.render()
            if rendered_active:
                _add(rendered_active, "active_skills", 1, "elastic")
        except Exception:
            logger.debug("Skipped active skill context injection", exc_info=True)

    # ── Skill catalog (Agent Skills standard) ───
    # Uses SkillIndex which excludes blocked/quarantine skills. Background
    # automation also excludes skills that need separate human approval.
    if not is_heartbeat:
        _DESC_LIMIT = 250
        common_label = t("skill.label_common")
        procedure_label = t("skill.label_procedure")
        settings = _load_skill_catalog_router_settings()
        all_skills = _dedupe_skill_catalog_metas(
            [
                meta
                for meta in skill_index.all_skills
                if _skill_visible_in_prompt_context(meta, is_background_auto=is_background_auto)
            ]
        )
        catalog_entries: list[str] = []

        if settings.enabled and message.strip():
            from core.skills.router import SkillRouter

            dense_scores = None
            if settings.dense_enabled:
                import time

                from core.skills.dense import skill_dense_scores

                t0 = time.monotonic()
                dense_scores = skill_dense_scores(message, all_skills)
                logger.debug(
                    "skill dense scoring took %.3fs (skills=%d)",
                    time.monotonic() - t0,
                    len(all_skills),
                )

            candidates = SkillRouter(
                min_score=settings.min_score,
                include_body=settings.include_body,
                dense_weight=settings.dense_weight,
            ).route(message, all_skills, top_k=settings.top_k, dense_scores=dense_scores)
            metas_by_pointer = {_skill_catalog_pointer(meta): meta for meta in all_skills}
            for candidate in candidates:
                meta = metas_by_pointer.get(candidate.path)
                if meta is None:
                    continue
                catalog_entries.append(
                    _format_skill_catalog_line(
                        meta,
                        path=candidate.path,
                        common_label=common_label,
                        procedure_label=procedure_label,
                        desc_limit=_DESC_LIMIT,
                        match_confidence=candidate.confidence,
                    )
                )
            if not catalog_entries:
                for meta in all_skills:
                    catalog_entries.append(
                        _format_skill_catalog_line(
                            meta,
                            path=_skill_catalog_pointer(meta),
                            common_label=common_label,
                            procedure_label=procedure_label,
                            desc_limit=_DESC_LIMIT,
                        )
                    )
        else:
            for meta in all_skills:
                catalog_entries.append(
                    _format_skill_catalog_line(
                        meta,
                        path=_skill_catalog_pointer(meta),
                        common_label=common_label,
                        procedure_label=procedure_label,
                        desc_limit=_DESC_LIMIT,
                    )
                )

        catalog_entries = _limit_skill_catalog_entries(catalog_entries, getattr(settings, "max_items", 3))
        if catalog_entries or not (settings.enabled and message.strip()):
            out.extend(_skill_catalog_sections(catalog_entries))

    return out


def _build_group5(
    pd: Path,
    memory: MemoryManager,
    other_animas: list[str],
    execution_mode: str,
    is_background_auto: bool,
    is_inbox: bool,
    is_task: bool,
    _ss: dict[str, str],
    _fs: dict[str, str],
    *,
    tier: str = TIER_FULL,
) -> list[SectionEntry]:
    """Group 5: Org context, messaging, human notification."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group5_header", "# 4. Organization and Communication"), "group5_header", 1)

    if tier == TIER_MICRO:
        return out

    oc = _build_org_context(pd.name, other_animas, execution_mode)
    if oc:
        _add(oc, "org_context", 2)

    msg = _build_messaging_section(pd, other_animas, execution_mode)
    if is_background_auto and not is_inbox and len(msg) > 500:
        msg = msg[:500] + "\n" + _fs.get("summary", "(summary)")
    _add(msg, "messaging", 2)
    try:
        from core.config import load_config as _lc

        cfg = _lc()
        pcfg = cfg.animas.get(pd.name)
        if (pcfg is None or pcfg.supervisor is None) and cfg.human_notification.enabled:
            _add(_build_human_notification_guidance(execution_mode), "human_notification", 4)
    except Exception:
        logger.debug("Skipped human notification guidance injection", exc_info=True)
    return out


def _build_group6(
    execution_mode: str,
    is_chat: bool,
    is_background_auto: bool,
    is_task: bool,
    _ss: dict[str, str],
    *,
    tier: str = TIER_FULL,
) -> list[SectionEntry]:
    """Group 6: Emotion, reflection, Codex response requirement."""
    out: list[SectionEntry] = []

    def _add(c: str, sid: str, pri: int = 2, kind: str = "rigid") -> None:
        if c and c.strip():
            out.append(SectionEntry(id=sid, priority=pri, kind=kind, content=c))

    _add(_ss.get("group6_header", "# 5. Meta Settings"), "group6_header", 1)

    if tier == TIER_MICRO:
        return out

    if is_chat:
        ei = _build_emotion_instruction()
        if ei:
            _add(ei, "emotion_instruction", 4)
    if execution_mode == "c" and not is_background_auto:
        _add(t("builder.c_response_requirement"), "c_response_requirement", 2)
    return out


# ── Main entry point ──────────────────────────────────────────


def build_system_prompt(
    memory: MemoryManager,
    tool_registry: list[str] | None = None,
    personal_tools: dict[str, str] | None = None,
    priming_section: str = "",
    execution_mode: str = "s",
    message: str = "",
    retriever: object | None = None,
    *,
    trigger: str = "",
    context_window: int = 200_000,
    system_budget: int | None = None,
    pending_human_notifications: str = "",
    thread_id: str = "default",
    shortterm_text: str = "",
) -> BuildResult:
    """Construct the full system prompt from Markdown files.

    Sections are organised into 6 hierarchical groups.
    """
    pd = memory.anima_dir
    data_dir = get_data_dir()
    budget = _compute_system_budget(context_window, system_budget)
    scale = min(context_window / _REFERENCE_WINDOW, 1.0)
    tier = resolve_prompt_tier(context_window)
    _ss = _load_section_strings()
    _fs = _load_fallback_strings()

    from core.execution.session.session_types import (
        SESSION_TYPE_CRON,
        SESSION_TYPE_HEARTBEAT,
        SESSION_TYPE_INBOX,
        SESSION_TYPE_TASK,
        resolve_runtime_session_type,
        trigger_uses_chat_session,
    )

    session_type = resolve_runtime_session_type(trigger)
    is_heartbeat = session_type == SESSION_TYPE_HEARTBEAT
    is_cron = session_type == SESSION_TYPE_CRON
    is_task = session_type == SESSION_TYPE_TASK
    is_inbox = session_type == SESSION_TYPE_INBOX
    is_consolidation = trigger.startswith("consolidation:")
    is_background_auto = is_heartbeat or is_cron or is_consolidation
    is_chat = trigger_uses_chat_session(trigger)

    other_animas = _discover_other_animas(pd)

    from core.paths import get_common_skills_dir
    from core.skills import SkillIndex

    skill_index = SkillIndex(
        skills_dir=pd / "skills",
        common_skills_dir=get_common_skills_dir(),
        procedures_dir=pd / "procedures",
        anima_dir=pd,
    )
    permissions = memory.read_permissions()

    group1 = _build_group1(
        pd,
        data_dir,
        memory,
        is_task,
        _ss,
        tier=tier,
        is_heartbeat=is_heartbeat,
        is_chat=is_chat,
        is_consolidation=is_consolidation,
    )
    group2 = _build_group2(memory, permissions, is_background_auto, is_task, _ss)
    group3 = _build_group3(
        pd,
        memory,
        scale,
        priming_section,
        pending_human_notifications,
        execution_mode,
        is_heartbeat,
        is_chat,
        is_task,
        _ss,
        _fs,
        shortterm_text=shortterm_text,
    )
    g4 = _build_group4(
        pd,
        data_dir,
        memory,
        scale,
        execution_mode,
        skill_index,
        is_heartbeat,
        is_background_auto,
        is_chat,
        is_task,
        tool_registry,
        personal_tools,
        _ss,
        _fs,
        message=message,
        thread_id=thread_id,
    )
    group5 = _build_group5(
        pd,
        memory,
        other_animas,
        execution_mode,
        is_background_auto,
        is_inbox,
        is_task,
        _ss,
        _fs,
        tier=tier,
    )
    group6 = _build_group6(
        execution_mode,
        is_chat,
        is_background_auto,
        is_task,
        _ss,
        tier=tier,
    )

    # Dynamic current-state content stays last so the preceding static groups
    # form a stable prefix that providers can cache across turns.
    sections = group1 + group2 + g4 + group5 + group6 + group3

    # Budget allocation + Final assembly
    allocated = _allocate_sections(sections, budget)
    prompt = _assemble_with_tags(allocated)
    logger.debug(
        "System prompt built: %d/%d sections, total_len=%d, target=%d, ceiling=%d, tier=%s, cw=%d",
        len(allocated),
        len(sections),
        len(prompt),
        budget.target,
        budget.ceiling,
        tier,
        context_window,
    )
    return BuildResult(system_prompt=prompt)


_VOICE_FRONT_EMOTION_NAMES = "neutral/smile/laugh/troubled/surprised/thinking/embarrassed"


def _read_text_file(path: Path) -> str:
    """Read a UTF-8 text file, returning empty string on any failure."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def build_voice_front_prompt(
    anima_dir: Path,
    *,
    anima_name: str | None = None,
    channel: VoiceChannel = "web",
    emotion_style: EmotionStyle = "emoji",
) -> str:
    """Build a minimal, *fixed* system prompt for the voice front lane.

    The prompt is intentionally static for the whole voice session so the
    front model's prefix cache stays valid (nothing per-turn like timestamps
    or current state is injected).  Only persona / speciality material and
    the fixed voice-mode speaking rules are included.

    Args:
        anima_dir: Directory of the target Anima.
        anima_name: Canonical anima name (defaults to the directory name).

    Returns:
        The assembled system prompt string.
    """
    name = anima_name or (anima_dir.name if anima_dir else "anima")
    parts: list[str] = []

    identity = _read_text_file(anima_dir / IDENTITY_FILE) if anima_dir else ""
    soul = _read_text_file(anima_dir / SOUL_FILE) if anima_dir else ""
    persona = "\n\n".join(part for part in (identity, soul) if part)
    if persona:
        parts.append(persona)

    specialty = _read_text_file(anima_dir / "specialty_prompt.md") if anima_dir else ""
    if specialty:
        parts.append(f"Speciality:\n{specialty}")

    phone_rules = ""
    if channel == "phone":
        phone_rules = (
            "- This is speech-recognition output from a phone call and may contain errors; ask for clarification "
            "when the meaning is ambiguous.\n"
            "- The phone confirmation rule overrides the same-turn ask_anima instruction above: for requests involving "
            "a money transfer, deletion, production change, or external message, repeat the request and wait for the "
            "caller's explicit yes before calling ask_anima.\n"
            "- Never read URLs aloud; omit URLs from spoken replies.\n"
        )

    parts.append(
        "You are " + name + ", speaking to a person by voice. "
        "Respond in natural, conversational spoken language.\n\n"
        "MOST IMPORTANT — the ask_anima tool:\n"
        "- You cannot look anything up, check anything, or do any work "
        "yourself; only the ask_anima tool does work. Whenever the person asks "
        "for something to be done or looked into (調べる・確認する・チェックする・"
        "探す・やっておく・対応する・調査・実装・記憶の検索や保存・タスク化), "
        "call ask_anima in that same turn with the full request, then reply "
        "briefly like 'やっておくね'. Saying 「調べておくね」「確認しておきますね」 "
        "in words alone does NOTHING — the person will wait forever.\n"
        "- Do NOT call ask_anima when nothing was asked to be done: greetings "
        "(おはよう), thanks (ありがとう), small talk, feelings, opinions, or "
        "questions you can answer from what you already know. Just talk.\n"
        "- When several things are asked, call ask_anima once per item. When a "
        "result prefixed with [ask_anima完了] arrives, report it in your own "
        "words.\n"
        "- read_memory is read-only and safe to call any time you want to recall "
        "something (recent events, notes, procedures).\n\n"
        "Rules:\n"
        "- Reply in one or two short spoken sentences — aim for 60 characters, "
        f"never exceed {150 if channel == 'phone' else 120}. This is a voice conversation: one thing per turn, "
        "no lists, no long explanations unless explicitly asked.\n"
        f"{front_prompt_emotion_instruction(emotion_style)}"
        f"{phone_rules}"
        "- Write large numbers and years in speakable form "
        "(「三千八百億」「にせんさんねん」), not raw digits.\n"
        "- Every alphabet-spelled term — English words, acronyms, product/"
        "service names, people, command names — MUST be followed by its "
        "katakana reading in full-width parentheses, no exceptions: "
        "GitHub（ギットハブ）, API（エーピーアイ）, PR（ピーアール）, "
        "Claude Code（クロードコード）. The reading feeds only the TTS; "
        "subtitles show the original spelling.\n"
        "- Do NOT use Markdown (headings, bold, lists, code blocks).\n"
        "- Never promise work (やっておく／調べておく／確認しておく) without "
        "having called ask_anima in the same turn.\n"
        "- End every reply with exactly one emotion tag on its own final line:\n"
        '  <!-- emotion: {"emotion": "<emotion>"} -->\n'
        f"  (emotions: {_VOICE_FRONT_EMOTION_NAMES}; prefer a non-neutral one "
        "unless neutral fits best)"
    )

    return "\n\n".join(p for p in parts if p)


def inject_shortterm(
    base_prompt: str,
    shortterm: ShortTermMemory,
) -> str:
    """Append short-term memory content to the system prompt."""
    md_content = shortterm.render_for_injection()
    if not md_content:
        return base_prompt
    return base_prompt + "\n\n---\n\n" + md_content
