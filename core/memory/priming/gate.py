from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic priming gate for cue-oriented memory injection."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.priming.result import PrimingResult


class PrimingRenderMode(StrEnum):
    """How a memory candidate should be rendered in priming."""

    ANCHOR = "anchor"
    GUARDRAIL = "guardrail"
    POINTER = "pointer"
    EVIDENCE = "evidence"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class MemoryCandidate:
    """A channel-level memory candidate considered by the priming gate."""

    channel: str
    content: str
    trust: str = "medium"
    risk_tags: frozenset[str] = frozenset()
    score_features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PrimingGateDecision:
    """Gate decision for one priming candidate."""

    channel: str
    visible: bool
    render_mode: PrimingRenderMode
    reason: str
    priority: int
    budget_chars: int | None = None
    require_search_before_action: bool = False


@dataclass(frozen=True)
class PrimingPlan:
    """Deterministic gate plan for a priming run."""

    channel_decisions: dict[str, PrimingGateDecision]
    evidence_mode: bool
    require_search_before_action: bool
    risk_tags: frozenset[str] = frozenset()


EXPLICIT_RECALL_TERMS = frozenset(
    {
        "前に",
        "昨日",
        "続き",
        "あの件",
        "以前",
        "思い出",
        "覚えて",
        "remember",
        "previous",
        "yesterday",
        "continue",
        "that issue",
        "이전",
        "어제",
        "계속",
        "기억",
    }
)

EXTERNAL_ACTION_TERMS = frozenset(
    {
        "gmail",
        "メール",
        "mail",
        "下書き",
        "draft",
        "返信",
        "reply",
        "送信",
        "send",
        "chatwork",
        "slack",
        "call_human",
    }
)

EVIDENCE_REQUEST_TERMS = frozenset(
    {
        "根拠",
        "原文",
        "引用",
        "source",
        "evidence",
        "quote",
    }
)

RESUME_TERMS = frozenset(
    {
        "再開",
        "続き",
        "resume",
        "continue",
        "task",
        "재개",
        "계속",
    }
)

GUARDRAIL_TERMS = frozenset(
    {
        "承認",
        "approval",
        "二重送信",
        "duplicate send",
        "機密",
        "confidential",
        "[ACTION-RULE]",
        "승인",
        "기밀",
        "중복 전송",
    }
)

_POINTER_MARKERS = (
    "read_memory_file(path=",
    "-> read_memory_file",
    "→ read_memory_file",
)

_ALWAYS_VISIBLE_CHANNELS = frozenset(
    {
        "sender_profile",
        "recent_activity",
        "pending_tasks",
        "recent_outbound",
        "pending_human_notifications",
    }
)

_RELATED_CHANNELS = frozenset(
    {
        "related_knowledge",
        "related_knowledge_untrusted",
        "episodes",
    }
)


def build_candidates_from_result(result: PrimingResult) -> list[MemoryCandidate]:
    """Build channel-level gate candidates from a PrimingResult."""

    candidates: list[MemoryCandidate] = []
    field_specs = (
        ("sender_profile", "medium"),
        ("recent_activity", "untrusted"),
        ("related_knowledge", "medium"),
        ("related_knowledge_untrusted", "untrusted"),
        ("pending_tasks", "medium"),
        ("recent_outbound", "trusted"),
        ("episodes", "medium"),
        ("pending_human_notifications", "medium"),
        ("graph_context", "medium"),
    )
    for channel, trust in field_specs:
        content = str(getattr(result, channel, "") or "")
        if not content:
            continue
        candidates.append(
            MemoryCandidate(
                channel=channel,
                content=content,
                trust=trust,
                risk_tags=classify_text_risk_tags(content),
            )
        )
    return candidates


def build_priming_plan(
    message: str,
    channel: str,
    intent: str,
    candidates: Sequence[MemoryCandidate],
    recent_human_messages: Sequence[str] | None = None,
) -> PrimingPlan:
    """Build a deterministic gate plan for the current priming candidates."""

    message_risk_tags = classify_risk_tags(message, candidates, recent_human_messages=recent_human_messages)
    message_evidence_mode = evidence_needed(
        message,
        channel,
        message_risk_tags,
        candidates,
        recent_human_messages=recent_human_messages,
    )
    require_search = "external_action" in message_risk_tags or "evidence_request" in message_risk_tags

    decisions: dict[str, PrimingGateDecision] = {}
    for candidate in candidates:
        decision = decide_candidate(
            candidate,
            evidence_mode=message_evidence_mode,
            risk_tags=message_risk_tags,
            message=message,
            channel=channel,
            intent=intent,
            require_search_before_action=require_search,
        )
        decisions[candidate.channel] = decision

    return PrimingPlan(
        channel_decisions=decisions,
        evidence_mode=message_evidence_mode,
        require_search_before_action=require_search,
        risk_tags=message_risk_tags,
    )


def apply_priming_plan(result: PrimingResult, plan: PrimingPlan) -> PrimingResult:
    """Return a PrimingResult with suppressed channels removed."""

    updates: dict[str, Any] = {"gate_plan": plan}
    for channel, decision in plan.channel_decisions.items():
        if not decision.visible:
            updates[channel] = ""
    return replace(result, **updates)


def decide_candidate(
    candidate: MemoryCandidate,
    *,
    evidence_mode: bool,
    risk_tags: frozenset[str],
    message: str,
    channel: str,
    intent: str,
    require_search_before_action: bool,
) -> PrimingGateDecision:
    """Decide whether and how to render one candidate."""

    if not candidate.content.strip():
        return PrimingGateDecision(
            channel=candidate.channel,
            visible=False,
            render_mode=PrimingRenderMode.SUPPRESS,
            reason="empty_candidate",
            priority=99,
        )

    if _is_guardrail_candidate(candidate, risk_tags):
        return PrimingGateDecision(
            channel=candidate.channel,
            visible=True,
            render_mode=PrimingRenderMode.GUARDRAIL,
            reason="guardrail_risk_match",
            priority=1,
            require_search_before_action=require_search_before_action,
        )

    if candidate.channel in _ALWAYS_VISIBLE_CHANNELS:
        return PrimingGateDecision(
            channel=candidate.channel,
            visible=True,
            render_mode=PrimingRenderMode.ANCHOR,
            reason="always_visible_channel",
            priority=2,
            require_search_before_action=require_search_before_action,
        )

    if candidate.channel in _RELATED_CHANNELS:
        if is_pointer_like(candidate.content):
            return PrimingGateDecision(
                channel=candidate.channel,
                visible=True,
                render_mode=PrimingRenderMode.POINTER,
                reason="pointer_like_related_memory",
                priority=3,
                require_search_before_action=require_search_before_action,
            )
        if evidence_mode:
            return PrimingGateDecision(
                channel=candidate.channel,
                visible=True,
                render_mode=PrimingRenderMode.EVIDENCE,
                reason="evidence_mode_related_memory",
                priority=4,
                require_search_before_action=require_search_before_action,
            )
        return PrimingGateDecision(
            channel=candidate.channel,
            visible=False,
            render_mode=PrimingRenderMode.SUPPRESS,
            reason="non_pointer_related_memory_without_evidence_mode",
            priority=80,
        )

    if candidate.channel == "graph_context" and _is_empty_background_query(message, channel):
        return PrimingGateDecision(
            channel=candidate.channel,
            visible=False,
            render_mode=PrimingRenderMode.SUPPRESS,
            reason="empty_background_graph_query",
            priority=80,
        )

    return PrimingGateDecision(
        channel=candidate.channel,
        visible=True,
        render_mode=PrimingRenderMode.POINTER if is_pointer_like(candidate.content) else PrimingRenderMode.ANCHOR,
        reason="pass_through",
        priority=5,
        require_search_before_action=require_search_before_action,
    )


def classify_risk_tags(
    message: str,
    candidates: Sequence[MemoryCandidate],
    *,
    recent_human_messages: Sequence[str] | None = None,
) -> frozenset[str]:
    """Classify deterministic risk/evidence tags for mixed Japanese/English text."""

    text_parts = [message or ""]
    if recent_human_messages:
        text_parts.extend(str(m) for m in recent_human_messages if m)
    return classify_text_risk_tags("\n".join(text_parts))


def classify_text_risk_tags(text: str) -> frozenset[str]:
    """Classify deterministic risk/evidence tags for one text blob."""

    haystack = _normalize_search_text(text)

    tags: set[str] = set()
    if _contains_any(haystack, EXPLICIT_RECALL_TERMS):
        tags.add("explicit_recall")
    if _contains_any(haystack, EXTERNAL_ACTION_TERMS):
        tags.add("external_action")
    if _contains_any(haystack, EVIDENCE_REQUEST_TERMS):
        tags.add("evidence_request")
    if _contains_any(haystack, RESUME_TERMS):
        tags.add("resume")
    if _contains_any(haystack, GUARDRAIL_TERMS):
        tags.add("guardrail")
    if _contains_any(haystack, frozenset({"duplicate send", "二重送信", "중복 전송"})):
        tags.add("duplicate_prevention")
    if _contains_any(haystack, frozenset({"confidential", "機密", "기밀"})):
        tags.add("confidentiality")
    return frozenset(tags)


def evidence_needed(
    message: str,
    channel: str,
    risk_tags: frozenset[str],
    candidates: Sequence[MemoryCandidate],
    *,
    recent_human_messages: Sequence[str] | None = None,
) -> bool:
    """Return whether raw evidence/excerpts are allowed for this priming turn."""

    if risk_tags & {"explicit_recall", "external_action", "evidence_request"}:
        return True
    return "resume" in risk_tags and any(
        c.channel in {"pending_tasks", "pending_human_notifications"} for c in candidates
    )


def is_pointer_like(content: str) -> bool:
    """Return True if content is already a pointer cue instead of raw payload."""

    return any(marker in content for marker in _POINTER_MARKERS)


def _is_guardrail_candidate(candidate: MemoryCandidate, risk_tags: frozenset[str]) -> bool:
    if not (risk_tags & {"external_action", "guardrail", "duplicate_prevention", "confidentiality"}):
        return False
    if candidate.channel in {"related_knowledge", "related_knowledge_untrusted", "pending_tasks", "recent_outbound"}:
        content = _normalize_search_text(candidate.content)
        # Candidate text is only corroborating evidence. Everyday words in a
        # retrieved document must never create message-level action risk.
        return _contains_any(content, GUARDRAIL_TERMS)
    return False


def _is_empty_background_query(message: str, channel: str) -> bool:
    return channel in {"heartbeat", "cron", "inbox", "task"} and not str(message or "").strip()


def _contains_any(haystack: str, terms: frozenset[str]) -> bool:
    return any(_normalize_search_text(term) in haystack for term in terms)


def _normalize_search_text(text: str) -> str:
    return str(text or "").casefold()
