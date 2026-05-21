from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Hermes-compatible Pydantic models for skill metadata."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from core.schemas import SkillMeta


# ── Usage Event Models ─────────────────────────────────────


class SkillUsageEventType(str, Enum):  # noqa: UP042
    """Classification of skill/procedure usage events."""

    view = "view"
    use = "use"
    success = "success"
    failure = "failure"
    patch = "patch"
    create = "create"


class SkillUsageEvent(BaseModel):
    """Single usage event appended to skill_usage.jsonl."""

    ts: str
    skill_name: str
    event_type: SkillUsageEventType
    is_common: bool = False
    notes: str | None = None


class SkillUsageStats(BaseModel):
    """Aggregated statistics for a single skill, computed from event replay."""

    skill_name: str
    view_count: int = 0
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    patch_count: int = 0
    create_count: int = 0
    created_at: str | None = None
    last_used_at: str | None = None
    is_common: bool = False


# ── Enumerations ────────────────────────────────────────────


class SkillTrustLevel(str, Enum):  # noqa: UP042
    """Trust tier describing provenance and policy for a skill."""

    builtin = "builtin"
    official = "official"
    trusted = "trusted"
    community = "community"
    untrusted = "untrusted"
    quarantine = "quarantine"
    blocked = "blocked"


class SkillScanVerdict(str, Enum):  # noqa: UP042
    """Outcome classification from automated skill security review."""

    safe = "safe"
    caution = "caution"
    warn = "warn"
    dangerous = "dangerous"


class SkillLifecycleState(str, Enum):  # noqa: UP042
    """Curator-managed skill lifecycle state."""

    active = "active"
    review = "review"
    stale = "stale"
    archived = "archived"
    blocked = "blocked"
    deleted = "deleted"


class SkillCuratorEventType(str, Enum):  # noqa: UP042
    """Append-only event types emitted by SkillCurator."""

    state_changed = "state_changed"
    merge_proposed = "merge_proposed"
    reference_rewrite_proposed = "reference_rewrite_proposed"


# ── Constants ──────────────────────────────────────────────

_SEVERITY_RANKS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

# ── Supporting models ───────────────────────────────────────


class SkillSource(BaseModel):
    """Where a skill came from and how it is attributed."""

    type: str = "local"
    identifier: str | None = None
    owner_anima: str | None = None
    origin: str | None = None


class ThreatPattern(BaseModel):
    """A single regex-based threat detection rule."""

    name: str
    pattern: str
    severity: str  # "low", "medium", "high", "critical"
    category: str

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_RANKS.get(self.severity, 0)


class ScanFinding(BaseModel):
    """A single finding from the security scanner."""

    pattern_name: str
    category: str
    severity: str
    line_number: int | None = None
    file_path: str | None = None
    matched_text: str = ""

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_RANKS.get(self.severity, 0)


class ScanResult(BaseModel):
    """Complete result of scanning a skill directory."""

    verdict: SkillScanVerdict = SkillScanVerdict.safe
    findings: list[ScanFinding] = Field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    size_violations: list[str] = Field(default_factory=list)


class SkillSecurityScan(BaseModel):
    """Security scan summary attached to skill metadata."""

    verdict: SkillScanVerdict = SkillScanVerdict.safe
    scan_status: str = "not_scanned"
    findings: list[dict] = Field(default_factory=list)
    scanned_at: datetime | None = None
    scanner_version: str | None = None


class SkillCuratorEvent(BaseModel):
    """Single append-only lifecycle event for skill curator state."""

    ts: str
    event_type: SkillCuratorEventType | str
    skill_name: str
    from_state: SkillLifecycleState | None = None
    to_state: SkillLifecycleState | None = None
    reason: str = ""
    actor: str = "curator"
    related_skill: str | None = None
    absorbed_into: str | None = None
    proposal_path: str | None = None


class SkillBundle(BaseModel):
    """A staged skill bundle resolved from an import source."""

    source_type: str
    source_identifier: str
    staging_path: str
    skill_md_path: str
    resolved_commit: str | None = None


class SkillHubLockEntry(BaseModel):
    """Append-only provenance lock entry for Skill Hub operations."""

    ts: str
    action: str
    skill_name: str
    target: str
    source_type: str | None = None
    source_identifier: str | None = None
    resolved_commit: str | None = None
    scan_verdict: str | None = None
    installed_path: str | None = None
    actor: str = "cli"
    reason: str = ""


def _coerce_str_list(value: object) -> list[str]:
    """Normalize YAML scalar/list metadata fields into a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


class SkillRiskMetadata(BaseModel):
    """Routing and approval hints for a skill.

    These fields are advisory only. They help route and present skill hints,
    but permission enforcement remains the responsibility of tool policy and
    sandbox layers.
    """

    read_only: bool = False
    destructive: bool = False
    external_send: bool = False
    handles_untrusted_data: bool = False
    credential: bool = False
    production: bool = False
    billing: bool = False
    private_data: bool = False
    requires_human_approval: bool = False
    open_world: bool = False


class SkillRoutingMetadata(BaseModel):
    """Metadata used by the shadow skill router."""

    use_when: list[str] = Field(default_factory=list)
    do_not_use_when: list[str] = Field(default_factory=list)
    trigger_phrases: list[str] = Field(default_factory=list)
    negative_phrases: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    routing_examples: list[str] = Field(default_factory=list)
    risk: SkillRiskMetadata = Field(default_factory=SkillRiskMetadata)

    @field_validator(
        "use_when",
        "do_not_use_when",
        "trigger_phrases",
        "negative_phrases",
        "domains",
        "routing_examples",
        mode="before",
    )
    @classmethod
    def _normalize_list_fields(cls, value: object) -> list[str]:
        return _coerce_str_list(value)


class SkillPolicyMetadata(BaseModel):
    """How strongly a skill may influence prompt context and behavior."""

    use_mode: Literal["primary_guidance", "candidate_hint"] = "primary_guidance"
    injection: Literal["body_allowed", "pointer_preferred", "blocked"] = "body_allowed"


# ── SkillMetadata ─────────────────────────────────────────


class SkillMetadata(BaseModel):
    """Full skill metadata record aligned with Hermes skill manifests."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    requires_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    use_when: list[str] = Field(default_factory=list)
    do_not_use_when: list[str] = Field(default_factory=list)
    trigger_phrases: list[str] = Field(default_factory=list)
    negative_phrases: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    routing_examples: list[str] = Field(default_factory=list)
    risk: SkillRiskMetadata = Field(default_factory=SkillRiskMetadata)
    routing: SkillRoutingMetadata = Field(default_factory=SkillRoutingMetadata)
    skill_policy: SkillPolicyMetadata = Field(default_factory=SkillPolicyMetadata)
    trust_level: SkillTrustLevel = SkillTrustLevel.trusted
    source: SkillSource = Field(default_factory=SkillSource)
    version: int = 1
    promotion_status: str | None = None
    lifecycle_state: SkillLifecycleState = SkillLifecycleState.active
    absorbed_into: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_callback_id: str | None = None
    trusted_by: str | None = None
    trusted_at: datetime | None = None
    trust_reason: str | None = None
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    patch_count: int = 0
    last_used_at: datetime | None = None
    last_updated_at: datetime | None = None
    pinned: bool = False
    protected: bool = False
    security: SkillSecurityScan = Field(default_factory=SkillSecurityScan)
    path: Path | None = None
    is_common: bool = False
    is_procedure: bool = False

    @field_validator(
        "tags",
        "platforms",
        "requires_tools",
        "allowed_tools",
        "use_when",
        "do_not_use_when",
        "trigger_phrases",
        "negative_phrases",
        "domains",
        "routing_examples",
        mode="before",
    )
    @classmethod
    def _normalize_list_fields(cls, value: object) -> list[str]:
        return _coerce_str_list(value)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source_field(cls, value: object) -> object:
        """Accept legacy scalar ``source`` frontmatter values.

        Older procedure / skill files frequently stored provenance as a simple
        string such as ``activity_log`` or ``error_trace_analysis``. The newer
        schema expects a ``SkillSource`` object, so we normalize a scalar into
        ``{\"type\": <value>}`` to preserve backward compatibility.
        """
        if value is None:
            return value
        if isinstance(value, SkillSource):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            return {"type": stripped} if stripped else None
        return value

    def to_legacy(self) -> SkillMeta:
        """Convert to the legacy ``SkillMeta`` dataclass used elsewhere in core.

        Returns:
            ``SkillMeta`` suitable for code paths that still expect frontmatter-
            style metadata only.
        """
        from core.schemas import SkillMeta

        return SkillMeta(
            name=self.name,
            description=self.description,
            path=self.path or Path("."),
            is_common=self.is_common,
            allowed_tools=self.allowed_tools,
        )
