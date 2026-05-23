from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Retrieval confidence gate — abstain when evidence is weak."""

from dataclasses import dataclass
from typing import Any


@dataclass
class GateResult:
    """Output of confidence gating."""

    candidates: list[dict[str, Any]]
    abstain: bool = False
    reason: str = ""


def apply_confidence_gate(
    candidates: list[dict[str, Any]],
    *,
    threshold: float,
    score_field: str = "score",
) -> GateResult:
    """Return empty candidates when max score is below *threshold*."""
    if not candidates:
        return GateResult(candidates=[], abstain=True, reason="low_confidence")

    max_score = max(float(c.get(score_field, 0.0) or 0.0) for c in candidates)
    if max_score < threshold:
        return GateResult(candidates=[], abstain=True, reason="low_confidence")

    return GateResult(candidates=candidates, abstain=False)
