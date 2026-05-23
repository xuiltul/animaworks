from __future__ import annotations

# AnimaWorks - Digital Anima Framework
"""Shared retrieval result types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalCandidate:
    """Normalized retrieval hit for pipeline stages."""

    content: str
    score: float
    source_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> RetrievalCandidate:
        """Build from legacy search result dict."""
        meta = {
            k: v
            for k, v in item.items()
            if k not in ("content", "score", "source_file")
        }
        return cls(
            content=str(item.get("content", "") or ""),
            score=float(item.get("score", 0.0) or 0.0),
            source_file=str(item.get("source_file", "") or ""),
            metadata=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert back to legacy search result dict."""
        out: dict[str, Any] = {
            "content": self.content,
            "score": self.score,
            "source_file": self.source_file,
        }
        out.update(self.metadata)
        return out
