# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Data schemas for memory performance evaluation.

Defines data structures for memory files, datasets, scenarios, and ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ── Memory Base Structures ──────────────────────────────────────────────────


@dataclass
class MemoryFile:
    """Individual memory file in a dataset."""

    path: Path
    content: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Get filename without path."""
        return self.path.name

    @property
    def category(self) -> str:
        """Get memory category (knowledge/episodes/skills)."""
        # Assumes structure like: .../knowledge/file.md
        return self.path.parent.name


@dataclass
class MemoryBase:
    """Complete memory base for a single domain and size."""

    domain: Literal["business", "tech_support", "education"]
    size: Literal["small", "medium", "large"]
    knowledge_files: list[MemoryFile] = field(default_factory=list)
    episode_files: list[MemoryFile] = field(default_factory=list)
    skill_files: list[MemoryFile] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total token count across all memory files."""
        return (
            sum(f.tokens for f in self.knowledge_files) +
            sum(f.tokens for f in self.episode_files) +
            sum(f.tokens for f in self.skill_files)
        )

    @property
    def total_files(self) -> int:
        """Total number of files."""
        return len(self.knowledge_files) + len(self.episode_files) + len(self.skill_files)

    @property
    def all_files(self) -> list[MemoryFile]:
        """Get all files as a single list."""
        return self.knowledge_files + self.episode_files + self.skill_files


# ── Scenario Structures ──────────────────────────────────────────────────────


@dataclass
class ConversationTurn:
    """Single turn in a conversation scenario."""

    message: str
    relevant_memories: list[Path] = field(default_factory=list)
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """Conversation scenario for evaluation."""

    scenario_id: str
    scenario_type: Literal["factual", "episodic", "multihop", "long"]
    domain: Literal["business", "tech_support", "education"]
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        """Get number of turns in scenario."""
        return len(self.turns)


# ── Ground Truth Structures ──────────────────────────────────────────────────


@dataclass
class RelevantMemory:
    """Annotated relevant memory for a query."""

    file_path: Path
    relevance: Literal["high", "medium", "low"]
    section: str | None = None
    notes: str = ""


@dataclass
class GroundTruth:
    """Ground truth annotation for a single query."""

    query: str
    scenario_id: str
    turn_index: int
    relevant_memories: list[RelevantMemory] = field(default_factory=list)
    irrelevant_memories: list[Path] = field(default_factory=list)
    annotator_id: str = ""
    timestamp: str = ""

    @property
    def high_relevance_files(self) -> list[Path]:
        """Get files marked as high relevance."""
        return [rm.file_path for rm in self.relevant_memories if rm.relevance == "high"]

    @property
    def all_relevant_files(self) -> list[Path]:
        """Get all relevant files regardless of relevance level."""
        return [rm.file_path for rm.file_path in self.relevant_memories]


@dataclass
class AnnotationSet:
    """Complete set of annotations from one annotator."""

    annotator_id: str
    annotations: dict[str, GroundTruth] = field(default_factory=dict)  # query_id -> GroundTruth
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_annotation(self, query_id: str, gt: GroundTruth) -> None:
        """Add a ground truth annotation."""
        self.annotations[query_id] = gt

    def get_annotation(self, query_id: str) -> GroundTruth | None:
        """Get annotation for a query."""
        return self.annotations.get(query_id)


# ── Size Configuration ──────────────────────────────────────────────────────


@dataclass
class SizeConfig:
    """Configuration for dataset sizes."""

    knowledge_count: int
    episode_count: int
    skill_count: int
    target_tokens: int

    @classmethod
    def get_config(cls, size: Literal["small", "medium", "large"]) -> SizeConfig:
        """Get predefined configuration for a size level."""
        configs = {
            "small": cls(
                knowledge_count=50,
                episode_count=30,
                skill_count=10,
                target_tokens=50_000
            ),
            "medium": cls(
                knowledge_count=500,
                episode_count=300,
                skill_count=100,
                target_tokens=500_000
            ),
            "large": cls(
                knowledge_count=5000,
                episode_count=3000,
                skill_count=1000,
                target_tokens=5_000_000
            )
        }
        return configs[size]


# ── Scenario Type Configuration ──────────────────────────────────────────────


@dataclass
class ScenarioTypeConfig:
    """Configuration for scenario type distribution."""

    factual_ratio: float = 0.40
    episodic_ratio: float = 0.30
    multihop_ratio: float = 0.20
    long_ratio: float = 0.10

    def get_counts(self, total: int) -> dict[str, int]:
        """Calculate scenario counts for each type."""
        return {
            "factual": int(total * self.factual_ratio),
            "episodic": int(total * self.episodic_ratio),
            "multihop": int(total * self.multihop_ratio),
            "long": int(total * self.long_ratio)
        }
