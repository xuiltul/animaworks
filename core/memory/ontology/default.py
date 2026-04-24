# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""Pydantic models for entity / fact extraction results."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────

ENTITY_TYPES = Literal["Person", "Place", "Organization", "Concept", "Event", "Object", "Time"]

EDGE_TYPES = Literal[
    "WORKS_AT",
    "LIVES_IN",
    "KNOWS",
    "PREFERS",
    "SKILLED_IN",
    "PARTICIPATED_IN",
    "CREATED",
    "REPORTED",
    "DEPENDS_ON",
    "RELATES_TO",
]

EDGE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "WORKS_AT": "Employment / affiliation",
    "LIVES_IN": "Residence / location",
    "KNOWS": "Personal acquaintance",
    "PREFERS": "Preference / taste",
    "SKILLED_IN": "Skill / ability",
    "PARTICIPATED_IN": "Participation / involvement",
    "CREATED": "Creation / authorship",
    "REPORTED": "Report / notification",
    "DEPENDS_ON": "Dependency",
    "RELATES_TO": "General (fallback)",
}

DEFAULT_EDGE_TYPE: str = "RELATES_TO"

# ── Entity models ──────────────────────────────────────────


class ExtractedEntity(BaseModel):
    """A single entity extracted from text."""

    name: str = Field(..., description="Canonical name")
    entity_type: ENTITY_TYPES = Field(default="Concept")
    summary: str = Field(default="", description="1-2 sentence summary")


class ExtractedFact(BaseModel):
    """A relationship between two entities."""

    source_entity: str = Field(..., description="Source entity name")
    target_entity: str = Field(..., description="Target entity name")
    fact: str = Field(..., description="Natural language relationship description")
    valid_at: str | None = Field(default=None, description="ISO datetime when fact became true")
    edge_type: str = Field(default="RELATES_TO", description="Relationship type from EDGE_TYPES")


# ── Extraction results ────────────────────────────────────


class EntityExtractionResult(BaseModel):
    """LLM response for entity extraction."""

    entities: list[ExtractedEntity] = Field(default_factory=list)


class FactExtractionResult(BaseModel):
    """LLM response for fact extraction."""

    facts: list[ExtractedFact] = Field(default_factory=list)
