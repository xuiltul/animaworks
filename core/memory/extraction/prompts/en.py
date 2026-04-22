# Copyright 2026 AnimaWorks
# Licensed under the Apache License, Version 2.0
"""English prompts for entity / fact extraction."""

from __future__ import annotations

# ── Entity extraction ──────────────────────────────────────

ENTITY_SYSTEM = (
    "You are an information extraction agent. "
    "Extract entities (Person, Place, Organization, Concept, Event, Object, Time) "
    "from the given text in JSON format."
)

ENTITY_USER = """## Text
{content}

## Known Entities (reference)
{previous_entities}

## Instructions
Extract entities from the text above and return them in the following JSON format. Return an empty list if no entities are found.

```json
{{
  "entities": [
    {{"name": "canonical name", "entity_type": "Person|Place|Organization|Concept|Event|Object|Time", "summary": "1-2 sentence description"}}
  ]
}}
```"""

# ── Fact extraction ────────────────────────────────────────

FACT_SYSTEM = "You are a relationship extraction agent. Extract relationships between entity pairs in JSON format."

FACT_USER = """## Text
{content}

## Extracted Entities
{entities_json}

## Instructions
Extract relationships (facts) between the entities above and return them in the following JSON format. Return an empty list if no relationships are found.

```json
{{
  "facts": [
    {{"source_entity": "EntityA", "target_entity": "EntityB", "fact": "natural language description of relationship", "valid_at": "YYYY-MM-DDTHH:MM:SS or null"}}
  ]
}}
```"""
