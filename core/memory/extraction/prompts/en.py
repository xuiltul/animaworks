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

# ── Entity deduplication ──────────────────────────────────

DEDUPE_SYSTEM = (
    "You are an entity deduplication agent. Determine whether a new entity is the same as any existing candidate."
)

DEDUPE_USER = """## New Entity
Name: {new_entity_name}
Type: {new_entity_type}
Summary: {new_entity_summary}

## Existing Entity Candidates
{candidates_json}

## Instructions
If the new entity refers to the same real-world entity as one of the candidates, return its UUID and a merged summary in JSON format.
If it is not a duplicate, or you are not confident, set duplicate_of_uuid to null.

```json
{{"duplicate_of_uuid": "existing UUID or null", "merged_summary": "merged 1-2 sentence description"}}
```"""

# ── Fact invalidation ─────────────────────────────────────

INVALIDATE_SYSTEM = (
    "You are a fact contradiction detection agent. Determine whether a new fact contradicts any existing facts."
)

INVALIDATE_USER = """## New Fact
{new_fact}

## Existing Facts (still valid)
{existing_facts_json}

## Instructions
If the new fact contradicts any existing facts, return the UUIDs of the contradicted facts in a list.
A contradiction means both facts cannot be true at the same time. Complementary information is NOT a contradiction.
If you are not confident, return an empty list.

```json
{{"contradicted_uuids": ["UUID of contradicted fact", ...]}}
```"""

# ── Community summarization ───────────────────────────────

COMMUNITY_SYSTEM = "You are a group analysis agent. Assign a name and summary to a group of related entities."

COMMUNITY_USER = """## Group Members
{members}

## Instructions
Based on the common theme of the members above, assign a name and summary to this group.

```json
{{"name": "group name (short)", "summary": "1-2 sentence description of this group"}}
```"""
