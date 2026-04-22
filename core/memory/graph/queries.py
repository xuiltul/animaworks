from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable Cypher query templates for the Neo4j graph backend."""

# ── Node operations ──────────

DELETE_ALL_BY_GROUP = "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n"

COUNT_NODES_BY_GROUP = """
MATCH (n)
WHERE n.group_id = $group_id
RETURN labels(n)[0] AS label, count(n) AS cnt
"""

COUNT_EDGES_BY_GROUP = """
MATCH ()-[r]->()
WHERE r.group_id = $group_id OR (EXISTS(r.group_id) = false)
RETURN type(r) AS rel_type, count(r) AS cnt
"""

# ── Episode ──────────

CREATE_EPISODE = """
CREATE (e:Episode {
  uuid: $uuid,
  content: $content,
  source: $source,
  source_description: $source_description,
  group_id: $group_id,
  created_at: datetime($created_at),
  valid_at: datetime($valid_at)
})
RETURN e.uuid AS uuid
"""

# ── Entity ──────────

CREATE_ENTITY = """
CREATE (e:Entity {
  uuid: $uuid,
  name: $name,
  summary: $summary,
  group_id: $group_id,
  created_at: datetime($created_at),
  name_embedding: $name_embedding
})
RETURN e.uuid AS uuid
"""

# ── RELATES_TO (Fact) ──────────

CREATE_FACT = """
MATCH (s:Entity {uuid: $source_uuid}), (t:Entity {uuid: $target_uuid})
CREATE (s)-[r:RELATES_TO {
  uuid: $uuid,
  fact: $fact,
  fact_embedding: $fact_embedding,
  group_id: $group_id,
  created_at: datetime($created_at),
  valid_at: datetime($valid_at),
  invalid_at: null,
  expired_at: null,
  source_episode_uuids: $source_episode_uuids
}]->(t)
RETURN r.uuid AS uuid
"""

# ── MENTIONS ──────────

CREATE_MENTION = """
MATCH (ep:Episode {uuid: $episode_uuid}), (en:Entity {uuid: $entity_uuid})
CREATE (ep)-[r:MENTIONS {uuid: $uuid, created_at: datetime($created_at)}]->(en)
RETURN r.uuid AS uuid
"""
