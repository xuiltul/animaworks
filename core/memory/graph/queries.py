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
  entity_type: $entity_type,
  summary: $summary,
  group_id: $group_id,
  created_at: datetime($created_at),
  name_embedding: $name_embedding
})
RETURN e.uuid AS uuid
"""

UPDATE_ENTITY_SUMMARY = """
MATCH (e:Entity {uuid: $uuid})
SET e.summary = $summary
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

# ── Entity Resolution ──────────

FIND_ENTITIES_BY_NAME = """
MATCH (e:Entity)
WHERE e.group_id = $group_id
  AND e.name =~ $name_pattern
RETURN e.uuid AS uuid, e.name AS name, e.summary AS summary, e.entity_type AS entity_type
LIMIT $limit
"""

FIND_ENTITIES_BY_VECTOR = """
CALL db.index.vector.queryNodes('entity_name_embedding', $top_k, $embedding)
YIELD node, score
WHERE node.group_id = $group_id
  AND score >= $min_score
RETURN node.uuid AS uuid, node.name AS name, node.summary AS summary, node.entity_type AS entity_type, score
"""

UPDATE_ENTITY_SUMMARY = """
MATCH (e:Entity {uuid: $uuid})
SET e.summary = $summary
"""

REDIRECT_MENTIONS = """
MATCH (ep:Episode)-[old:MENTIONS]->(old_entity:Entity {uuid: $old_uuid})
MATCH (new_entity:Entity {uuid: $new_uuid})
CREATE (ep)-[:MENTIONS {uuid: old.uuid, created_at: old.created_at}]->(new_entity)
DELETE old
"""

REDIRECT_OUTGOING_FACTS = """
MATCH (old_entity:Entity {uuid: $old_uuid})-[old:RELATES_TO]->(target:Entity)
MATCH (new_entity:Entity {uuid: $new_uuid})
CREATE (new_entity)-[r:RELATES_TO {
  uuid: old.uuid, fact: old.fact, fact_embedding: old.fact_embedding,
  group_id: old.group_id, created_at: old.created_at, valid_at: old.valid_at,
  invalid_at: old.invalid_at, expired_at: old.expired_at,
  source_episode_uuids: old.source_episode_uuids
}]->(target)
DELETE old
"""

REDIRECT_INCOMING_FACTS = """
MATCH (source:Entity)-[old:RELATES_TO]->(old_entity:Entity {uuid: $old_uuid})
MATCH (new_entity:Entity {uuid: $new_uuid})
CREATE (source)-[r:RELATES_TO {
  uuid: old.uuid, fact: old.fact, fact_embedding: old.fact_embedding,
  group_id: old.group_id, created_at: old.created_at, valid_at: old.valid_at,
  invalid_at: old.invalid_at, expired_at: old.expired_at,
  source_episode_uuids: old.source_episode_uuids
}]->(new_entity)
DELETE old
"""

DELETE_ENTITY = """
MATCH (e:Entity {uuid: $uuid})
DETACH DELETE e
"""

# ── Temporal Fact queries ──────────

FIND_ACTIVE_FACTS_FOR_PAIR = """
MATCH (s:Entity {uuid: $source_uuid})-[r:RELATES_TO]->(t:Entity {uuid: $target_uuid})
WHERE r.invalid_at IS NULL
  AND r.expired_at IS NULL
  AND r.uuid <> $new_fact_uuid
  AND r.valid_at <= datetime($new_valid_at)
RETURN r.uuid AS uuid, r.fact AS fact, toString(r.valid_at) AS valid_at
"""

FIND_ACTIVE_FACTS_FOR_PAIR_REVERSE = """
MATCH (s:Entity {uuid: $target_uuid})-[r:RELATES_TO]->(t:Entity {uuid: $source_uuid})
WHERE r.invalid_at IS NULL
  AND r.expired_at IS NULL
  AND r.uuid <> $new_fact_uuid
  AND r.valid_at <= datetime($new_valid_at)
RETURN r.uuid AS uuid, r.fact AS fact, toString(r.valid_at) AS valid_at
"""

INVALIDATE_FACT = """
MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
WHERE r.group_id = $group_id
SET r.invalid_at = datetime($invalid_at)
"""

EXPIRE_FACT = """
MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
WHERE r.group_id = $group_id
SET r.expired_at = datetime($expired_at)
"""

FIND_VALID_FACTS_BY_GROUP = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.group_id = $group_id
  AND (r.invalid_at IS NULL OR r.invalid_at > datetime($as_of_time))
  AND (r.expired_at IS NULL OR r.expired_at > datetime($as_of_time))
RETURN r.uuid AS uuid, r.fact AS fact,
       s.name AS source_name, t.name AS target_name,
       toString(r.valid_at) AS valid_at,
       r.group_id AS group_id
ORDER BY r.valid_at DESC
LIMIT $limit
"""

# ── Hybrid Search queries ──────────

VECTOR_SEARCH_FACTS = """
CALL db.index.vector.queryRelationships('fact_embedding', $top_k, $embedding)
YIELD relationship, score
WHERE relationship.group_id = $group_id
  AND (relationship.invalid_at IS NULL OR relationship.invalid_at > datetime($as_of_time))
  AND (relationship.expired_at IS NULL OR relationship.expired_at > datetime($as_of_time))
WITH relationship AS r, score,
     startNode(relationship) AS s, endNode(relationship) AS t
RETURN r.uuid AS uuid, r.fact AS fact, s.name AS source_name, t.name AS target_name,
       toString(r.valid_at) AS valid_at, score
"""

VECTOR_SEARCH_ENTITIES = """
CALL db.index.vector.queryNodes('entity_name_embedding', $top_k, $embedding)
YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid AS uuid, node.name AS name, node.summary AS summary,
       node.entity_type AS entity_type, score
"""

FULLTEXT_SEARCH_FACTS = """
CALL db.index.fulltext.queryRelationships('fact_fulltext', $query, {limit: $top_k})
YIELD relationship, score
WHERE relationship.group_id = $group_id
  AND (relationship.invalid_at IS NULL OR relationship.invalid_at > datetime($as_of_time))
  AND (relationship.expired_at IS NULL OR relationship.expired_at > datetime($as_of_time))
WITH relationship AS r, score,
     startNode(relationship) AS s, endNode(relationship) AS t
RETURN r.uuid AS uuid, r.fact AS fact, s.name AS source_name, t.name AS target_name,
       toString(r.valid_at) AS valid_at, score
"""

FULLTEXT_SEARCH_ENTITIES = """
CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query, {limit: $top_k})
YIELD node, score
WHERE node.group_id = $group_id
RETURN node.uuid AS uuid, node.name AS name, node.summary AS summary,
       node.entity_type AS entity_type, score
"""

BFS_FACTS_FROM_ENTITY = """
MATCH (seed:Entity {uuid: $entity_uuid})-[:RELATES_TO*1..2]-(related:Entity)
WHERE related.group_id = $group_id
WITH DISTINCT related
MATCH (related)-[r:RELATES_TO]-(other:Entity)
WHERE r.group_id = $group_id
  AND (r.invalid_at IS NULL OR r.invalid_at > datetime($as_of_time))
  AND (r.expired_at IS NULL OR r.expired_at > datetime($as_of_time))
WITH r, startNode(r) AS s, endNode(r) AS t
RETURN r.uuid AS uuid, r.fact AS fact, s.name AS source_name, t.name AS target_name,
       toString(r.valid_at) AS valid_at
LIMIT $limit
"""

# ── Community ──────────

FETCH_ENTITIES_FOR_COMMUNITY = """
MATCH (e:Entity)
WHERE e.group_id = $group_id
RETURN e.uuid AS uuid, e.name AS name, e.summary AS summary
"""

FETCH_EDGES_FOR_COMMUNITY = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.group_id = $group_id
  AND r.invalid_at IS NULL
RETURN s.uuid AS source_uuid, t.uuid AS target_uuid, r.uuid AS edge_uuid
"""

DELETE_COMMUNITIES_BY_GROUP = """
MATCH (c:Community)
WHERE c.group_id = $group_id
DETACH DELETE c
"""

CREATE_COMMUNITY = """
CREATE (c:Community {
  uuid: $uuid,
  name: $name,
  summary: $summary,
  group_id: $group_id,
  created_at: datetime($created_at)
})
RETURN c.uuid AS uuid
"""

CREATE_HAS_MEMBER = """
MATCH (c:Community {uuid: $community_uuid}), (e:Entity {uuid: $entity_uuid})
CREATE (c)-[:HAS_MEMBER]->(e)
"""

FIND_COMMUNITY_FOR_ENTITY = """
MATCH (c:Community)-[:HAS_MEMBER]->(e:Entity {uuid: $entity_uuid})
RETURN c.uuid AS community_uuid, c.name AS name, c.summary AS summary
"""

FIND_ENTITY_NEIGHBORS = """
MATCH (e:Entity {uuid: $entity_uuid})-[:RELATES_TO]-(neighbor:Entity)
WHERE neighbor.group_id = $group_id
RETURN DISTINCT neighbor.uuid AS uuid
LIMIT $limit
"""

SEARCH_COMMUNITIES = """
MATCH (c:Community)
WHERE c.group_id = $group_id
RETURN c.uuid AS uuid, c.name AS name, c.summary AS summary
ORDER BY c.created_at DESC
LIMIT $limit
"""

FIND_RECENT_FACTS = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.group_id = $group_id
  AND r.invalid_at IS NULL
  AND r.expired_at IS NULL
  AND r.created_at >= datetime($since)
RETURN r.uuid AS uuid, r.fact AS fact, s.name AS source_name, t.name AS target_name,
       toString(r.valid_at) AS valid_at, toString(r.created_at) AS created_at
ORDER BY r.created_at DESC
LIMIT $limit
"""
