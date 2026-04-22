from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Community detection via Label Propagation + LLM summarization."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from core.memory.graph.driver import Neo4jDriver

logger = logging.getLogger(__name__)


# ── Data ──────────


@dataclass
class Community:
    """A detected community of entities."""

    uuid: str
    name: str
    summary: str
    member_uuids: list[str]
    member_names: list[str]


# ── Detector ──────────


class CommunityDetector:
    """Detect communities in entity graph using Label Propagation."""

    def __init__(
        self,
        driver: Neo4jDriver,
        group_id: str,
        *,
        model: str = "claude-sonnet-4-6",
        locale: str = "ja",
        min_members: int = 3,
    ) -> None:
        self._driver = driver
        self._group_id = group_id
        self._model = model
        self._locale = locale
        self._min_members = min_members

    # ── Public API ──────────

    async def detect_and_store(self) -> list[Community]:
        """Full pipeline: fetch graph -> detect -> summarize -> store.

        Steps:
            1. Fetch all entities + valid RELATES_TO edges
            2. Build NetworkX graph
            3. Run Label Propagation
            4. Filter communities < min_members
            5. LLM summarize each community
            6. Delete old communities in Neo4j
            7. Create new communities + HAS_MEMBER edges

        Returns:
            List of detected communities.
        """
        from core.memory.graph.queries import (
            FETCH_EDGES_FOR_COMMUNITY,
            FETCH_ENTITIES_FOR_COMMUNITY,
        )

        entities = await self._driver.execute_query(FETCH_ENTITIES_FOR_COMMUNITY, {"group_id": self._group_id})
        edges = await self._driver.execute_query(FETCH_EDGES_FOR_COMMUNITY, {"group_id": self._group_id})

        if len(entities) < self._min_members:
            logger.info("Too few entities (%d) for community detection", len(entities))
            return []

        G, entity_map = self._build_networkx_graph(entities, edges)
        raw_communities = self._run_label_propagation(G)
        if raw_communities is None:
            return []

        valid = [c for c in raw_communities if len(c) >= self._min_members]
        if not valid:
            logger.info("No communities with >= %d members", self._min_members)
            return []

        results = await self._summarize_all(valid, entity_map)
        await self._persist(results)

        logger.info(
            "Detected %d communities for group_id=%s",
            len(results),
            self._group_id,
        )
        return results

    async def dynamic_update(self, entity_uuid: str, neighbor_uuids: list[str]) -> str | None:
        """Assign new entity to existing community via majority vote.

        Args:
            entity_uuid: UUID of newly created/resolved entity.
            neighbor_uuids: UUIDs of neighboring entities.

        Returns:
            Community UUID if assigned, None otherwise.
        """
        if not neighbor_uuids:
            return None

        from core.memory.graph.queries import (
            CREATE_HAS_MEMBER,
            FIND_COMMUNITY_FOR_ENTITY,
        )

        community_votes: dict[str, int] = {}
        for n_uuid in neighbor_uuids:
            rows = await self._driver.execute_query(FIND_COMMUNITY_FOR_ENTITY, {"entity_uuid": n_uuid})
            for row in rows:
                c_uuid = row.get("community_uuid", "")
                if c_uuid:
                    community_votes[c_uuid] = community_votes.get(c_uuid, 0) + 1

        if not community_votes:
            return None

        best_community = max(community_votes, key=community_votes.get)  # type: ignore[arg-type]

        try:
            await self._driver.execute_write(
                CREATE_HAS_MEMBER,
                {
                    "community_uuid": best_community,
                    "entity_uuid": entity_uuid,
                },
            )
            return best_community
        except Exception:
            logger.debug("Dynamic community update failed", exc_info=True)
            return None

    # ── Graph construction ──────────

    @staticmethod
    def _build_networkx_graph(entities: list[dict], edges: list[dict]) -> tuple:
        """Build undirected weighted NetworkX graph from entities and edges."""
        import networkx as nx

        G = nx.Graph()
        entity_map: dict[str, dict] = {}

        for e in entities:
            uid = e.get("uuid", "")
            G.add_node(uid)
            entity_map[uid] = e

        for edge in edges:
            src = edge.get("source_uuid", "")
            tgt = edge.get("target_uuid", "")
            if src in entity_map and tgt in entity_map:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
                else:
                    G.add_edge(src, tgt, weight=1)

        return G, entity_map

    @staticmethod
    def _run_label_propagation(G) -> list[list[str]] | None:  # noqa: N803
        """Run asynchronous Label Propagation on the graph."""
        try:
            from networkx.algorithms.community import asyn_lpa_communities

            communities_gen = asyn_lpa_communities(G, weight="weight")
            return [list(c) for c in communities_gen]
        except Exception:
            logger.warning("Label Propagation failed", exc_info=True)
            return None

    # ── Summarization ──────────

    async def _summarize_all(self, communities: list[list[str]], entity_map: dict[str, dict]) -> list[Community]:
        """Summarize each community via LLM."""
        results: list[Community] = []
        for member_uuids in communities:
            member_names = [entity_map[uid].get("name", "") for uid in member_uuids if uid in entity_map]
            member_summaries = [entity_map[uid].get("summary", "") for uid in member_uuids if uid in entity_map]

            try:
                name, summary = await self._summarize_community(member_names, member_summaries)
            except Exception:
                logger.warning("Community summarization failed", exc_info=True)
                ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
                name = f"cluster_{ts}"
                summary = ", ".join(member_names[:10])

            results.append(
                Community(
                    uuid=str(uuid4()),
                    name=name,
                    summary=summary,
                    member_uuids=member_uuids,
                    member_names=member_names,
                )
            )
        return results

    async def _summarize_community(self, names: list[str], summaries: list[str]) -> tuple[str, str]:
        """LLM-generate name and summary for a community."""
        import litellm

        prompts = self._select_prompts()
        members_text = "\n".join(f"- {n}: {s}" for n, s in zip(names[:20], summaries[:20], strict=False))
        user_prompt = prompts.COMMUNITY_USER.format(members=members_text)

        response = await litellm.acompletion(
            model=self._model,
            messages=[
                {"role": "system", "content": prompts.COMMUNITY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=30,
        )

        text = response.choices[0].message.content or ""
        return self._parse_community_response(text)

    # ── Persistence ──────────

    async def _persist(self, communities: list[Community]) -> None:
        """Delete old communities and write new ones to Neo4j."""
        from core.memory.graph.queries import (
            CREATE_COMMUNITY,
            CREATE_HAS_MEMBER,
            DELETE_COMMUNITIES_BY_GROUP,
        )

        await self._driver.execute_write(DELETE_COMMUNITIES_BY_GROUP, {"group_id": self._group_id})

        now_str = datetime.now(tz=UTC).isoformat()
        for comm in communities:
            await self._driver.execute_write(
                CREATE_COMMUNITY,
                {
                    "uuid": comm.uuid,
                    "name": comm.name,
                    "summary": comm.summary,
                    "group_id": self._group_id,
                    "created_at": now_str,
                },
            )
            for member_uuid in comm.member_uuids:
                await self._driver.execute_write(
                    CREATE_HAS_MEMBER,
                    {
                        "community_uuid": comm.uuid,
                        "entity_uuid": member_uuid,
                    },
                )

    # ── Parsing helpers ──────────

    @staticmethod
    def _parse_community_response(text: str) -> tuple[str, str]:
        """Parse LLM community summary response.

        Tries JSON extraction (fenced, then raw), falls back to line split.
        """
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        body = fence.group(1) if fence else text

        for candidate in (body, text):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return (
                        data.get("name", "unnamed"),
                        data.get("summary", ""),
                    )
            except (json.JSONDecodeError, ValueError):
                continue

        lines = text.strip().split("\n")
        name = lines[0][:100] if lines else "unnamed"
        summary = " ".join(lines[1:])[:300] if len(lines) > 1 else name
        return (name, summary)

    def _select_prompts(self):  # noqa: ANN202
        """Return locale-appropriate prompt module."""
        if self._locale == "en":
            from core.memory.extraction.prompts import en as p
        else:
            from core.memory.extraction.prompts import ja as p
        return p
