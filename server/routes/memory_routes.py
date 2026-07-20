from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import re
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path
from typing import Any

import networkx as nx
from fastapi import APIRouter, HTTPException, Query, Request
from networkx.readwrite import json_graph

from core.memory.conversation import ConversationMemory
from core.memory.frontmatter import parse_frontmatter
from core.memory.manager import MemoryManager
from core.time_utils import get_app_timezone

logger = logging.getLogger("animaworks.routes.memory")

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_GRAPH_CACHE_FILE = "knowledge_graph.json"
_SIMILARITY_EDGE_LIMIT = 500
_GRAPH_VISIBLE_MEMORY_TYPES = frozenset({"knowledge", "procedures"})


def _make_memory_node_id(rel_key: str, memory_type: str) -> str:
    """Return the node ID format used by ``KnowledgeGraph``."""
    if memory_type == "knowledge":
        return rel_key
    return f"{memory_type}:{rel_key}"


def _resolve_link_target(graph: nx.DiGraph, target: str) -> str | None:
    """Resolve a wikilink using the same rules as ``KnowledgeGraph``."""
    target_stem = target.replace(".md", "")
    if target_stem in graph:
        return target_stem
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("stem") == target_stem:
            return str(node_id)
    return None


def _build_explicit_graph(memory: MemoryManager) -> nx.DiGraph:
    """Build a lightweight knowledge/procedure graph without embeddings."""
    graph = nx.DiGraph()
    sources = {
        "knowledge": memory.knowledge_dir,
        "procedures": memory.procedures_dir,
    }

    for memory_type, source_dir in sources.items():
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.rglob("*.md")):
            rel_key = str(path.relative_to(source_dir).with_suffix(""))
            graph.add_node(
                _make_memory_node_id(rel_key, memory_type),
                node_type="memory_file",
                path=str(path),
                memory_type=memory_type,
                stem=path.stem,
                rel_key=rel_key,
            )

    for node_id, attrs in list(graph.nodes(data=True)):
        path = Path(attrs["path"])
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to scan wikilinks in %s: %s", path, exc)
            continue
        for match in _WIKILINK_PATTERN.findall(content):
            target = _resolve_link_target(graph, match.strip())
            if target is not None and target != node_id:
                graph.add_edge(
                    node_id,
                    target,
                    link_type="explicit",
                    similarity=1.0,
                )
    return graph


def _load_cached_graph(cache_path: Path) -> nx.DiGraph:
    """Load the existing NetworkX node-link graph cache."""
    with cache_path.open(encoding="utf-8") as cache_file:
        data = json.load(cache_file)
    return json_graph.node_link_graph(data, directed=True)


def _node_file_path(anima_dir: Path, node_id: str, attrs: dict[str, Any]) -> Path | None:
    """Find a cached node's file while keeping reads inside the anima."""
    memory_type = str(attrs.get("memory_type", "knowledge"))
    rel_key = attrs.get("rel_key")
    if not isinstance(rel_key, str) or not rel_key:
        prefix = f"{memory_type}:"
        rel_key = node_id[len(prefix) :] if node_id.startswith(prefix) else node_id

    try:
        candidate = (anima_dir / memory_type / f"{rel_key}.md").resolve()
        candidate.relative_to(anima_dir.resolve())
    except (OSError, ValueError):
        candidate = None
    if candidate is not None and candidate.is_file():
        return candidate

    raw_path = attrs.get("path")
    if not isinstance(raw_path, str):
        return None
    try:
        resolved_path = Path(raw_path).resolve()
        resolved_path.relative_to(anima_dir.resolve())
    except (OSError, ValueError):
        return None
    return resolved_path if resolved_path.is_file() else None


def _frontmatter_fields(path: Path) -> dict[str, Any]:
    """Read graph-facing metadata, using mtime when created_at is absent."""
    metadata: dict[str, Any] = {}
    try:
        metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read graph metadata from %s: %s", path, exc)

    created_at = metadata.get("created_at")
    if created_at is None:
        try:
            created_at = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=get_app_timezone(),
            ).isoformat()
        except OSError:
            created_at = None
    return {
        "created_at": created_at,
        "updated_at": metadata.get("updated_at"),
        "confidence": metadata.get("confidence"),
        "source_episodes": metadata.get("source_episodes", []),
    }


def _graph_response(anima_dir: Path, graph: nx.DiGraph, *, partial: bool) -> dict[str, Any]:
    """Convert a cached or fallback graph into the public API shape."""
    nodes: list[dict[str, Any]] = []
    included_node_ids: set[str] = set()
    for raw_node_id, attrs in graph.nodes(data=True):
        node_id = str(raw_node_id)
        if attrs.get("node_type", "memory_file") != "memory_file":
            continue
        path = _node_file_path(anima_dir, node_id, attrs)
        if path is None:
            continue
        memory_type = str(attrs.get("memory_type", "knowledge"))
        if memory_type not in _GRAPH_VISIBLE_MEMORY_TYPES:
            continue
        nodes.append(
            {
                "id": node_id,
                "memory_type": memory_type,
                "stem": str(attrs.get("stem") or path.stem),
                **_frontmatter_fields(path),
            }
        )
        included_node_ids.add(node_id)

    explicit_edges: list[dict[str, Any]] = []
    similarity_edges: list[dict[str, Any]] = []
    for raw_source, raw_target, attrs in graph.edges(data=True):
        source = str(raw_source)
        target = str(raw_target)
        if source not in included_node_ids or target not in included_node_ids:
            continue
        link_type = str(attrs.get("link_type", "implicit"))
        default_similarity = 1.0 if link_type == "explicit" else 0.0
        try:
            similarity = float(attrs.get("similarity", default_similarity))
        except (TypeError, ValueError):
            similarity = default_similarity
        edge = {
            "source": source,
            "target": target,
            "link_type": link_type,
            "similarity": similarity,
        }
        if link_type == "explicit":
            explicit_edges.append(edge)
        else:
            similarity_edges.append(edge)

    similarity_edges.sort(key=lambda edge: edge["similarity"], reverse=True)
    edges_capped = len(similarity_edges) > _SIMILARITY_EDGE_LIMIT
    return {
        "nodes": nodes,
        "edges": explicit_edges + similarity_edges[:_SIMILARITY_EDGE_LIMIT],
        "partial": partial,
        "edges_capped": edges_capped,
    }


def create_memory_router() -> APIRouter:
    router = APIRouter()

    # ── Episodes ──────────────────────────────────────────

    @router.get("/animas/{name}/episodes")
    async def list_episodes(name: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        return {"files": memory.list_episode_files()}

    @router.get("/animas/{name}/episodes/calendar")
    async def episode_calendar(
        name: str,
        request: Request,
        year: int = Query(..., ge=1, le=9999),
        month: int = Query(..., ge=1, le=12),
    ):
        """Return lightweight episode availability for every day in a month."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        episode_dates = set(memory.list_episode_files())
        days = []
        for day_number in range(1, monthrange(year, month)[1] + 1):
            episode_date = date(year, month, day_number).isoformat()
            has_episode = episode_date in episode_dates
            size_bytes = 0
            if has_episode:
                try:
                    size_bytes = (memory.episodes_dir / f"{episode_date}.md").stat().st_size
                except OSError:
                    has_episode = False
            days.append(
                {
                    "date": episode_date,
                    "has_episode": has_episode,
                    "size_bytes": size_bytes,
                }
            )
        return {"year": year, "month": month, "days": days}

    @router.get("/animas/{name}/episodes/{date}")
    async def get_episode(name: str, date: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        path = memory.episodes_dir / f"{date}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Episode not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}") from None
        return {"date": date, "content": content}

    # ── Knowledge ─────────────────────────────────────────

    @router.get("/animas/{name}/knowledge")
    async def list_knowledge(name: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        return {"files": memory.list_knowledge_files()}

    @router.get("/animas/{name}/knowledge/{topic}")
    async def get_knowledge(name: str, topic: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        path = memory.knowledge_dir / f"{topic}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Knowledge not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}") from None
        return {"topic": topic, "content": content}

    # ── Procedures ────────────────────────────────────────

    @router.get("/animas/{name}/procedures")
    async def list_procedures(name: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        return {"files": memory.list_procedure_files()}

    @router.get("/animas/{name}/procedures/{proc}")
    async def get_procedure(name: str, proc: str, request: Request):
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        path = memory.procedures_dir / f"{proc}.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Procedure not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {e}") from None
        return {"name": proc, "content": content}

    # ── Conversation ──────────────────────────────────────

    @router.get("/animas/{name}/conversation")
    async def get_conversation(name: str, request: Request):
        """View current conversation state."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        # Read model config from anima directory
        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        conv = ConversationMemory(anima_dir, model_config)
        state = conv.load()
        return {
            "anima": name,
            "total_turn_count": state.total_turn_count,
            "raw_turns": len(state.turns),
            "compressed_turn_count": state.compressed_turn_count,
            "has_summary": bool(state.compressed_summary),
            "summary_preview": (state.compressed_summary[:300] if state.compressed_summary else ""),
            "total_token_estimate": state.total_token_estimate,
            "turns": [
                {
                    "role": t.role,
                    "content": (t.content[:200] + "..." if len(t.content) > 200 else t.content),
                    "timestamp": t.timestamp,
                    "token_estimate": t.token_estimate,
                }
                for t in state.turns
            ],
        }

    # NOTE: /conversation/full has been removed.
    # Use GET /animas/{name}/conversation/history (sessions.py) instead,
    # which reads from the permanent activity_log.

    @router.delete("/animas/{name}/conversation")
    async def clear_conversation(name: str, request: Request):
        """Clear conversation history for a fresh start."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        conv = ConversationMemory(anima_dir, model_config)
        conv.clear()
        return {"status": "cleared", "anima": name}

    @router.post("/animas/{name}/conversation/compress")
    async def compress_conversation(name: str, request: Request):
        """Manually trigger conversation compression."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        from core.config.models import load_model_config

        model_config = load_model_config(anima_dir)

        conv = ConversationMemory(anima_dir, model_config)
        compressed = await conv.compress_if_needed()
        state = conv.load()
        return {
            "compressed": compressed,
            "anima": name,
            "total_turn_count": state.total_turn_count,
            "total_token_estimate": state.total_token_estimate,
        }

    # ── Stats ─────────────────────────────────────────────

    @router.get("/animas/{name}/memory/graph")
    async def memory_graph(name: str, request: Request):
        """Return the cached memory graph or an explicit-link-only fallback."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)
        cache_path = anima_dir / "vectordb" / _GRAPH_CACHE_FILE
        partial = not cache_path.is_file()
        if partial:
            graph = _build_explicit_graph(memory)
        else:
            try:
                graph = _load_cached_graph(cache_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, nx.NetworkXError, KeyError, TypeError) as exc:
                logger.warning("Failed to load graph cache %s: %s", cache_path, exc)
                graph = _build_explicit_graph(memory)
                partial = True
        return _graph_response(anima_dir, graph, partial=partial)

    @router.get("/animas/{name}/memory/stats")
    async def memory_stats(name: str, request: Request):
        """Return memory storage statistics for an anima."""
        animas_dir = request.app.state.animas_dir
        anima_dir = animas_dir / name
        if not anima_dir.exists():
            raise HTTPException(status_code=404, detail=f"Anima not found: {name}")

        memory = MemoryManager(anima_dir)

        def dir_stats(directory):
            if not directory.exists():
                return {"count": 0, "total_bytes": 0}
            files = list(directory.glob("*.md"))
            return {
                "count": len(files),
                "total_bytes": sum(f.stat().st_size for f in files),
            }

        return {
            "anima": name,
            "episodes": dir_stats(memory.episodes_dir),
            "knowledge": dir_stats(memory.knowledge_dir),
            "procedures": dir_stats(memory.procedures_dir),
        }

    return router
