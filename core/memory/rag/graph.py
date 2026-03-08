from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Knowledge graph for spreading activation in memory retrieval.

Implements:
- Graph construction from markdown links and vector similarity
- Multi-source support (knowledge + episodes + other memory types)
- Personalized PageRank for multi-hop activation (edge-weighted)
- Spreading activation to expand search results with real content
- Graph persistence (JSON) and incremental updates

Based on: docs/design/priming-layer-design.md Phase 3
"""

import json
import logging
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

logger = logging.getLogger("animaworks.rag.graph")

# ── Configuration ───────────────────────────────────────────────────

# Similarity threshold for implicit links (cosine similarity)
IMPLICIT_LINK_THRESHOLD = 0.75

# PageRank parameters
PAGERANK_ALPHA = 0.85  # Damping factor
PAGERANK_MAX_ITER = 100
PAGERANK_TOL = 1e-6

# Spreading activation parameters
MAX_HOPS = 2  # Maximum hops for spreading activation

# Graph cache filename
GRAPH_CACHE_FILE = "knowledge_graph.json"


# ── KnowledgeGraph ──────────────────────────────────────────────────


class KnowledgeGraph:
    """Knowledge graph for spreading activation.

    Builds a directed graph where:
    - Nodes = memory files (knowledge, episodes, etc.)
    - Edges = explicit links (Markdown [[filename]]) + implicit links (vector similarity)

    Each node stores ``memory_type`` and ``path`` attributes so that
    doc_id construction and vector-store lookups use the correct collection.

    Provides Personalized PageRank (edge-weighted) for multi-hop activation.
    Supports JSON persistence and incremental graph updates.
    """

    def __init__(
        self,
        vector_store,  # VectorStore instance
        indexer,  # MemoryIndexer instance
    ) -> None:
        """Initialize knowledge graph.

        Args:
            vector_store: VectorStore instance for similarity queries
            indexer: MemoryIndexer instance for embedding generation
        """
        self.vector_store = vector_store
        self.indexer = indexer
        self.graph: nx.DiGraph | None = None

    # ── Graph construction ──────────────────────────────────────────

    def build_graph(
        self,
        anima_name: str,
        knowledge_dir: Path,
        *,
        memory_dirs: dict[str, Path] | None = None,
        implicit_link_threshold: float = IMPLICIT_LINK_THRESHOLD,
    ) -> nx.DiGraph:
        """Build knowledge graph from memory files.

        Scans *knowledge_dir* (treated as ``memory_type="knowledge"``)
        plus any additional directories in *memory_dirs* (e.g.
        ``{"episodes": episodes_dir}``).

        Args:
            anima_name: Anima name (for collection selection)
            knowledge_dir: Path to knowledge directory
            memory_dirs: Additional ``{memory_type: directory}`` to include.

        Returns:
            NetworkX directed graph
        """
        logger.info("Building knowledge graph for anima=%s", anima_name)

        self._implicit_link_threshold = implicit_link_threshold
        graph = nx.DiGraph()

        sources: dict[str, Path] = {}
        if knowledge_dir.is_dir():
            sources["knowledge"] = knowledge_dir

        if memory_dirs:
            for mt, d in memory_dirs.items():
                if d.is_dir():
                    sources[mt] = d

        if not sources:
            logger.warning("No valid source directories for graph")
            self.graph = graph
            return graph

        # Add nodes from each source directory (recursive)
        for memory_type, src_dir in sources.items():
            md_files = sorted(src_dir.rglob("*.md"))
            for md_file in md_files:
                rel_key = str(md_file.relative_to(src_dir).with_suffix(""))
                node_id = self._make_node_id(rel_key, memory_type)
                graph.add_node(
                    node_id,
                    path=str(md_file),
                    memory_type=memory_type,
                    stem=md_file.stem,
                    rel_key=rel_key,
                )

        if graph.number_of_nodes() == 0:
            logger.warning("No markdown files found in source directories")
            self.graph = graph
            return graph

        logger.debug("Added %d nodes to graph", graph.number_of_nodes())

        # Add explicit links (Markdown [[filename]])
        for node_id, attrs in list(graph.nodes(data=True)):
            node_path = Path(attrs["path"])
            try:
                content = node_path.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    target_stem = target.replace(".md", "")
                    target_node = self._resolve_link_target(graph, target_stem)
                    if target_node and target_node != node_id:
                        graph.add_edge(
                            node_id,
                            target_node,
                            link_type="explicit",
                            similarity=1.0,
                        )
                        logger.debug("Explicit link: %s -> %s", node_id, target_node)

            except Exception as e:
                logger.warning("Failed to extract links from %s: %s", node_path, e)

        logger.debug("Added %d explicit edges", graph.number_of_edges())

        # Add implicit links (vector similarity)
        implicit_count = self._add_implicit_links(graph, anima_name)
        logger.debug("Added %d implicit edges", implicit_count)

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        self.graph = graph
        return graph

    @staticmethod
    def _make_node_id(rel_key: str, memory_type: str) -> str:
        """Create a unique node ID combining memory type and relative path key.

        ``rel_key`` is the relative path from the source directory with the
        ``.md`` suffix removed (e.g. ``"structure"`` or ``"organization/structure"``).

        Knowledge nodes use bare rel_key for backward compatibility;
        other types are prefixed (e.g. ``episodes:2026-03-01``).
        """
        if memory_type == "knowledge":
            return rel_key
        return f"{memory_type}:{rel_key}"

    @staticmethod
    def _resolve_link_target(graph: nx.DiGraph, target_stem: str) -> str | None:
        """Resolve ``[[target]]`` to a graph node id.

        Tries bare stem first (knowledge nodes), then checks
        prefixed variants for other memory types.
        """
        if target_stem in graph:
            return target_stem
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("stem") == target_stem:
                return node_id
        return None

    def _extract_markdown_links(self, content: str) -> list[str]:
        """Extract [[filename]] style links from markdown content.

        Args:
            content: Markdown file content

        Returns:
            List of linked filenames
        """
        # Match [[filename]] or [[filename|display text]]
        pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
        matches = re.findall(pattern, content)
        return [m.strip() for m in matches]

    def _add_implicit_links(self, graph: nx.DiGraph, anima_name: str) -> int:
        """Add implicit links based on vector similarity.

        Queries the correct collection per-node based on the node's
        ``memory_type`` attribute (e.g. ``{anima}_knowledge``,
        ``{anima}_episodes``).

        Args:
            graph: NetworkX graph to modify
            anima_name: Anima name for collection selection

        Returns:
            Number of implicit edges added
        """
        added_count = 0

        for node_id, attrs in list(graph.nodes(data=True)):
            node_path = Path(attrs["path"])
            memory_type = attrs.get("memory_type", "knowledge")
            collection_name = f"{anima_name}_{memory_type}"

            try:
                content = node_path.read_text(encoding="utf-8")
                embedding = self.indexer._generate_embeddings([content])[0]

                results = self.vector_store.query(
                    collection=collection_name,
                    embedding=embedding,
                    top_k=5,
                )

                threshold = getattr(self, "_implicit_link_threshold", IMPLICIT_LINK_THRESHOLD)
                for result in results:
                    target_node = self._match_result_to_node(
                        graph,
                        result.document.id,
                        result.score,
                    )
                    if (
                        target_node is not None
                        and target_node != node_id
                        and result.score >= threshold
                        and not graph.has_edge(node_id, target_node)
                    ):
                        graph.add_edge(
                            node_id,
                            target_node,
                            link_type="implicit",
                            similarity=result.score,
                        )
                        added_count += 1
                        logger.debug(
                            "Implicit link: %s -> %s (score=%.3f)",
                            node_id,
                            target_node,
                            result.score,
                        )

            except Exception as e:
                logger.warning("Failed to add implicit links for %s: %s", node_id, e)

        return added_count

    @staticmethod
    def _match_result_to_node(
        graph: nx.DiGraph,
        doc_id: str,
        score: float,
    ) -> str | None:
        """Map a vector-store doc_id back to a graph node.

        doc_id format: ``{prefix}/{memory_type}/{rel_path}.md#{chunk_index}``
        where ``rel_path`` may include subdirectories.
        """
        parts = doc_id.split("/")
        if len(parts) < 3:
            return None
        memory_type = parts[1]
        tail = "/".join(parts[2:])
        rel_path_md = tail.split("#")[0]
        rel_key = str(Path(rel_path_md).with_suffix(""))

        candidate = KnowledgeGraph._make_node_id(rel_key, memory_type)
        if candidate in graph:
            return candidate

        # Fallback: match by stem for flat files
        stem = Path(rel_path_md).stem
        for nid, attrs in graph.nodes(data=True):
            if attrs.get("stem") == stem and attrs.get("memory_type") == memory_type:
                return nid
        return None

    # ── Graph persistence ───────────────────────────────────────────

    def save_graph(self, cache_dir: Path) -> None:
        """Persist graph to JSON file.

        Args:
            cache_dir: Directory to save the cache file
        """
        if self.graph is None:
            logger.warning("No graph to save")
            return

        cache_path = cache_dir / GRAPH_CACHE_FILE
        data = json_graph.node_link_data(self.graph)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Graph saved: %d nodes, %d edges -> %s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            cache_path,
        )

    def load_graph(self, cache_dir: Path) -> bool:
        """Load graph from JSON cache.

        Args:
            cache_dir: Directory containing the cache file

        Returns:
            True if cache loaded successfully, False otherwise
        """
        cache_path = cache_dir / GRAPH_CACHE_FILE
        if not cache_path.exists():
            logger.debug("No graph cache found at %s", cache_path)
            return False

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)

            graph = json_graph.node_link_graph(data, directed=True)
            self.graph = graph

            logger.info(
                "Graph loaded from cache: %d nodes, %d edges",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return True

        except Exception as e:
            logger.warning("Failed to load graph cache: %s", e)
            return False

    # ── Incremental update ──────────────────────────────────────────

    @staticmethod
    def _infer_memory_type(file_path: Path, anima_dir: Path) -> str:
        """Infer memory_type from file path relative to anima directory.

        Uses the first directory component of the relative path
        (e.g. ``knowledge``, ``episodes``, ``procedures``).

        Args:
            file_path: Absolute path to the memory file.
            anima_dir: Anima data root directory.

        Returns:
            Memory type string, defaults to ``"knowledge"`` on failure.
        """
        try:
            rel = file_path.relative_to(anima_dir)
            return rel.parts[0] if rel.parts else "knowledge"
        except ValueError:
            return "knowledge"

    def update_graph_incremental(
        self,
        changed_files: list[Path],
        anima_name: str,
        memory_type: str | None = None,
        anima_dir: Path | None = None,
    ) -> None:
        """Update graph incrementally for changed files.

        Steps:
        1. Remove nodes and edges for changed files
        2. Re-add nodes for changed files (if file still exists)
        3. Re-scan explicit links for changed files
        4. Re-compute implicit links for changed files

        When *memory_type* is ``None``, each file's type is inferred from
        its path relative to *anima_dir* (via ``_infer_memory_type``).
        If *anima_dir* is also ``None``, the graph's existing node
        attribute is used, falling back to ``"knowledge"``.

        Args:
            changed_files: List of changed file paths
            anima_name: Anima name for collection selection
            memory_type: Memory type of the changed files (None = auto-detect)
            anima_dir: Anima data root for type inference
        """
        if self.graph is None:
            logger.warning("No graph to update incrementally")
            return

        def _resolve_type(f: Path) -> str:
            if memory_type is not None:
                return memory_type
            # Try exact path match in existing graph nodes first
            f_str = str(f)
            for _nid, attrs in self.graph.nodes(data=True):
                if attrs.get("path") == f_str:
                    mt = attrs.get("memory_type")
                    if mt:
                        return mt
            # Fallback: stem match (less precise but handles new files
            # that share a stem with an existing node of known type)
            for _nid, attrs in self.graph.nodes(data=True):
                if attrs.get("stem") == f.stem:
                    mt = attrs.get("memory_type")
                    if mt:
                        return mt
            if anima_dir is not None:
                return self._infer_memory_type(f, anima_dir)
            return "knowledge"

        changed_node_ids: set[str] = set()
        file_types: dict[Path, str] = {}
        file_rel_keys: dict[Path, str] = {}
        for f in changed_files:
            mt = _resolve_type(f)
            file_types[f] = mt
            # Compute rel_key: try existing node attrs first, then anima_dir inference
            rel_key = f.stem
            f_str = str(f)
            for _nid, attrs in self.graph.nodes(data=True):
                if attrs.get("path") == f_str:
                    rel_key = attrs.get("rel_key", f.stem)
                    break
            else:
                if anima_dir is not None:
                    mt_dir = anima_dir / mt
                    try:
                        rel_key = str(f.relative_to(mt_dir).with_suffix(""))
                    except ValueError:
                        rel_key = f.stem
            file_rel_keys[f] = rel_key
            nid = self._make_node_id(rel_key, mt)
            changed_node_ids.add(nid)

        logger.info(
            "Incremental graph update for %d files: %s",
            len(changed_files),
            changed_node_ids,
        )

        # 1. Remove nodes (and their edges) for changed files
        for node_id in changed_node_ids:
            if node_id in self.graph:
                self.graph.remove_node(node_id)
                logger.debug("Removed node: %s", node_id)

        # 2. Re-add nodes for files that still exist
        for file_path in changed_files:
            if file_path.exists():
                mt = file_types[file_path]
                rel_key = file_rel_keys[file_path]
                nid = self._make_node_id(rel_key, mt)
                self.graph.add_node(
                    nid,
                    path=str(file_path),
                    memory_type=mt,
                    stem=file_path.stem,
                    rel_key=rel_key,
                )
                logger.debug("Re-added node: %s", nid)

        # 3. Re-scan explicit links for changed files
        for file_path in changed_files:
            if not file_path.exists():
                continue

            mt = file_types[file_path]
            source_id = self._make_node_id(file_rel_keys[file_path], mt)
            try:
                content = file_path.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    target_stem = target.replace(".md", "")
                    target_node = self._resolve_link_target(self.graph, target_stem)
                    if target_node and target_node != source_id:
                        self.graph.add_edge(
                            source_id,
                            target_node,
                            link_type="explicit",
                            similarity=1.0,
                        )

            except Exception as e:
                logger.warning("Failed to extract links from %s: %s", file_path, e)

        # Re-scan explicit links from OTHER nodes that might point to changed files
        for node_id in list(self.graph.nodes()):
            if node_id in changed_node_ids:
                continue

            node_path = Path(self.graph.nodes[node_id].get("path", ""))
            if not node_path.exists():
                continue

            try:
                content = node_path.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    target_stem = target.replace(".md", "")
                    target_node = self._resolve_link_target(self.graph, target_stem)
                    if (
                        target_node
                        and target_node in changed_node_ids
                        and not self.graph.has_edge(node_id, target_node)
                    ):
                        self.graph.add_edge(
                            node_id,
                            target_node,
                            link_type="explicit",
                            similarity=1.0,
                        )

            except Exception as e:
                logger.warning("Failed to re-scan links from %s: %s", node_path, e)

        # 4. Re-compute implicit links for changed files
        for file_path in changed_files:
            if not file_path.exists():
                continue

            mt = file_types[file_path]
            collection_name = f"{anima_name}_{mt}"
            node_id = self._make_node_id(file_rel_keys[file_path], mt)
            try:
                content = file_path.read_text(encoding="utf-8")
                embedding = self.indexer._generate_embeddings([content])[0]

                results = self.vector_store.query(
                    collection=collection_name,
                    embedding=embedding,
                    top_k=5,
                )

                threshold = getattr(self, "_implicit_link_threshold", IMPLICIT_LINK_THRESHOLD)
                for result in results:
                    target_node = self._match_result_to_node(
                        self.graph,
                        result.document.id,
                        result.score,
                    )
                    if (
                        target_node is not None
                        and target_node != node_id
                        and result.score >= threshold
                        and not self.graph.has_edge(node_id, target_node)
                    ):
                        self.graph.add_edge(
                            node_id,
                            target_node,
                            link_type="implicit",
                            similarity=result.score,
                        )

            except Exception as e:
                logger.warning("Failed to add implicit links for %s: %s", node_id, e)

        logger.info(
            "Incremental update complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ── Personalized PageRank ───────────────────────────────────────

    def personalized_pagerank(
        self,
        query_nodes: list[str],
        alpha: float = PAGERANK_ALPHA,
    ) -> dict[str, float]:
        """Compute Personalized PageRank scores with edge weights.

        Uses edge attribute "similarity" as weight. Explicit links have
        weight 1.0 (maximum), implicit links use their similarity score.

        Args:
            query_nodes: Starting nodes (files related to query)
            alpha: Damping factor (default: 0.85)

        Returns:
            Dict mapping node_id -> PageRank score
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("Graph not built or empty")
            return {}

        if not query_nodes:
            logger.warning("No query nodes provided")
            return {}

        # Filter query nodes to only those in graph
        valid_query_nodes = [n for n in query_nodes if n in self.graph]
        if not valid_query_nodes:
            logger.warning("No valid query nodes in graph")
            return {}

        # Create personalization vector (uniform over query nodes)
        personalization = {node: 0.0 for node in self.graph.nodes()}
        weight = 1.0 / len(valid_query_nodes)
        for node in valid_query_nodes:
            personalization[node] = weight

        logger.debug("Computing Personalized PageRank from %d query nodes", len(valid_query_nodes))

        try:
            # Compute Personalized PageRank with edge weights
            scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=PAGERANK_MAX_ITER,
                tol=PAGERANK_TOL,
                weight="similarity",  # Use edge "similarity" as weight
            )

            logger.debug("PageRank computed for %d nodes", len(scores))
            return scores

        except ImportError as e:
            logger.error(
                "PageRank computation failed: missing dependency (numpy required). "
                "Install with: pip install 'animaworks[rag]'"
            )
            logger.debug("Original error: %s", e)
            return {}
        except Exception as e:
            logger.error("PageRank computation failed: %s", e)
            return {}

    # ── Spreading activation ────────────────────────────────────────

    def expand_search_results(
        self,
        initial_results: list,  # List of RetrievalResult
        max_hops: int = MAX_HOPS,
    ) -> list:
        """Expand search results using spreading activation.

        Activated nodes fetch real file content instead of placeholders.
        Works across memory types (knowledge, episodes, etc.).

        Args:
            initial_results: Initial retrieval results
            max_hops: Maximum hops for expansion (default: 2)

        Returns:
            Expanded list of results (original + activated neighbors)
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.debug("Graph not available, returning original results")
            return initial_results

        if not initial_results:
            return initial_results

        logger.debug(
            "Expanding %d initial results with spreading activation (max_hops=%d)",
            len(initial_results),
            max_hops,
        )

        # Extract query nodes from initial results
        query_nodes: list[str] = []
        initial_node_ids: set[str] = set()
        for result in initial_results:
            node = self._match_result_to_node(self.graph, result.doc_id, 0.0)
            if node is not None:
                query_nodes.append(node)
                initial_node_ids.add(node)

        if not query_nodes:
            logger.debug("No query nodes found in graph")
            return initial_results

        # Compute Personalized PageRank
        pagerank_scores = self.personalized_pagerank(query_nodes)

        if not pagerank_scores:
            return initial_results

        # Find activated nodes (top K by PageRank score, excluding initial results)
        activated_nodes = [
            (nid, score) for nid, score in pagerank_scores.items() if nid not in initial_node_ids and score > 0.001
        ]

        activated_nodes.sort(key=lambda x: x[1], reverse=True)
        top_activated = activated_nodes[:5]

        logger.debug("Found %d activated nodes, adding top %d", len(activated_nodes), len(top_activated))

        from core.memory.rag.retriever import RetrievalResult

        expanded_results = list(initial_results)

        for node_id, score in top_activated:
            attrs = self.graph.nodes[node_id]
            node_path = Path(attrs.get("path", ""))
            memory_type = attrs.get("memory_type", "knowledge")
            stem = attrs.get("stem", node_id)

            content = self._fetch_node_content(node_id, node_path, memory_type)

            rel_key = attrs.get("rel_key", stem)
            rel_md = f"{rel_key}.md"
            expanded_results.append(
                RetrievalResult(
                    doc_id=f"{self.indexer.anima_name}/{memory_type}/{rel_md}#0",
                    content=content,
                    score=score * 0.5,
                    metadata={
                        "source_file": f"{memory_type}/{rel_md}",
                        "memory_type": memory_type,
                        "activation": "spreading",
                        "pagerank_score": score,
                    },
                    source_scores={
                        "pagerank": score,
                    },
                )
            )

        logger.info(
            "Expanded from %d to %d results",
            len(initial_results),
            len(expanded_results),
        )

        return expanded_results

    def _fetch_node_content(
        self,
        node_id: str,
        node_path: Path,
        memory_type: str = "knowledge",
    ) -> str:
        """Fetch real content for an activated node.

        Tries file read first, falls back to vector store chunk retrieval
        using ``source_file`` metadata filter, then stem-based vector search.

        Args:
            node_id: Node identifier
            node_path: Path to the file
            memory_type: Memory type for collection lookup

        Returns:
            File content string
        """
        if node_path.exists():
            try:
                return node_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read file %s: %s", node_path, e)

        try:
            collection_name = f"{self.indexer.anima_name}_{memory_type}"
            stem = node_path.stem if node_path.stem else (node_id.split(":", 1)[-1] if ":" in node_id else node_id)

            # 1st attempt: filter by source_file metadata (supports subdirectories)
            try:
                source_file_value = str(node_path.relative_to(self.indexer.anima_dir))
            except ValueError:
                source_file_value = f"{memory_type}/{stem}.md"
            query_text = stem.replace("-", " ").replace("_", " ")
            embedding = self.indexer._generate_embeddings([f"{memory_type} {query_text}"])[0]
            results = self.vector_store.query(
                collection=collection_name,
                embedding=embedding,
                top_k=3,
                filter_metadata={"source_file": source_file_value},
            )
            if results:
                return results[0].document.content

            # 2nd attempt: stem-based vector search without filter
            embedding_fallback = self.indexer._generate_embeddings([query_text])[0]
            results = self.vector_store.query(
                collection=collection_name,
                embedding=embedding_fallback,
                top_k=1,
            )
            if results:
                return results[0].document.content
        except Exception as e:
            logger.warning("Failed to fetch content from vector store for %s: %s", node_id, e)

        return f"[Content unavailable: {node_id}]"


# ── Public API ──────────────────────────────────────────────────────


def create_knowledge_graph(
    anima_name: str,
    knowledge_dir: Path,
    vector_store,
    indexer,
    *,
    memory_dirs: dict[str, Path] | None = None,
) -> KnowledgeGraph:
    """Create and build a knowledge graph for an anima.

    Args:
        anima_name: Anima name
        knowledge_dir: Path to knowledge directory
        vector_store: VectorStore instance
        indexer: MemoryIndexer instance
        memory_dirs: Additional ``{memory_type: directory}`` to include
            (e.g. ``{"episodes": episodes_dir}``).

    Returns:
        Built KnowledgeGraph instance
    """
    graph = KnowledgeGraph(vector_store, indexer)
    graph.build_graph(anima_name, knowledge_dir, memory_dirs=memory_dirs)
    return graph
