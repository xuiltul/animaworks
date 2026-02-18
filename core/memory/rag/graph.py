from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Knowledge graph for spreading activation in memory retrieval.

Implements:
- Graph construction from markdown links and vector similarity
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
    - Nodes = knowledge files
    - Edges = explicit links (Markdown [[filename]]) + implicit links (vector similarity)

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

    def build_graph(self, anima_name: str, knowledge_dir: Path) -> nx.DiGraph:
        """Build knowledge graph from knowledge files.

        Args:
            anima_name: Anima name (for collection selection)
            knowledge_dir: Path to knowledge directory

        Returns:
            NetworkX directed graph
        """
        logger.info("Building knowledge graph for anima=%s", anima_name)

        graph = nx.DiGraph()

        if not knowledge_dir.is_dir():
            logger.warning("Knowledge directory not found: %s", knowledge_dir)
            self.graph = graph
            return graph

        # Collect all markdown files
        md_files = sorted(knowledge_dir.glob("*.md"))
        if not md_files:
            logger.warning("No markdown files found in %s", knowledge_dir)
            self.graph = graph
            return graph

        # Add nodes
        for md_file in md_files:
            node_id = md_file.stem  # Use filename without extension
            graph.add_node(node_id, path=str(md_file))

        logger.debug("Added %d nodes to graph", graph.number_of_nodes())

        # Add explicit links (Markdown [[filename]])
        for md_file in md_files:
            source_id = md_file.stem
            try:
                content = md_file.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    # Normalize target (remove .md extension if present)
                    target_id = target.replace(".md", "")

                    # Only add edge if target node exists
                    if target_id in graph:
                        graph.add_edge(
                            source_id, target_id,
                            link_type="explicit",
                            similarity=1.0,  # Explicit links get max weight
                        )
                        logger.debug("Explicit link: %s -> %s", source_id, target_id)

            except Exception as e:
                logger.warning("Failed to extract links from %s: %s", md_file, e)

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

        Args:
            graph: NetworkX graph to modify
            anima_name: Anima name for collection selection

        Returns:
            Number of implicit edges added
        """
        collection_name = f"{anima_name}_knowledge"
        added_count = 0

        # For each node, find similar nodes via vector similarity
        for node_id in list(graph.nodes()):
            node_path = Path(graph.nodes[node_id]["path"])

            try:
                # Read file content
                content = node_path.read_text(encoding="utf-8")

                # Generate embedding for the entire file
                embedding = self.indexer._generate_embeddings([content])[0]

                # Query vector store for similar documents
                results = self.vector_store.query(
                    collection=collection_name,
                    embedding=embedding,
                    top_k=5,  # Top 5 most similar
                )

                # Add edges to similar documents
                for result in results:
                    # Extract target node ID from doc_id
                    # Format: "{anima}/{memory_type}/{filename}#{chunk_index}"
                    doc_id_parts = result.id.split("/")
                    if len(doc_id_parts) >= 3:
                        filename_with_chunk = doc_id_parts[2]
                        target_filename = filename_with_chunk.split("#")[0]
                        target_id = Path(target_filename).stem

                        # Only add if:
                        # 1. Target node exists
                        # 2. Not self-loop
                        # 3. Similarity above threshold
                        # 4. Edge doesn't already exist
                        if (
                            target_id in graph
                            and target_id != node_id
                            and result.score >= IMPLICIT_LINK_THRESHOLD
                            and not graph.has_edge(node_id, target_id)
                        ):
                            graph.add_edge(
                                node_id,
                                target_id,
                                link_type="implicit",
                                similarity=result.score,
                            )
                            added_count += 1
                            logger.debug(
                                "Implicit link: %s -> %s (score=%.3f)",
                                node_id,
                                target_id,
                                result.score,
                            )

            except Exception as e:
                logger.warning("Failed to add implicit links for %s: %s", node_id, e)

        return added_count

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

            self.graph = json_graph.node_link_graph(data, directed=True)

            logger.info(
                "Graph loaded from cache: %d nodes, %d edges",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )
            return True

        except Exception as e:
            logger.warning("Failed to load graph cache: %s", e)
            return False

    # ── Incremental update ──────────────────────────────────────────

    def update_graph_incremental(
        self,
        changed_files: list[Path],
        anima_name: str,
    ) -> None:
        """Update graph incrementally for changed files.

        Steps:
        1. Remove nodes and edges for changed files
        2. Re-add nodes for changed files (if file still exists)
        3. Re-scan explicit links for changed files
        4. Re-compute implicit links for changed files

        Args:
            changed_files: List of changed file paths
            anima_name: Anima name for collection selection
        """
        if self.graph is None:
            logger.warning("No graph to update incrementally")
            return

        changed_node_ids = {f.stem for f in changed_files}
        logger.info("Incremental graph update for %d files: %s", len(changed_files), changed_node_ids)

        # 1. Remove nodes (and their edges) for changed files
        for node_id in changed_node_ids:
            if node_id in self.graph:
                self.graph.remove_node(node_id)
                logger.debug("Removed node: %s", node_id)

        # Also remove edges from other nodes pointing to changed files
        # (already handled by remove_node above)

        # 2. Re-add nodes for files that still exist
        for file_path in changed_files:
            if file_path.exists():
                node_id = file_path.stem
                self.graph.add_node(node_id, path=str(file_path))
                logger.debug("Re-added node: %s", node_id)

        # 3. Re-scan explicit links for changed files
        for file_path in changed_files:
            if not file_path.exists():
                continue

            source_id = file_path.stem
            try:
                content = file_path.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    target_id = target.replace(".md", "")
                    if target_id in self.graph:
                        self.graph.add_edge(
                            source_id, target_id,
                            link_type="explicit",
                            similarity=1.0,
                        )

            except Exception as e:
                logger.warning("Failed to extract links from %s: %s", file_path, e)

        # Re-scan explicit links from OTHER nodes that might point to changed files
        for node_id in list(self.graph.nodes()):
            if node_id in changed_node_ids:
                continue  # Already handled above

            node_path = Path(self.graph.nodes[node_id].get("path", ""))
            if not node_path.exists():
                continue

            try:
                content = node_path.read_text(encoding="utf-8")
                explicit_links = self._extract_markdown_links(content)

                for target in explicit_links:
                    target_id = target.replace(".md", "")
                    if target_id in changed_node_ids and target_id in self.graph:
                        if not self.graph.has_edge(node_id, target_id):
                            self.graph.add_edge(
                                node_id, target_id,
                                link_type="explicit",
                                similarity=1.0,
                            )

            except Exception as e:
                logger.warning("Failed to re-scan links from %s: %s", node_path, e)

        # 4. Re-compute implicit links for changed files
        collection_name = f"{anima_name}_knowledge"
        for file_path in changed_files:
            if not file_path.exists():
                continue

            node_id = file_path.stem
            try:
                content = file_path.read_text(encoding="utf-8")
                embedding = self.indexer._generate_embeddings([content])[0]

                results = self.vector_store.query(
                    collection=collection_name,
                    embedding=embedding,
                    top_k=5,
                )

                for result in results:
                    doc_id_parts = result.id.split("/")
                    if len(doc_id_parts) >= 3:
                        filename_with_chunk = doc_id_parts[2]
                        target_filename = filename_with_chunk.split("#")[0]
                        target_id = Path(target_filename).stem

                        if (
                            target_id in self.graph
                            and target_id != node_id
                            and result.score >= IMPLICIT_LINK_THRESHOLD
                            and not self.graph.has_edge(node_id, target_id)
                        ):
                            self.graph.add_edge(
                                node_id, target_id,
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

        logger.debug(
            "Computing Personalized PageRank from %d query nodes", len(valid_query_nodes)
        )

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
        query_nodes = []
        for result in initial_results:
            # Extract filename from doc_id
            # Format: "{anima}/{memory_type}/{filename}#{chunk_index}"
            doc_id_parts = result.doc_id.split("/")
            if len(doc_id_parts) >= 3:
                filename_with_chunk = doc_id_parts[2]
                filename = filename_with_chunk.split("#")[0]
                node_id = Path(filename).stem

                if node_id in self.graph:
                    query_nodes.append(node_id)

        if not query_nodes:
            logger.debug("No query nodes found in graph")
            return initial_results

        # Compute Personalized PageRank
        pagerank_scores = self.personalized_pagerank(query_nodes)

        if not pagerank_scores:
            return initial_results

        # Find activated nodes (top K by PageRank score, excluding initial results)
        initial_node_ids = {
            Path(result.doc_id.split("/")[2].split("#")[0]).stem
            for result in initial_results if "/" in result.doc_id
        }

        activated_nodes = []
        for node_id, score in pagerank_scores.items():
            if node_id not in initial_node_ids and score > 0.001:  # Threshold
                activated_nodes.append((node_id, score))

        # Sort by score and take top K
        activated_nodes.sort(key=lambda x: x[1], reverse=True)
        top_activated = activated_nodes[:5]  # Add top 5 activated nodes

        logger.debug("Found %d activated nodes, adding top %d", len(activated_nodes), len(top_activated))

        # Convert activated nodes to RetrievalResult format with real content
        from core.memory.rag.retriever import RetrievalResult

        expanded_results = list(initial_results)

        for node_id, score in top_activated:
            node_path_str = self.graph.nodes[node_id].get("path", "")
            node_path = Path(node_path_str)

            # Fetch real content from file
            content = self._fetch_node_content(node_id, node_path)

            expanded_results.append(
                RetrievalResult(
                    doc_id=f"{self.indexer.anima_name}/knowledge/{node_id}.md#0",
                    content=content,
                    score=score * 0.5,  # Reduce score for activated nodes
                    metadata={
                        "source_file": f"knowledge/{node_id}.md",
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

    def _fetch_node_content(self, node_id: str, node_path: Path) -> str:
        """Fetch real content for an activated node.

        Tries file read first, falls back to vector store chunk retrieval.

        Args:
            node_id: Node identifier (filename stem)
            node_path: Path to the knowledge file

        Returns:
            File content string
        """
        # Try reading from file
        if node_path.exists():
            try:
                return node_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read file %s: %s", node_path, e)

        # Fallback: try fetching from vector store
        try:
            collection_name = f"{self.indexer.anima_name}_knowledge"
            # Use a dummy embedding to search by doc_id pattern
            # This is a best-effort fallback
            doc_id_prefix = f"{self.indexer.anima_name}/knowledge/{node_id}.md"
            embedding = self.indexer._generate_embeddings([node_id])[0]
            results = self.vector_store.query(
                collection=collection_name,
                embedding=embedding,
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
) -> KnowledgeGraph:
    """Create and build a knowledge graph for an anima.

    Args:
        anima_name: Anima name
        knowledge_dir: Path to knowledge directory
        vector_store: VectorStore instance
        indexer: MemoryIndexer instance

    Returns:
        Built KnowledgeGraph instance
    """
    graph = KnowledgeGraph(vector_store, indexer)
    graph.build_graph(anima_name, knowledge_dir)
    return graph
