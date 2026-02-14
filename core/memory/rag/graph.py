from __future__ import annotations
# AnimaWorks - Digital Person Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Knowledge graph for spreading activation in memory retrieval.

Implements:
- Graph construction from markdown links and vector similarity
- Personalized PageRank for multi-hop activation
- Spreading activation to expand search results

Based on: docs/design/priming-layer-design.md Phase 3
"""

import logging
import re
from pathlib import Path

import networkx as nx

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


# ── KnowledgeGraph ──────────────────────────────────────────────────


class KnowledgeGraph:
    """Knowledge graph for spreading activation.

    Builds a directed graph where:
    - Nodes = knowledge files
    - Edges = explicit links (Markdown [[filename]]) + implicit links (vector similarity)

    Provides Personalized PageRank for multi-hop activation.
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

    def build_graph(self, person_name: str, knowledge_dir: Path) -> nx.DiGraph:
        """Build knowledge graph from knowledge files.

        Args:
            person_name: Person name (for collection selection)
            knowledge_dir: Path to knowledge directory

        Returns:
            NetworkX directed graph
        """
        logger.info("Building knowledge graph for person=%s", person_name)

        graph = nx.DiGraph()

        if not knowledge_dir.is_dir():
            logger.warning("Knowledge directory not found: %s", knowledge_dir)
            return graph

        # Collect all markdown files
        md_files = sorted(knowledge_dir.glob("*.md"))
        if not md_files:
            logger.warning("No markdown files found in %s", knowledge_dir)
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
                        graph.add_edge(source_id, target_id, link_type="explicit")
                        logger.debug("Explicit link: %s -> %s", source_id, target_id)

            except Exception as e:
                logger.warning("Failed to extract links from %s: %s", md_file, e)

        logger.debug("Added %d explicit edges", graph.number_of_edges())

        # Add implicit links (vector similarity)
        implicit_count = self._add_implicit_links(graph, person_name)
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

    def _add_implicit_links(self, graph: nx.DiGraph, person_name: str) -> int:
        """Add implicit links based on vector similarity.

        Args:
            graph: NetworkX graph to modify
            person_name: Person name for collection selection

        Returns:
            Number of implicit edges added
        """
        collection_name = f"{person_name}_knowledge"
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
                    # Format: "{person}/{memory_type}/{filename}#{chunk_index}"
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

    # ── Personalized PageRank ───────────────────────────────────────

    def personalized_pagerank(
        self,
        query_nodes: list[str],
        alpha: float = PAGERANK_ALPHA,
    ) -> dict[str, float]:
        """Compute Personalized PageRank scores.

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
            # Compute Personalized PageRank
            scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=PAGERANK_MAX_ITER,
                tol=PAGERANK_TOL,
            )

            logger.debug("PageRank computed for %d nodes", len(scores))
            return scores

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

        Args:
            initial_results: Initial retrieval results (from hybrid search)
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
            # Format: "{person}/{memory_type}/{filename}#{chunk_index}"
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
        initial_node_ids = {Path(result.doc_id.split("/")[2].split("#")[0]).stem for result in initial_results if "/" in result.doc_id}

        activated_nodes = []
        for node_id, score in pagerank_scores.items():
            if node_id not in initial_node_ids and score > 0.001:  # Threshold
                activated_nodes.append((node_id, score))

        # Sort by score and take top K
        activated_nodes.sort(key=lambda x: x[1], reverse=True)
        top_activated = activated_nodes[:5]  # Add top 5 activated nodes

        logger.debug("Found %d activated nodes, adding top %d", len(activated_nodes), len(top_activated))

        # Convert activated nodes to RetrievalResult format
        # Note: This is a simplified implementation
        # In production, you'd fetch actual content from vector store
        from core.memory.rag.retriever import RetrievalResult

        expanded_results = list(initial_results)

        for node_id, score in top_activated:
            node_path = self.graph.nodes[node_id].get("path", "")

            # Create pseudo result
            # In real implementation, fetch actual chunk from vector store
            expanded_results.append(
                RetrievalResult(
                    doc_id=f"{self.indexer.person_name}/knowledge/{node_id}.md#0",
                    content=f"[Activated node: {node_id}]",  # Placeholder
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


# ── Public API ──────────────────────────────────────────────────────


def create_knowledge_graph(
    person_name: str,
    knowledge_dir: Path,
    vector_store,
    indexer,
) -> KnowledgeGraph:
    """Create and build a knowledge graph for a person.

    Args:
        person_name: Person name
        knowledge_dir: Path to knowledge directory
        vector_store: VectorStore instance
        indexer: MemoryIndexer instance

    Returns:
        Built KnowledgeGraph instance
    """
    graph = KnowledgeGraph(vector_store, indexer)
    graph.build_graph(person_name, knowledge_dir)
    return graph
