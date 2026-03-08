from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for recursive directory indexing (Issue #20).

Verifies the full pipeline: subdirectory files are indexed, stored in
ChromaDB, and retrievable via MemoryRetriever.search.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

chromadb = pytest.importorskip(
    "chromadb", reason="ChromaDB not installed. Install with: pip install 'animaworks[rag]'"
)
pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed. Install with: pip install 'animaworks[rag]'",
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_dirs():
    """Create temp directory tree with subdirectory-based common_knowledge."""
    tmpdir = Path(tempfile.mkdtemp())
    data_dir = tmpdir / "data"
    data_dir.mkdir()

    # Anima directory
    anima_dir = data_dir / "animas" / "test_anima"
    anima_dir.mkdir(parents=True)
    for sub in ("knowledge", "episodes", "procedures", "skills", "state"):
        (anima_dir / sub).mkdir()

    (anima_dir / "knowledge" / "personal-notes.md").write_text(
        "# Personal Notes\n\n## Summary\nMy private knowledge.\n",
        encoding="utf-8",
    )

    # common_knowledge with subdirectories
    ckdir = data_dir / "common_knowledge"
    ckdir.mkdir()

    (ckdir / "top-level-guide.md").write_text(
        "# Top Level Guide\n\n## Overview\nA guide at the root level.\n",
        encoding="utf-8",
    )

    org_dir = ckdir / "organization"
    org_dir.mkdir()
    (org_dir / "structure.md").write_text(
        "# Organization Structure\n\n## Hierarchy\n"
        "CEO at the top, followed by VPs, directors, and team leads.\n"
        "Each team has a dedicated manager responsible for sprint planning.\n",
        encoding="utf-8",
    )
    (org_dir / "roles.md").write_text(
        "# Roles and Responsibilities\n\n## Engineer\n"
        "Engineers write production code and unit tests.\n"
        "## Manager\nManagers handle sprint planning and stakeholder communication.\n",
        encoding="utf-8",
    )

    comm_dir = ckdir / "communication"
    comm_dir.mkdir()
    (comm_dir / "messaging-guide.md").write_text(
        "# Messaging Guide\n\n## DM Rules\n"
        "Maximum 2 recipients per run. Use Board for 3+ recipients.\n"
        "## Rate Limits\n30 messages per hour, 100 per day.\n",
        encoding="utf-8",
    )

    # common_skills with SKILL.md + templates (should be excluded)
    csdir = data_dir / "common_skills"
    csdir.mkdir()
    s1 = csdir / "tool-creator"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "# Tool Creator\n\nCreate external tools for AnimaWorks.\n",
        encoding="utf-8",
    )
    tpl = s1 / "templates"
    tpl.mkdir()
    (tpl / "template.md").write_text(
        "# Template\n\nThis is a template and should NOT be indexed.\n",
        encoding="utf-8",
    )

    yield anima_dir, ckdir, csdir, data_dir

    shutil.rmtree(tmpdir)


@pytest.fixture
def vector_store(temp_dirs):
    """Create temporary ChromaDB vector store."""
    anima_dir, _, _, _ = temp_dirs
    from core.memory.rag.store import ChromaVectorStore

    vectordb_dir = anima_dir.parent.parent / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(persist_dir=vectordb_dir)


# ── Test: Subdirectory files are indexed and searchable ─────────


def test_common_knowledge_subdirectory_indexing_and_search(temp_dirs, vector_store):
    """Files in common_knowledge subdirectories are indexed and found by RAG search."""
    anima_dir, ckdir, _, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever

    # Index personal knowledge
    personal_indexer = MemoryIndexer(vector_store, "test_anima", anima_dir)
    personal_indexer.index_directory(anima_dir / "knowledge", "knowledge")

    # Index common_knowledge (recursive — including subdirectories)
    shared_indexer = MemoryIndexer(
        vector_store, "shared", data_dir, collection_prefix="shared",
    )
    ck_chunks = shared_indexer.index_directory(ckdir, "common_knowledge")

    # All 4 files should be indexed: top-level-guide + organization/structure +
    # organization/roles + communication/messaging-guide
    assert ck_chunks >= 4, (
        f"Expected at least 4 chunks from 4 files (with heading sections), got {ck_chunks}"
    )

    # Verify collection exists
    collections = vector_store.list_collections()
    assert "shared_common_knowledge" in collections

    # Search for content that only exists in a subdirectory file
    retriever = MemoryRetriever(
        vector_store, personal_indexer, anima_dir / "knowledge",
    )
    results = retriever.search(
        query="organization hierarchy CEO VPs directors",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
        include_shared=True,
    )

    assert len(results) > 0, "Search should return results"

    shared_results = [r for r in results if r.metadata.get("anima") == "shared"]
    assert len(shared_results) > 0, (
        "Should find results from shared common_knowledge subdirectory files"
    )

    found_org = any(
        "organization" in str(r.metadata.get("source_file", "")).lower()
        or "hierarchy" in r.content.lower()
        for r in shared_results
    )
    assert found_org, (
        f"Expected organization/structure.md content in results. "
        f"Got: {[(r.metadata.get('source_file'), r.content[:80]) for r in shared_results]}"
    )


def test_common_knowledge_messaging_guide_searchable(temp_dirs, vector_store):
    """communication/messaging-guide.md is searchable via RAG."""
    anima_dir, ckdir, _, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.retriever import MemoryRetriever

    personal_indexer = MemoryIndexer(vector_store, "test_anima", anima_dir)
    personal_indexer.index_directory(anima_dir / "knowledge", "knowledge")

    shared_indexer = MemoryIndexer(
        vector_store, "shared", data_dir, collection_prefix="shared",
    )
    shared_indexer.index_directory(ckdir, "common_knowledge")

    retriever = MemoryRetriever(
        vector_store, personal_indexer, anima_dir / "knowledge",
    )
    results = retriever.search(
        query="DM rate limits messaging rules recipients",
        anima_name="test_anima",
        memory_type="knowledge",
        top_k=5,
        include_shared=True,
    )

    shared_results = [r for r in results if r.metadata.get("anima") == "shared"]
    found_messaging = any(
        "messaging" in str(r.metadata.get("source_file", "")).lower()
        or "rate limit" in r.content.lower()
        or "recipients" in r.content.lower()
        for r in shared_results
    )
    assert found_messaging, (
        "Should find communication/messaging-guide.md content via RAG search"
    )


# ── Test: Skills only index SKILL.md ────────────────────────────


def test_common_skills_only_indexes_skill_md(temp_dirs, vector_store):
    """common_skills indexing only picks up SKILL.md, not templates."""
    _, _, csdir, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer

    shared_indexer = MemoryIndexer(
        vector_store, "shared", data_dir, collection_prefix="shared",
    )
    cs_chunks = shared_indexer.index_directory(csdir, "common_skills")

    assert cs_chunks > 0, "Should index at least one SKILL.md"

    collections = vector_store.list_collections()
    assert "shared_common_skills" in collections

    # Verify that template files were NOT indexed
    from core.memory.rag.singleton import get_embedding_dimension

    dim = get_embedding_dimension()
    query_embedding = [0.0] * dim
    results = vector_store.query(
        "shared_common_skills", query_embedding, top_k=20,
    )

    for r in results:
        sf = r.document.metadata.get("source_file", "")
        assert "template" not in sf.lower(), (
            f"Template file should not be indexed, found: {sf}"
        )
        assert sf.endswith("SKILL.md") or "SKILL" in sf, (
            f"Only SKILL.md should be indexed, found: {sf}"
        )


# ── Test: source_file metadata has subdirectory path ────────────


def test_source_file_metadata_includes_subdirectory(temp_dirs, vector_store):
    """Indexed chunks have source_file metadata with subdirectory paths."""
    _, ckdir, _, data_dir = temp_dirs

    from core.memory.rag.indexer import MemoryIndexer
    from core.memory.rag.singleton import get_embedding_dimension

    shared_indexer = MemoryIndexer(
        vector_store, "shared", data_dir, collection_prefix="shared",
    )
    shared_indexer.index_directory(ckdir, "common_knowledge")

    dim = get_embedding_dimension()
    query_embedding = [0.0] * dim
    results = vector_store.query(
        "shared_common_knowledge", query_embedding, top_k=20,
    )

    source_files = {r.document.metadata.get("source_file", "") for r in results}

    has_subdir = any("/" in sf and "organization" in sf for sf in source_files)
    assert has_subdir, (
        f"Expected source_file with subdirectory path (organization/...), "
        f"got: {source_files}"
    )

    has_top_level = any("top-level-guide" in sf for sf in source_files)
    assert has_top_level, (
        f"Expected top-level file in source_files, got: {source_files}"
    )
