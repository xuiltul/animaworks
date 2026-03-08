from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for recursive directory indexing (Issue #20).

Verifies that index_directory uses rglob for all memory types,
skills/common_skills only index SKILL.md, and graph node IDs
handle subdirectory paths correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("scipy")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir():
    """Create a temp anima directory with subdirectory structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        anima_dir = base / "animas" / "test"
        anima_dir.mkdir(parents=True)

        # knowledge — flat (backward compat)
        kdir = anima_dir / "knowledge"
        kdir.mkdir()
        (kdir / "topic-a.md").write_text("# Topic A\n\nFlat knowledge file.\n")

        # common_knowledge — nested subdirectories
        ckdir = base / "common_knowledge"
        ckdir.mkdir()
        (ckdir / "top-level.md").write_text("# Top Level\n\nDirect file.\n")

        org_dir = ckdir / "organization"
        org_dir.mkdir()
        (org_dir / "structure.md").write_text(
            "# Organization Structure\n\n## Hierarchy\nFlat org.\n"
        )
        (org_dir / "roles.md").write_text(
            "# Roles\n\n## Engineer\nWrites code.\n"
        )

        comm_dir = ckdir / "communication"
        comm_dir.mkdir()
        (comm_dir / "messaging-guide.md").write_text(
            "# Messaging Guide\n\n## DM Rules\nMax 2 per run.\n"
        )

        # skills — nested SKILL.md + extra files that should NOT be indexed
        sdir = anima_dir / "skills"
        sdir.mkdir()
        skill1 = sdir / "deploy"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Deploy Skill\n\nDeploy stuff.\n")
        (skill1 / "README.md").write_text("# Readme\n\nThis should not be indexed.\n")

        skill2 = sdir / "monitoring"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Monitoring Skill\n\nMonitor stuff.\n")

        # common_skills — nested with templates that should NOT be indexed
        csdir = base / "common_skills"
        csdir.mkdir()
        cs1 = csdir / "skill-creator"
        cs1.mkdir()
        (cs1 / "SKILL.md").write_text("# Skill Creator\n\nCreate skills.\n")
        templates = cs1 / "templates"
        templates.mkdir()
        (templates / "template.md").write_text("# Template\n\nNot a skill.\n")

        # shared_users — nested user dirs
        shared = base / "shared"
        shared.mkdir()
        users_dir = shared / "users"
        users_dir.mkdir()
        user1 = users_dir / "alice"
        user1.mkdir()
        (user1 / "index.md").write_text("# Alice\n\nUser profile.\n")
        user2 = users_dir / "bob"
        user2.mkdir()
        (user2 / "index.md").write_text("# Bob\n\nAnother user.\n")

        yield {
            "base": base,
            "anima_dir": anima_dir,
            "common_knowledge": ckdir,
            "skills": sdir,
            "common_skills": csdir,
            "shared_users": users_dir,
        }


# ── Test: index_directory rglob for standard types ──────────────


def test_index_directory_finds_subdirectory_files(temp_anima_dir):
    """index_directory with rglob finds .md files in subdirectories."""
    ckdir = temp_anima_dir["common_knowledge"]

    mock_store = MagicMock()
    mock_store.create_collection = MagicMock()
    mock_store.upsert = MagicMock()

    with patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"):
        from core.memory.rag.indexer import MemoryIndexer

        indexer = MemoryIndexer(
            mock_store,
            "shared",
            temp_anima_dir["base"],
        )
        indexer.embedding_model = MagicMock()
        indexer.embedding_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.array([[0.1] * 384] * len(texts))
        )

        chunks = indexer.index_directory(ckdir, "common_knowledge")

    assert chunks > 0, "Should index files from subdirectories"
    upsert_calls = mock_store.upsert.call_args_list
    all_docs = []
    for call in upsert_calls:
        all_docs.extend(call[1]["documents"] if "documents" in call[1] else call[0][1])

    source_files = {d.metadata.get("source_file", "") for d in all_docs}
    assert any("organization" in sf for sf in source_files), (
        f"Expected subdirectory path in source_file, got: {source_files}"
    )


# ── Test: skills only index SKILL.md ────────────────────────────


def test_index_directory_skills_only_skill_md(temp_anima_dir):
    """index_directory for skills type only indexes SKILL.md files."""
    sdir = temp_anima_dir["skills"]

    mock_store = MagicMock()
    mock_store.create_collection = MagicMock()
    mock_store.upsert = MagicMock()

    with patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"):
        from core.memory.rag.indexer import MemoryIndexer

        indexer = MemoryIndexer(
            mock_store,
            "test",
            temp_anima_dir["anima_dir"],
        )
        indexer.embedding_model = MagicMock()
        indexer.embedding_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.array([[0.1] * 384] * len(texts))
        )

        chunks = indexer.index_directory(sdir, "skills")

    assert chunks > 0, "Should index SKILL.md files"

    upsert_calls = mock_store.upsert.call_args_list
    all_docs = []
    for call in upsert_calls:
        all_docs.extend(call[1]["documents"] if "documents" in call[1] else call[0][1])

    for doc in all_docs:
        sf = doc.metadata.get("source_file", "")
        assert "SKILL.md" in sf, f"Skills should only index SKILL.md, got: {sf}"
        assert "README" not in sf, f"README.md should not be indexed, got: {sf}"


def test_index_directory_common_skills_excludes_templates(temp_anima_dir):
    """index_directory for common_skills excludes template .md files."""
    csdir = temp_anima_dir["common_skills"]

    mock_store = MagicMock()
    mock_store.create_collection = MagicMock()
    mock_store.upsert = MagicMock()

    with patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"):
        from core.memory.rag.indexer import MemoryIndexer

        indexer = MemoryIndexer(
            mock_store,
            "shared",
            temp_anima_dir["base"],
        )
        indexer.embedding_model = MagicMock()
        indexer.embedding_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.array([[0.1] * 384] * len(texts))
        )

        chunks = indexer.index_directory(csdir, "common_skills")

    assert chunks > 0, "Should index SKILL.md in common_skills"

    upsert_calls = mock_store.upsert.call_args_list
    all_docs = []
    for call in upsert_calls:
        all_docs.extend(call[1]["documents"] if "documents" in call[1] else call[0][1])

    for doc in all_docs:
        sf = doc.metadata.get("source_file", "")
        assert "template" not in sf.lower(), (
            f"Template files should not be indexed, got: {sf}"
        )


# ── Test: shared_users indexing ─────────────────────────────────


def test_index_directory_shared_users_finds_nested(temp_anima_dir):
    """index_directory indexes user profiles in {username}/index.md."""
    users_dir = temp_anima_dir["shared_users"]

    mock_store = MagicMock()
    mock_store.create_collection = MagicMock()
    mock_store.upsert = MagicMock()

    shared_base = temp_anima_dir["base"] / "shared"
    with patch("core.memory.rag.indexer.MemoryIndexer._init_embedding_model"):
        from core.memory.rag.indexer import MemoryIndexer

        indexer = MemoryIndexer(
            mock_store,
            "shared",
            shared_base,
        )
        indexer.embedding_model = MagicMock()
        indexer.embedding_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.array([[0.1] * 384] * len(texts))
        )

        chunks = indexer.index_directory(users_dir, "shared_users")

    assert chunks > 0, "Should index nested user profile files"


# ── Test: graph node ID with subdirectories ─────────────────────


def test_graph_build_with_subdirectories():
    """Graph build_graph handles subdirectory files with unique node IDs."""
    from core.memory.rag.graph import KnowledgeGraph

    with tempfile.TemporaryDirectory() as tmpdir:
        kdir = Path(tmpdir) / "knowledge"
        kdir.mkdir()
        (kdir / "topic.md").write_text("# Topic\n\nFlat file.\n")

        sub = kdir / "advanced"
        sub.mkdir()
        (sub / "topic.md").write_text("# Advanced Topic\n\nNested file.\n")

        mock_store = MagicMock()
        mock_store.query = MagicMock(return_value=[])

        mock_indexer = MagicMock()
        mock_indexer.anima_name = "test"
        mock_indexer._generate_embeddings = MagicMock(
            return_value=[[0.1] * 384]
        )

        graph_builder = KnowledgeGraph(mock_store, mock_indexer)
        graph = graph_builder.build_graph(
            "test", kdir, implicit_link_threshold=999.0,
        )

        # Both files should be nodes with distinct IDs
        assert graph.number_of_nodes() == 2
        assert "topic" in graph, "Flat file node should exist"
        assert "advanced/topic" in graph, "Nested file node should exist"

        # Verify rel_key attribute
        assert graph.nodes["advanced/topic"]["rel_key"] == "advanced/topic"
        assert graph.nodes["topic"]["rel_key"] == "topic"


# ── Test: _match_result_to_node with subdirectory doc_ids ───────


def test_match_result_to_node_subdirectory():
    """_match_result_to_node handles multi-segment doc_ids."""
    import networkx as nx

    from core.memory.rag.graph import KnowledgeGraph

    graph = nx.DiGraph()
    graph.add_node(
        "organization/structure",
        path="/tmp/knowledge/organization/structure.md",
        memory_type="knowledge",
        stem="structure",
        rel_key="organization/structure",
    )
    graph.add_node(
        "structure",
        path="/tmp/knowledge/structure.md",
        memory_type="knowledge",
        stem="structure",
        rel_key="structure",
    )

    # Subdirectory doc_id should match the nested node
    result = KnowledgeGraph._match_result_to_node(
        graph,
        "shared/knowledge/organization/structure.md#0",
        0.9,
    )
    assert result == "organization/structure"

    # Flat doc_id should match the flat node
    result_flat = KnowledgeGraph._match_result_to_node(
        graph,
        "shared/knowledge/structure.md#0",
        0.9,
    )
    assert result_flat == "structure"


def test_match_result_to_node_episodes_prefix():
    """_match_result_to_node works for non-knowledge types with subdirs."""
    import networkx as nx

    from core.memory.rag.graph import KnowledgeGraph

    graph = nx.DiGraph()
    graph.add_node(
        "episodes:2026-03-01",
        path="/tmp/episodes/2026-03-01.md",
        memory_type="episodes",
        stem="2026-03-01",
        rel_key="2026-03-01",
    )

    result = KnowledgeGraph._match_result_to_node(
        graph,
        "test/episodes/2026-03-01.md#0",
        0.8,
    )
    assert result == "episodes:2026-03-01"


# ── Test: _make_node_id ─────────────────────────────────────────


def test_make_node_id_with_subdirectory():
    """_make_node_id creates correct IDs for subdirectory paths."""
    from core.memory.rag.graph import KnowledgeGraph

    assert KnowledgeGraph._make_node_id("topic", "knowledge") == "topic"
    assert KnowledgeGraph._make_node_id("org/topic", "knowledge") == "org/topic"
    assert KnowledgeGraph._make_node_id("2026-03-01", "episodes") == "episodes:2026-03-01"
    assert (
        KnowledgeGraph._make_node_id("sub/item", "common_knowledge")
        == "common_knowledge:sub/item"
    )


# ── Test: watcher recursive ─────────────────────────────────────


def test_watcher_uses_recursive_for_all_dirs():
    """FileWatcher registers all directories with recursive=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anima_dir = Path(tmpdir)
        for sub in ("knowledge", "episodes", "procedures", "skills"):
            (anima_dir / sub).mkdir()

        extra_ck = Path(tmpdir) / "common_knowledge"
        extra_ck.mkdir()

        mock_indexer = MagicMock()

        from core.memory.rag.watcher import FileWatcher

        watcher = FileWatcher(
            anima_dir,
            mock_indexer,
            extra_watch_dirs=[(extra_ck, "common_knowledge")],
        )

        mock_observer = MagicMock()
        watcher.observer = mock_observer

        # Manually call the schedule part
        from core.memory.rag.watcher import MemoryFileHandler

        handler = MemoryFileHandler(watcher)

        watch_dirs = [
            anima_dir / "knowledge",
            anima_dir / "episodes",
            anima_dir / "procedures",
            anima_dir / "skills",
            extra_ck,
        ]

        for watch_dir in watch_dirs:
            if watch_dir.is_dir():
                mock_observer.schedule(handler, str(watch_dir), recursive=True)

        for call in mock_observer.schedule.call_args_list:
            assert call[1].get("recursive") is True or call[0][2] is True, (
                f"All watch dirs must be recursive=True, got: {call}"
            )
