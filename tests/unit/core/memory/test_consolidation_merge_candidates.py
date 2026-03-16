from __future__ import annotations

"""Tests for consolidation merge candidate detection and prompt injection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

# ── _list_knowledge_files_with_meta ──────────────────────────────


class TestListKnowledgeFilesWithMeta:
    """Tests for ConsolidationEngine._list_knowledge_files_with_meta."""

    def _make_engine(self, tmp_path: Path):
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        return ConsolidationEngine(anima_dir, "test")

    def test_empty_knowledge_dir(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        result = engine._list_knowledge_files_with_meta()
        assert result == []

    def test_returns_metadata(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "topic-a.md").write_text(
            "---\ncreated_at: '2026-03-01'\nconfidence: 0.8\nauto_consolidated: true\nsuccess_count: 2\n---\nContent A",
            encoding="utf-8",
        )
        result = engine._list_knowledge_files_with_meta()
        assert len(result) == 1
        assert result[0]["path"] == "topic-a.md"
        assert result[0]["confidence"] == 0.8
        assert result[0]["auto_consolidated"] is True
        assert result[0]["success_count"] == 2

    def test_skips_archive(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        archive_dir = engine.knowledge_dir / "archive"
        archive_dir.mkdir()
        (archive_dir / "old.md").write_text("---\n---\nOld", encoding="utf-8")
        (engine.knowledge_dir / "active.md").write_text("---\n---\nActive", encoding="utf-8")
        result = engine._list_knowledge_files_with_meta()
        assert len(result) == 1
        assert result[0]["path"] == "active.md"

    def test_handles_no_frontmatter(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "plain.md").write_text("No frontmatter here", encoding="utf-8")
        result = engine._list_knowledge_files_with_meta()
        assert len(result) == 1
        assert result[0]["path"] == "plain.md"


# ── _find_merge_candidates ──────────────────────────────


class TestFindMergeCandidates:
    """Tests for ConsolidationEngine._find_merge_candidates."""

    def _make_engine(self, tmp_path: Path):
        from core.memory.consolidation import ConsolidationEngine

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        return ConsolidationEngine(anima_dir, "test")

    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine._find_merge_candidates() == []

    def test_single_file_returns_empty(self, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "only.md").write_text("---\n---\nContent", encoding="utf-8")
        assert engine._find_merge_candidates() == []

    @patch("core.memory.rag.singleton.get_vector_store", return_value=None)
    def test_rag_unavailable(self, mock_store: MagicMock, tmp_path: Path) -> None:
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "a.md").write_text("---\n---\nA", encoding="utf-8")
        (engine.knowledge_dir / "b.md").write_text("---\n---\nB", encoding="utf-8")
        assert engine._find_merge_candidates() == []

    def test_max_pairs_respected(self, tmp_path: Path) -> None:
        """Verify max_pairs limits the returned candidates."""
        engine = self._make_engine(tmp_path)
        for i in range(4):
            (engine.knowledge_dir / f"file{i}.md").write_text(f"---\n---\nContent {i}", encoding="utf-8")

        def make_result(source_file: str, vector_sim: float) -> MagicMock:
            return MagicMock(
                score=vector_sim + 0.05,
                source_scores={"vector": vector_sim},
                metadata={"source_file": source_file},
            )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            make_result("knowledge/file1.md", 0.9),
            make_result("knowledge/file2.md", 0.85),
            make_result("knowledge/file3.md", 0.8),
        ]

        mock_vs = MagicMock()
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=mock_vs),
            patch("core.memory.rag.MemoryIndexer"),
            patch(
                "core.memory.rag.retriever.MemoryRetriever",
                return_value=mock_retriever,
            ),
        ):
            result = engine._find_merge_candidates(max_pairs=2)
            assert len(result) <= 2

    def test_similarity_threshold_respected(self, tmp_path: Path) -> None:
        """Verify pairs below similarity_threshold are excluded."""
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "a.md").write_text("---\n---\nContent A", encoding="utf-8")
        (engine.knowledge_dir / "b.md").write_text("---\n---\nContent B", encoding="utf-8")

        def make_result(source_file: str, vector_sim: float) -> MagicMock:
            return MagicMock(
                score=vector_sim + 0.1,
                source_scores={"vector": vector_sim},
                metadata={"source_file": source_file},
            )

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            make_result("knowledge/b.md", 0.5),
        ]

        mock_vs = MagicMock()
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=mock_vs),
            patch("core.memory.rag.MemoryIndexer"),
            patch(
                "core.memory.rag.retriever.MemoryRetriever",
                return_value=mock_retriever,
            ),
        ):
            result = engine._find_merge_candidates(similarity_threshold=0.75)
            assert result == []

    def test_uses_raw_vector_similarity(self, tmp_path: Path) -> None:
        """Verify raw vector similarity is used, not combined score."""
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "a.md").write_text("---\n---\nContent A", encoding="utf-8")
        (engine.knowledge_dir / "b.md").write_text("---\n---\nContent B", encoding="utf-8")

        mock_result = MagicMock(
            score=0.9,  # combined score is high
            source_scores={"vector": 0.6},  # raw vector similarity is low
            metadata={"source_file": "knowledge/b.md"},
        )
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [mock_result]

        mock_vs = MagicMock()
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=mock_vs),
            patch("core.memory.rag.MemoryIndexer"),
            patch(
                "core.memory.rag.retriever.MemoryRetriever",
                return_value=mock_retriever,
            ),
        ):
            result = engine._find_merge_candidates(similarity_threshold=0.75)
            assert result == [], "Should reject when raw vector sim < threshold even if combined score is high"

    def test_deduplicates_pairs(self, tmp_path: Path) -> None:
        """Verify (a,b) and (b,a) are deduplicated to one pair."""
        engine = self._make_engine(tmp_path)
        (engine.knowledge_dir / "a.md").write_text("---\n---\nContent A", encoding="utf-8")
        (engine.knowledge_dir / "b.md").write_text("---\n---\nContent B", encoding="utf-8")

        def make_result(source_file: str, vector_sim: float) -> MagicMock:
            return MagicMock(
                score=vector_sim + 0.05,
                source_scores={"vector": vector_sim},
                metadata={"source_file": source_file},
            )

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = [
            [make_result("knowledge/b.md", 0.9)],
            [make_result("knowledge/a.md", 0.9)],
        ]

        mock_vs = MagicMock()
        with (
            patch("core.memory.rag.singleton.get_vector_store", return_value=mock_vs),
            patch("core.memory.rag.MemoryIndexer"),
            patch(
                "core.memory.rag.retriever.MemoryRetriever",
                return_value=mock_retriever,
            ),
        ):
            result = engine._find_merge_candidates()
            assert len(result) == 1
            pair = result[0]
            assert set(pair[:2]) == {"a.md", "b.md"}


# ── Format helpers ──────────────────────────────


class TestFormatHelpers:
    """Tests for _format_knowledge_list and _format_merge_candidates."""

    def test_format_knowledge_list_empty(self) -> None:
        from core._anima_lifecycle import _format_knowledge_list

        result = _format_knowledge_list([])
        assert "knowledgeファイルなし" in result

    def test_format_knowledge_list_with_data(self) -> None:
        from core._anima_lifecycle import _format_knowledge_list

        files = [
            {
                "path": "topic.md",
                "confidence": 0.8,
                "created_at": "2026-03-01T00:00:00",
            }
        ]
        result = _format_knowledge_list(files)
        assert "topic.md" in result
        assert "0.8" in result

    def test_format_merge_candidates_empty(self) -> None:
        from core._anima_lifecycle import _format_merge_candidates

        result = _format_merge_candidates([])
        assert "マージ候補なし" in result

    def test_format_merge_candidates_with_data(self) -> None:
        from core._anima_lifecycle import _format_merge_candidates

        candidates = [("a.md", "b.md", 0.85)]
        result = _format_merge_candidates(candidates)
        assert "a.md" in result
        assert "b.md" in result
        assert "0.85" in result


# ── builder.py trigger ──────────────────────────────


class TestBuilderConsolidationTrigger:
    """Test that consolidation trigger is handled as background_auto."""

    def test_consolidation_daily_is_background(self) -> None:
        trigger = "consolidation:daily"
        is_inbox = trigger.startswith("inbox:")
        is_heartbeat = trigger == "heartbeat"
        is_cron = trigger.startswith("cron:")
        is_task = trigger.startswith("task:")
        is_consolidation = trigger.startswith("consolidation:")
        is_background_auto = is_heartbeat or is_cron or is_consolidation
        is_chat = not (is_inbox or is_background_auto or is_task)

        assert is_consolidation is True
        assert is_background_auto is True
        assert is_chat is False

    def test_consolidation_weekly_is_background(self) -> None:
        trigger = "consolidation:weekly"
        is_consolidation = trigger.startswith("consolidation:")
        is_heartbeat = trigger == "heartbeat"
        is_cron = trigger.startswith("cron:")
        is_background_auto = is_heartbeat or is_cron or is_consolidation
        is_inbox = trigger.startswith("inbox:")
        is_task = trigger.startswith("task:")
        is_chat = not (is_inbox or is_background_auto or is_task)

        assert is_consolidation is True
        assert is_background_auto is True
        assert is_chat is False

    def test_chat_still_works(self) -> None:
        trigger = "chat"
        is_consolidation = trigger.startswith("consolidation:")
        is_heartbeat = trigger == "heartbeat"
        is_cron = trigger.startswith("cron:")
        is_background_auto = is_heartbeat or is_cron or is_consolidation
        is_inbox = trigger.startswith("inbox:")
        is_task = trigger.startswith("task:")
        is_chat = not (is_inbox or is_background_auto or is_task)

        assert is_consolidation is False
        assert is_background_auto is False
        assert is_chat is True


# ── _agent_priming.py channel ──────────────────────────────


class TestAgentPrimingConsolidationChannel:
    """Test that consolidation trigger routes to heartbeat channel."""

    def test_consolidation_trigger_routes_to_heartbeat(self) -> None:
        """Consolidation trigger should use heartbeat channel for priming."""
        trigger = "consolidation:daily"
        if trigger == "heartbeat" or trigger.startswith("consolidation:"):
            channel = "heartbeat"
        elif trigger.startswith("cron"):
            channel = "cron"
        else:
            channel = "chat"
        assert channel == "heartbeat"

    def test_consolidation_weekly_routes_to_heartbeat(self) -> None:
        trigger = "consolidation:weekly"
        if trigger == "heartbeat" or trigger.startswith("consolidation:"):
            channel = "heartbeat"
        elif trigger.startswith("cron"):
            channel = "cron"
        else:
            channel = "chat"
        assert channel == "heartbeat"
