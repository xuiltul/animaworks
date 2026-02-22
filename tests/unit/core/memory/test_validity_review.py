from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.


"""Unit tests for knowledge validity review during daily consolidation.

Tests the _select_review_candidates(), _run_validity_review(), and
_process_review_verdicts() methods added to ConsolidationEngine.
All external dependencies (LLM, RAG) are mocked.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def temp_anima_dir(tmp_path: Path) -> Path:
    """Create a temporary anima directory structure."""
    anima_dir = tmp_path / "test_anima"
    (anima_dir / "episodes").mkdir(parents=True)
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "procedures").mkdir(parents=True)
    (anima_dir / "skills").mkdir(parents=True)
    (anima_dir / "state").mkdir(parents=True)
    (anima_dir / "archive" / "superseded").mkdir(parents=True)
    return anima_dir


@pytest.fixture
def engine(temp_anima_dir: Path):
    """Create a ConsolidationEngine instance."""
    from core.memory.consolidation import ConsolidationEngine

    return ConsolidationEngine(
        anima_dir=temp_anima_dir,
        anima_name="test_anima",
    )


def _write_knowledge_file(
    anima_dir: Path, name: str, content: str, meta: dict | None = None,
) -> Path:
    """Helper to write a knowledge file with frontmatter."""
    from core.memory.manager import MemoryManager

    mm = MemoryManager(anima_dir)
    path = anima_dir / "knowledge" / name
    if meta is None:
        meta = {
            "created_at": "2026-02-20T09:00:00",
            "confidence": 0.7,
            "auto_consolidated": True,
            "success_count": 0,
            "failure_count": 0,
            "version": 1,
            "last_used": "",
        }
    mm.write_knowledge_with_meta(path, content, meta)
    return path


def _make_llm_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_rag_result(source_file: str, content: str = "", score: float = 0.8):
    """Build a mock RAG RetrievalResult."""
    result = MagicMock()
    result.metadata = {"source_file": source_file}
    result.content = content
    result.score = score
    return result


# ── Candidate Selection Tests ──────────────────────────────


class TestSelectReviewCandidates:
    """Tests for _select_review_candidates()."""

    def test_from_rag(self, engine: object, temp_anima_dir: Path) -> None:
        """RAG results are included as candidates."""
        kfile = _write_knowledge_file(
            temp_anima_dir, "api-design.md", "# API Design\n\nREST patterns.",
        )

        mock_results = [_make_rag_result("api-design.md")]

        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            return_value=mock_results,
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="API設計の見直し",
                        resolved_events=[],
                        exclude_files=[],
                    )

        assert any(c.name == "api-design.md" for c in candidates)

    def test_excludes_new_files(self, engine: object, temp_anima_dir: Path) -> None:
        """Files created in this cycle are excluded."""
        _write_knowledge_file(
            temp_anima_dir, "new-file.md", "# New\n\nJust created.",
        )

        mock_results = [_make_rag_result("new-file.md")]

        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            return_value=mock_results,
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="Something",
                        resolved_events=[],
                        exclude_files=["new-file.md"],
                    )

        assert not any(c.name == "new-file.md" for c in candidates)

    def test_excludes_superseded(self, engine: object, temp_anima_dir: Path) -> None:
        """Files with valid_until set are excluded."""
        _write_knowledge_file(
            temp_anima_dir, "old-info.md", "# Old\n\nSuperseded.",
            meta={
                "created_at": "2026-02-10T09:00:00",
                "confidence": 0.7,
                "valid_until": "2026-02-20T09:00:00",
                "auto_consolidated": True,
            },
        )

        mock_results = [_make_rag_result("old-info.md")]

        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            return_value=mock_results,
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="Something",
                        resolved_events=[],
                        exclude_files=[],
                    )

        assert not any(c.name == "old-info.md" for c in candidates)

    def test_max_cap(self, engine: object, temp_anima_dir: Path) -> None:
        """Candidates are capped at max_candidates."""
        # Create 15 files
        mock_results = []
        for i in range(15):
            name = f"file-{i:02d}.md"
            _write_knowledge_file(
                temp_anima_dir, name, f"# File {i}\n\nContent {i}.",
            )
            mock_results.append(_make_rag_result(name))

        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            return_value=mock_results,
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="Something",
                        resolved_events=[],
                        exclude_files=[],
                        max_candidates=10,
                    )

        assert len(candidates) <= 10

    def test_low_confidence_source_c(self, engine: object, temp_anima_dir: Path) -> None:
        """Source C: low-confidence files with recent last_used are included."""
        from core.time_utils import now_iso

        _write_knowledge_file(
            temp_anima_dir, "shaky.md", "# Shaky\n\nUncertain info.",
            meta={
                "created_at": "2026-02-10T09:00:00",
                "confidence": 0.3,
                "last_used": now_iso(),
                "auto_consolidated": True,
            },
        )

        # No RAG results — only source C should find it
        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            return_value=[],
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="Something",
                        resolved_events=[],
                        exclude_files=[],
                    )

        assert any(c.name == "shaky.md" for c in candidates)

    def test_rag_unavailable_fallback(self, engine: object, temp_anima_dir: Path) -> None:
        """When RAG is unavailable, source C still works."""
        from core.time_utils import now_iso

        _write_knowledge_file(
            temp_anima_dir, "low-conf.md", "# Low\n\nContent.",
            meta={
                "created_at": "2026-02-10T09:00:00",
                "confidence": 0.2,
                "last_used": now_iso(),
                "auto_consolidated": True,
            },
        )

        with patch(
            "core.memory.rag.retriever.MemoryRetriever",
            side_effect=ImportError("RAG not available"),
        ):
            with patch(
                "core.memory.rag.singleton.get_vector_store",
                side_effect=ImportError("RAG not available"),
            ):
                candidates = engine._select_review_candidates(
                    episodes_text="Something",
                    resolved_events=[],
                    exclude_files=[],
                )

        assert any(c.name == "low-conf.md" for c in candidates)


# ── Verdict Processing Tests ───────────────────────────────


class TestProcessReviewVerdicts:
    """Tests for _process_review_verdicts()."""

    @pytest.mark.asyncio
    async def test_stale_file_archives(self, engine: object, temp_anima_dir: Path) -> None:
        """Stale verdict archives the file and creates replacement."""
        path = _write_knowledge_file(
            temp_anima_dir, "stale-info.md",
            "# Stale Info\n\nIPC接続問題は未解決。",
        )

        verdicts = [{
            "file": "stale-info.md",
            "verdict": "stale",
            "reason": "IPC接続問題はすでに解決済み",
            "correction": "# IPC接続\n\nIPC接続問題は2026-02-22に解決済み。dedicated connectionパターンを採用。",
        }]

        result = await engine._process_review_verdicts(
            verdicts, [path], )

        assert result["stale"] == 1
        assert result["reviewed"] == 1

        # Original moved to archive
        archive = temp_anima_dir / "archive" / "superseded" / "stale-info.md"
        assert archive.exists()

        # Replacement created
        replacement = temp_anima_dir / "knowledge" / "stale-info.md"
        assert replacement.exists()

        from core.memory.manager import MemoryManager

        mm = MemoryManager(temp_anima_dir)
        new_content = mm.read_knowledge_content(replacement)
        assert "解決済み" in new_content

        new_meta = mm.read_knowledge_metadata(replacement)
        assert new_meta.get("validity_reviewed") is True
        assert "stale-info.md" in new_meta.get("supersedes", [])

    @pytest.mark.asyncio
    async def test_stale_without_correction(self, engine: object, temp_anima_dir: Path) -> None:
        """Stale verdict without correction just archives, no replacement."""
        path = _write_knowledge_file(
            temp_anima_dir, "obsolete.md", "# Obsolete\n\nOld info.",
        )

        verdicts = [{
            "file": "obsolete.md",
            "verdict": "stale",
            "reason": "Completely outdated",
            "correction": None,
        }]

        result = await engine._process_review_verdicts(
            verdicts, [path], )

        assert result["stale"] == 1
        archive = temp_anima_dir / "archive" / "superseded" / "obsolete.md"
        assert archive.exists()
        # No replacement
        assert not (temp_anima_dir / "knowledge" / "obsolete.md").exists()

    @pytest.mark.asyncio
    async def test_needs_update_appends(self, engine: object, temp_anima_dir: Path) -> None:
        """needs_update verdict appends correction to existing content."""
        path = _write_knowledge_file(
            temp_anima_dir, "partial.md",
            "# Partial\n\nSome correct info.",
        )

        verdicts = [{
            "file": "partial.md",
            "verdict": "needs_update",
            "reason": "追加情報が必要",
            "correction": "新しいAPIエンドポイントが追加された。",
        }]

        result = await engine._process_review_verdicts(
            verdicts, [path], )

        assert result["needs_update"] == 1
        assert result["reviewed"] == 1

        from core.memory.manager import MemoryManager

        mm = MemoryManager(temp_anima_dir)
        content = mm.read_knowledge_content(path)
        assert "Some correct info." in content
        assert "[VALIDITY-REVIEWED:" in content
        assert "新しいAPIエンドポイント" in content

        meta = mm.read_knowledge_metadata(path)
        assert "last_reviewed" in meta
        assert "updated_at" in meta

    @pytest.mark.asyncio
    async def test_valid_updates_metadata(self, engine: object, temp_anima_dir: Path) -> None:
        """valid verdict only updates last_reviewed metadata."""
        path = _write_knowledge_file(
            temp_anima_dir, "good.md", "# Good\n\nAccurate content.",
        )

        verdicts = [{
            "file": "good.md",
            "verdict": "valid",
            "reason": "内容は正確",
            "correction": None,
        }]

        result = await engine._process_review_verdicts(
            verdicts, [path], )

        assert result["valid"] == 1
        assert result["reviewed"] == 1

        from core.memory.manager import MemoryManager

        mm = MemoryManager(temp_anima_dir)
        meta = mm.read_knowledge_metadata(path)
        assert "last_reviewed" in meta

        # Content unchanged
        content = mm.read_knowledge_content(path)
        assert content.strip() == "# Good\n\nAccurate content."

    @pytest.mark.asyncio
    async def test_unknown_file_counts_as_error(self, engine: object) -> None:
        """Verdict referencing unknown file increments error count."""
        verdicts = [{
            "file": "nonexistent.md",
            "verdict": "stale",
            "reason": "test",
            "correction": None,
        }]

        result = await engine._process_review_verdicts(
            verdicts, [], )

        assert result["errors"] == 1
        assert result["reviewed"] == 0

    @pytest.mark.asyncio
    async def test_needs_update_without_correction_is_error(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """needs_update without correction increments error count."""
        path = _write_knowledge_file(
            temp_anima_dir, "incomplete.md", "# Incomplete\n\nSome info.",
        )

        verdicts = [{
            "file": "incomplete.md",
            "verdict": "needs_update",
            "reason": "追加情報が必要",
            "correction": None,
        }]

        result = await engine._process_review_verdicts(
            verdicts, [path],
        )

        assert result["errors"] == 1
        assert result["needs_update"] == 0
        assert result["reviewed"] == 1

    @pytest.mark.asyncio
    async def test_source_b_resolved_events(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """Source B: resolved events RAG search includes candidates."""
        kfile = _write_knowledge_file(
            temp_anima_dir, "ipc-problem.md",
            "# IPC Problem\n\nIPC接続が不安定。",
        )

        mock_results_empty = []
        mock_results_resolved = [_make_rag_result("ipc-problem.md")]

        call_count = 0

        def _mock_search(**kwargs):
            nonlocal call_count
            call_count += 1
            query = kwargs.get("query", "")
            # Source A queries use episodes text, Source B uses resolved text
            if "IPC接続問題を解決" in query:
                return mock_results_resolved
            return mock_results_empty

        with patch(
            "core.memory.rag.retriever.MemoryRetriever.search",
            side_effect=_mock_search,
        ):
            with patch("core.memory.rag.singleton.get_vector_store"):
                with patch("core.memory.rag.MemoryIndexer"):
                    candidates = engine._select_review_candidates(
                        episodes_text="今日の活動",
                        resolved_events=[
                            {"content": "IPC接続問題を解決した"},
                        ],
                        exclude_files=[],
                    )

        assert any(c.name == "ipc-problem.md" for c in candidates)

    @pytest.mark.asyncio
    async def test_stale_procedure_file_archives(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """Stale verdict on a procedure file archives and replaces correctly."""
        from core.memory.manager import MemoryManager

        mm = MemoryManager(temp_anima_dir)
        proc_dir = temp_anima_dir / "procedures"
        proc_path = proc_dir / "old-deploy.md"
        mm.write_knowledge_with_meta(
            proc_path, "# Deploy\n\n古いデプロイ手順。",
            {
                "created_at": "2026-02-10T09:00:00",
                "confidence": 0.7,
                "auto_consolidated": True,
            },
        )

        verdicts = [{
            "file": "old-deploy.md",
            "verdict": "stale",
            "reason": "デプロイ手順が変更された",
            "correction": "# Deploy\n\n新しいデプロイ手順。CI/CDパイプラインを使用。",
        }]

        result = await engine._process_review_verdicts(
            verdicts, [proc_path],
        )

        assert result["stale"] == 1

        # Original moved to archive
        archive = temp_anima_dir / "archive" / "superseded" / "old-deploy.md"
        assert archive.exists()

        # Replacement created in procedures/ (not knowledge/)
        replacement = proc_dir / "old-deploy.md"
        assert replacement.exists()

        new_content = mm.read_knowledge_content(replacement)
        assert "CI/CDパイプライン" in new_content


# ── Run Validity Review Integration Tests ──────────────────


class TestRunValidityReview:
    """Tests for _run_validity_review()."""

    @pytest.mark.asyncio
    async def test_no_candidates_early_return(self, engine: object) -> None:
        """No candidates returns early with zero counts."""
        with patch.object(
            engine, "_select_review_candidates", return_value=[],
        ):
            result = await engine._run_validity_review(
                episode_entries=[{
                    "date": "2026-02-22", "time": "10:00",
                    "content": "Test episode",
                }],
                resolved_events=[],
                files_created=[],
                files_updated=[],
                model="test-model",
            )

        assert result["reviewed"] == 0
        assert result["stale"] == 0

    @pytest.mark.asyncio
    async def test_no_episodes_early_return(self, engine: object) -> None:
        """No episodes returns early with zero counts."""
        result = await engine._run_validity_review(
            episode_entries=[],
            resolved_events=[],
            files_created=[],
            files_updated=[],
            model="test-model",
        )

        assert result["reviewed"] == 0

    @pytest.mark.asyncio
    async def test_llm_parse_failure_safe(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """JSON parse failure returns safely without crashing."""
        path = _write_knowledge_file(
            temp_anima_dir, "review-target.md", "# Target\n\nContent.",
        )

        with patch.object(
            engine, "_select_review_candidates", return_value=[path],
        ):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
                # Return invalid JSON
                mock_llm.return_value = _make_llm_response(
                    "I cannot parse this as JSON properly.",
                )

                result = await engine._run_validity_review(
                    episode_entries=[{
                        "date": "2026-02-22", "time": "10:00",
                        "content": "Test episode",
                    }],
                    resolved_events=[],
                    files_created=[],
                    files_updated=[],
                    model="test-model",
                )

        # Should return zero counts, not crash
        assert result["reviewed"] == 0
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_full_review_flow(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """End-to-end: LLM returns mixed verdicts, each processed correctly."""
        stale_path = _write_knowledge_file(
            temp_anima_dir, "stale.md", "# Stale\n\nOld problem description.",
        )
        valid_path = _write_knowledge_file(
            temp_anima_dir, "valid.md", "# Valid\n\nStill accurate.",
        )

        llm_verdicts = json.dumps([
            {
                "file": "stale.md",
                "verdict": "stale",
                "reason": "問題は解決済み",
                "correction": "# Updated\n\n問題は解決済み。",
            },
            {
                "file": "valid.md",
                "verdict": "valid",
                "reason": "内容は正確",
                "correction": None,
            },
        ])

        with patch.object(
            engine, "_select_review_candidates",
            return_value=[stale_path, valid_path],
        ):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = _make_llm_response(llm_verdicts)

                result = await engine._run_validity_review(
                    episode_entries=[{
                        "date": "2026-02-22", "time": "10:00",
                        "content": "IPC問題を解決した",
                    }],
                    resolved_events=[],
                    files_created=[],
                    files_updated=[],
                    model="test-model",
                )

        assert result["stale"] == 1
        assert result["valid"] == 1
        assert result["reviewed"] == 2

        # Stale file archived
        assert (temp_anima_dir / "archive" / "superseded" / "stale.md").exists()
        # Valid file still exists
        assert valid_path.exists()


# ── Pipeline Integration Test ──────────────────────────────


class TestDailyConsolidateIncludesValidityReview:
    """Test that daily_consolidate() includes validity review results."""

    @pytest.mark.asyncio
    async def test_daily_consolidate_includes_validity_review(
        self, engine: object, temp_anima_dir: Path,
    ) -> None:
        """daily_consolidate result dict includes validity_review key."""
        today = datetime.now().date()
        episode_file = engine.episodes_dir / f"{today}.md"
        episode_file.write_text(
            "## 10:00 — テスト\n\nテストエピソード。\n",
            encoding="utf-8",
        )

        llm_response = (
            "## 既存ファイル更新\n(なし)\n\n"
            "## 新規ファイル作成\n(なし)"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_llm_response(llm_response)

            with patch(
                "core.memory.consolidation.ConsolidationEngine."
                "_validate_consolidation",
                new_callable=AsyncMock,
            ) as mock_validate:
                mock_validate.return_value = llm_response

                with patch.object(
                    engine, "_run_validity_review",
                    new_callable=AsyncMock,
                    return_value={
                        "reviewed": 2, "stale": 1,
                        "needs_update": 0, "valid": 1, "errors": 0,
                    },
                ) as mock_review:
                    result = await engine.daily_consolidate(min_episodes=1)

        assert "validity_review" in result
        assert result["validity_review"]["reviewed"] == 2
        assert result["validity_review"]["stale"] == 1
        mock_review.assert_awaited_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
