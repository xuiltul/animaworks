from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────

SAMPLE_CONVERSATION = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1": [
            {"speaker": "Alice", "text": "I went to Paris last summer."},
            {"speaker": "Bob", "text": "That sounds amazing! How was the Eiffel Tower?"},
        ],
        "session_1_date_time": "July 15, 2023",
        "session_2": [
            {"speaker": "Alice", "text": "I'm planning a trip to London next month."},
            {"speaker": "Bob", "text": "You should visit the British Museum!"},
        ],
        "session_2_date_time": "October 3, 2023",
    },
    "qa": [
        {
            "question": "Where did Alice go last summer?",
            "answer": "Paris",
            "category": 4,
        },
        {
            "question": "What does Alice plan to visit?",
            "answer": "London",
            "category": 2,
        },
        {
            "question": "Where did Alice and Bob meet in Tokyo?",
            "category": 5,
            "adversarial_answer": "No information available.",
        },
    ],
}


def _make_dataset_file(tmp_path: Path) -> Path:
    p = tmp_path / "locomo_test.json"
    p.write_text(json.dumps([SAMPLE_CONVERSATION]), encoding="utf-8")
    return p


# ── Adapter mock tests ──────────


class _FakeRetrievalResult:
    def __init__(self, content: str, score: float):
        self.content = content
        self.score = score
        self.metadata = {"source": "test"}


class TestAdapterIngestAndRetrieve:
    """Test adapter with mocked RAG internals."""

    def _make_adapter(self):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = None
            adapter._episodes_dir = None
            adapter._vector_store = None
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = None
            adapter._bm25_index = None
            return adapter

    def test_retrieval_to_dicts(self):
        from core.memory.rag.retriever import RetrievalResult

        adapter = self._make_adapter()
        r1 = RetrievalResult(
            doc_id="1",
            content="context about paris",
            score=0.95,
            metadata={"source": "test"},
            source_scores={},
        )
        r2 = RetrievalResult(
            doc_id="2",
            content="context about london",
            score=0.80,
            metadata={"source": "test2"},
            source_scores={},
        )
        dicts = adapter._retrieval_to_dicts([r1, r2])
        assert len(dicts) == 2
        assert dicts[0]["content"] == "context about paris"
        assert dicts[0]["score"] == pytest.approx(0.95)
        assert "source" in dicts[0]["metadata"]

    def test_retrieval_to_dicts_ignores_non_results(self):
        adapter = self._make_adapter()
        dicts = adapter._retrieval_to_dicts(["not a result", 42])
        assert dicts == []

    def test_rrf_merge(self):
        adapter = self._make_adapter()
        vec = [
            {"content": "A", "score": 1.0, "metadata": {}},
            {"content": "B", "score": 0.8, "metadata": {}},
        ]
        bm25 = [
            {"content": "B", "score": 2.0, "metadata": {}},
            {"content": "C", "score": 1.5, "metadata": {}},
        ]
        merged = adapter._rrf_merge(vec, bm25, k=60)
        assert len(merged) == 3
        contents = [m["content"] for m in merged]
        assert "B" in contents[0]
        assert all(m["metadata"]["search_method"] == "rrf" for m in merged)

    def test_rrf_merge_empty(self):
        adapter = self._make_adapter()
        assert adapter._rrf_merge([], [], k=60) == []


class TestAdapterAnswer:
    """Test answer generation with mocked LLM."""

    def test_answer_calls_litellm(self):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = None
            adapter._episodes_dir = None
            adapter._vector_store = None
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = None
            adapter._bm25_index = None

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Paris"

        with patch("litellm.completion", return_value=mock_resp) as mock_comp:
            result = adapter.answer(
                "Where did Alice go?",
                [{"content": "Alice went to Paris"}],
                model="test-model",
            )
            assert result == "Paris"
            mock_comp.assert_called_once()

    def test_answer_empty_context(self):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = None
            adapter._episodes_dir = None
            adapter._vector_store = None
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = None
            adapter._bm25_index = None

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "No information available."

        with patch("litellm.completion", return_value=mock_resp):
            result = adapter.answer("Where?", [], model="test-model")
            assert "No information" in result


class TestAdapterCleanup:
    """Test cleanup and context manager behavior."""

    def test_cleanup_restores_env(self, tmp_path):
        import os

        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = str(tmp_path / "temp")
            (tmp_path / "temp").mkdir()
            adapter._previous_animaworks_data = "original_value"
            adapter._own_data_env = True
            adapter._anima_dir = None
            adapter._episodes_dir = None
            adapter._vector_store = None
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = None
            adapter._bm25_index = None

        os.environ["ANIMAWORKS_DATA_DIR"] = str(tmp_path / "temp")
        adapter.cleanup()
        assert os.environ.get("ANIMAWORKS_DATA_DIR") == "original_value"
        os.environ.pop("ANIMAWORKS_DATA_DIR", None)

    def test_context_manager(self, tmp_path):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = None
            adapter._episodes_dir = None
            adapter._vector_store = None
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = None
            adapter._bm25_index = None

        with patch.object(adapter, "cleanup") as mock_cleanup:
            with adapter:
                pass
            mock_cleanup.assert_called_once()


# ── Runner mock tests ──────────


class TestRunnerWithMockedAdapter:
    """Test runner with mocked adapter to verify orchestration logic."""

    def test_run_benchmark_with_mock(self, tmp_path):
        import argparse

        data_file = _make_dataset_file(tmp_path)

        mock_adapter = MagicMock()
        mock_adapter.__enter__ = MagicMock(return_value=mock_adapter)
        mock_adapter.__exit__ = MagicMock(return_value=False)
        mock_adapter.ingest_conversation.return_value = 5
        mock_adapter.retrieve.return_value = [
            {"content": "Alice went to Paris", "score": 0.9, "metadata": {}},
        ]
        mock_adapter.answer.return_value = "Paris"

        with patch(
            "benchmarks.locomo.runner.AnimaWorksLoCoMoAdapter",
            return_value=mock_adapter,
        ):
            from benchmarks.locomo.runner import run_benchmark

            args = argparse.Namespace(
                data=str(data_file),
                output=str(tmp_path / "results"),
                mode="vector",
                conversations=1,
                top_k=5,
                judge=False,
                judge_model="gpt-4o",
                answer_model="gpt-4o-mini",
                verbose=False,
            )
            results, errors = run_benchmark(args)

        assert "vector" in results
        assert results["vector"]["summary"]["overall_f1"] > 0
        result_files = list((tmp_path / "results").glob("*.json"))
        assert len(result_files) == 1

    def test_run_benchmark_with_judge(self, tmp_path):
        import argparse

        data_file = _make_dataset_file(tmp_path)

        mock_adapter = MagicMock()
        mock_adapter.__enter__ = MagicMock(return_value=mock_adapter)
        mock_adapter.__exit__ = MagicMock(return_value=False)
        mock_adapter.ingest_conversation.return_value = 3
        mock_adapter.retrieve.return_value = [{"content": "Paris trip", "score": 0.9, "metadata": {}}]
        mock_adapter.answer.return_value = "Paris"

        with (
            patch("benchmarks.locomo.runner.AnimaWorksLoCoMoAdapter", return_value=mock_adapter),
            patch(
                "benchmarks.locomo.runner.llm_judge_sync",
                return_value={"verdict": "correct", "score": 1.0},
            ),
        ):
            from benchmarks.locomo.runner import run_benchmark

            args = argparse.Namespace(
                data=str(data_file),
                output=str(tmp_path / "results_judge"),
                mode="vector",
                conversations=1,
                top_k=5,
                judge=True,
                judge_model="gpt-4o",
                answer_model="gpt-4o-mini",
                verbose=False,
            )
            results, errors = run_benchmark(args)

        assert "vector" in results
        judge_scores = [r["judge_score"] for r in results["vector"]["results"] if r["judge_score"] is not None]
        assert len(judge_scores) > 0

    def test_run_benchmark_all_modes(self, tmp_path):
        import argparse

        data_file = _make_dataset_file(tmp_path)

        mock_adapter = MagicMock()
        mock_adapter.__enter__ = MagicMock(return_value=mock_adapter)
        mock_adapter.__exit__ = MagicMock(return_value=False)
        mock_adapter.ingest_conversation.return_value = 3
        mock_adapter.retrieve.return_value = [{"content": "data", "score": 0.5, "metadata": {}}]
        mock_adapter.answer.return_value = "Paris"

        with patch(
            "benchmarks.locomo.runner.AnimaWorksLoCoMoAdapter",
            return_value=mock_adapter,
        ):
            from benchmarks.locomo.runner import run_benchmark

            args = argparse.Namespace(
                data=str(data_file),
                output=str(tmp_path / "results_all"),
                mode="all",
                conversations=1,
                top_k=5,
                judge=False,
                judge_model="gpt-4o",
                answer_model="gpt-4o-mini",
                verbose=False,
            )
            results, errors = run_benchmark(args)

        assert "vector" in results
        assert "vector_graph" in results
        assert "scope_all" in results
        result_files = list((tmp_path / "results_all").glob("*.json"))
        assert len(result_files) == 3

    def test_run_benchmark_error_recovery(self, tmp_path):
        import argparse

        data_file = _make_dataset_file(tmp_path)

        mock_adapter = MagicMock()
        mock_adapter.__enter__ = MagicMock(return_value=mock_adapter)
        mock_adapter.__exit__ = MagicMock(return_value=False)
        mock_adapter.ingest_conversation.return_value = 3
        mock_adapter.retrieve.side_effect = RuntimeError("test error")
        mock_adapter.answer.return_value = ""

        with patch(
            "benchmarks.locomo.runner.AnimaWorksLoCoMoAdapter",
            return_value=mock_adapter,
        ):
            from benchmarks.locomo.runner import run_benchmark

            args = argparse.Namespace(
                data=str(data_file),
                output=str(tmp_path / "results_err"),
                mode="vector",
                conversations=1,
                top_k=5,
                judge=False,
                judge_model="gpt-4o",
                answer_model="gpt-4o-mini",
                verbose=False,
            )
            results, errors = run_benchmark(args)

        assert errors > 0
        assert "vector" in results


# ── Metrics llm_judge_sync mock tests ──────────


class TestLlmJudgeSync:
    def test_correct_verdict(self):
        from benchmarks.locomo.metrics import llm_judge_sync

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "CORRECT"

        with patch("litellm.completion", return_value=mock_resp):
            result = llm_judge_sync("Q", "ref", "pred", model="test")
            assert result["verdict"] == "correct"
            assert result["score"] == 1.0

    def test_incorrect_verdict(self):
        from benchmarks.locomo.metrics import llm_judge_sync

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "INCORRECT"

        with patch("litellm.completion", return_value=mock_resp):
            result = llm_judge_sync("Q", "ref", "pred", model="test")
            assert result["verdict"] == "incorrect"
            assert result["score"] == 0.0

    def test_partial_verdict(self):
        from benchmarks.locomo.metrics import llm_judge_sync

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "PARTIALLY_CORRECT"

        with patch("litellm.completion", return_value=mock_resp):
            result = llm_judge_sync("Q", "ref", "pred", model="test")
            assert result["verdict"] == "partially_correct"
            assert result["score"] == 0.5

    def test_api_failure_returns_error(self):
        from benchmarks.locomo.metrics import llm_judge_sync

        with patch("litellm.completion", side_effect=RuntimeError("API down")):
            result = llm_judge_sync("Q", "ref", "pred", model="test")
            assert result["verdict"] == "error"
            assert result["score"] == 0.0


# ── BM25 cache tests ──────────


class TestBM25Cache:
    def test_cache_invalidated_by_ingest(self, tmp_path):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "scope_all"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = tmp_path
            adapter._episodes_dir = tmp_path / "episodes"
            adapter._episodes_dir.mkdir()
            adapter._vector_store = None
            adapter._indexer = MagicMock()
            adapter._indexer.index_file.return_value = 3
            adapter._retriever = None
            adapter._bm25_corpus = [("old data", {})]
            adapter._bm25_index = MagicMock()

        (adapter._episodes_dir / "test.md").write_text("## Session 1\nHello", encoding="utf-8")
        sample = {
            "sample_id": "test",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "text": "Hi"}],
                "session_1_date_time": "2023-01-01",
            },
        }
        adapter.ingest_conversation(sample)
        assert adapter._bm25_corpus is None
        assert adapter._bm25_index is None

    def test_cache_invalidated_by_reset(self, tmp_path):
        from benchmarks.locomo.adapter import AnimaWorksLoCoMoAdapter

        with (
            patch("benchmarks.locomo.adapter._ensure_rag_stack"),
            patch("benchmarks.locomo.adapter.AnimaWorksLoCoMoAdapter._init_isolated_rag"),
        ):
            adapter = AnimaWorksLoCoMoAdapter.__new__(AnimaWorksLoCoMoAdapter)
            adapter._search_mode = "vector"
            adapter._top_k = 5
            adapter._temp_dir = None
            adapter._previous_animaworks_data = None
            adapter._own_data_env = False
            adapter._anima_dir = tmp_path
            adapter._episodes_dir = tmp_path / "episodes"
            adapter._episodes_dir.mkdir()
            adapter._vector_store = MagicMock()
            adapter._vector_store.list_collections.return_value = []
            adapter._indexer = None
            adapter._retriever = None
            adapter._bm25_corpus = [("cached", {})]
            adapter._bm25_index = MagicMock()

        adapter.reset()
        assert adapter._bm25_corpus is None
        assert adapter._bm25_index is None
