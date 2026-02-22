"""Tests for IPC streaming residual issues — setting_sources and per-anima ChromaDB.

Problem A: ClaudeAgentOptions.setting_sources=[] disables CLI hook loading
Problem B: get_vector_store(anima_name) returns per-anima ChromaVectorStore
"""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Problem A: setting_sources=[] ──────────────────────────────────


class TestSettingSourcesDisabled:
    """Verify ClaudeAgentOptions receives setting_sources=[] to block CLI hooks."""

    def test_execute_passes_empty_setting_sources(self, tmp_path: Path):
        """execute() ClaudeAgentOptions should include setting_sources=[]."""
        # We capture the kwargs passed to ClaudeAgentOptions
        captured_kwargs = {}
        original_init = None

        class FakeOptions:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        with patch.dict("sys.modules", {
            "claude_agent_sdk": MagicMock(
                ClaudeAgentOptions=FakeOptions,
                ClaudeSDKClient=MagicMock,
                HookMatcher=MagicMock,
                ResultMessage=MagicMock,
                SystemMessage=MagicMock,
                TextBlock=MagicMock,
                ToolResultBlock=MagicMock,
                ToolUseBlock=MagicMock,
                UserMessage=MagicMock,
                AssistantMessage=MagicMock,
            ),
        }):
            from core.execution.agent_sdk import AgentSDKExecutor

            executor = AgentSDKExecutor.__new__(AgentSDKExecutor)
            executor._anima_dir = tmp_path
            executor._model_config = MagicMock()
            executor._model_config.model = "claude-sonnet-4-20250514"
            executor._model_config.max_turns = 10
            executor._model_config.max_tokens = 4096
            executor._resolve_agent_sdk_model = MagicMock(return_value="claude-sonnet-4-20250514")
            executor._build_env = MagicMock(return_value={})
            executor._build_mcp_env = MagicMock(return_value={})

            # We need to trigger the ClaudeAgentOptions construction
            # The easiest way is to check the source code directly
            import inspect
            from core.execution.agent_sdk import AgentSDKExecutor as _Cls
            source = inspect.getsource(_Cls.execute)
            assert "setting_sources=[]" in source, (
                "execute() must pass setting_sources=[] to ClaudeAgentOptions"
            )

    def test_execute_streaming_passes_empty_setting_sources(self):
        """execute_streaming() ClaudeAgentOptions should include setting_sources=[]."""
        import inspect
        from core.execution.agent_sdk import AgentSDKExecutor
        source = inspect.getsource(AgentSDKExecutor.execute_streaming)
        assert "setting_sources=[]" in source, (
            "execute_streaming() must pass setting_sources=[] to ClaudeAgentOptions"
        )


# ── Problem B: Per-anima ChromaDB isolation ────────────────────────


@pytest.fixture(autouse=True)
def _reset_vector_stores():
    """Reset singleton stores before and after each test."""
    from core.memory.rag.singleton import _reset_for_testing
    _reset_for_testing()
    yield
    _reset_for_testing()


class TestGetAnimaVectordbDir:
    """Verify get_anima_vectordb_dir returns correct path."""

    def test_returns_correct_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        from core.paths import get_anima_vectordb_dir
        result = get_anima_vectordb_dir("yuki")
        assert result == tmp_path / "animas" / "yuki" / "vectordb"

    def test_different_animas_different_dirs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        from core.paths import get_anima_vectordb_dir
        d1 = get_anima_vectordb_dir("yuki")
        d2 = get_anima_vectordb_dir("sakura")
        assert d1 != d2
        assert "yuki" in str(d1)
        assert "sakura" in str(d2)


class TestPerAnimaVectorStore:
    """Verify get_vector_store(anima_name) creates isolated stores."""

    def test_different_animas_get_different_stores(self):
        """get_vector_store('a') and get_vector_store('b') return different instances."""
        store_a = MagicMock(name="store_a")
        store_b = MagicMock(name="store_b")
        call_count = 0

        def mock_ctor(persist_dir=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return store_a
            return store_b

        with patch("core.memory.rag.store.ChromaVectorStore", side_effect=mock_ctor):
            from core.memory.rag.singleton import get_vector_store
            a = get_vector_store("anima_a")
            b = get_vector_store("anima_b")
            assert a is not b
            assert a is store_a
            assert b is store_b

    def test_same_anima_returns_same_store(self):
        """get_vector_store('a') called twice returns the same instance."""
        mock_store = MagicMock()
        with patch("core.memory.rag.store.ChromaVectorStore", return_value=mock_store):
            from core.memory.rag.singleton import get_vector_store
            s1 = get_vector_store("yuki")
            s2 = get_vector_store("yuki")
            assert s1 is s2

    def test_none_uses_legacy_shared_store(self):
        """get_vector_store(None) uses default persist_dir (legacy shared)."""
        mock_store = MagicMock()
        with patch("core.memory.rag.store.ChromaVectorStore", return_value=mock_store) as mock_cls:
            from core.memory.rag.singleton import get_vector_store
            get_vector_store(None)
            mock_cls.assert_called_once_with(persist_dir=None)

    def test_anima_name_passes_correct_persist_dir(self, tmp_path: Path, monkeypatch):
        """get_vector_store('yuki') creates ChromaVectorStore with per-anima path."""
        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        mock_store = MagicMock()
        with patch("core.memory.rag.store.ChromaVectorStore", return_value=mock_store) as mock_cls:
            from core.memory.rag.singleton import get_vector_store
            get_vector_store("yuki")
            expected = tmp_path / "animas" / "yuki" / "vectordb"
            mock_cls.assert_called_once_with(persist_dir=expected)

    def test_reset_clears_all_stores(self):
        """_reset_for_testing() clears the per-anima store dict."""
        mock_store = MagicMock()
        with patch("core.memory.rag.store.ChromaVectorStore", return_value=mock_store) as mock_cls:
            from core.memory.rag.singleton import get_vector_store, _reset_for_testing
            get_vector_store("a")
            get_vector_store("b")
            _reset_for_testing()
            get_vector_store("a")
            # Should be called 3 times total (original a, original b, recreated a)
            assert mock_cls.call_count == 3

    def test_thread_safety_per_anima(self):
        """Multiple threads calling get_vector_store('x') get the same instance."""
        mock_store = MagicMock()
        results = []
        errors = []

        with patch("core.memory.rag.store.ChromaVectorStore", return_value=mock_store):
            from core.memory.rag.singleton import get_vector_store

            def worker():
                try:
                    store = get_vector_store("thread_test")
                    results.append(store)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors
        assert len(results) == 10
        assert all(r is mock_store for r in results)


class TestCallersSendAnimaName:
    """Verify all production callers pass anima_name to get_vector_store."""

    def test_forgetting_passes_anima_name(self):
        """ForgettingEngine._get_vector_store passes self.anima_name."""
        from core.memory.forgetting import ForgettingEngine
        engine = ForgettingEngine.__new__(ForgettingEngine)
        engine.anima_name = "yuki"

        with patch("core.memory.rag.singleton.get_vector_store") as mock_gvs:
            mock_gvs.return_value = MagicMock()
            engine._get_vector_store()
            mock_gvs.assert_called_once_with("yuki")

    def test_consolidation_passes_anima_name(self, tmp_path: Path):
        """ConsolidationEngine uses get_vector_store(self.anima_name) when no rag_store."""
        anima_dir = tmp_path / "animas" / "sakura"
        (anima_dir / "knowledge").mkdir(parents=True)
        (anima_dir / "knowledge" / "test.md").write_text("test", encoding="utf-8")

        from core.memory.consolidation import ConsolidationEngine
        engine = ConsolidationEngine(anima_dir, "sakura")

        with patch("core.memory.rag.MemoryIndexer") as MockIndexer, \
             patch("core.memory.rag.singleton.get_vector_store") as mock_gvs:
            mock_gvs.return_value = MagicMock()
            MockIndexer.return_value = MagicMock()
            engine._update_rag_index(["test.md"])
            mock_gvs.assert_called_once_with("sakura")

    def test_priming_passes_anima_name(self):
        """PrimingEngine uses get_vector_store(anima_name)."""
        import inspect
        from core.memory.priming import PrimingEngine
        source = inspect.getsource(PrimingEngine._get_or_create_retriever)
        assert "get_vector_store(anima_name)" in source

    def test_contradiction_passes_anima_name(self):
        """ContradictionDetector uses get_vector_store(self.anima_name)."""
        import inspect
        from core.memory.contradiction import ContradictionDetector
        source = inspect.getsource(ContradictionDetector._find_candidates_via_rag)
        assert "get_vector_store(self.anima_name)" in source

    def test_distillation_passes_anima_name(self):
        """ProceduralDistiller uses get_vector_store(self.anima_name)."""
        import inspect
        from core.memory.distillation import ProceduralDistiller
        source = inspect.getsource(ProceduralDistiller._check_rag_duplicate)
        assert "get_vector_store(self.anima_name)" in source
