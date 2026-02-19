"""Unit tests for MemoryManager lazy RAG indexer initialization."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def anima_dir(tmp_path: Path) -> Path:
    """Create a minimal anima directory structure."""
    d = tmp_path / "animas" / "test-anima"
    d.mkdir(parents=True)
    (d / "identity.md").write_text("test identity")
    return d


class TestMemoryManagerLazyIndexer:
    """Verify that the RAG indexer is initialized lazily, not at construction."""

    @patch("core.memory.manager.get_common_skills_dir", return_value=Path("/tmp/skills"))
    @patch("core.memory.manager.get_company_dir", return_value=Path("/tmp/company"))
    @patch("core.memory.manager.get_shared_dir", return_value=Path("/tmp/shared"))
    def test_constructor_does_not_call_init_indexer(
        self, mock_shared, mock_company, mock_skills, anima_dir
    ):
        """MemoryManager() should NOT call _init_indexer during construction."""
        with patch(
            "core.memory.manager.MemoryManager._init_indexer"
        ) as mock_init:
            from core.memory.manager import MemoryManager

            mgr = MemoryManager(anima_dir)

            # _init_indexer should NOT have been called
            mock_init.assert_not_called()

            # Flags should indicate not yet initialized
            assert mgr._indexer is None
            assert mgr._indexer_initialized is False

    @patch("core.memory.manager.get_common_skills_dir", return_value=Path("/tmp/skills"))
    @patch("core.memory.manager.get_company_dir", return_value=Path("/tmp/company"))
    @patch("core.memory.manager.get_shared_dir", return_value=Path("/tmp/shared"))
    def test_get_indexer_triggers_init_once(
        self, mock_shared, mock_company, mock_skills, anima_dir
    ):
        """_get_indexer() should call _init_indexer on first access only."""
        from core.memory.manager import MemoryManager

        mgr = MemoryManager(anima_dir)
        assert mgr._indexer_initialized is False

        # First call triggers init
        mgr._get_indexer()
        assert mgr._indexer_initialized is True

        # Record state after first init
        indexer_after_first = mgr._indexer

        # Second call does NOT re-initialize
        mgr._get_indexer()
        assert mgr._indexer is indexer_after_first

    @patch("core.memory.manager.get_common_skills_dir", return_value=Path("/tmp/skills"))
    @patch("core.memory.manager.get_company_dir", return_value=Path("/tmp/company"))
    @patch("core.memory.manager.get_shared_dir", return_value=Path("/tmp/shared"))
    def test_get_indexer_returns_none_when_deps_missing(
        self, mock_shared, mock_company, mock_skills, anima_dir
    ):
        """When RAG dependencies are missing, _get_indexer() returns None."""
        from core.memory.manager import MemoryManager

        # Patch the import inside _init_indexer to fail
        with patch.dict("sys.modules", {"core.memory.rag": None}):
            mgr = MemoryManager(anima_dir)
            result = mgr._get_indexer()
            assert result is None
            assert mgr._indexer_initialized is True
