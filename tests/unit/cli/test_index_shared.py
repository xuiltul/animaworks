"""Unit tests for CLI index --shared flag."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.commands.index_cmd import (
    _index_shared_collections,
    _is_anima_enabled,
    index_command,
    setup_index_command,
)

# ── _is_anima_enabled ─────────────────────────────────────


class TestIsAnimaEnabled:
    def test_enabled_when_no_status_file(self, tmp_path: Path) -> None:
        assert _is_anima_enabled(tmp_path) is True

    def test_enabled_explicitly(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text('{"enabled": true}')
        assert _is_anima_enabled(tmp_path) is True

    def test_disabled(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text('{"enabled": false}')
        assert _is_anima_enabled(tmp_path) is False

    def test_enabled_default_when_key_missing(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text('{"model": "claude-sonnet-4-6"}')
        assert _is_anima_enabled(tmp_path) is True

    def test_corrupted_json_treated_as_enabled(self, tmp_path: Path) -> None:
        (tmp_path / "status.json").write_text("{bad")
        assert _is_anima_enabled(tmp_path) is True


# ── setup_index_command (--shared flag registration) ──────


class TestSetupSharedFlag:
    def test_shared_flag_registered(self) -> None:
        """--shared flag is available in the argument parser."""
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        setup_index_command(subs)
        args = parser.parse_args(["index", "--shared"])
        assert args.shared is True

    def test_shared_defaults_false(self) -> None:
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        setup_index_command(subs)
        args = parser.parse_args(["index"])
        assert args.shared is False


# ── _index_shared_collections ─────────────────────────────


_PATCH_STORE = "core.memory.rag.store.ChromaVectorStore"
_PATCH_INDEXER = "core.memory.rag.MemoryIndexer"
_PATCH_VDBDIR = "core.paths.get_anima_vectordb_dir"


class TestIndexSharedCollections:
    @pytest.fixture
    def base_dir(self, tmp_path: Path) -> Path:
        """Set up a minimal base directory with common_knowledge."""
        d = tmp_path / "data"
        d.mkdir(exist_ok=True)
        ck = d / "common_knowledge"
        ck.mkdir(exist_ok=True)
        (ck / "ref.md").write_text("# Reference")
        return d

    @pytest.fixture
    def anima_dirs(self, base_dir: Path) -> list[Path]:
        animas = base_dir / "animas"
        alice = animas / "alice"
        alice.mkdir(parents=True, exist_ok=True)
        bob = animas / "bob"
        bob.mkdir(parents=True, exist_ok=True)
        return [alice, bob]

    def test_dry_run_does_not_write_meta(
        self,
        anima_dirs: list[Path],
        base_dir: Path,
        tmp_path: Path,
    ) -> None:
        with patch(_PATCH_STORE), patch(_PATCH_INDEXER), patch(_PATCH_VDBDIR, return_value=tmp_path / "vdb"):
            _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=True,
            )
        for d in anima_dirs:
            assert not (d / "index_meta.json").exists()

    def test_indexes_into_each_anima_db(
        self,
        anima_dirs: list[Path],
        base_dir: Path,
        tmp_path: Path,
    ) -> None:
        with patch(_PATCH_STORE), patch(_PATCH_INDEXER) as MockIdx, patch(_PATCH_VDBDIR, return_value=tmp_path / "vdb"):
            mock_indexer = MagicMock()
            mock_indexer.index_directory.return_value = 3
            MockIdx.return_value = mock_indexer

            total = _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=False,
            )

        assert total == 3 * len(anima_dirs)
        for d in anima_dirs:
            meta_path = d / "index_meta.json"
            assert meta_path.exists()
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            assert "shared_common_knowledge_hash" in data

    def test_skips_repair_locked_anima(
        self,
        anima_dirs: list[Path],
        base_dir: Path,
    ) -> None:
        """Shared indexing must not write into an anima under RAG repair."""
        with (
            patch("core.memory.rag.repair.is_repair_locked", side_effect=lambda name: name == "alice"),
            patch("core.memory.rag.singleton.get_vector_store") as mock_get_vs,
            patch(_PATCH_INDEXER) as MockIdx,
        ):
            mock_indexer = MagicMock()
            mock_indexer.index_directory.return_value = 3
            MockIdx.return_value = mock_indexer

            total = _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=False,
            )

        assert total == 3
        mock_get_vs.assert_called_once_with("bob")

    def test_skips_when_no_shared_dirs(self, tmp_path: Path) -> None:
        """Returns 0 when common_knowledge/ and common_skills/ don't exist."""
        base = tmp_path / "empty"
        base.mkdir()
        total = _index_shared_collections([], base, full=False, dry_run=False)
        assert total == 0

    def test_hash_skip_on_second_call(
        self,
        anima_dirs: list[Path],
        base_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Second call with unchanged files skips indexing."""
        with patch(_PATCH_STORE), patch(_PATCH_INDEXER) as MockIdx, patch(_PATCH_VDBDIR, return_value=tmp_path / "vdb"):
            mock_indexer = MagicMock()
            mock_indexer.index_directory.return_value = 3
            MockIdx.return_value = mock_indexer

            _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=False,
            )
            MockIdx.reset_mock()

            _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=False,
            )
            MockIdx.assert_not_called()

    def test_full_flag_forces_reindex(
        self,
        anima_dirs: list[Path],
        base_dir: Path,
        tmp_path: Path,
    ) -> None:
        """--full ignores stored hash and re-indexes."""
        with patch(_PATCH_STORE), patch(_PATCH_INDEXER) as MockIdx, patch(_PATCH_VDBDIR, return_value=tmp_path / "vdb"):
            mock_indexer = MagicMock()
            mock_indexer.index_directory.return_value = 2
            MockIdx.return_value = mock_indexer

            _index_shared_collections(
                anima_dirs,
                base_dir,
                full=False,
                dry_run=False,
            )
            MockIdx.reset_mock()
            mock_indexer.reset_mock()
            mock_indexer.index_directory.return_value = 2
            MockIdx.return_value = mock_indexer

            total = _index_shared_collections(
                anima_dirs,
                base_dir,
                full=True,
                dry_run=False,
            )
            assert total == 2 * len(anima_dirs)


def test_index_command_skips_repair_locked_anima(tmp_path: Path) -> None:
    """CLI indexing must not open a local vector store while repair lock is held."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "knowledge" / "note.md").write_text("# Note", encoding="utf-8")

    args = argparse.Namespace(anima="alice", full=False, shared=False, dry_run=False)

    with (
        patch("cli.commands.index_cmd.get_data_dir", return_value=tmp_path),
        patch("cli.commands.index_cmd._setup_server_delegation", return_value=False),
        patch("cli.commands.index_cmd._check_model_change", return_value="test-model"),
        patch("core.memory.rag.repair.is_repair_locked", return_value=True),
        patch("core.memory.rag.singleton.get_vector_store") as mock_get_vs,
        patch("core.memory.rag.MemoryIndexer") as mock_indexer,
    ):
        index_command(args)

    mock_get_vs.assert_not_called()
    mock_indexer.assert_not_called()


def test_index_command_rebuilds_longterm_bm25(tmp_path: Path) -> None:
    """CLI indexing rebuilds the persisted long-term BM25 index."""
    animas_dir = tmp_path / "animas"
    anima_dir = animas_dir / "alice"
    (anima_dir / "knowledge").mkdir(parents=True)
    (anima_dir / "knowledge" / "note.md").write_text("# Note", encoding="utf-8")

    args = argparse.Namespace(anima="alice", full=False, shared=False, dry_run=False)
    mock_store = MagicMock()

    with (
        patch("cli.commands.index_cmd.get_data_dir", return_value=tmp_path),
        patch("cli.commands.index_cmd._setup_server_delegation", return_value=False),
        patch("cli.commands.index_cmd._setup_offline_vector_worker_if_needed", return_value=None),
        patch("cli.commands.index_cmd._check_model_change", return_value="test-model"),
        patch("core.memory.rag.repair.is_repair_locked", return_value=False),
        patch("core.memory.rag.singleton.get_vector_store", return_value=mock_store),
        patch("core.memory.rag.MemoryIndexer") as mock_indexer_cls,
        patch("core.memory.bm25.rebuild_longterm_bm25_index") as mock_rebuild,
    ):
        mock_indexer = MagicMock()
        mock_indexer.index_directory.return_value = 1
        mock_indexer_cls.return_value = mock_indexer
        mock_rebuild.return_value = MagicMock(documents=1)

        index_command(args)

    mock_rebuild.assert_called_once_with(anima_dir)
