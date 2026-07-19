from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from cli.commands.index_cmd import _index_shared_collections
from core.memory.rag.indexer import IndexDirectoryResult


def _anima(base: Path, name: str, company: str | None) -> Path:
    directory = base / "animas" / name
    directory.mkdir(parents=True)
    payload = {"company": company} if company is not None else {}
    (directory / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _knowledge(base: Path, company: str) -> Path:
    directory = base / "companies" / company / "knowledge"
    directory.mkdir(parents=True)
    (directory / "guide.md").write_text(f"# {company}", encoding="utf-8")
    return directory


def test_shared_index_targets_each_animas_own_company_only(tmp_path: Path) -> None:
    alice = _anima(tmp_path, "alice", "alpha")
    bob = _anima(tmp_path, "bob", "beta")
    legacy = _anima(tmp_path, "legacy", None)
    alpha = _knowledge(tmp_path, "alpha")
    beta = _knowledge(tmp_path, "beta")
    _knowledge(tmp_path, "other")

    stores = {name: MagicMock(name=f"store-{name}") for name in ("alice", "bob", "legacy")}
    indexed: list[tuple[MagicMock, Path, str]] = []

    def make_indexer(store, **_kwargs):
        indexer = MagicMock()
        indexer.index_directory.side_effect = lambda directory, label, force=False: (
            indexed.append((store, directory, label))
            or IndexDirectoryResult(chunks_indexed=1, files_indexed=1)
        )
        return indexer

    with (
        patch("core.memory.rag.repair.is_repair_locked", return_value=False),
        patch("core.memory.rag.singleton.get_vector_store", side_effect=lambda name: stores[name]),
        patch("core.memory.rag.MemoryIndexer", side_effect=make_indexer),
    ):
        total = _index_shared_collections([alice, bob, legacy], tmp_path, full=False, dry_run=False)

    assert total == 2
    assert (stores["alice"], alpha, "common_knowledge") in indexed
    assert (stores["bob"], beta, "common_knowledge") in indexed
    assert not any(store is stores["legacy"] for store, _directory, _label in indexed)
    assert not any("other" in directory.parts for _store, directory, _label in indexed)


def test_company_shared_index_dry_run_does_not_mutate(tmp_path: Path) -> None:
    alice = _anima(tmp_path, "alice", "alpha")
    _knowledge(tmp_path, "alpha")
    store = MagicMock()

    with (
        patch("core.memory.rag.repair.is_repair_locked", return_value=False),
        patch("core.memory.rag.singleton.get_vector_store", return_value=store),
        patch("core.memory.rag.MemoryIndexer") as indexer,
    ):
        assert _index_shared_collections([alice], tmp_path, full=False, dry_run=True) == 0

    store.delete_collection.assert_not_called()
    indexer.assert_not_called()
    assert not (alice / "index_meta.json").exists()

