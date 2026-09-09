from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.i18n import t
from core.memory.rag import repair_rebuild, repair_snapshot
from core.memory.rag.indexer import MemoryIndexer
from core.supervisor.memory_service import MemoryService
from core.time_utils import ensure_aware


@pytest.fixture
def sources(data_dir: Path, monkeypatch):
    anima = data_dir / "animas" / "alice"
    (anima / "knowledge").mkdir(parents=True)
    (anima / "state").mkdir()
    (anima / "status.json").write_text('{"company":"example"}')
    (anima / "knowledge" / "note.md").write_text("# Synthetic note\n\n" + "Evidence from a fixture. " * 8)
    (anima / "knowledge" / "ignored.md").write_text("A deliberately excluded fixture." * 4)
    (data_dir / ".ragignore").write_text(str(anima / "knowledge" / "ignored.md") + "\n")
    # Corrupt registry recovery used to write/rename the original source.
    (anima / "state" / "entity_registry.json").write_text("{broken")
    (anima / "state" / "rag_upsert_failures.json").write_text('{"quarantined": []}')
    for directory in (data_dir / "common_knowledge", data_dir / "companies" / "example" / "knowledge"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "shared.md").write_text("# Shared fixture\n\n" + "Shared synthetic evidence. " * 6)
    (anima / "index_meta.json").write_text('{"old": {"hash":"old"}}')
    (anima / "shared_index_meta.json").write_text('{"shared_common_knowledge_hash":"old-shared"}')
    (data_dir / "index_meta.json").write_text('{"global-old": {"hash":"global"}}')
    live = anima / "vectordb"
    live.mkdir()
    (live / "old.marker").write_text("old database")
    monkeypatch.setattr(MemoryIndexer, "_init_embedding_model", lambda self: None)
    monkeypatch.setattr(
        "core.memory.rag.singleton.generate_embeddings", lambda texts, **kwargs: [[0.1, 0.2] for _ in texts]
    )
    return anima


def _source_bytes(anima: Path) -> dict[Path, bytes]:
    base = anima.parent.parent
    paths = [
        *anima.glob("knowledge/*.md"),
        *anima.glob("state/*"),
        anima / "status.json",
        anima / "index_meta.json",
        anima / "shared_index_meta.json",
        base / "index_meta.json",
        base / ".ragignore",
        *base.glob("common_knowledge/*.md"),
        *base.glob("companies/*/knowledge/*.md"),
    ]
    # Locks are coordination artifacts, not the data restored by rollback.
    return {path: path.read_bytes() for path in paths if path.is_file() and path.suffix != ".lock"}


def _build(anima: Path):
    staging = anima / "vectordb.staging-test"
    result = repair_rebuild.build_staging_vectordb(
        anima.name,
        include_shared=True,
        anima_dir=anima,
        staging=staging,
    )
    return staging, *result


def test_staging_build_keeps_inputs_metadata_and_live_db_unchanged(sources):
    before = _source_bytes(sources)
    staging, chunks, hashes = _build(sources)
    assert chunks == 3
    assert _source_bytes(sources) == before
    assert (sources / "vectordb" / "old.marker").read_text() == "old database"
    assert not list(sources.glob(".rag-rebuild-inputs-*"))
    metadata = json.loads((staging / ".rebuild" / "index_meta.json").read_text())
    assert set(metadata) == {"knowledge/note.md"}
    assert hashes["shared_common_knowledge_hash"]
    assert hashes["shared_company_knowledge_hash"]
    repair_snapshot.validate_rebuild_sources(staging, sources)


def test_snapshot_indexer_preserves_ids_timestamps_and_absolute_exclusions(sources):
    original = sources / "knowledge" / "note.md"
    stat = original.stat()
    with repair_snapshot.snapshot_inputs(sources, include_shared=True) as snapshot:
        store = MagicMock()
        store.create_collection.return_value = True
        store.upsert.return_value = True
        store.get_by_metadata.return_value = []
        indexer = MemoryIndexer(
            store, "alice", snapshot.anima_dir, source_data_dir=snapshot.data_dir, source_file_stats=snapshot.file_stats
        )
        copied = snapshot.anima_dir / "knowledge" / "note.md"
        assert indexer.index_file(copied, "knowledge", force=True) == 1
        document = store.upsert.call_args.args[1][0]
        assert document.metadata["source_file"] == "knowledge/note.md"
        assert "rag-rebuild-inputs" not in document.id
        assert document.metadata["created_at"] == ensure_aware(datetime.fromtimestamp(stat.st_ctime)).isoformat()
        assert document.metadata["source_mtime_ns"] == stat.st_mtime_ns
        store.upsert.reset_mock()
        assert indexer.index_file(snapshot.anima_dir / "knowledge" / "ignored.md", "knowledge", force=True) == 0
        store.upsert.assert_not_called()


def test_failed_build_does_not_restore_over_a_concurrent_source_writer(sources, monkeypatch):
    metadata_before = (sources / "index_meta.json").read_bytes()
    changed = sources / "knowledge" / "note.md"

    def encode(texts, **kwargs):
        changed.write_text("Concurrent writer's new source survives failed repair.")
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("core.memory.rag.singleton.generate_embeddings", encode)
    with pytest.raises(RuntimeError, match=re.escape(t("rag.rebuild_input_changed"))):
        _build(sources)
    assert changed.read_text() == "Concurrent writer's new source survives failed repair."
    assert (sources / "index_meta.json").read_bytes() == metadata_before
    assert (sources / "vectordb" / "old.marker").exists()
    assert not list(sources.glob("vectordb.staging-*"))
    assert not list(sources.glob(".rag-rebuild-inputs-*"))


@pytest.mark.parametrize("change", ["add", "delete", "modify", "policy", "config"])
async def test_source_change_after_build_aborts_before_owner_closes_live_db(sources, change):
    staging, chunks, hashes = _build(sources)
    path = sources / "knowledge" / "note.md"
    if change == "add":
        (path.parent / "new.md").write_text("new source")
    elif change == "delete":
        path.unlink()
    elif change == "modify":
        path.write_text("updated source")
    elif change == "policy":
        (sources.parent.parent / ".ragignore").write_text("*.md")
    else:
        config = sources.parent.parent / "config.json"
        config.write_text(config.read_text() + "\n")
    before = _source_bytes(sources)
    original = MagicMock()
    service = MemoryService("alice", sources, opener=lambda: original)
    service._build_staging_subprocess = AsyncMock(return_value=(staging, chunks, hashes))
    await service.start()
    try:
        with pytest.raises(RuntimeError, match=re.escape(t("rag.rebuild_input_changed"))):
            await service.repair(include_shared=True)
        original.close.assert_not_called()
        assert _source_bytes(sources) == before
        assert (sources / "vectordb" / "old.marker").exists()
    finally:
        await service.close()


@pytest.mark.parametrize("owner", ["phase3", "legacy"])
async def test_publication_failure_restores_db_index_shared_and_bm25_metadata(sources, monkeypatch, owner):
    from core.memory.bm25 import longterm_bm25_delta_path, longterm_bm25_dirty_path, longterm_bm25_index_path

    for path in (
        longterm_bm25_index_path(sources),
        longterm_bm25_dirty_path(sources),
        longterm_bm25_delta_path(sources),
    ):
        path.write_text("old-" + path.name)
    before = _source_bytes(sources)
    publish = repair_snapshot.publish_rebuild_metadata

    def fail_after_publish(anima_dir):
        publish(anima_dir)
        assert json.loads((anima_dir / "index_meta.json").read_text()) != {"old": {"hash": "old"}}
        (anima_dir / "shared_index_meta.json").write_text('{"shared_common_knowledge_hash":"new"}')
        raise OSError("injected metadata publication failure")

    monkeypatch.setattr(repair_snapshot, "publish_rebuild_metadata", fail_after_publish)
    if owner == "legacy":
        monkeypatch.setattr(repair_rebuild, "_has_active_repair_fence", lambda *args, **kwargs: True)
        monkeypatch.setattr(repair_rebuild, "reset_worker_vector_store", lambda *args: True)
        monkeypatch.setattr(repair_rebuild, "verify_worker_vector_store", lambda *args, **kwargs: True)
        monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", lambda *args: None)
        with pytest.raises(OSError, match="publication failure"):
            repair_rebuild.atomic_rebuild_vectordb("alice", include_shared=True, anima_dir=sources)
    else:
        staging, chunks, hashes = _build(sources)
        reopened = MagicMock()
        reopened.verify_rebuilt_data.return_value = {"chunks": chunks}
        service = MemoryService("alice", sources, opener=MagicMock(side_effect=[MagicMock(), reopened, MagicMock()]))
        service._build_staging_subprocess = AsyncMock(return_value=(staging, chunks, hashes))
        await service.start()
        try:
            with pytest.raises(OSError, match="publication failure"):
                await service.repair(include_shared=True)
        finally:
            await service.close()
    assert _source_bytes(sources) == before
    assert (sources / "vectordb" / "old.marker").exists()
    assert list((sources / "archive").glob("vectordb-rebuild-failed-*/.rebuild/index_meta.json"))


def test_snapshot_rejects_symlink_inputs(sources, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside the selected input scope")
    (sources / "knowledge" / "link.md").symlink_to(outside)
    message = t("rag.rebuild_symlink_input", path=sources / "knowledge" / "link.md")
    with pytest.raises(ValueError, match=re.escape(message)):
        _build(sources)
    assert (sources / "vectordb" / "old.marker").exists()


def test_normal_indexer_never_builds_a_snapshot(sources, monkeypatch):
    snapshot = MagicMock(side_effect=AssertionError("normal indexing must not snapshot"))
    monkeypatch.setattr(repair_snapshot, "snapshot_inputs", snapshot)
    store = MagicMock()
    store.get_by_metadata.return_value = []
    indexer = MemoryIndexer(store, "alice", sources)
    assert indexer.index_file(sources / "knowledge" / "note.md", "knowledge", force=True) == 1
    snapshot.assert_not_called()


def test_source_mutation_during_copy_aborts_without_metadata_writes(sources, monkeypatch):
    copy = repair_snapshot.shutil.copy2
    original = sources / "knowledge" / "note.md"
    metadata = (sources / "index_meta.json").read_bytes()

    def racing_copy(source, target, *args, **kwargs):
        result = copy(source, target, *args, **kwargs)
        if source == original:
            original.write_text("A concurrent user's edit must survive.")
        return result

    monkeypatch.setattr(repair_snapshot.shutil, "copy2", racing_copy)
    with pytest.raises(RuntimeError, match=re.escape(t("rag.rebuild_input_changed"))):
        _build(sources)
    assert original.read_text() == "A concurrent user's edit must survive."
    assert (sources / "index_meta.json").read_bytes() == metadata
    assert (sources / "vectordb" / "old.marker").exists()
    assert not list(sources.glob(".rag-rebuild-inputs-*"))


async def test_success_publishes_personal_metadata_and_migrates_legacy_shared_keys(sources):
    (sources / "shared_index_meta.json").unlink()
    (sources / "index_meta.json").write_text('{"old": {}, "shared_company_name": "example"}')
    staging, chunks, hashes = _build(sources)
    rebuilt = json.loads((staging / ".rebuild" / "index_meta.json").read_text())
    reopened = MagicMock()
    reopened.verify_rebuilt_data.return_value = {"chunks": chunks}
    service = MemoryService("alice", sources, opener=MagicMock(side_effect=[MagicMock(), reopened]))
    service._build_staging_subprocess = AsyncMock(return_value=(staging, chunks, hashes))
    await service.start()
    try:
        result = await service.repair(include_shared=True)
        assert result["ok"] is True
        assert json.loads((sources / "index_meta.json").read_text()) == rebuilt
        shared = json.loads((sources / "shared_index_meta.json").read_text())
        assert shared["shared_company_name"] == "example"
        assert shared["shared_common_knowledge_hash"] == hashes["shared_common_knowledge_hash"]
        assert (Path(result["archive_path"]) / "old.marker").exists()
        assert not list((sources / "state").glob(".rag-repair-metadata-*"))
    finally:
        await service.close()


async def test_incomplete_rollback_retains_metadata_backup_for_recovery(sources, monkeypatch):
    before = (sources / "index_meta.json").read_bytes()
    staging, chunks, hashes = _build(sources)
    reopened = MagicMock()
    reopened.verify_rebuilt_data.side_effect = RuntimeError("verification failed")
    service = MemoryService("alice", sources, opener=MagicMock(side_effect=[MagicMock(), reopened]))
    service._build_staging_subprocess = AsyncMock(return_value=(staging, chunks, hashes))
    monkeypatch.setattr(service, "_rollback_sync", MagicMock(side_effect=OSError("rollback disk error")))
    await service.start()
    try:
        with pytest.raises(RuntimeError, match="rollback incomplete"):
            await service.repair(include_shared=True)
        backups = list((sources / "state").glob(".rag-repair-metadata-*"))
        assert len(backups) == 1
        assert (backups[0] / "index_meta.json").read_bytes() == before
        assert list((sources / "archive").glob("vectordb-corrupt-*/old.marker"))
        assert service._store is None
    finally:
        await service.close()


@pytest.mark.parametrize("owner", ["legacy", "phase3"])
@pytest.mark.parametrize("failed_move", ["retire_original", "install_staging"])
async def test_rename_failure_keeps_original_db_and_metadata(sources, monkeypatch, owner, failed_move):
    import shutil

    before = _source_bytes(sources)
    live = sources / "vectordb"
    move = shutil.move
    rejected = []

    def fail_selected_move(source, destination, *args, **kwargs):
        source_path = Path(source)
        selected = (
            source_path == live
            if failed_move == "retire_original"
            else source_path.name.startswith("vectordb.staging-")
        )
        if selected and not rejected:
            rejected.append(source_path)
            raise PermissionError("injected rename denied")
        return move(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", fail_selected_move)
    if owner == "legacy":
        monkeypatch.setattr(repair_rebuild, "_has_active_repair_fence", lambda *args, **kwargs: True)
        monkeypatch.setattr(repair_rebuild, "reset_worker_vector_store", lambda *args: True)
        monkeypatch.setattr("core.memory.rag.singleton.reset_vector_store", lambda *args: None)
        verify = MagicMock(return_value=True)
        monkeypatch.setattr(repair_rebuild, "verify_worker_vector_store", verify)
        with pytest.raises(PermissionError, match="rename denied"):
            repair_rebuild.atomic_rebuild_vectordb("alice", include_shared=True, anima_dir=sources)
        verify.assert_not_called()
    else:
        staging, chunks, hashes = _build(sources)
        original, reopened = MagicMock(), MagicMock()
        service = MemoryService("alice", sources, opener=MagicMock(side_effect=[original, reopened]))
        service._build_staging_subprocess = AsyncMock(return_value=(staging, chunks, hashes))
        await service.start()
        try:
            with pytest.raises(PermissionError, match="rename denied"):
                await service.repair(include_shared=True)
            original.close.assert_called_once()
            assert service._store is reopened
            reopened.verify_rebuilt_data.assert_not_called()
        finally:
            await service.close()
    assert len(rejected) == 1
    assert (live / "old.marker").read_text() == "old database"
    assert _source_bytes(sources) == before
    assert not list((sources / "archive").glob("*/old.marker"))
    assert not list(sources.glob("vectordb.staging-*"))
    assert not list((sources / "state").glob(".rag-repair-metadata-*"))
