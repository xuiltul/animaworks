# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for orphan prevention, detection, and cleanup.

Covers:
- detect_orphan_animas auto-removal and logging
- _auto_cleanup_orphan config.json cleanup
- create_blank / create_from_template rollback
- ChromaVectorStore mkdir guard
- sync_org_structure config pruning
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from core.memory.rag.direct_access import OWNER_CAPABILITY


class TestDetectOrphanAnimas:
    """Tests for detect_orphan_animas."""

    def test_no_orphans_when_all_valid(self, data_dir, make_anima):
        """Directories with valid IDENTITY.md are not flagged as orphans."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        from core.org.org_sync import detect_orphan_animas

        animas_dir = data_dir / "animas"
        shared_dir = data_dir / "shared"
        orphans = detect_orphan_animas(animas_dir, shared_dir, age_threshold_s=0)
        assert orphans == []

    def test_nontrivial_orphan_archived(self, data_dir, make_anima):
        """A directory without IDENTITY.md but with state/ is archived and removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
        assert orphans[0]["action"] == "archived"
        assert not orphan_dir.exists()

        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        assert archives[0].name.startswith("rie_")

    def test_trivial_orphan_auto_removed_empty(self, data_dir, make_anima):
        """An empty orphan directory is trivially removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_trivial_orphan_auto_removed_vectordb_only(self, data_dir, make_anima):
        """An orphan directory with only vectordb/ is trivially removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "vectordb").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_trivial_orphan_auto_removed_shared_index_meta_only(self, data_dir, make_anima):
        """Shared metadata and its flock sidecar remain trivial leftovers."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "shared_index_meta.json").write_text("{}\n", encoding="utf-8")
        (orphan_dir / "shared_index_meta.json.lock").touch()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert orphans == [{"name": "rie", "action": "auto_removed"}]
        assert not orphan_dir.exists()

    def test_trivial_orphan_with_empty_identity(self, data_dir, make_anima):
        """An orphan with empty IDENTITY.md and no other content is auto-removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "IDENTITY.md").write_text("", encoding="utf-8")

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_trivial_orphan_with_undefined_identity(self, data_dir, make_anima):
        """An orphan with IDENTITY.md containing '未定義' is auto-removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "IDENTITY.md").write_text("未定義", encoding="utf-8")

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_skips_young_directories(self, data_dir):
        """Directories younger than age_threshold_s are skipped."""
        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared")
        assert orphans == []

    def test_archives_nontrivial_even_with_marker(self, data_dir, make_anima):
        """Non-trivial orphans with .orphan_notified marker are now archived."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()
        (orphan_dir / ".orphan_notified").write_text("logged")

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "archived"
        assert not orphan_dir.exists()

    def test_trivial_orphan_with_marker_still_removed(self, data_dir, make_anima):
        """Trivial orphans are auto-removed even if they have .orphan_notified."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / ".orphan_notified").write_text("already notified")
        (orphan_dir / "vectordb").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_skips_hidden_directories(self, data_dir):
        """Directories starting with '.' or '_' are skipped."""
        (data_dir / "animas" / ".hidden").mkdir()
        (data_dir / "animas" / "_internal").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert orphans == []

    def test_no_notification_sent_to_anima(self, data_dir, make_anima):
        """Orphan detection does NOT send messages to any Anima."""
        make_anima("rin", supervisor="sakura")
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()

        from core.org.org_sync import detect_orphan_animas

        detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)

        rin_inbox = data_dir / "shared" / "inbox" / "rin"
        sakura_inbox = data_dir / "shared" / "inbox" / "sakura"
        rin_msgs = list(rin_inbox.glob("*.json")) if rin_inbox.exists() else []
        sakura_msgs = list(sakura_inbox.glob("*.json")) if sakura_inbox.exists() else []
        assert len(rin_msgs) == 0
        assert len(sakura_msgs) == 0

    def test_nontrivial_orphan_archived_with_episodes(self, data_dir, make_anima):
        """Non-trivial orphan with episodes is archived (directory is removed)."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "episodes").mkdir()

        from core.org.org_sync import detect_orphan_animas

        detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)

        assert not orphan_dir.exists()
        archive_root = data_dir / "archive" / "orphans"
        assert archive_root.is_dir()
        archives = list(archive_root.iterdir())
        assert len(archives) == 1

    def test_empty_animas_dir(self, data_dir):
        """No orphans reported for empty animas directory."""
        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert orphans == []

    def test_nonexistent_animas_dir(self, data_dir):
        """Returns empty list for nonexistent directory."""
        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "nonexistent", data_dir / "shared", age_threshold_s=0)
        assert orphans == []


class TestAutoCleanupOrphanConfig:
    """Tests for _auto_cleanup_orphan config.json cleanup."""

    def test_auto_cleanup_removes_config_entry(self, data_dir, make_anima):
        """Auto-cleanup should unregister the orphan from config.json."""
        make_anima("sakura")

        from core.config.models import AnimaModelConfig, invalidate_cache, load_config, save_config

        config = load_config(data_dir / "config.json")
        config.animas["rie"] = AnimaModelConfig()
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "vectordb").mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

        invalidate_cache()
        updated = load_config(data_dir / "config.json")
        assert "rie" not in updated.animas

    def test_auto_cleanup_handles_missing_config_entry(self, data_dir, make_anima):
        """Auto-cleanup should not fail if config has no entry for the orphan."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()

        from core.org.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(data_dir / "animas", data_dir / "shared", age_threshold_s=0)
        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"


class TestCreateBlankRollback:
    """Tests for create_blank rollback on failure."""

    def test_rollback_on_failure(self, data_dir, monkeypatch):
        """create_blank should remove the directory if _ensure_runtime_subdirs fails."""
        from core.anima import factory as anima_factory

        animas_dir = data_dir / "animas"

        def _failing_subdirs(pd):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(anima_factory, "_ensure_runtime_subdirs", _failing_subdirs)

        with pytest.raises(RuntimeError, match="simulated failure"):
            anima_factory.create_blank(animas_dir, "test-fail")

        assert not (animas_dir / "test-fail").exists()

    def test_successful_create_blank(self, data_dir):
        """create_blank creates an anima directory on success."""
        from core.anima import factory as anima_factory

        animas_dir = data_dir / "animas"
        anima_dir = anima_factory.create_blank(animas_dir, "test-ok")

        assert anima_dir.exists()
        assert (anima_dir / "episodes").is_dir()
        assert (anima_dir / "knowledge").is_dir()
        assert (anima_dir / "state").is_dir()


class TestCreateFromTemplateRollback:
    """Tests for create_from_template rollback on failure."""

    def test_rollback_on_failure(self, data_dir, monkeypatch):
        """create_from_template should remove the directory on post-copy failure."""
        from core.anima import factory as anima_factory

        animas_dir = data_dir / "animas"

        def _failing_subdirs(pd):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(anima_factory, "_ensure_runtime_subdirs", _failing_subdirs)

        template_dir = anima_factory.ANIMA_TEMPLATES_DIR
        templates = (
            [d.name for d in template_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
            if template_dir.exists()
            else []
        )

        if not templates:
            pytest.skip("No non-blank templates available")

        with pytest.raises(RuntimeError, match="simulated failure"):
            anima_factory.create_from_template(
                animas_dir,
                templates[0],
                anima_name="test-fail",
            )

        assert not (animas_dir / "test-fail").exists()


class TestChromaVectorStoreMkdirGuard:
    """Tests for ChromaVectorStore mkdir guard against orphan creation."""

    def test_parent_exists_creates_vectordb(self, tmp_path):
        """When parent dir exists, vectordb/ is created normally."""
        import sys

        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True)
        persist_dir = anima_dir / "vectordb"

        mock_chromadb = MagicMock()
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            from core.memory.rag.store import ChromaVectorStore

            ChromaVectorStore(persist_dir=persist_dir, allow_direct=OWNER_CAPABILITY)

        assert persist_dir.exists()

    def test_parent_missing_creates_with_warning(self, tmp_path, caplog):
        """When parent dir is missing, dir is still created but with a warning."""
        import sys

        persist_dir = tmp_path / "animas" / "ghost" / "vectordb"
        assert not persist_dir.parent.exists()

        mock_chromadb = MagicMock()
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}), caplog.at_level(logging.WARNING):
            from core.memory.rag.store import ChromaVectorStore

            ChromaVectorStore(persist_dir=persist_dir, allow_direct=OWNER_CAPABILITY)

        assert persist_dir.exists()
        assert "Parent directory does not exist" in caplog.text
