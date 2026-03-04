# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for heartbeat guard, orphan detection, and org tree.

These tests exercise the full stack: build_system_prompt with real
MemoryManager, detect_orphan_animas with filesystem I/O, etc.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory.manager import MemoryManager
from core.prompt.builder import build_system_prompt


class TestOrgTreeE2E:
    """E2E: Top-level anima system prompt includes full org tree."""

    def test_top_level_sees_full_tree(self, data_dir, make_anima):
        """Top-level anima's system prompt should contain full org tree."""
        make_anima("sakura", speciality="経営")
        make_anima("rin", supervisor="sakura", speciality="開発")
        make_anima("kotoha", supervisor="sakura", speciality="広報")
        make_anima("aoi", supervisor="rin")

        anima_dir = data_dir / "animas" / "sakura"
        memory = MemoryManager(anima_dir)

        prompt = build_system_prompt(memory=memory)

        assert "組織全体" in prompt
        assert "rin (開発" in prompt
        assert "kotoha (広報" in prompt
        assert "aoi" in prompt
        assert "← あなた" in prompt

    def test_non_top_level_no_full_tree(self, data_dir, make_anima):
        """Non-top-level anima should not see the full org tree header."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        anima_dir = data_dir / "animas" / "rin"
        memory = MemoryManager(anima_dir)

        prompt = build_system_prompt(memory=memory)

        # Should not contain the "org tree" header
        assert "組織全体" not in prompt
        # But should mention their supervisor
        assert "sakura" in prompt

    def test_top_level_with_speciality_shows_in_prompt(self, data_dir, make_anima):
        """Top-level anima's own speciality should appear in the prompt."""
        make_anima("sakura", speciality="経営")
        make_anima("rin", supervisor="sakura", speciality="開発")

        anima_dir = data_dir / "animas" / "sakura"
        memory = MemoryManager(anima_dir)

        prompt = build_system_prompt(memory=memory)

        assert "経営" in prompt

    def test_single_anima_no_tree(self, data_dir, make_anima):
        """A single top-level anima without subordinates shouldn't show a complex tree."""
        make_anima("sakura")

        anima_dir = data_dir / "animas" / "sakura"
        memory = MemoryManager(anima_dir)

        prompt = build_system_prompt(memory=memory)

        # With only one anima, there should be no other names in org context
        # (the org tree section is only shown when len(all_animas) > 1)
        assert "組織全体" not in prompt


class TestOrphanDetectionE2E:
    """E2E: Orphan detection auto-cleans trivial orphans and logs non-trivial."""

    def test_nontrivial_orphan_archived(self, data_dir, make_anima):
        """Non-trivial orphan is archived and removed."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()
        (orphan_dir / "status.json").write_text(
            json.dumps({"supervisor": "rin"}), encoding="utf-8"
        )

        from core.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
        assert orphans[0]["action"] == "archived"
        assert not orphan_dir.exists()

        archive_root = data_dir / "archive" / "orphans"
        archives = list(archive_root.iterdir())
        assert len(archives) == 1
        assert (archives[0] / "state").is_dir()

        # Second run — orphan already gone, nothing to do
        orphans2 = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )
        assert orphans2 == []

    def test_no_inbox_message_sent(self, data_dir, make_anima):
        """Orphan detection must NOT send messages to any Anima's inbox."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "state").mkdir()
        (orphan_dir / "status.json").write_text(
            json.dumps({"supervisor": "rin"}), encoding="utf-8"
        )

        from core.org_sync import detect_orphan_animas

        detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        inbox = data_dir / "shared" / "inbox" / "rin"
        messages = list(inbox.glob("*.json")) if inbox.exists() else []
        assert len(messages) == 0

    def test_trivial_orphan_auto_removed(self, data_dir, make_anima):
        """A trivial orphan (empty dir) is automatically removed."""
        make_anima("sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()

        from core.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

    def test_auto_removed_orphan_cleans_config(self, data_dir, make_anima):
        """Auto-removal of orphan should also remove its config.json entry."""
        make_anima("sakura")

        from core.config.models import (
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )

        config = load_config(data_dir / "config.json")
        config.animas["ghost"] = AnimaModelConfig()
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        orphan_dir = data_dir / "animas" / "ghost"
        orphan_dir.mkdir()
        (orphan_dir / "vectordb").mkdir()

        from core.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        assert len(orphans) == 1
        assert orphans[0]["action"] == "auto_removed"
        assert not orphan_dir.exists()

        invalidate_cache()
        updated = load_config(data_dir / "config.json")
        assert "ghost" not in updated.animas
        assert "sakura" in updated.animas


class TestSyncOrgPruneE2E:
    """E2E: sync_org_structure prunes stale config entries."""

    def test_prune_stale_config_entry(self, data_dir, make_anima):
        """Config entries for deleted animas are pruned by sync."""
        make_anima("sakura")

        from core.config.models import (
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )

        config = load_config(data_dir / "config.json")
        config.animas["deleted"] = AnimaModelConfig(supervisor="sakura")
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        from core.org_sync import sync_org_structure

        sync_org_structure(data_dir / "animas", data_dir / "config.json")

        invalidate_cache()
        updated = load_config(data_dir / "config.json")
        assert "deleted" not in updated.animas
        assert "sakura" in updated.animas

    def test_existing_dir_without_identity_not_pruned(self, data_dir, make_anima):
        """Dirs that exist on disk (even without identity.md) keep their config entry."""
        make_anima("sakura")

        from core.config.models import (
            AnimaModelConfig,
            invalidate_cache,
            load_config,
            save_config,
        )

        orphan = data_dir / "animas" / "orphan"
        orphan.mkdir()

        config = load_config(data_dir / "config.json")
        config.animas["orphan"] = AnimaModelConfig()
        save_config(config, data_dir / "config.json")
        invalidate_cache()

        from core.org_sync import sync_org_structure

        sync_org_structure(data_dir / "animas", data_dir / "config.json")

        invalidate_cache()
        updated = load_config(data_dir / "config.json")
        assert "orphan" in updated.animas

    def test_full_lifecycle_create_orphan_detect_sync(self, data_dir, make_anima):
        """Full lifecycle: create anima, delete dir, detect orphan, sync prune."""
        make_anima("sakura")
        make_anima("temp")

        from core.config.models import invalidate_cache, load_config

        invalidate_cache()
        config_before = load_config(data_dir / "config.json")
        assert "temp" in config_before.animas

        import shutil
        shutil.rmtree(data_dir / "animas" / "temp")

        from core.org_sync import sync_org_structure

        sync_org_structure(data_dir / "animas", data_dir / "config.json")

        invalidate_cache()
        config_after = load_config(data_dir / "config.json")
        assert "temp" not in config_after.animas
        assert "sakura" in config_after.animas
