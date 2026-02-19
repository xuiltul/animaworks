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
        assert "rin (開発)" in prompt
        assert "kotoha (広報)" in prompt
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
    """E2E: Orphan detection finds incomplete directories and notifies."""

    def test_detect_and_notify_orphan(self, data_dir, make_anima):
        """Full detection + notification flow for an orphan directory."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        # Create an orphan (no identity.md, has status.json with supervisor)
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
        assert orphans[0]["supervisor"] == "rin"

        # Notification marker should be created
        assert (orphan_dir / ".orphan_notified").exists()

        # Second run should skip the already-notified orphan
        orphans2 = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )
        assert orphans2 == []

    def test_orphan_notification_reaches_inbox(self, data_dir, make_anima):
        """Orphan notification should create a message file in the supervisor's inbox."""
        make_anima("sakura")
        make_anima("rin", supervisor="sakura")

        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()
        (orphan_dir / "status.json").write_text(
            json.dumps({"supervisor": "rin"}), encoding="utf-8"
        )

        from core.org_sync import detect_orphan_animas

        detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        inbox = data_dir / "shared" / "inbox" / "rin"
        messages = list(inbox.glob("*.json"))
        assert len(messages) >= 1

        # Verify message content
        msg_data = json.loads(messages[0].read_text(encoding="utf-8"))
        assert "rie" in msg_data.get("content", "")
        assert msg_data.get("type") == "system_alert"

    def test_orphan_without_supervisor_still_detected(self, data_dir, make_anima):
        """An orphan with no resolvable supervisor is still listed in results."""
        # Create orphan with no status.json and no config entry
        orphan_dir = data_dir / "animas" / "rie"
        orphan_dir.mkdir()

        from core.org_sync import detect_orphan_animas

        orphans = detect_orphan_animas(
            data_dir / "animas", data_dir / "shared", age_threshold_s=0
        )

        # Should still be detected even if notification fails
        assert len(orphans) == 1
        assert orphans[0]["name"] == "rie"
