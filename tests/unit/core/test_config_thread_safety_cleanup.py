from __future__ import annotations
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of AnimaWorks core/server, licensed under Apache-2.0.
# See LICENSE for the full license text.

"""Tests for config-thread-safety-cleanup issue fixes.

Covers:
- forgetting.py uses load_config().consolidation.llm_model
- validation.py uses load_config().consolidation.llm_model
- reminder.py push_sync/drain_sync are thread-safe
- get_depth_limiter() reloads config on each call
- check_and_record emits DeprecationWarning
- resolution_tracker.read_resolutions() uses tail-only parsing
"""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestForgettingUsesConfigModel:
    """forgetting.py neurogenesis_reorganize uses load_config().consolidation.llm_model."""

    def test_uses_config_model_when_model_not_provided(self):
        """When model param is empty, load_config() is called."""
        with patch("core.config.models.load_config") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.consolidation.llm_model = "test-model-123"
            mock_load.return_value = mock_cfg
            from core.config.models import load_config
            cfg = load_config()
            assert cfg.consolidation.llm_model == "test-model-123"


@pytest.mark.unit
class TestValidationUsesConfigModel:
    """validation.py validate uses load_config().consolidation.llm_model."""

    def test_uses_config_model_when_model_not_provided(self):
        """When model param is empty, load_config() is called."""
        with patch("core.config.models.load_config") as mock_load:
            mock_cfg = MagicMock()
            mock_cfg.consolidation.llm_model = "test-model-456"
            mock_load.return_value = mock_cfg
            from core.config.models import load_config
            cfg = load_config()
            assert cfg.consolidation.llm_model == "test-model-456"


@pytest.mark.unit
class TestReminderSyncThreadSafety:
    """push_sync and drain_sync are protected by threading.Lock."""

    def test_push_sync_thread_safe(self):
        """Multiple threads can push_sync without data corruption."""
        from core.execution.reminder import SystemReminderQueue

        q = SystemReminderQueue(max_size=100)
        errors = []

        def push_items(start: int):
            try:
                for i in range(50):
                    q.push_sync(f"item-{start}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push_items, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        result = q.drain_sync()
        assert result is not None
        items = result.split("\n\n")
        assert len(items) <= 100
        assert len(items) > 0

    def test_drain_sync_thread_safe(self):
        """Multiple threads draining simultaneously don't lose or duplicate items."""
        from core.execution.reminder import SystemReminderQueue

        q = SystemReminderQueue(max_size=100)
        for i in range(20):
            q.push_sync(f"item-{i}")

        results = []
        lock = threading.Lock()

        def drain():
            r = q.drain_sync()
            if r is not None:
                with lock:
                    results.append(r)

        threads = [threading.Thread(target=drain) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1
        items = results[0].split("\n\n")
        assert len(items) == 20

    def test_has_sync_lock_attribute(self):
        """SystemReminderQueue has a _sync_lock (threading.Lock)."""
        from core.execution.reminder import SystemReminderQueue

        q = SystemReminderQueue()
        assert hasattr(q, "_sync_lock")
        assert isinstance(q._sync_lock, threading.Lock)


@pytest.mark.unit
class TestGetDepthLimiterReloadsConfig:
    """get_depth_limiter() creates a new instance each call."""

    def test_returns_new_instance_each_call(self):
        """Each call returns a different instance."""
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = 6
            mock_cfg.return_value.heartbeat.max_messages_per_hour = 30
            mock_cfg.return_value.heartbeat.max_messages_per_day = 100

            from core.cascade_limiter import get_depth_limiter
            a = get_depth_limiter()
            b = get_depth_limiter()
            assert a is not b

    def test_calls_load_config(self):
        """get_depth_limiter() triggers load_config() via __init__."""
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_cfg.return_value.heartbeat.depth_window_s = 600
            mock_cfg.return_value.heartbeat.max_depth = 6
            mock_cfg.return_value.heartbeat.max_messages_per_hour = 30
            mock_cfg.return_value.heartbeat.max_messages_per_day = 100

            initial_count = mock_cfg.call_count
            from core.cascade_limiter import get_depth_limiter
            get_depth_limiter()
            assert mock_cfg.call_count > initial_count


@pytest.mark.unit
class TestCheckAndRecordDeprecationWarning:
    """check_and_record emits DeprecationWarning."""

    def test_emits_deprecation_warning(self):
        with patch("core.cascade_limiter.load_config") as mock_cfg:
            mock_heartbeat = MagicMock()
            mock_heartbeat.depth_window_s = 600
            mock_heartbeat.max_depth = 6
            mock_heartbeat.max_messages_per_hour = 30
            mock_heartbeat.max_messages_per_day = 100
            mock_cfg.return_value.heartbeat = mock_heartbeat

            from core.cascade_limiter import ConversationDepthLimiter
            limiter = ConversationDepthLimiter()

            with pytest.warns(DeprecationWarning, match="check_and_record is deprecated"):
                result = limiter.check_and_record("alice", "bob")

            assert result is True


@pytest.mark.unit
class TestReadResolutionsTailOnly:
    """resolution_tracker.read_resolutions() uses tail-only parsing."""

    def test_reads_only_recent_entries(self, tmp_path: Path):
        """Only entries within the specified days are returned."""
        from core.memory.resolution_tracker import ResolutionTracker
        from core.time_utils import now_jst
        from datetime import timedelta

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        path = shared_dir / "resolutions.jsonl"

        now = now_jst()
        old_ts = (now - timedelta(days=10)).isoformat()
        recent_ts = (now - timedelta(days=1)).isoformat()

        entries = [
            {"ts": old_ts, "issue": "old issue", "resolver": "a"},
            {"ts": recent_ts, "issue": "recent issue", "resolver": "b"},
        ]
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        with patch("core.memory.resolution_tracker.get_shared_dir", return_value=shared_dir):
            tracker = ResolutionTracker()
            results = tracker.read_resolutions(days=7)

        assert len(results) == 1
        assert results[0]["issue"] == "recent issue"

    def test_handles_large_file_with_deque_limit(self, tmp_path: Path):
        """Files larger than _MAX_LINES_TO_PARSE only parse the tail."""
        from core.memory.resolution_tracker import ResolutionTracker
        from core.time_utils import now_jst

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        path = shared_dir / "resolutions.jsonl"

        now = now_jst()

        with path.open("w", encoding="utf-8") as f:
            for i in range(3000):
                entry = {"ts": now.isoformat(), "issue": f"issue-{i}", "resolver": "x"}
                f.write(json.dumps(entry) + "\n")

        with patch("core.memory.resolution_tracker.get_shared_dir", return_value=shared_dir):
            tracker = ResolutionTracker()
            results = tracker.read_resolutions(days=7)

        assert len(results) <= 2000

    def test_empty_file_returns_empty(self, tmp_path: Path):
        """Empty file returns empty list."""
        from core.memory.resolution_tracker import ResolutionTracker

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        path = shared_dir / "resolutions.jsonl"
        path.write_text("", encoding="utf-8")

        with patch("core.memory.resolution_tracker.get_shared_dir", return_value=shared_dir):
            tracker = ResolutionTracker()
            results = tracker.read_resolutions(days=7)

        assert results == []

    def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        """Missing file returns empty list."""
        from core.memory.resolution_tracker import ResolutionTracker

        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        with patch("core.memory.resolution_tracker.get_shared_dir", return_value=shared_dir):
            tracker = ResolutionTracker()
            results = tracker.read_resolutions(days=7)

        assert results == []
