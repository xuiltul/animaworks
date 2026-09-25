# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Issue B: Verify current_state trim is disabled by default and configurable."""

from unittest.mock import MagicMock, patch

import pytest

from core._anima_heartbeat import HeartbeatMixin
from tests.helpers.filesystem import create_anima_dir, create_test_data_dir


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    from core.config import invalidate_cache
    from core.paths import _prompt_cache

    d = create_test_data_dir(tmp_path)
    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(d))
    invalidate_cache()
    _prompt_cache.clear()
    yield d
    invalidate_cache()
    _prompt_cache.clear()


@pytest.fixture
def anima_dir(data_dir):
    return create_anima_dir(data_dir, "test-trim")


@pytest.fixture
def mixin(anima_dir):
    m = MagicMock(spec=HeartbeatMixin)
    m.name = "test-trim"
    m.anima_dir = anima_dir
    m.memory = MagicMock()
    m._get_heartbeat_md_max_bytes = MagicMock(return_value=0)
    m._get_current_state_cleanup_chars = MagicMock(return_value=0)
    m._archive_heartbeat_md_before_cleanup = lambda: HeartbeatMixin._archive_heartbeat_md_before_cleanup(m)
    m._build_heartbeat_md_cleanup_instruction = lambda hb: HeartbeatMixin._build_heartbeat_md_cleanup_instruction(m, hb)
    return m


class TestEnforceStateSizeLimit:
    """_enforce_state_size_limit respects heartbeat.current_state_max_chars."""

    def test_noop_when_disabled(self, mixin):
        """When max_chars=0, state is never trimmed."""
        mixin._get_current_state_max_chars = MagicMock(return_value=0)
        mixin.memory.read_current_state.return_value = "x" * 50000

        HeartbeatMixin._enforce_state_size_limit(mixin)

        mixin.memory.update_state.assert_not_called()
        mixin.memory.append_episode.assert_not_called()

    def test_trims_when_over_limit(self, mixin):
        """When max_chars>0 and state exceeds it, state is trimmed."""
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "a" * 200

        HeartbeatMixin._enforce_state_size_limit(mixin)

        mixin.memory.update_state.assert_called_once()
        trimmed_state = mixin.memory.update_state.call_args[0][0]
        assert len(trimmed_state) <= 100

        mixin.memory.append_episode.assert_called_once()
        overflow = mixin.memory.append_episode.call_args[0][0]
        assert "overflow" in overflow

    def test_no_trim_when_under_limit(self, mixin):
        """When max_chars>0 but state is short enough, no trim."""
        mixin._get_current_state_max_chars = MagicMock(return_value=3000)
        mixin.memory.read_current_state.return_value = "short state"

        HeartbeatMixin._enforce_state_size_limit(mixin)

        mixin.memory.update_state.assert_not_called()


class TestHeartbeatPromptCleanupInstruction:
    """Heartbeat prompt injects cleanup only when threshold is active."""

    @pytest.mark.asyncio
    async def test_no_injection_when_disabled(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=0)
        mixin._build_state_cleanup_instruction = lambda: HeartbeatMixin._build_state_cleanup_instruction(mixin)
        mixin.memory.read_heartbeat_config.return_value = None
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="hb"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mixin)

        assert parts == ["hb"]

    @pytest.mark.asyncio
    async def test_injection_when_over_threshold(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "x" * 200
        mixin._build_state_cleanup_instruction = lambda: HeartbeatMixin._build_state_cleanup_instruction(mixin)
        mixin.memory.read_heartbeat_config.return_value = None
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="hb"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mixin)

        assert len(parts) == 2
        assert "current_state.md" in parts[1]
        assert "200" in parts[1] and "100" in parts[1]


class TestStateCleanupInstruction:
    """_build_state_cleanup_instruction fires at 80% of the limit."""

    def test_i18n_string_defined(self):
        """Regression: the cleanup key must exist in the i18n catalog.

        When the key is missing, t() silently returns the raw key string and
        animas receive a meaningless token instead of an instruction.
        """
        from core.i18n import _STRINGS

        assert "heartbeat.current_state_cleanup_required" in _STRINGS

    def test_none_when_disabled(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=0)

        assert HeartbeatMixin._build_state_cleanup_instruction(mixin) is None

    def test_fires_above_soft_threshold(self, mixin):
        """State between 80% and 100% of the limit must trigger the nudge.

        The hard trim keeps the file just under the limit, so a 100% trigger
        would never fire again once trimming starts.
        """
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "x" * 90

        result = HeartbeatMixin._build_state_cleanup_instruction(mixin)

        assert result is not None
        assert "current_state.md" in result
        assert result != "heartbeat.current_state_cleanup_required"

    def test_none_below_soft_threshold(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "x" * 70

        assert HeartbeatMixin._build_state_cleanup_instruction(mixin) is None

    def test_configured_cleanup_threshold_fires_independently_of_hard_limit(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=8000)
        mixin._get_current_state_cleanup_chars = MagicMock(return_value=2000)
        mixin.memory.read_current_state.return_value = "x" * 2001

        result = HeartbeatMixin._build_state_cleanup_instruction(mixin)

        assert result is not None
        assert "2001" in result
        assert "1000" in result

    def test_configured_cleanup_threshold_does_not_fire_below_limit(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=8000)
        mixin._get_current_state_cleanup_chars = MagicMock(return_value=2000)
        mixin.memory.read_current_state.return_value = "x" * 2000

        assert HeartbeatMixin._build_state_cleanup_instruction(mixin) is None

    def test_cron_prompt_includes_cleanup(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "x" * 200
        mixin._build_state_cleanup_instruction = lambda: HeartbeatMixin._build_state_cleanup_instruction(mixin)
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="cron"):
            prompt = HeartbeatMixin._build_cron_prompt(mixin, "task", "desc")

        assert "current_state.md" in prompt


class TestHeartbeatMdCleanupInstruction:
    """_build_heartbeat_md_cleanup_instruction respects heartbeat.heartbeat_md_max_bytes."""

    def test_i18n_string_defined(self):
        from core.i18n import _STRINGS

        assert "heartbeat.heartbeat_md_cleanup_required" in _STRINGS

    def test_none_when_disabled(self, mixin):
        mixin._get_heartbeat_md_max_bytes = MagicMock(return_value=0)

        result = HeartbeatMixin._build_heartbeat_md_cleanup_instruction(mixin, "x" * 50000)

        assert result is None

    def test_none_when_under_limit(self, mixin):
        mixin._get_heartbeat_md_max_bytes = MagicMock(return_value=100)

        result = HeartbeatMixin._build_heartbeat_md_cleanup_instruction(mixin, "x" * 100)

        assert result is None

    def test_fires_when_over_limit_by_bytes(self, mixin):
        """Multibyte text must be measured in bytes, not characters."""
        mixin._get_heartbeat_md_max_bytes = MagicMock(return_value=100)
        hb = "あ" * 40  # 40 chars, 120 bytes in UTF-8

        result = HeartbeatMixin._build_heartbeat_md_cleanup_instruction(mixin, hb)

        assert result is not None
        assert "heartbeat.md" in result
        assert result != "heartbeat.heartbeat_md_cleanup_required"

    def test_default_limits(self, data_dir):
        from core.config.schemas import HeartbeatConfig

        config = HeartbeatConfig()
        assert config.current_state_cleanup_chars == 2000
        assert config.heartbeat_md_max_bytes == 8000

    def test_archives_before_cleanup_once_per_day(self, mixin):
        from core.time_utils import now_local

        source = mixin.anima_dir / "heartbeat.md"
        source.write_text("original heartbeat content", encoding="utf-8")
        mixin._get_heartbeat_md_max_bytes = MagicMock(return_value=10)

        first = HeartbeatMixin._build_heartbeat_md_cleanup_instruction(mixin, "x" * 20)
        archive = mixin.anima_dir / "archive" / "heartbeat" / f"heartbeat.md.{now_local().strftime('%Y%m%d')}"
        assert first is not None
        assert "archive/heartbeat/" in first
        assert archive.read_text(encoding="utf-8") == "original heartbeat content"

        source.write_text("updated heartbeat content", encoding="utf-8")
        HeartbeatMixin._build_heartbeat_md_cleanup_instruction(mixin, "x" * 20)
        assert archive.read_text(encoding="utf-8") == "original heartbeat content"

    @pytest.mark.asyncio
    async def test_heartbeat_prompt_includes_instruction(self, mixin):
        mixin.memory.read_heartbeat_config.return_value = "x" * 30000
        mixin._get_heartbeat_md_max_bytes = MagicMock(return_value=8000)
        mixin._build_state_cleanup_instruction = MagicMock(return_value=None)
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with (
            patch("core._anima_heartbeat._build_cron_rejected_notice", return_value=None),
            patch("core._anima_heartbeat._build_stale_task_scoreboard", return_value=None),
            patch("core._anima_heartbeat._build_curator_review_part", return_value=None),
        ):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mixin)

        assert any("heartbeat.md" in p and "KB" in p for p in parts[1:])
