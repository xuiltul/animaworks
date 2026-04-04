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
        mixin.memory.read_heartbeat_config.return_value = None
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="hb"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mixin)

        assert parts == ["hb"]

    @pytest.mark.asyncio
    async def test_injection_when_over_threshold(self, mixin):
        mixin._get_current_state_max_chars = MagicMock(return_value=100)
        mixin.memory.read_current_state.return_value = "x" * 200
        mixin.memory.read_heartbeat_config.return_value = None
        mixin._build_background_context_parts = MagicMock(return_value=[])

        with patch("core._anima_heartbeat.load_prompt", return_value="hb"):
            parts = await HeartbeatMixin._build_heartbeat_prompt(mixin)

        assert len(parts) == 2
        assert "current_state" in parts[1] or "200" in parts[1] or "100" in parts[1]
