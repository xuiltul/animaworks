from __future__ import annotations

"""Unit tests for Heartbeat 2-stage timeout (soft + hard).

Covers:
  - _calc_effective_max_turns with hb_max_turns override
  - Soft timeout: Mode A reminder_queue injection
  - Hard timeout: Mode A loop break + recovery_note
  - Mode S session_stats flags for PreToolUse hook
  - HeartbeatConfig validation (soft < hard, bounds)
"""

import math
import time
from typing import Any

import pytest
from pydantic import ValidationError

from core._anima_heartbeat import _calc_effective_max_turns
from core.config.models import HeartbeatConfig

# ── _calc_effective_max_turns ──────────────────────────────────


class TestCalcEffectiveMaxTurns:
    """Test max_turns calculation with HB-specific override."""

    def test_no_hb_override_activity_100(self):
        result = _calc_effective_max_turns(20, 100, hb_max_turns=None)
        assert result is None

    def test_no_hb_override_low_activity(self):
        result = _calc_effective_max_turns(20, 50, hb_max_turns=None)
        assert result == max(3, math.ceil(20 * 50 / 100))

    def test_hb_override_activity_100(self):
        result = _calc_effective_max_turns(20, 100, hb_max_turns=15)
        assert result == 15

    def test_hb_override_low_activity(self):
        result = _calc_effective_max_turns(20, 50, hb_max_turns=15)
        assert result == max(3, math.ceil(15 * 50 / 100))

    def test_hb_override_very_low_activity_floor(self):
        result = _calc_effective_max_turns(20, 10, hb_max_turns=5)
        assert result == 3

    def test_default_no_override(self):
        result = _calc_effective_max_turns(20, 100)
        assert result is None


# ── HeartbeatConfig validation ────────────────────────────────


class TestHeartbeatConfigValidation:
    """Test HeartbeatConfig field constraints."""

    def test_defaults(self):
        cfg = HeartbeatConfig()
        assert cfg.soft_timeout_seconds == 300
        assert cfg.hard_timeout_seconds == 600
        assert cfg.max_turns is None

    def test_custom_values(self):
        cfg = HeartbeatConfig(
            soft_timeout_seconds=120,
            hard_timeout_seconds=360,
            max_turns=10,
        )
        assert cfg.soft_timeout_seconds == 120
        assert cfg.hard_timeout_seconds == 360
        assert cfg.max_turns == 10

    def test_soft_timeout_min(self):
        with pytest.raises(ValidationError):
            HeartbeatConfig(soft_timeout_seconds=10)

    def test_hard_timeout_min(self):
        with pytest.raises(ValidationError):
            HeartbeatConfig(hard_timeout_seconds=30)

    def test_max_turns_min(self):
        with pytest.raises(ValidationError):
            HeartbeatConfig(max_turns=1)

    def test_max_turns_none_allowed(self):
        cfg = HeartbeatConfig(max_turns=None)
        assert cfg.max_turns is None


# ── Mode A soft timeout (reminder_queue) ──────────────────────


class TestModeASoftTimeout:
    """Test that soft timeout injects a reminder into the queue."""

    @pytest.fixture()
    def mock_reminder_queue(self):
        from core.execution.reminder import SystemReminderQueue

        return SystemReminderQueue()

    def test_reminder_pushed_after_soft_timeout(self, mock_reminder_queue):
        mock_reminder_queue.push_sync("⏰ Heartbeat time limit approaching")
        content = mock_reminder_queue.drain_sync()
        assert content is not None
        assert "Heartbeat" in content

    def test_reminder_not_pushed_before_timeout(self, mock_reminder_queue):
        content = mock_reminder_queue.drain_sync()
        assert content is None


# ── Mode S session_stats flags ─────────────────────────────────


class TestModeSSessionStats:
    """Test that session_stats includes HB timeout fields."""

    def test_session_stats_heartbeat_fields(self):
        session_stats: dict[str, Any] = {
            "tool_call_count": 0,
            "total_result_bytes": 0,
            "system_prompt_tokens": 1000,
            "user_prompt_tokens": 500,
            "force_chain": False,
            "trigger": "heartbeat",
            "start_time": time.monotonic(),
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
        }
        assert session_stats["trigger"] == "heartbeat"
        assert session_stats["hb_soft_warned"] is False
        assert session_stats["hb_soft_timeout"] == 300
        assert isinstance(session_stats["start_time"], float)

    def test_session_stats_chat_no_hb_trigger(self):
        session_stats: dict[str, Any] = {
            "trigger": "chat",
            "start_time": time.monotonic(),
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
        }
        assert session_stats["trigger"] != "heartbeat"


def _sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except ImportError:
        return False


# ── Mode S PreToolUse hook soft timeout ────────────────────────


@pytest.mark.skipif(not _sdk_available(), reason="claude_agent_sdk not installed")
class TestPreToolHookSoftTimeout:
    """Test the PreToolUse hook heartbeat soft timeout injection."""

    @pytest.fixture()
    def session_stats_expired(self):
        return {
            "tool_call_count": 5,
            "total_result_bytes": 10000,
            "system_prompt_tokens": 1000,
            "user_prompt_tokens": 500,
            "force_chain": False,
            "trigger": "heartbeat",
            "start_time": time.monotonic() - 400,
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
            "min_trust_seen": 2,
        }

    @pytest.fixture()
    def session_stats_not_expired(self):
        return {
            "tool_call_count": 1,
            "total_result_bytes": 100,
            "system_prompt_tokens": 1000,
            "user_prompt_tokens": 500,
            "force_chain": False,
            "trigger": "heartbeat",
            "start_time": time.monotonic(),
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
            "min_trust_seen": 2,
        }

    @pytest.fixture()
    def session_stats_chat_trigger(self):
        return {
            "tool_call_count": 5,
            "total_result_bytes": 10000,
            "system_prompt_tokens": 1000,
            "user_prompt_tokens": 500,
            "force_chain": False,
            "trigger": "chat",
            "start_time": time.monotonic() - 400,
            "hb_soft_warned": False,
            "hb_soft_timeout": 300,
            "min_trust_seen": 2,
        }

    @pytest.mark.asyncio
    async def test_hook_injects_warning_on_expired(self, tmp_path, session_stats_expired):
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            tmp_path,
            session_stats=session_stats_expired,
        )
        result = await hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "test.txt")}},
            "test-id",
            {},
        )
        assert session_stats_expired["hb_soft_warned"] is True
        # SyncHookJSONOutput is a TypedDict-like dict
        output = (
            result.hookSpecificOutput if hasattr(result, "hookSpecificOutput") else result.get("hookSpecificOutput")
        )
        assert output is not None
        ctx = (
            output.get("additionalContext") if isinstance(output, dict) else getattr(output, "additionalContext", None)
        )
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_hook_no_warning_before_timeout(self, tmp_path, session_stats_not_expired):
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            tmp_path,
            session_stats=session_stats_not_expired,
        )
        await hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "test.txt")}},
            "test-id",
            {},
        )
        assert session_stats_not_expired["hb_soft_warned"] is False

    @pytest.mark.asyncio
    async def test_hook_no_warning_for_chat_trigger(self, tmp_path, session_stats_chat_trigger):
        from core.execution._sdk_hooks import _build_pre_tool_hook

        hook = _build_pre_tool_hook(
            tmp_path,
            session_stats=session_stats_chat_trigger,
        )
        await hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(tmp_path / "test.txt")}},
            "test-id",
            {},
        )
        assert session_stats_chat_trigger["hb_soft_warned"] is False


# ── Hard timeout recovery note ─────────────────────────────────


class TestHardTimeoutRecoveryNote:
    """Test that hard timeout writes recovery_note.md."""

    def test_recovery_note_written(self, tmp_path):
        from core.i18n import t

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        recovery_path = state_dir / "recovery_note.md"

        content = t("reminder.hb_hard_timeout_recovery", timeout=600)
        recovery_path.write_text(content, encoding="utf-8")

        assert recovery_path.exists()
        text = recovery_path.read_text(encoding="utf-8")
        assert "600" in text

    def test_recovery_note_content_ja(self):
        from core.i18n import t

        content = t("reminder.hb_hard_timeout_recovery", locale="ja", timeout=600)
        assert "制限時間" in content or "600" in content

    def test_recovery_note_content_en(self):
        from core.i18n import t

        content = t("reminder.hb_hard_timeout_recovery", locale="en", timeout=600)
        assert "600" in content


# ── i18n strings exist ──────────────────────────────────────────


class TestI18nStrings:
    """Verify the new i18n keys are registered."""

    def test_hb_time_limit_key_exists(self):
        from core.i18n import _STRINGS

        assert "reminder.hb_time_limit" in _STRINGS

    def test_hb_hard_timeout_recovery_key_exists(self):
        from core.i18n import _STRINGS

        assert "reminder.hb_hard_timeout_recovery" in _STRINGS

    def test_hb_time_limit_has_both_locales(self):
        from core.i18n import _STRINGS

        entry = _STRINGS["reminder.hb_time_limit"]
        assert "ja" in entry
        assert "en" in entry

    def test_hb_hard_timeout_recovery_has_both_locales(self):
        from core.i18n import _STRINGS

        entry = _STRINGS["reminder.hb_hard_timeout_recovery"]
        assert "ja" in entry
        assert "en" in entry
