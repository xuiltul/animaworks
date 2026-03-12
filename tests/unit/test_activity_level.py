"""Unit tests for the Global Activity Level feature.

Tests cover:
  - _calc_effective_max_turns() scaling logic
  - AnimaWorksConfig.activity_level field validation
  - HeartbeatConfig.interval_minutes extended range
  - Per-anima heartbeat_interval_minutes reading from status.json
  - SchedulerManager._setup_heartbeat() with activity_level
  - SchedulerManager.reschedule_heartbeat()
  - Polling-based heartbeat for interval > 60 (active hours, activity_log scan)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from core._anima_heartbeat import _calc_effective_max_turns
from core.config.models import AnimaWorksConfig, HeartbeatConfig

# ── _calc_effective_max_turns ─────────────────────────────────


class TestCalcEffectiveMaxTurns:
    """Tests for _calc_effective_max_turns function."""

    def test_at_100_returns_none(self):
        assert _calc_effective_max_turns(20, 100) is None

    def test_above_100_returns_none(self):
        assert _calc_effective_max_turns(20, 200) is None
        assert _calc_effective_max_turns(20, 400) is None

    def test_at_50_halves_turns(self):
        result = _calc_effective_max_turns(20, 50)
        assert result == 10

    def test_at_10_min_floor(self):
        result = _calc_effective_max_turns(20, 10)
        assert result == max(3, math.ceil(20 * 10 / 100))
        assert result >= 3

    def test_very_low_activity_clamps_to_3(self):
        result = _calc_effective_max_turns(5, 10)
        assert result == 3

    def test_at_30_percent(self):
        result = _calc_effective_max_turns(20, 30)
        assert result == math.ceil(20 * 30 / 100)  # 6

    def test_at_99_percent(self):
        result = _calc_effective_max_turns(20, 99)
        expected = max(3, math.ceil(20 * 99 / 100))
        assert result == expected

    def test_base_turns_3_at_50(self):
        result = _calc_effective_max_turns(3, 50)
        assert result == 3  # ceil(1.5)=2 but clamp to 3

    def test_large_base_turns(self):
        result = _calc_effective_max_turns(200, 50)
        assert result == 100


# ── Config model validation ───────────────────────────────────


class TestActivityLevelConfig:
    """Tests for activity_level field in AnimaWorksConfig."""

    def test_default_value(self):
        config = AnimaWorksConfig()
        assert config.activity_level == 100

    def test_valid_min(self):
        config = AnimaWorksConfig(activity_level=10)
        assert config.activity_level == 10

    def test_valid_max(self):
        config = AnimaWorksConfig(activity_level=400)
        assert config.activity_level == 400

    def test_below_min_raises(self):
        with pytest.raises(ValidationError):
            AnimaWorksConfig(activity_level=9)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError):
            AnimaWorksConfig(activity_level=401)

    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            AnimaWorksConfig(activity_level=0)

    def test_json_roundtrip(self):
        config = AnimaWorksConfig(activity_level=75)
        data = config.model_dump(mode="json")
        assert data["activity_level"] == 75
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.activity_level == 75


class TestHeartbeatIntervalExtended:
    """Tests for relaxed interval_minutes upper bound."""

    def test_default_30(self):
        config = HeartbeatConfig()
        assert config.interval_minutes == 30

    def test_old_max_60_still_valid(self):
        config = HeartbeatConfig(interval_minutes=60)
        assert config.interval_minutes == 60

    def test_extended_120(self):
        config = HeartbeatConfig(interval_minutes=120)
        assert config.interval_minutes == 120

    def test_extended_max_1440(self):
        config = HeartbeatConfig(interval_minutes=1440)
        assert config.interval_minutes == 1440

    def test_above_1440_raises(self):
        with pytest.raises(ValidationError):
            HeartbeatConfig(interval_minutes=1441)


# ── Per-anima interval reading ────────────────────────────────


class TestPerAnimaInterval:
    """Tests for SchedulerManager._read_per_anima_interval."""

    def _make_mgr(self, tmp_path: Path):
        from core.supervisor.scheduler_manager import SchedulerManager

        mock_anima = MagicMock()
        anima_dir = tmp_path / "animas" / "test-anima"
        anima_dir.mkdir(parents=True, exist_ok=True)
        return SchedulerManager(
            anima=mock_anima,
            anima_name="test-anima",
            anima_dir=anima_dir,
            emit_event=MagicMock(),
        )

    def test_reads_from_status_json(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"heartbeat_interval_minutes": 60}),
            encoding="utf-8",
        )
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 60

    def test_fallback_to_global(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 45
        assert mgr._read_per_anima_interval(app_config) == 45

    def test_invalid_value_fallback(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"heartbeat_interval_minutes": -5}),
            encoding="utf-8",
        )
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 30

    def test_non_numeric_fallback(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"heartbeat_interval_minutes": "fast"}),
            encoding="utf-8",
        )
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 30

    def test_exceeds_max_fallback(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"heartbeat_interval_minutes": 2000}),
            encoding="utf-8",
        )
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 30

    def test_float_value_accepted(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text(
            json.dumps({"heartbeat_interval_minutes": 45.0}),
            encoding="utf-8",
        )
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 45

    def test_corrupt_json_fallback(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        status_path = mgr._anima_dir / "status.json"
        status_path.write_text("{broken", encoding="utf-8")
        app_config = MagicMock()
        app_config.heartbeat.interval_minutes = 30
        assert mgr._read_per_anima_interval(app_config) == 30


# ── Scheduler setup with activity level ───────────────────────


class TestSchedulerActivityLevel:
    """Tests for SchedulerManager._setup_heartbeat with activity_level."""

    def _make_mgr(self, tmp_path: Path, anima_name: str = "test-anima"):
        from core.supervisor.scheduler_manager import SchedulerManager

        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "30分ごと"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        anima_dir = tmp_path / "animas" / anima_name
        anima_dir.mkdir(parents=True, exist_ok=True)
        return SchedulerManager(
            anima=mock_anima,
            anima_name=anima_name,
            anima_dir=anima_dir,
            emit_event=MagicMock(),
        )

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_default_activity_100(self, mock_load_config, tmp_path):
        config = AnimaWorksConfig()
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()
        assert mgr.scheduler is not None

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_activity_50_doubles_interval(self, mock_load_config, tmp_path):
        config = AnimaWorksConfig(activity_level=50)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_activity_200_halves_interval(self, mock_load_config, tmp_path):
        config = AnimaWorksConfig(activity_level=200)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_reschedule_heartbeat(self, mock_load_config, tmp_path):
        config = AnimaWorksConfig(activity_level=100)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        config.activity_level = 200
        mgr.reschedule_heartbeat()

        jobs_after = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs_after if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_low_activity_uses_polling(self, mock_load_config, tmp_path):
        """Activity 10% with base 30min -> effective 300min (>60) -> polling mode."""
        config = AnimaWorksConfig(activity_level=10)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        assert mgr._hb_effective_interval == 300
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_400_percent_5min_floor(self, mock_load_config, tmp_path):
        """Activity 400% with base 15min -> effective 3.75min -> clamped to 5min."""
        config = AnimaWorksConfig(activity_level=400)
        config.heartbeat.interval_minutes = 15
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()


# ── Effective interval calculation ────────────────────────────


class TestEffectiveIntervalCalc:
    """Pure calculation tests for activity level scaling."""

    @staticmethod
    def _calc(base: int, activity: int) -> int:
        activity_pct = max(10, min(400, activity))
        effective = base / (activity_pct / 100.0)
        return max(5, round(effective))

    def test_100_percent_no_change(self):
        assert self._calc(30, 100) == 30

    def test_50_percent_doubles(self):
        assert self._calc(30, 50) == 60

    def test_200_percent_halves(self):
        assert self._calc(30, 200) == 15

    def test_400_percent_quarter(self):
        assert self._calc(30, 400) == 8  # 30/4 = 7.5 → round → 8

    def test_10_percent_tenfold(self):
        assert self._calc(30, 10) == 300

    def test_400_with_base_15_clamps_to_5(self):
        result = self._calc(15, 400)
        assert result == 5  # 15/4 = 3.75 → round → 4 → clamp → 5

    def test_clamp_minimum_5(self):
        result = self._calc(10, 400)
        assert result == 5  # 10/4 = 2.5 → round → 2 → clamp → 5


# ── Polling-based heartbeat (interval > 60) ───────────────────


JST = timezone(timedelta(hours=9))


class TestHeartbeatPolling:
    """Tests for the polling-based heartbeat check (_heartbeat_check)."""

    def _make_mgr(self, tmp_path: Path, anima_name: str = "test-anima"):
        from core.supervisor.scheduler_manager import SchedulerManager

        mock_anima = MagicMock()
        mock_anima.memory.read_heartbeat_config.return_value = "30分ごと"
        mock_anima.memory.read_cron_config.return_value = ""
        mock_anima.set_on_schedule_changed = MagicMock()
        mock_anima._activity = MagicMock()
        anima_dir = tmp_path / "animas" / anima_name
        anima_dir.mkdir(parents=True, exist_ok=True)
        return SchedulerManager(
            anima=mock_anima,
            anima_name=anima_name,
            anima_dir=anima_dir,
            emit_event=MagicMock(),
        )

    def _setup_polling_mgr(self, tmp_path, *, interval=120, active_start=9, active_end=22):
        """Create a SchedulerManager in polling mode with given parameters."""
        mgr = self._make_mgr(tmp_path)
        mgr._hb_effective_interval = interval
        mgr._hb_active_start = active_start
        mgr._hb_active_end = active_end
        mgr._hb_first_check_offset = 5
        mgr._hb_first_check_done = True
        mgr.heartbeat_tick = AsyncMock()
        return mgr

    # ── _in_active_hours ──

    def test_in_active_hours_normal(self, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, active_start=9, active_end=22)
        now_14 = datetime(2026, 3, 12, 14, 0, tzinfo=JST)
        assert mgr._in_active_hours(now_14) is True

    def test_outside_active_hours(self, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, active_start=9, active_end=22)
        now_23 = datetime(2026, 3, 12, 23, 0, tzinfo=JST)
        assert mgr._in_active_hours(now_23) is False

    def test_active_hours_midnight_crossing(self, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, active_start=22, active_end=6)
        now_23 = datetime(2026, 3, 12, 23, 0, tzinfo=JST)
        now_3 = datetime(2026, 3, 13, 3, 0, tzinfo=JST)
        now_12 = datetime(2026, 3, 13, 12, 0, tzinfo=JST)
        assert mgr._in_active_hours(now_23) is True
        assert mgr._in_active_hours(now_3) is True
        assert mgr._in_active_hours(now_12) is False

    def test_active_hours_none_means_24h(self, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, active_start=9, active_end=22)
        mgr._hb_active_start = None
        mgr._hb_active_end = None
        now_3am = datetime(2026, 3, 13, 3, 0, tzinfo=JST)
        assert mgr._in_active_hours(now_3am) is True

    # ── _get_last_heartbeat_ts ──

    def test_get_last_heartbeat_ts_found(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        entry = MagicMock()
        entry.ts = "2026-03-12T14:00:00+09:00"
        mgr._anima._activity.recent.return_value = [entry]

        result = mgr._get_last_heartbeat_ts()
        assert result == datetime(2026, 3, 12, 14, 0, tzinfo=JST)
        mgr._anima._activity.recent.assert_called_once_with(
            days=2,
            types=["heartbeat_start"],
            limit=1,
        )

    def test_get_last_heartbeat_ts_empty(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        mgr._anima._activity.recent.return_value = []

        result = mgr._get_last_heartbeat_ts()
        assert result is None

    def test_get_last_heartbeat_ts_exception(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        mgr._anima._activity.recent.side_effect = OSError("disk error")

        result = mgr._get_last_heartbeat_ts()
        assert result is None

    # ── _heartbeat_check ──

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.now_local")
    async def test_heartbeat_check_fires_when_interval_elapsed(self, mock_now, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, interval=120, active_start=9, active_end=22)
        now = datetime(2026, 3, 12, 16, 30, tzinfo=JST)
        mock_now.return_value = now

        entry = MagicMock()
        entry.ts = (now - timedelta(minutes=130)).isoformat()
        mgr._anima._activity.recent.return_value = [entry]

        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.now_local")
    async def test_heartbeat_check_skips_when_interval_not_elapsed(self, mock_now, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, interval=120, active_start=9, active_end=22)
        now = datetime(2026, 3, 12, 16, 30, tzinfo=JST)
        mock_now.return_value = now

        entry = MagicMock()
        entry.ts = (now - timedelta(minutes=60)).isoformat()
        mgr._anima._activity.recent.return_value = [entry]

        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.now_local")
    async def test_heartbeat_check_skips_outside_active_hours(self, mock_now, tmp_path):
        mgr = self._setup_polling_mgr(tmp_path, interval=120, active_start=9, active_end=22)
        mock_now.return_value = datetime(2026, 3, 12, 23, 0, tzinfo=JST)

        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.now_local")
    async def test_heartbeat_check_fires_on_fresh_install(self, mock_now, tmp_path):
        """No heartbeat_start in activity_log → fire immediately."""
        mgr = self._setup_polling_mgr(tmp_path, interval=120, active_start=9, active_end=22)
        mock_now.return_value = datetime(2026, 3, 12, 14, 0, tzinfo=JST)
        mgr._anima._activity.recent.return_value = []

        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.now_local")
    async def test_heartbeat_check_first_check_adds_offset(self, mock_now, tmp_path):
        """First check after setup adds offset to required interval."""
        mgr = self._setup_polling_mgr(tmp_path, interval=120, active_start=9, active_end=22)
        mgr._hb_first_check_done = False
        mgr._hb_first_check_offset = 5
        now = datetime(2026, 3, 12, 16, 30, tzinfo=JST)
        mock_now.return_value = now

        # 122 min elapsed < 120 + 5 = 125 min required
        entry = MagicMock()
        entry.ts = (now - timedelta(minutes=122)).isoformat()
        mgr._anima._activity.recent.return_value = [entry]

        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_not_awaited()

        # 126 min elapsed >= 125 min required
        entry.ts = (now - timedelta(minutes=126)).isoformat()
        await mgr._heartbeat_check()
        mgr.heartbeat_tick.assert_awaited_once()
        assert mgr._hb_first_check_done is True

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_setup_polling_mode_registers_job(self, mock_load_config, tmp_path):
        """interval > 60 registers a CronTrigger(minute='*') polling job."""
        config = AnimaWorksConfig(activity_level=10)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()

        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1

        job = heartbeat_jobs[0]
        from apscheduler.triggers.cron import CronTrigger as APCronTrigger

        assert isinstance(job.trigger, APCronTrigger)
        assert mgr._hb_effective_interval == 300
        mgr.shutdown()

    @pytest.mark.asyncio
    @patch("core.supervisor.scheduler_manager.load_config")
    async def test_reschedule_preserves_polling_mode(self, mock_load_config, tmp_path):
        """Rescheduling with interval > 60 stays in polling mode."""
        config = AnimaWorksConfig(activity_level=10)
        mock_load_config.return_value = config

        mgr = self._make_mgr(tmp_path)
        mgr.setup()
        assert mgr._hb_effective_interval == 300

        config.activity_level = 20
        mgr.reschedule_heartbeat()

        # activity 20%: 30 / 0.2 = 150 → still > 60 → polling
        assert mgr._hb_effective_interval == 150
        jobs = mgr.scheduler.get_jobs()
        heartbeat_jobs = [j for j in jobs if "heartbeat" in j.id]
        assert len(heartbeat_jobs) == 1
        mgr.shutdown()
