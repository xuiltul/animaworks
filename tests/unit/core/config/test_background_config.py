"""Tests for background task config models in core/config/models.py."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.config.models import (
    AnimaWorksConfig,
    BackgroundTaskConfig,
    BackgroundToolConfig,
    GPUConfig,
    PrimingConfig,
    ServerConfig,
    resolve_background_worker_pool_size,
)

# ── ServerConfig ─────────────────────────────────────────────


class TestServerConfig:
    def test_server_config_defaults(self):
        """ServerConfig defaults to ipc_stream_timeout=60 and keepalive_interval=30."""
        sc = ServerConfig()
        assert sc.ipc_stream_timeout == 60
        assert sc.keepalive_interval == 30
        assert sc.anima_startup_ready_timeout == 120
        assert sc.anima_stop_timeout == 60.0
        assert sc.health_check_warmup_seconds == 300
        assert sc.runner_warmup_seconds == 180
        assert sc.spawn_timeout == 300
        assert sc.supervisor_respawn_max_retries == 3
        assert sc.supervisor_respawn_retry_interval_seconds == 30.0

    def test_server_config_rejects_invalid_intervals(self):
        """keepalive_interval must be less than ipc_stream_timeout."""
        with pytest.raises(ValueError, match="keepalive_interval"):
            ServerConfig(keepalive_interval=120, ipc_stream_timeout=60)

    def test_server_config_rejects_equal_intervals(self):
        """keepalive_interval equal to ipc_stream_timeout is rejected."""
        with pytest.raises(ValueError, match="keepalive_interval"):
            ServerConfig(keepalive_interval=60, ipc_stream_timeout=60)

    def test_server_config_accepts_valid_intervals(self):
        """keepalive_interval < ipc_stream_timeout is accepted."""
        sc = ServerConfig(keepalive_interval=15, ipc_stream_timeout=120)
        assert sc.keepalive_interval == 15
        assert sc.ipc_stream_timeout == 120

    def test_server_config_custom_anima_startup_ready_timeout(self):
        sc = ServerConfig(anima_startup_ready_timeout=300)
        assert sc.anima_startup_ready_timeout == 300

    def test_server_config_custom_anima_stop_timeout(self):
        sc = ServerConfig(anima_stop_timeout=45.5)
        assert sc.anima_stop_timeout == 45.5

    def test_server_config_custom_warmup_and_respawn_settings(self):
        sc = ServerConfig(
            health_check_warmup_seconds=10,
            runner_warmup_seconds=20,
            spawn_timeout=30,
            supervisor_respawn_max_retries=4,
            supervisor_respawn_retry_interval_seconds=0.5,
        )
        assert sc.health_check_warmup_seconds == 10
        assert sc.runner_warmup_seconds == 20
        assert sc.spawn_timeout == 30
        assert sc.supervisor_respawn_max_retries == 4
        assert sc.supervisor_respawn_retry_interval_seconds == 0.5


class TestPrimingConfig:
    def test_channel_timeout_default(self):
        pc = PrimingConfig()
        assert pc.channel_timeout_seconds == 60.0


class TestGPUConfig:
    def test_embedding_bulk_yield_batches_default(self):
        gpu = GPUConfig()
        assert gpu.embedding_bulk_yield_batches == 5


# ── BackgroundTaskConfig ─────────────────────────────────────


class TestBackgroundTaskConfig:
    def test_background_task_config_defaults(self):
        """BackgroundTaskConfig has expected defaults."""
        btc = BackgroundTaskConfig()
        assert btc.enabled is True
        assert btc.result_retention_hours == 24
        assert btc.result_memory_retention_minutes == 60
        assert btc.max_completed_tasks_in_memory == 200
        assert btc.worker_pool_size == 1
        assert btc.shutdown_drain_seconds == 600
        assert isinstance(btc.eligible_tools, dict)

    def test_retired_task_control_keys_are_ignored(self):
        """A config.json left over from before the task-control teardown loads."""
        btc = BackgroundTaskConfig(
            completion_declaration_required=True,
            blocked_recovery_enabled=True,
            blocked_reprobe_after_hours=6.0,
            blocked_reprobe_batch_limit=3,
            blocked_recovery_scan_minutes=15.0,
            blocked_max_reprobes=4,
            blocked_check_timeout_seconds=60,
            blocked_checkless_reprobe_enabled=True,
        )
        assert btc.enabled is True
        assert not hasattr(btc, "completion_declaration_required")
        assert not hasattr(btc, "blocked_recovery_enabled")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("result_memory_retention_minutes", -1),
            ("max_completed_tasks_in_memory", -1),
            ("shutdown_drain_seconds", -1),
        ],
    )
    def test_background_task_config_rejects_negative_memory_limits(self, field, value):
        with pytest.raises(ValueError):
            BackgroundTaskConfig(**{field: value})

    @pytest.mark.parametrize("value", [0, 11, -1])
    def test_background_task_config_rejects_invalid_worker_pool_size(self, value):
        with pytest.raises(ValueError):
            BackgroundTaskConfig(worker_pool_size=value)

    def test_background_task_config_accepts_worker_pool_size_bounds(self):
        assert BackgroundTaskConfig(worker_pool_size=1).worker_pool_size == 1
        assert BackgroundTaskConfig(worker_pool_size=10).worker_pool_size == 10

    def test_background_worker_pool_status_override(self, tmp_path):
        (tmp_path / "status.json").write_text(
            '{"background_worker_pool_size": 4}',
            encoding="utf-8",
        )

        assert resolve_background_worker_pool_size(tmp_path, default=2) == 4

    @pytest.mark.parametrize(
        "status",
        [
            '{"background_worker_pool_size": 0}',
            '{"background_worker_pool_size": 11}',
            '{"background_worker_pool_size": true}',
            '{"background_worker_pool_size": "3"}',
            "not-json",
        ],
    )
    def test_invalid_background_worker_pool_override_uses_default(self, tmp_path, status):
        (tmp_path / "status.json").write_text(status, encoding="utf-8")

        assert resolve_background_worker_pool_size(tmp_path, default=2) == 2

    def test_background_task_config_eligible_tools(self):
        """Eligible tools include image_gen schema names, local_llm, run_command."""
        btc = BackgroundTaskConfig()

        # Image gen schema names (all threshold 30)
        for name in (
            "generate_character_assets",
            "generate_fullbody",
            "generate_bustup",
            "generate_icon",
            "generate_chibi",
            "generate_3d_model",
            "generate_rigged_model",
            "generate_animations",
        ):
            assert name in btc.eligible_tools, f"{name} missing"
            assert btc.eligible_tools[name].threshold_s == 30

        # Other background tools
        assert "local_llm" in btc.eligible_tools
        assert "run_command" in btc.eligible_tools
        assert btc.eligible_tools["local_llm"].threshold_s == 60
        assert btc.eligible_tools["run_command"].threshold_s == 60

        # Old category name must NOT be present
        assert "image_generation" not in btc.eligible_tools

    def test_background_tool_config_defaults(self):
        """BackgroundToolConfig has a default threshold_s."""
        btool = BackgroundToolConfig()
        assert btool.threshold_s == 30

    def test_background_tool_config_custom(self):
        """BackgroundToolConfig accepts custom threshold."""
        btool = BackgroundToolConfig(threshold_s=120)
        assert btool.threshold_s == 120


# ── AnimaWorksConfig integration ─────────────────────────────


class TestAnimaWorksConfigBackground:
    def test_animaworks_config_has_server(self):
        """AnimaWorksConfig includes server field with correct type."""
        config = AnimaWorksConfig()
        assert isinstance(config.server, ServerConfig)
        assert config.server.ipc_stream_timeout == 60
        assert config.server.keepalive_interval == 30

    def test_animaworks_config_has_background_task(self):
        """AnimaWorksConfig includes background_task field with correct type."""
        config = AnimaWorksConfig()
        assert isinstance(config.background_task, BackgroundTaskConfig)
        assert config.background_task.enabled is True

    def test_config_serialization_roundtrip(self):
        """model_dump / model_validate round-trip works for background config."""
        config = AnimaWorksConfig()
        # Modify background_task settings
        config.background_task.enabled = False
        config.background_task.result_retention_hours = 48
        config.background_task.result_memory_retention_minutes = 30
        config.background_task.max_completed_tasks_in_memory = 50
        config.background_task.eligible_tools["custom_tool"] = BackgroundToolConfig(
            threshold_s=90,
        )
        config.server.ipc_stream_timeout = 600

        # Serialize and deserialize
        data = config.model_dump(mode="json")
        restored = AnimaWorksConfig.model_validate(data)

        # Verify background_task round-trip
        assert restored.background_task.enabled is False
        assert restored.background_task.result_retention_hours == 48
        assert restored.background_task.result_memory_retention_minutes == 30
        assert restored.background_task.max_completed_tasks_in_memory == 50
        assert "custom_tool" in restored.background_task.eligible_tools
        assert restored.background_task.eligible_tools["custom_tool"].threshold_s == 90

        # Verify server round-trip
        assert restored.server.ipc_stream_timeout == 600

        # Verify default eligible tools survived round-trip
        assert "generate_character_assets" in restored.background_task.eligible_tools
        assert restored.background_task.eligible_tools["generate_character_assets"].threshold_s == 30
