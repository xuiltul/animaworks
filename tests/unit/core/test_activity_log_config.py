from __future__ import annotations

"""Tests for ActivityLogConfig and its integration with AnimaWorksConfig."""

import pytest

from pydantic import ValidationError

from core.config.models import ActivityLogConfig, AnimaWorksConfig


class TestActivityLogConfig:
    def test_defaults(self) -> None:
        cfg = ActivityLogConfig()
        assert cfg.rotation_enabled is True
        assert cfg.rotation_mode == "size"
        assert cfg.max_size_mb == 1024
        assert cfg.max_age_days == 7
        assert cfg.rotation_time == "05:00"

    def test_custom_values(self) -> None:
        cfg = ActivityLogConfig(
            rotation_enabled=False,
            rotation_mode="both",
            max_size_mb=512,
            max_age_days=14,
            rotation_time="03:30",
        )
        assert cfg.rotation_enabled is False
        assert cfg.rotation_mode == "both"
        assert cfg.max_size_mb == 512
        assert cfg.max_age_days == 14
        assert cfg.rotation_time == "03:30"

    def test_invalid_rotation_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActivityLogConfig(rotation_mode="invalid")

    def test_negative_max_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActivityLogConfig(max_size_mb=-1)

    def test_negative_max_age_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ActivityLogConfig(max_age_days=-1)


class TestAnimaWorksConfigActivityLog:
    def test_default_activity_log_section(self) -> None:
        config = AnimaWorksConfig()
        assert hasattr(config, "activity_log")
        assert config.activity_log.rotation_enabled is True
        assert config.activity_log.max_size_mb == 1024

    def test_json_roundtrip(self) -> None:
        config = AnimaWorksConfig(
            activity_log=ActivityLogConfig(rotation_mode="time", max_age_days=30),
        )
        data = config.model_dump(mode="json")
        assert data["activity_log"]["rotation_mode"] == "time"
        assert data["activity_log"]["max_age_days"] == 30

        restored = AnimaWorksConfig.model_validate(data)
        assert restored.activity_log.rotation_mode == "time"
        assert restored.activity_log.max_age_days == 30

    def test_missing_activity_log_uses_defaults(self) -> None:
        """Config without activity_log section should use defaults."""
        data = {"version": 1, "setup_complete": True}
        config = AnimaWorksConfig.model_validate(data)
        assert config.activity_log.rotation_enabled is True
        assert config.activity_log.rotation_mode == "size"
