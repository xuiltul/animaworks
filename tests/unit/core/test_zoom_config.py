"""Unit tests for Zoom RTMS configuration schema and source registration."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from core.config.models import AnimaWorksConfig, ExternalMessagingConfig
from core.config.schemas import ZoomRTMSConfig
from core.schemas import EXTERNAL_PLATFORM_SOURCES


class TestZoomRTMSConfigDefaults:
    def test_defaults(self):
        cfg = ZoomRTMSConfig()
        assert cfg.enabled is False
        assert cfg.default_anima == ""
        assert cfg.meeting_mapping == {}
        assert cfg.chunk_interval_seconds == 300
        assert cfg.chunk_max_chars == 4000

    def test_custom_values(self):
        cfg = ZoomRTMSConfig(
            enabled=True,
            default_anima="kotoha",
            meeting_mapping={"123456789": "sakura"},
            chunk_interval_seconds=120,
            chunk_max_chars=2000,
        )
        assert cfg.enabled is True
        assert cfg.default_anima == "kotoha"
        assert cfg.meeting_mapping["123456789"] == "sakura"
        assert cfg.chunk_interval_seconds == 120
        assert cfg.chunk_max_chars == 2000


class TestExternalMessagingZoomWiring:
    def test_zoom_present_and_default_disabled(self):
        cfg = ExternalMessagingConfig()
        assert isinstance(cfg.zoom, ZoomRTMSConfig)
        assert cfg.zoom.enabled is False

    def test_in_animaworks_config(self):
        cfg = AnimaWorksConfig()
        assert isinstance(cfg.external_messaging.zoom, ZoomRTMSConfig)

    def test_enabled_config_validates(self):
        cfg = ExternalMessagingConfig(
            zoom=ZoomRTMSConfig(enabled=True, default_anima="kotoha"),
        )
        assert cfg.zoom.enabled is True
        assert cfg.zoom.default_anima == "kotoha"

    def test_serialization_roundtrip(self):
        cfg = AnimaWorksConfig(
            external_messaging=ExternalMessagingConfig(
                zoom=ZoomRTMSConfig(
                    enabled=True,
                    meeting_mapping={"123": "kotoha"},
                    chunk_interval_seconds=60,
                ),
            ),
        )
        data = cfg.model_dump(mode="json")
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.external_messaging.zoom.enabled is True
        assert restored.external_messaging.zoom.meeting_mapping["123"] == "kotoha"
        assert restored.external_messaging.zoom.chunk_interval_seconds == 60


class TestZoomInExternalSources:
    def test_zoom_registered(self):
        assert "zoom" in EXTERNAL_PLATFORM_SOURCES

    def test_existing_sources_preserved(self):
        for source in ("slack", "chatwork", "discord", "googlechat"):
            assert source in EXTERNAL_PLATFORM_SOURCES
