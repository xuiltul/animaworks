"""Unit tests for UIConfig and theme settings."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestUIConfig:
    """Tests for UIConfig model."""

    def test_default_theme(self):
        from core.config.models import UIConfig

        cfg = UIConfig()
        assert cfg.theme == "default"

    def test_business_theme(self):
        from core.config.models import UIConfig

        cfg = UIConfig(theme="business")
        assert cfg.theme == "business"

    def test_custom_theme_string(self):
        from core.config.models import UIConfig

        cfg = UIConfig(theme="custom")
        assert cfg.theme == "custom"


class TestAnimaWorksConfigUI:
    """Tests for UIConfig in AnimaWorksConfig."""

    def test_default_has_ui(self):
        from core.config.models import AnimaWorksConfig

        cfg = AnimaWorksConfig()
        assert hasattr(cfg, "ui")
        assert cfg.ui.theme == "default"

    def test_missing_ui_field_in_json(self):
        """Existing config without 'ui' field should load without error."""
        from core.config.models import AnimaWorksConfig

        config_data = {"version": 1, "setup_complete": True}
        cfg = AnimaWorksConfig(**config_data)
        assert cfg.ui.theme == "default"
        assert cfg.setup_complete is True

    def test_ui_field_in_json(self):
        from core.config.models import AnimaWorksConfig

        config_data = {"version": 1, "ui": {"theme": "business"}}
        cfg = AnimaWorksConfig(**config_data)
        assert cfg.ui.theme == "business"

    def test_serialization_roundtrip(self):
        from core.config.models import AnimaWorksConfig

        cfg = AnimaWorksConfig(ui={"theme": "business"})
        data = json.loads(cfg.model_dump_json())
        assert data["ui"]["theme"] == "business"
        cfg2 = AnimaWorksConfig(**data)
        assert cfg2.ui.theme == "business"

    def test_config_load_with_ui(self, tmp_path: Path):
        """Test loading config from file with UI settings."""
        from core.config.models import AnimaWorksConfig

        config_file = tmp_path / "config.json"
        config_data = {
            "version": 1,
            "setup_complete": True,
            "ui": {"theme": "business"},
        }
        config_file.write_text(json.dumps(config_data))
        loaded = json.loads(config_file.read_text())
        cfg = AnimaWorksConfig(**loaded)
        assert cfg.ui.theme == "business"
