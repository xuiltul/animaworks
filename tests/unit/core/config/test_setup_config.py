"""Unit tests for setup-related config fields (setup_complete, locale)."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.models import (
    AnimaWorksConfig,
    invalidate_cache,
    load_config,
    save_config,
)


class TestSetupCompleteField:
    """Tests for the setup_complete field on AnimaWorksConfig."""

    def test_defaults_to_false(self):
        config = AnimaWorksConfig()
        assert config.setup_complete is False

    def test_set_to_true(self):
        config = AnimaWorksConfig(setup_complete=True)
        assert config.setup_complete is True

    def test_serialization(self):
        config = AnimaWorksConfig(setup_complete=True)
        data = config.model_dump(mode="json")
        assert data["setup_complete"] is True

    def test_deserialization(self):
        data = {"version": 1, "setup_complete": True}
        config = AnimaWorksConfig.model_validate(data)
        assert config.setup_complete is True

    def test_missing_in_json_defaults_false(self):
        data = {"version": 1}
        config = AnimaWorksConfig.model_validate(data)
        assert config.setup_complete is False


class TestLocaleField:
    """Tests for the locale field on AnimaWorksConfig."""

    def test_defaults_to_ja(self):
        config = AnimaWorksConfig()
        assert config.locale == "ja"

    def test_set_to_en(self):
        config = AnimaWorksConfig(locale="en")
        assert config.locale == "en"

    def test_serialization(self):
        config = AnimaWorksConfig(locale="en")
        data = config.model_dump(mode="json")
        assert data["locale"] == "en"

    def test_deserialization(self):
        data = {"version": 1, "locale": "en"}
        config = AnimaWorksConfig.model_validate(data)
        assert config.locale == "en"

    def test_missing_in_json_defaults_ja(self):
        data = {"version": 1}
        config = AnimaWorksConfig.model_validate(data)
        assert config.locale == "ja"


class TestSetupConfigRoundTrip:
    """Tests for save/load round-trip of setup fields."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        invalidate_cache()
        yield
        invalidate_cache()

    def test_roundtrip_preserves_setup_complete(self, tmp_path: Path):
        config = AnimaWorksConfig(setup_complete=True)
        path = tmp_path / "config.json"
        save_config(config, path)
        invalidate_cache()

        loaded = load_config(path)
        assert loaded.setup_complete is True

    def test_roundtrip_preserves_locale(self, tmp_path: Path):
        config = AnimaWorksConfig(locale="en")
        path = tmp_path / "config.json"
        save_config(config, path)
        invalidate_cache()

        loaded = load_config(path)
        assert loaded.locale == "en"

    def test_roundtrip_both_fields(self, tmp_path: Path):
        config = AnimaWorksConfig(setup_complete=True, locale="en")
        path = tmp_path / "config.json"
        save_config(config, path)
        invalidate_cache()

        loaded = load_config(path)
        assert loaded.setup_complete is True
        assert loaded.locale == "en"

    def test_json_file_contains_fields(self, tmp_path: Path):
        config = AnimaWorksConfig(setup_complete=True, locale="en")
        path = tmp_path / "config.json"
        save_config(config, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["setup_complete"] is True
        assert raw["locale"] == "en"

    def test_defaults_roundtrip(self, tmp_path: Path):
        config = AnimaWorksConfig()
        path = tmp_path / "config.json"
        save_config(config, path)
        invalidate_cache()

        loaded = load_config(path)
        assert loaded.setup_complete is False
        assert loaded.locale == "ja"
