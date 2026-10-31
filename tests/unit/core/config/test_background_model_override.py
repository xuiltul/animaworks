"""Tests for background_model override feature in core/config/models.py.

Covers:
- HeartbeatConfig.default_model field
- AnimaDefaults.background_model/background_credential fields
- _load_status_json mapping for background_model/background_credential
- update_status_model with background_model/background_credential
- resolve_anima_config propagation of background_model
- load_model_config propagation of background_model
"""
from __future__ import annotations

import json
from pathlib import Path

from core.config.models import (
    AnimaDefaults,
    AnimaWorksConfig,
    CredentialConfig,
    HeartbeatConfig,
    _load_status_json,
    resolve_anima_config,
    update_status_model,
)

# ── HeartbeatConfig.default_model ─────────────────────────────


class TestHeartbeatConfigDefaultModel:
    def test_default_model_defaults_to_none(self):
        hc = HeartbeatConfig()
        assert hc.default_model is None

    def test_default_model_accepts_string(self):
        hc = HeartbeatConfig(default_model="claude-sonnet-4-6")
        assert hc.default_model == "claude-sonnet-4-6"

    def test_config_roundtrip(self):
        config = AnimaWorksConfig(
            heartbeat=HeartbeatConfig(default_model="openai/gpt-4.1-mini"),
        )
        data = config.model_dump(mode="json")
        restored = AnimaWorksConfig.model_validate(data)
        assert restored.heartbeat.default_model == "openai/gpt-4.1-mini"


# ── AnimaDefaults.background_model ────────────────────────────


class TestAnimaDefaultsBackgroundModel:
    def test_background_model_defaults_to_none(self):
        ad = AnimaDefaults()
        assert ad.background_model is None
        assert ad.background_credential is None

    def test_background_model_accepts_values(self):
        ad = AnimaDefaults(
            background_model="claude-sonnet-4-6",
            background_credential="anthropic",
        )
        assert ad.background_model == "claude-sonnet-4-6"
        assert ad.background_credential == "anthropic"


# ── _load_status_json ─────────────────────────────────────────


class TestLoadStatusJsonBackgroundModel:
    def test_reads_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        status = {
            "model": "claude-opus-4-6",
            "background_model": "claude-sonnet-4-6",
            "background_credential": "anthropic",
            "enabled": True,
        }
        (anima_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )
        result = _load_status_json(anima_dir)
        assert result["background_model"] == "claude-sonnet-4-6"
        assert result["background_credential"] == "anthropic"

    def test_skips_empty_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        status = {"model": "claude-opus-4-6", "background_model": "", "enabled": True}
        (anima_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )
        result = _load_status_json(anima_dir)
        assert "background_model" not in result

    def test_skips_null_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        status = {"model": "claude-opus-4-6", "background_model": None, "enabled": True}
        (anima_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )
        result = _load_status_json(anima_dir)
        assert "background_model" not in result

    def test_missing_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "test-anima"
        anima_dir.mkdir()
        status = {"model": "claude-opus-4-6", "enabled": True}
        (anima_dir / "status.json").write_text(
            json.dumps(status), encoding="utf-8",
        )
        result = _load_status_json(anima_dir)
        assert "background_model" not in result


# ── update_status_model ───────────────────────────────────────


class TestUpdateStatusModelBackground:
    def _write_status(self, anima_dir: Path, data: dict) -> Path:
        anima_dir.mkdir(parents=True, exist_ok=True)
        p = anima_dir / "status.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def test_set_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        self._write_status(anima_dir, {"model": "claude-opus-4-6"})
        update_status_model(anima_dir, background_model="claude-sonnet-4-6")
        data = json.loads((anima_dir / "status.json").read_text())
        assert data["background_model"] == "claude-sonnet-4-6"
        assert data["model"] == "claude-opus-4-6"

    def test_clear_background_model(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        self._write_status(anima_dir, {
            "model": "claude-opus-4-6",
            "background_model": "claude-sonnet-4-6",
        })
        update_status_model(anima_dir, background_model="")
        data = json.loads((anima_dir / "status.json").read_text())
        assert "background_model" not in data

    def test_set_background_credential(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        self._write_status(anima_dir, {"model": "claude-opus-4-6"})
        update_status_model(
            anima_dir,
            background_model="openai/gpt-4.1-mini",
            background_credential="azure",
        )
        data = json.loads((anima_dir / "status.json").read_text())
        assert data["background_model"] == "openai/gpt-4.1-mini"
        assert data["background_credential"] == "azure"

    def test_clear_background_credential(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        self._write_status(anima_dir, {
            "model": "claude-opus-4-6",
            "background_credential": "azure",
        })
        update_status_model(anima_dir, background_credential="")
        data = json.loads((anima_dir / "status.json").read_text())
        assert "background_credential" not in data

    def test_sentinel_leaves_background_model_unchanged(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        self._write_status(anima_dir, {
            "model": "claude-opus-4-6",
            "background_model": "claude-sonnet-4-6",
        })
        update_status_model(anima_dir, model="claude-haiku-4-5-20251001")
        data = json.loads((anima_dir / "status.json").read_text())
        assert data["model"] == "claude-haiku-4-5-20251001"
        assert data["background_model"] == "claude-sonnet-4-6"


# ── resolve_anima_config ──────────────────────────────────────


class TestResolveAnimaConfigBackgroundModel:
    def test_background_model_from_status_json(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({
            "model": "claude-opus-4-6",
            "background_model": "claude-sonnet-4-6",
        }), encoding="utf-8")

        config = AnimaWorksConfig()
        resolved, _cred = resolve_anima_config(config, "test", anima_dir)
        assert resolved.background_model == "claude-sonnet-4-6"

    def test_background_model_fallback_to_defaults(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({
            "model": "claude-opus-4-6",
        }), encoding="utf-8")

        config = AnimaWorksConfig(
            anima_defaults=AnimaDefaults(background_model="openai/gpt-4.1-mini"),
        )
        resolved, _cred = resolve_anima_config(config, "test", anima_dir)
        assert resolved.background_model == "openai/gpt-4.1-mini"

    def test_background_model_none_when_unset(self, tmp_path: Path):
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(json.dumps({
            "model": "claude-opus-4-6",
        }), encoding="utf-8")

        config = AnimaWorksConfig()
        resolved, _cred = resolve_anima_config(config, "test", anima_dir)
        assert resolved.background_model is None


# ── ModelConfig.background_model ──────────────────────────────


class TestModelConfigBackgroundModel:
    def test_schema_has_background_fields(self):
        from core.schemas import ModelConfig

        mc = ModelConfig(
            background_model="claude-sonnet-4-6",
            background_credential="azure",
        )
        assert mc.background_model == "claude-sonnet-4-6"
        assert mc.background_credential == "azure"

    def test_defaults_to_none(self):
        from core.schemas import ModelConfig

        mc = ModelConfig()
        assert mc.background_model is None
        assert mc.background_credential is None


# ── ConfigReader.read_model_config ────────────────────────────


class TestConfigReaderBackgroundModel:
    """Verify ConfigReader propagates background_model/background_credential."""

    def test_config_reader_passes_background_fields(self, tmp_path: Path):
        from unittest.mock import patch

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        from core.memory.config_reader import ConfigReader

        config_json = tmp_path / "config.json"
        config_json.write_text("{}")

        with patch("core.config.models.get_config_path", return_value=config_json), \
             patch("core.config.models.load_config", return_value=AnimaWorksConfig()), \
             patch("core.config.models.resolve_anima_config") as mock_resolve, \
             patch("core.config.models.resolve_execution_mode", return_value="A"):
            resolved = AnimaDefaults(
                model="claude-opus-4-6",
                background_model="claude-sonnet-4-6",
                background_credential="azure",
            )
            mock_resolve.return_value = (resolved, CredentialConfig(api_key="key1"))

            reader = ConfigReader(anima_dir)
            mc = reader.read_model_config()

        assert mc.background_model == "claude-sonnet-4-6"
        assert mc.background_credential == "azure"

    def test_config_reader_none_when_unset(self, tmp_path: Path):
        from unittest.mock import patch

        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)

        from core.memory.config_reader import ConfigReader

        config_json = tmp_path / "config.json"
        config_json.write_text("{}")

        with patch("core.config.models.get_config_path", return_value=config_json), \
             patch("core.config.models.load_config", return_value=AnimaWorksConfig()), \
             patch("core.config.models.resolve_anima_config") as mock_resolve, \
             patch("core.config.models.resolve_execution_mode", return_value="A"):
            resolved = AnimaDefaults(model="claude-opus-4-6")
            mock_resolve.return_value = (resolved, CredentialConfig(api_key="key1"))

            reader = ConfigReader(anima_dir)
            mc = reader.read_model_config()

        assert mc.background_model is None
        assert mc.background_credential is None
