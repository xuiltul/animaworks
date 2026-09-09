"""E2E tests for background_model override feature.

Tests the full lifecycle:
- status.json with background_model → resolve_anima_config → ModelConfig
- update_status_model set/clear cycle
- anima_factory role defaults merge
- HeartbeatConfig global default propagation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config.models import (
    AnimaWorksConfig,
    _load_status_json,
    resolve_anima_config,
    update_status_model,
)
from core.schemas import ModelConfig

# ── Full lifecycle: set → resolve → build ModelConfig ─────────


class TestBackgroundModelLifecycle:
    """End-to-end: write status.json → resolve config → get ModelConfig."""

    def _make_anima_dir(self, tmp_path: Path, status: dict) -> Path:
        anima_dir = tmp_path / "animas" / "test"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        return anima_dir

    def test_full_lifecycle_with_background_model(self, tmp_path: Path):
        """Set background_model in status.json, resolve, and verify ModelConfig."""
        anima_dir = self._make_anima_dir(
            tmp_path,
            {
                "model": "claude-opus-4-6",
                "background_model": "claude-sonnet-4-6",
                "enabled": True,
            },
        )

        config = AnimaWorksConfig()
        resolved, _cred = resolve_anima_config(config, "test", anima_dir)

        assert resolved.model == "claude-opus-4-6"
        assert resolved.background_model == "claude-sonnet-4-6"

    def test_update_then_resolve_cycle(self, tmp_path: Path):
        """update_status_model → _load_status_json → resolve_anima_config round-trip."""
        anima_dir = self._make_anima_dir(
            tmp_path,
            {
                "model": "claude-opus-4-6",
                "enabled": True,
            },
        )

        update_status_model(anima_dir, background_model="openai/gpt-4.1-mini")

        raw = _load_status_json(anima_dir)
        assert raw["background_model"] == "openai/gpt-4.1-mini"

        config = AnimaWorksConfig()
        resolved, _ = resolve_anima_config(config, "test", anima_dir)
        assert resolved.background_model == "openai/gpt-4.1-mini"

    def test_clear_then_resolve_cycle(self, tmp_path: Path):
        """Set then clear background_model, verify it's gone."""
        anima_dir = self._make_anima_dir(
            tmp_path,
            {
                "model": "claude-opus-4-6",
                "background_model": "claude-sonnet-4-6",
                "enabled": True,
            },
        )

        update_status_model(anima_dir, background_model="")

        raw = _load_status_json(anima_dir)
        assert "background_model" not in raw

        config = AnimaWorksConfig()
        resolved, _ = resolve_anima_config(config, "test", anima_dir)
        assert resolved.background_model is None


# ── Global default propagation ────────────────────────────────


class TestGlobalDefaultPropagation:
    """Test heartbeat.default_model as global fallback."""

    def test_heartbeat_default_model_used_by_resolve_background_config(self):
        """When per-anima is unset, _resolve_background_config uses global."""
        from core._anima_heartbeat import HeartbeatMixin

        mc = ModelConfig(model="claude-opus-4-6")

        class FakeMixin(HeartbeatMixin):
            pass

        mixin = FakeMixin.__new__(FakeMixin)
        mixin.agent = MagicMock()
        mixin.agent.model_config = mc

        with patch("core.config.models.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.heartbeat.default_model = "claude-sonnet-4-6"
            mock_config.return_value = mock_cfg

            result = mixin._resolve_background_config()

        assert result is not None
        assert result.model == "claude-sonnet-4-6"

    @pytest.mark.parametrize("credential_available", [False, True])
    def test_per_anima_overrides_global(self, credential_available):
        """Cross-provider priority requires its own explicitly configured auth."""
        from core._anima_heartbeat import HeartbeatMixin
        from core.config.models import CredentialConfig

        mc = ModelConfig(
            model="claude-opus-4-6",
            background_model="openai/gpt-4.1",
            api_key="synthetic-main-provider-key",
        )

        class FakeMixin(HeartbeatMixin):
            pass

        mixin = FakeMixin.__new__(FakeMixin)
        mixin.agent = MagicMock()
        mixin.agent.model_config = mc

        config = AnimaWorksConfig(
            credentials={"openai": CredentialConfig(api_key="synthetic-openai-key")} if credential_available else {}
        )
        config.heartbeat.default_model = "claude-sonnet-4-6"
        with patch("core.config.models.load_config", return_value=config):
            if not credential_available:
                with pytest.raises(ValueError, match="No credential configured"):
                    mixin._resolve_background_config()
                assert mixin.agent.model_config.api_key == "synthetic-main-provider-key"
                return
            result = mixin._resolve_background_config()

        assert result is not None
        assert result.model == "openai/gpt-4.1"
        assert result.api_key == "synthetic-openai-key"


# ── Role defaults merge ──────────────────────────────────────


class TestRoleDefaultsMerge:
    """Test that anima_factory merges background_model from role defaults."""

    def test_engineer_defaults_include_background_model(self):
        defaults_path = (
            Path(__file__).resolve().parents[3] / "templates" / "_shared" / "roles" / "engineer" / "defaults.json"
        )
        if not defaults_path.exists():
            pytest.skip("Template not found in test environment")

        data = json.loads(defaults_path.read_text(encoding="utf-8"))
        assert "background_model" in data
        assert data["background_model"] == "claude-sonnet-4-6"

    def test_manager_defaults_include_background_model(self):
        defaults_path = (
            Path(__file__).resolve().parents[3] / "templates" / "_shared" / "roles" / "manager" / "defaults.json"
        )
        if not defaults_path.exists():
            pytest.skip("Template not found in test environment")

        data = json.loads(defaults_path.read_text(encoding="utf-8"))
        assert "background_model" in data
        assert data["background_model"] == "claude-sonnet-4-6"

    def test_general_defaults_no_background_model(self):
        defaults_path = (
            Path(__file__).resolve().parents[3] / "templates" / "_shared" / "roles" / "general" / "defaults.json"
        )
        if not defaults_path.exists():
            pytest.skip("Template not found in test environment")

        data = json.loads(defaults_path.read_text(encoding="utf-8"))
        assert "background_model" not in data


# ── Cross-provider credential ─────────────────────────────────


class TestCrossProviderCredential:
    """Test background_credential resolves a different provider's creds."""

    def test_credential_applied_to_background_config(self):
        from core._anima_heartbeat import HeartbeatMixin
        from core.config.models import CredentialConfig

        mc = ModelConfig(
            model="claude-opus-4-6",
            background_model="azure/gpt-4.1-mini",
            background_credential="azure",
        )

        class FakeMixin(HeartbeatMixin):
            pass

        mixin = FakeMixin.__new__(FakeMixin)
        mixin.agent = MagicMock()
        mixin.agent.model_config = mc

        with patch("core.config.models.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.credentials = {
                "azure": CredentialConfig(
                    api_key="az-key-123",
                    base_url="https://myresource.openai.azure.com",
                    keys={"api_version": "2024-12-01-preview"},
                ),
            }
            mock_config.return_value = mock_cfg

            result = mixin._resolve_background_config()

        assert result.model == "azure/gpt-4.1-mini"
        assert result.api_key == "az-key-123"
        assert result.api_base_url == "https://myresource.openai.azure.com"
        assert result.extra_keys["api_version"] == "2024-12-01-preview"

    def test_missing_credential_rejects_cross_provider_reuse(self):
        """An unknown background credential must not leak the main provider key."""
        from core._anima_heartbeat import HeartbeatMixin

        mc = ModelConfig(
            model="claude-opus-4-6",
            background_model="openai/gpt-4.1",
            background_credential="nonexistent",
            api_key="main-key",
        )

        class FakeMixin(HeartbeatMixin):
            pass

        mixin = FakeMixin.__new__(FakeMixin)
        mixin.agent = MagicMock()
        mixin.agent.model_config = mc

        with patch("core.config.models.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.credentials = {}
            mock_config.return_value = mock_cfg

            with pytest.raises(ValueError, match="No credential configured"):
                mixin._resolve_background_config()
        assert mixin.agent.model_config.api_key == "main-key"


# ── Inbox uses _resolve_background_config via MRO ─────────────


class TestInboxBackgroundModelResolution:
    """Verify InboxMixin accesses _resolve_background_config through MRO."""

    def test_inbox_mixin_resolves_background_via_heartbeat_mixin(self):
        """InboxMixin + HeartbeatMixin MRO gives inbox access to background config."""
        from core._anima_heartbeat import HeartbeatMixin
        from core._anima_inbox import InboxMixin

        mc = ModelConfig(
            model="claude-opus-4-6",
            background_model="claude-sonnet-4-6",
        )

        class FakeAnima(HeartbeatMixin, InboxMixin):
            pass

        obj = FakeAnima.__new__(FakeAnima)
        obj.agent = MagicMock()
        obj.agent.model_config = mc

        result = obj._resolve_background_config()

        assert result is not None
        assert result.model == "claude-sonnet-4-6"

    def test_inbox_credential_resolution_e2e(self, tmp_path: Path):
        """Full E2E: status.json with background_model + credential → inbox resolution."""
        from core._anima_heartbeat import HeartbeatMixin
        from core._anima_inbox import InboxMixin
        from core.config.models import CredentialConfig

        anima_dir = tmp_path / "animas" / "test_inbox"
        anima_dir.mkdir(parents=True)
        (anima_dir / "status.json").write_text(
            json.dumps(
                {
                    "model": "claude-opus-4-6",
                    "background_model": "azure/gpt-4.1-mini",
                    "background_credential": "azure",
                    "enabled": True,
                }
            ),
            encoding="utf-8",
        )

        config = AnimaWorksConfig()
        resolved, _cred = resolve_anima_config(config, "test_inbox", anima_dir)
        assert resolved.background_model == "azure/gpt-4.1-mini"
        assert resolved.background_credential == "azure"

        mc = ModelConfig(
            model=resolved.model,
            background_model=resolved.background_model,
            background_credential=resolved.background_credential,
        )

        class FakeAnima(HeartbeatMixin, InboxMixin):
            pass

        obj = FakeAnima.__new__(FakeAnima)
        obj.agent = MagicMock()
        obj.agent.model_config = mc

        with patch("core.config.models.load_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.credentials = {
                "azure": CredentialConfig(
                    api_key="az-inbox-key",
                    base_url="https://inbox.openai.azure.com",
                    keys={"api_version": "2024-12-01"},
                ),
            }
            mock_config.return_value = mock_cfg

            bg = obj._resolve_background_config()

        assert bg is not None
        assert bg.model == "azure/gpt-4.1-mini"
        assert bg.api_key == "az-inbox-key"
        assert bg.api_base_url == "https://inbox.openai.azure.com"
