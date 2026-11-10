"""Tests for moving plaintext config credentials into the vault on save."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from core.config.io import (
    PlaintextCredential,
    find_plaintext_credentials,
    invalidate_cache,
    load_config,
    save_config,
    update_config,
)
from core.config.schemas import AnimaWorksConfig, CredentialConfig
from core.config.vault import VaultManager


@pytest.fixture(autouse=True)
def _clear_config_cache():
    invalidate_cache()
    yield
    invalidate_cache()


def _generate_vault_key(data_dir: Path) -> VaultManager:
    vault = VaultManager(data_dir)
    assert vault.generate_key()
    return vault


def _write_raw_config(data_dir: Path, payload: dict) -> Path:
    path = data_dir / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    invalidate_cache()
    return path


def _raw_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestSaveConfigMovesPlaintextCredentials:
    def test_save_stores_api_key_in_vault_and_references_it(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        config_path = tmp_path / "config.json"
        secret = "sk-live-plaintext-secret"
        config = AnimaWorksConfig(credentials={"gpu40-direct": CredentialConfig(api_key=secret)})

        save_config(config, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["gpu40-direct"]["api_key"] == {"$vault": "GPU40_DIRECT_API_KEY"}
        assert vault.get("shared", "GPU40_DIRECT_API_KEY") == secret
        assert secret not in config_path.read_text(encoding="utf-8")

        invalidate_cache()
        assert load_config(config_path).credentials["gpu40-direct"].api_key == secret

    def test_update_stores_api_key_in_vault_and_references_it(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        config_path = _write_raw_config(
            tmp_path,
            {"credentials": {"openai": {"type": "api_key", "api_key": "old-placeholder"}}},
        )
        secret = "sk-update-plaintext-secret"

        def apply_update(config: AnimaWorksConfig) -> AnimaWorksConfig:
            config.credentials["openai"] = CredentialConfig(type="api_key", api_key=secret)
            return config

        update_config(apply_update, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["openai"]["api_key"] == {"$vault": "OPENAI_API_KEY"}
        assert vault.get("shared", "OPENAI_API_KEY") == secret
        assert secret not in config_path.read_text(encoding="utf-8")

        invalidate_cache()
        assert load_config(config_path).credentials["openai"].api_key == secret

    def test_keys_are_moved_but_non_secret_fields_are_left_alone(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        config_path = tmp_path / "config.json"
        config = AnimaWorksConfig(
            credentials={
                "bedrock": CredentialConfig(
                    keys={
                        "aws_secret_access_key": "plaintext-aws-secret",
                        "aws_access_key_id": "plaintext-aws-id",
                        "region": "us-east-1",
                    },
                    base_url="https://bedrock.example.invalid",
                )
            }
        )

        save_config(config, config_path)

        raw = _raw_config(config_path)
        keys = raw["credentials"]["bedrock"]["keys"]
        assert keys["aws_secret_access_key"] == {"$vault": "BEDROCK_AWS_SECRET_ACCESS_KEY"}
        assert keys["aws_access_key_id"] == {"$vault": "BEDROCK_AWS_ACCESS_KEY_ID"}
        assert keys["region"] == "us-east-1"
        assert raw["credentials"]["bedrock"]["base_url"] == "https://bedrock.example.invalid"
        assert vault.get("shared", "BEDROCK_AWS_SECRET_ACCESS_KEY") == "plaintext-aws-secret"
        assert vault.get("shared", "BEDROCK_AWS_ACCESS_KEY_ID") == "plaintext-aws-id"

        text = config_path.read_text(encoding="utf-8")
        assert "plaintext-aws-secret" not in text
        assert "plaintext-aws-id" not in text

    def test_existing_key_with_same_value_is_reused(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        vault.store("shared", "BEDROCK_API_KEY", "shared-value")
        config_path = tmp_path / "config.json"
        config = AnimaWorksConfig(credentials={"bedrock": CredentialConfig(api_key="shared-value")})

        save_config(config, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["bedrock"]["api_key"] == {"$vault": "BEDROCK_API_KEY"}
        assert vault.get("shared", "BEDROCK_API_KEY") == "shared-value"

    def test_existing_key_with_different_value_uses_config_namespace(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        vault.store("shared", "BEDROCK_API_KEY", "other-value")
        config_path = tmp_path / "config.json"
        config = AnimaWorksConfig(credentials={"bedrock": CredentialConfig(api_key="new-value")})

        save_config(config, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["bedrock"]["api_key"] == {"$vault": "CONFIG_BEDROCK_API_KEY"}
        assert vault.get("shared", "CONFIG_BEDROCK_API_KEY") == "new-value"
        assert vault.get("shared", "BEDROCK_API_KEY") == "other-value"

    def test_existing_reference_name_is_not_changed(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        vault.store("shared", "AZURE_API_KEY", "azure-value")
        config_path = _write_raw_config(
            tmp_path,
            {"credentials": {"azure": {"api_key": {"$vault": "AZURE_API_KEY"}}}},
        )
        config = load_config(config_path)
        config.server.session_ttl_days = 91

        save_config(config, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["azure"]["api_key"] == {"$vault": "AZURE_API_KEY"}
        assert vault.get("shared", "AZURE_API_KEY") == "azure-value"


class TestVaultUnavailable:
    def test_plaintext_is_kept_when_no_vault_key_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr("core.config.io._vault_unavailable_warned", False)
        config_path = tmp_path / "config.json"
        secret = "sk-no-vault-secret"
        config = AnimaWorksConfig(credentials={"anthropic": CredentialConfig(api_key=secret)})

        with caplog.at_level(logging.WARNING, logger="animaworks.config"):
            save_config(config, config_path)
            save_config(config, config_path)

        raw = _raw_config(config_path)
        assert raw["credentials"]["anthropic"]["api_key"] == secret
        assert secret in config_path.read_text(encoding="utf-8")
        assert not (tmp_path / "vault.json").exists()
        assert caplog.text.count("Vault is unavailable") == 1


class TestFindPlaintextCredentials:
    def test_returns_paths_and_key_names_without_values(self, tmp_path: Path) -> None:
        _generate_vault_key(tmp_path)
        raw = {
            "credentials": {
                "gpu40-direct": {"api_key": "finding-secret", "base_url": "https://example.invalid"},
                "bedrock": {
                    "keys": {
                        "aws_secret_access_key": "finding-secret-two",
                        "region": "us-east-1",
                    }
                },
            }
        }

        found = find_plaintext_credentials(raw, tmp_path)

        assert found == [
            PlaintextCredential(path="credentials.gpu40-direct.api_key", vault_key="GPU40_DIRECT_API_KEY"),
            PlaintextCredential(
                path="credentials.bedrock.keys.aws_secret_access_key",
                vault_key="BEDROCK_AWS_SECRET_ACCESS_KEY",
            ),
        ]
        assert "finding-secret" not in repr(found)
        assert "finding-secret-two" not in repr(found)

    def test_planned_key_respects_existing_vault_values(self, tmp_path: Path) -> None:
        vault = _generate_vault_key(tmp_path)
        vault.store("shared", "BEDROCK_API_KEY", "other-value")
        raw = {"credentials": {"bedrock": {"api_key": "new-value"}}}

        found = find_plaintext_credentials(raw, tmp_path)

        assert found == [PlaintextCredential(path="credentials.bedrock.api_key", vault_key="CONFIG_BEDROCK_API_KEY")]


class TestRoutePaths:
    async def test_anthropic_auth_route_stores_key_in_vault(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi import FastAPI

        from server.routes.config_routes import create_config_router

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        _generate_vault_key(tmp_path)
        config_path = _write_raw_config(
            tmp_path,
            {"credentials": {"anthropic": {"type": "api_key", "api_key": ""}}},
        )
        secret = "sk-ant-route-secret"
        app = FastAPI()
        app.include_router(create_config_router(), prefix="/api")
        transport = ASGITransport(app=app)

        with patch("server.routes.config_routes.is_claude_code_available", return_value=False):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/settings/anthropic-auth",
                    json={"auth_mode": "api_key", "api_key": secret},
                )

        assert resp.status_code == 200
        text = config_path.read_text(encoding="utf-8")
        assert secret not in text
        raw = json.loads(text)
        assert raw["credentials"]["anthropic"]["api_key"] == {"$vault": "ANTHROPIC_API_KEY"}
        assert VaultManager(tmp_path).get("shared", "ANTHROPIC_API_KEY") == secret

    async def test_setup_complete_route_stores_key_in_vault(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi import FastAPI

        from server.routes.setup import create_setup_router

        monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
        _generate_vault_key(tmp_path)
        secret = "sk-ant-setup-secret"
        app = FastAPI()
        app.state.setup_complete = False
        app.state.animas_dir = tmp_path / "animas"
        app.state.supervisor = AsyncMock()
        app.include_router(create_setup_router())
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/setup/complete",
                json={
                    "locale": "en",
                    "provider": "anthropic",
                    "credentials": {"anthropic": {"api_key": secret}},
                },
            )

        assert resp.status_code == 200
        config_path = tmp_path / "config.json"
        text = config_path.read_text(encoding="utf-8")
        assert secret not in text
        raw = json.loads(text)
        assert raw["credentials"]["anthropic"]["api_key"] == {"$vault": "ANTHROPIC_API_KEY"}
        assert VaultManager(tmp_path).get("shared", "ANTHROPIC_API_KEY") == secret
