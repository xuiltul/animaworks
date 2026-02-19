"""Unit tests for CredentialConfig with type and keys fields."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.config.models import CredentialConfig


class TestCredentialConfigType:
    def test_default_type(self):
        cc = CredentialConfig()
        assert cc.type == "api_key"

    def test_custom_type(self):
        cc = CredentialConfig(type="api_token")
        assert cc.type == "api_token"

    def test_bearer_token_type(self):
        cc = CredentialConfig(type="bearer_token")
        assert cc.type == "bearer_token"


class TestCredentialConfigKeys:
    def test_default_empty_keys(self):
        cc = CredentialConfig()
        assert cc.keys == {}

    def test_single_extra_key(self):
        cc = CredentialConfig(keys={"client_secret": "sec-123"})
        assert cc.keys["client_secret"] == "sec-123"

    def test_multiple_keys(self):
        cc = CredentialConfig(keys={
            "client_id": "id-123",
            "client_secret": "sec-456",
        })
        assert cc.keys["client_id"] == "id-123"
        assert cc.keys["client_secret"] == "sec-456"

    def test_api_key_and_keys_coexist(self):
        cc = CredentialConfig(
            api_key="primary-key",
            keys={"secondary": "sec-key"},
        )
        assert cc.api_key == "primary-key"
        assert cc.keys["secondary"] == "sec-key"


class TestCredentialConfigBackwardCompat:
    def test_api_key_still_works(self):
        cc = CredentialConfig(api_key="sk-123")
        assert cc.api_key == "sk-123"

    def test_base_url_still_works(self):
        cc = CredentialConfig(base_url="http://localhost:8080")
        assert cc.base_url == "http://localhost:8080"

    def test_full_llm_config(self):
        cc = CredentialConfig(
            type="api_key",
            api_key="sk-ant-xxx",
            base_url="https://api.anthropic.com",
        )
        assert cc.type == "api_key"
        assert cc.api_key == "sk-ant-xxx"
        assert cc.base_url == "https://api.anthropic.com"
        assert cc.keys == {}


class TestCredentialConfigSerialization:
    def test_round_trip(self):
        original = CredentialConfig(
            type="api_token",
            api_key="cwt-xxx",
            keys={"extra": "val"},
            base_url=None,
        )
        data = original.model_dump()
        restored = CredentialConfig.model_validate(data)
        assert restored == original

    def test_json_round_trip(self):
        original = CredentialConfig(
            type="bearer_token",
            api_key="token-xxx",
            keys={"refresh_token": "refresh-xxx"},
        )
        json_str = original.model_dump_json()
        restored = CredentialConfig.model_validate_json(json_str)
        assert restored == original
