# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Issue A + B: Verify safe defaults for PR #149 features.

- auto_response defaults to False
- Usage Governor defaults to disabled
- Ollama keep_alive defaults to empty string (Ollama native default)
- Ollama total_timeout defaults to 0 (unlimited)
- Heartbeat current_state_max_chars defaults to 0 (disabled)
"""

from core.config.schemas import (
    ExternalMessagingChannelConfig,
    HeartbeatConfig,
    ServerConfig,
    UsageGovernorConfig,
)


class TestAutoResponseDefault:
    def test_default_false(self):
        cfg = ExternalMessagingChannelConfig()
        assert cfg.auto_response is False

    def test_explicit_true(self):
        cfg = ExternalMessagingChannelConfig(auto_response=True)
        assert cfg.auto_response is True


class TestUsageGovernorDefault:
    def test_default_disabled(self):
        cfg = UsageGovernorConfig()
        assert cfg.enabled is False

    def test_explicit_enabled(self):
        cfg = UsageGovernorConfig(enabled=True)
        assert cfg.enabled is True

    def test_nested_in_server_config(self):
        cfg = ServerConfig()
        assert cfg.usage_governor.enabled is False


class TestOllamaDefaults:
    def test_keep_alive_empty(self):
        cfg = ServerConfig()
        assert cfg.ollama_keep_alive == ""

    def test_total_timeout_zero(self):
        cfg = ServerConfig()
        assert cfg.ollama_total_timeout == 0


class TestCurrentStateMaxCharsDefault:
    def test_default_zero(self):
        cfg = HeartbeatConfig()
        assert cfg.current_state_max_chars == 0

    def test_explicit_value(self):
        cfg = HeartbeatConfig(current_state_max_chars=3000)
        assert cfg.current_state_max_chars == 3000
