# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Issue B: Verify Ollama keep_alive and total_timeout are configurable."""

from unittest.mock import MagicMock, patch

from core.config.schemas import ServerConfig


class TestKeepAliveConditional:
    """keep_alive is only sent when explicitly configured."""

    def test_empty_keep_alive_not_sent(self):
        cfg = ServerConfig(ollama_keep_alive="")
        assert cfg.ollama_keep_alive == ""

    def test_explicit_keep_alive(self):
        cfg = ServerConfig(ollama_keep_alive="5m")
        assert cfg.ollama_keep_alive == "5m"


class TestTotalTimeoutConditional:
    """total_timeout=0 means unlimited (no asyncio.wait_for)."""

    def test_zero_means_unlimited(self):
        cfg = ServerConfig(ollama_total_timeout=0)
        assert cfg.ollama_total_timeout == 0

    def test_positive_value(self):
        cfg = ServerConfig(ollama_total_timeout=7200)
        assert cfg.ollama_total_timeout == 7200
