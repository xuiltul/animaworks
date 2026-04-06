# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Discord webhook manager (message splitting, TTL, persistence)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.discord_webhooks import (
    DISCORD_MESSAGE_LIMIT,
    DiscordWebhookManager,
    _split_message,
)


# ── _split_message ───────────────────────────────────────


class TestSplitMessage:
    def test_short_message_no_split(self):
        assert _split_message("hello") == ["hello"]

    def test_exact_limit_no_split(self):
        msg = "x" * DISCORD_MESSAGE_LIMIT
        assert _split_message(msg) == [msg]

    def test_over_limit_splits(self):
        msg = "x" * (DISCORD_MESSAGE_LIMIT + 100)
        chunks = _split_message(msg)
        assert len(chunks) == 2
        assert all(len(c) <= DISCORD_MESSAGE_LIMIT for c in chunks)
        assert "".join(chunks) == msg

    def test_split_at_newline(self):
        line = "a" * 1000
        msg = f"{line}\n{line}\n{line}"
        chunks = _split_message(msg)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= DISCORD_MESSAGE_LIMIT

    def test_empty_message(self):
        assert _split_message("") == [""]

    def test_multi_split(self):
        msg = "x" * (DISCORD_MESSAGE_LIMIT * 3 + 50)
        chunks = _split_message(msg)
        assert len(chunks) == 4
        assert "".join(chunks) == msg


# ── DiscordWebhookManager ────────────────────────────────


class TestDiscordWebhookManager:
    @pytest.fixture
    def mgr(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DiscordWebhookManager:
        monkeypatch.setattr("core.discord_webhooks.get_data_dir", lambda: tmp_path)
        return DiscordWebhookManager()

    def test_thread_map_ttl_expiry(self, mgr: DiscordWebhookManager):
        mgr.record_thread_mapping("msg1", "sakura")
        assert mgr.lookup_thread_anima("msg1") == "sakura"

        # Simulate expiry by backdating the timestamp
        mgr._thread_map["msg1"]["ts"] = time.time() - 8 * 86400
        assert mgr.lookup_thread_anima("msg1") is None

    def test_thread_map_unknown_id(self, mgr: DiscordWebhookManager):
        assert mgr.lookup_thread_anima("nonexistent") is None

    def test_persistence_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("core.discord_webhooks.get_data_dir", lambda: tmp_path)

        mgr1 = DiscordWebhookManager()
        mgr1._webhooks["ch1"] = {"id": "wh1", "token": "tok1"}
        mgr1._persist()

        mgr2 = DiscordWebhookManager()
        assert mgr2._webhooks.get("ch1") == {"id": "wh1", "token": "tok1"}

    def test_persistence_thread_map_prunes_expired(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("core.discord_webhooks.get_data_dir", lambda: tmp_path)

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - 30 * 86400
        data = {
            "fresh": {"anima": "sakura", "ts": time.time()},
            "stale": {"anima": "kotoha", "ts": old_ts},
        }
        (run_dir / "discord_thread_map.json").write_text(json.dumps(data))

        mgr = DiscordWebhookManager()
        assert "fresh" in mgr._thread_map
        assert "stale" not in mgr._thread_map

    @patch("core.discord_webhooks.get_credential", return_value="test-token")
    @patch("core.discord_webhooks.DiscordClient")
    def test_send_as_anima_basic(self, MockClient, _mock_cred, mgr: DiscordWebhookManager):
        mock_client = MockClient.return_value
        mock_client.list_webhooks.return_value = [
            {"name": "AnimaWorks", "id": "wh1", "token": "tok1"},
        ]
        mock_client.execute_webhook.return_value = {"id": "msg123"}

        msg_id = mgr.send_as_anima("ch1", "sakura", "hello")
        assert msg_id == "msg123"
        mock_client.execute_webhook.assert_called_once()
        _, kwargs = mock_client.execute_webhook.call_args
        assert kwargs.get("username") == "sakura"
