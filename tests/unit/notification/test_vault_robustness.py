"""Tests for notification channel vault/shared credential resolution and robustness fixes.

Covers:
- call_human._get_bot_token vault/shared lookup
- HumanNotifier partial failure logging
- ChatworkChannel room_id type safety
- TelegramChannel truncate-before-escape
- All channels: vault/shared credential resolution via _resolve_credential_with_vault
"""
from __future__ import annotations

import asyncio
import html
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.notification.notifier import HumanNotifier, NotificationChannel


# ── _resolve_credential_with_vault (base class) ─────────────


class _DummyChannel(NotificationChannel):
    """Minimal concrete channel for testing base-class helpers."""

    @property
    def channel_type(self) -> str:
        return "dummy"

    async def send(self, subject: str, body: str, priority: str = "normal",
                   *, anima_name: str = "") -> str:
        return "dummy: OK"


class TestResolveCredentialWithVault:
    def test_returns_env_var_first(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "from-env")
        ch = _DummyChannel({"token_env": "MY_TOKEN"})
        assert ch._resolve_credential_with_vault("token_env") == "from-env"

    def test_falls_back_to_per_anima_vault(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        ch = _DummyChannel({"token_env": "MY_TOKEN"})

        with patch("core.tools._base._lookup_vault_credential") as mock_vault, \
             patch("core.tools._base._lookup_shared_credentials", return_value=None):
            mock_vault.side_effect = lambda k: "vault-val" if k == "MY_TOKEN__alice" else None
            result = ch._resolve_credential_with_vault("token_env", anima_name="alice")
        assert result == "vault-val"

    def test_falls_back_to_generic_vault(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        ch = _DummyChannel({"token_env": "MY_TOKEN"})

        with patch("core.tools._base._lookup_vault_credential") as mock_vault, \
             patch("core.tools._base._lookup_shared_credentials", return_value=None):
            mock_vault.side_effect = lambda k: "generic-vault" if k == "MY_TOKEN" else None
            result = ch._resolve_credential_with_vault("token_env", anima_name="alice")
        assert result == "generic-vault"

    def test_falls_back_to_shared_credentials(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        ch = _DummyChannel({"token_env": "MY_TOKEN"})

        with patch("core.tools._base._lookup_vault_credential", return_value=None), \
             patch("core.tools._base._lookup_shared_credentials") as mock_shared:
            mock_shared.side_effect = lambda k: "shared-val" if k == "MY_TOKEN__alice" else None
            result = ch._resolve_credential_with_vault("token_env", anima_name="alice")
        assert result == "shared-val"

    def test_returns_empty_when_nothing_found(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        ch = _DummyChannel({"token_env": "MY_TOKEN"})

        with patch("core.tools._base._lookup_vault_credential", return_value=None), \
             patch("core.tools._base._lookup_shared_credentials", return_value=None):
            result = ch._resolve_credential_with_vault("token_env", anima_name="alice")
        assert result == ""

    def test_uses_fallback_env_when_config_key_missing(self, monkeypatch):
        monkeypatch.delenv("FALLBACK", raising=False)
        ch = _DummyChannel({})

        with patch("core.tools._base._lookup_vault_credential") as mock_vault, \
             patch("core.tools._base._lookup_shared_credentials", return_value=None):
            mock_vault.side_effect = lambda k: "fb-vault" if k == "FALLBACK" else None
            result = ch._resolve_credential_with_vault(
                "nonexistent_key", fallback_env="FALLBACK",
            )
        assert result == "fb-vault"

    def test_no_anima_name_skips_per_anima_lookup(self, monkeypatch):
        monkeypatch.delenv("MY_TOKEN", raising=False)
        ch = _DummyChannel({"token_env": "MY_TOKEN"})

        with patch("core.tools._base._lookup_vault_credential") as mock_vault, \
             patch("core.tools._base._lookup_shared_credentials", return_value=None):
            mock_vault.side_effect = lambda k: "generic" if k == "MY_TOKEN" else None
            result = ch._resolve_credential_with_vault("token_env", anima_name="")
        assert result == "generic"


# ── HumanNotifier partial failure logging ────────────────────


class _OkChannel(NotificationChannel):
    @property
    def channel_type(self) -> str:
        return "ok_ch"

    async def send(self, subject, body, priority="normal", *, anima_name=""):
        return "ok_ch: OK"


class _FailChannel(NotificationChannel):
    @property
    def channel_type(self) -> str:
        return "fail_ch"

    async def send(self, subject, body, priority="normal", *, anima_name=""):
        return "fail_ch: ERROR - broken"


class _RaiseChannel(NotificationChannel):
    @property
    def channel_type(self) -> str:
        return "raise_ch"

    async def send(self, subject, body, priority="normal", *, anima_name=""):
        raise ConnectionError("connection refused")


class TestNotifierPartialFailureLogging:
    @pytest.mark.asyncio
    async def test_all_success_logs_info(self, caplog):
        notifier = HumanNotifier([_OkChannel({}), _OkChannel({})])
        with caplog.at_level(logging.INFO, logger="animaworks.notification"):
            await notifier.notify("ok", "body")
        assert any("Human notification sent" in r.message for r in caplog.records)
        assert not any("partial failure" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_partial_failure_logs_warning(self, caplog):
        notifier = HumanNotifier([_OkChannel({}), _FailChannel({})])
        with caplog.at_level(logging.WARNING, logger="animaworks.notification"):
            results = await notifier.notify("test", "body")
        assert any("partial failure" in r.message for r in caplog.records)
        warning = [r for r in caplog.records if "partial failure" in r.message][0]
        assert "failed=1" in warning.message
        assert "success=1" in warning.message

    @pytest.mark.asyncio
    async def test_exception_channel_counted_as_failure(self, caplog):
        notifier = HumanNotifier([_OkChannel({}), _RaiseChannel({})])
        with caplog.at_level(logging.WARNING, logger="animaworks.notification"):
            results = await notifier.notify("test", "body")
        assert any("partial failure" in r.message for r in caplog.records)
        assert "ERROR" in results[1]

    @pytest.mark.asyncio
    async def test_all_fail_logs_warning(self, caplog):
        notifier = HumanNotifier([_FailChannel({}), _RaiseChannel({})])
        with caplog.at_level(logging.WARNING, logger="animaworks.notification"):
            await notifier.notify("test", "body")
        assert any("partial failure" in r.message for r in caplog.records)
        warning = [r for r in caplog.records if "partial failure" in r.message][0]
        assert "failed=2" in warning.message
        assert "success=0" in warning.message


# ── ChatworkChannel room_id type safety ──────────────────────


class TestChatworkRoomIdTypeSafety:
    @pytest.mark.asyncio
    async def test_room_id_int_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("CW_TOKEN", "test-token")
        from core.notification.channels.chatwork import ChatworkChannel

        ch = ChatworkChannel({"api_token_env": "CW_TOKEN", "room_id": 12345})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body")

        assert result == "chatwork: OK"

    @pytest.mark.asyncio
    async def test_room_id_str_int_works(self, monkeypatch):
        monkeypatch.setenv("CW_TOKEN", "test-token")
        from core.notification.channels.chatwork import ChatworkChannel

        ch = ChatworkChannel({"api_token_env": "CW_TOKEN", "room_id": "12345"})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body")

        assert result == "chatwork: OK"

    @pytest.mark.asyncio
    async def test_room_id_non_numeric_rejected(self, monkeypatch):
        monkeypatch.setenv("CW_TOKEN", "test-token")
        from core.notification.channels.chatwork import ChatworkChannel

        ch = ChatworkChannel({"api_token_env": "CW_TOKEN", "room_id": "abc"})
        result = await ch.send("Test", "body")
        assert "room_id must be numeric" in result


# ── TelegramChannel truncate-before-escape ───────────────────


class TestTelegramTruncateBeforeEscape:
    @pytest.mark.asyncio
    async def test_long_body_does_not_break_html_entities(self, monkeypatch):
        monkeypatch.setenv("TG_BOT_TOKEN", "test-token")
        from core.notification.channels.telegram import TelegramChannel

        ampersand_body = "A & B " * 1000

        ch = TelegramChannel({"bot_token_env": "TG_BOT_TOKEN", "chat_id": "123"})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        captured_payload: dict[str, Any] = {}

        async def capture_post(url, json=None):
            captured_payload.update(json or {})
            return mock_response

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Subject", ampersand_body)

        assert result == "telegram: OK"
        text = captured_payload["text"]
        assert len(text) <= 4096
        # Verify no broken HTML entities: &amp; should never be split
        assert "&am" not in text or "&amp;" in text
        # Every & should be properly escaped
        assert "& " not in text  # raw & should not appear

    @pytest.mark.asyncio
    async def test_short_body_unchanged(self, monkeypatch):
        monkeypatch.setenv("TG_BOT_TOKEN", "test-token")
        from core.notification.channels.telegram import TelegramChannel

        ch = TelegramChannel({"bot_token_env": "TG_BOT_TOKEN", "chat_id": "123"})

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        captured_payload: dict[str, Any] = {}

        async def capture_post(url, json=None):
            captured_payload.update(json or {})
            return mock_response

        with patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await ch.send("Hello & World", "Body & More")

        text = captured_payload["text"]
        assert "Hello &amp; World" in text
        assert "Body &amp; More" in text


# ── call_human._get_bot_token vault lookup ───────────────────


class TestCallHumanGetBotToken:
    def test_direct_token_returned(self):
        from core.tools.call_human import _get_bot_token

        assert _get_bot_token({"bot_token": "xoxb-direct"}) == "xoxb-direct"

    def test_env_var_returned(self, monkeypatch):
        from core.tools.call_human import _get_bot_token

        monkeypatch.setenv("MY_SLACK_TOKEN", "xoxb-env")
        assert _get_bot_token({"bot_token_env": "MY_SLACK_TOKEN"}) == "xoxb-env"

    def test_vault_per_anima_returned(self, monkeypatch, tmp_path):
        from core.tools.call_human import _get_bot_token

        anima_dir = tmp_path / "animas" / "alice"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

        with patch("core.tools._base._lookup_vault_credential") as mock_vault, \
             patch("core.tools._base._lookup_shared_credentials", return_value=None), \
             patch("core.tools._base.get_credential", side_effect=Exception("no cred")):
            mock_vault.side_effect = lambda k: "xoxb-vault" if k == "SLACK_BOT_TOKEN__alice" else None
            result = _get_bot_token({})

        assert result == "xoxb-vault"

    def test_vault_not_found_falls_back_to_get_credential(self, monkeypatch, tmp_path):
        from core.tools.call_human import _get_bot_token

        anima_dir = tmp_path / "animas" / "bob"
        anima_dir.mkdir(parents=True)
        monkeypatch.setenv("ANIMAWORKS_ANIMA_DIR", str(anima_dir))

        with patch("core.tools._base._lookup_vault_credential", return_value=None), \
             patch("core.tools._base._lookup_shared_credentials", return_value=None), \
             patch("core.tools._base.get_credential", return_value="xoxb-cred"):
            result = _get_bot_token({})

        assert result == "xoxb-cred"


# ── Channel vault credential integration ─────────────────────


class TestChannelVaultIntegration:
    """Verify that each non-Slack channel uses _resolve_credential_with_vault."""

    @pytest.mark.asyncio
    async def test_line_uses_vault(self, monkeypatch):
        monkeypatch.delenv("LINE_TOKEN", raising=False)
        from core.notification.channels.line import LineChannel

        ch = LineChannel({
            "channel_access_token_env": "LINE_TOKEN",
            "user_id": "U123",
        })

        with patch.object(ch, "_resolve_credential_with_vault", return_value="vault-token") as mock_rcv, \
             patch("core.notification.channels.line.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body", anima_name="alice")

        assert result == "line: OK"
        mock_rcv.assert_called_once_with(
            "channel_access_token_env",
            anima_name="alice",
            fallback_env="LINE_CHANNEL_ACCESS_TOKEN",
        )

    @pytest.mark.asyncio
    async def test_telegram_uses_vault(self, monkeypatch):
        monkeypatch.delenv("TG_TOKEN", raising=False)
        from core.notification.channels.telegram import TelegramChannel

        ch = TelegramChannel({
            "bot_token_env": "TG_TOKEN",
            "chat_id": "123",
        })

        with patch.object(ch, "_resolve_credential_with_vault", return_value="vault-token") as mock_rcv, \
             patch("core.notification.channels.telegram.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body", anima_name="bob")

        assert result == "telegram: OK"
        mock_rcv.assert_called_once_with(
            "bot_token_env", anima_name="bob", fallback_env="TELEGRAM_BOT_TOKEN",
        )

    @pytest.mark.asyncio
    async def test_chatwork_uses_vault(self, monkeypatch):
        monkeypatch.delenv("CW_TOKEN", raising=False)
        from core.notification.channels.chatwork import ChatworkChannel

        ch = ChatworkChannel({
            "api_token_env": "CW_TOKEN",
            "room_id": "999",
        })

        with patch.object(ch, "_resolve_credential_with_vault", return_value="vault-token") as mock_rcv, \
             patch("core.notification.channels.chatwork.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body", anima_name="charlie")

        assert result == "chatwork: OK"
        mock_rcv.assert_called_once_with(
            "api_token_env", anima_name="charlie", fallback_env="CHATWORK_API_TOKEN",
        )

    @pytest.mark.asyncio
    async def test_ntfy_uses_vault_for_optional_token(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOKEN", raising=False)
        from core.notification.channels.ntfy import NtfyChannel

        ch = NtfyChannel({
            "server_url": "https://ntfy.sh",
            "topic": "test",
            "token_env": "NTFY_TOKEN",
        })

        with patch.object(ch, "_resolve_credential_with_vault", return_value="vault-token") as mock_rcv, \
             patch("core.notification.channels.ntfy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await ch.send("Test", "body", anima_name="dave")

        assert result == "ntfy: OK"
        mock_rcv.assert_called_once_with(
            "token_env", anima_name="dave", fallback_env="NTFY_TOKEN",
        )
