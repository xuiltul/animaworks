from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config.schemas import ExternalMessagingChannelConfig
from core.messaging.messenger import ChannelMeta, load_channel_meta, save_channel_meta
from server.gateways.slack_channel_sync import SlackChannelSync, _ensure_board


class _FakeSlackManager:
    def __init__(self, app_map: dict[str, object]) -> None:
        self._app_map = app_map


def _mock_config() -> SimpleNamespace:
    return SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(
                channel_discovery=True,
                default_anima="sakura",
                board_mapping={},
                default_channel_company="",
            ),
        ),
    )


def _patch_update_config(monkeypatch: pytest.MonkeyPatch, cfg: SimpleNamespace, save_calls=None) -> None:
    def update(fn, *args, **kwargs):
        previous = dict(cfg.external_messaging.slack.board_mapping)
        result = fn(cfg)
        if save_calls is not None and cfg.external_messaging.slack.board_mapping != previous:
            save_calls.append(cfg)
        return result or cfg

    monkeypatch.setattr("core.config.io.update_config", update)


def test_update_config_mapping_skips_save_when_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mock_config()
    cfg.external_messaging.slack.board_mapping = {"C123": "general"}
    save_calls: list[object] = []
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)
    _patch_update_config(monkeypatch, cfg, save_calls)

    sync = SlackChannelSync()
    sync.board_mapping = {"C123": "general"}
    sync._update_config_mapping()

    assert save_calls == []


def test_update_config_mapping_saves_when_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mock_config()
    save_calls: list[object] = []
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)
    _patch_update_config(monkeypatch, cfg, save_calls)

    sync = SlackChannelSync()
    sync.board_mapping = {"C123": "general"}
    sync._update_config_mapping()

    assert save_calls == [cfg]
    assert cfg.external_messaging.slack.board_mapping == {"C123": "general"}


@pytest.mark.asyncio
async def test_sync_marks_missing_slack_channel_and_skips_reverse_recreate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "general.jsonl").write_text("", encoding="utf-8")

    cfg = _mock_config()
    cfg.external_messaging.slack.board_mapping = {"C123": "general"}
    save_calls: list[object] = []

    monkeypatch.setattr("server.gateways.slack_channel_sync.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr("server.gateways.slack_socket.SlackSocketModeManager", _FakeSlackManager)

    async def _list_public_channels(_token: str):
        return []

    monkeypatch.setattr("server.gateways.slack_channel_sync._list_public_channels", _list_public_channels)

    async def _create_channel(*_args, **_kwargs):
        raise AssertionError("deleted Slack channel should not be recreated")

    monkeypatch.setattr("server.gateways.slack_channel_sync._create_channel", _create_channel)
    monkeypatch.setattr("server.gateways.slack_channel_sync._join_channel_if_needed", lambda *_a, **_k: False)
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)
    _patch_update_config(monkeypatch, cfg, save_calls)

    manager = _FakeSlackManager(
        {"sakura": SimpleNamespace(client=SimpleNamespace(token="xoxb-sakura"))},
    )
    sync = SlackChannelSync()
    sync.board_mapping = {"C123": "general"}

    await sync.sync(manager)

    meta = load_channel_meta(shared_dir, "general")
    assert meta is not None
    assert meta.slack_sync_disabled is True
    assert meta.slack_deleted_at
    assert sync.board_mapping == {}
    assert save_calls


@pytest.mark.asyncio
async def test_sync_clears_tombstone_when_slack_channel_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)
    (channels_dir / "general.jsonl").write_text("", encoding="utf-8")
    save_channel_meta(
        shared_dir,
        "general",
        ChannelMeta(
            members=[],
            slack_sync_disabled=True,
            slack_deleted_at="2026-04-02T00:00:00+00:00",
        ),
    )

    cfg = _mock_config()
    monkeypatch.setattr("server.gateways.slack_channel_sync.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr("server.gateways.slack_socket.SlackSocketModeManager", _FakeSlackManager)

    async def _list_public_channels(_token: str):
        return [{"id": "C999", "name": "general", "is_private": False}]

    async def _join_channel_if_needed(*_args, **_kwargs):
        return False

    async def _create_channel(*_args, **_kwargs):
        raise AssertionError("existing Slack channel should not be recreated")

    monkeypatch.setattr("server.gateways.slack_channel_sync._list_public_channels", _list_public_channels)
    monkeypatch.setattr("server.gateways.slack_channel_sync._join_channel_if_needed", _join_channel_if_needed)
    monkeypatch.setattr("server.gateways.slack_channel_sync._create_channel", _create_channel)
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)
    _patch_update_config(monkeypatch, cfg)

    manager = _FakeSlackManager(
        {"sakura": SimpleNamespace(client=SimpleNamespace(token="xoxb-sakura"))},
    )
    sync = SlackChannelSync()

    await sync.sync(manager)

    meta = load_channel_meta(shared_dir, "general")
    assert meta is not None
    assert meta.slack_sync_disabled is False
    assert meta.slack_deleted_at == ""
    assert sync.board_mapping == {"C999": "general"}


@pytest.mark.asyncio
async def test_sync_falls_back_to_available_bot_when_default_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When default_anima is not in app_map, fallback to any available bot."""
    caplog.set_level("INFO", logger="animaworks.slack_channel_sync")
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)

    cfg = SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(channel_discovery=True, default_anima="kotoha", board_mapping={}),
        ),
    )

    monkeypatch.setattr("server.gateways.slack_channel_sync.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr("server.gateways.slack_socket.SlackSocketModeManager", _FakeSlackManager)

    list_channels_calls: list[str] = []

    async def _list_public_channels(token: str):
        list_channels_calls.append(token)
        return [{"id": "C111", "name": "dev", "is_private": False}]

    async def _join_channel_if_needed(*_args, **_kwargs):
        return False

    monkeypatch.setattr("server.gateways.slack_channel_sync._list_public_channels", _list_public_channels)
    monkeypatch.setattr("server.gateways.slack_channel_sync._join_channel_if_needed", _join_channel_if_needed)
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)
    _patch_update_config(monkeypatch, cfg)

    # sakura is available but kotoha (default) is not
    manager = _FakeSlackManager(
        {"sakura": SimpleNamespace(client=SimpleNamespace(token="xoxb-sakura"))},
    )
    sync = SlackChannelSync()

    await sync.sync(manager)

    # Should have used sakura's token as fallback
    assert list_channels_calls == ["xoxb-sakura"]
    assert sync.board_mapping == {"C111": "dev"}
    assert "falling back to 'sakura' for channel discovery" in caplog.text


def test_ensure_board_applies_default_channel_company(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_dir = tmp_path / "shared"
    cfg = _mock_config()
    cfg.external_messaging.slack.default_channel_company = "alpha"
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)

    created = _ensure_board(shared_dir, "dev-room", "dev-room")
    assert created is True
    meta = load_channel_meta(shared_dir, "dev-room")
    assert meta is not None
    assert meta.company == "alpha"
    assert meta.members == []
    # second call is no-op
    assert _ensure_board(shared_dir, "dev-room", "dev-room") is False


def test_ensure_board_empty_company_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_dir = tmp_path / "shared"
    cfg = _mock_config()
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)

    assert _ensure_board(shared_dir, "open-room", "open-room") is True
    meta = load_channel_meta(shared_dir, "open-room")
    assert meta is not None
    assert meta.company == ""


@pytest.mark.asyncio
async def test_sync_returns_empty_when_no_bots_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When default_anima missing and no fallback bots exist, return empty mapping."""
    shared_dir = tmp_path / "shared"
    channels_dir = shared_dir / "channels"
    channels_dir.mkdir(parents=True)

    cfg = SimpleNamespace(
        external_messaging=SimpleNamespace(
            slack=SimpleNamespace(
                channel_discovery=True,
                default_anima="kotoha",
                board_mapping={},
                default_channel_company="",
            ),
        ),
    )

    monkeypatch.setattr("server.gateways.slack_channel_sync.get_shared_dir", lambda: shared_dir)
    monkeypatch.setattr("server.gateways.slack_socket.SlackSocketModeManager", _FakeSlackManager)
    monkeypatch.setattr("core.config.models.load_config", lambda: cfg)

    # No bots available at all (only __shared__ which is excluded)
    manager = _FakeSlackManager({"__shared__": SimpleNamespace()})
    sync = SlackChannelSync()

    result = await sync.sync(manager)

    assert result == {}
    assert "no Slack bots available for channel discovery" in caplog.text


def test_slack_channel_discovery_defaults_to_disabled() -> None:
    assert ExternalMessagingChannelConfig().channel_discovery is False


@pytest.mark.asyncio
async def test_startup_and_periodic_sync_skip_when_discovery_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.app import _slack_channel_resync, _sync_slack_channels_if_enabled

    config = SimpleNamespace(
        external_messaging=SimpleNamespace(slack=ExternalMessagingChannelConfig()),
    )
    monkeypatch.setattr("server.app.load_config", lambda: config)
    sync = AsyncMock()
    monkeypatch.setattr(SlackChannelSync, "sync", sync)

    startup_app = SimpleNamespace(
        state=SimpleNamespace(slack_socket_manager=_FakeSlackManager({}), slack_channel_sync=None),
    )
    await _sync_slack_channels_if_enabled(startup_app)
    sync.assert_not_awaited()

    periodic_app = SimpleNamespace(
        state=SimpleNamespace(slack_socket_manager=_FakeSlackManager({}), slack_channel_sync=None),
    )
    await _slack_channel_resync(periodic_app)
    sync.assert_not_awaited()
    assert startup_app.state.slack_channel_sync is None
    assert periodic_app.state.slack_channel_sync is None


def test_ensure_board_skips_closed_name(
    tmp_path: Path,
) -> None:
    shared_dir = tmp_path / "shared"
    save_channel_meta(shared_dir, "general", ChannelMeta(members=[], closed=True))

    assert _ensure_board(shared_dir, "general", "general") is False
    assert not (shared_dir / "channels" / "general.jsonl").exists()
