"""Tests for chat-UI reply DM redirection gating (chat_dm_redirect).

user_aliases entries serve inbound trust elevation; they must not force
chat-UI replies out to Slack/Chatwork DMs unless chat_dm_redirect is
explicitly enabled.
"""

from unittest.mock import patch

from core._anima_messaging import MessagingMixin
from core.config.schemas import AnimaWorksConfig as Config
from core.config.schemas import UserAliasConfig


class _StubAnima(MessagingMixin):
    def __init__(self) -> None:
        self.name = "yoru"


def _config_with_alias(chat_dm_redirect: bool, outbound_dm: bool = True) -> Config:
    config = Config()
    config.external_messaging.chat_dm_redirect = chat_dm_redirect
    config.external_messaging.user_aliases["owner"] = UserAliasConfig(
        slack_user_id="U06MJKLV0TG",
        outbound_dm=outbound_dm,
    )
    return config


def _resolve(anima: _StubAnima, config: Config, tmp_path):
    with (
        patch("core.config.models.load_config", return_value=config),
        patch("core.paths.get_animas_dir", return_value=tmp_path),
    ):
        return anima._resolve_chat_external_recipient("owner")


def test_redirect_disabled_by_default(tmp_path):
    config = Config()
    config.external_messaging.user_aliases["owner"] = UserAliasConfig(
        slack_user_id="U06MJKLV0TG",
    )
    assert config.external_messaging.chat_dm_redirect is False
    assert _resolve(_StubAnima(), config, tmp_path) is None


def test_redirect_off_returns_none_despite_alias(tmp_path):
    config = _config_with_alias(chat_dm_redirect=False)
    assert _resolve(_StubAnima(), config, tmp_path) is None


def test_redirect_on_resolves_external_recipient(tmp_path):
    config = _config_with_alias(chat_dm_redirect=True)
    resolved = _resolve(_StubAnima(), config, tmp_path)
    assert resolved is not None
    assert resolved.is_internal is False
    assert resolved.channel == "slack"
    assert resolved.slack_user_id == "U06MJKLV0TG"


def test_redirect_on_but_outbound_dm_off_returns_none(tmp_path):
    """chat_dm_redirect alone is not enough: the alias must opt in to outbound DM."""
    config = _config_with_alias(chat_dm_redirect=True, outbound_dm=False)
    assert _resolve(_StubAnima(), config, tmp_path) is None


def test_external_platform_source_never_redirects(tmp_path):
    config = _config_with_alias(chat_dm_redirect=True)
    anima = _StubAnima()
    with (
        patch("core.config.models.load_config", return_value=config),
        patch("core.paths.get_animas_dir", return_value=tmp_path),
    ):
        assert anima._resolve_chat_external_recipient("owner", source="slack") is None
