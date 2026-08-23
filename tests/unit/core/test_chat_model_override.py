"""Tests for the chat per-message model override (Cursor-style)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core._anima_messaging import _apply_chat_model_override
from core.schemas import CycleResult, ModelConfig
from server.routes.chat_models import ChatRequest


def _owner() -> MagicMock:
    owner = MagicMock()
    owner.name = "test-anima"
    return owner


def _base() -> ModelConfig:
    return ModelConfig(model="claude-sonnet-4-6", execution_mode="s", resolved_mode="S")


# ── ChatRequest backward compatibility ─────────────────────────────────


def test_chat_request_model_backward_compatible() -> None:
    """No ``model`` in the payload must keep the previous behaviour (None)."""
    req = ChatRequest(message="hello")
    assert req.model is None

    req2 = ChatRequest(message="hello", model="gpt-4.1")
    assert req2.model == "gpt-4.1"


# ── _apply_chat_model_override ────────────────────────────────────────


def test_apply_valid_override_builds_and_logs() -> None:
    owner = _owner()
    base = _base()
    override = base.model_copy(
        update={"model": "gpt-4.1", "execution_mode": "s", "resolved_mode": "S"}
    )
    config = MagicMock()
    with (
        patch("core.config.io.load_config", return_value=config),
        patch(
            "core.config.model_mode.parse_fallback_entry",
            return_value=("s", "gpt-4.1"),
        ) as parse,
        patch(
            "core.config.model_config.build_model_override_config",
            return_value=override,
        ) as build,
    ):
        result = _apply_chat_model_override(owner, base, "gpt-4.1", thread_id="default")

    assert result is override
    parse.assert_called_once_with("gpt-4.1", config)
    build.assert_called_once_with(base, "s", "gpt-4.1", config)
    owner._activity.log.assert_called_once()
    call = owner._activity.log.call_args
    assert call.args[0] == "model_override"
    assert call.kwargs["channel"] == "chat"
    assert call.kwargs["meta"]["requested"] == "gpt-4.1"
    assert call.kwargs["meta"]["resolved"] == "S:gpt-4.1"
    assert call.kwargs["safe"] is True


def test_apply_invalid_entry_keeps_base_without_event() -> None:
    owner = _owner()
    base = _base()
    config = MagicMock()
    with (
        patch("core.config.io.load_config", return_value=config),
        patch("core.config.model_mode.parse_fallback_entry", return_value=None),
    ):
        result = _apply_chat_model_override(owner, base, "not a model", thread_id="default")

    assert result is base
    owner._activity.log.assert_not_called()


def test_apply_unresolvable_credential_keeps_base_without_event() -> None:
    owner = _owner()
    base = _base()
    config = MagicMock()
    with (
        patch("core.config.io.load_config", return_value=config),
        patch(
            "core.config.model_mode.parse_fallback_entry",
            return_value=("s", "unknown-model"),
        ),
        patch("core.config.model_config.build_model_override_config", return_value=None),
    ):
        result = _apply_chat_model_override(
            owner, base, "unknown-model", thread_id="default"
        )

    assert result is base
    owner._activity.log.assert_not_called()


def test_apply_empty_model_returns_base() -> None:
    owner = _owner()
    base = _base()
    assert _apply_chat_model_override(owner, base, "", thread_id="default") is base


# ── process_message integration (run_cycle wiring) ────────────────────


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.set_on_message_sent = MagicMock()
    agent.set_on_schedule_changed = MagicMock()
    agent.set_interrupt_event = MagicMock()
    agent.background_manager = None
    agent._tool_handler = MagicMock()
    # Return a real contextvar token so ``active_session_type.reset()`` works.
    from core.tooling import handler as _tool_handler_mod

    _tool_handler_mod.active_session_type.set("chat")
    agent._tool_handler.set_active_session_type = MagicMock(
        side_effect=lambda _st: _tool_handler_mod.active_session_type.set("chat")
    )
    agent.supports_message_injection = True
    agent._resolve_execution_mode = MagicMock(return_value="s")
    agent.run_cycle = AsyncMock(
        return_value=CycleResult(
            trigger="message:human",
            action="responded",
            summary="ok",
        )
    )
    return agent


class TestProcessMessageModelOverride:
    @pytest.fixture
    def anima_dir(self, tmp_path):
        d = tmp_path / "test-anima"
        d.mkdir()
        (d / "identity.md").write_text("# Test Anima", encoding="utf-8")
        (d / "injection.md").write_text("test injection", encoding="utf-8")
        (d / "status.json").write_text(
            '{"enabled": true, "model": "claude-sonnet-4-6"}',
            encoding="utf-8",
        )
        for sub in (
            "state",
            "episodes",
            "knowledge",
            "procedures",
            "skills",
            "shortterm",
            "assets",
        ):
            (d / sub).mkdir()
        (d / "shortterm" / "chat").mkdir()
        (d / "shortterm" / "heartbeat").mkdir()
        (d / "activity_log").mkdir()
        return d

    @pytest.fixture
    def shared_dir(self, tmp_path):
        d = tmp_path / "shared"
        d.mkdir()
        (d / "channels").mkdir()
        (d / "users").mkdir()
        (d / "dm_logs").mkdir()
        return d

    @patch("core.anima.AgentCore")
    @patch("core.anima.MemoryManager")
    @patch("core.anima.Messenger")
    async def test_model_override_reaches_run_cycle(
        self, mock_messenger_cls, mock_mm_cls, mock_agent_cls, anima_dir, shared_dir
    ):
        """A requested model must be passed as ``model_config_override``."""
        base = _base()
        override = base.model_copy(
            update={"model": "gpt-4.1", "execution_mode": "s", "resolved_mode": "S"}
        )
        mock_mm_cls.return_value.read_model_config.return_value = base
        mock_agent_cls.return_value = _agent()

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)
        config = MagicMock()

        with (
            patch("core._anima_messaging.resolve_effective_model_config", return_value=base),
            patch("core._anima_messaging.log_model_fallback", return_value=None),
            patch("core.config.io.load_config", return_value=config),
            patch(
                "core.config.model_mode.parse_fallback_entry",
                return_value=("s", "gpt-4.1"),
            ),
            patch(
                "core.config.model_config.build_model_override_config",
                return_value=override,
            ),
        ):
            await anima.process_message("hello", from_person="human", model="gpt-4.1")

        call = mock_agent_cls.return_value.run_cycle.await_args
        assert call is not None
        assert call.kwargs["model_config_override"] is override

    @patch("core.anima.AgentCore")
    @patch("core.anima.MemoryManager")
    @patch("core.anima.Messenger")
    async def test_invalid_model_continues_with_default(
        self, mock_messenger_cls, mock_mm_cls, mock_agent_cls, anima_dir, shared_dir
    ):
        """An invalid model must fall back to the default (base) model."""
        base = _base()
        mock_mm_cls.return_value.read_model_config.return_value = base
        mock_agent_cls.return_value = _agent()

        from core.anima import DigitalAnima

        anima = DigitalAnima(anima_dir, shared_dir)
        config = MagicMock()

        with (
            patch("core._anima_messaging.resolve_effective_model_config", return_value=base),
            patch("core._anima_messaging.log_model_fallback", return_value=None),
            patch("core.config.io.load_config", return_value=config),
            patch("core.config.model_mode.parse_fallback_entry", return_value=None),
        ):
            await anima.process_message("hello", from_person="human", model="bad!!")

        call = mock_agent_cls.return_value.run_cycle.await_args
        assert call is not None
        assert call.kwargs["model_config_override"] is base
