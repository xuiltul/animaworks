# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the rate-guard wiring in one_shot_completion()."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.llm.oneshot as llm_utils
from core.config.schemas import LlmRateGuardConfig
from core.llm.guard.rate_guard import LlmRateGuard

pytestmark = pytest.mark.asyncio


class _ApiError(Exception):
    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        headers: dict | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if headers is not None:
            self.headers = headers


def _make_guard(tmp_path: Path, **cfg) -> LlmRateGuard:
    return LlmRateGuard(config=LlmRateGuardConfig(**cfg), path=tmp_path / "guard.json")


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Skip the real backoff sleep so rate-limit retries run instantly."""
    with patch("core.llm.oneshot.asyncio.sleep", new=AsyncMock()):
        yield


@pytest.fixture(autouse=True)
def _anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give LiteLLM an Anthropic key so its stage runs (it is skipped without one)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _fixed_mode_s_realm(data_dir: Path):
    """Pin the Mode-S auth realm through the temporary config file."""
    from core.config import invalidate_cache

    config_path = data_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["anima_defaults"]["mode_s_auth"] = "max"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    invalidate_cache()


@pytest.fixture(autouse=True)
def agent_sdk_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the public Agent SDK client boundary used by one-shot fallback."""

    class _Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def query(self, _prompt: str) -> None:
            return None

        async def receive_response(self):
            yield SimpleNamespace(content=[SimpleNamespace(text="sdk")])

    sdk_module = ModuleType("claude_agent_sdk")
    sdk_module.ClaudeAgentOptions = _Options
    client_factory = MagicMock(side_effect=lambda **_kwargs: _Client())
    sdk_module.ClaudeSDKClient = client_factory
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk_module)
    monkeypatch.setattr("core.execution.engines.claude.apply_sdk_transport_patch", lambda: None)
    monkeypatch.setattr("core.execution.engines.claude.resolve_sdk_cli_path", lambda: None)
    return client_factory


async def test_content_policy_returns_none_without_fallback(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path)
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", side_effect=_ApiError("violates our usage policies", status_code=400)),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result is None
    agent_sdk_client.assert_not_called()


async def test_api_rate_limit_does_not_block_mode_s_sdk(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    # The key requirement: a 429 on the API key (anthropic:api) must NOT block
    # the independently-authenticated Agent SDK (anthropic:max).
    guard = _make_guard(tmp_path)
    mock_litellm = AsyncMock(side_effect=_ApiError("Too Many Requests", status_code=429))
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result == "sdk"
    agent_sdk_client.assert_called_once()
    assert mock_litellm.await_count == 2  # initial + one short-backoff retry
    assert guard.blocked_remaining("anthropic:api") > 0
    assert guard.blocked_remaining("anthropic:max") == 0.0


async def test_prior_api_block_skips_litellm_but_runs_max_sdk(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path)
    guard.report_block("anthropic:api", 300, "rate_limit")  # a peer already blocked the API realm
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result == "sdk"
    mock_litellm.assert_not_called()
    agent_sdk_client.assert_called_once()


async def test_mode_s_realm_block_skips_agent_sdk(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    # Same-realm protection: a block on anthropic:max skips the Agent SDK stage.
    guard = _make_guard(tmp_path)
    guard.report_block("anthropic:max", 300, "rate_limit")
    mock_litellm = AsyncMock(side_effect=RuntimeError("weird"))  # unknown → fall through to SDK stage
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result is None
    assert mock_litellm.await_count == 1  # LiteLLM ran (api realm free)
    agent_sdk_client.assert_not_called()  # but the max-realm SDK stage was skipped


async def test_all_realms_guarded_returns_none(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path)
    guard.report_block("anthropic:api", 300, "rate_limit")
    guard.report_block("anthropic:max", 300, "rate_limit")
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result is None
    mock_litellm.assert_not_called()
    agent_sdk_client.assert_not_called()


async def test_large_retry_after_skips_inline_retry(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    # A Retry-After beyond the inline budget reports the full fleet block but
    # skips the in-process retry; the SDK (max realm) still runs.
    guard = _make_guard(tmp_path)
    mock_litellm = AsyncMock(
        side_effect=_ApiError("Too Many Requests", status_code=429, headers={"retry-after": "300"})
    )
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result == "sdk"
    assert mock_litellm.await_count == 1  # no inline retry for a >15s wait
    agent_sdk_client.assert_called_once()
    assert guard.blocked_remaining("anthropic:api") > 60  # full (clamped) block


async def test_auth_error_logs_error_and_falls_back(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path)
    with (
        caplog.at_level(logging.ERROR, logger="core.llm.oneshot"),
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=AsyncMock(side_effect=_ApiError("unauthorized", status_code=401))),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    # auth does not block any realm, so the SDK fallback still runs.
    assert result == "sdk"
    agent_sdk_client.assert_called_once()
    assert guard.blocked_remaining("anthropic:api") == 0.0
    assert any(rec.levelno == logging.ERROR and "human attention" in rec.message for rec in caplog.records)


async def test_codex_realm_block_skips_codex_sdk_and_litellm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_module = ModuleType("openai_codex")
    codex_client = MagicMock()
    codex_module.ApprovalMode = SimpleNamespace(deny_all="deny_all")
    codex_module.AsyncCodex = codex_client
    codex_module.CodexConfig = MagicMock()
    codex_module.Sandbox = SimpleNamespace(read_only="read_only")
    monkeypatch.setitem(sys.modules, "openai_codex", codex_module)

    guard = _make_guard(tmp_path)
    guard.report_block("openai:codex", 300, "rate_limit")  # codex realm blocked
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "codex/gpt-5.4-mini"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="codex/gpt-5.4-mini")

    assert result is None
    mock_litellm.assert_not_called()  # codex/* models are Codex-SDK-only
    codex_client.assert_not_called()  # openai:codex realm blocked


@pytest.mark.parametrize(
    ("model", "realm"),
    [
        ("bedrock/jp.anthropic.claude-sonnet-4-6", "bedrock"),
        ("vertex_ai/claude-sonnet-4-6", "vertex"),
    ],
)
async def test_litellm_realm_recorded_from_model_prefix(tmp_path: Path, model: str, realm: str) -> None:
    # A bedrock/vertex 429 is recorded on its own realm and does not block the
    # ``anthropic:api`` (direct API key) realm.
    guard = _make_guard(tmp_path)
    mock_litellm = AsyncMock(side_effect=_ApiError("Too Many Requests", status_code=429))
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": model}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        await llm_utils.one_shot_completion("hi", model=model)

    assert guard.blocked_remaining(f"anthropic:{realm}") > 0
    assert guard.blocked_remaining("anthropic:api") == 0.0


async def test_disabled_guard_does_not_record_block(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path, enabled=False)
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=AsyncMock(side_effect=_ApiError("rate limit", status_code=429))),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result == "sdk"
    agent_sdk_client.assert_called_once()
    assert guard.blocked_remaining("anthropic:api") == 0.0
    assert not (tmp_path / "guard.json").exists()


async def test_unknown_error_degrades_to_current_fallback(
    tmp_path: Path,
    agent_sdk_client: MagicMock,
) -> None:
    guard = _make_guard(tmp_path)
    mock_litellm = AsyncMock(side_effect=RuntimeError("weird"))
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "anthropic/claude-sonnet-4-6"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model="anthropic/claude-sonnet-4-6")

    assert result == "sdk"
    # No same-backend retry for unknown; single attempt then fallback.
    assert mock_litellm.await_count == 1
    agent_sdk_client.assert_called_once()
    assert guard.blocked_remaining("anthropic:api") == 0.0


# ── LiteLLM is skipped when it has no Anthropic key ─────────────────────────


def _no_anthropic_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "anthropic/claude-sonnet-4-6"])
async def test_keyless_anthropic_model_skips_litellm_and_uses_agent_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    agent_sdk_client: MagicMock,
    model: str,
) -> None:
    # claude_code_login (or a codex_login named for a Claude model) has no key.
    _no_anthropic_key_env(monkeypatch)
    guard = _make_guard(tmp_path)
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        caplog.at_level(logging.DEBUG, logger="core.llm.oneshot"),
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": model}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=guard),
    ):
        result = await llm_utils.one_shot_completion("hi", model=model, credential="anthropic")

    assert result == "sdk"
    mock_litellm.assert_not_called()
    agent_sdk_client.assert_called_once()
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)
    assert guard.blocked_remaining("anthropic:api") == 0.0


async def test_keyless_anthropic_model_without_sdk_fallback_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    agent_sdk_client: MagicMock,
) -> None:
    _no_anthropic_key_env(monkeypatch)
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        caplog.at_level(logging.DEBUG, logger="core.llm.oneshot"),
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "claude-opus-5-5"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=_make_guard(tmp_path)),
    ):
        result = await llm_utils.one_shot_completion("hi", model="claude-opus-5-5", allow_agent_sdk_fallback=False)

    assert result is None
    mock_litellm.assert_not_called()
    agent_sdk_client.assert_not_called()
    assert not any("human attention" in rec.message for rec in caplog.records)


def test_sync_one_shot_skips_keyless_litellm_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_sdk_client: MagicMock,
) -> None:
    _no_anthropic_key_env(monkeypatch)
    mock_litellm = AsyncMock(return_value="should-not-run")
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value={"model": "claude-opus-5-5"}),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=_make_guard(tmp_path)),
    ):
        result = llm_utils.one_shot_completion_sync("hi", model="claude-opus-5-5")

    assert result == "sdk"
    mock_litellm.assert_not_called()
    agent_sdk_client.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "env"),
    [
        ({"model": "claude-opus-5-5", "api_key": "sk-from-credential"}, {}),
        ({"model": "claude-opus-5-5"}, {"ANTHROPIC_API_KEY": "sk-env"}),
        ({"model": "claude-opus-5-5"}, {"ANTHROPIC_AUTH_TOKEN": "token-env"}),
        ({"model": "bedrock/jp.anthropic.claude-sonnet-4-6"}, {}),
        ({"model": "vertex_ai/claude-sonnet-4-6"}, {}),
        ({"model": "openai/gpt-test"}, {}),
    ],
)
async def test_litellm_still_runs_when_it_can_authenticate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_sdk_client: MagicMock,
    kwargs: dict,
    env: dict,
) -> None:
    _no_anthropic_key_env(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    mock_litellm = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="litellm"))])
    )
    with (
        patch("core.llm.oneshot.get_llm_kwargs_for_model", return_value=dict(kwargs)),
        patch("litellm.acompletion", new=mock_litellm),
        patch("core.llm.guard.rate_guard.get_rate_guard", return_value=_make_guard(tmp_path)),
    ):
        result = await llm_utils.one_shot_completion("hi", model=kwargs["model"])

    assert result == "litellm"
    mock_litellm.assert_awaited_once()
    agent_sdk_client.assert_not_called()
