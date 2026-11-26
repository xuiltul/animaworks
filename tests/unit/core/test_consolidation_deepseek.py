"""Tests for the consolidation DeepSeek-ification plan (2026-10-03).

Covers T2 (explicit daily fallback model ordering), T4 (one-shot Agent SDK
fallback toggle + anthropic temperature sanitization), T5 (fact extraction
max_tokens), and T6 (deadline cutoff + root cancel propagation).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import CycleResult, ModelConfig

# ── T3: fact reconcile model override & credential ────────────────────────


@pytest.mark.unit
def test_reconcile_uses_fact_reconcile_model_when_set() -> None:
    from core.memory.facts import invalidation_llm

    consolidation = SimpleNamespace(
        llm_model="openai/deepseek-v4-flash",
        llm_credential="gpu40-direct",
        fact_reconcile_model="anthropic/claude-sonnet-4-6",
        fact_reconcile_credential="bsky-cred",
    )
    rag = SimpleNamespace(fact_extraction_timeout_seconds=30)
    defs = SimpleNamespace(
        consolidation=consolidation,
        rag=rag,
        anima_defaults=SimpleNamespace(background_model="", model="", background_credential="", credential=""),
        locale="ja",
    )
    with patch("core.config.load_config", return_value=defs):
        model, _extra, timeout, credential = invalidation_llm._resolve_reconcile_llm_config("/tmp/no-status")
    assert model == "anthropic/claude-sonnet-4-6"
    assert credential == "bsky-cred"
    assert timeout == 30


# ── T2: explicit daily-summary fallback sits right after primary ─────────────


def _fallback_cfg(llm_fallback_model: str = "anthropic/claude-sonnet-4-6"):
    return SimpleNamespace(
        consolidation=SimpleNamespace(
            llm_model="openai/deepseek-v4-flash",
            llm_credential="gpu40-direct",
            llm_fallback_model=llm_fallback_model,
            llm_fallback_credential="anthropic",
        ),
        credentials={
            "gpu40-direct": SimpleNamespace(api_key="k", base_url="http://gpu40:8000/v1", keys={}),
            "openai": SimpleNamespace(api_key="k", base_url=None, keys={}),
            "anthropic": SimpleNamespace(api_key="k", base_url=None, keys={}, type="api_key", is_org=False),
            "ollama": SimpleNamespace(api_key=None, base_url="http://localhost:11434/v1", keys={}),
        },
        model_modes={},
        local_llm=None,
        anima_defaults=SimpleNamespace(background_credential=""),
    )


@pytest.mark.unit
def test_episode_summary_fallback_model_sits_right_after_primary() -> None:
    from core.anima.lifecycle import _episode_summary_model_configs

    base = ModelConfig(
        model="anthropic/claude-opus-5-5",
        background_model="anthropic/claude-opus-5-5",
        fallback_models=[],
    )
    cfg = _fallback_cfg()
    candidates = _episode_summary_model_configs(base, "openai/deepseek-v4-flash", cfg)

    # Only the primary and explicit consolidation fallback are helper candidates.
    assert [candidate.model for candidate in candidates] == [
        "openai/deepseek-v4-flash",
        "anthropic/claude-sonnet-4-6",
    ]
    assert all("claude-opus-5-5" not in candidate.model for candidate in candidates)


# ── T4: allow_agent_sdk_fallback=False skips Agent SDK ──────────────────────


@pytest.mark.asyncio
async def test_one_shot_agent_sdk_fallback_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")  # LiteLLM runs only with a key
    calls: dict = {}

    async def fake_litellm_stage(prompt, **kwargs):
        calls["litellm"] = True
        return ("fallback", None)

    with (
        patch("core.llm.oneshot._litellm_stage_with_guard", side_effect=fake_litellm_stage),
        patch("core.llm.oneshot._try_agent_sdk", new_callable=AsyncMock) as mock_sdk,
        patch("core.llm.oneshot._is_codex_model", return_value=False),
        patch(
            "core.llm.oneshot._is_anthropic_model",
            return_value=True,
        ),
    ):
        result = await llm.oneshot.one_shot_completion(
            "hello",
            model="anthropic/claude-sonnet-4-6",
            allow_agent_sdk_fallback=False,
        )

    assert result is None
    assert calls.get("litellm") is True
    mock_sdk.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_temperature_zero_dropped_for_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")  # LiteLLM runs only with a key
    captured: dict = {}

    async def fake_litellm_stage(prompt, *, resolved_model, llm_kwargs, **kwargs):
        captured.update(llm_kwargs)
        return ("success", "ok")

    with (
        patch("core.llm.oneshot._litellm_stage_with_guard", side_effect=fake_litellm_stage),
        patch("core.llm.oneshot._is_codex_model", return_value=False),
    ):
        result = await llm.oneshot.one_shot_completion(
            "hello",
            model="anthropic/claude-opus-5-5",
            temperature=0.0,
            structured_output=False,
        )

    assert result == "ok"
    # temperature must be dropped so LiteLLM does not raise UnsupportedParamsError.
    assert "temperature" not in captured
    assert captured.get("drop_params") is True


# ── T4: extractor and reconcile pass allow_agent_sdk_fallback=False ─────────


@pytest.mark.unit
def test_extractor_forwards_agent_sdk_fallback_false() -> None:
    from core.memory.facts.extractor import FactExtractor

    ext = FactExtractor(model="qwen-model", max_retries=1)
    with patch(
        "core.llm.oneshot.one_shot_completion",
        new_callable=AsyncMock,
        return_value='{"entities": []}',
    ) as mock_one_shot:
        asyncio.run(ext._call_llm("system", "user"))
    assert mock_one_shot.call_args.kwargs["allow_agent_sdk_fallback"] is False


# ── T5: fact-extraction max_tokens follows rag config ──────────────────────


@pytest.mark.unit
def test_extractor_uses_max_tokens() -> None:
    from core.memory.facts.extractor import FactExtractor

    ext = FactExtractor(model="qwen-model", max_retries=1, max_tokens=4096)
    with patch(
        "core.llm.oneshot.one_shot_completion",
        new_callable=AsyncMock,
        return_value='{"entities": []}',
    ) as mock_one_shot:
        asyncio.run(ext._call_llm("system", "user"))
    assert mock_one_shot.call_args.kwargs["max_tokens"] == 4096


@pytest.mark.unit
def test_resolve_extraction_max_tokens_from_config() -> None:
    from core.memory.facts.config import _resolve_extraction_max_tokens

    rag = SimpleNamespace(fact_extraction_max_tokens=5000)
    with patch("core.config.load_config", return_value=SimpleNamespace(rag=rag)):
        assert _resolve_extraction_max_tokens() == 5000

    with patch(
        "core.config.load_config", return_value=SimpleNamespace(rag=SimpleNamespace(fact_extraction_max_tokens=500))
    ):
        assert _resolve_extraction_max_tokens() == 8192  # below ge floor -> default

    with patch(
        "core.config.load_config",
        return_value=SimpleNamespace(rag=SimpleNamespace(fact_extraction_max_tokens=None)),
    ):
        assert _resolve_extraction_max_tokens() == 8192


# ── T6: consolidation deadline cutoff clears consolidation mode ──────────────


class _SlowWeeklyLifecycle:
    """Minimal LifecycleMixin-compatible host whose weekly run exceeds deadline."""

    def __init__(self) -> None:
        self.name = "deepseek-anima"
        self.anima_dir = "/tmp/deepseek-anima"
        self._background_lock = asyncio.Lock()
        self._mark_busy_start = MagicMock()
        self._status_slots = {"background": "idle"}
        self._task_slots = {"background": ""}
        self._notify_lock_released = MagicMock()
        self._last_activity = None

        class Handler:
            def set_active_session_type(self, *a, **k):
                return "tok"

            def set_session_origin(self, *a, **k):
                return None

        agent = MagicMock()
        agent._tool_handler = Handler()
        agent.set_interrupt_event = MagicMock()
        self.agent = agent

        self._get_interrupt_event = lambda _name: asyncio.Event()
        self._activity = MagicMock()
        self._activity.alog = AsyncMock()

    async def _keepalive_while_busy(self, interval: float = 60.0) -> None:
        await asyncio.sleep(interval)

    async def _run_weekly_consolidation(self, engine):
        await asyncio.sleep(30)  # far beyond the deadline
        return CycleResult(trigger="consolidation:weekly", action="completed", summary="late")


@pytest.mark.asyncio
async def test_run_consolidation_clears_mode_when_deadline_exceeded() -> None:
    from core.anima.lifecycle import LifecycleMixin

    obj = _SlowWeeklyLifecycle()
    fake_writer = MagicMock()
    fake_writer.set_consolidation_mode = AsyncMock(return_value=None)

    with (
        patch("core.platform.state_writer.get_state_writer", return_value=fake_writer),
        patch("core.memory.maintenance.consolidation.ConsolidationEngine", return_value=MagicMock()),
        patch("core.tooling.handler.active_session_type") as ast,
    ):
        ast.reset = MagicMock()
        with pytest.raises(TimeoutError):
            await LifecycleMixin.run_consolidation(obj, "weekly", deadline_s=0.05)

    # The mode flag must be cleared even though the consolidation was cut off.
    assert fake_writer.set_consolidation_mode.call_count == 2  # True then False
    assert fake_writer.set_consolidation_mode.await_args.args == (False,)
    assert obj._status_slots["background"] == "idle"


@pytest.mark.unit
def test_execute_background_contract_forwards_deadline_to_run_consolidation() -> None:
    from core.runtime import task_runner

    anima = MagicMock()
    anima.run_consolidation = AsyncMock(return_value=CycleResult(trigger="consolidation:daily", action="completed"))
    result = asyncio.run(
        task_runner.execute_background_contract(
            anima,
            kind="consolidation",
            payload={"consolidation_type": "daily", "deadline_s": 123.0},
        )
    )
    assert anima.run_consolidation.await_args.kwargs["deadline_s"] == 123.0
    assert result["task_type"] == "consolidation"


@pytest.mark.unit
def test_cancel_consolidation_sends_cancel_event_to_background_job() -> None:
    from core.runtime.task_runner_supervisor import IPCV2Identity, TaskRunnerJob, TaskRunnerSupervisor

    loop = asyncio.new_event_loop()
    job = TaskRunnerJob(
        identity=IPCV2Identity(
            job_id="bg-consolidation-1",
            root_epoch="e",
            attempt=1,
            lane="background",
            display_lane="background",
        ),
        request_id="req-1",
        params={},
        result=loop.create_future(),
        peer_state=MagicMock(),
    )
    conn = MagicMock()
    conn.send_event = AsyncMock(return_value=1)
    job.connection = conn

    real = TaskRunnerSupervisor.__new__(TaskRunnerSupervisor)
    real.anima_name = "deepseek-anima"
    real._jobs = {job.identity.job_id: job}

    result = asyncio.run(real.cancel_consolidation())
    loop.close()
    assert result["status"] == "cancel_requested"
    conn.send_event.assert_awaited_once_with("cancel", {})
