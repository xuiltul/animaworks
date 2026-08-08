# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the fleet LLM rate guard.

Exercises the guard end-to-end through the Mode A executor and across real
OS processes — no LLM is called; failures are injected and the shared guard
file on disk is the observable contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config.schemas import (
    AnimaWorksConfig,
    CredentialConfig,
    LlmRateGuardConfig,
)
from core.exceptions import LLMAPIError
from core.execution.rate_guard import LlmRateGuard
from core.schemas import ModelConfig
from tests.helpers.mocks import make_litellm_response


class _RateLimit429(Exception):
    def __init__(self) -> None:
        super().__init__("Too Many Requests")
        self.status_code = 429


@pytest.fixture
def shared_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the runtime data dir and reset the guard singleton/config cache."""
    import core.execution.rate_guard as rate_guard
    from core.config import invalidate_cache

    monkeypatch.setenv("ANIMAWORKS_DATA_DIR", str(tmp_path))
    invalidate_cache()
    rate_guard._shared_guard = None
    yield tmp_path
    rate_guard._shared_guard = None
    invalidate_cache()


def _make_executor(anima_dir: Path):
    from core.execution.litellm_loop import LiteLLMExecutor
    from core.memory import MemoryManager
    from core.tooling.handler import ToolHandler

    anima_dir.mkdir(parents=True, exist_ok=True)
    (anima_dir / "permissions.md").write_text("", encoding="utf-8")
    (anima_dir / "identity.md").write_text("# Test", encoding="utf-8")
    for sub in ["episodes", "knowledge", "procedures", "skills", "state", "shortterm"]:
        (anima_dir / sub).mkdir(exist_ok=True)

    memory = MagicMock(spec=MemoryManager)
    memory.read_permissions.return_value = ""
    memory.search_memory_text.return_value = []
    memory.anima_dir = anima_dir

    model_config = ModelConfig(
        model="anthropic/claude-sonnet-4-6",
        api_key="sk-test",
        max_tokens=1024,
        context_threshold=0.50,
        max_chains=2,
    )
    tool_handler = ToolHandler(anima_dir=anima_dir, memory=memory, tool_registry=[])
    return LiteLLMExecutor(
        model_config=model_config,
        anima_dir=anima_dir,
        tool_handler=tool_handler,
        tool_registry=[],
        memory=memory,
    )


def _guard_file(data_dir: Path) -> Path:
    return data_dir / "shared" / "llm_rate_guard.json"


async def test_rate_error_records_block_in_shared_file(shared_data_dir: Path) -> None:
    executor = _make_executor(shared_data_dir / "animas" / "aoi")
    mock = AsyncMock(side_effect=_RateLimit429())

    with (
        patch("core.execution.litellm_loop.decorrelated_jitter", return_value=0.0),
        patch("litellm.acompletion", mock),
        pytest.raises(LLMAPIError),
    ):
        await executor.execute("hello", system_prompt="sys")

    # Mode A authenticates with the API key, so the block is keyed on the
    # ``anthropic:api`` realm (not the Max-auth SDK realm).
    state = json.loads(_guard_file(shared_data_dir).read_text())
    assert "anthropic:api" in state
    assert state["anthropic:api"]["reason"] == "rate_limit"
    assert state["anthropic:api"]["blocked_until"] > 0


async def test_guarded_family_start_continues_not_deferred(
    shared_data_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A peer process has already recorded a block on the API realm.
    from core.execution.rate_guard import get_rate_guard

    get_rate_guard().report_block("anthropic:api", 300, "rate_limit")
    assert get_rate_guard().blocked_remaining("anthropic:api") > 0

    executor = _make_executor(shared_data_dir / "animas" / "yuki")
    resp = make_litellm_response(content="done")
    mock = AsyncMock(side_effect=[resp])

    with caplog.at_level("INFO", logger="animaworks.execution.litellm_loop"), patch("litellm.acompletion", mock):
        result = await executor.execute("hello", system_prompt="sys")

    # The session ran (not deferred) despite the active block ...
    assert "done" in result.text
    mock.assert_awaited()
    # ... and the guard was observed at session start.
    assert any("rate-guarded" in rec.message for rec in caplog.records)


def test_cross_process_writes_stay_valid(shared_data_dir: Path) -> None:
    guard_path = _guard_file(shared_data_dir)
    guard_path.parent.mkdir(parents=True, exist_ok=True)

    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(Path.cwd())!r})
        from core.config.schemas import LlmRateGuardConfig
        from core.execution.rate_guard import LlmRateGuard

        family = sys.argv[1]
        guard = LlmRateGuard(config=LlmRateGuardConfig(), path=Path({str(guard_path)!r}))
        for _ in range(40):
            guard.report_block(family, 120, "rate_limit")
        """
    )

    procs = [subprocess.Popen([sys.executable, "-c", script, fam]) for fam in ("anthropic", "openai")]
    for p in procs:
        assert p.wait(timeout=60) == 0

    state = json.loads(guard_path.read_text())
    assert isinstance(state, dict)
    assert any(fam in state for fam in ("anthropic", "openai"))


def test_quota_block_is_not_clamped_to_transient_max(
    shared_data_dir: Path,
) -> None:
    guard = LlmRateGuard(
        config=LlmRateGuardConfig(
            max_block_seconds=600,
            quota_block_seconds=1800,
        ),
        path=_guard_file(shared_data_dir),
    )

    guard.report_block("openai:codex", 1800, "quota_exhausted")

    assert guard.blocked_remaining("openai:codex") > 1700


def test_grok_quota_block_selects_sonnet_fallback(
    shared_data_dir: Path,
) -> None:
    from core.config.model_config import resolve_effective_model_config
    from core.execution.grok_cli import _grok_error_metadata

    primary = ModelConfig(
        model="grok/grok-4.5",
        execution_mode="X",
        resolved_mode="X",
        credential="grok",
        fallback_models=["s:claude-sonnet-4-6"],
    )
    config = AnimaWorksConfig(
        credentials={
            "anthropic": CredentialConfig(type="claude_code_login"),
        },
    )
    message = "API error (status 402 Payment Required): Grok Build usage balance exhausted"

    with patch("core.config.io.load_config", return_value=config):
        metadata = _grok_error_metadata(message, primary.model)
        effective = resolve_effective_model_config(primary)

    state = json.loads(_guard_file(shared_data_dir).read_text(encoding="utf-8"))
    assert metadata == {"terminal": True, "reason": "quota_exhausted"}
    assert state["grok:grok"]["reason"] == "quota_exhausted"
    assert effective.model == "claude-sonnet-4-6"
    assert effective.resolved_mode == "S"
