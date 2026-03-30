"""E2E-specific fixtures for constructing AgentCore and DigitalAnima instances."""

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.agent import AgentCore
from core.anima import DigitalAnima
from core.memory import MemoryManager
from core.messenger import Messenger


@pytest.fixture(autouse=True)
def _bypass_completion_gate():
    """Disable completion_gate enforcement for all e2e tests.

    E2E tests mock LLM responses with a fixed number of side effects.
    completion_gate injects an extra retry loop iteration when the gate
    is not called, causing StopIteration on the mock.  Since e2e tests
    test specific agent behaviors (not completion_gate itself), bypass
    enforcement globally here.  Unit-level gate tests live in tests/unit/.
    """
    with patch(
        "core.execution.litellm_loop.completion_gate_applies_to_trigger",
        return_value=False,
    ):
        yield


@pytest.fixture
def make_agent_core(data_dir: Path, make_anima):
    """Factory to create an AgentCore instance with isolated filesystem.

    Bypasses DigitalAnima to test AgentCore directly.
    """

    def _make(name: str = "test-agent", **kwargs: Any) -> AgentCore:
        anima_dir = make_anima(name, **kwargs)
        memory = MemoryManager(anima_dir)
        model_config = memory.read_model_config()
        messenger = Messenger(data_dir / "shared", name)
        return AgentCore(anima_dir, memory, model_config, messenger)

    return _make


@pytest.fixture
def make_digital_anima(data_dir: Path, make_anima):
    """Factory to create a DigitalAnima instance with isolated filesystem."""

    def _make(name: str = "test-anima", **kwargs: Any) -> DigitalAnima:
        anima_dir = make_anima(name, **kwargs)
        shared_dir = data_dir / "shared"
        return DigitalAnima(anima_dir, shared_dir)

    return _make
