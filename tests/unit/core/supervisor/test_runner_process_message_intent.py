"""Unit tests for AnimaRunner intent normalization in process_message handler."""
# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.supervisor.runner import AnimaRunner


@pytest.mark.asyncio
async def test_handle_process_message_normalizes_none_intent(tmp_path: Path):
    runner = AnimaRunner(
        anima_name="sakura",
        socket_path=tmp_path / "sakura.sock",
        animas_dir=tmp_path / "animas",
        shared_dir=tmp_path / "shared",
    )
    runner.anima = MagicMock()
    runner.anima.process_message = AsyncMock(return_value="ok")
    runner.anima.drain_notifications.return_value = []

    result = await runner._handle_process_message(
        {"message": "hello", "from_person": "human", "intent": None},
    )

    assert result["response"] == "ok"
    runner.anima.process_message.assert_awaited_once_with(
        "hello",
        from_person="human",
        intent="",
        images=None,
        attachment_paths=None,
    )
