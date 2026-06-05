from __future__ import annotations

import json
from unittest.mock import AsyncMock

from tests.helpers.mocks import patch_agent_sdk, patch_agent_sdk_streaming


async def test_heartbeat_cycle_has_isolated_runtime_metadata(make_agent_core):
    with patch_agent_sdk(response_text="HEARTBEAT_OK"):
        agent = make_agent_core(name="runtime-hb", model="claude-sonnet-4-6")
        agent._sdk_available = True
        agent._run_priming = AsyncMock(return_value=("", []))

        result = await agent.run_cycle("heartbeat prompt", trigger="heartbeat")

    assert result.session_type == "heartbeat"
    assert result.thread_id == "default"
    assert result.request_id
    assert result.tool_session_id

    log_entries = []
    for file in (agent.anima_dir / "prompt_logs").glob("*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("request_id") == result.request_id:
                log_entries.append(entry)
    assert log_entries
    assert {entry["session_type"] for entry in log_entries} == {"heartbeat"}
    assert {entry["thread_id"] for entry in log_entries} == {"default"}
    assert {entry["tool_session_id"] for entry in log_entries} == {result.tool_session_id}
    assert {entry["trigger"] for entry in log_entries} == {"heartbeat"}


async def test_streaming_cycle_done_carries_runtime_metadata(make_agent_core):
    with patch_agent_sdk_streaming(text_deltas=["HEART", "BEAT"]):
        agent = make_agent_core(name="runtime-stream", model="claude-sonnet-4-6")
        agent._sdk_available = True
        agent._run_priming = AsyncMock(return_value=("", []))

        events = []
        async for chunk in agent.run_cycle_streaming(
            "stream heartbeat",
            trigger="heartbeat",
            thread_id="default",
        ):
            events.append(chunk)

    final = [event for event in events if event["type"] == "cycle_done"][0]["cycle_result"]
    assert final["summary"] == "HEARTBEAT"
    assert final["session_type"] == "heartbeat"
    assert final["thread_id"] == "default"
    assert final["request_id"]
    assert final["tool_session_id"]
