from __future__ import annotations

from unittest.mock import MagicMock


def test_runtime_session_context_env_roundtrip(monkeypatch):
    from core.execution.session_context import RuntimeSessionContext

    ctx = RuntimeSessionContext.create(
        session_type="heartbeat",
        thread_id="default",
        trigger="heartbeat",
    )
    for key, value in ctx.to_env().items():
        monkeypatch.setenv(key, value)

    restored = RuntimeSessionContext.from_env()
    assert restored is not None
    assert restored.request_id == ctx.request_id
    assert restored.session_type == "heartbeat"
    assert restored.thread_id == "default"
    assert restored.trigger == "heartbeat"
    assert restored.tool_session_id == ctx.tool_session_id


def test_runtime_session_scope_resets():
    from core.execution.session_context import RuntimeSessionContext, current_runtime_session, runtime_session_scope

    ctx = RuntimeSessionContext.create(
        session_type="chat",
        thread_id="thread-a",
        trigger="message:mio",
    )

    assert current_runtime_session() is None
    with runtime_session_scope(ctx):
        assert current_runtime_session() == ctx
    assert current_runtime_session() is None


def test_agent_sdk_env_includes_runtime_session(tmp_path):
    from core.execution.agent_sdk import AgentSDKExecutor
    from core.execution.session_context import RuntimeSessionContext, runtime_session_scope
    from core.schemas import ModelConfig

    executor = AgentSDKExecutor(ModelConfig(model="claude-sonnet-4-6"), tmp_path)
    ctx = RuntimeSessionContext.create(
        session_type="cron",
        thread_id="default",
        trigger="cron:daily",
    )

    with runtime_session_scope(ctx):
        env = executor._build_env()
        mcp_env = executor._build_mcp_env()

    for built_env in (env, mcp_env):
        assert built_env["ANIMAWORKS_REQUEST_ID"] == ctx.request_id
        assert built_env["ANIMAWORKS_SESSION_TYPE"] == "cron"
        assert built_env["ANIMAWORKS_THREAD_ID"] == "default"
        assert built_env["ANIMAWORKS_TRIGGER"] == "cron:daily"
        assert built_env["ANIMAWORKS_TOOL_SESSION_ID"] == ctx.tool_session_id


def test_no_runtime_session_env_when_unscoped(tmp_path):
    from core.execution.agent_sdk import AgentSDKExecutor
    from core.schemas import ModelConfig

    executor = AgentSDKExecutor(ModelConfig(model="claude-sonnet-4-6"), tmp_path)
    env = executor._build_env()
    mcp_env = executor._build_mcp_env()

    for built_env in (env, mcp_env):
        assert "ANIMAWORKS_REQUEST_ID" not in built_env
        assert built_env["ANIMAWORKS_ANIMA_DIR"] == str(tmp_path)


def test_tool_handler_bind_runtime_session_clears_new_context_only(tmp_path):
    from core.execution.session_context import RuntimeSessionContext
    from core.tooling.handler import ToolHandler

    handler = ToolHandler(anima_dir=tmp_path, memory=MagicMock())
    ctx = RuntimeSessionContext.create(
        session_type="heartbeat",
        thread_id="default",
        trigger="heartbeat",
    )

    handler.bind_runtime_session(ctx)
    handler._replied_to["heartbeat"].add("owner")
    handler._posted_channels["heartbeat"].add("ops")

    handler.bind_runtime_session(ctx)
    assert handler._replied_to["heartbeat"] == {"owner"}
    assert handler._posted_channels["heartbeat"] == {"ops"}

    new_ctx = RuntimeSessionContext.create(
        session_type="heartbeat",
        thread_id="default",
        trigger="heartbeat",
    )
    handler.bind_runtime_session(new_ctx)

    assert handler._replied_to["heartbeat"] == set()
    assert handler._posted_channels["heartbeat"] == set()
