"""Visibility matrix for MCP-backed modes and the Mode A/B schema builder."""

from __future__ import annotations

import pytest

# Triggers observed at executor call sites (message/inbox/heartbeat/cron/task,
# explicit background task-authoring, and memory consolidation).
TRIGGER_CASES = (
    ("chat", "scoped", "interactive"),
    ("message:user", "message", "interactive"),
    ("inbox:alice", "scoped", "normal"),
    ("heartbeat", "heartbeat", "heartbeat"),
    ("cron:daily", "scoped", "normal"),
    ("task:task-42", "scoped", "normal"),
    ("background:manual", "background", "background"),
    ("submit_tasks:manual", "background", "background"),
    ("consolidation:daily", "consolidation", "consolidation"),
)
MCP_MODES = ("S", "C", "D", "G", "X", "Y")
BUILDER_MODES = ("A", "B")  # B is the legacy alias normalized to the A tool-use path.

# Expected names for a normal Anima without notification channels, subordinates,
# or the newstaff skill. MCP call_human now follows the builder's notification gate.
# Role-specific additions are applied below.
MCP_SCOPED = frozenset(
    {
        "archive_memory_file",
        "create_skill",
        "grant_workspace_access",
        "list_tasks",
        "read_memory_file",
        "report_knowledge_outcome",
        "report_procedure_outcome",
        "search_memory",
        "send_message",
        "update_task",
        "write_memory_file",
    }
)
MCP_FULL_WITH_SUBMIT = frozenset(
    {
        "archive_memory_file",
        "archive_skill",
        "block_skill",
        "create_skill",
        "curate_skills",
        "delete_skill",
        "grant_workspace_access",
        "list_tasks",
        "promote_procedure_to_skill",
        "read_memory_file",
        "report_knowledge_outcome",
        "report_procedure_outcome",
        "restore_skill",
        "search_memory",
        "send_message",
        "set_skill_lifecycle",
        "submit_tasks",
        "unblock_skill",
        "update_task",
        "write_memory_file",
    }
)
# grant_workspace_access now follows Mode A's stricter consolidation surface.
MCP_CONSOLIDATION = MCP_FULL_WITH_SUBMIT - {
    "grant_workspace_access",
    "send_message",
    "submit_tasks",
}

# Scoped triggers now hide curator tools in Mode A as MCP did before unification.
_BUILDER_SKILL_MANAGEMENT = frozenset(
    {
        "curate_skills",
        "archive_skill",
        "restore_skill",
        "block_skill",
        "unblock_skill",
        "delete_skill",
        "set_skill_lifecycle",
    }
)
_BUILDER_BASE = frozenset(
    {
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "Read",
        "WebFetch",
        "WebSearch",
        "Write",
        "create_skill",
        "grant_workspace_access",
        "list_tasks",
        "read_memory_file",
        "report_knowledge_outcome",
        "report_procedure_outcome",
        "search_memory",
        "send_message",
        "todo_write",
        "update_task",
        "write_memory_file",
    }
)
BUILDER_FULL = _BUILDER_BASE | _BUILDER_SKILL_MANAGEMENT
# During consolidation Mode A already withheld workspace, messaging, and
# delegation tools; the shared resolver applies the same restrictions to MCP.
BUILDER_CONSOLIDATION = BUILDER_FULL - {"grant_workspace_access", "send_message"}


def _expected_mcp(
    kind: str,
    *,
    has_subordinates: bool,
    has_newstaff: bool,
    include_notification_tools: bool,
) -> frozenset[str]:
    if kind == "consolidation":
        names = set(MCP_CONSOLIDATION)
    elif kind == "heartbeat":
        names = set(MCP_FULL_WITH_SUBMIT)
    else:
        names = set(MCP_SCOPED)
        if kind == "message":
            names.add("attach_file")
        if kind == "background":
            names.add("submit_tasks")
    if has_subordinates and kind != "consolidation":
        names.add("delegate_task")
    if has_newstaff:
        names.add("create_anima")
    if include_notification_tools:
        names.add("call_human")
    return frozenset(names)


def _expected_builder(
    kind: str,
    *,
    has_subordinates: bool,
    include_notification_tools: bool,
    trigger: str = "",
) -> frozenset[str]:
    if kind == "consolidation":
        names = set(BUILDER_CONSOLIDATION)
    elif kind == "heartbeat":
        names = set(BUILDER_FULL)
        names.add("submit_tasks")
    else:
        names = set(_BUILDER_BASE)
        if kind == "interactive":
            names.add("trust_skill")
        if trigger.strip().startswith("message:"):
            names.add("attach_file")
        if kind == "background":
            names.add("submit_tasks")
    if include_notification_tools:
        names.add("call_human")
    if has_subordinates and kind != "consolidation":
        names.update({"delegate_task", "ping_subordinate"})
    return frozenset(names)


@pytest.mark.parametrize(("trigger", "mcp_kind", "builder_kind"), TRIGGER_CASES)
async def test_resolver_and_mcp_tool_surface_table(
    trigger: str,
    mcp_kind: str,
    builder_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every MCP-backed mode resolves to the same names as the MCP endpoint."""
    import core.mcp.server as mcp
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    monkeypatch.delenv("ANIMAWORKS_MCP_TOOLS", raising=False)
    monkeypatch.delenv("ANIMAWORKS_ENABLE_SUBMIT_TASKS", raising=False)
    for key in (
        "ANIMAWORKS_REQUEST_ID",
        "ANIMAWORKS_SESSION_TYPE",
        "ANIMAWORKS_THREAD_ID",
        "ANIMAWORKS_TOOL_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANIMAWORKS_TRIGGER", trigger)

    tools, _exposed = mcp._build_mcp_tools()
    monkeypatch.setattr(mcp, "MCP_TOOLS", tools)
    monkeypatch.setattr(mcp, "_trigger_scoped_tools_enabled", lambda: True)
    monkeypatch.setattr(mcp, "_is_consolidation_mode", lambda: mcp_kind == "consolidation")

    for has_subordinates in (False, True):
        for has_newstaff in (False, True):
            for include_notification_tools in (False, True):
                monkeypatch.setattr(mcp, "_is_supervisor", has_subordinates)
                monkeypatch.setattr(mcp, "_has_newstaff", has_newstaff)
                monkeypatch.setattr(
                    mcp,
                    "_has_notification_channels_for_anima",
                    lambda enabled=include_notification_tools: enabled,
                )
                context = ToolSurfaceContext(
                    has_subordinates=has_subordinates,
                    has_newstaff_skill=has_newstaff,
                    include_notification_tools=include_notification_tools,
                    consolidation_mode=mcp_kind == "consolidation",
                )
                expected = _expected_mcp(
                    mcp_kind,
                    has_subordinates=has_subordinates,
                    has_newstaff=has_newstaff,
                    include_notification_tools=include_notification_tools,
                )
                actual = frozenset(tool.name for tool in await mcp.list_tools())
                for mode in MCP_MODES:
                    assert set(resolve_tool_surface(context, trigger, mode)) == expected, (
                        trigger,
                        mode,
                        has_subordinates,
                        has_newstaff,
                        include_notification_tools,
                    )
                    assert actual == expected, (trigger, mode, has_subordinates, has_newstaff)


@pytest.mark.parametrize(("trigger", "mcp_kind", "builder_kind"), TRIGGER_CASES)
def test_resolver_and_builder_tool_surface_table(
    trigger: str,
    mcp_kind: str,
    builder_kind: str,
) -> None:
    """Mode A/B schemas are exactly the names selected by the shared resolver."""
    from core.tooling.policy.schemas import build_unified_tool_list
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    for has_subordinates in (False, True):
        for include_notification_tools in (False, True):
            context = ToolSurfaceContext(
                has_subordinates=has_subordinates,
                include_notification_tools=include_notification_tools,
            )
            expected = _expected_builder(
                builder_kind,
                has_subordinates=has_subordinates,
                include_notification_tools=include_notification_tools,
                trigger=trigger,
            )
            actual = frozenset(
                tool["name"]
                for tool in build_unified_tool_list(
                    include_supervisor_tools=has_subordinates,
                    include_notification_tools=include_notification_tools,
                    trigger=trigger,
                )
            )
            for mode in BUILDER_MODES:
                assert set(resolve_tool_surface(context, trigger, mode)) == expected, (
                    trigger,
                    mode,
                    has_subordinates,
                    include_notification_tools,
                )
                assert actual == expected, (trigger, mode, has_subordinates, include_notification_tools)


@pytest.mark.parametrize(
    "trigger",
    ("", "chat", "heartbeat", "inbox:alice", "cron:daily", "task:task-42", "consolidation:daily"),
)
def test_attach_file_is_hidden_outside_message_replies(trigger: str) -> None:
    from core.tooling.policy.schemas import build_unified_tool_list
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    context = ToolSurfaceContext()
    assert "attach_file" not in resolve_tool_surface(context, trigger, "S")
    assert "attach_file" not in resolve_tool_surface(context, trigger, "A")
    assert "attach_file" not in {tool["name"] for tool in build_unified_tool_list(trigger=trigger)}


def test_attach_file_is_visible_for_message_replies() -> None:
    from core.tooling.policy.schemas import build_unified_tool_list
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    context = ToolSurfaceContext()
    assert "attach_file" in resolve_tool_surface(context, "message:user", "S")
    assert "attach_file" in resolve_tool_surface(context, "message:user", "A")
    assert "attach_file" in {tool["name"] for tool in build_unified_tool_list(trigger="message:user")}


def test_consolidation_disagreement_is_resolved_by_the_stricter_surface() -> None:
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    # Mode A already omitted workspace tools in consolidation, so MCP now hides grant_workspace_access too.
    context = ToolSurfaceContext(has_subordinates=True, include_notification_tools=True)
    mcp_names = set(resolve_tool_surface(context, "consolidation:daily", "S"))
    builder_names = set(resolve_tool_surface(context, "consolidation:daily", "A"))
    assert "grant_workspace_access" not in mcp_names
    assert "grant_workspace_access" not in builder_names
    assert "delegate_task" not in mcp_names
    assert "delegate_task" not in builder_names


def test_mode_specific_schema_names_remain_distinct() -> None:
    from core.tooling.policy.surface import ToolSurfaceContext, resolve_tool_surface

    context = ToolSurfaceContext(has_subordinates=True, include_notification_tools=True)
    mcp_names = set(resolve_tool_surface(context, "chat", "S"))
    builder_names = set(resolve_tool_surface(context, "chat", "A"))
    assert "archive_memory_file" in mcp_names and "archive_memory_file" not in builder_names
    assert "todo_write" in builder_names and "todo_write" not in mcp_names
