from __future__ import annotations

"""Human-readable one-liner summaries for tool arguments.

Used by all execution engines to generate ``tool_detail`` streaming events
so the UI can show *what* a tool is doing while it runs.
"""

from typing import Any


def summarize_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    """Return a concise human-readable summary of *args* for *tool_name*.

    Returns empty string for unknown tools (caller should skip
    emitting ``tool_detail`` in that case).
    """
    match tool_name:
        case "Bash" | "execute_command":
            return (args.get("command") or "")[:120]
        case "Read" | "read_file":
            return args.get("file_path") or args.get("path") or ""
        case "Write" | "Edit" | "write_file" | "edit_file":
            return args.get("file_path") or args.get("path") or ""
        case "Grep" | "search_files":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            return f"{pattern} in {path}"
        case "Glob" | "glob_files":
            return args.get("pattern") or args.get("glob_pattern") or ""
        case "Task":
            return (args.get("description") or "")[:80]
        case "send_message":
            return f"→ {args.get('to', '')}"
        case "delegate_task":
            name = args.get("name", "")
            summary = (args.get("summary") or args.get("instruction", ""))[:60]
            return f"→ {name}: {summary}"
        case "web_search":
            return (args.get("query") or "")[:80]
        case "x_search":
            return (args.get("query") or "")[:80]
        case "skill":
            return args.get("name") or ""
        case "search_memory":
            return (args.get("query") or "")[:80]
        case "save_memory":
            return args.get("category", "")
        case "read_channel":
            return f"#{args.get('channel', '')}"
        case "post_channel":
            return f"#{args.get('channel', '')}"
        case "manage_channel":
            return f"{args.get('action', '')} #{args.get('channel', '')}"
        case _:
            return ""


def make_tool_detail_chunk(
    tool_name: str, tool_id: str, args: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a ``tool_detail`` chunk dict, or ``None`` if no summary."""
    detail = summarize_tool_args(tool_name, args)
    if not detail:
        return None
    return {
        "type": "tool_detail",
        "tool_id": tool_id,
        "tool_name": tool_name,
        "detail": detail,
    }
