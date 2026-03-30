---
name: tool-creator
description: >-
  Meta-skill for building AnimaWorks Python external tools: ExternalToolDispatcher, get_credential, and permissions.
  Use when: adding a module under core/tools, wrapping a Web API, or exposing commands via animaworks-tool.
---

# tool-creator

## Overview

AnimaWorks tools are categorized into three types:

| Type | Location | Discovery |
|------|----------|-----------|
| **Core tools** | `core/tools/*.py` | `discover_core_tools()` → TOOL_MODULES (fixed at startup) |
| **Shared tools** | `{data_dir}/common_tools/*.py` | discover_common_tools() |
| **Personal tools** | `{anima_dir}/tools/*.py` | discover_personal_tools() |

`{data_dir}` is typically `~/.animaworks/`. Personal and shared tools are re-scanned by `ExternalToolDispatcher` via `refresh_tools` and can be hot-reloaded. ToolHandler checks the tool creation permission in permissions.json when writing to `tools/*.py` with `write_memory_file`.

Personal and shared tools are invoked via **Bash** with `animaworks-tool <tool> <subcommand> [args]`. Schema name format is `{tool_name}_{action}` (e.g., `my_tool` + `query` → `my_tool_query`).

## Procedure

### Step 1: Design the Tool

1. Decide the tool name (snake_case, e.g., `my_api_tool`)
2. Define the schema(s) (operations) to provide
3. Define required parameters

### Step 2: Create the Module File

Create a Python file following the template below.

#### Single-Schema Tool (Simple)

For file `my_tool.py`, `animaworks-tool my_tool action [args]` invokes it; schema name `my_tool_action` is passed to dispatch.

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


def get_tool_schemas() -> list[dict]:
    """Return tool schema(s) (required)."""
    return [
        {
            "name": "my_tool_action",
            "description": "Description of what this tool does",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description",
                    },
                    "param2": {
                        "type": "integer",
                        "description": "Optional parameter",
                        "default": 10,
                    },
                },
                "required": ["param1"],
            },
        }
    ]


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Execute handling by schema name (recommended)."""
    args.pop("anima_dir", None)  # Injected by framework; unused in this tool
    if name == "my_tool_action":
        return _do_action(
            param1=args["param1"],
            param2=args.get("param2", 10),
        )
    raise ValueError(f"Unknown tool: {name}")


def _do_action(param1: str, param2: int = 10) -> dict[str, Any]:
    """Actual logic implementation."""
    # Implement here
    return {"result": f"Processed {param1} with {param2}"}
```

#### Multi-Schema Tool (API Integration, etc.)

Schema names use `{tool_name}_{action}` format. `animaworks-tool myapi query [args]` passes `myapi_query` to dispatch. File name: `myapi.py`.

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("animaworks.tools")


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": "myapi_query",
            "description": "Send query to API and get results",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "myapi_post",
            "description": "Send data to API",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Data to send"},
                },
                "required": ["data"],
            },
        },
    ]


class MyAPIClient:
    """API client."""

    def __init__(self) -> None:
        from core.tools._base import get_credential
        self._api_key = get_credential(
            "myapi", "myapi_tool", env_var="MYAPI_KEY",
        )

    def query(self, query: str, limit: int = 10) -> list[dict]:
        import httpx
        resp = httpx.get(
            "https://api.example.com/search",
            params={"q": query, "limit": limit},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def post(self, data: str) -> dict:
        import httpx
        resp = httpx.post(
            "https://api.example.com/data",
            json={"data": data},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def dispatch(name: str, args: dict[str, Any]) -> Any:
    args.pop("anima_dir", None)  # Injected by framework; unused in this tool
    client = MyAPIClient()
    if name == "myapi_query":
        return client.query(
            query=args["query"],
            limit=args.get("limit", 10),
        )
    elif name == "myapi_post":
        return client.post(data=args["data"])
    raise ValueError(f"Unknown tool: {name}")
```

### Step 3: Save the File

Save as a personal tool (path in `write_memory_file` is relative to anima_dir):

```
write_memory_file(path="tools/my_tool.py", content=<code>)
```

Writing to `tools/` requires **personal tool** permission in the "Tool creation" section of permissions.json.

### Step 4: Enable the Tool

After saving, call `refresh_tools` for hot reload:

```
refresh_tools()
```

The tool becomes available immediately without restarting the session. Personal tools do not need to be listed in permissions.json external_tools; once discovered by `refresh_tools`, they are callable via **Bash** with `animaworks-tool <tool> <subcommand>`.

### Step 5: Share (Optional)

To let other Anima use it, share the tool:

```
share_tool(tool_name="my_tool")
```

This copies it to `~/.animaworks/common_tools/` and makes it available to all Anima. Sharing requires **shared tool** permission in permissions.json.

## Required Interface

| Function | Required | Description |
|----------|----------|-------------|
| `get_tool_schemas()` | ✅ Required | Return list of tool schemas. Must include `name`, `description`, `input_schema` (or `parameters`) |
| `dispatch(name, args)` | 🔵 Recommended | Dispatch by schema name. ExternalToolDispatcher prefers this. Remove `anima_dir` with `args.pop("anima_dir", None)` when passing args to other functions |
| Function with same name as schema | 🟡 Alternative | Can be used instead of `dispatch()` |
| `cli_main(argv)` | ⚪ Optional | For standalone execution via `animaworks-tool <tool_name>` |
| `EXECUTION_PROFILE` | ⚪ Optional | For long-running tools. Enables background submission via `animaworks-tool submit` |

## Bash Invocation

Anima invokes personal/shared tools via **Bash** with `animaworks-tool <tool> <subcommand>`:

```bash
animaworks-tool myapi query "search term" [--limit 10]
```

`schema_name = tool_name + "_" + action` is passed to `dispatch(name, args)`. In the example above, `name="myapi_query"`.

## Schema Definition Conventions

Both `input_schema` and `parameters` are supported and normalized (`core/tooling/schemas._normalise_schema`).

```python
{
    "name": "tool_action_name",       # snake_case. Format: {tool_name}_{action}
    "description": "1-2 sentence description",  # Used by LLM for tool selection
    "input_schema": {                  # JSON Schema format (parameters also accepted)
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

## Credential Retrieval (get_credential)

Obtain API keys etc. via `get_credential()`. Never hardcode.

```python
from core.tools._base import get_credential

api_key = get_credential(
    credential_name="myapi",   # Key in config.json credentials
    tool_name="myapi_tool",   # For error messages
    key_name="api_key",       # Default. Can specify other keys in keys
    env_var="MYAPI_KEY",      # Fallback environment variable
)
```

**Resolution order**: config.json → vault.json (encrypted vault) → shared/credentials.json → environment variable. ToolConfigError if none found.

## Tool Creation Permission in permissions.json

Add the following to permissions.json for tool creation and sharing:

```markdown
## Tool creation
- Personal tools: yes
- Shared tools: yes
```

`OK`, `enabled`, or `true` are also valid instead of `yes`.

## Validation Checklist

- [ ] Filename: snake_case, `.py` extension (e.g., `my_tool.py`)
- [ ] `from __future__ import annotations` at top of file
- [ ] `get_tool_schemas()` exists and returns a list
- [ ] Schema names follow `{tool_name}_{action}` format (animaworks-tool integration)
- [ ] Schema has `name`, `description`, `input_schema` (or `parameters`)
- [ ] `dispatch()` or same-name function exists
- [ ] `args.pop("anima_dir", None)` in dispatch when passing args to other functions
- [ ] Handler exists for all schemas
- [ ] Appropriate exceptions raised on error
- [ ] Timeout set for external APIs

## Security Guidelines

1. **Credentials**: Obtain via `get_credential()`. Never hardcode

2. **Access control**: Do not access other Anima's directories

3. **Timeout**: Always set timeout for external APIs (recommended: 30 seconds)

4. **Logging**: Use `logging.getLogger("animaworks.tools")`

5. **Dependencies**: Import external libraries inside functions (lazy import)

## Mode S (MCP) exposure for core tools

When adding a module under `core/tools/`, only names listed in `core/mcp/server.py` `_EXPOSED_TOOL_NAMES` are exposed to Claude Code via MCP (curated subset). As of 2026-03, that set includes: `search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`, `send_message`, `post_channel`, `call_human`, `delegate_task`, `submit_tasks`, `update_task`, `create_skill`, `completion_gate` (pre-completion self-check). External-service core tools such as Slack, Gmail, and `web_search` are **not** on the MCP list; they are reached via `use_tool`, Bash (`animaworks-tool`), or skills.

## Notes

- Tools are Python code, different from Skills (Markdown procedure documents)
- Tool creation requires **personal tools: yes** in the "Tool creation" section of permissions.json
- Sharing tools requires **shared tools: yes** permission
- Created tools are discovered immediately on `refresh_tools` call (hot reload)
- Personal tools do not need to be in permissions.json external_tools; once discovered, they are available via **Bash** with `animaworks-tool <tool> <subcommand>`
- Schema names use `{tool_name}_{action}` format; keep them unique across tools
- Personal or shared tools with the same name as core tools are shadowed and skipped (`core/tools/__init__.py`)
