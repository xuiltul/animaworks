---
name: tool-creator
description: >-
  Meta-skill for creating AnimaWorks Python tool modules (core/tools/) with correct interfaces.
  Provides procedure for ExternalToolDispatcher integration, API key management,
  ToolHandler registration, and permissions.md allowance.
  Use when developing custom tools for Web API integration or external service integration.
  "create tool", "toolify", "new tool", "custom tool", "Python tool"
---

# tool-creator

## Procedure

### Step 1: Design the Tool

1. Decide the tool name (snake_case, e.g., `my_api_tool`)
2. Define the schema(s) (operations) to provide
3. Define required parameters

### Step 2: Create the Module File

Create a Python file following this template.

#### Single-Schema Tool (Simple)

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
        self._api_key = get_credential("myapi", "myapi_tool", env_var="MYAPI_KEY")

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

Save as a personal tool:
```
write_memory_file("tools/my_tool.py", <code>)
```

### Step 4: Enable the Tool

After saving, call `refresh_tools` for hot reload:
```
refresh_tools()
```

The tool becomes available immediately without restarting the session.

### Step 5: Share (Optional)

To let other Anima use it, share the tool:
```
share_tool(tool_name="my_tool")
```

This copies it to `common_tools/` and makes it available to all Anima.

## Required Interface

| Function | Required | Description |
|----------|----------|-------------|
| `get_tool_schemas()` | ✅ Required | Return tool schema list |
| `dispatch(name, args)` | 🔵 Recommended | Dispatch by schema name |
| Function with same name as schema | 🟡 Alternative | Can be used instead of `dispatch()` |
| `cli_main(argv)` | ⚪ Optional | For standalone CLI execution |
| `get_cli_guide()` | ⚪ Optional | CLI guide for agents |

## Schema Definition Conventions

```python
{
    "name": "tool_action_name",       # snake_case, prefix with tool name
    "description": "1-2 sentence description",  # Used by LLM for tool selection
    "input_schema": {                  # JSON Schema
        "type": "object",
        "properties": { ... },
        "required": [ ... ],
    },
}
```

## Validation Checklist

- [ ] Filename: snake_case, `.py` extension (e.g., `my_tool.py`)
- [ ] `from __future__ import annotations` at top of file
- [ ] `get_tool_schemas()` exists and returns a list
- [ ] Schema has `name`, `description`, `input_schema`
- [ ] `dispatch()` or same-name function exists
- [ ] Handler exists for all schemas
- [ ] Appropriate exceptions on error
- [ ] Timeout set for external APIs

## Security Guidelines

1. **Credentials**: Obtain via `get_credential()`. Never hardcode
   ```python
   from core.tools._base import get_credential
   api_key = get_credential("service_name", "tool_name", env_var="ENV_VAR_NAME")
   ```

2. **Access control**: Do not access other Anima's directories

3. **Timeout**: Always set timeout for external APIs (recommended: 30 seconds)

4. **Logging**: Use `logging.getLogger("animaworks.tools")`

5. **Dependencies**: Import external libraries inside functions (lazy import)

## Notes

- Tools are Python code, different from Skills (Markdown procedure documents)
- Tool creation requires "tool creation: yes" in permissions.md
- Created tools are discovered on the next `refresh_tools` call or session start
- Schema names must be globally unique (no conflict with other tools)
