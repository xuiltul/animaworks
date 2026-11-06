---
name: tool-creator
description: >-
  A meta-skill for creating Python external tool modules for AnimaWorks.core/integrationsHandles integrations, get_credential, and permissions.
  Use when: core/integrationsWhen you need to add a new module, implement a Web API wrapper, or develop a custom tool called by animaworks-tool.
---


# tool-creator

## Overview

AnimaWorks tools are divided into three types:

| Type | Location | Discovery method |
|------|--------|----------|
| **Core tools** | `core/integrations/*.py` (files with the `_` prefix are excluded) | `discover_core_tools()` → `TOOL_MODULES` (package import) |
| **Shared tools** | `{data_dir}/common_tools/*.py` | `discover_common_tools()` |
| **Personal tools** | `{anima_dir}/tools/*.py` | `discover_personal_tools()` |

`{data_dir}` is usually `~/.animaworks/`.

- **Dispatch**: `ExternalToolDispatcher` has deprecated `_DISPATCH_TABLE` and standardized on `dispatch(name, args)` in each module (or a function with the same name as the schema) (`core/tooling/dispatch.py`).
- **Merging**: At startup, both `_discover_personal_tools()` and `refresh_tools` for `AgentCore` merge in the order **shared → personal**, with **personal tools overriding entries with the same name** (`{**common, **personal}`). The merged result is held in `_personal_tools` of `ExternalToolDispatcher` (the name is historical, but it **also includes shared tools**).
- **Core conflicts**: Files with the same name as core `TOOL_MODULES` are **skipped** when discovering shared and personal tools (warning log only).
- **Writing tool files**: When writing to `tools/*.py` via `write_memory_file`, the **tool_creation.personal** requirement of `permissions` (**`permissions.json` takes precedence**) must be met (`core/tooling/handler_memory.py`).

## Execution Path (How the LLM Calls It)

| Mode | Typical Path |
|--------|-----------|
| **A (LiteLLM, etc.)** | Integrated tool **`use_tool(tool_name, action, args)`** → module's `dispatch` (`core/tooling/handler.py`). Details are designed so that each tool's skill is read via **`read_memory_file`** (`USE_TOOL` of `core/tooling/schemas/skill.py`). |
| **S (Agent SDK)** | Via Claude Code's built-in **Bash** with `animaworks-tool <ツール> …`, or via MCP (only a curated subset is exposed to MCP; see "Adding Core Tools to the Repository" below). |
| **Anthropic fallback, etc.** | `include_use_tool=False` configuration may exist in `build_tool_list` → externally, this assumes **Bash + animaworks-tool** or skills. |

At startup, the merged map above is passed to `ToolHandler`, so common and personal tools placed **before process startup** are available from the beginning via `use_tool` / `ExternalToolDispatcher`. For **`.py` newly added during a session**, `use_tool` will not recognize the tool name unless a rescan is performed via `refresh_tools` (because the map is not updated).

## Procedure

### Step 1: Tool Design

1. Decide the tool name (module name) (snake_case, e.g., `my_api_tool`). This becomes the first argument of `animaworks-tool my_api_tool …`.
2. Decide the **actions** (subcommands). Schema names should generally follow **`{tool_name}_{action}`** (e.g., `myapi_query`). In `use_tool`, use `tool_name="myapi"`, `action="query"`.
3. Define parameters using JSON Schema (`input_schema` or `parameters`).

### Step 2: Creating the Module File

#### Single-Action Example

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_tool_schemas() -> list[dict]:
    """ツールスキーマを返す。個人・共有ツールでは必須推奨（スキーマ読み込み・ログ用）。"""
    return [
        {
            "name": "my_tool_action",
            "description": "このツールが何をするかの説明",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "パラメータの説明",
                    },
                    "param2": {
                        "type": "integer",
                        "description": "オプションパラメータ",
                        "default": 10,
                    },
                },
                "required": ["param1"],
            },
        }
    ]


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """スキーマ名に応じた処理を実行する（推奨）。"""
    args.pop("anima_dir", None)  # フレームワークから注入。必要なら Path(anima_dir) で利用
    if name == "my_tool_action":
        return _do_action(
            param1=args["param1"],
            param2=args.get("param2", 10),
        )
    raise ValueError(f"Unknown tool: {name}")


def _do_action(param1: str, param2: int = 10) -> dict[str, Any]:
    return {"result": f"Processed {param1} with {param2}"}
```

When calling from `animaworks-tool`, **`cli_main` must be implemented** afterward (see the "`cli_main`" section below).

#### Multiple Actions + Authentication (API Integration)

The resolution order for `get_credential(credential_name, tool_name, key_name="api_key", env_var=...)` is **`config.json`'s `credentials.{credential_name}`** (`api_key` or `keys[key_name]`) → **the `shared` section of `vault.json`** (the key name is the string passed as argument `env_var`) → **`shared/credentials.json` (legacy, key is `env_var`)** → **environment variable `env_var`** (`core/credentials.py`).

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_tool_schemas() -> list[dict]:
    return [
        {
            "name": "myapi_query",
            "description": "APIにクエリを送信して結果を取得する",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "limit": {"type": "integer", "description": "最大件数", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "myapi_post",
            "description": "APIにデータを送信する",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "送信データ"},
                },
                "required": ["data"],
            },
        },
    ]


class MyAPIClient:
    def __init__(self) -> None:
        from core.credentials import get_credential

        self._api_key = get_credential(
            "myapi",
            "myapi_tool",
            env_var="MYAPI_KEY",
        )

    def query(self, query: str, limit: int = 10) -> list[dict]:
        import httpx

        resp = httpx.get(
            "https://api.example.com/search",
            params={"q": query, "limit": limit},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def post(self, data: str) -> dict:
        import httpx

        resp = httpx.post(
            "https://api.example.com/data",
            json={"data": data},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()


def dispatch(name: str, args: dict[str, Any]) -> Any:
    args.pop("anima_dir", None)
    client = MyAPIClient()
    if name == "myapi_query":
        return client.query(query=args["query"], limit=args.get("limit", 10))
    if name == "myapi_post":
        return client.post(data=args["data"])
    raise ValueError(f"Unknown tool: {name}")
```

**Per-Anima authentication** (Chatwork, etc.): There is a pattern where the Anima name is taken from `args.get("anima_dir")` and an **Anima-specific key** like `CHATWORK_API_TOKEN__{anima_name}` is resolved via `resolve_env_style_credential(...)` (e.g., `resolve_identity` of `core/integrations/_chatwork_identity.py`. If not registered, raise an error without falling back). The same key naming can be used in custom tools.

#### `cli_main` (for animaworks-tool)

`animaworks-tool <tool_name> …` cannot run the CLI for core, common, or personal modules unless the module has `cli_main` (see `cli_dispatch` in `cli/tool_dispatch.py`). The typical pattern is to parse subcommands with `argparse` and call `dispatch(f"{tool}_{action}", args_dict)` internally. If you want to generate usage from a schema, also refer to `auto_cli_guide` in `core/integrations/_base.py`.

### Step 3: Saving the File

Personal tools:

```
write_memory_file(path="tools/my_tool.py", content=<コード>)
```

`tool_creation.personal` must be permitted.

### Step 4: Tool Activation (Hot Reload)

Only when adding or changing `tools/*.py` or `common_tools/*.py` **after** process startup:

```
refresh_tools()
```

The file-based map of `ExternalToolDispatcher` within the same session is rescanned, and new module names are resolved from `use_tool` (usually unnecessary for files that existed before startup).

### Step 5: Sharing (Optional)

```
share_tool(tool_name="my_tool")
```

Copied to `~/.animaworks/common_tools/`. `tool_creation.shared` is required. Other Animas each need their own `refresh_tools` (or automatic discovery at restart).

## Required Interfaces

| Function / Constant | Required | Description |
|-------------|------|------|
| `get_tool_schemas()` | **Strongly recommended** for personal and shared | For schema loading and guide generation. Even in core, some modules have empty lists (e.g., `web_search` is `[]`). **Important**: `ExternalToolDispatcher.dispatch` (the path that passes the schema name directly via `tool_use`) matches modules only for schemas included in the **`name` list of `get_tool_schemas()`** in core. Empty modules are not hit on the core side via that path. On the other hand, **`use_tool`** imports the module directly from `TOOL_MODULES` and calls `dispatch`, so **even with an empty schema list, execution is possible if `dispatch` exists**. Custom tools should be aware of both paths; defining the schema is usually safer. |
| `dispatch(name, args)` | **Recommended** | `ExternalToolDispatcher._call_module` takes priority. |
| Function with the same name as the schema name | Alternative | Used when `dispatch` is absent: `getattr(mod, name)(**args)`. |
| `cli_main(argv)` | **Required for CLI use** | `animaworks-tool` entry. |
| `EXECUTION_PROFILE` | Optional | `expected_seconds`, `background_eligible`; in core tools, **`gated: True`** can require an allowlist for send-type operations (`core/tooling/permissions.py`). |

## Invocation and schema names

- **`use_tool`**: Passed to the module’s `dispatch` (or a function with the same name) by `schema_name = f"{tool_name}_{action}"`. Authorization checks: **core**: `tool_registry` (`tool_name` must be included in the result of `get_permitted_tools`); **file-based (shared and personal)**: the merged `_personal_tools` must contain `tool_name` (`_handle_use_tool` of `core/tooling/handler.py`).
- **`animaworks-tool`**: If the first token is `submit`, it is submitted as a background job (see below). **Core** imports from `TOOL_MODULES` and `cli_main`; **shared and personal** load from a file and `cli_main`. An unknown first argument may fall back to the main CLI (`animaworks`) (`cli/tool_dispatch.py`).
- **Gated subcommands (core only)**: If the relevant action in `EXECUTION_PROFILE` has `"gated": True`, it is blocked in both the CLI and dispatch unless **`{tool_name}_{action}`** (e.g., `gmail_send`) is included in the allowed set of `permissions`. **File-based personal and shared tools** are not in `TOOL_MODULES`, so they are not subject to this gate mechanism.

## Schema Normalization

`_normalise_schema` of `core/tooling/schemas/loader.py` receives `input_schema` / `parameters` and normalizes them to `parameters` in the internal representation.

## permissions (tool_creation, external tools)

- **Loading**: `load_permissions(anima_dir)` (`core/config/schemas.py`). **`permissions.json` takes priority**. Only when absent, parse `permissions.md` to generate JSON and migrate (`migrate_permissions_md_to_json`).
- **Tool creation** (JSON example):

```json
{
  "version": 1,
  "tool_creation": {
    "personal": true,
    "shared": false
  }
}
```

The Markdown "Tool Creation" section (`個人ツール` / `共有ツール` lines) also becomes the same structure during migration.

- **External tools (core)**: `external_tools` collects, via `get_permitted_tools`, the **module names of core `TOOL_MODULES`** and the **`{tool}_{action}`** strings for gate release (e.g., `gmail_send`). In `use_tool`, core tools use names in this set on the `tool_registry` side; **personal and shared tools** are executed even outside the core set if the name exists in **`_personal_tools` after startup merge or `refresh_tools`** (files with the same name as core are skipped during discovery, so no conflict).

## EXECUTION_PROFILE

- **`background_eligible: True`**: `animaworks-tool submit <tool> <subcommand> …` registers the input from `task_type="command"` into the TaskStore, and `PendingTaskExecutor` retrieves and executes the attempt (see `_handle_submit` in `cli/tool_dispatch.py`). Execution results are also saved to `state/background_tasks/{task_id}.json`. Profile references are only applied to **importable core modules** (file tools are less likely to be flagged as warnings at submit time).
- **`gated: True`**: For the corresponding core tool actions, explicit permission for `tool_action` must be granted via permissions.

```python
EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline": {"expected_seconds": 1800, "background_eligible": True},
    "send": {"expected_seconds": 15, "background_eligible": False, "gated": True},
}
```

## When adding core tools to the repository

1. Add `core/integrations/{name}.py` (items beginning with `_` are not scanned).
2. `TOOL_MODULES` is registered automatically in `discover_core_tools()`. No manual list of `core/integrations/__init__.py` is needed.
3. The curated MCP allowlist is `core/tooling/surface.py`'s `MCP_TOOL_NAMES`. `resolve_tool_surface` filters by trigger and role according to the MCP usage mode (S / C / D / G / X). Examples: `search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`, `send_message`, `call_human`, `delegate_task`, `submit_tasks`, `update_task`, `create_skill`. Board reads and posts are done via Bash (`animaworks-tool board read/post`). **Core tools for external services such as Slack / Gmail / `web_search` do not appear in MCP** — they are usually accessed via **`use_tool` / Bash (`animaworks-tool`) / skills**.
4. Add tests to `tests/`. If schemas or reference documents are generated automatically, also check what is covered by `scripts/generate_reference.py`.
5. For destructive operations, consider updating `gated: True` and the description on the permissions side.

## Validation Checklist

- [ ] File name: snake_case, `.py`, no leading `_` (to be included in scanning)
- [ ] Add `from __future__ import annotations` at the top (project convention)
- [ ] `get_tool_schemas()` returns the correct schema name (personal, shared)
- [ ] Process all schemas via `dispatch` or the schema-name function
- [ ] If `anima_dir` is not used, avoid side effects with `args.pop("anima_dir", None)`
- [ ] Implement `cli_main` and verify operation with `animaworks-tool`
- [ ] Add `timeout=` for external HTTP
- [ ] Authentication uses `get_credential` (or per-anima resolution of the same type as core)
- [ ] Use `logging.getLogger(__name__)` for logging (recommended)

## Security

1. Do not embed secrets in code. Use `get_credential` / vault / config.
2. Do not touch other Animas' directories.
3. In core, design "write/send" operations with `gated` and permissions as a set.

## Reference Implementations

- Thin entry + `_client` / `_cli` split: `core/integrations/chatwork.py`, `slack.py`, `discord.py`
- Authentication and API: `core/integrations/gmail.py`, `github.py`, `notion.py`, `google_calendar.py`, `google_tasks.py`
- Long-running and pipeline: `core/integrations/image_gen.py` (facade, `image/` subpackage + `EXECUTION_PROFILE`)
- Search and local LLM: `core/integrations/web_search.py` (if `get_tool_schemas` is empty, it will not match via the core path in `ExternalToolDispatcher.dispatch`; `use_tool` is possible with `dispatch`), `x_search.py`, `local_llm.py`
- Dispatcher and CLI entry: `core/tooling/dispatch.py`, `cli/tool_dispatch.py` (`cli_dispatch` / `_handle_submit`)

## Notes

- Tools are executable Python. They are distinct from skills (Markdown).
- Only tools added **after startup** require **`refresh_tools`** (files that existed before startup are already scanned at startup).
- Personal and shared files with the same name as core files are not adopted.
