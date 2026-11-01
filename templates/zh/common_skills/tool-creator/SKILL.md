---
name: tool-creator
description: >-
  为AnimaWorks创建Python外部工具模块的元技能。core/integrations处理集成、get_credential、permissions。
  Use when: core/integrations需要新增模块、实现Web API包装器、开发从animaworks-tool调用的自定义工具时。
---


# tool-creator

## 概述

AnimaWorks的工具分为3类：

| 类型 | 放置位置 | 发现方法 |
|------|--------|----------|
| **核心工具** | `core/integrations/*.py`（排除`_`前缀的文件） | `discover_core_tools()` → `TOOL_MODULES`（包 import） |
| **共享工具** | `{data_dir}/common_tools/*.py` | `discover_common_tools()` |
| **个人工具** | `{anima_dir}/tools/*.py` | `discover_personal_tools()` |

`{data_dir}` 通常为 `~/.animaworks/`。

- **分发**：`ExternalToolDispatcher` 废除了 `_DISPATCH_TABLE`，统一为各模块的 `dispatch(name, args)`（或与schema名同名的函数）（`core/tooling/dispatch.py`）。
- **合并**：`AgentCore` 启动时的 `_discover_personal_tools()` 和 `refresh_tools` 均按 **共享→个人** 的顺序合并，**个人覆盖同名项**（`{**common, **personal}`）。合并结果保存在 `ExternalToolDispatcher` 的 `_personal_tools` 中（名称是historical但**也包含共享工具**）。
- **与核心的冲突**：与核心 `TOOL_MODULES` 同名的文件在发现共享·个人工具时会被**跳过**（仅记录警告日志）。
- **工具文件写入**：在 `write_memory_file` 中写入 `tools/*.py` 时，必须满足 `permissions`（**优先 `permissions.json`**）的 **tool_creation.personal**（`core/tooling/handler_memory.py`）。

## 执行路径（LLM如何调用）

| 模式 | 典型路径 |
|--------|-----------|
| **A（LiteLLM 等）** | 集成工具 **`use_tool(tool_name, action, args)`** → 模块的 `dispatch`（`core/tooling/handler.py`）。详细设计为在 **`read_memory_file`** 中读取各工具的技能（`core/tooling/schemas/skill.py` 的 `USE_TOOL`）。 |
| **S（Agent SDK）** | 通过 Claude Code 内置 **Bash** 执行 `animaworks-tool <ツール> …`，或经由 MCP（MCP 上仅加载精选子集。参见下文「向仓库添加核心工具时」）。 |
| **Anthropic 回退等** | `build_tool_list` 中可能存在 `include_use_tool=False` 的配置 → 外部以 **Bash + animaworks-tool** 或技能为前提。 |

启动时上述合并后的映射会传给 `ToolHandler`，因此**在进程启动前放置的**共享·个人工具从一开始就能通过 `use_tool` / `ExternalToolDispatcher` 引用。**仅在会话中新增的** `.py`，如果不通过 `refresh_tools` 重新扫描，`use_tool` 就无法识别工具名（因为映射未更新）。

## 步骤

### Step 1: 工具设计

1. 确定工具名（模块名）（snake_case，例：`my_api_tool`）。它将成为 `animaworks-tool my_api_tool …` 的第1个参数。
2. 确定**动作**（子命令）。schema名原则上为 **`{tool_name}_{action}`**（例：`myapi_query`）。在 `use_tool` 中为 `tool_name="myapi"`, `action="query"`。
3. 用 JSON Schema 定义参数（`input_schema` 或 `parameters`）。

### Step 2: 创建模块文件

#### 单动作示例

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

若要从 `animaworks-tool` 调用，之后**必须实现 `cli_main`**（参见下文「`cli_main`」节）。

#### 多动作 + 认证（API 集成）

`get_credential(credential_name, tool_name, key_name="api_key", env_var=...)` 的解析顺序为 **`config.json` 的 `credentials.{credential_name}`**（`api_key` 或 `keys[key_name]`）→ **`vault.json` 的 `shared` 节**（键名为通过参数 `env_var` 传入的字符串）→ **`shared/credentials.json`（legacy，键为 `env_var`）** → **环境变量 `env_var`**（`core/credentials.py`）。

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

**Per-Anima 认证**（Chatwork 等）：存在从 `args.get("anima_dir")` 获取 Anima 名，并通过 `resolve_env_style_credential(...)` 解析 `CHATWORK_API_TOKEN__{anima_name}` 这类 **Anima 专用键**的模式（`core/integrations/_chatwork_identity.py` 的 `resolve_identity` 等。未注册时不回退而是报错）。自定义工具也可使用同样的键命名方式。

#### `cli_main`（用于 animaworks-tool）

`animaworks-tool <tool_name> …` 无论核心·共享·个人，**若模块中没有 `cli_main` 则无法通过 CLI 执行**（`cli/tool_dispatch.py` 的 `cli_dispatch`）。常见做法是在 `argparse` 中解析子命令，内部调用 `dispatch(f"{tool}_{action}", args_dict)`。若想从 schema 生成用法，也可参考 `core/integrations/_base.py` 的 `auto_cli_guide`。

### Step 3: 保存文件

个人工具：

```
write_memory_file(path="tools/my_tool.py", content=<コード>)
```

需允许 `tool_creation.personal`。

### Step 4: 启用工具（热重载）

仅在进程启动**后**新增或修改了 `tools/*.py` 或 `common_tools/*.py` 时：

```
refresh_tools()
```

同一会话内 `ExternalToolDispatcher` 的基于文件的映射会被重新扫描，`use_tool` 可解析新的模块名（启动前已存在的文件通常不需要）。

### Step 5: 共享（可选）

```
share_tool(tool_name="my_tool")
```

会被复制到 `~/.animaworks/common_tools/`。需要 `tool_creation.shared`。其他 Anima 各自需要 `refresh_tools`（或重启时的自动发现）。

## 必需接口

| 函数 / 常量 | 必需 | 说明 |
|-------------|------|------|
| `get_tool_schemas()` | 个人·共享中**强烈推荐** | 用于schema读取·指南生成。核心中也有空列表的模块（例：`web_search` 为 `[]`）。**重要**：`ExternalToolDispatcher.dispatch`（在 `tool_use` 中直接传入schema名的路径）对于核心，**仅匹配 `get_tool_schemas()` 的 `name` 列表中包含的schema**。空模块在该路径下不会命中核心侧。另一方面，**`use_tool`** 会从 `TOOL_MODULES` 直接 import 模块并调用 `dispatch`，因此**即使schema列表为空，只要有 `dispatch` 就能执行**。自定义工具需兼顾两条路径，通常定义schema更安全。 |
| `dispatch(name, args)` | **推荐** | `ExternalToolDispatcher._call_module` 优先使用。 |
| 与schema名同名的函数 | 替代 | 当没有 `dispatch` 时使用 `getattr(mod, name)(**args)`。 |
| `cli_main(argv)` | **使用 CLI 时必需** | `animaworks-tool` 入口。 |
| `EXECUTION_PROFILE` | 任意 | `expected_seconds`, `background_eligible`，在核心工具中可通过 **`gated: True`** 将发送类等设为必须的允许列表（`core/tooling/permissions.py`）。 |

## 调用与schema名

- **`use_tool`**：通过 `schema_name = f"{tool_name}_{action}"` 传给模块的 `dispatch`（或同名函数）。允许判定为 **核心**：`tool_registry`（`get_permitted_tools` 的结果中包含 `tool_name`），**基于文件（共享·个人）**：合并后的 `_personal_tools` 中存在 `tool_name`（`core/tooling/handler.py` 的 `_handle_use_tool`）。
- **`animaworks-tool`**：若第1个token为 `submit` 则后台提交（见下文）。**核心**从 `TOOL_MODULES` import 并 `cli_main`，**共享·个人**从文件加载并 `cli_main`。未知的第1个参数有时会回退到主 CLI（`animaworks`）（`cli/tool_dispatch.py`）。
- **带门控的子命令（仅核心）**：若 `EXECUTION_PROFILE` 的对应动作有 `"gated": True`，则当 `permissions` 的允许集合中不包含 **`{tool_name}_{action}`**（例：`gmail_send`）时，CLI / 分发两者都会被阻止。**基于文件的个人·共享工具**没有 `TOOL_MODULES`，因此不在此门控机制范围内。

## schema规范化

`core/tooling/schemas/loader.py` 的 `_normalise_schema` 接收 `input_schema` / `parameters`，在内部表示中统一为 `parameters`。

## permissions（tool_creation·外部工具）

- **读取**：`load_permissions(anima_dir)`（`core/config/schemas.py`）。**优先 `permissions.json`**。仅在没有时解析 `permissions.md` 并生成·迁移 JSON（`migrate_permissions_md_to_json`）。
- **工具创建**（JSON 示例）：

```json
{
  "version": 1,
  "tool_creation": {
    "personal": true,
    "shared": false
  }
}
```

Markdown 的「工具创建」节（`個人ツール` / `共有ツール` 行）在迁移时也会变为相同结构。

- **外部工具（核心）**：`external_tools` 在 `get_permitted_tools` 中收集 **核心 `TOOL_MODULES` 的模块名** 以及用于解除门控的 **`{tool}_{action}`** 字符串（例：`gmail_send`）。在 `use_tool` 中，核心工具使用进入该集合的名称在 `tool_registry` 侧使用，**个人·共享工具**若在启动时合并或 `refresh_tools` 后的 **`_personal_tools` 中有名称**，则即使不在核心集合内也能执行（与核心同名的文件在发现时会被跳过，因此不会冲突）。

## EXECUTION_PROFILE

- **`background_eligible: True`**：`animaworks-tool submit <tool> <subcommand> …` 将 `task_type="command"` 的输入注册到 TaskStore，`PendingTaskExecutor` 获取尝试并执行（`cli/tool_dispatch.py` 的 `_handle_submit`）。执行结果也会保存到 `state/background_tasks/{task_id}.json`。profile 引用仅对**可 import 的核心模块**执行（文件工具容易成为 submit 时的警告对象之外）。
- **`gated: True`**：对于核心工具的对应动作，需要在 permissions 中显式允许 `tool_action`。

```python
EXECUTION_PROFILE: dict[str, dict[str, object]] = {
    "pipeline": {"expected_seconds": 1800, "background_eligible": True},
    "send": {"expected_seconds": 15, "background_eligible": False, "gated": True},
}
```

## 向仓库添加核心工具时

1. 添加 `core/integrations/{name}.py`（以 `_` 开头的不是扫描对象）。
2. `TOOL_MODULES` 通过 `discover_core_tools()` 自动注册。不需要 `core/integrations/__init__.py` 的手动列表。
3. MCP 的精选允许列表为 `core/tooling/surface.py` 的 `MCP_TOOL_NAMES`。`resolve_tool_surface` 在 MCP 使用模式（S / C / D / G / X）下根据触发器和角色进行筛选。例：`search_memory`, `read_memory_file`, `write_memory_file`, `archive_memory_file`, `send_message`, `post_channel`, `call_human`, `delegate_task`, `submit_tasks`, `update_task`, `create_skill`。**Slack / Gmail / `web_search` 等外部服务类核心工具不会出现在 MCP 中** — 通常走 **`use_tool` / Bash（`animaworks-tool`）/ 技能** 路径。
4. 在 `tests/` 中添加测试。若自动生成schema或参考文档，也需确认 `scripts/generate_reference.py` 的对象。
5. 破坏性操作需考虑 `gated: True` 和 permissions 侧的说明更新。

## 验证检查清单

- [ ] 文件名：snake_case、`.py`、开头无 `_`（以便纳入扫描对象）
- [ ] 在开头添加 `from __future__ import annotations`（项目规范）
- [ ] `get_tool_schemas()` 返回正确的schema名（个人·共享）
- [ ] 通过 `dispatch` 或schema名函数处理所有schema
- [ ] 若不使用 `anima_dir`，则通过 `args.pop("anima_dir", None)` 避免副作用
- [ ] 实现 `cli_main` 并通过 `animaworks-tool` 确认运行
- [ ] 外部 HTTP 添加 `timeout=`
- [ ] 认证使用 `get_credential`（或与核心同型的 per-anima 解析）
- [ ] 日志推荐使用 `logging.getLogger(__name__)`

## 安全

1. 不要将机密信息嵌入代码。使用 `get_credential` / vault / config。
2. 不要触碰其他 Anima 的目录。
3. 核心中「写入·发送」类需将 `gated` 与 permissions 配套设计。

## 参考实现

- 薄入口 + `_client` / `_cli` 拆分：`core/integrations/chatwork.py`, `slack.py`, `discord.py`
- 认证・API：`core/integrations/gmail.py`, `github.py`, `notion.py`, `google_calendar.py`, `google_tasks.py`
- 长时间・管道：`core/integrations/image_gen.py`（门面、`image/` 子包 + `EXECUTION_PROFILE`）
- 搜索・本地 LLM：`core/integrations/web_search.py`（`get_tool_schemas` 为空 → `ExternalToolDispatcher.dispatch` 的核心路径中不匹配。`use_tool` 可在 `dispatch` 中实现）、`x_search.py`, `local_llm.py`
- 调度器・CLI 入口：`core/tooling/dispatch.py`, `cli/tool_dispatch.py`（`cli_dispatch` / `_handle_submit`）

## 注意事项

- 工具是可执行的 Python。与技能（Markdown）是两回事。
- 仅在**启动后**添加的工具需要 **`refresh_tools`**（启动前已存在的文件已在启动时扫描）。
- 与核心同名的个人・共享文件不会被采用。
