# 项目配置方法

AnimaWorks 的配置结构与 Anima 添加步骤的参考。
在需要修改配置的场景中进行搜索与参考。

## 运行时目录的初始化

运行时数据放置在 `~/.animaworks/`（或 `ANIMAWORKS_DATA_DIR`）中。
首次设置时，使用 `animaworks init` 从模板进行初始化。

### init 命令

| 命令 | 说明 |
|---------|------|
| `animaworks init` | 初始化运行时目录（非交互）。若已存在则不做任何操作 |
| `animaworks init --force` | 将模板差异合并到现有数据。仅添加新文件，`prompts/` 会被覆盖 |
| `animaworks init --skip-anima` | 仅初始化基础设施。跳过 Anima 创建 |
| `animaworks init --template NAME` | 从模板以非交互方式创建 Anima |
| `animaworks init --from-md PATH` | 从 MD 文件以非交互方式创建 Anima |
| `animaworks init --blank NAME` | 以非交互方式创建空白 Anima |

**推荐流程**：执行 `animaworks init` 后，用 `animaworks start` 启动服务器，并通过 Web UI 的设置向导添加 Anima。

### 初始化时创建的目录

`ensure_runtime_dir`（`core/infra/runtime_init.py`）会创建以下目录:

- `animas/` — Anima 目录
- `shared/inbox/` — 接收消息队列
- `shared/users/` — 用户配置文件
- `shared/channels/` — 共享频道（general、ops 的初始文件）
- `tmp/attachments/` — 附件临时存储
- `common_skills/` / `common_knowledge/` — 通用技能与知识
- `prompts/` / `company/` — 提示词与组织模板
- `tool_prompts.sqlite3` — 工具提示词数据库
- `models.json` — 模型名称→运行模式映射（从 `config_defaults/` 复制）

每次启动时，`common_skills` 和 `common_knowledge` 都会从模板增量同步，仅添加新条目（保留现有文件）。

## config.json 的整体结构

AnimaWorks 的集成配置文件放置在 `~/.animaworks/config.json` 中。
所有配置均由 `AnimaWorksConfig` 模型定义，具有以下顶层字段。

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-5-5", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "max_tokens": 2000 },
  "image_gen": {}
}
```

**注意**：`animas` 部分仅保留组织布局（`supervisor`, `speciality`）。模型名・credential 等模型配置记录在各 Anima 的 `status.json` 中（参见后文「Anima 配置的解析」）。

各部分的角色：

| 部分 | 说明 |
|-----------|------|
| `version` | 配置模式版本（当前为 `1`） |
| `setup_complete` | 首次设置完成标志 |
| `locale` | UI 语言（`"ja"` / `"en"`） |
| `system` | 服务器模式・时区 |
| `credentials` | API 密钥・端点（命名） |
| `model_modes` | 模型名→执行模式的覆盖映射 |
| `anima_defaults` | 所有 Anima 通用的默认配置 |
| `animas` | Anima 的组织布局（supervisor, speciality）。模型配置为 status.json |
| `consolidation` | 记忆整合（每日/每周）的配置 |
| `rag` | RAG（嵌入向量搜索）的配置 |
| `priming` | 预激活（自动记忆获取）的 token 预算 |
| `image_gen` | 图像生成的样式配置 |

<!-- AUTO-GENERATED:START config_fields -->
### 配置项参考（自动生成）

#### Anima 配置 (per-anima overrides)

| 字段 | 类型 | 默认值 | 说明 |
|-----------|-----|----------|------|
| `supervisor` | `str | None` | None |  |
| `company` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |
| `heartbeat_enabled` | `bool | None` | None |  |
| `background_review_enabled` | `bool | None` | None |  |
| `token_budget_monthly` | `int | None` | None |  |
| `aliases` | `list[str]` | `[]` |  |

#### 默认值 (anima_defaults)

| 字段 | 类型 | 默认值 | 说明 |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-5-5"` |  |
| `fallback_model` | `str | None` | None |  |
| `fallback_models` | `list[str]` | `[]` |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `background_thinking_effort` | `str | None` | None |  |
| `voice_thinking_effort` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `context_absolute_ceiling` | `float` | `0.75` |  |
| `task_compaction_tokens` | `int` | `0` |  |
| `task_compaction_max` | `int` | `6` |  |
| `max_session_age_hours` | `float` | `24.0` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `extra_mcp_servers` | `dict[str, dict]` | `{}` |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `mode_s_auth` | `str | None` | None |  |
| `default_workspace` | `str` | `""` |  |
| `consolidation_enabled` | `bool` | `True` |  |
| `heartbeat_enabled` | `bool` | `True` |  |
| `token_budget_monthly` | `int | None` | None |  |

#### AnimaWorksConfig 顶层

| 部分 | 说明 |
|-----------|------|
| `version` | 配置文件版本 |
| `setup_complete` | 设置完成标志 |
| `locale` | 区域设置 |
| `system` | 系统配置（模式、日志级别） |
| `credentials` | API 认证信息 |
| `model_modes` | 模型名→执行模式映射 |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima 配置默认值 |
| `animas` | Anima 个别配置覆盖 |
| `consolidation` | 记忆整合配置 |
| `helper_models` |  |
| `background_review` |  |
| `rag` | RAG（检索增强生成）配置 |
| `gpu` |  |
| `memory` |  |
| `skills` |  |
| `chatwork_tool` |  |
| `prompt` |  |
| `priming` | 预激活（自动记忆回忆）配置 |
| `image_gen` | 图像生成配置 |
| `human_notification` |  |
| `interaction` |  |
| `server` |  |
| `llm_rate_guard` |  |
| `mcp` |  |
| `external_messaging` |  |
| `external_tasks` |  |
| `github_webhook` |  |
| `event_export` |  |
| `background_task` |  |
| `activity_log` |  |
| `logging` |  |
| `heartbeat` |  |
| `voice` |  |
| `phone` |  |
| `housekeeping` |  |
| `inbox` |  |
| `local_llm` |  |
| `workspaces` |  |
| `github_identities` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `icon_url_template` |  |
| `ui` |  |
| `cli` |  |
| `enclave` |  |
| `enclaves` |  |

<!-- AUTO-GENERATED:END -->

## 添加新 Anima 的方法

添加 Anima 的方法有 3 种。均通过 `animaworks anima create` 或 `animaworks init` 执行。
`animaworks anima create` 具有 `--role` 和 `--supervisor` 选项，推荐使用。

### 方法 1：从模板创建

使用预先定义的模板（`templates/ja/anima_templates/` 或 `templates/en/anima_templates/` 下）的方法。
模板包含 IDENTITY.md, SOUL.md, AGENTS.md, permissions.json、技能等。

```bash
# テンプレートから作成（テンプレート名はディレクトリ名）
animaworks anima create --template <テンプレート名>

# 名前を変えて作成
animaworks anima create --template <テンプレート名> --name <anima名>
```

模板是最推荐的方法。角色设定已完备，可跳过 bootstrap（首次启动的自我定义）。

### 方法 2：从 Markdown 文件创建

准备角色表（Markdown），并基于它生成 Anima。

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima 名（省略时从表中提取）
- `--role`: 角色模板（engineer, researcher, manager, writer, ops, general）。默认：general
- `--supervisor`: 上司 Anima 名（覆盖角色表中的「上司」）

Markdown 文件会作为 `character_sheet.md` 复制到 Anima 目录中。
表中的「基本信息」反映到 IDENTITY.md，「人格」反映到 SOUL.md（个人资料标题为 IDENTITY.md），「角色・行动方针」反映到 AGENTS.md。
从角色模板应用 permissions.json 和 specialty_prompt.md。

Markdown 文件 SHOULD 包含以下内容：
- `# Character: 名前` 格式的标题，或基本信息表中的「英名」行（用于自动提取名称）
- `## 基本情報` — 英名、上司、模型等表格。反映到 IDENTITY.md
- `## 人格` — 反映到 SOUL.md（个人资料类标题分配到 IDENTITY.md）
- `## 役割・行動方針` — 反映到 AGENTS.md

### 方法 3：空白创建

使用最小骨架文件创建 Anima。

```bash
animaworks anima create --name aoi
```

`--name` 为必填。空白创建会生成将 `{name}` 占位符替换为实际名称的骨架文件。
在首次启动的 bootstrap 中，智能体通过与用户对话自我定义角色。

### 创建后的目录结构

无论采用哪种方法，都会生成以下目录和文件：

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # プロフィール・外見（不変ベースライン）
├── SOUL.md              # 人格・価値観（不変ベースライン）
├── AGENTS.md            # 役割・行動指針（可変）
├── BOOTSTRAP.md         # 初回起動指示（完了後に削除）
├── permissions.json       # ツール・コマンド権限
├── HEARTBEAT.md         # ハートビート設定
├── cron.md              # 定時タスク設定
├── episodes/            # エピソード記憶（日別ログ）
├── knowledge/           # 意味記憶（学んだ知識）
├── procedures/          # 手続き記憶（手順書）
├── skills/              # 個人スキル
├── state/               # ワーキングメモリ
│   └── current_state.md  # 現在のタスク
└── shortterm/           # 短期記憶（セッション継続用）
    └── archive/
```

### Anima 名规范

Anima 名必须（MUST）遵循以下规则：
- 仅可使用小写英数字、连字符(`-`)、下划线(`_`)
- 首字符必须为英文字母（`a-z`）
- 不可以下划线开头（模板保留）
- 例：`aoi`, `taro-dev`, `worker01`

## 执行模式（S / C / D / G / A / B）

AnimaWorks 具有 **6** 种执行模式。由模型名自动判定，但也可通过 `status.json` 的 `execution_mode` 覆盖。

### Mode S (SDK): Claude Agent SDK

Claude 模型专用。使用 Claude Code 子进程，可实现最丰富的工具执行。

- **目标模型**：`claude-*`（例：`claude-sonnet-5-5`, `claude-opus-5-5`）
- **特点**：文件操作、Bash 执行、记忆的自主搜索全部经由 Claude Agent SDK 进行
- **credential**：使用 `anthropic`（MUST）

### Mode C (Codex): Codex CLI

通过 Codex CLI 包装器执行 OpenAI Codex 系列。

- **目标模型**: `codex/*`（例如: `codex/o4-mini`, `codex/gpt-4.1`）
- **特点**: MCP 工具与 AnimaWorks 外部工具的集成路径与 Mode S/D/G 同系
- **credential**: 遵循 Codex / OpenAI 侧的要求

### Mode D (Cursor Agent): Cursor Agent CLI

将 Cursor 的 `cursor-agent` CLI 作为子进程执行。通过 MCP 使用 AnimaWorks 工具。

- **目标模型**: `cursor/*`
- **特点**: 主机需要 CLI 和认证。失败时可切换到 Mode A（LiteLLM）
- **credential**: 依赖 Cursor / `agent login` 的认证状态

### Mode G (Gemini CLI): Gemini CLI

将 Google 的 `gemini` CLI 作为子进程执行。MCP 集成。

- **目标模型**: `gemini/*`
- **特点**: 需要 CLI 或 `GEMINI_API_KEY`。回退时在 Mode A 中可能会被重映射为 `gemini/` → `google/` 等
- **credential**: CLI 登录或 API 密钥

### Mode A (Autonomous): LiteLLM + tool_use 循环

面向支持 tool_use 的云端・本地模型。通过 LiteLLM 统一提供商。

- **目标模型**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b` 等
- **特点**: 通过 LiteLLM 运行 tool_use 循环。工具执行由框架调度
- **credential**: 指定与各提供商对应的 credential

Mode B 已废弃。不建议使用不支持 tool_use 的模型。配置中残留的 `execution_mode: "B"` 会被视为 Mode A。

### 模式自动判定的机制

可在 `~/.animaworks/models.json` 中显式添加映射。
未指定时，使用代码内的默认模式（fnmatch 格式）进行匹配。

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "A"
  }
}
```

判定优先级:
1. Anima 的 `execution_mode` 字段（per-anima override）
2. `~/.animaworks/models.json`（完全匹配 → 通配符）
3. `config.json` 的 `model_modes`（已弃用的回退）
4. 代码的默认模式（完全匹配 → 通配符）
5. 均不匹配时为 Mode A

## 凭据配置

API 密钥在 `credentials` 节中按名称管理。

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-...",
      "base_url": null
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": null
    },
    "ollama": {
      "api_key": "",
      "base_url": "http://localhost:11434"
    }
  }
}
```

每个 Anima 通过 `credential` 字段指定使用哪个 credential。

- `api_key` — API 密钥字符串。为空字符串时，尝试从环境变量回退
- `base_url` — 自定义端点。使用 Ollama 或代理时配置。默认由 `null` 提供

**安全**: config.json 以文件权限 `0600` 保存（MUST）。由于包含 API 密钥，防止其他用户读取。

## 权限配置（permissions.json）

通过每个 Anima 的 `permissions.json` 定义可用的工具・可访问的路径・可执行的命令。

```markdown
# Permissions: aoi

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
全般的なコマンド

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
- image_gen: yes
- web_search: yes
- slack: no
```

权限相关规则:
- 每个 Anima 在启动时读取自己的 `permissions.json`（MUST）
- ToolHandler 进行权限检查，阻止未允许的操作
- 外部工具（Slack, Gmail, GitHub 等）在 `外部ツール` 节中单独允许/拒绝
- `読める場所` / `書ける場所` 以自然语言描述，由 ToolHandler 解释

### 阻止命令

在 `permissions.json` 中记载 `## 実行できないコマンド` 节后，指定命令的执行会被阻止。
除系统整体的硬编码阻止列表（`rm -rf /` 等危险命令）之外，还会应用 Anima 单独的阻止列表。

```markdown
## 実行できないコマンド
rm -rf, docker rm, git push --force
```

管道中的命令也会被单独检查（例如: `cat file | rm -rf` 中 `rm -rf` 会被阻止）。

## Anima 配置的解析（2 层合并）

Anima 的模型配置以 **`status.json` 为 Single Source of Truth（SSoT）**。

### 配置解析的 2 层结构

| 优先级 | 来源 | 说明 |
|--------|--------|------|
| 1（最高优先） | `status.json` | 放置于各 Anima 目录。保存模型・执行参数的全部配置 |
| 2（回退） | `anima_defaults` | `config.json` 的整体默认值。应用于 `status.json` 中未设置的字段 |

`config.json` 的 `animas` 节仅保存 **组织布局**（`supervisor`, `speciality`）。
模型名・credential 等模型配置记录在 `status.json` 中。

### status.json 的结构

文件路径: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-5-5",
  "credential": "anthropic",
  "max_tokens": 16384,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### 模型变更

要变更模型，使用 CLI 命令:

```bash
animaworks anima set-model <anima名> <モデル名> [--credential <credential名>]

# 全 Anima のモデルを一括変更
animaworks anima set-model --all <モデル名>
```

上司变更部下的模型时，使用 `set_subordinate_model` 工具。

### 配置重载

在 root/CLI 允许的 `status.json` 配置变更后，为在不重启进程的情况下反映模型配置，使用 `reload` 命令。`set-model` 命令要求启动中的 Anima 进行 reload:

```bash
# 単一 Anima のリロード
animaworks anima reload <anima名>

# 全 Anima のリロード
animaworks anima reload --all
```

重载通过 IPC 即时反映（无停机）。运行中的会话以旧配置完成，从下一个会话起应用新配置。

**典型的配置变更工作流**:

1. 通过 `animaworks anima set-model <name> <model>` 变更模型（root API 更新 `status.json`，并要求启动中的 Anima 进行 reload）
2. 在不会被自动通知的 root/CLI 更新之后，使用 `animaworks anima reload <name>`

不要从 Anima 进程直接编辑 `status.json`。使用 root 所有的 CLI/API 操作。

### 默认值列表（anima_defaults）

| 字段 | 默认值 | 说明 |
|-----------|-------------|------|
| `model` | `claude-sonnet-5-5` | 使用的 LLM 模型 |
| `max_tokens` | `8192` | 单次响应的最大 token 数 |
| `credential` | `"anthropic"` | 使用的 credential 名 |
| `context_threshold` | `0.50` | 上下文使用率超过此阈值时，将短期记忆外部化 |

### 层级结构

仅通过 `supervisor` 字段定义组织层级（记载于 `config.json` 的 `animas` 节）。

- `supervisor: null` — 顶层 Anima（指挥链的最上位）
- `supervisor: "aoi"` — 作为 aoi 的部下运作

层级通过消息传递的指令・报告发挥作用。上司可将任务委派给部下，部下向上司报告结果。

## Anima 管理命令列表

用于日常 Anima 操作・管理的 CLI 命令。
在服务器启动中（`animaworks start` 已完成）的状态下执行。

| 命令 | 说明 | 停机 |
|---------|------|-----------|
| `animaworks anima list` | 显示所有 Anima 的列表与状态 | 无 |
| `animaworks anima status [name]` | 显示指定 Anima（省略时为整体）的进程状态 | 无 |
| `animaworks anima reload <name>` | 重新读取 status.json 并即时反映模型配置（不重启进程） | 无 |
| `animaworks anima reload --all` | 批量重载所有 Anima 的配置 | 无 |
| `animaworks anima restart <name>` | 完全重启 Anima 进程（用于反映代码变更） | 15-30 秒 |
| `animaworks anima set-model <name> <model>` | 更新 root 所有的 status，并 reload 启动中 Anima 的模型配置 | 无 |
| `animaworks anima set-model --all <model>` | 批量变更所有 Anima 的模型 | 无 |
| `animaworks anima enable <name>` | 启用休眠中的 Anima 并启动进程 | — |
| `animaworks anima disable <name>` | 使 Anima 休眠（停止进程，status.json 的 enabled=false） | — |
| `animaworks anima create` | 创建新的 Anima（`--from-md`, `--template`, `--blank`） | — |
| `animaworks anima delete <name>` | 删除 Anima（默认归档保存） | — |

### 服务器管理命令

| 命令 | 说明 |
|---------|------|
| `animaworks start` | 启动服务器 |
| `animaworks stop` | 停止服务器 |
| `animaworks restart` | 完全重启服务器（重新生成所有进程） |
| `animaworks status` | 显示系统整体的状态 |
| `animaworks reset` | 服务器停止后，删除运行时目录并重新初始化（破坏性） |
| `animaworks reset --restart` | 在上述之后，重启服务器 |

### reload / restart / system reload 的区分使用

| 命令 | 动作 | 停机 | 用例 |
|---------|------|-----------|-------------|
| `anima reload` | 通过 IPC 交换 ModelConfig | 无 | status.json 的模型/参数变更 |
| `anima restart` | 进程 kill → 重新生成 | 15-30 秒 | 反映代码变更、应对内存泄漏 |
| 服务器 restart | 重启所有 Anima + 检测新增 | 15-30 秒 | 反映 Anima 的添加/删除 |
