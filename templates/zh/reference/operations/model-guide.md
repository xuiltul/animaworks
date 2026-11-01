# 模型选择・配置指南

关于 AnimaWorks 模型配置的综合指南。
讲解执行模式、支持的模型、配置方法以及上下文窗口的机制。

---

## 执行模式

AnimaWorks 会根据模型名称自动判定执行模式。共有 6 种执行模式：

| 模式 | 名称 | 概要 | 目标模型示例 |
|--------|------|------|-------------|
| **S** | SDK | 经由 Claude Agent SDK。功能最强大 | `claude-opus-5-5`, `claude-sonnet-5-5` |
| **C** | Codex | 经由 Codex CLI | `codex/o4-mini`, `codex/gpt-4.1` |
| **D** | Cursor Agent | 经由 Cursor Agent CLI（`cursor-agent`）。MCP 集成 | `cursor/*` |
| **G** | Gemini CLI | 经由 Gemini CLI。MCP 集成 | `gemini/*` |
| **X** | Grok Build | 经由 Grok Build CLI 包装器（ACP stdio） | `grok/*` |
| **A** | Autonomous | LiteLLM + tool_use 循环 | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `ollama/qwen3:14b` |

### 模式判定的优先级

1. Per-anima `status.json` 的 `execution_mode` 显式指定
2. `~/.animaworks/models.json`（用户可编辑）
3. `config.json` `model_modes`（不推荐）
4. 代码默认的模式匹配
5. 未知 → Mode A

使用 Mode X 之前，需要安装 `grok` CLI，并通过 `grok login` 进行认证。

---

## 支持的模型列表

可通过 `animaworks models list` 显示最新列表。主要模型：

### Claude / Anthropic（Mode S）

| 模型 | 说明 |
|--------|------|
| `claude-opus-5-5` | 最高性能・推荐 |
| `claude-sonnet-5-5` | 均衡型・推荐 |
| `claude-haiku-4-5-20251001` | 轻量・高速 |

### OpenAI（Mode A）

| 模型 | 说明 |
|--------|------|
| `openai/gpt-4.1` | 最新・编码能力强 |
| `openai/gpt-4.1-mini` | 高速・低成本 |
| `openai/o3-2025-04-16` | 推理专用 |

### Google Gemini（Mode A）

| 模型 | 说明 |
|--------|------|
| `google/gemini-2.5-pro` | 最高性能 |
| `google/gemini-2.5-flash` | 高速均衡 |

### Grok Build（Mode X）

| 模型 | 说明 |
|--------|------|
| `grok/grok-4.5` | 经由 ACP stdio 的 Grok Build CLI |
| `grok/grok-composer-2.5-fast` | 高速的 Grok Build 模型 |

### Azure OpenAI（Mode A）

| 模型 | 说明 |
|--------|------|
| `azure/gpt-4.1-mini` | Azure OpenAI |
| `azure/gpt-4.1` | Azure OpenAI |

### Vertex AI（Mode A）

| 模型 | 说明 |
|--------|------|
| `vertex_ai/gemini-2.5-flash` | Vertex AI Flash |
| `vertex_ai/gemini-2.5-pro` | Vertex AI Pro |

### 本地模型 / vLLM / Ollama

| 模型 | 模式 | 说明 |
|--------|--------|------|
| `openai/qwen3.5-35b-a3b` | A | **推荐** — 与 Sonnet 同等性能（已通过基准验证） |
| `ollama/qwen3:14b` | A | 中型・支持 tool_use |
| `ollama/glm-4.7` | A | 支持 tool_use |
| `ollama/gemma3:4b` | B | 轻量 |

### AWS Bedrock

| 模型 | 模式 | 说明 |
|--------|--------|------|
| `openai/zai.glm-4.7` | A | 经由 Bedrock Mantle。适合单次任务 |
| `bedrock/qwen.qwen3-next-80b-a3b` | A | 工具调用能力不足（不推荐） |

---

## 推荐的 OSS 模型（已通过基准验证）

### Qwen3.5-35B — 本地 GPU 推荐模型

`openai/qwen3.5-35b-a3b`（经由 vLLM）作为 AnimaWorks Mode A 智能体，是已通过基准验证的**推荐本地模型**。
取得了与 Claude Sonnet 4.6 同等的综合得分，**最适合作为 background_model**。

#### 基准数据（2026-03-11 实施）

测量条件：Mode A（LiteLLM tool_use 循环）统一，15 个任务×每个模型 3 次运行

| 模型 | T1 基本操作 | T2 多步骤 | T3 判断・错误 | 综合 | 平均时间 | 成本 |
|--------|:----------:|:----------------:|:--------------:|:----:|:-------:|:-----:|
| **Qwen3.5-35B (local)** | **100%** | **100%** | 60% | **88%** | 9.6s | **$0** |
| Claude Sonnet 4.6 | 100% | 100% | 60% | 88% | 8.5s | ~$0.015/task |
| GLM-4.7 (Bedrock) | 87% | 33% | 53% | 55% | 5.9s | ~$0.003/task |
| Qwen3-Next 80B (Bedrock) | 40% | 27% | 40% | 35% | 5.2s | ~$0.005/task |

#### 特别事项

- **T1（基本操作：文件 I/O、工具调用）**：与 Sonnet 完全一致，为 100%
- **T2（多步骤：CSV 汇总、JSON 解析→写入等）**：与 Sonnet 完全一致，为 100%
- **计算精度（T3-3）**：Qwen3.5 为 3/3，Sonnet 为 1/3，Qwen3.5 更优
- **提示词注入耐受性（T3-4）**：所有模型均为 0/3（需要框架层面的对策）
- 参数数量并不直接决定性能（35B 的 Qwen3.5 大幅优于 80B 的 Qwen3-Next）

#### 推荐配置

```bash
# vLLM credential設定
# config.json > credentials に追加:
# "vllm-local": { "api_key": "dummy", "base_url": "http://<vllm-host>:8000/v1" }

# models.json に追加
# "openai/qwen3.5*": { "mode": "A", "context_window": 64000 }

# background_model として設定（Chat=Sonnet, HB/Inbox/Cron=Qwen3.5）
animaworks anima set-background-model {名前} openai/qwen3.5-35b-a3b --credential vllm-local
```

#### vLLM 启动示例

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

#### 按用途选择模型的指南

| 用途 | 推荐模型 | 理由 |
|------|-----------|------|
| background_model（HB/Inbox/Cron） | **Qwen3.5-35B** | 成本 $0 且具备与 Sonnet 同等的稳定性 |
| foreground（人类 Chat） | Sonnet 4.6 | 错误处理的稳定性与日语质量 |
| TaskExec（委派任务执行） | Qwen3.5-35B | 成本 $0 且工具链稳定 |
| 轻量简单响应（分类・摘要） | GLM-4.7 | 最快但无法多步骤 |

---

## models.json

在 `~/.animaworks/models.json` 中定义每个模型的执行模式与上下文窗口。
可使用 fnmatch 通配符模式。

### 架构

```json
{
  "パターン": {
    "mode": "S" | "C" | "D" | "G" | "X" | "A" | "B",
    "context_window": トークン数
  }
}
```

### 示例

```json
{
  "claude-opus-5-5":    { "mode": "S", "context_window": 1000000 },
  "claude-sonnet-5-5":  { "mode": "S", "context_window": 1000000 },
  "claude-*":           { "mode": "S", "context_window": 200000 },
  "grok/*":             { "mode": "X", "context_window": 500000 },
  "openai/gpt-4.1*":   { "mode": "A", "context_window": 1000000 },
  "openai/*":           { "mode": "A", "context_window": 128000 },
  "ollama/gemma3*":     { "mode": "B", "context_window": 8192 }
}
```

更具体的模式优先。`claude-opus-5-5` 会先于 `claude-*` 匹配。

### 确认命令

```bash
animaworks models show       # models.json の内容表示
animaworks models info {モデル名}  # 解決結果の確認
```

---

## 模型变更步骤

### 变更特定 Anima 的模型

```bash
# 1. モデル設定（status.json を更新）
animaworks anima set-model {名前} {モデル名}

# 2. credential が必要な場合
animaworks anima set-model {名前} {モデル名} --credential {credential名}

# 3. サーバー起動中なら再起動
animaworks anima restart {名前}
```

### 批量变更所有 Anima

```bash
animaworks anima set-model --all {モデル名}
```

### 确认当前配置

```bash
animaworks anima info {名前}    # モデル・実行モード・credential等を表示
animaworks anima list --local   # 全Animaのモデル一覧
```

---

## 上下文窗口

### 解析顺序

1. `models.json` 的 `context_window`
2. `config.json` 的 `model_context_windows`（通配符模式）
3. 代码的硬编码默认值（`MODEL_CONTEXT_WINDOWS`）
4. 最终回退：128,000 个 token

### 阈值的自动缩放

压缩阈值会根据上下文窗口大小自动调整：

- **200K 以上**：按配置值原样使用（默认 0.50）
- **低于 200K**：向 0.98 线性缩放

对于小模型，仅系统提示词就会占据上下文的大部分，因此提高阈值以防止误触发。

---

## 各提供商的 credential 配置

### Anthropic（默认）

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-..."
    }
  }
}
```

### Azure OpenAI

```json
{
  "credentials": {
    "azure": {
      "api_key": "",
      "base_url": "https://YOUR_RESOURCE.openai.azure.com",
      "keys": { "api_version": "2024-12-01-preview" }
    }
  }
}
```

### Vertex AI

```json
{
  "credentials": {
    "vertex": {
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "us-central1",
        "vertex_credentials": "/path/to/service-account.json"
      }
    }
  }
}
```

### vLLM（本地 GPU 推理）

```json
{
  "credentials": {
    "vllm-local": {
      "api_key": "dummy",
      "base_url": "http://192.168.1.100:8000/v1"
    }
  }
}
```

配置 credential 后，将其与 Anima 关联：

```bash
animaworks anima set-model {名前} {モデル名} --credential {credential名}
```

---

## 后台模型（成本优化）

Heartbeat / Inbox / Cron 可以使用与主模型不同的轻量模型来执行。
设置 `background_model` 后，可大幅降低这些后台处理的成本。

### foreground / background 的区分

| 区分 | 使用模型 | 目标触发器 |
|------|-----------|-------------|
| **foreground** | 主模型（`model`） | `chat`（与人类对话）、`task:*`（TaskExec 实际作业） |
| **background** | `background_model`（未设置时为���模型） | `heartbeat`、`inbox:*`（Anima 间 DM）、`cron:*` |

Heartbeat / Inbox / Cron 主要以「判断・分诊」为目的，实际执行由 TaskExec（主模型）承担。

### 解析顺序

1. Per-anima `status.json` 的 `background_model`
2. `config.json` 的 `heartbeat.default_model`（全局默认）
3. 回退到主模型（`model`）

### 配置方法

```bash
# 特定Animaにbackground_model を設定
animaworks anima set-background-model {名前} claude-sonnet-5-5

# credential が異なるプロバイダの場合
animaworks anima set-background-model {名前} azure/gpt-4.1-mini --credential azure

# 全Animaに一括設定
animaworks anima set-background-model --all claude-sonnet-5-5

# background_model を削除（メインモデルにフォールバック）
animaworks anima set-background-model {名前} --clear

# サーバー起動中なら再起動
animaworks anima restart {名前}
```

### 在 status.json 中确认

```json
{
  "model": "claude-opus-5-5",
  "background_model": "claude-sonnet-5-5",
  "background_credential": null
}
```

当 `background_model` 未设置或与主模型相同时，会跳过切换。

---

## 角色模板与默认模型

在 `animaworks anima set-role` 中变更角色时，默认模型也会随之变更：

| 角色 | 默认模型 | background_model | context_threshold | conversation_history_threshold |
|--------|---------------|-----------------|-------------------|----------------------------------|
| engineer | claude-opus-5-5 | claude-sonnet-5-5 | 0.80 | 0.40 |
| manager | claude-opus-5-5 | claude-sonnet-5-5 | 0.60 | 0.30 |
| writer | claude-sonnet-5-5 | — | 0.70 | 0.30 |
| researcher | claude-sonnet-5-5 | — | 0.50 | 0.30 |
| ops | ollama/glm-4.7 | — | 0.50 | 0.30 |
| general | claude-sonnet-5-5 | — | 0.50 | 0.30 |

Opus 系角色（engineer, manager）会自动将 Sonnet 设置为 `background_model`。
Sonnet 以下的角色成本效率已经很好，因此 `background_model` 未设置。

---

## 常见问题

### 修改了模型却没有生效

`set-model` 通过 root API 更新 root 所有的 `status.json`，并重新加载运行中 Anima 的模型配置。已停止的 Anima 会在下次启动时读取新配置。不要从 Anima 进程直接编辑此文件。

### 编辑了 models.json 却没有生效

models.json 会根据文件的 mtime 自动重新加载。也可通过 `anima reload` 使其生效。

### 想增大上下文窗口

修改 `models.json` 的 `context_window`，或通过 `config.json` 的 `model_context_windows` 进行覆盖。

### 不知道该选哪个模型

- **需要高质量・自主执行** → `claude-opus-5-5`（Mode S）
- **均衡・重视成本** → `claude-sonnet-5-5`（Mode S）
- **低成本・大量处理** → `openai/gpt-4.1-mini`（Mode A）
- **本地 GPU・成本 $0** → `openai/qwen3.5-35b-a3b`（Mode A, vLLM）**推荐**
- **本地・轻量** → `ollama/qwen3:14b`（Mode A）

### 想降低 Heartbeat / Cron 的成本

配置 `background_model`。详情请参阅上文的“后台模型（成本优化）”章节。
如果主要使用 Opus，只需将 `background_model` 配置为 Sonnet，即可将 Heartbeat + Inbox 成本削减约73%。

在 vLLM 中将 `openai/qwen3.5-35b-a3b` 配置为 `background_model`，即可**将后台处理成本完全降为$0**。已确认其综合得分达到与 Sonnet 同等的88%。
