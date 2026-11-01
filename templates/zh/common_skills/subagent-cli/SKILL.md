---
name: subagent-cli
description: >-
  通过 Bash 非交互式运行外部 AI 代理 CLI 的技能。提供使用 codex exec 或 cursor-agent 委派编码工作的步骤。
  Use when: 委派复杂实现、代码审查、多文件批量修改、从 Bash 启动子代理时。
---


# subagent-cli

通过 Bash 将外部 AI 代理 CLI 作为子进程运行，委派复杂的编码任务。
在保持自身身份、判断和记忆的同时，作为扩展执行能力的“强力工具”使用。

## 与框架执行模式的关系

此技能仅在 **Bash 工具可用时** 适用。

| 模式 | 实现 | Bash | 此技能的适用 |
|--------|------|------|------------------|
| **Mode S** | `core/execution/engines/claude/executor.py` (Claude Agent SDK) | 默认可用 | 适用。在 Claude Code 子进程内可使用 Read/Write/Edit/Bash/Grep/Glob/WebFetch/WebSearch + MCP(send_message 等) + Task/Agent。Bash 执行时的 cwd 为 anima_dir |
| **Mode C** | `core/execution/engines/codex/executor.py` (Codex SDK) | 取决于 Codex CLI 的工具集 | **不需要 codex exec** — 框架直接运行 Codex。cursor-agent / claude -p 可通过 Bash 调用（Bash 可用时） |
| **Mode D** | Cursor Agent（cursor-agent 子进程） | 取决于 Cursor CLI 的工具集 | **不需要 cursor-agent -p** — 框架直接运行 cursor-agent。MCP 集成。工具访问接近 Mode S，但实体是 cursor-agent 二进制。codex exec / claude -p 可通过 Bash 调用（Bash 可用时） |
| **Mode G** | Gemini CLI（gemini 子进程） | 取决于 Gemini CLI 的工具集 | **不需要手动启动 Gemini CLI** — 框架直接运行。MCP 集成、stream-json 输出。其他 CLI 可通过 Bash 调用（Bash 可用时） |
| **Mode A/B** | LiteLLM + tool_use / 单次执行 | 仅在 permissions.json 允许时 | 若有 Bash 许可则适用 |

**重要**: Mode C（`codex/*`）、Mode D（`cursor/*`）、Mode G（`gemini/*`）的 Anima 由框架直接运行各引擎。此时无需自己从 Bash 调用 `codex exec`（Mode C）、`cursor-agent -p`（Mode D）、Gemini CLI（Mode G）。仅当想显式使用其他 CLI（cursor-agent / claude -p / codex exec 等）时，才参照此技能的相应章节。

**Windows 例外**: 在原生 Windows 环境中 shell 执行变为 `policy blocked` 时，或 `codex exec exited with code 1` 反复发生时，应停止本地 `codex exec` 的重试。shell 必需任务升级给上司（`supervisor`）。

## 工具选择的优先级

**按成本效率顺序选择。**

| 优先级 | 工具 | 成本 | 擅长领域 |
|--------|--------|--------|----------|
| 1 | `codex exec` | 最便宜（Codex） | 代码生成、编辑、审查 |
| 2 | `cursor-agent -p` | 便宜（Cursor） | 代码生成、编辑、多文件 |
| 3 | `claude -p` | 昂贵（Claude API） | 最终手段。仅在上两个无法解决时使用 |

**原则**: 在非 Windows 或 shell 执行健康的环境中首先尝试 `codex exec`。在原生 Windows 中 shell 执行 blocked / unstable 时跳过 `codex exec`，升级给上司（`supervisor`）。其他失败时或仅在不擅长的任务中，按 cursor-agent → claude 的顺序回退。

## 应使用的时机

- 跨多个文件的代码修改
- 测试编写、测试修正
- 代码审查
- 重构
- Bug 修复的调查与实现
- 新功能的实现

## 不应使用的时机

- 单个文件的小修改（自己直接做）
- 记忆的读写（使用自己的工具）
- 外部 API 调用（使用专用工具）
- 仅信息搜索、调查（web_search 或 Read 足够）

---

## 1. codex exec（推荐）

**适用条件**: Mode S 或 Mode A/B（Bash 允许）时。Mode C 时框架会运行 Codex，因此本节不需要。Mode D/G 时框架也会运行各引擎，因此若无必要作为替代使用 codex，则无需参照。

### 基本语法

```bash
codex exec --full-auto -C /path/to/workspace "プロンプト"
```

工作目录 `-C` 中指定目标项目的绝对路径。Mode S 的 Bash 执行时，`ANIMAWORKS_ANIMA_DIR`（Anima 的数据目录）和 `ANIMAWORKS_PROJECT_DIR`（AnimaWorks 框架的根目录）会作为环境变量设置。若开发对象是 AnimaWorks 本身，可使用 `-C "$ANIMAWORKS_PROJECT_DIR"`。

### 重要选项

| 选项 | 说明 |
|-----------|------|
| `--full-auto` | 自动批准＋沙箱（workspace-write） |
| `-C /path` | 指定工作目录（**必需**） |
| `-m model` | 指定模型（例: `o4-mini`, `o3`） |
| `--sandbox workspace-write` | 允许工作区写入（包含在 full-auto 中） |
| `--json` | 以 JSONL 格式输出 |
| `-o file` | 将最终消息写入文件 |
| `--ephemeral` | 不保存会话文件 |

### 执行示例

#### 代码生成

```bash
codex exec --full-auto --ephemeral -C /home/user/dev/myproject \
  "src/utils/parser.py にMarkdownパーサーを実装して。既存のテストを壊さないこと。"
```

#### 代码审查

```bash
codex exec --full-auto --ephemeral -C /home/user/dev/myproject \
  review
```

#### 测试编写

```bash
codex exec --full-auto --ephemeral -C /home/user/dev/myproject \
  "src/utils/parser.py のユニットテストを tests/test_parser.py に作成して。"
```

#### 将结果保存到文件

```bash
codex exec --full-auto --ephemeral -C /home/user/dev/myproject \
  -o "$ANIMAWORKS_SCRATCH_DIR"/codex_result.txt \
  "このプロジェクトのアーキテクチャを分析して改善案を出して。"
```

---

## 2. cursor-agent -p（替代）

**适用条件**: Mode S 或 Mode A/B（Bash 允许）。Mode C/G 中若 Bash 可用则也可适用。Mode D 中框架会运行 cursor-agent，因此本节的手动 `cursor-agent -p` 原则上不需要。

### 基本语法

```bash
cursor-agent -p --trust --force --workspace /path/to/workspace "プロンプト"
```

### 重要选项

| 选项 | 说明 |
|-----------|------|
| `-p` / `--print` | 非交互模式（**必需**） |
| `--trust` | 自动信任工作区 |
| `--force` | 自动批准命令 |
| `--workspace /path` | 指定工作目录（**必需**） |
| `--model model` | 指定模型（例: `sonnet-4`, `gpt-5`） |
| `--output-format text\|json` | 输出格式 |
| `--mode plan\|ask` | 只读模式（用于调查） |

### 执行示例

#### 代码生成

```bash
cursor-agent -p --trust --force \
  --workspace /home/user/dev/myproject \
  "src/api/routes.py にPOST /users エンドポイントを追加して。バリデーション付き。"
```

#### 只读调查

```bash
cursor-agent -p --trust --mode ask \
  --workspace /home/user/dev/myproject \
  "この認証フローにセキュリティ上の問題はある？"
```

#### 将结果保存到文件

```bash
cursor-agent -p --trust --force \
  --workspace /home/user/dev/myproject \
  --output-format text \
  "テストカバレッジが低いモジュールを特定して改善して" > "$ANIMAWORKS_SCRATCH_DIR"/cursor_result.txt
```

---

## 3. claude -p（回退）

**适用条件**: Mode S 或 Mode A/B（Bash 允许）。Mode C/D/G 中若 Bash 可用则也可适用。

仅在 codex/cursor-agent 无法应对时使用。API 成本较高。

### 基本语法

```bash
claude -p --dangerously-skip-permissions --output-format text "プロンプト"
```

### 重要选项

| 选项 | 说明 |
|-----------|------|
| `-p` / `--print` | 非交互模式（**必需**） |
| `--dangerously-skip-permissions` | 省略权限检查 |
| `--model model` | 指定模型（例: `sonnet`, `haiku`） |
| `--allowedTools "tools"` | 限制允许工具（例: `"Read Edit Bash(git:*)"`) |
| `--output-format text\|json` | 输出格式 |
| `--max-budget-usd N` | 成本上限（美元）|
| `--no-session-persistence` | 不保存会话 |

### 执行示例

```bash
claude -p --dangerously-skip-permissions --no-session-persistence \
  --model haiku --max-budget-usd 0.5 \
  --output-format text \
  "src/core/parser.py のエラーハンドリングを改善して"
```

---

## 提示词的写法

子代理没有 AnimaWorks 的上下文。应编写明确且自包含的提示词。

### 好的提示词

```
以下の要件でPythonモジュールを実装して:

ファイル: src/utils/validator.py

要件:
- Pydantic v2のBaseModelを使ったバリデータ
- email, username, passwordフィールド
- パスワードは8文字以上、英数字混合
- バリデーションエラー時にカスタム例外を投げる

制約:
- from __future__ import annotations を先頭に
- Google-style docstring
- 既存のテストを壊さないこと
```

### 差的提示词

```
いい感じにバリデーションを直して
```

→ 没有上下文，“感觉不错”不明确。

---

## 输出的处理

### 捕获标准输出

```bash
RESULT=$(codex exec --full-auto --ephemeral -C /path "プロンプト" 2>/dev/null)
echo "$RESULT"
```

### 经由文件（推荐 codex）

```bash
codex exec --full-auto --ephemeral -C /path \
  -o "$ANIMAWORKS_SCRATCH_DIR"/result.txt "プロンプト"
# 結果を読む
cat "$ANIMAWORKS_SCRATCH_DIR"/result.txt
```

### 通过退出码判定成败

```bash
codex exec --full-auto --ephemeral -C /path "プロンプト"
if [ $? -eq 0 ]; then
  echo "成功"
else
  echo "失敗 — cursor-agentにフォールバック"
  cursor-agent -p --trust --force --workspace /path "同じプロンプト"
fi
```

---

## 后台执行（重要）

子代理的执行可能需要 **5 分钟〜20 分钟以上**。
在前台等待会阻塞会话，因此 **务必在后台执行**。

### 基本模式: nohup + 结果文件

```bash
nohup codex exec --full-auto --ephemeral -C /path/to/workspace \
  -o "$ANIMAWORKS_SCRATCH_DIR"/codex_result.txt \
  "プロンプト" > "$ANIMAWORKS_SCRATCH_DIR"/codex_stdout.log 2>&1 &
echo "PID: $!"
```

cursor-agent 的情况:

```bash
nohup cursor-agent -p --trust --force \
  --workspace /path/to/workspace \
  "プロンプト" > "$ANIMAWORKS_SCRATCH_DIR"/cursor_result.txt 2>&1 &
echo "PID: $!"
```

### 完成确认

```bash
# プロセスがまだ動いているか確認
ps -p <PID> > /dev/null 2>&1 && echo "実行中" || echo "完了"

# 結果を読む（完了後）
cat "$ANIMAWORKS_SCRATCH_DIR"/codex_result.txt
# または
cat "$ANIMAWORKS_SCRATCH_DIR"/cursor_result.txt
```

### 带超时的执行

为防止失控，并用 `timeout`:

```bash
nohup timeout 30m codex exec --full-auto --ephemeral -C /path \
  -o "$ANIMAWORKS_SCRATCH_DIR"/codex_result.txt \
  "プロンプト" > "$ANIMAWORKS_SCRATCH_DIR"/codex_stdout.log 2>&1 &
```

- 推荐超时: **30 分钟**（`30m`）
- 小任务: **10 分钟**（`10m`）
- 大型重构: **60 分钟**（`60m`）

### 执行中继续其他工作

后台执行后，可以不等完成而推进其他任务。
定期确认进程是否存活，完成后读取结果并记录到 episodes/。

---

## 安全指南

1. **务必指定工作目录** — 未指定时会在当前目录执行
2. **不要在提示词中包含机密信息** — API 密钥、密码等
3. **codex 使用 `--full-auto` 在沙箱内执行** — 对工作区外的写入会受到限制
4. **执行后用 git diff 确认变更** — 检查是否有非预期变更
5. **加上 --ephemeral** — 防止会话文件不必要地累积

---

## 回退策略

```
1. codex exec で試行
   ↓ 失敗 or 品質不足
2. cursor-agent -p で再試行
   ↓ 失敗 or 品質不足
3. claude -p（--max-budget-usd でコスト制限）で最終試行
   ↓ それでも失敗
4. 自分で実行を試みるか、上司に報告する
```

## 注意事项

- 子代理无法访问 AnimaWorks 的记忆、工具。终究只是“编码之手”
- 执行结果应记录到自己的 episodes/，学到的模式应积累到 knowledge/
- 执行需要 5 分钟〜20 分钟以上。务必在后台执行，并设置 timeout
- 应在 git 管理的仓库中工作（便于追踪、撤销变更）
- 在 Mode S 中，Bash 执行时 `ANIMAWORKS_ANIMA_DIR`（Anima 的数据目录）和 `ANIMAWORKS_PROJECT_DIR`（AnimaWorks 框架的根目录）会作为环境变量设置（在 `core/execution/engines/claude/executor.py` 的 `_build_env()` 中注入）
