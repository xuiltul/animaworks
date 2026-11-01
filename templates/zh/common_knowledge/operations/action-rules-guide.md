# 动作规则（Action Rules）

## 概述

动作规则是在发送、发布、通知、记忆写入等有副作用的操作之前插入确认的知识。在 `knowledge/action-rule-*.md` 中写入 `[ACTION-RULE]` 和 `trigger_tools:` 后，会在执行相应工具前进行搜索。

## 基本格式

```markdown
## [ACTION-RULE] ルール名
trigger_tools: gmail_draft, gmail_send
keywords: メール, 下書き, 重複確認
---
実行前に必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
必要な確認が終わってから同じツールを再実行する。
```

| 字段 | 必填 | 说明 |
|-----------|------|------|
| `trigger_tools` | 必填 | 目标工具名。多个用逗号分隔 |
| `keywords` | 可选 | 用于提高搜索精度的词 |
| 正文 | 必填 | 停止时显示的确认内容。必读文件写作 `read_memory_file(path="...")` |

## ToolHandler 的目标工具名

- `call_human`
- `send_message`
- `post_channel`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI 支持

| CLI | 动作规则中的名称 |
|-----|--------------------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool chatwork upload` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...` 不在动作规则范围内。在后台提交目标的运行时，会重新判定目标子命令。

## 门控行为

- 相关度分数低于 `0.80` 的规则不会停止。
- 搜索失败、vector store 不存在、无匹配规则时采用 fail-open，不会妨碍执行。
- 正文中包含 `read_memory_file(path="...")` 时，在同一 action-gate 会话内会停止，直到读取完所有路径。
- 没有必读文件的审查专用规则，在同一 action-gate 会话的 `tool:rule` 中只停止一次。
- 没有全局的“最多停止 2 次”限制。
- 被停止后，请阅读显示的规则，执行必要的 `read_memory_file` 或确认，然后重试同一操作。

## 创建示例

```markdown
## [ACTION-RULE] Gmail下書き前の重複確認
trigger_tools: gmail_draft, gmail_send
keywords: Gmail, 下書き, 重複, thread
---
Gmail下書きや送信の前に、必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
既存スレッドと既存下書きの重複を確認してから実行する。
```

```markdown
## [ACTION-RULE] 顧客メモ更新前の確認
trigger_tools: write_memory_file
keywords: 顧客, customer, profile
---
顧客関連の `knowledge/` を更新する前に、関連する既存ファイルを読んで矛盾がないか確認する。
```

## 放置位置

通常创建在 `knowledge/action-rule-{topic}.md` 中。创建前请用 `search_memory(scope="knowledge")` 查找类似规则，如果已有规则，请优先更新。
