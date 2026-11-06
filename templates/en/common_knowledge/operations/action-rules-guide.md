# Action Rules

## Overview

Action rules are knowledge used to insert a confirmation immediately before operations that have side effects, such as sending, posting, notifications, and memory writes. Writing `knowledge/action-rule-*.md` and `[ACTION-RULE]` in `trigger_tools:` will cause a search to be performed before the corresponding tool is executed.

## Basic Format

```markdown
## [ACTION-RULE] ルール名
trigger_tools: gmail_draft, gmail_send
keywords: メール, 下書き, 重複確認
---
実行前に必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
必要な確認が終わってから同じツールを再実行する。
```

| Field | Required | Description |
|-----------|------|------|
| `trigger_tools` | Required | Target tool name. Multiple names are comma-separated |
| `keywords` | Optional | Terms that improve search precision |
| Body | Required | Confirmation content shown at shutdown. Required reading files are written as `read_memory_file(path="...")` |

## Tool Names Targeted by ToolHandler

- `call_human`
- `send_message`
- `animaworks-tool board post`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI Support

| CLI | Name in Action Rules |
|-----|--------------------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool chatwork upload` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...` is not covered by action rules. When the background submission target runs, the target subcommand is evaluated again at runtime.

## Gate Behavior

- Rules with a relevance score below `0.80` will not trigger a shutdown.
- Search failure, missing vector store, or no matching rule will fail open and not block execution.
- If the body contains `read_memory_file(path="...")`, shutdown will occur until all paths are read within the same action-gate session.
- Review-only rules without required reading files will trigger a shutdown only once per `tool:rule` in the same action-gate session.
- There is no global "maximum two shutdowns" limit.
- If shutdown occurs, read the displayed rules, perform any necessary `read_memory_file` or confirmations, then retry the same operation.

## Creation Example

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

## Location

Normally, create it in `knowledge/action-rule-{topic}.md`. Before creating, search for similar rules using `search_memory(scope="knowledge")`, and if an existing rule is found, prioritize updating it.
