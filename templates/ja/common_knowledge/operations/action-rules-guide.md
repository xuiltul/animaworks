# アクションルール（Action Rules）

## 概要

アクションルールは、送信・投稿・通知・記憶書き込みなど副作用のある操作の直前に確認を入れるための知識です。`knowledge/action-rule-*.md` に `[ACTION-RULE]` と `trigger_tools:` を書くと、該当ツール実行前に検索されます。

## 基本形式

```markdown
## [ACTION-RULE] ルール名
trigger_tools: gmail_draft, gmail_send
keywords: メール, 下書き, 重複確認
---
実行前に必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
必要な確認が終わってから同じツールを再実行する。
```

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `trigger_tools` | 必須 | 対象ツール名。複数はカンマ区切り |
| `keywords` | 任意 | 検索精度を上げる語 |
| 本文 | 必須 | 停止時に表示される確認内容。必読ファイルは `read_memory_file(path="...")` と書く |

## ToolHandler の対象ツール名

- `call_human`
- `send_message`
- `post_channel`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI 対応

| CLI | アクションルール上の名前 |
|-----|--------------------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...` はアクションルール対象外です。バックグラウンド投入先の実行時に、対象サブコマンドが改めて判定されます。

## ゲート動作

- 関連度スコア `0.80` 未満のルールは停止しません。
- 検索失敗、vector store不在、一致ルールなしの場合は fail-open で実行を妨げません。
- 本文に `read_memory_file(path="...")` が含まれる場合、同じ action-gate セッション内で全パスを読むまで停止します。
- 必読ファイルがないレビュー専用ルールは、同じ action-gate セッションの `tool:rule` ごとに1回だけ停止します。
- グローバルな「最大2回停止」制限はありません。
- 停止されたら、表示されたルールを読み、必要な `read_memory_file` や確認を実行してから同じ操作を再試行します。

## 作成例

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

## 置き場所

通常は `knowledge/action-rule-{topic}.md` に作成します。作成前に `search_memory(scope="knowledge")` で類似ルールを探し、既存ルールがあれば更新を優先してください。
