---
name: machine-tool
description: >-
  外部エージェントCLI（工作機械）にタスクを委託。コード変更・調査・分析など重い作業を
  claude/codex/cursor-agent/geminiに丸投げできる。「machine」「工作機械」「外部エージェント」
tags: [machine, agent, external, delegation]
---

# Machine ツール（工作機械）

外部エージェントCLI（claude, codex, cursor-agent, gemini）にタスクを委託するツール。
コード変更・調査・分析など、自分で直接やると重い作業を外部エージェントに丸投げできる。

## 設計思想

あなたは**棟梁（craftsperson）**。machineは**工作機械（CNC、レーザーカッター等）**。
工作機械は精密な加工ができるが、何を作るか決めない。記憶も通信もない。
**正確な設計図（instruction）を渡すのがあなたの仕事。**

## CLI使用法（Sモード）

```bash
# 基本形
animaworks-tool machine run "詳細な作業指示" -d /path/to/workdir

# エンジン指定
animaworks-tool machine run -e cursor-agent "指示" -d /path/to/workdir
animaworks-tool machine run -e claude "指示" -d /path/to/workdir
animaworks-tool machine run -e gemini "指示" -d /path/to/workdir

# モデル上書き
animaworks-tool machine run -e claude -m claude-sonnet-4-6 "指示" -d /path/to/workdir

# バックグラウンド実行（結果は次回heartbeatで取得）
animaworks-tool machine run --background "指示" -d /path/to/workdir

# タイムアウト指定（秒）
animaworks-tool machine run -t 300 "指示" -d /path/to/workdir
```

## use_tool での呼び出し（A/Bモード）

```json
{
  "tool": "use_tool",
  "arguments": {
    "tool_name": "machine",
    "action": "run",
    "args": {
      "engine": "cursor-agent",
      "instruction": "詳細な作業指示",
      "working_directory": "/path/to/workdir"
    }
  }
}
```

## パラメータ

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| engine | YES | エンジン名（cursor-agent, claude, codex, gemini） |
| instruction | YES | 作業指示。ゴール・対象・制約・期待出力を明記 |
| working_directory | YES | 作業ディレクトリの絶対パス |
| background | no | true で非同期実行（デフォルト: false） |
| model | no | モデル上書き（省略時はエンジンのデフォルト） |
| timeout | no | タイムアウト秒数（同期: 600、非同期: 1800） |

## エンジン一覧を確認する

利用可能なエンジンと優先順位を確認するには:

```json
{"tool": "use_tool", "arguments": {"tool_name": "machine", "action": "run", "args": {"engine": "__list__", "instruction": "", "working_directory": ""}}}
```

## instruction の書き方（重要）

曖昧な指示 → 低品質な結果。以下を必ず含める:

1. **達成すべきゴール** — 何を完成させるか
2. **対象ファイル・モジュール** — どこを触るか
3. **制約条件** — コーディング規約、既存APIとの整合性等
4. **期待する出力形式** — コード、レポート、diff等

## 使い分けの目安

| 場面 | 向いている |
|------|-----------|
| マルチファイルのコード変更 | YES |
| バグ調査・原因分析 | YES |
| テストコード生成 | YES |
| リファクタリング | YES |
| 短い質問への回答 | NO（自分で回答） |
| 記憶・通信が必要な作業 | NO（自分でやる） |

## 注意事項

- 工作機械はAnimaWorksインフラにアクセスできない（記憶・メッセージ・組織情報なし）
- レート制限あり（セッション5回、heartbeat2回）
- background=true の結果は `state/background_notifications/` に書かれ、次回heartbeatで確認
