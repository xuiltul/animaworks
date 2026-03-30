---
name: machine-tool
description: >-
  外部エージェントCLI（工作機械）へタスクを委託し、大規模なコード変更・調査・分析を外部に任せる。
  Use when: machineコマンドで別エージェントに実装依頼、リファクタ・調査の丸投げ、重いバッチ実行が必要なとき。
tags: [machine, agent, external, delegation]
---

# Machine ツール（工作機械）

外部エージェントCLI にタスクを委託するツール。
コード変更・調査・分析など、自分で直接やると重い作業を外部エージェントに丸投げできる。

## 設計思想

あなたは**棟梁（craftsperson）**。machineは**工作機械（CNC、レーザーカッター等）**。
工作機械は精密な加工ができるが、何を作るか決めない。記憶も通信もない。
**正確な設計図（instruction）を渡すのがあなたの仕事。**

## 呼び出し方法

```bash
animaworks-tool machine run [オプション] "指示" -d /path/to/workdir
```

### CLIオプション一覧

| オプション | 説明 |
|-----------|------|
| `-e ENGINE` | エンジン指定（省略時: デフォルトが自動選択される。指定する場合は `-h` で一覧確認） |
| `-d PATH` | 作業ディレクトリ（省略時: カレントディレクトリ） |
| `-t SECONDS` | タイムアウト秒数（デフォルト: 同期600秒、バックグラウンド1800秒） |
| `-m MODEL` | モデル上書き（省略時: エンジンのデフォルト） |
| `--background` | バックグラウンド実行（タイムアウト1800秒。出力は `state/cmd_output/` にストリーミング） |
| `-j / --json` | 結果をJSON形式で出力 |

### 基本例

```bash
# 最小限（デフォルトエンジン・カレントディレクトリ）
animaworks-tool machine run "詳細な作業指示"

# エンジンとディレクトリを指定
animaworks-tool machine run -e cursor-agent "指示" -d /path/to/workdir

# バックグラウンド実行
animaworks-tool machine run --background "指示" -d /path/to/workdir

# タイムアウト指定
animaworks-tool machine run -t 300 "指示" -d /path/to/workdir
```

## instruction の書き方（重要）

曖昧な指示 → 低品質な結果。以下を必ず含める:

1. **達成すべきゴール** — 何を完成させるか
2. **対象ファイル・モジュール** — どこを触るか
3. **制約条件** — コーディング規約、既存APIとの整合性等
4. **期待する出力形式** — コード、レポート、diff等

### 長い instruction はファイル経由で渡す

Bash特殊文字（`|`, `` ` ``, `$`）を含む長い指示は、直接引数にするとシェルエラーになる。
ファイルに書いてから渡す:

```bash
# ファイルに指示を書き出す
cat > /tmp/instruction.txt << 'INSTRUCTION'
## タスク: PR #2087 コードレビュー

| 観点 | 確認内容 |
|------|---------|
| 正しさ | Issue要件を満たしているか |
| 保守性 | 読みやすさ・テスト・責務分離 |

対象ファイル: `app/Services/Movacal/MovacalApiClient.php`
INSTRUCTION

# ファイルから読み込んで実行
animaworks-tool machine run "$(cat /tmp/instruction.txt)" -d /path/to/workdir
```

## 並列実行（`--background`）

`--background` を使えば複数のmachineを同時に起動できる。
出力は `state/cmd_output/` にリアルタイムでストリーミングされる。

### 3並列レビューの例

```bash
# 3つの観点を並列で起動
animaworks-tool machine run --background "Correctness観点でレビュー..." -d /path &
animaworks-tool machine run --background "Maintainability観点でレビュー..." -d /path &
animaworks-tool machine run --background "Consistency観点でレビュー..." -d /path &
wait

# 結果を確認（state/cmd_output/ のファイルをReadで読む）
```

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
- background実行の出力は `state/cmd_output/` にストリーミングされ、Read/Globで確認可能
