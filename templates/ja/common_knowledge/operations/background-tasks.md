# バックグラウンドタスク実行ガイド

## 概要

一部の外部ツール（画像生成、3Dモデル生成、ローカルLLM推論、音声文字起こし等）は
実行に数分〜数十分かかる。これらを直接実行すると、実行中ずっとロックが保持され、
メッセージの受信やheartbeatが停止してしまう。

`animaworks-tool submit` を使うことで、タスクをバックグラウンドで実行し、
自分自身はすぐに次の作業に移ることができる。

## いつ submit を使うか

### 必ず submit を使うべきツール

ツールガイド（システムプロンプト）に ⚠ マークが付いているサブコマンド:

- `image_gen pipeline` / `fullbody` / `bustup` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe`

### submit 不要のツール

実行時間が短い（30秒未満）ツール:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`（通常の操作）
- `github`, `aws_collector`

### 判断基準

- ⚠ マークあり → 必ず submit
- ⚠ マークなし → 直接実行

## 使い方

### 基本構文

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 実行例

```bash
# 3Dモデル生成（Meshy API、最大10分）
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# キャラクター画像一括生成（全ステップ、最大30分）
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# ローカルLLM推論（Ollama、最大5分）
animaworks-tool submit local_llm generate "要約してください: ..."

# 音声文字起こし（Whisper + Ollama整形、最大2分）
animaworks-tool submit transcribe "/path/to/audio.wav" --language ja
```

### 戻り値

submit は即座に以下のJSONを返して終了する:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。"
}
```

## 結果の受け取り

1. submit 後、タスクはバックグラウンドで実行される
2. 完了すると `state/background_notifications/{task_id}.md` に結果が書かれる
3. 次回の heartbeat で自動的にこの通知を確認できる
4. 通知には成功/失敗のステータスと結果サマリが含まれる

## 失敗時の対応

- 通知に「失敗」と記載されている場合:
  1. エラー内容を確認する
  2. 原因を特定する（APIキー未設定、タイムアウト、引数ミス等）
  3. 修正して再度 submit する
  4. 解決できない場合は上司に報告する

- 実行中にプロセスがクラッシュした場合、`state/background_tasks/pending/processing/` または `state/pending/processing/` に残ったタスクは次回起動時に `pending/failed/` へ自動移動される

## よくある間違い

### 直接実行してしまう

```bash
# 悪い例: 直接実行 → 10分間ロックされる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

直接実行してしまった場合、タスクが完了するまで待つしかない。
次回から必ず submit を使うこと。

### submit 後に結果を待ち続ける

submit したらすぐに次の作業に移ること。
結果は自動的に通知されるので、ポーリングや待機は不要。

## 技術的な仕組み（参考）

PendingTaskExecutor は2種類のタスクを監視・実行する。

### コマンド型タスク（animaworks-tool submit）

1. `animaworks-tool submit` が `state/background_tasks/pending/*.json` にタスク記述子を書く
2. PendingTaskExecutor の watcher が3秒間隔で `state/background_tasks/pending/` を監視（`wake()` で即時チェックも可能）
3. タスクを検出すると `pending/*.json` を `pending/processing/` に移動して実行開始
4. `execute_pending_task` が BackgroundTaskManager.submit に投入。Anima のロック外でサブプロセス実行（タイムアウト30分）
5. 投入成功時: processing 内のファイルを削除。投入失敗時: `pending/failed/` に移動
6. 完了時に `_on_background_task_complete` コールバックが `state/background_notifications/{task_id}.md` に通知を書く
7. 次回 heartbeat で `drain_background_notifications()` が通知を読み取り、コンテキストに注入される

### LLM型タスク（state/pending/）

Heartbeat や `submit_tasks` ツールが書き出す LLM タスクは **別ディレクトリ** `state/pending/` に投入される。

1. `submit_tasks` が `state/pending/{task_id}.json` にタスク記述子を書く（`task_type: "llm"`, `batch_id` 等）
2. watcher が `state/pending/` を同様に3秒間隔で監視
3. `batch_id` 付きタスクはバッチに蓄積し、`_dispatch_batch` で DAG に基づき実行
4. `parallel: true` のタスクはセマフォ（`config.json` の `background_task.max_parallel_llm_tasks`、デフォルト3）で並列実行
5. `depends_on` で依存関係を指定したタスクは、依存完了後に実行
6. 結果は `state/task_results/{task_id}.md` に保存。`reply_to` に DM で完了/失敗通知を送信
7. 24時間経過したタスク（TTL）はスキップされる

本ガイドの `animaworks-tool submit` とは入口・ディレクトリが異なる。

### ファイルのライフサイクル

**コマンド型**（animaworks-tool submit）:

```
state/background_tasks/pending/*.json
  → pending/processing/*.json
  → 成功: 削除 | 失敗: pending/failed/*.json
```

**LLM型**（submit_tasks / Heartbeat）:

```
state/pending/*.json
  → pending/processing/*.json
  → 成功: 削除 | 失敗: pending/failed/*.json
```

起動時には両方の `processing/` に残った孤立ファイル（クラッシュ等）を `failed/` へ移動してリカバリする。
