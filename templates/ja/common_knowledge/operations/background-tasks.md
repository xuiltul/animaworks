# バックグラウンドタスク実行ガイド

## 概要

一部の外部ツール（画像生成、3Dモデル生成、ローカルLLM推論、音声文字起こし等）は
実行に数分〜数十分かかる。これらを直接実行すると、実行中ずっとロックが保持され、
メッセージの受信や heartbeat が停止してしまう。

`animaworks-tool submit` を使うことで、タスクをバックグラウンドで実行し、
自分自身はすぐに次の作業に移ることができる。

ランタイムでは `core/background.py` の **BackgroundTaskManager** がツール実行と
`state/background_tasks/{task_id}.json` への状態永続化を担当し、
Anima 子プロセス内の **PendingTaskExecutor**（`core/supervisor/pending_executor.py`）が
`animaworks-tool submit` が書いた待ちキューを監視して `BackgroundTaskManager` に載せ替える。

## いつ submit を使うか

### 必ず submit を使うべきツール

ツールガイド（システムプロンプト）に ⚠ マークが付いているサブコマンド:

- `image_gen pipeline` / `fullbody` / `bustup` / `icon` / `chibi` / `3d` / `rigging` / `animations`
- `local_llm generate` / `chat`
- `transcribe audio`（サブコマンド名は `audio`）

各ツールの `EXECUTION_PROFILE` で `background_eligible: true` のものは、プロファイル経由で
バックグラウンド実行の候補に登録される（例: `chatwork sync` / `download` など）。
運用方針は引き続き **⚠ マーク** を優先すること。

### submit 不要のツール

実行時間が短い（数十秒未満が目安）ツール:

- `web_search`, `x_search`
- `slack`, `chatwork`, `gmail`（通常の操作）
- `github`, `aws_collector`

### 判断基準

- ⚠ マークあり → 必ず submit
- ⚠ マークなし → 直接実行（`animaworks-tool submit` 実行時、プロファイル上「短時間」の場合は stderr に警告が出ることがある）

## 使い方

### 基本構文

```bash
animaworks-tool submit <ツール名> <サブコマンド> [引数...]
```

### 実行例

```bash
# 3Dモデル生成（Meshy API 等）
animaworks-tool submit image_gen 3d assets/avatar_chibi.png

# キャラクター画像一括生成（全ステップ）
animaworks-tool submit image_gen pipeline "1girl, black hair, ..." --negative "lowres, ..." --anima-dir $ANIMAWORKS_ANIMA_DIR

# ローカルLLM推論（Ollama）
animaworks-tool submit local_llm generate "要約してください: ..."

# 音声文字起こし（Whisper 等）— サブコマンドは audio
animaworks-tool submit transcribe audio "/path/to/audio.wav" --language ja
```

### 戻り値

submit は即座に JSON を標準出力へ出して終了する（`task_id` は 12 桁の hex）:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "submitted",
  "tool": "image_gen",
  "subcommand": "3d",
  "message": "バックグラウンドタスクを投入しました。完了時にinboxに通知されます。(task_id: a1b2c3d4e5f6)"
}
```

## 結果の受け取り

1. submit 後、**PendingTaskExecutor** が `state/background_tasks/pending/` の記述子を取り込み、
   `BackgroundTaskManager.submit` でバックグラウンド実行する（Anima の会話ロック外）。
2. 実行中〜完了まで、同一 `task_id` の状態は **`state/background_tasks/{task_id}.json`** にも保存される。
   `BackgroundTaskManager.submit` / `submit_async` が書き出すタイミングでは **初回から `running`**（UUID 12 桁の `task_id` を採番した直後にディスクへ保存）。完了で `completed`、例外で `failed`。
   データ型上は `pending` 列挙値もあるが、マネージャの通常投入フローでは使われない。**キュー待ち**は別物で、`state/background_tasks/pending/*.json` の記述子が表す。
3. 完了時に `_on_background_task_complete` が **`state/background_notifications/{task_id}.md`** に Markdown 通知を書く。
4. 次回の **heartbeat** で `drain_background_notifications()` が当該 `.md` を読み取り・削除し、コンテキストへ注入される。
5. Web UI の WebSocket や `call_human` 系の人間通知が有効なら、同じ完了タイミングでそちらにも載る場合がある。

ツール **`list_background_tasks`** でメモリとディスクをマージした一覧を、**`check_background_task`** で `task_id` 指定の状態を参照できる（いずれも `BackgroundTaskManager` の `list_tasks` / `get_task` に相当）。

## 失敗時の対応

- 通知に「失敗」と記載されている場合:
  1. エラー内容を確認する
  2. 原因を特定する（APIキー未設定、タイムアウト、引数ミス等）
  3. 修正して再度 submit する
  4. 解決できない場合は上司に報告する

- **クラッシュや異常終了**で `processing/` に JSON が残った場合、Anima プロセス起動時に **PendingTaskExecutor** が回収する:
  - **コマンド型**（`animaworks-tool submit`）: `state/background_tasks/pending/processing/*.json` → `state/background_tasks/pending/failed/`
  - **LLM 型**（`submit_tasks` / Heartbeat 書き出し）: `state/pending/processing/*.json` → `state/pending/failed/`
  いずれも本ガイドの submit とは **ディレクトリが別**（後者は `state/pending/` ツリー）。

## よくある間違い

### 直接実行してしまう

```bash
# 悪い例: 直接実行 → 長時間ロックされうる
animaworks-tool image_gen 3d assets/avatar_chibi.png -j

# 良い例: submit で非同期実行
animaworks-tool submit image_gen 3d assets/avatar_chibi.png
```

直接実行してしまった場合、タスクが完了するまで待つしかない。
次回から必ず submit を使うこと。

### transcribe のサブコマンド省略

```bash
# 悪い例: audio サブコマンドがないと意図したプロファイル判定にならない
animaworks-tool submit transcribe "/path/to/audio.wav"

# 良い例
animaworks-tool submit transcribe audio "/path/to/audio.wav"
```

### submit 後に結果を待ち続ける

submit したらすぐに次の作業に移ること。
結果は heartbeat 向けの通知ファイル経由で取り込まれるので、ポーリングや待機は不要。

## 技術的な仕組み（参考）

### BackgroundTaskManager（`core/background.py`）

- **役割**: 長時間ツール呼び出しを `asyncio` タスクとしてバックグラウンド実行し、完了・失敗時に `on_complete`（任意の非同期コールバック）を `await` する。コンストラクタで `state/background_tasks/` を `mkdir(parents=True)` する。
- **同期ツール**: `submit(tool_name, tool_args, execute_fn)` → `execute_fn(name, args) -> str | None` を `run_in_executor(None, ...)` でスレッドプール実行。
- **非同期ツール**: `submit_async`（同シグネチャで `execute_fn` が `Awaitable[str]`）→ イベントループ上で `await execute_fn(...)`。
- **スケジューリング**: `asyncio.create_task(..., name=f"bg-{task_id}")` でラップ。完了時 `_async_tasks` から該当エントリを除去。
- **永続化**: 各変更後 `_save_task` で `state/background_tasks/{task_id}.json` に `to_dict()`（`ensure_ascii=False`, `indent=2`）。破損 JSON は `_load_task` で警告ログのうえ `None`。
- **照会**: `get_task` はインメモリ優先、なければディスク。`list_tasks(status=...)` はディスク上の `*.json` とマージし、`created_at` 降順。`active_count` はインメモリの `RUNNING` 件数。
- **`on_complete`**: コールバック内で例外が出てもタスクの完了/失敗状態は維持され、失敗はログに記録されるのみ。
- **資格あるツール名**（`is_eligible`）は次の **3 層**をマージ（後勝ち）。キーはそのまま辞書照合（Mode A のスキーマ名 `generate_3d_model` と Mode S 提出用の `image_gen:3d` の**両方**があり得る）:
  1. コード内デフォルト `_DEFAULT_ELIGIBLE_TOOLS`（値は目安秒数。現状のキー）:
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`（各 30）、`local_llm` / `run_command`（各 60）
  2. `BackgroundTaskManager.from_profiles` 経由で、各モジュールの `EXECUTION_PROFILE` から `background_eligible: true` のサブコマンドを抽出（`core.tools._base.get_eligible_tools_from_profiles`）。キーは `"{tool_name}:{subcmd}"`、秒数は `expected_seconds`（未設定時 60）
  3. `config.json` の `background_task.eligible_tools` — 各キーに対し `threshold_s` を秒数として上書き
- **無効化**: `config.json` で `background_task.enabled: false` にすると `BackgroundTaskManager` 自体が作られない（その場合、submit キューは取り込まれても実行側で警告になる）。
- **掃除**: `cleanup_old_tasks(max_age_hours=24)` は、(1) `completed` / `failed` で `completed_at` が **24 時間超**過去の JSON を削除、(2) `running` のまま `created_at` が **48 時間超**過去のファイル（プロセスクラッシュ等の孤児）を削除。戻り値は削除件数。`config.json` の `background_task.result_retention_hours` はスキーマに存在するが、**現行の `BackgroundTaskManager` 実装からは参照されない**（呼び出し側が `cleanup_old_tasks` に任意の `max_age_hours` を渡す API のみ）。

### 同ファイル内のその他 API: `rotate_dm_logs`

`core/background.py` には **バックグラウンドツール実行とは独立**に、`rotate_dm_logs(shared_dir, max_age_days=7)` がある。`shared/dm_logs/*.jsonl` のうち、エントリの `ts` が閾値より古い行を `{stem}.{YYYYMMDD}.archive.jsonl` へ追記アーカイブし、アクティブファイルからは除く。実処理は `_rotate_dm_logs_sync`（`run_in_executor` でオフロード）。

### コマンド型タスク（`animaworks-tool submit`）

1. `animaworks-tool submit` が `state/background_tasks/pending/{task_id}.json` に記述子を書く（`ANIMAWORKS_ANIMA_DIR` 必須）。
2. PendingTaskExecutor の watcher が最大 **3 秒**間隔（`wake()` で即時も可）で `pending/` を監視。
3. `pending/*.json` → `pending/processing/` にリネームしてから `execute_pending_task`。
4. コマンド型は内部で `animaworks-tool … -j` を **サブプロセス**として起動。**1 回あたりの壁時計タイムアウトは 1800 秒（30 分）**（`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`）。成功後、processing 内のファイルは削除。例外時は `pending/failed/` へ。
5. 実処理は `BackgroundTaskManager.submit(composite_name, tool_args, execute_fn)` に委譲。`composite_name` は `tool:subcommand`（例: `image_gen:3d`）。これが `is_eligible` と照合される。
6. 完了時に `_on_background_task_complete` が `state/background_notifications/{task_id}.md` を書き、heartbeat で `drain_background_notifications()` が読み取る。

### LLM 型タスク（`state/pending/`）

Heartbeat や `submit_tasks` ツールが書き出す LLM タスクは **別ディレクトリ** `state/pending/` に投入される。

1. `submit_tasks` が `state/pending/{task_id}.json` にタスク記述子を書く（`task_type: "llm"`, `batch_id` 等）
2. watcher が `state/pending/` を同様に監視
3. `batch_id` 付きタスクはバッチに蓄積し、`_dispatch_batch` で DAG に基づき実行
4. `parallel: true` のタスクはセマフォ（`config.json` の `background_task.max_parallel_llm_tasks`、デフォルト 3）で並列実行
5. `depends_on` で依存関係を指定したタスクは、依存完了後に実行
6. 結果は `state/task_results/{task_id}.md` に保存（要約は長さ上限あり）。`reply_to` があれば DM で完了/失敗通知
7. 24 時間経過したタスク（TTL）はスキップされる

本ガイドの `animaworks-tool submit` とは入口・ディレクトリが異なる。

### ファイルのライフサイクル

**コマンド型**（`animaworks-tool submit` の待ちキュー）:

```
state/background_tasks/pending/*.json
  → pending/processing/*.json
  → 成功: 削除 | 失敗: pending/failed/*.json
```

併せて **タスク状態ファイル**（実行全体）:

```
state/background_tasks/{task_id}.json   # running → completed / failed
```

**LLM 型**（`submit_tasks` / Heartbeat）:

```
state/pending/*.json
  → pending/processing/*.json
  → 成功: 削除 | 失敗: pending/failed/*.json
```

起動時には、**それぞれの** `processing/` に残った孤立ファイルを `failed/` へ移動してリカバリする。
