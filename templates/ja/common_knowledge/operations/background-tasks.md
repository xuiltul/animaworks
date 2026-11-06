# バックグラウンドタスク実行ガイド

## 概要

一部の外部ツール（画像生成、3Dモデル生成、ローカルLLM推論、音声文字起こし等）は
実行に数分〜数十分かかる。これらを直接実行すると、実行中ずっとロックが保持され、
メッセージの受信や heartbeat が停止してしまう。

`animaworks-tool submit` を使うことで、タスクをバックグラウンドで実行し、
自分自身はすぐに次の作業に移ることができる。

`animaworks-tool submit` は `task_type="command"` として TaskStore に登録され、
Anima 子プロセス内の **PendingTaskExecutor**（`core/tasks/pending_executor.py`）が試行を取得する。
**BackgroundTaskManager**（`core/tasks/background.py`）がツールをバックグラウンド実行し、
試行は TaskStore、互換用の結果 JSON は `state/background_tasks/{task_id}.json` に保存する。

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

1. submit はツール名・引数・Anima を含む `task_type="command"` の入力を TaskStore に登録する。PendingTaskExecutor が実行試行を取得し、会話ロック外で BackgroundTaskManager に実行を依頼する。
2. TaskStore は試行と結果参照を記録する。`state/background_tasks/{task_id}.json` は従来の照会 API との互換用に同じ task_id で `running` → `completed` / `failed` を記録する。
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

- プロセス異常終了時は TaskStore の試行所有者を確認し、所有者の終了が確定したタスクを pending（要確認）として記録する。副作用が発生済みの可能性があるため、自動再実行はしない。結果 JSON が `running` のまま残っていても完了と判断せず、実際の状態を確認してから明示的に再開する。

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

### BackgroundTaskManager（`core/tasks/background.py`）

- **役割**: 長時間ツール呼び出しを `asyncio` タスクとしてバックグラウンド実行し、完了・失敗時に `on_complete`（任意の非同期コールバック）を `await` する。コンストラクタで `state/background_tasks/` を `mkdir(parents=True)` する。
- **同期ツール**: `submit(tool_name, tool_args, execute_fn, task_id=None)` → `execute_fn(name, args) -> str | None` を `run_in_executor(None, ...)` でスレッドプール実行。`task_id` が省略されれば生成し、CLI command task では TaskStore と同じ ID を渡す。
- **非同期ツール**: `submit_async`（同シグネチャで `execute_fn` が `Awaitable[str]`）→ イベントループ上で `await execute_fn(...)`。
- **スケジューリング**: `asyncio.create_task(..., name=f"bg-{task_id}")` でラップ。完了時 `_async_tasks` から該当エントリを除去。
- **永続化**: 各変更後 `_save_task` で `state/background_tasks/{task_id}.json` に `to_dict()`（`ensure_ascii=False`, `indent=2`）。破損 JSON は `_load_task` で警告ログのうえ `None`。
- **照会**: `get_task` はインメモリ優先、なければディスク。`list_tasks(status=...)` はディスク上の `*.json` とマージし、`created_at` 降順。`active_count` はインメモリの `RUNNING` 件数。
- **`on_complete`**: コールバック内で例外が出てもタスクの完了/失敗状態は維持され、失敗はログに記録されるのみ。
- **資格あるツール名**（`is_eligible`）は次の **3 層**をマージ（後勝ち）。キーはそのまま辞書照合（Mode A のスキーマ名 `generate_3d_model` と Mode S 提出用の `image_gen:3d` の**両方**があり得る）:
  1. コード内デフォルト `_DEFAULT_ELIGIBLE_TOOLS`（値は目安秒数。現状のキー）:
     `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`（各 30）、`local_llm` / `run_command`（各 60）
  2. `BackgroundTaskManager.from_profiles` 経由で、各モジュールの `EXECUTION_PROFILE` から `background_eligible: true` のサブコマンドを抽出（`core.integrations._base.get_eligible_tools_from_profiles`）。キーは `"{tool_name}:{subcmd}"`、秒数は `expected_seconds`（未設定時 60）
  3. `config.json` の `background_task.eligible_tools` — 各キーに対し `threshold_s` を秒数として上書き
- **無効化**: `config.json` で `background_task.enabled: false` にすると `BackgroundTaskManager` 自体が作られない。submit は TaskStore に登録されるが、実行試行は pending に戻り、要確認通知が発生する。
- **掃除**: `cleanup_old_tasks(max_age_hours=24)` は、(1) `completed` / `failed` で `completed_at` が指定時間を超えた JSON、(2) `running` のまま `created_at` が **48 時間超**過去のファイル（プロセスクラッシュ等の孤児）を削除する。戻り値は削除件数。保持時間は呼び出し側が `max_age_hours` で指定し、対応する `config.json` 設定キーはない。

### コマンド型タスク（`animaworks-tool submit`）

1. `animaworks-tool submit` は `task_type="command"`、ツール名、引数、Anima、`task_id` を含む入力を TaskStore に原子的に登録する（`ANIMAWORKS_ANIMA_DIR` 必須）。
2. TaskStore watcher は既存の claim / attempt 機構でタスクを取得する。`BackgroundTaskManager` は元の `task_id` を結果 JSON にも使う。
3. コマンド型は内部で `animaworks-tool … -j` を **サブプロセス**として起動する。**1 回あたりの壁時計タイムアウトは 1800 秒（30 分）**（`pending_executor._PENDING_TASK_SUBPROCESS_TIMEOUT`）。
4. stdout またはエラーは `state/task_results/{task_id}/{attempt_token}.md` に保存し、TaskStore の試行に結果参照を記録する。状態・引数・試行の正本は TaskStore。
5. BackgroundTaskManager は `state/background_tasks/{task_id}.json` に `running` → `completed` / `failed` を保存し、`_on_background_task_complete` は `state/background_notifications/{task_id}.md` に通知を書く。heartbeat が通知を取り込む。

### 旧コマンド記述子の移行

以前の `state/background_tasks/pending/*.json` は新ランタイムでは監視しない。停止中に `animaworks task-store migrate --anima NAME --backup PATH` を実行すると、未処理のトップレベル記述子を検証して TaskStore に取り込む。元ファイルは証跡として残す。`processing/` と `failed/` は自動再実行・移行せず、警告とともに残すため、副作用を確認してから運用者が判断する。

### LLM 型タスク（正規タスクストア）

1. `submit_tasks` / `delegate_task` が完全な指示・文脈・完了条件・制約・モデル・依存関係を原子的に登録する。
2. watcherは実行可能なタスクの取得と試行作成を同じtransactionで行う。依存先と設定済みワーカー容量を確認し、非並列の制約は同じバッチ内に限定する。
3. `done` / `cancelled` 宣言なしで試行が終わるとpendingに戻るが自動再実行しない。副作用を確認し、必要なら `submit_tasks(batch_id="resume", tasks=[{"task_id":"ID","resume":true}])` で明示再開する。
4. 結果は `state/task_results/{task_id}.md` を参照できる場合があるが、状態・入力・試行は正規ストアが正本。完了は `update_task` で宣言し、LLMの応答やファイルの存在から推測しない。
5. 要確認・完了通知は永続化し、定期heartbeatに依存せず届ける。DMから実行入力を再構成しない。

SQLiteや旧タスクファイルを直接編集せず、タスクツールを使う。上記のコマンド型パイプラインとは別の仕組み。

### ファイルのライフサイクル

**コマンド型**（`animaworks-tool submit`）:

```
TaskStore の command 入力 → claim / attempt
  → 成功: done | 失敗・中断: pending（要確認。自動再実行なし）
state/task_results/{task_id}/{attempt_token}.md  # attempt output / error
state/background_tasks/{task_id}.json           # 互換用の running → completed / failed 結果
```

**LLM 型**（`submit_tasks` / `delegate_task`）:

```
保存済み入力 → 実行可能pending → 取得済み試行（in_progress）
  → done/cancelled宣言、またはpending + 永続的な要確認通知
  → 明示resumeで保存済み入力を使う新しい試行
```

コマンド型とLLM型は別の入力形式だが、どちらも TaskStore の claim / attempt で実行する。旧LLMのJSONL・`state/pending/` 記述子は移行・export用であり、実行開始の合図ではない。
