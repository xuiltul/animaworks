# ワーキングメモリ（state/）技術リファレンス

Anima の作業状態を管理する `state/` ディレクトリの詳細仕様。
プロンプトへの注入ロジック、サイズ制御、マイグレーション、ロック制御を含む。

---

## state/ ディレクトリ構成

```
state/
├── current_state.md          # ワーキングメモリ（自由形式Markdown）
├── task_queue.jsonl           # タスクレジストリ（append-only JSONL）
├── pending/                   # LLMタスク実行キュー（JSON）
│   ├── {task_id}.json         # 投入されたタスク
│   ├── processing/            # 実行中（PendingTaskExecutorが移動）
│   └── failed/                # 失敗タスク
├── task_results/              # TaskExec完了結果
│   └── {task_id}.md           # 結果要約（最大2000文字、7日TTL）
├── conversation.json          # 会話状態
├── conversations/             # スレッド別会話ファイル
├── recovery_note.md           # クラッシュ復旧ノート
├── heartbeat_checkpoint.json  # Heartbeatチェックポイント
└── pending_procedures.json    # 保留中の手続き追跡
```

---

## current_state.md

### 役割

Anima のワーキングメモリ。「今まさに何をしているか」「何を観察したか」「どんなブロッカーがあるか」を自由形式で記録する。タスク管理用ではなく、状況認識のための場所。

タスクの公式な追跡・管理は `task_queue.jsonl`（Layer 2）が担う。

### サイズ制御

| パラメータ | 値 | ソース |
|-----------|-----|-------|
| 表示上限 | 3000文字 | `_CURRENT_STATE_MAX_CHARS`（builder.py） |
| ディスク trim 上限 | デフォルト無効 | `heartbeat.current_state_max_chars`（0 = 無効） |
| Inbox時上限 | 500文字 | builder.py 内で `min(_state_max, 500)` |

**セッション境界**:

- 通常の Heartbeat / cron / 会話最終化では `current_state.md` を保持する
- セッション要約に現在状態が含まれる場合も、`current_state.md` が空/idle のときだけ書き込む
- active な可視タスクがない古い state は TaskBoard housekeeping によりアーカイブされる場合がある

**Heartbeat 時の任意クリーンアップ**:

1. `heartbeat.current_state_max_chars` が 0 より大きく、Heartbeat 開始前に `current_state.md` がその値を超過している場合、「整理して圧縮せよ」という指示が Heartbeat プロンプトに注入される
2. Heartbeat または cron 完了後、`_enforce_state_size_limit()` が実行される
3. 設定上限の超過分は当日のエピソード記憶（`episodes/{date}.md`）に `## current_state.md overflow archived` として移動
4. 末尾の設定文字数を保持し、改行位置で調整（先頭20%以内に改行があればそこで切る）

### プロンプトへの注入

| トリガー | 挙動 |
|---------|------|
| `chat` | 全文注入（3000文字上限、スケール適用） |
| `inbox` | 最大500文字に制限 |
| `heartbeat` / `cron` | 全文注入（3000文字上限） |
| `task` | **注入しない**（Minimal ティア） |

注入時、`status: idle` のみの場合はセクション自体が省略される。
それ以外の場合は `builder/task_in_progress` テンプレートで強調ヘッダー付きで注入される。

### ロック制御

`core/anima.py` の `_state_file_lock`（`asyncio.Lock`）が `current_state.md` への並行書き込みを防止する。

`_is_state_file(path)` は `state/current_state.md` のみに `True` を返す。`write_memory_file` 経由の書き込みでは、このファイルに対してロックが自動取得される。

### パス解決（後方互換）

`read_memory_file` / `write_memory_file` で `state/current_task.md` が指定された場合、自動的に `state/current_state.md` に解決される（`handler_memory.py`）。

---

## pending.md（廃止済み）

`state/pending.md` は `current_state.md` に統合された後、自動削除される。

### マイグレーション（MemoryManager 初期化時）

1. `state/current_task.md` が存在し `state/current_state.md` が存在しない → リネーム
2. 両方存在 → `current_state.md` を優先、警告ログ
3. `state/pending.md` が存在し内容がある → `current_state.md` に `## Migrated from pending.md` として追記後、削除
4. `state/pending.md` が空 → 削除

### API

| メソッド | 挙動 |
|---------|------|
| `read_pending()` | 常に空文字 `""` を返す。非推奨警告をログ出力 |
| `update_pending()` | 何もしない（no-op）。非推奨警告をログ出力 |

---

## task_queue.jsonl

タスクレジストリ。詳細は `common_knowledge/anatomy/task-architecture.md`（Layer 2）を参照。

### エントリスキーマ（TaskEntry）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `task_id` | string | 一意ID |
| `ts` | ISO8601 | 作成日時 |
| `source` | `"human"` / `"anima"` | タスク元 |
| `original_instruction` | string | 元の指示文 |
| `assignee` | string | 担当Anima名 |
| `status` | string | `pending` / `in_progress` / `done` / `cancelled` / `blocked` / `delegated` / `failed` |
| `summary` | string | 1行要約 |
| `deadline` | ISO8601 / null | 期限 |
| `relay_chain` | array | 委譲チェーン |
| `updated_at` | ISO8601 | 最終更新日時 |
| `meta` | object | `executor`, `batch_id`, `task_desc`, `origin` 等 |

---

## pending/ ディレクトリ

LLM タスクの実行キュー。詳細は `common_knowledge/anatomy/task-architecture.md`（Layer 1）を参照。

### ライフサイクル

```
pending/{task_id}.json → processing/{task_id}.json → 成功: 削除 / 失敗: failed/ に移動
```

- TTL: 24時間（`_LLM_TASK_TTL_HOURS`）。超過したタスクはスキップされる
- ポーリング間隔: 3秒（`_PENDING_WATCHER_POLL_INTERVAL`）
- `task_queue.jsonl` で `cancelled` のタスクは自動スキップ → `failed/` に移動

### JSON スキーマ

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `task_type` | string | Yes | `"llm"` |
| `task_id` | string | Yes | 一意ID |
| `batch_id` | string | No | バッチID（submit_tasks） |
| `title` | string | Yes | タイトル |
| `description` | string | Yes | 指示内容 |
| `parallel` | boolean | No | 並列実行可否 |
| `depends_on` | array | No | 先行タスクID |
| `context` | string | No | 追加コンテキスト |
| `acceptance_criteria` | array | No | 完了条件 |
| `constraints` | array | No | 制約 |
| `file_paths` | array | No | 関連ファイル |
| `workspace` | string | No | 作業ディレクトリ（エイリアス） |
| `submitted_by` | string | Yes | 投入者 |
| `submitted_at` | ISO8601 | Yes | 投入日時 |
| `source` | string | No | `"delegation"` 等 |

---

## task_results/ ディレクトリ

TaskExec が完了したタスクの結果要約を保存する。

| パラメータ | 値 |
|-----------|-----|
| ファイル名 | `{task_id}.md` |
| 最大文字数 | 2000（`_TASK_RESULT_MAX_CHARS`） |
| TTL | 7日（ハウスキーピングで自動削除） |

依存タスク（`depends_on`）はこのファイルの内容をコンテキストとして自動的に受け取る。

---

## read_subordinate_state

上司が `read_subordinate_state(name="部下名")` を呼ぶと、部下の `state/current_state.md` のみが読み取られる（`pending.md` は対象外）。
