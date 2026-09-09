# ワーキングメモリ（state/）技術リファレンス

Anima の作業状態を管理する `state/` ディレクトリの詳細仕様。
プロンプトへの注入ロジック、サイズ制御、マイグレーション、ロック制御を含む。

---

## state/ ディレクトリ構成

```
state/
├── current_state.md          # ワーキングメモリ（自由形式Markdown）
├── task_results/              # TaskExec完了結果
│   └── {task_id}/{attempt_token}.md
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

タスクの追跡はホスト管理の正本 TaskStore が担う。確認は `list_tasks`、変更はタスクツールを使い、DB やキューファイルを直接編集しない。

### サイズ制御

| パラメータ | 値 | ソース |
|-----------|-----|-------|
| 表示上限 | 3000文字 | `_CURRENT_STATE_MAX_CHARS`（builder.py） |
| ディスク trim 上限 | 8000文字（デフォルト） | `heartbeat.current_state_max_chars`（0 = 無効） |
| Inbox時上限 | 500文字 | builder.py 内で `min(_state_max, 500)` |

**セッション境界**:

- 通常の Heartbeat / cron / 会話最終化では `current_state.md` を保持する
- セッション要約に現在状態が含まれる場合も、`current_state.md` が空/idle のときだけ書き込む
- active なタスクがない古い state は TaskBoard housekeeping によりアーカイブされる場合がある。非表示でも active なタスクは state を保護する

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

## 旧タスクファイル

`state/task_queue.jsonl` と `state/pending/` は移行・エクスポート用の証跡としてのみ保持する。稼働中のキューではない。運用者が旧書き込み処理を停止し、バックアップ付きで明示的にインポートしてから正本ランタイムを起動する。再開のためにファイルを削除・再投入・捏造しない。

## タスク実行と結果

ホストが原指示とタスクを一括保存し、実行可能な仕事を取得して各試行を記録する。`in_progress` はホスト管理。エージェントは `update_task` で `done` / `pending` / `cancelled` を宣言する。`list_tasks(detail=true)` で依存関係と要対応理由を確認する。pending は再試行を意味しない。原因の解消後、`submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])` で同じタスクを明示的に再開する。

受理された結果要約は `state/task_results/{task_id}/{attempt_token}.md`（最大2000文字）に保存される。後続にはホストが選んだ受理済み結果を渡す。古いファイルの存在だけで完了と判断しない。原記録を保存し、結果を書いて成功した試行を装わない。

長時間コマンドツールは別経路のまま。`animaworks-tool submit` は `state/background_tasks/pending/` に投入し、BackgroundTaskManager がコマンド状態・通知を管理する。詳細は `operations/background-tasks.md` と `operations/task-management.md`。

## read_subordinate_state

上司が `read_subordinate_state(name="部下名")` を呼ぶと、部下の `state/current_state.md` のみが読み取られる（`pending.md` は対象外）。
