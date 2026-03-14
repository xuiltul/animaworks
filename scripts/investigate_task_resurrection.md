# 古いタスクの蒸し返し問題 — コードレベル調査レポート

## 概要

Animaが完了済み・古いタスクを再度取り上げてしまう問題の根本原因を、タスクがシステムプロンプトに注入される経路ごとに追跡した結果をまとめる。

---

## 1. Priming Engine のタスク注入（チャネルE）

**コードパス**: `core/memory/priming.py` → `_channel_e_pending_tasks()` (L1141-1208)

### 収集ソース

| ソース | フィルタリング | 備考 |
|--------|----------------|------|
| TaskQueueManager.format_for_priming() | ✅ pending/in_progress のみ | get_pending() 経由 |
| _active_parallel_tasks | 実行中タスク | メモリ上の辞書 |
| state/task_results/*.md | なし（完了済み） | 直近5件を表示 |

### 発見事項

- **TaskQueueManager 経路**: `format_for_priming()` は `get_pending()` を呼び、`status in ("pending", "in_progress")` のみを返す。done/cancelled は除外されている。
- **task_results**: 完了済みタスクの結果を表示。ヘッダーは「完了したバックグラウンドタスク」だが、Animaが「未完了」と誤認する可能性は低い。ただし、完了結果のプレビューから「フォローアップが必要」と判断して新規タスクを生む可能性はある。

---

## 2. TaskQueueManager

**コードパス**: `core/memory/task_queue.py`

### task_queue.jsonl の読み込み・フィルタリング

```python
# _load_all() (L297-339): ログをリプレイして最新状態を再構築
# - 各行は create または update イベント
# - update イベント: existing.status を上書き（latest wins）
# - 結果: task_id → 最新 TaskEntry の dict
```

- **append-only**: 完了しても行は削除されない。`_load_all()` がリプレイするため、最新 status が正しく反映される。
- **get_pending()** (L341-344): `status in ("pending", "in_progress")` のみ返却。done/cancelled/failed/blocked/delegated は除外。
- **format_for_priming()** (L384-448): `get_pending()` の結果のみを表示。done/cancelled は含まれない。

### compact() の役割

- **呼び出し**: `core/_anima_heartbeat.py` L536-553（heartbeat 成功後に実行）
- **処理**: done/cancelled/failed を task_queue_archive.jsonl に退避し、task_queue.jsonl から削除
- **compact 未実行時**: ファイルは肥大化するが、`_load_all()` の結果は正しい。get_pending() に done は含まれない。

### 結論

TaskQueueManager のフィルタリングは正しく、**task_queue.jsonl 経由では古いタスクの蒸し返しは起きない**。

---

## 3. builder.py のタスク関連セクション

**コードパス**: `core/prompt/builder.py` L804-878

### 注入されるセクション

| セクション | ソース | フィルタリング |
|------------|--------|----------------|
| current_state | memory.read_current_state() | なし |
| pending | memory.read_pending() | なし |
| task_queue | TaskQueueManager.format_for_priming() | ✅ pending/in_progress のみ |
| priming | format_priming_section(result) | 含む: pending_tasks（同上） |

### 問題点

**pending.md と current_task.md は自動クリアされない**:

- `read_pending()`: `state/pending.md` の生の内容をそのまま返す
- `read_current_state()`: `state/current_task.md` の生の内容をそのまま返す
- どちらも Anima が `update_task` や `write_memory_file` で書き込む。**更新・クリアは Anima の明示的な操作に依存**

→ 古いタスク記述が残り続け、**毎回プロンプトに注入される**可能性が高い。

---

## 4. state 管理

### current_task.md

- **読み込み**: `MemoryManager.read_current_state()` → `state/current_task.md`
- **書き込み**: `memory.update_state()` / `write_memory_file(path="state/current_task.md")`
- **自動クリア**: なし
- **部分的な整理**: `conversation.py` の `_prune_auto_detected_resolved()` が `- ✅ ...` 形式の自動検出完了行を古い順に削除（最大保持数あり）。ただし、メインの「現在のタスク」本文は対象外。

### pending.md

- **読み込み**: `MemoryManager.read_pending()` → `state/pending.md`
- **書き込み**: `write_memory_file(path="state/pending.md")` のみ
- **自動クリア**: なし
- **フィルタリング**: なし

### state/pending/ ディレクトリ（JSON ファイル）

- **用途**: TaskExec が LLM タスクを実行するための Layer 1 データ
- **ライフサイクル**: 実行成功 → ファイル削除、失敗 → `failed/` へ移動
- **builder への注入**: なし（直接は builder に渡されない）

---

## 5. Heartbeat のタスク処理

**コードパス**: `core/_anima_heartbeat.py`

- 計画フェーズで `state/pending/` に JSON を書き出す
- 完了後に `TaskQueueManager.compact()` を実行
- `_trigger_pending_task_execution()` で PendingTaskExecutor にウェイク

**古いタスクの再取り上げ**: Heartbeat 自体は task_queue.jsonl の pending タスクを「直接」実行するロジックは持たない。計画結果を `state/pending/` に書き、TaskExec が実行する。

**問題になりうる経路**: Heartbeat のプロンプトに `current_task.md` と `pending.md` が含まれる。ここに古いタスクが残っていると、**「未完了タスク」として再計画・再投入する**可能性がある。

---

## 6. STALE マーカー

**コードパス**: `core/memory/task_queue.py` L319-320, L416-418

```python
_STALE_TASK_THRESHOLD_SEC = 1800  # 30分

# format_for_priming 内
if elapsed_sec is not None and elapsed_sec >= _STALE_TASK_THRESHOLD_SEC:
    line += " ⚠️ STALE"
```

- **対象**: 表示対象は `get_pending()` の結果のみ（pending/in_progress）
- **STALE の意味**: 「30分以上 updated_at が更新されていない」

### 蒸し返しとの関係

- **ケース1**: タスクは完了したが Anima が `update_task(status="done")` を呼ばなかった  
  → 状態は pending のまま → 引き続き表示され、STALE になる  
  → Anima が「未着手」と判断して再取り組みする可能性

- **ケース2**: backlog_task で登録されたタスク  
  → TaskExec は実行しない。Anima が chat/heartbeat で処理する必要がある  
  → 完了時に `update_task` を呼ばないと、永遠に pending のまま残る

---

## 7. 根本原因の整理

### 高リスク要因

1. **pending.md / current_task.md の永続化**
   - 自動クリア・自動整理がなく、古い記述が残り続ける
   - 毎回プロンプトに注入され、Anima が「未完了」と解釈する可能性

2. **update_task が呼ばれない**
   - 完了時に `update_task(status="done")` を呼ばないと、task_queue.jsonl 上は pending のまま
   - 特に backlog_task 由来のタスクは TaskExec を経由しないため、Anima の明示的な更新が必須

3. **STALE の解釈**
   - STALE は「未更新」の注意喚起だが、Anima が「未完了」と解釈する可能性
   - 実際には完了済みでも、update されていなければ STALE 表示になる

### 低リスク要因

- task_queue.jsonl の append-only と _load_all のリプレイ: 最新 status が正しく反映される
- format_for_priming の get_pending フィルタ: done/cancelled は表示されない
- compact 未実行: ファイル肥大化はあるが、表示内容の正確性には影響しない

---

## 8. 推奨対策

### 短期

1. **プロンプトの明確化**
   - 完了時は必ず `update_task(status="done")` を呼ぶこと
   - STALE は「未更新」であり、必ずしも「未完了」ではないことを明記

2. **pending.md / current_task.md の運用**
   - タスク完了時に「完了」や「クリア」を明示的に書き込むようプロンプトで指示

### 中期

1. **current_task.md の自動整理**
   - 一定期間更新されていない行や、明示的な完了マーカー付き行を削除するロジックの検討

2. **backlog_task 完了時の自動 update**
   - 会話や heartbeat の結果から「完了」を検出した場合に、自動で `update_task` を呼ぶ仕組みの検討

3. **task_results の扱い**
   - 完了タスクの表示が「フォローアップが必要」と誤解されないよう、ヘッダーや文言の見直し

### 長期

1. **compact の定期実行**
   - heartbeat 失敗時にも compact が走るよう、別スケジュールでの実行を検討

2. **pending.md の構造化**
   - 自由形式ではなく、タスクID や status を紐づけられる形式にし、task_queue と連携する設計の検討

---

## 付録: コードパス一覧

| 機能 | ファイル | 関数/行 |
|------|----------|---------|
| Priming チャネルE | priming.py | _channel_e_pending_tasks L1141 |
| タスクキュー要約 | task_queue.py | format_for_priming L384, get_pending L341 |
| ログリプレイ | task_queue.py | _load_all L297 |
| builder 注入 | builder.py | L812-846, L873 |
| current_state 読み込み | manager.py | read_current_state L187 |
| pending 読み込み | manager.py | read_pending L190 |
| 自動整理（一部） | conversation.py | _prune_auto_detected_resolved L47 |
| heartbeat compact | _anima_heartbeat.py | L536-553 |
| update_task ハンドラ | handler_skills.py | _handle_update_task L330 |
