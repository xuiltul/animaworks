# Priming チャネル技術リファレンス

PrimingEngine が実行する全チャネルの詳細仕様。
バジェット、検索ソース、フィルタリング、動的調整を含む。

並列取得は **5 チャネル**（A / B / C / E / F）である。C0（important_knowledge）は Channel C と同一パイプライン内の補助ブロック。旧 Distilled Knowledge に相当した Channel D や「6 チャネル」表記は廃止済み。

---

## チャネル一覧

| チャネル | バジェット（トークン） | ソース | trust |
|---------|---------------------|--------|-------|
| A: sender_profile | 500 | `shared/users/{sender}/index.md` | medium |
| B: recent_activity | 1300 | `activity_log/` + shared channels | trusted |
| C: related_knowledge | 1200 | RAG ベクトル検索（knowledge + common_knowledge） | medium / untrusted |
| C0: important_knowledge | 300 | `[IMPORTANT]` タグ付きチャンク | medium |
| E: pending_tasks | 500 | `task_queue.jsonl` + `task_results/` | trusted |
| F: episodes | 500 | RAG ベクトル検索（episodes/） | medium |

追加注入:

| 項目 | バジェット | ソース | trust |
|------|-----------|--------|-------|
| Recent outbound | 上限なし（最大3件） | activity_log（直近2時間、`channel_post` / `message_sent`） | trusted |
| Pending human notifications | 500 | `human_notify` イベント（直近24時間） | trusted |

スキル・手続きの本文は Priming では注入されない。システムプロンプトのスキルカタログに示されたパス（例: `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`, `procedures/baz.md`）を `read_memory_file` で読み込む。

---

## Channel A: sender_profile

送信者のユーザープロファイルを注入する。

- **ソース**: `shared/users/{sender}/index.md` を直接読み取り
- **バジェット**: 500トークン
- **送信者不明時**: スキップ

---

## Channel B: recent_activity

直近の活動タイムラインを注入する。

- **ソース**: `activity_log/{date}.jsonl` + 共有チャネルの最新投稿
- **バジェット**: 1300トークン

**Priming 注入と明示検索の違い**: Channel B は **自動**でバジェット内に要約注入する。過去の行動ログを **キーワードで広く探す** 用途は `search_memory(scope="activity_log")`（BM25。`scope="all"` では activity_log BM25 を RRF でベクトル結果と統合）。注入は予算・フィルタ制約あり、ツール検索はクエリ主導で別物。

### トリガー別フィルタリング

| トリガー | 除外されるイベントタイプ |
|---------|----------------------|
| `heartbeat` / `cron:*` | `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` |
| `chat` | `cron_executed` |

---

## Channel C: related_knowledge

RAG ベクトル検索で関連知識を注入する。

- **バジェット**: 1200トークン
- **検索方式**: Dual-query（メッセージコンテキスト + キーワードのみ）
- **検索対象**: 個人 `knowledge/` + `shared_common_knowledge` コレクション
- **最小スコア**: `config.json` の `rag.min_retrieval_score`（デフォルト 0.3）

### trust 分離

検索結果はチャンクの `origin` に基づいて trust レベルで分離される:

| trust | 対象 | 処理 |
|-------|------|------|
| `medium` | 個人 knowledge、common_knowledge | 優先的にバジェットを消費 |
| `untrusted` | 外部プラットフォーム由来（`origin_chain` に `external_platform` を含む） | 残りバジェットで注入。`origin=ORIGIN_EXTERNAL_PLATFORM` タグ付き |

---

## Channel C0: important_knowledge

`[IMPORTANT]` タグ付きチャンクの概要ポインタを常時注入する。

- **バジェット**: 300トークン
- **対象**: `knowledge/` 内の `[IMPORTANT]` タグ付きチャンク
- **注入形式**: 概要ポインタのみ（全文ではない）。詳細は `read_memory_file` で取得
- **用途**: 重要な業務ルール・判断基準の確実な想起

---

## Channel E: pending_tasks

タスクキューの要約を注入する。

- **バジェット**: 500トークン
- **ソース**: `TaskQueueManager.format_for_priming()`
- **内容**:
  - `pending` / `in_progress` タスクの一覧と要約
  - `source: human` タスクに 🔴 HIGH マーカー
  - 30分以上更新なしのタスクに ⚠️ STALE マーカー
  - 期限超過タスクに 🔴 OVERDUE マーカー
  - アクティブな並列タスク（submit_tasks バッチ）の進捗
  - `task_results/` からの完了タスク結果
  - `status: failed` + `meta.executor == "taskexec"` の失敗タスク

---

## Channel F: episodes

RAG ベクトル検索で関連エピソードを注入する。

- **バジェット**: 500トークン
- **検索対象**: `episodes/` コレクション（ChromaDB）
- **最小スコア**: Channel C と共通（`rag.min_retrieval_score`）

---

## 動的バジェット調整

`config.json` の `priming.dynamic_budget: true`（デフォルト）で有効。

### メッセージタイプ別バジェット

| メッセージタイプ | バジェット | 設定キー |
|----------------|-----------|---------|
| greeting | 500 | `priming.budget_greeting` |
| question | 1500 | `priming.budget_question` |
| request | 3000 | `priming.budget_request` |
| heartbeat（フォールバック） | 200 | `priming.budget_heartbeat` |

### Heartbeat バジェット計算

```
heartbeat_budget = max(budget_heartbeat, context_window × heartbeat_context_pct)
```

- `heartbeat_context_pct`: デフォルト 0.05（コンテキストウィンドウの5%）
- 例: context_window=200000 → `max(200, 200000 × 0.05)` = 10000

---

## Hebbian LTP（長期増強）

Priming で検索・表示されたチャンクは `record_access()` により活性度が更新される。これにより、頻繁に想起される記憶の忘却が防止される（Forgetting エンジンとの連携）。
