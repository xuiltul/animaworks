# 統一アクティビティログ — 全インタラクションの単一時系列記録化

## Overview

Animaの全インタラクション（ユーザー会話、チャネル投稿/閲覧、DM送受信、Human通知、ツール使用、ハートビート等）を**単一のアクティビティログ** (`activity_log/{date}.jsonl`) に時系列で記録し、Primingレイヤーのデータソースを統一する。

現在5箇所に分散している活動記録を1本化し、「何をいつ・誰と・どのチャネルで行ったか」をAnimaの視点で一貫して記録・想起できるようにする。

## Problem / Background

### 現状の問題

Animaの活動記録が複数の場所・形式に分散しており、Priming時の想起に非対称性がある:

| インタラクション | 記録先 | Primingでの想起 |
|---|---|---|
| ユーザー会話 | `state/conversation.json` + `transcripts/{date}.jsonl` | 毎回全履歴を参照（build_chat_prompt） |
| チャネル投稿/閲覧 | `shared/channels/{channel}.jsonl` | Priming Channel Eで最新5件+24h |
| DM送受信 | `shared/dm_logs/{pair}.jsonl` + `shared/inbox/` | **想起されない**（専用チャネルなし） |
| Human通知 | ログのみ | **想起されない** |
| ツール使用 | ログのみ | **想起されない** |
| ハートビート | `heartbeat_history/` | 直近5件のみ（system prompt直接注入） |

### 具体的に何が起きるか

1. **文脈の断絶**: 「チャネルでsakuraの投稿を見た → taroにDMで相談した → ユーザーに報告した」という一連の流れが分断される。Primingでは会話とチャネルは別々に想起され、DM部分は想起されない
2. **エピソード化の偏り**: `finalize_session()` はユーザー会話のみをエピソード化する。DM/チャネル活動/ツール使用はエピソードにならない
3. **DMの不可視**: Anima間のDMやり取りがPrimingに載らないため、「さっきtaroと話した内容」を思い出せない
4. **Human通知の忘却**: call_humanで人間に通知した事実が記録されず、重複通知のリスクがある

### Primingチャネルの分類

実装当時は episodes と shared channels が別の活動チャネルだった。現行の Priming は「参照データ系」と「活動記録系」に分かれる:

```
参照データ系（変更不要）:
  A: Sender Profile     → shared/users/
  C: Related Knowledge   → knowledge/ (RAG)
  C0: Important Knowledge → [IMPORTANT] knowledge pointers
  F: Episodes            → episodes/ (RAG)
  G: Graph Context       → MemoryBackend community context + recent facts

活動記録系（統一対象）:
  B: Recent Activity     → activity_log/{date}.jsonl + shared channels
  E: Pending Tasks       → task_queue.jsonl + task_results/
```

Channel B・Eが活動記録を扱っているが、データソースが別々で網羅性がない。

## Solution

### 1. 統一アクティビティログの導入

**ファイル配置:**
```
~/.animaworks/animas/{name}/activity_log/
├── 2026-02-17.jsonl
├── 2026-02-18.jsonl
└── ...
```

**レコード形式:**
```jsonl
{"ts":"2026-02-17T10:30:00","type":"message_received","from":"sakura","channel":"chat","content":"レポート作成して","meta":{}}
{"ts":"2026-02-17T10:30:15","type":"response_sent","to":"sakura","channel":"chat","content":"承知しました","meta":{}}
{"ts":"2026-02-17T10:35:00","type":"channel_read","channel":"general","summary":"最新5件確認","meta":{"entries_count":5}}
{"ts":"2026-02-17T10:36:00","type":"channel_post","channel":"general","content":"レポート完了しました","meta":{}}
{"ts":"2026-02-17T10:37:00","type":"dm_received","from":"taro","content":"レポートの件、確認しました","meta":{"thread_id":"..."}}
{"ts":"2026-02-17T10:38:00","type":"dm_sent","to":"taro","content":"ありがとうございます","meta":{"thread_id":"..."}}
{"ts":"2026-02-17T10:40:00","type":"human_notify","via":"slack","content":"sakuraさんへの報告完了","meta":{}}
{"ts":"2026-02-17T10:42:00","type":"tool_use","tool":"web_search","summary":"市場レポート検索","meta":{"query":"市場レポート 2026","results_count":5}}
{"ts":"2026-02-17T11:00:00","type":"heartbeat_start","meta":{}}
{"ts":"2026-02-17T11:01:00","type":"heartbeat_end","summary":"チャネル確認、未読DMなし","meta":{}}
{"ts":"2026-02-17T11:30:00","type":"cron_executed","task":"daily_report","summary":"日次レポート送信完了","meta":{}}
```

**イベントタイプ一覧:**

| type | 説明 | 必須フィールド |
|---|---|---|
| `message_received` | ユーザー/Animaからのメッセージ受信 | `from`, `channel`, `content` |
| `response_sent` | 応答送信 | `to`, `channel`, `content` |
| `channel_read` | チャネル閲覧 | `channel`, `summary` |
| `channel_post` | チャネル投稿 | `channel`, `content` |
| `dm_received` | DM受信 | `from`, `content` |
| `dm_sent` | DM送信 | `to`, `content` |
| `human_notify` | Human通知送信 | `via`, `content` |
| `tool_use` | 外部ツール使用 | `tool`, `summary` |
| `heartbeat_start` | ハートビート開始 | — |
| `heartbeat_end` | ハートビート終了 | `summary` |
| `cron_executed` | cronタスク実行 | `task`, `summary` |
| `memory_write` | 記憶書き込み | `target`, `summary` |
| `error` | エラー発生 | `summary` |

### 2. ActivityLogger クラス

```
core/memory/activity.py
```

**責務:**
- 全インタラクションを `activity_log/{date}.jsonl` に append-only で記録
- 直近N件/N日分の読み込み（フィルタリング対応）
- トークン予算内でのフォーマット出力

**主要API:**

```python
class ActivityLogger:
    def __init__(self, anima_dir: Path): ...

    # 記録
    def log(self, type: str, *, content: str = "", summary: str = "",
            from_person: str = "", to_person: str = "",
            channel: str = "", tool: str = "", via: str = "",
            meta: dict | None = None) -> None: ...

    # 取得
    def recent(self, days: int = 2, limit: int = 100,
               types: list[str] | None = None,
               involving: str | None = None) -> list[ActivityEntry]: ...

    # Priming用フォーマット出力
    def format_for_priming(self, entries: list[ActivityEntry],
                           budget_tokens: int = 1300) -> str: ...
```

### 3. Primingレイヤーの変更

**Before (旧5チャネル):**
```
A: Sender Profile       (500 toks)
B: Recent Episodes       (600 toks)  ← episodes/{date}.md
C: Related Knowledge     (700 toks)
E: Shared Channels       (700 toks)  ← channels/*.jsonl
```

**After (現行):**
```
A: Sender Profile        (500 toks)   ← 据え置き
B: Recent Activity       (1300 toks)  ← ★ activity_log/{date}.jsonl（B+E統合）
C: Related Knowledge     (1000 toks)
C0: Important Knowledge  (500 toks)
E: Pending Tasks         (500 toks)
F: Episodes              (800 toks)
G: Graph Context         (500 toks)
```

Channel B の新しい実装:

```python
async def _channel_b_recent_activity(self, sender: str, keywords: list[str]) -> str:
    """統一アクティビティログから直近の活動を取得"""
    logger = ActivityLogger(self.anima_dir)
    entries = logger.recent(days=2, limit=50)

    # 優先度フィルタリング:
    # 1. sender関連のエントリ（会話相手に関する記憶を優先）
    # 2. 直近のエントリ（時間的近接性）
    # 3. keywordsと関連するエントリ（話題の関連性）
    prioritized = self._prioritize_entries(entries, sender, keywords)

    return logger.format_for_priming(prioritized, budget_tokens=1300)
```

### 4. 記録ポイントの統合（各モジュールへの組み込み）

ActivityLoggerの `log()` 呼び出しを各所に追加:

| 場所 | イベント | 現在の記録先 |
|---|---|---|
| `anima.py::process_message()` | `message_received`, `response_sent` | conversation.json + transcripts/ |
| `messenger.py::send()` | `dm_sent` | dm_logs/ |
| `messenger.py::receive_and_archive()` | `dm_received` | inbox/ → processed/ |
| `messenger.py::post_channel()` | `channel_post` | channels/ |
| `messenger.py::read_channel()` | `channel_read` | （記録なし） |
| `notification/notifier.py` | `human_notify` | （記録なし） |
| `tooling/handler.py` | `tool_use` | （記録なし） |
| `agent.py` (heartbeat) | `heartbeat_start/end` | heartbeat_history/ |
| `agent.py` (cron) | `cron_executed` | （ログのみ） |

### 5. 既存ファイルとの関係

| 既存 | 統一後の扱い |
|---|---|
| `state/conversation.json` | **維持** — 会話ステート管理（圧縮・ターン管理）は引き続き必要。ただし `build_chat_prompt()` の履歴構成はactivity_logからも参照可能に |
| `transcripts/{date}.jsonl` | **廃止** — activity_logの `message_received`/`response_sent` で完全に代替 |
| `shared/dm_logs/{pair}.jsonl` | **廃止** — activity_logの `dm_sent`/`dm_received` で代替。ペア別の閲覧が必要な場合はactivity_logからフィルタリング |
| `shared/channels/{channel}.jsonl` | **維持** — 共有リソース（複数Animaが読み書き）。自Animaの行為記録はactivity_logにも二重記録 |
| `shared/inbox/` | **維持** — メッセージキューとしての役割は残す。受信イベントをactivity_logにも記録 |
| `episodes/{date}.md` | **維持** — consolidation（日次統合）の入力ソースをactivity_logに拡大 |
| `heartbeat_history/` | **廃止** — activity_logの `heartbeat_start/end` で代替 |

### 6. エピソード化の拡張

`finalize_session()` の入力ソースをactivity_log全体に拡大:

```python
# Before: 会話ターンのみからエピソード生成
turns = conversation.json の turns[]

# After: 当該セッション中の全活動からエピソード生成
activities = activity_log から session 期間のエントリ全件
# → 会話 + DM + チャネル + ツール使用を含む統合エピソード
```

## Implementation Plan

### Phase 1: ActivityLogger 基盤

1. `core/memory/activity.py` — ActivityLogger クラス実装
   - `ActivityEntry` データモデル（Pydantic）
   - `log()` — append-only JSONL 書き込み
   - `recent()` — 日数/件数/タイプでのフィルタ読み込み
   - `format_for_priming()` — トークン予算内フォーマット
2. テスト: `tests/test_activity_logger.py`

### Phase 2: 記録ポイント組み込み

3. `core/anima.py` — `process_message()` に `message_received`/`response_sent` 記録追加
4. `core/messenger.py` — `send()`/`receive_and_archive()`/`post_channel()`/`read_channel()` に記録追加
5. `core/notification/notifier.py` — `human_notify` 記録追加
6. `core/tooling/handler.py` — `tool_use` 記録追加
7. `core/agent.py` — heartbeat/cron の記録追加

### Phase 3: Priming統合

8. `core/memory/priming.py` — Channel B+E を統合した `_channel_b_recent_activity()` 実装
9. 旧 `_channel_b_recent_episodes()` と `_channel_e_shared_channels()` を削除
10. トークン予算の再配分（B: 1300 toks）

### Phase 4: 旧記録の廃止

11. `transcripts/` への書き込みを停止（activity_logで代替）
12. `dm_logs/` への書き込みを停止（activity_logで代替）
13. `heartbeat_history/` への書き込みを停止（activity_logで代替）
14. `builder.py` — heartbeat_history 読み込みを activity_log 経由に切り替え
15. 旧ファイル読み込みの互換レイヤー（移行期間中のみ）

### Phase 5: エピソード化拡張

16. `core/memory/conversation.py::finalize_session()` — 入力ソースをactivity_logに拡大
17. 統合エピソード形式の定義とテスト

## Scope

### In Scope

- ActivityLogger クラスの新規実装
- 全インタラクションの記録ポイント追加
- Priming Channel B+E の統合
- 旧記録先（transcripts, dm_logs, heartbeat_history）の廃止
- エピソード化の入力ソース拡大

### Out of Scope

- `conversation.json` の廃止（会話ステート管理として引き続き必要）
- `shared/channels/` の廃止（共有リソースとして維持）
- `shared/inbox/` の廃止（メッセージキューとして維持）
- RAGインデックスへの activity_log 組み込み（将来検討）
- 過去データのマイグレーション（新規記録から開始）

## Risk

- **二重記録の一時期**: Phase 2〜4の間、activity_logと旧記録先の両方に書き込む移行期間が発生する。Phase 4完了で解消
- **ログサイズ**: 活発なAnimaの activity_log が肥大化する可能性。日次ローテーション（日付別ファイル）で軽減済み。長期的には consolidation で圧縮
- **Priming精度**: Channel B+E統合時のトークン配分変更により、チャネル情報の想起量が変わる。優先度フィルタリングのチューニングが必要
