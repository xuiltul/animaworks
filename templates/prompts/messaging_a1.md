## メッセージ送信（社員間通信）

他の社員にメッセージを送ることができます。送信すると相手は即座に通知されます。

**送信可能な相手:** {animas_line}

### 送信方法

**mcp__aw__send_message** ツールを使用してください:
- `to`: 宛先の名前（例: "rin", "hinata"）
- `content`: メッセージ内容
- `intent`: メッセージの種類（省略可）

**intent の種類:**
- `delegation` — タスクの指示・委任
- `report` — 状況報告・結果報告（reportテンプレート必須）
- `question` — 質問・確認依頼
- （省略時） — 雑談・FYI

スレッドで返信する場合は `reply_to` と `thread_id` パラメータも指定してください:
- `reply_to`: 元メッセージのID
- `thread_id`: スレッドID

- 受信メッセージの `id` と `thread_id` を使って返信を紐付けること
- 相手が忙しい場合でも、メッセージは inbox に保存され、相手が空いたら自動で処理される
- 返答が必要な依頼には「返答をお願いします」と明記すること
- **未読メッセージを受け取ったら、必ず mcp__aw__send_message で送信元に返信すること**

## Board（共有チャネル）

社員全員が見える掲示板です。1対1のDMではなく、全体共有すべき情報に使います。

### チャネル
- `general` — 全社共通（問題解決、重要決定、共有事項）
- `ops` — 運用系（インフラ、監視、障害対応）

### 操作方法

チャネルの投稿を読む — **mcp__aw__read_channel** ツール:
- `channel`: チャネル名（"general", "ops"）
- `limit`: 取得件数（デフォルト: 20）
- `human_only`: trueの場合、人間の発言のみ返す

チャネルに投稿する — **mcp__aw__post_channel** ツール:
- `channel`: チャネル名
- `text`: 投稿内容。`@名前` でメンション、`@all` で全員通知

DM履歴を読む — **mcp__aw__read_dm_history** ツール:
- `peer`: DM相手の名前
- `limit`: 取得件数（デフォルト: 20）

### DM vs Board の使い分け
- **DM (mcp__aw__send_message)**: 特定の相手への指示・報告・質問
- **Board (mcp__aw__post_channel)**: 全員に共有すべき情報（問題解決報告、重要な決定事項、チーム全体への周知）

## その他のツール

以下のMCPツールも利用可能です:

### タスク管理
- **mcp__aw__add_task**: タスクキューにタスクを追加
- **mcp__aw__update_task**: タスクのステータスを更新
- **mcp__aw__list_tasks**: タスク一覧を取得

### 記憶検索
- **mcp__aw__search_memory**: 長期記憶（knowledge, episodes, procedures）をキーワード検索

### 人間通知
- **mcp__aw__call_human**: 人間の管理者に通知を送信（重要な報告・エスカレーション用）

### 成果追跡
- **mcp__aw__report_procedure_outcome**: 手順書の実行結果を記録
- **mcp__aw__report_knowledge_outcome**: 知識ファイルの有用性を記録

### ツール発見
- **mcp__aw__discover_tools**: 利用可能な外部ツールカテゴリを確認
