# Heartbeat: sora

## チェックリスト
- **MUST**: current_task.mdに進行中タスクがあるか確認。ある場合はそのタスクの続きから開始
- Inboxに未読メッセージがあるか確認
- ボード確認: read_channel(channel="general", limit=5)
- 実装中のタスクの進捗を確認
- テスト結果に異常がないか確認
- **何もなければ何もしない（HEARTBEAT_OK）**
