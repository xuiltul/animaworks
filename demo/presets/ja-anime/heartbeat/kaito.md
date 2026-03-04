# Heartbeat: kaito

## チェックリスト
- **MUST**: current_task.mdに進行中タスクがあるか確認。ある場合はそのタスクの続きから開始
- Inboxに未読メッセージがあるか確認
- ボード確認: read_channel(channel="general", limit=5)
- 部下（sora, hina）の状況確認: org_dashboard
- ブロッカーや遅延がないか確認
- **何もなければ何もしない（HEARTBEAT_OK）**
