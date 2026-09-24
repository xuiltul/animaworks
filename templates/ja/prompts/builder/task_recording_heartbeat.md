### タスク記録（Heartbeat中）

- heartbeat で検出した作業は list_tasks で重複を確認してから着手する。
- 外部依存で進められないタスクは理由付きで cancelled にする。
- 継続的な定時チェック項目は heartbeat.md または cron.md に追記して内在化する。
