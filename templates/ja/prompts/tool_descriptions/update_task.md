タスクの結果を宣言する。検証後は status='done'、待機理由があれば 'pending'、中止時は 'cancelled'。in_progress はホスト管理で設定しない。中断した未終了タスクを意図的に再開するには submit_tasks に既存 task_id と resume=true を渡す。保存済み入力と履歴を保持する。
