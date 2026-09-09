【重要】直属部下のAnimaにタスクを委譲する（部下のTaskExecが実行する。あなた自身は実行しない）。
【書く前に読む（MUST）】呼ぶ前に `list_tasks(status="delegated")` と `list_tasks(status="in_progress")` を読み、同じ PR / Issue / 対象に自分が既に頼んだ未完タスクが無いか確認する。あれば新しいタスクを作らず、既存の task_id を添えて担当者へ `send_message` で追加指示する。
【書いた後に読む（MUST）】返ってきた task_id を `list_tasks` で読み戻し、summary に対象（例: `[PR #5215]`）が入って登録されていることを確認する。summary の先頭には必ず `[PR #N]` / `[Issue #N]` / 対象名を付ける。
instruction は自己完結させる（TaskExecは会話履歴を持たない）。PR番号・URL・直してほしい点・完了の目安だけを書き、手順条件や台帳の状態語で縛らない。
