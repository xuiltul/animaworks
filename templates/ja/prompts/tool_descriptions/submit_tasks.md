【重要】このツールで投入したタスクはあなた自身のTaskExecが実行する（部下には送られない）。部下に任せるなら delegate_task。
【書く前に読む（MUST）】投入前に `list_tasks`（pending / in_progress）を読み、同じ PR / Issue / 対象のタスクが既に無いか確認する。あれば投入せず、既存タスクを続ける。
【書いた後に読む（MUST）】投入後に `list_tasks` で読み戻し、登録と summary（先頭に `[PR #N]` 等の対象）を確認する。
