【重要】このツールで投入したタスクはあなた自身のTaskExecが実行する（部下には送られない）。部下に任せるなら delegate_task。
【書く前に読む（MUST）】投入前に `list_tasks`（pending / in_progress）を読み、同じ PR / Issue / 対象のタスクが既に無いか確認する。あれば新しい task_id を作らず、その既存 task_id を指定して投入する。
【pending の再投入】自分の台帳の pending（前回の run が完了宣言なしで終わったものを含む）を続けるときは、同じ task_id・元の指示（original_instruction）・必要な workspaceでこのツールに投入する。これが再実行の経路。
【書いた後に読む（MUST）】投入後に `list_tasks` で読み戻し、登録と summary（先頭に `[PR #N]` 等の対象）を確認する。
