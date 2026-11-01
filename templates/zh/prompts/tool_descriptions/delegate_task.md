【重要】将任务委派给直属部下 Anima（由部下的 TaskExec 执行。你自己不执行）。
【写之前要读（MUST）】调用前先读 `list_tasks(status="delegated")` 和 `list_tasks(status="in_progress")`，确认对同一个 PR / Issue / 对象是否已有自己委托过且未完成的任务。若有，则不要新建任务，而是带上已有的 task_id，用 `send_message` 向负责人追加指令。若可能对同一对象反复委托，还要指定 subject（例：`github:owner/repo#123`）和可选的 version（例：head SHA）。若同一对方、同一 subject、同一 version 存在未完成的委托，系统不会新建，而是返回已有的 task_id。
【写之后要读（MUST）】用 `list_tasks` 读回返回的 task_id，确认 summary 中已登记对象（例：`[PR #5215]`）。summary 开头必须带上 `[PR #N]` / `[Issue #N]` / 对象名。若返回值 JSON 的 deduplicated 为 true，则未新建任务，而是合并到了已有的委托中（不会发送 DM）。
instruction 要自包含（TaskExec 没有对话历史）。只写 PR 编号、URL、希望修正的点、完成的基准，不要用手顺条件或台账的状态词来约束。
