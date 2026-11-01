# 任务投入与委派

## 执行路径

不使用 Agent/Task 的原生子代理启动，而是使用公开的任务工具。
能在普通聊天中完成的工作直接执行。仅需持续跟踪时使用 `backlog_task`，
自己的后台执行使用 `submit_tasks`，向有效的直属部下委派使用
`delegate_task(name="担当名", instruction="原指示と完了条件", summary="要約")`。
遵循工具的提供范围和权限，不向无效的负责人委派，也不擅自切换到其他路径。
Heartbeat 用于判断和投入，长时间的实际作业交给 TaskExec。

## 交接的信息

执行者不会自动共享对话历史。需传递原始指令、目的、相关文件及可确定范围内的位置、
现状、完成条件、批准条件、禁止事项。不编造不存在的路径或行号。
根据用途使用 `description` 和 `context`、`acceptance_criteria`、`constraints`、`file_paths`。
也要保留模型和已注册 workspace 的指定。不要指示向其他 Anima 的个人目录写入。

`submit_tasks(batch_id="work", tasks=[{"task_id":"job","title":"仕事","description":"具体的な依頼"}])`
会一次性公开任务和执行输入。相同 ID 的重新发送不是重新执行。
`parallel:true` 可在 worker 数上限内并行，`depends_on` 会等待先行任务完成和尝试结束。
若依赖对象已取消或未完成则需要确认，不要推测为成功。

## 状态、结果、恢复

状态通过 `list_tasks(detail=true)` 确认，委派跟踪通过 `task_tracker()` 确认。
跟踪 ID 是部下拥有的同一任务的 alias，无需台账同步或文件补救。
`task_tracker(status="all")` 为全部，`status="completed"` 为 done/cancelled。
执行权限，`in_progress` 由主机管理。结果以带依据的 `done`、
带具体等待理由的 `pending`、明确中止的 `cancelled` 来声明。

收到未完成通知后，确认已完成的外部操作和成果，仅在适合继续时
用 `submit_tasks(batch_id="resume-job", tasks=[{"task_id":"job","resume":true}])` 复用已保存的输入。
执行中、已完成、已取消的工作无法用此方法恢复。不要无限重新投入。
结果摘要为 `state/task_results/{task_id}/{attempt_token}.md`。不要仅因文件存在就视为完成。

## 重复与报告

若已知同一请求存在未完成的工作，则向其 ID 传递追加信息。
不要仅因怀疑重复就自动取消或覆盖旧工作，应确认负责人和执行状态。
保留必要的批准和独立审查。结果报告给需要判断的委托者，不要求向所有层级
转发相同内容，也不要求向另一份手写台账重复记录。
保存的详细信息请参照 `common_knowledge/anatomy/task-architecture.md`。
