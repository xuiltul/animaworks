# 正本任务架构

## 唯一的持久正本

LLM 任务的正本是主机管理的 TaskStore。任务 ID、完整执行输入、依赖关系、结果、执行尝试、委派别名、持久唤醒通知按所需单位一次性确定。并非将独立的文件执行队列与上司台账进行比对的结构。

确认使用 `list_tasks` / `task_tracker`，变更使用 `submit_tasks` / `delegate_task` / `update_task`。不要直接编辑 DB 或任务文件。`backlog_task` 登记仅追踪的工作，不获取执行权。

## 执行契约

1. 新建 `submit_tasks` 会原子性地公开任务与完整输入。保留原指令、约束、workspace、完成条件。
2. 主机检查依赖关系，并以唯一尝试令牌获取执行权。只有主机设置 `in_progress`。
3. 代理在 `update_task` 中声明 `done` / `pending` / `cancelled`。旧尝试不能覆盖新尝试的完成或已受理结果。
4. 依赖方的完成与持久唤醒由主机处理，不需要定期 Heartbeat。取消、异常终止、中断会留下证据与需处理理由。
5. 中断任务不要盲目重试。确认已完成操作并消除原因后，用 `submit_tasks(..., tasks=[{"task_id": "ID", "resume": true}])` 显式恢复同一未完成任务。输入与历史会被保留。无 resume 的重新投递是幂等的。

上司的委派视图是部下正本任务的别名。不经过其他可变台账或 Heartbeat 同步，双方均反映最新状态。依赖方的终止未必成功，不要将已取消视为 done 并驱动后续。

## 工作上下文与证据

`state/current_state.md` 是简洁的工作上下文，不是任务的正本。保留观察、计划、阻塞项，在通常的会话边界中保持。持久知识与步骤保存到专用记忆区域。

TaskExec 的结果摘要放在 `state/task_results/{task_id}/{attempt_token}.md`，由 TaskStore 选择已受理结果。仅凭文件名或旧摘要无法证明完成。将活动日志与原指令作为证据保留。

## 旧保存位置与命令任务

旧 `state/task_queue.jsonl` 与 `state/pending/` 仅是迁移、导出用的证据，予以保存。迁移在停止旧写入处理并备份后由运维人员显式执行。不要通过任意读取导入运行中的旧数据。

长时间命令工具也作为 TaskStore 的 `task_type="command"` 记录执行尝试。为兼容性，`animaworks-tool submit` 的结果确认仍可使用 `state/background_tasks/{task_id}.json` 与完成通知。详情参见 `reference/operations/task-management.md` 与 `operations/background-tasks.md`。

工具示例参见 `reference/operations/task-management.md`，命令执行参见 `operations/background-tasks.md`。
