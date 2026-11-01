# 任务管理的方法

## 正本只有一个

确认时使用 `list_tasks(detail=true)` 或 `animaworks-tool task list`。将指令、依赖关系、执行尝试、结果、委派别名持久化到由主机管理的 TaskStore 中。禁止直接编辑数据库、伪造执行权、通过写文件来修复队列。旧的 `state/task_queue.jsonl` 和 `state/pending/` 是用于迁移和导出的证据，不是运行中的投入目标。为便于运维人员迁移而保留。

可以在普通聊天中处理的请求可直接应对。仅在需要后台执行、并行化、持续跟踪时才登记任务。人类来源的请求最优先，同等优先级时上司的请求优先于同事。交接时保留原指令、完成条件、约束、必要的上下文。

## 选择执行路径

- Inbox 处理消息和轻量回复。
- Heartbeat 确认有意义的变化并判断应对。不进行长时间编码或大量工具执行，自己的 TaskExec 使用 `submit_tasks`，直属部下使用 `delegate_task`。
- TaskExec 带工具执行已持久化的任务。执行权的获取、并行数、依赖关系、取消、尝试的恢复由主机管理，不依赖定期 Heartbeat。
- Agent/Task 的子代理启动工具无效。使用上述的投入、委派工具。

## 投入与确认

`submit_tasks` 的执行者不是部下，而是**自己的 TaskExec**。

```
submit_tasks(batch_id="report-build", tasks=[
  {"task_id": "collect", "title": "根拠収集", "description": "依頼された根拠を出典付きで収集する。", "parallel": true},
  {"task_id": "report", "title": "報告作成", "description": "収集結果から依頼された報告書を作る。", "depends_on": ["collect"]}
])
list_tasks(detail=true)
```

新任务需要 `task_id`、`title`、`description`。可选项目为 `context`、`acceptance_criteria`、`constraints`、`file_paths`、`workspace`、`parallel`、`depends_on`、`reply_to`、`model`。`workspace` 是已登记工作区的别名。模型选择遵循通常的运行时配置。不要用简短摘要替换长原指令。

投入时验证批次，一次性确定任务和执行输入。同一次投入的重复投递是幂等的，不是重试。在依赖目标正常完成之前，后续不会执行。不要仅凭文件不存在或 `pending` 显示就判断可执行。

## 结果声明与显式重启

```
update_task(task_id="TASK_ID", status="done", summary="検証済みの結果", result="根拠と成果物の場所")
update_task(task_id="TASK_ID", status="pending", summary="指定した入力を待っている")
update_task(task_id="TASK_ID", status="cancelled", summary="不要になった理由")
```

`in_progress` 是获取执行权时由主机设置的查看用状态。不要用 `update_task` 设置。没有完成声明而尝试结束的任务，有时会变成带需应对原因的 pending。pending 不是自动重试的承诺。不要为了重启而用不同 ID 复制。

解决中断原因后，显式重启同一个未结束任务。

```
submit_tasks(batch_id="resume-report", tasks=[{"task_id": "TASK_ID", "resume": true}])
```

复用已保存的输入并保留历史。运行中的尝试无法重启，已完成、已取消的任务也无法用此方法重启。若依赖目标已取消或需应对，确认详情，向委托者确认或取消已不需要的工作。不要伪造成功。

无法推进时，将事实、尝试过的事情、缺少的权限、信息、下一步告知委托者，不要重复同样的失败。仅在需要时搜索相关知识，不要仪式化。等待期间可以处理其他已允许的工作。委派工作的完成要向委托者报告，避免重复通知和不必要的知悉回复。

## 委派给部下

```
delegate_task(name="dave", instruction="API テストを実施し検証した結果を報告する", summary="API テスト")
task_tracker()
```

创建一个部下拥有的任务，以及一个上司可见的别名。双方会立即反映相同的最新状态，无需另一套台账或 Heartbeat 同步。`task_tracker(status="all")` 包含已结束的，`status="completed"` 显示 done/cancelled。若指令不明，向委派来源确认，完成时报告结果。

## 工作上下文与结果

在 `state/current_state.md` 中简洁记录观察、上下文、计划、阻塞项。不要放置任务列表的副本或永久性步骤。若没有工作上下文，则为 `status: idle`。在通常的会话边界会保留，显示限制与磁盘整理限制是分开的（`anatomy/working-memory.md`）。

TaskExec 的结果摘要保存在 `state/task_results/{task_id}/{attempt_token}.md` 中。将主机已受理的尝试结果传给后续，不要随意采用旧文件。禁止伪造结果文件，或仅凭存在就判定完成。活动日志、情节作为证据保留，无需手动双重记录每次状态迁移。

## 长时间命令工具走另一条路径

对于图像生成或 run_command 等对应的长时间外部工具，使用 `animaworks-tool submit TOOL ...`。与 `submit_tasks` 的 LLM 任务登记在同一个 TaskStore 中，作为 `task_type="command"`，由 BackgroundTaskManager 执行。执行尝试记录到 TaskStore，为兼容性保留 `state/background_tasks/{task_id}.json` 和完成通知。用 `list_background_tasks` / `check_background_task` 确认结果。详情见 `operations/background-tasks.md`。
