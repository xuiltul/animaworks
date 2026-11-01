这是心跳。请进行你负责的定期确认。

## 确认事项
{checklist}

对未处理的请求、异常、有截止日期的工作，选择必要的应对。
- 实际作业用 `submit_tasks`，委派用 `delegate_task`。向既有任务追加信息时绑定到既有ID，不要新建内容相同的工作。向人联系、审批时，用 `send_message` / `call_human` 直接发送给需要判断、应对的对方，不要将相同内容转发到所有层级。
- 对收到的指令，留下应对、委派、明确说明的等待理由中的任一项。
- 用 `animaworks-tool task board` 盘点自己的台账。是否关闭由持有者根据当前状况判断，优先于任务正文中“不取消”“维持pending”等过去的指定。自己无法推进的等待，写明等待理由和解除条件后记入 `task cancel ID --reason 理由`（如有保留台账则在那里写1行）；前提已变化、暂时无法着手的则关闭，之后的确认汇总为1个汇总任务；已得出结论的记入 `task done ID --note 結果`。委派方用 `--all` 查看，无法推进的委派按 `task claim ID` → `task cancel` 关闭。
- 未完成执行的 通知 先确认已完成的 操作、成果后再判断是否继续，继续时在 `submit_tasks` 的 tasks 中写入 `{{"task_id":"既存ID","resume":true}}`（原始输入已保存）。如不需要则 `update_task(status="cancelled", summary="理由")`。

以修复为目的的巡回、批量重新投入不需要。需要时参照业务步骤、审批条件。如无应对事项，仅返回 HEARTBEAT_OK。
