# 任务板与面向人类的报告

## 正本与显示

TaskBoard 由正本任务和显示用 metadata 构成。
状态确认使用 `list_tasks(detail=true)` / `task_tracker()`。
`state/current_state.md` 是工作上下文，`state/task_results/` 是尝试结果，并非另一个执行台账。
显示列和 archive 不能替代执行结果或取消。

## 不做手写的双重管理

每次委派、完成、Heartbeat 时，没有义务重写 `shared/task-board.md`。
可以创建人类要求的案件报告，但需从正本中参照必要范围，并注明创建时间和未确认事项。
现有报告文件不得未经批准删除或进行每周重置。不要仅凭报告的状态就重新投入工作。

## 外部共享

仅在人类委托或已有获准运用的情况下，才进行 Slack 的发布和更新。
遵守 `slack_channel_post` / `slack_channel_update` 的权限和批准条件，
确认公司边界、机密范围、已完成的发布。不要增加自动的新发布或通知。

## 通过 CLI 读写

用 `animaworks-tool task board [--anima NAME | --all] [--stale DAYS] [--limit N]` 列出未完成任务，
用 `task show ID` 确认详情。用 `task claim ID [--ttl 30m]` 获取 lease，
用 `task note ID TEXT` 添加注记，用 `task done ID --note TEXT` 完成，
用 `task cancel ID --reason REASON` 取消，用 `task release ID` 释放 lease。
如需 JSON，则在 `board` / `show` 上附加 `--json`。

lease 是为了关闭或注记他人任务而设置的带截止日期的锁。仅在获取期间可以修改，默认 30 分钟后自然失效。
自己的任务在没有他人 lease 的情况下，即使没有 lease 也可以修改。
进行观察和判断的是 anima，机器不会自动关闭任务。
持有者的判断优先于任务正文中“无取消”“维持 pending”等过去的指定。
当持有者对自己的任务执行 `done` / `cancel` 时，会向委派方的 anima 自动发送附带理由的通知。委派方在重新发出相同工作之前会查看此通知。
