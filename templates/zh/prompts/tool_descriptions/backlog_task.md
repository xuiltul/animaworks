向任务队列添加新任务。来自人类的指令必须记录为 source='human'。Anima 之间的委派记录为 source='anima'。
【写前必读（MUST）】添加前先读 `list_tasks`，确认同一 PR / Issue / 对象没有未完成任务。若有则不添加。
【写后必读（MUST）】添加后用 `list_tasks` 回读，确认 summary（开头为 `[PR #N]` 等）。
