### Task Recording in Chat

- Handle ordinary chat directly; use `submit_tasks` only when background or parallel execution is actually needed
- For human work with explicit completion criteria that may outlive this conversation, use `backlog_task` for durable tracking before starting. It remains `pending` while you work in chat; only the host claims TaskExec execution. Preserve the task through conversation boundaries and interim reports. Declare `done` only when criteria are verified, `pending` with a concrete waiting reason, or `cancelled` when explicitly stopped
- Set `done` only after verifying evidence for every completion criterion
