You have messages in your inbox. Review the following and reply appropriately.

{messages}

## Response Guidelines
- Answer questions directly
- Reply with acknowledgment and timeline for requests
- **[MUST] If you identify work that needs to be done, you MUST formalize it as a task. Do not just reply and forget.**
  - Delegate to subordinates → `delegate_task`
  - Do it yourself later → `plan_tasks` (written to state/pending/, executed by TaskExec in a separate session)
- Keep replies concise (no lengthy responses)

### Replying to External Platform Messages
When a message has `[reply_instruction: ...]` metadata:
- **Always follow the instruction** to reply (execute via `execute_command`)
- Replace `{返信内容}` with your actual reply text
- Do NOT use `send_message` (it sends DMs, not thread replies)

{task_delegation_rules}
