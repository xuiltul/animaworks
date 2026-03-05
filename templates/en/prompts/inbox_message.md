You have messages in your inbox. Review the following and reply appropriately.

{messages}

## Response Guidelines
- Answer questions directly
- Reply with acknowledgment and timeline for requests
- **[MUST] If you identify work that needs to be done, you MUST formalize it as a task. Do not just reply and forget.**
  - Delegate to subordinates → `delegate_task`
  - Do it yourself later → `plan_tasks` (written to state/pending/, executed by TaskExec in a separate session)
- Keep replies concise (no lengthy responses)

### Replying to External Platform (Slack/Chatwork) Messages
When a message has `[platform=slack channel=CHANNEL_ID ts=TS]` metadata:
- **Always reply in thread**: `animaworks-tool slack send '#channel-or-CHANNEL_ID' 'message' --thread TS`
- Pass the ts value directly to `--thread`

{task_delegation_rules}
