Use tools for **observation, reporting, and planning** during Heartbeat.
- OK: Channel reads, memory search, message sending, task updates, pending creation
- NG: Code changes, bulk file edits, long-running analysis or research

**[MUST] Heartbeat tool usage is limited to a maximum of 5 steps.**
Complete observation → planning → task file creation within 5 steps.
Exceeding this will block other paths (Chat/Inbox) from responding.

If heavy work is needed, write a task file to state/pending/.
Task execution is handled automatically in a separate session (TaskExec).
