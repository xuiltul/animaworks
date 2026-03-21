# Common Issues and Troubleshooting

Reference for problems commonly encountered during work and how to address them.
Each issue is documented in the format: "Symptoms → Causes → Steps".

When stuck, read this document first and follow the steps for the relevant section.
If it doesn't help, see `troubleshooting/escalation-flowchart.md` and escalate appropriately.

---

## Messages Not Received

### Symptoms

- No reply to a message you thought you sent
- Recipient says they never received it
- `send_message` ran but recipient did not react

### Causes

1. Wrong recipient name (Anima name)
2. Server is stopped
3. Recipient is between Heartbeat intervals (message remains unread until next run)
4. Send operation failed with an error
5. `intent` is unspecified or invalid (only report / question allowed; use delegate_task for task delegation)
6. Session DM limit exceeded (one send per recipient per session; max recipients per session varies by role, e.g. general: 2)

### Steps

1. **Verify recipient name**
   - Confirm the name in `send_message` `to` parameter is correct
   - Names are case-sensitive. Use the official name from `identity.md`
   - To verify:
     ```
     search_memory(query="organization", scope="common_knowledge")
     ```
     Or `read_memory_file(path="reference/organization/structure.md")` to see all Anima names in the org
   - **Note**: When chatting with a human, do not use `send_message`; reply directly in text (humans receive it). Use `call_human` to contact humans outside of chat (e.g. during heartbeat)

2. **Check server status**
   - If you are running, the server should be up
   - If still uncertain, report to your supervisor that "messages are not being received"

3. **Wait for recipient**
   - Recipients check Inbox on Heartbeat intervals (e.g. every 30 min)
   - No immediate reply is normal; they will process it on the next Heartbeat
   - If urgent, report to supervisor that you need to reach them urgently and request manual trigger

4. **If send failed**
   - Log the error message
   - Record it in `state/current_state.md` as a blocker
   - Report to supervisor

### Examples

```
# Wrong name
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # May fail if name differs

# DM requires intent (report / question only). Use delegate_task for task delegation
# Max recipients per session varies by role (general: 2). One send per recipient
send_message(
    to="aoi",
    content="Understood. Starting work.",
    intent="report",           # Required: report / question
    reply_to="msg-abc123",     # Optional: original message ID
    thread_id="thread-xyz789"  # Optional: thread ID
)

# DM for confirmation/thanks/notice only is not allowed → use post_channel (Board)
```

---

## Task Blocked

### Symptoms

- Cannot proceed because required information or permissions are missing
- Waiting for another Anima's work to complete
- External service returns errors

### Causes

1. Dependency task not completed
2. Insufficient permissions (attempted operation not allowed in permissions.json)
3. Missing required information
4. External service outage

### Steps

1. **Clarify the blocker**
   - Identify specifically what is missing
   - Organize: whose work, what work, and by when

2. **Update `state/current_state.md`**
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## Current Task\n\nXXX implementation\n\n### Blocked\n- Cause: Waiting for YYY to complete\n- Waiting on: ZZZ\n- Since: 2026-02-15 10:00",
       mode="overwrite"
   )
   ```

3. **Decide if you can resolve it yourself**
   - Consider whether an alternative approach can avoid the blocker
   - Search memory for similar past issues:
     ```
     search_memory(query="blocked", scope="episodes")
     search_memory(query="workaround", scope="knowledge")
     ```

4. **Escalate if unresolved** (see `troubleshooting/escalation-flowchart.md`)
   - Report to supervisor. Include:
     - What you tried to do
     - What is blocking you
     - Since when
     - What you tried to resolve it
   ```
   send_message(
       to="supervisor_name",
       content="[Blocked Report]\nTask: XXX implementation\nBlocked by: YYY API permission missing\nSince: 2026-02-15 10:00\nTried: Checked permissions.json, no relevant setting\nRequest: Please add API permission",
       intent="report"
   )
   ```

5. **Check for other work**
   - Review persistent task queue (`Bash: animaworks-tool task list`) and tasks under `state/pending/`
   - Start another task that is not blocked

---

## Memory Not Found

### Symptoms

- Cannot recall past work
- Procedure should exist but cannot find it
- Search returns no relevant results

### Causes

1. Search keywords are not appropriate
2. Search scope is too narrow
3. Not yet recorded (first time doing this task)
4. Wrong file path

### Steps

1. **Broaden scope and search again**
   - First try `all` scope:
     ```
     search_memory(query="search keyword", scope="all")
     ```
   - If too many results, narrow scope:
     ```
     search_memory(query="Slack setup", scope="procedures")    # Procedures only
     search_memory(query="Slack incident", scope="episodes")   # Past events only
     search_memory(query="Slack", scope="knowledge")          # Learned knowledge only
     ```

2. **Try different keywords**
   - Synonyms and related terms (e.g. "send", "message", "notify", "contact")
   - English keywords (e.g. "slack", "message", "send")
   - Partial match (e.g. "Chatwork" → "chatwork", "chat work")

3. **Search common knowledge**
   - If not in personal memory, it may be in shared knowledge:
     ```
     search_memory(query="search keyword", scope="common_knowledge")
     ```
   - Check the index:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **Check directories directly**
   - If search fails, use `Glob` to list directory contents. Omitting `path` shows the anima_dir root and subdirectories such as knowledge/, procedures/, episodes/
   - Find the file by name and read directly:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **If memory does not exist**
   - May be first-time work
   - Check common knowledge (`common_knowledge/`) for related guides
   - Ask supervisor or peers if they have knowledge
   - MUST record as memory after completing the work (for next time)
   - Old or duplicate memories can be archived with `archive_memory_file(path="...", reason="...")` to archive/ (moved, not deleted; `reason` is required)

### Search Scope Reference

| scope | Searches | Use for |
|-------|----------|---------|
| `knowledge` | Learned knowledge, know-how | Approach, tech notes |
| `episodes` | Past action logs | Fact-checking "what was done when" |
| `procedures` | Procedures | "How to" steps |
| `common_knowledge` | Shared knowledge across all Anima | Org rules, system guides |
| `all` | All of the above | Keyword existence check, broad search |

---

## Permission Denied

### Symptoms

- "Permission denied" or similar error when running a tool
- Cannot read/write files
- Command execution rejected

### Causes

1. Attempted operation not allowed in `permissions.json`
2. External tool category not enabled
3. File path outside allowed scope

### Steps

1. **Check your permissions**
   ```
   check_permissions()
   ```
   - Returns a list of available internal tools, external tools, file access, and restrictions
   - Details in `read_memory_file(path="permissions.json")`
   - Main sections in `permissions.json`:
     - "File operations" / "Readable paths": Paths you can read
     - "Command execution" / "Allowed commands": Whitelist of executable commands
     - "Disallowed commands": Blocked commands
     - External tools: Categories allowed in permissions.json are enabled

2. **Confirm the operation is allowed**
   - Your anima_dir is readable and writable. Shared dirs, subordinate management files, etc. — check with `check_permissions`
   - Commands: Only commands listed in "Allowed commands" can be executed

3. **If you need permission**
   - Reconsider whether the operation is really necessary
   - Consider alternatives within allowed scope
   - If no alternative, ask supervisor for permission:
   ```
   send_message(
       to="supervisor_name",
       content="[Permission Request]\nPurpose: XXX work\nNeeded: Read /path/to/dir\nReason: Need to reference YYY information",
       intent="question"
   )
   ```

4. **Never do the following**
   - Attempt to bypass permission checks
   - Execute disallowed commands by other means
   - Use another Anima's permissions

---

## Tools Don't Work

### Symptoms

- "Tool not found" or similar error when calling a tool
- External tools (Slack, Gmail, etc.) unavailable

### Causes

1. Tool not allowed in `permissions.json`
2. Skill file not found
3. External service credentials not configured

### Steps

1. **Look up tool usage via skill**
   - Use the `skill` tool with the skill name to get the full procedure. Skill list is shown in the `<available_skills>` block of the tool description
   - In B-mode, external tools are invoked via **Bash** with `animaworks-tool <tool> <subcommand>` when allowed

2. **Check permissions**
   ```
   check_permissions()
   ```
   - `external_tools.enabled` shows currently enabled categories; `external_tools.available_but_not_enabled` shows allowed but not yet enabled categories
   - Categories not allowed in permissions.json cannot be used

3. **If category is not allowed**
   - Ask supervisor for permission
   - Clearly state why the tool is needed when requesting

4. **MCP-integrated modes (S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI)**
   - Built-in tools are available directly (e.g. `send_message`). Restart process if not found
   - External tools: look up usage via `skill` tool and execute via **Bash** with `animaworks-tool <tool> <subcommand>`
   - Long-running tools (image gen, local LLM, etc.) run asynchronously via `animaworks-tool submit`

5. **D-mode (Cursor Agent) — common issues**
   - **CLI not found**: Ensure the `cursor-agent` CLI is installed on the host
   - **Authentication error**: Run `agent login` in a terminal
   - **Fallback**: If unresolved, set `execution_mode` to `A` or switch the model to a LiteLLM (Mode A) path

6. **G-mode (Gemini CLI) — common issues**
   - **CLI not found**: Ensure the `gemini` CLI is installed on the host
   - **Authentication error**: Run `gemini auth login` or set `GEMINI_API_KEY`
   - **Fallback**: If unresolved, set `execution_mode` to `A` or switch to LiteLLM (Mode A). `gemini/` may be remapped to `google/` for the Google provider

7. **A-mode (LiteLLM)**
   - External tools: look up usage via `skill` and execute via **Bash** with `animaworks-tool <tool> <subcommand>`

8. **If tool returns an error**
   - Record the error message accurately
   - Report to supervisor for auth errors (credential setup is admin responsibility)
   - Retry on timeout (up to 3 times)
   - If still failing after retries, report as blocker

See `operations/tool-usage-overview.md` for the full tool overview.

---

## Context Too Long

### Symptoms

- Session has run for a long time
- Responses are getting slower
- System notice about approaching context limit

### Causes

- Long work or many tool calls consuming the context window
- Large file contents loaded

### Steps

1. **Save work state to short-term memory** (MUST)
   - Write current state to `shortterm/` (use `shortterm/chat/` for chat sessions):
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## Session State\n\n### Current Task\n- XXX implementation (50% done)\n\n### Next Steps\n1. Complete YYY\n2. Test ZZZ\n\n### Important Interim Results\n- AAA findings: BBB\n- CCC setting: DDD",
       mode="overwrite"
   )
   ```
   - Use `shortterm/heartbeat/session_state.md` for Heartbeat sessions

2. **Update `state/current_state.md`** (MUST)
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## Current Task\n\nXXX implementation\n\n### Progress\n- 50% done\n- Resume from YYY next time\n\n### Notes\n- Important findings here",
       mode="overwrite"
   )
   ```

3. **Save important learnings to persistent memory** (SHOULD)
   - Save insights from the work to `knowledge/`:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# Findings on XXX\n\n## Summary\n...",
       mode="overwrite"
   )
   ```

4. **Wait for session continuation**
   - System will start a new session automatically
   - New session will include `shortterm/chat/` (or `shortterm/heartbeat/`) in context
   - Re-read `state/current_state.md` and resume work

### Prevention

- Don't read whole large files; search for needed parts only
- Update `state/current_state.md` regularly during long work
- Save interim results to memory frequently

---

## Message Sending Limited

### Symptoms

- `send_message` or `post_channel` returns an error
- Message like `GlobalOutboundLimitExceeded: Hourly send limit (N) reached...` is shown
- `ConversationDepthExceeded: Conversation with {recipient} reached 6 turns in 10 minutes...` is shown

### Causes

- **Role-based limits**: Hourly and daily send limits are applied based on default values for `role` in `status.json` (e.g. general: 15/50, manager: 60/300). Can be overridden per-anima with `max_outbound_per_hour` / `max_outbound_per_day` in `status.json`
- Repeated post to same channel within cooldown period (`heartbeat.channel_post_cooldown_s` in config.json, default 300 seconds)
- DM exchange between two parties exceeded depth limit (6 turns in 10 minutes) (`heartbeat.depth_window_s` / `heartbeat.max_depth`)

### Steps

1. **Check the error message**: Identify whether it's hourly limit, 24-hour limit, or depth limit
2. **Review send history**: Check if there were unnecessary sends
3. **Wait**: Until next hour for hourly limit, next day for 24-hour limit, or next Heartbeat cycle for depth limit
4. **Record what you wanted to send**: For this turn, don't use `send_message`; write the content to `state/current_state.md` and send in the next session
5. **Urgent contact**: `call_human` is not rate-limited; human contact remains available
6. **Combine sends**: Merge multiple reports into one message. If depth limit is reached, move complex discussions to a Board channel

See `communication/sending-limits.md` for details.

---

## Command Blocked

### Symptoms

- "PermissionDenied", "Command blocked" or similar error when trying to run a command
- Certain commands cannot be executed

### Causes

1. Command in system-wide block list (e.g. `rm -rf /`)
2. Command listed in `permissions.json` "Disallowed commands" section

### Steps

1. **Check your permissions**
   ```
   read_memory_file(path="permissions.json")
   ```
   - `## Disallowed commands` section lists blocked commands

2. **Consider alternatives**
   - See if equivalent operation is possible with allowed tools
   - Example: If `rm -rf` is blocked, single-file delete may be allowed

3. **If permission change is needed**
   - Ask supervisor to unblock
   - Clearly state why the command is needed when requesting

---

## Prompt Shortened

### Symptoms

- Information that normally appears (org context, memory guide, etc.) is missing from system prompt
- Fewer tools available
- Memory auto-recall (Priming) seems inactive

### Causes

When using a model with a small context window, the system prompt is reduced in stages (Tiered System Prompt).
Context window is inferred from model name in `status.json`; can be overridden in `~/.animaworks/models.json` or `model_context_windows` in config.json.

| Tier | Context Window | Omitted Information |
|------|----------------|----------------------|
| T1 (FULL) | 128k+ tokens | None (all info shown) |
| T2 (STANDARD) | 32k–128k tokens | Distilled knowledge (deprecated), Priming budget reduced |
| T3 (LIGHT) | 16k–32k tokens | bootstrap, vision, specialty, distilled knowledge, memory guide omitted |
| T4 (MINIMAL) | Under 16k tokens | Plus permissions, Priming, org, messaging, emotion omitted |

### Steps

1. **Fetch needed info yourself**: Use `search_memory` or `read_memory_file` to get omitted information explicitly
2. **Consult supervisor**: If model change is needed, ask supervisor

---

## Other Common Issues

### File Not Found

- **Cause**: Wrong path or file does not exist
- **Fix**: Use `Glob` to verify directory contents before specifying path
- **Note**: `read_memory_file` uses paths relative to Anima dir (e.g. `knowledge/xxx.md`, `reference/organization/structure.md`). `Read` uses absolute paths

### Cannot Specify Inbox in read_channel

- **Cause**: `read_channel` is for Board shared channels. Inbox (message inbox) is not a channel
- **Fix**: Inbox messages are processed automatically by the system. Specifying `inbox` or `inbox/` in `read_channel` will cause an error

### Command Timeout

- **Cause**: Execution exceeded `timeout`
- **Fix**: Increase `timeout` parameter for `Bash` (default: 30 seconds)
- **Note**: Set appropriate timeout for long-running commands

### Recipient Anima Does Not Exist

- **Cause**: Wrong Anima name, or that Anima has not been created yet
- **Fix**: Check with supervisor. See `reference/organization/structure.md` for org structure
