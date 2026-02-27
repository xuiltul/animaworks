# Common Issues and Fixes

Reference for common problems and how to fix them.
Use this first; if it doesn't help, see `troubleshooting/escalation-flowchart.md`.

---

## Messages Not Received

### Symptoms

- No reply to a sent message
- Recipient says they never got it
- `send_message` ran but recipient did not react

### Causes

1. Wrong Anima name
2. Server stopped
3. Recipient between Heartbeat runs (still unread)
4. Send failed with an error

### Steps

1. **Check recipient name**
   - Verify the name in `send_message` `to` parameter
   - Names are case-sensitive
   - Use exact name from `identity.md`
   - To check org:
     ```
     search_memory(query="organization", scope="common_knowledge")
     ```
     Or `read_memory_file(path="common_knowledge/organization/structure.md")`

2. **Check server**
   - If you are running, the server is up
   - If still uncertain, report to supervisor

3. **Wait for recipient**
   - Recipients check Inbox on Heartbeat (e.g. every 30 min)
   - No reply immediately is normal; they process on next Heartbeat
   - If urgent, report to supervisor for manual trigger

4. **If send failed**
   - Log the error
   - Add to `state/current_task.md` as a blocker
   - Report to supervisor

### Examples

```
# Wrong name
send_message(to="Sakura", content="...")   # OK
send_message(to="sakura", content="...")   # May fail if name differs

# Thread reply
send_message(
    to="sakura",
    content="Understood. Starting work.",
    reply_to="msg-abc123",
    thread_id="thread-xyz789"
)
```

---

## Task Blocked

### Symptoms

- Missing information or permissions to proceed
- Waiting on another Anima’s work
- External service errors

### Causes

1. Dependency task not done
2. Insufficient permissions (permissions.md)
3. Missing information
4. External service issues

### Steps

1. **Clarify blocker**
   - Identify what is missing
   - Clarify who, what, and when

2. **Update `state/current_task.md`**
   ```
   write_memory_file(
       path="state/current_task.md",
       content="## Current Task\n\nXXX implementation\n\n### Blocked\n- Cause: Waiting for YYY\n- Blocked by: ZZZ\n- Since: 2026-02-15 10:00",
       mode="overwrite"
   )
   ```

3. **Check if you can resolve**
   - Look for alternative approaches
   - Search memory:
     ```
     search_memory(query="blocked", scope="episodes")
     search_memory(query="workaround", scope="knowledge")
     ```

4. **Escalate if unresolved** (see `troubleshooting/escalation-flowchart.md`)
   - Report to supervisor with:
     - What you tried
     - What is blocking
     - Since when
     - What you tried
   ```
   send_message(
       to="supervisor_name",
       content="[Blocked] Task: XXX implementation\nBlocked by: YYY API permission\nSince: 2026-02-15 10:00\nTried: Checked permissions.md, no entry\nRequest: Add API permission"
   )
   ```

5. **Check other work**
   - Review `state/pending.md`
   - Start another unblocked task if possible

---

## Memory Not Found

### Symptoms

- Can’t recall past work
- Procedure should exist but can’t find it
- Search returns nothing

### Causes

1. Poor keywords
2. Scope too narrow
3. Not yet recorded
4. Wrong file path

### Steps

1. **Broaden scope**
   - First try `all`:
     ```
     search_memory(query="keyword", scope="all")
     ```
   - Then narrow:
     ```
     search_memory(query="Slack setup", scope="procedures")
     search_memory(query="Slack incident", scope="episodes")
     search_memory(query="Slack", scope="knowledge")
     ```

2. **Try other keywords**
   - Synonyms (e.g. "send", "message", "notify")
   - English (e.g. "slack", "message", "send")
   - Partial match (e.g. "Chatwork" → "chatwork", "チャットワーク")

3. **Search common_knowledge**
   - If personal memory has nothing:
     ```
     search_memory(query="keyword", scope="common_knowledge")
     ```
   - Check index:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **Check directories**
   - If search fails, list dirs:
     ```
     list_directory(path="knowledge/")
     list_directory(path="procedures/")
     list_directory(path="episodes/")
     ```
   - Read directly:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     ```

5. **If it doesn’t exist**
   - Might be new
   - Check common_knowledge for related guides
   - Ask supervisor or peers
   - Record after completing the work (MUST)

### Search Scopes

| scope | Searches | Use for |
|-------|----------|---------|
| `knowledge` | Learned knowledge | Approach, tech notes |
| `episodes` | Past actions | What you did when |
| `procedures` | Procedures | How-to steps |
| `common_knowledge` | Shared guides | Org rules, system guides |
| `all` | All above | Broad search |

---

## Permission Denied

### Symptoms

- "Permission denied" or similar from tools
- Can’t read/write files
- Commands rejected

### Causes

1. Operation not allowed in `permissions.md`
2. External tool category not enabled
3. Path outside allowed scope

### Steps

1. **Check your permissions**
   ```
   read_memory_file(path="permissions.md")
   ```
   - Read paths
   - Write paths
   - Allowed commands
   - External tool categories

2. **Confirm scope**
   - Read: only paths in read_paths
   - Write: only paths in write_paths
   - Commands: only in allowed_commands

3. **If you need more**
   - Re-evaluate if the operation is needed
   - Look for alternatives within allowed scope
   - If not possible:
   ```
   send_message(
       to="supervisor_name",
       content="[Permission request]\nPurpose: XXX\nNeeded: Read /path/to/dir\nReason: Need YYY"
   )
   ```

4. **Don’t**
   - Bypass permission checks
   - Run disallowed commands in other ways
   - Use another Anima’s permissions

---

## Tools Don’t Work

### Symptoms

- "Tool not found" or similar
- External tools (Slack, Gmail, etc.) unavailable
- `discover_tools` doesn’t show what you expect

### Causes

1. Category not enabled
2. Tool not allowed in `permissions.md`
3. Missing credentials/config for external service

### Steps

1. **List tool categories**
   ```
   discover_tools()
   ```
   (No args)

2. **Enable a category**
   ```
   discover_tools(category="slack")
   ```
   Then tools in that category become available.

3. **Check permissions**
   ```
   read_memory_file(path="permissions.md")
   ```
   - `tool_categories` lists allowed categories
   - If a category is missing, it is not allowed

4. **If not allowed**
   - Ask supervisor for permission
   - Explain why the tool is needed

5. **S-mode**
   - MCP tools (`mcp__aw__*`) should work; restart if not
   - External tools via Bash: `animaworks-tool <name> --help`
   - `mcp__aw__discover_tools` for categories, then `animaworks-tool <name> <subcmd>`

6. **If a tool errors**
   - Record the error
   - Auth errors → report to supervisor (admins manage credentials)
   - Timeout → retry (up to 3 times)
   - If still failing, treat as blocker and report

See `operations/tool-usage-overview.md` for tool overview.

---

## Context Too Long

### Symptoms

- Long session
- Slower responses
- Notice about context limit

### Causes

- Long runs or many tool calls using context
- Large file reads

### Steps

1. **Save state** (MUST)
   - Write current state to `shortterm/`:
   ```
   write_memory_file(
       path="shortterm/session_state.md",
       content="## Session State\n\n### Current Task\n- XXX (50% done)\n\n### Next Steps\n1. Finish YYY\n2. Test ZZZ\n\n### Interim Results\n- AAA: BBB\n- CCC: DDD",
       mode="overwrite"
   )
   ```

2. **Update `state/current_task.md`** (MUST)
   ```
   write_memory_file(
       path="state/current_task.md",
       content="## Current Task\n\nXXX\n\n### Progress\n- 50% done\n- Resume from YYY\n\n### Notes\n- Key findings",
       mode="overwrite"
   )
   ```

3. **Save important learnings** (SHOULD)
   - Write to `knowledge/`:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# Findings on XXX\n\n## Summary\n...",
       mode="overwrite"
   )
   ```

4. **Wait for session continuation**
   - New session will load `shortterm/`
   - Re-read `state/current_task.md` and resume

### Prevention

- Avoid reading whole large files; search for what you need
- Update `state/current_task.md` regularly
- Save interim results to memory

---

## Message Sending Limited

### Symptoms

- `send_message` or `post_channel` returns an error
- "Global outbound limit reached"

### Causes

- Hit 30/hour or 100/day limit
- Or posted to same channel again within cooldown

### Steps

1. Check the error: hour vs daily limit
2. Review recent sends
3. Wait: next hour for hourly limit, next day for daily limit
4. For urgent contact: use `call_human` (not rate-limited)
5. Combine reports into fewer messages

See `communication/sending-limits.md` for details.

---

## Command Blocked

### Symptoms

- "Command blocked" or similar
- Some commands fail

### Causes

1. System-wide block list (e.g. `rm -rf /`)
2. `permissions.md` "Disallowed commands" section

### Steps

1. **Check permissions**
   ```
   read_memory_file(path="permissions.md")
   ```
   - Look for `## Disallowed commands` (or equivalent)

2. **Find alternatives**
   - Use allowed tools or commands
   - Example: if `rm -rf` is blocked, try allowed single-file delete

3. **If you need the command**
   - Ask supervisor to unblock
   - Explain why it is needed

---

## Prompt Shortened

### Symptoms

- Org context, memory guide, etc. missing from prompt
- Fewer tools
- Priming seems inactive

### Causes

Small-context models get a shorter system prompt (Tiered System Prompt).

| Tier | Context Size | Omitted |
|------|--------------|---------|
| T1 (FULL) | 128k+ | None |
| T2 (STANDARD) | 32k–128k | Distilled knowledge, smaller Priming |
| T3 (LIGHT) | 16k–32k | bootstrap, vision, specialty, distilled knowledge, memory guide |
| T4 (MINIMAL) | Under 16k | Plus permissions, Priming, org, messaging, emotion |

### Steps

1. Check model in `status.json` to infer context size
2. Fetch omitted info with `search_memory` or `read_memory_file`
3. If model change is needed, ask supervisor

---

## Other Issues

### File Not Found

- **Cause**: Wrong path or missing file
- **Fix**: Use `list_directory` to verify paths
- **Note**: `read_memory_file` uses paths relative to Anima dir; `read_file` uses absolute paths

### Command Timeout

- **Cause**: Execution exceeded `timeout`
- **Fix**: Increase `timeout` for `execute_command` (default 30s)
- **Note**: Set appropriate timeout for long-running commands

### Recipient Anima Missing

- **Cause**: Wrong name or Anima not created
- **Fix**: Check with supervisor; see `common_knowledge/organization/structure.md` for org layout
