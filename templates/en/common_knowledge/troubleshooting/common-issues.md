# Common Issues and Troubleshooting

Reference for problems commonly encountered during work and how to address them.
Each issue is documented in the format: **Symptoms → Causes → Steps**.

When stuck, read this document first and follow the steps for the relevant section.
If that does not resolve the issue, see `troubleshooting/escalation-flowchart.md` and escalate appropriately.

---

## Messages Not Received

### Symptoms

- No reply to a message you thought you sent
- The recipient says they never received it
- You ran `send_message` but the recipient does not react

### Causes

1. Incorrect recipient specification (canonical Anima name, user alias, `slack:` / `chatwork:` prefixes, etc.) or a value that does not match the resolution order
2. Server is stopped
3. Recipient is between Heartbeat intervals (message stays unread until the next run)
4. Send failed with an error (global send limits, conversation depth limits, per-session DM limits, `RecipientResolutionError`, etc.)
5. `intent` missing or invalid. For DMs, only `report` / `question`. For task delegation use `delegate_task` (attaching `intent="delegation"` to `send_message` returns a deprecation message)
6. Per-session DM limits exceeded (**one message per recipient per session**. **Maximum distinct recipients** per session is `max_recipients_per_run` from `status.json` according to `role` — see table below. Override per Anima with the same field in `status.json`)

**`max_recipients_per_run` by role (`core/config/schemas.py` `ROLE_OUTBOUND_DEFAULTS`)**

| role | Max distinct recipients per session (one message each) |
|------|--------------------------------------------------------|
| manager | 10 |
| engineer | 5 |
| writer | 3 |
| researcher | 3 |
| ops | 2 |
| general | 2 |

### Steps

1. **Verify recipient name and address format**
   - Confirm the `send_message` `to` parameter resolves to the intended party
   - Resolution order in `core/outbound.py` `resolve_recipient` is roughly:
     1. **Exact** match with a known Anima name (case-sensitive) → internal
     2. **Alias** from `config.json` `external_messaging.user_aliases` (case-insensitive) → external (`preferred_channel`)
     3. `slack:USERID` / `chatwork:ROOMID` → external direct
     4. Bare Slack user ID (`U` + 8+ alphanumeric) → Slack direct
     5. **Case-insensitive** match with a known Anima name → internal
     6. Otherwise → resolution fails
   - For reliable internal delivery, use the **canonical name** from `~/.animaworks/animas/<name>/` or `reference/organization/structure.md`
   - How to verify:
     ```
     search_memory(query="organization", scope="common_knowledge")
     ```
     Or `read_memory_file(path="reference/organization/structure.md")` to list all Anima names in the org
   - **Note**: While chatting with a human, you cannot use `send_message`; reply in plain text and the human receives it. To contact a human outside chat (e.g. during Heartbeat), use `call_human`

2. **Check server status**
   - If you are running, the server should normally be up
   - If still uncertain, report to your supervisor that messages are not being received

3. **Wait for the recipient**
   - Recipients check Inbox on Heartbeat intervals (e.g. every 30 minutes)
   - No immediate reply is normal; processing happens on the next Heartbeat
   - If urgent, report to your supervisor that you need urgent contact and request a manual trigger

4. **If send failed**
   - Record the error message
   - Note the blocking reason in `state/current_state.md`
   - Report to your supervisor

### Examples

```
# Wrong name
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # May error if the name differs

# DM requires intent (report / question only). Delegation: delegate_task
# Max distinct recipients per session depends on role (e.g. general: up to 2, engineer: up to 5). One send per recipient
send_message(
    to="aoi",
    content="Understood. Starting work.",
    intent="report",           # Required: report / question
    reply_to="msg-abc123",     # Optional: original message ID
    thread_id="thread-xyz789"  # Optional: thread ID
)

# DM for acknowledgment, thanks, or FYI only is not allowed → use post_channel (Board)
```

---

## Task Blocked

### Symptoms

- You try to proceed but lack required information or permissions
- Waiting for another Anima to finish work
- External services return errors

### Causes

1. Dependency task not completed
2. Insufficient permissions (operation not allowed in `permissions.json`. Even with `permissions.md` only, first `load_permissions` generates JSON and renames MD to `.bak`)
3. Missing required information
4. External service outage

### Steps

1. **Clarify the blocker**
   - Identify specifically what is missing
   - Organize: **whose** work, **what** work, and **by when** it is needed

2. **Update `state/current_state.md`**
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## Current Task\n\nXXX implementation\n\n### Blocked\n- Cause: Waiting for YYY to complete\n- Waiting on: ZZZ\n- Since: 2026-02-15 10:00",
       mode="overwrite"
   )
   ```

3. **Decide if you can resolve it yourself**
   - Consider whether another approach can avoid the blocker
   - Search memory for similar past issues:
     ```
     search_memory(query="blocked", scope="episodes")
     search_memory(query="workaround", scope="knowledge")
     ```

4. **Escalate if unresolved** (see `troubleshooting/escalation-flowchart.md`)
   - Report to your supervisor. Include:
     - What you tried to do
     - What is blocking you
     - Since when you have been blocked
     - What you already tried
   ```
   send_message(
       to="supervisor_name",
       content="[Blocked Report]\nTask: XXX implementation\nBlocked by: YYY API permission missing\nSince: 2026-02-15 10:00\nTried: Checked permissions.json; no matching setting\nRequest: Please add API permission",
       intent="report"
   )
   ```

5. **Look for work you can still do while blocked**
   - Persistent task queue: if tools are available, use `list_tasks` or `Bash: animaworks-tool task list`
   - Check Heartbeat-emitted LLM tasks under `state/pending/*.json` for other work
   - Start another task that is not blocked

---

## Memory Not Found

### Symptoms

- Cannot recall past work
- A procedure should exist but you cannot find it
- Search returns no relevant hits

### Causes

1. Search keywords are not appropriate
2. Search scope (`scope`) is too narrow
3. Not yet written to memory (first time doing the task)
4. Wrong file path

### Steps

1. **Broaden scope and search again**
   - First try the `all` scope:
     ```
     search_memory(query="your keyword", scope="all")
     ```
   - If there are too many results, narrow scope:
     ```
     search_memory(query="Slack setup", scope="procedures")    # Procedures only
     search_memory(query="Slack incident", scope="episodes")   # Past events only
     search_memory(query="Slack", scope="knowledge")           # Learned knowledge only
     ```

2. **Try different keywords**
   - Synonyms and related terms (e.g. “send”, “message”, “notify”, “contact”)
   - English keywords (e.g. `"slack"`, `"message"`, `"send"`)
   - Partial matches (e.g. “Chatwork” → `chatwork`, “チャットワーク”)

3. **Search common knowledge**
   - If not in personal memory, it may exist in shared knowledge:
     ```
     search_memory(query="your keyword", scope="common_knowledge")
     ```
   - Check the index:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **Inspect directories directly**
   - In Mode S (Claude Agent SDK), the built-in `Glob` can list under the Anima directory. In Mode A and similar, open known paths with `read_memory_file` or cast a wide net with `search_memory`
   - If you know the file name, read it directly:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **If memory truly does not exist**
   - The task may be a first-time effort
   - Check `common_knowledge/` for related guides
   - Ask your supervisor or peers for know-how
   - After completing the work, you **MUST** record it as memory (for next time)
   - Old or duplicate memories can be moved (not deleted) to `archive/` with `archive_memory_file(path="...", reason="...")` (`reason` is required)

### Search Scope Reference

| scope | Searches | Use for |
|-------|----------|---------|
| `knowledge` | Learned knowledge, know-how | Approach, technical notes |
| `episodes` | Past action logs | Fact-checking “what happened when” |
| `procedures` | Procedures | “How to” steps |
| `common_knowledge` | Knowledge shared across all Anima | Org rules, system guides |
| `skills` | Skills and common skills (vector search) | Discovering and searching skills |
| `activity_log` | Unified activity timeline (tool results, messages, etc.) | “What did I just read?” recent actions |
| `all` | All of the above (vector search + activity_log BM25 fused via RRF) | Keyword existence, broad search |

---

## Permission Denied

### Symptoms

- Errors such as “no permission” or “Permission denied” when running a tool
- Cannot read or write files you expected to access
- Command execution rejected

### Causes

1. Operation not allowed in `permissions.json` (with `permissions.md` only, `load_permissions` normalizes to JSON on load; invalid JSON may fall back to permissive defaults with a warning)
2. External tool category not enabled in the registry (`check_permissions` → `available_but_not_enabled`)
3. File path outside allowed scope (writes to protected files, outside `file_roots`, etc.). **Global** denials also come from `permissions.global.json` and framework-side patterns

### Steps

1. **Check your permissions**
   ```
   check_permissions()
   ```
   - Returns JSON. Review `internal_tools`, `external_tools.enabled` / `available_but_not_enabled`, `file_access` (read/write), and `restrictions` (command deny lists, etc.)
   - Raw settings: `read_memory_file(path="permissions.json")` if present, otherwise `permissions.md`
   - The permission text injected into the system prompt is formatted at runtime from JSON

2. **Confirm the operation is allowed**
   - Under your `anima_dir`, read/write is generally allowed (except protected files such as `identity.md`). Reading supervisors’/peers’ `activity_log` or subordinates’ `state/` depends on role and appears under `check_permissions` → `file_access`
   - Shell commands: follow `commands` (allow/deny) in `permissions.json`. Globally dangerous patterns are also blocked by the framework

3. **If you need additional permission**
   - Reconsider whether the operation is truly necessary
   - See if an alternative within allowed scope works
   - If not, ask your supervisor to add permission:
   ```
   send_message(
       to="supervisor_name",
       content="[Permission Request]\nPurpose: XXX work\nNeeded: Read /path/to/dir\nReason: Need to reference YYY",
       intent="question"
   )
   ```

4. **Never do the following**
   - Try to bypass permission checks
   - Run disallowed commands through other means
   - Attempt to use another Anima’s permissions

---

## Tools Don't Work

### Symptoms

- “Tool not found” or similar when invoking a tool
- External tools (Slack, Gmail, etc.) unavailable

### Causes

1. The tool is not allowed in `permissions.json` (or MD-derived settings after normalization), or a gated action is not explicitly permitted
2. Skill file not found
3. External service credentials not configured

### Steps

1. **Confirm how to use the tool via skills**
   - Use `read_memory_file` with the path from the system prompt skill catalog (e.g. `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`) to load the full procedure
   - In B-mode, when external tools are allowed, you can call them with `Bash: animaworks-tool <tool> <subcommand>`

2. **Check permissions**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`: external tool categories on this Anima’s registry (what actually reaches the session)
   - `external_tools.available_but_not_enabled`: implemented in the framework but not on this Anima’s registry — cross-check with `permissions.json` allowances, gated actions, and execution mode

3. **If still unavailable**
   - Verify the tool/action is allowed in `permissions.json` and credentials exist (e.g. `shared/credentials.json`)
   - If it still fails, ask your supervisor, stating **why** the tool is needed

4. **MCP-integrated modes (S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI)**
   - Built-in tools are available without a prefix (e.g. `send_message`). If missing, restart the process
   - External tools: load CLI usage from the skill file with `read_memory_file`, then run `animaworks-tool <tool> <subcommand>` via **Bash** (the agent’s Bash tool)
   - Long-running tools (image generation, local LLM, etc.): use `animaworks-tool submit` for async execution

5. **D-mode (Cursor Agent) — common issues**
   - **CLI not found**: Confirm `cursor-agent` is installed on the host
   - **Auth error**: Run `agent login` in a terminal
   - **Fallback**: If unresolved, set `execution_mode` to `A` or switch the model to a LiteLLM (Mode A) path

6. **G-mode (Gemini CLI) — common issues**
   - **CLI not found**: Confirm the `gemini` CLI is installed on the host
   - **Auth error**: Run `gemini auth login` or set env var `GEMINI_API_KEY`
   - **Fallback**: If unresolved, set `execution_mode` to `A` or switch to LiteLLM (Mode A). The `gemini/` prefix may be remapped to `google/` for the Google provider

7. **A-mode (LiteLLM)**
   - External tools: confirm usage with `read_memory_file` on the skill path, then run `animaworks-tool <tool> <subcommand>` via **Bash**

8. **If the tool returns an error**
   - Record the error message accurately
   - For auth errors, report to your supervisor (credential setup is an admin responsibility)
   - For transient timeouts or rate limits, wait briefly and retry (retry count and spacing depend on the tool implementation and server settings)
   - If it does not improve, report as a blocker

See `operations/tool-usage-overview.md` for the full tool picture.

---

## Context Too Long

### Symptoms

- Session has been running a long time
- Responses are slowing down
- System notice that you are near the context limit

### Causes

- Long work or many tool calls consuming the context window
- Large amounts of file content loaded

### Steps

1. **Save work state to short-term memory** (MUST)
   - Write current state under `shortterm/` (for chat sessions use `shortterm/chat/`):
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## Work State\n\n### Task in progress\n- XXX implementation (50% done)\n\n### Next steps\n1. Finish YYY\n2. Test ZZZ\n\n### Important interim results\n- AAA findings: BBB\n- CCC setting: DDD",
       mode="overwrite"
   )
   ```
   - For Heartbeat sessions use `shortterm/heartbeat/session_state.md`

2. **Update `state/current_state.md`** (MUST)
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## Current Task\n\nXXX implementation\n\n### Progress\n- 50% done\n- Resume from YYY next time\n\n### Notes\n- Important findings here",
       mode="overwrite"
   )
   ```

3. **Persist important learnings** (SHOULD)
   - Save insights under `knowledge/`:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# Findings on XXX\n\n## Discoveries\n...",
       mode="overwrite"
   )
   ```

4. **Wait for session continuation**
   - The system starts a new session automatically
   - The new session includes `shortterm/chat/` (or `shortterm/heartbeat/`) in context
   - Re-read `state/current_state.md` and resume

### Prevention

- Avoid reading entire large files; search for only what you need
- Update `state/current_state.md` regularly during long work
- Write interim results to memory often

---

## Message Sending Limited

### Symptoms

- `send_message` or `post_channel` returns an error
- `GlobalOutboundLimitExceeded: Hourly send limit (N messages) reached...` or the 24-hour variant
- `GlobalOutboundLimitExceeded: Sending blocked because activity log could not be read` (`core/cascade_limiter.py` — when the sender’s `activity_log` cannot be read)
- `ConversationDepthExceeded: Conversation with {recipient} reached 6 turns in 10 minutes...`

### Causes

- **Role-based global limits**: Counts `dm_sent` / `message_sent` / `channel_post` from `activity_log` for 1-hour and 24-hour windows (`ConversationDepthLimiter.check_global_outbound`). Override per Anima with `max_outbound_per_hour` / `max_outbound_per_day` in `status.json`; if unset, use role defaults from `ROLE_OUTBOUND_DEFAULTS`

**Default hourly / 24-hour limits by role (code defaults)**

| role | 1 hour | 24 hours |
|------|--------|----------|
| manager | 60 | 300 |
| engineer | 40 | 200 |
| writer | 30 | 150 |
| researcher | 30 | 150 |
| ops | 20 | 80 |
| general | 15 | 50 |

- Repeated post to the same channel inside the cooldown window (`heartbeat.channel_post_cooldown_s` in `config.json`, default 300 seconds)
- Two-party back-and-forth exceeded depth limits (`Messenger.send` → `ConversationDepthLimiter.check_depth`. **Internal Anima DMs only**. `heartbeat.depth_window_s` / `heartbeat.max_depth`, defaults **600 seconds** and **max 6 turns**. Copy may say “10 minutes / 6 turns”)
- Activity log read error (disk, permissions, corruption, etc.) → sends blocked on the safe side

### Steps

1. **Read the error**: Identify hourly limit, 24-hour limit, depth limit, or activity_log failure
2. **Review send history**: Check for unnecessary sends
3. **Wait**: Until the next hour for hourly limits, next day for 24-hour limits, or until the depth window clears (the message may include an approximate “next send allowed” time)
4. **Record intended content**: When at a cap, as the message instructs, skip `send_message` this turn; write to `state/current_state.md` and send in a later session
5. **If activity_log failed**: Ask an admin to check logs, disk, and that Anima’s `activity_log/` (blocking depends on the sender’s log read)
6. **Urgent human contact**: `call_human` is exempt from these global limits
7. **Batch and move to Board**: Combine multiple reports into one message; when depth limits bite, move to Board (`post_channel`)

See `communication/sending-limits.md` for details.

---

## Command Blocked

### Symptoms

- “PermissionDenied”, “Command blocked”, or similar when running a command
- Only certain commands fail

### Causes

1. Command matches framework / `permissions.global.json` global deny patterns (e.g. `rm -rf /`, etc.)
2. Command listed under `commands.deny` in `permissions.json`

### Steps

1. **Check your permissions**
   ```
   check_permissions()
   ```
   - `restrictions` lists denied commands. Also read `read_memory_file(path="permissions.json")` directly (legacy: `permissions.md`)

2. **Consider alternatives**
   - See if an equivalent operation is possible with allowed tools
   - Example: If `rm -rf` is blocked, deleting individual files may still be allowed

3. **If a permission change is needed**
   - Ask your supervisor to unblock
   - Clearly explain **why** the command is needed

---

## Prompt Shortened

### Symptoms

- System prompt feels thinner than usual; Priming (auto-recall) is nearly empty
- After a long conversation or a large user message, the prompt seems rebuilt before the reply

### Causes

There are two major layers.

**1. Priming (auto-recall) tier** — `resolve_prompt_tier(context_window)` in `core/prompt/builder.py` chooses the tier from the estimated context window. Resolution order is `core/prompt/context.py` `resolve_context_window`: **`~/.animaworks/models.json` (SSoT)** → deprecated `config.json` `model_context_windows` → in-code fallbacks such as `MODEL_CONTEXT_WINDOWS` → default 128k.

| Tier | Condition (`context_window`) | Priming handling (`core/_agent_priming.py`) |
|------|------------------------------|---------------------------------------------|
| full | **≥ 128_000** | All six channels formatted with `format_priming_section` and included as-is |
| standard | **≥ 32_000 and < 128_000** | Same fetch as above; **if formatted text exceeds 4000 characters, keep first 4000 + ellipsis marker** |
| light | **≥ 16_000 and < 32_000** | **Sender profile (Channel A) only** (with i18n header). Other channels dropped |
| minimal | **< 16_000** | **Entire Priming skipped** (empty string) |

Heartbeat/cron query text is built from recent `[REFLECTION]` blocks in `activity_log` (not the full long template).

**2. System prompt body shrink** — `core/_agent_priming.py` `_fit_prompt_to_context_window`: when estimated system + user tokens plus tool schema overhead exceed **~80%** of the context window, `build_system_prompt` is rebuilt with system budget stepped **75% → 50% → 25%**. At the **≤25%** step, **Priming and human-notification blocks are cleared** before fitting. If it still does not fit, the system prompt is **hard-truncated by bytes**.

### Steps

1. **Fetch missing context explicitly**: Use `search_memory` / `read_memory_file` for org, procedures, and common knowledge (especially under `minimal` / `light`, Priming is weak)
2. **Persist state on disk**: Keep summaries in `state/current_state.md` and `shortterm/` so work can resume after a session break
3. **Talk to supervisor / admin**: If production feels tight, revisit `context_window` in `models.json` or change models

---

## Other Common Issues

### File Not Found

- **Cause**: Wrong path or file does not exist
- **Fix**: In Mode S use `Glob`; otherwise use `search_memory` or `read_memory_file` on known paths
- **Note**: `read_memory_file` accepts paths relative to the Anima directory (e.g. `knowledge/xxx.md`) plus `common_knowledge/`, `reference/`, and `common_skills/` prefixes for shared trees. The agent built-in `Read` resolves paths under different rules

### Cannot Use Inbox with read_channel

- **Cause**: `read_channel` is for Board shared channels. Inbox is not a channel
- **Fix**: Inbox messages are handled automatically by the system. Passing `inbox` or `inbox/` to `read_channel` errors

### Command Timeout

- **Cause**: Processing exceeded `timeout`
- **Fix**: Increase the `timeout` parameter for Bash runs (default: 30 seconds)
- **Note**: Set timeouts appropriately for long-running commands

### Recipient Anima Does Not Exist

- **Cause**: Wrong Anima name, or that Anima has not been created
- **Fix**: Confirm with your supervisor. Org structure: `reference/organization/structure.md`
