# Flowchart for When You're Stuck

A flowchart for deciding whether to **resolve an issue yourself** or **consult and report to someone else** when a problem occurs.

Refer to this document when you are unsure how to proceed. You do not need this flowchart for problems you can clearly solve on your own.

---

## Decision flowchart

When a problem occurs, work through the following steps in order.

### Step 1: Classify the problem type

Classify the problem into one of the following:

| Type | Description | Examples |
|------|-------------|----------|
| **A. Technical** | Issues with tools or system behavior | Tool errors, insufficient permissions, missing files |
| **B. Operational** | Issues with how to carry out tasks or make judgments | Unclear requirements, priority trade-offs, blockers |
| **C. Interpersonal** | Issues coordinating with other Anima | No response, conflicting instructions, unclear ownership |
| **D. Urgent** | Issues that need immediate action | Risk of data loss, security concerns |

### Step 2: Assess urgency

| Urgency | Criteria | Response |
|---------|----------|----------|
| **High** | Leaving it unaddressed risks data loss or security issues | MUST: Report to your supervisor immediately |
| **High** | Another Anima’s work is completely stopped | MUST: Report to your supervisor immediately |
| **Medium** | Your work is blocked, but you can switch to other tasks | SHOULD: Report to your supervisor within one hour |
| **Low** | Efficiency drops but you can still make progress | MAY: Report at the next Heartbeat |

**If urgency is High** → Go to Step 5 (escalate immediately).

### Step 3: Attempt self-resolution

Try to resolve the issue yourself using the steps below. Stop as soon as any step fixes the problem.

1. **Search memory**
   ```
   search_memory(query="keywords related to the problem", scope="all")
   ```
   - Check whether you have encountered the same problem before
   - Check whether `procedures/` contains remediation steps
   - To recall **recent actions** (e.g. a tool result or message from earlier in the session), use `search_memory(query="...", scope="activity_log")` in addition to `scope="all"` (which merges vector hits with activity_log BM25 via RRF)

2. **Search shared knowledge**
   ```
   search_memory(query="keywords related to the problem", scope="common_knowledge")
   ```
   - Check whether `troubleshooting/common-issues.md` covers the issue

3. **Consider other approaches**
   - Consider alternative ways to reach the goal
   - If permissions are insufficient, try another path; if a tool errors, try another tool

4. **Time limits for self-resolution**
   - Technical issues: Escalate if not resolved within 15 minutes
   - Operational issues: Escalate as soon as you are unsure (avoid the risk of a wrong judgment)
   - Interpersonal issues: Escalate after one retry

### Step 4: Choose where to escalate

| Problem type | Contact first | If that does not resolve it |
|--------------|---------------|------------------------------|
| **A. Technical** | Peer (when it matches their specialty) | Supervisor |
| **B. Operational** | Supervisor | — |
| **C. Interpersonal** | Supervisor (ask for mediation) | — |
| **D. Urgent** | Supervisor (immediately) | — |

**Decision rules:**
- **OK to ask a peer:** Same supervisor, and the issue relates to their specialty.
- **MUST report to supervisor:** Business judgment required, another area is involved, or urgency is high.
- **Do not** contact Anima in other departments directly (MUST go through your supervisor).

### Step 5: Execute the escalation

Your report MUST include:

1. **Situation:** What is happening
2. **Cause:** What you believe caused it (or “under investigation” if unknown)
3. **What you tried:** What you attempted on your own
4. **What you need:** What you want from your supervisor (decision, granting access, mediation, etc.)

**`send_message` constraints (aligned with implementation):**
- `intent` is REQUIRED: `report` or `question` only. It cannot be omitted. `intent="delegation"` is **rejected** (task delegation is only via `delegate_task`)
- Acknowledgments, thanks, and FYI must not use DM. Use the Board (`post_channel`)
- Per-run DM recipient caps are set by role / `status.json` (e.g. general/ops: 2, engineer: 5, manager: 10). Only one message per recipient. For broader broadcasts, use the Board
- DM and the Board **share the same outbound budget** (per-hour and 24-hour limits). See `communication/sending-limits.md`
- **Recipient:** Anima name, or a configured human alias (Slack / Chatwork / etc. when set in config)
- **During chat:** Reply to the human user in plain text. Use `send_message` only for other Anima (or external delivery via configured aliases)
- **Reaching a human** when `send_message` cannot (e.g. no alias): If you are a **top-level** Anima and notifications are configured, use `call_human` (below)
- For thread replies, set `reply_to` and `thread_id` to keep context
- If urgency is High and a human must act immediately, consider `call_human` (`subject`, `body`, `priority`)

**`post_channel` (Board) constraints** (use when reaching three or more people):
- Channels without extra metadata (e.g. `general`, `ops`) are open to everyone. Member-only channels allow posts only for members (ACL). If you lack access, use `manage_channel(action="info", channel="channel_name")` to inspect membership
- At most one post per channel per run. Further posts to the same channel need a cooldown (`heartbeat.channel_post_cooldown_s` in `config.json`, default 300 seconds)
- You can mention with `@name` in the body; mentioned parties receive a DM notification

**`call_human` and the human notification stack (`core/notification/` implementation):**

- **When the tool is enabled:** `human_notification.enabled` is `true` in `config.json`, and `HumanNotifier.from_config` successfully builds **at least one delivery channel** (only `channels[]` entries with `enabled: true` and a registered `type` count; `enabled: false` is skipped; unregistered `type` values are skipped after a warning log)
- **Top-level gate (`supervisor` gate):** When there is an entry for **that Anima’s name** in `config.animas` **and** `supervisor` is non-null, `HumanNotifier` is not attached and `call_human` is unavailable (subordinates escalate to their supervisor with `send_message`). **Anima not listed under `animas` do not pass this gate**, so in theory `call_human` may be available with channels alone. In operations, register every Anima under `animas` and give human notification only to those with `supervisor: null` for safety
- **Delivery:** `HumanNotifier.notify` sends to each enabled channel **in parallel** (`asyncio.gather(..., return_exceptions=True)`). Each channel returns a success string or a failure string containing `ERROR`; **exceptions in one channel are swallowed and others continue**
- **Channel types** (`human_notification.channels[].type`): `slack`, `chatwork`, `line`, `telegram`, `ntfy` (see `@register_channel` in `core/notification/channels/*.py`). You can define multiple channels in parallel
- **Parameters:** `subject` and `body` are required; `priority` is optional. Allowed values are `low` / `normal` / `high` / `urgent` (default `normal`). **Strings outside `PRIORITY_LEVELS` are normalized to `normal` inside `HumanNotifier.notify`**
- **How priority surfaces:**
  - **Slack / Chatwork / LINE / Telegram:** For `high` / `urgent`, **`[HIGH]` / `[URGENT]`** is prefixed at the start (`priority.upper()`). Not added for `low` / `normal`
  - **ntfy:** HTTP header `Priority` is set to `low=2`, `normal=3`, `high=4`, `urgent=5`. Body is the request body (max ~4096 characters); `Title` header carries the subject plus `(from AnimaName)` when needed
- **Slack** (`channels/slack.py`):
  - **Bot Token + `channel`** (`chat.postMessage`) or **Incoming Webhook**. Body text is formatted for Slack via `md_to_slack_mrkdwn`
  - **Bot mode with `anima_name` set:** The API `username` is set to the Anima name, so **`(from AnimaName)` is not added to the body** (Webhook mode adds `(from AnimaName)` to the body). With matching config and assets, `icon_url` can be set
  - **Thread reply routing** (`reply_routing.py`): Only when posting with a Bot, `anima_name` is non-empty, and the API response includes `ts`, an entry is saved to `notification_map.json`. Path: `{data_dir}/run/notification_map.json` (typically `~/.animaworks/run/`). Entries are discarded after **up to 7 days** from creation. Webhooks cannot obtain `ts`, so mapping is impossible
  - When routing, Slack thread summaries are fetched when possible; on failure, a saved summary of the notification text is used. External messages delivered to Inbox use `intent="question"`
- **Chatwork:** `room_id` must be **numeric only**. Body is converted with `md_to_chatwork` and wrapped in `[info][title]…[/title]…[/info]` format
- **LINE:** Push API. Text is truncated to a maximum of 5000 characters
- **Telegram:** `parse_mode=HTML`. Subject in `<b>…</b>`; overall length adjusted within 4096 characters (truncate after escaping)
- **Credentials:** Base `NotificationChannel._resolve_credential_with_vault` order: **config key env → `{key}__{anima_name}` (vault/shared) → plain key**. Slack Bot also has a `get_credential("slack", "notification", …)` fallback (see each `channels/*.py`)
- **Chat UI:** Streaming responses emit a **`notification_sent`** event (via `core/_anima_messaging.py`; separate path from external channels)
- **Logging:** On `call_human`, the unified activity log records **`human_notify`** (`via` is fixed in implementation as `configured_channels`), plus `tool_result`. Priming “Pending Human Notifications” aggregates **`human_notify` from the past 24 hours, up to 10 entries** (`core/memory/priming/outbound.py`)
- **Other `HumanNotifier` use:** The framework may notify humans via the **same `HumanNotifier`** for background tool completion, etc. (not only the `call_human` tool; top-level Anima restriction is the same)
- **Mode S (CLI):** You can send the same style of notification with `animaworks-tool call_human "subject" "body" [--priority …]`

**`call_human` parameters (summary):**
- `subject` and `body` are required; `priority` is optional (`low` / `normal` / `high` / `urgent`, default `normal`; invalid values are treated as `normal`)

---

## Escalation message templates

### Template 1: Blocker report

```
send_message(
    to="supervisor_name",
    content="""[Blocker report]

■ Situation
Task “Monthly report creation” is blocked.

■ Cause
I do not have read access to `/data/sales/`, where sales data is stored.

■ Already tried
- Checked `permissions.json` → `/data/sales/` not allowed
- Looked for alternate data sources → none found

■ Request
Please grant read access to `/data/sales/`.""",
    intent="report"
)
```

### Template 2: Decision request

```
send_message(
    to="supervisor_name",
    content="""[Decision request]

■ Situation
Task “Customer support flow improvement” has two plausible directions.

■ Options
Plan A: Incrementally adjust the existing flow (effort: small, risk: low, impact: medium)
Plan B: Redesign the flow end-to-end (effort: large, risk: medium, impact: high)

■ My view
I recommend Plan A. The pain points in the current flow are narrow, and incremental fixes should be enough.

■ Request
Please choose the direction.""",
    intent="question"
)
```

### Template 3: Technical question to a peer

```
send_message(
    to="peer_name",
    content="""[Technical question]

I am hitting the Slack API rate limit.

■ Situation
- Trying to send 100+ messages in one batch
- Around the 50th message I get `429 Too Many Requests`

■ Question
Do you know good ways to work around Slack API rate limits?
I am considering spacing out batch sends but am unsure what interval is appropriate.""",
    intent="question"
)
```

### Template 4: Urgent report

When urgency is High and a human must respond immediately, also use `call_human`. The tool is available only for **top-level** Anima when `human_notification` is enabled. Subordinate Anima should use `send_message` to their supervisor only.

```
send_message(
    to="supervisor_name",
    content="""[Urgent report]

■ Situation
Authentication errors from external API (XXX service) keep occurring.

■ Impact
- Task YYY is fully stopped
- Task ZZZ uses the same API and may be affected

■ Already tried
- Retried three times → all failed
- I cannot verify API key validity on my own

■ Request
Please verify the API key and assess blast radius.""",
    intent="report"
)
```

When immediate human notification is needed:

```
call_human(
    subject="[URGENT] Ongoing external API authentication errors",
    body="Authentication errors keep occurring from XXX service. Task YYY is stopped. Please verify the API key.",
    priority="urgent"
)
```

---

## Common escalation scenarios

### Scenario 1: Unclear instructions

**Situation:** Your supervisor said “create a report,” but the period, format, and delivery destination are unclear.

**Correct response:**
1. Organize what you can infer yourself
2. Ask concrete questions about what is unclear

```
send_message(
    to="supervisor_name",
    content="""About the report, please confirm:

1. Period: Should this be for the current month?
2. Format: Same Markdown format as last time?
3. Delivery: Should I save it under `knowledge/`?

If the above is fine, I will start.""",
    intent="question"
)
```

**Do not:**
- Proceed on your own interpretation without confirming
- Reply only “the instructions are unclear” (with no specific questions)
- Omit `intent` on `send_message` (`report` or `question` is required; omission causes an error)

### Scenario 2: Conflicting instructions from multiple supervisors

**Situation:** Your direct supervisor said “prioritize A,” but another Anima asked you to “do B first.”

**Correct response:**
1. Follow your direct supervisor’s instruction (MUST)
2. Report the situation to your direct supervisor

```
send_message(
    to="direct_supervisor_name",
    content="""[Priority check]

I am working on task A as you directed, but XXX asked me to prioritize task B.
I will keep prioritizing task A unless you say otherwise—is that OK?

Task B request: handle YYY (from XXX)""",
    intent="question"
)
```

### Scenario 3: Repeated errors while working

**Situation:** Sending a message via the Chatwork API failed three times in a row.

**Correct response:**
1. Record the errors
2. Escalate if three retries do not fix it
3. Move on to other tasks that are not blocked

```
# Record the errors
write_memory_file(
    path="state/current_state.md",
    content="## Blocked\n\nChatwork API repeated failures\n- 1st: 10:00 - 500 Internal Server Error\n- 2nd: 10:05 - 500 Internal Server Error\n- 3rd: 10:10 - 500 Internal Server Error\n\nReported to supervisor. Working on other tasks.",
    mode="overwrite"
)

# Report to supervisor
send_message(
    to="supervisor_name",
    content="[Blocker report] Chatwork API returned HTTP 500 three times in a row. Possible external incident. Waiting for recovery while working on other tasks.",
    intent="report"
)
```

### Scenario 4: Issue outside your scope

**Situation:** While working, you notice inconsistent data owned by another Anima.

**Correct response:**
1. Record what you found
2. Report to your direct supervisor (do not contact the other department’s Anima directly)

```
send_message(
    to="supervisor_name",
    content="""[Information share]

I noticed the following inconsistency while working. Outside my remit, but sharing.

■ Finding
`/data/reports/monthly.md` total sales and `/data/sales/summary.md` do not match.
- monthly.md: ¥1,234,567
- summary.md: ¥1,234,000

■ How I found it
Spotted while referencing data for the monthly report.

Whether to act is your call.""",
    intent="report"
)
```

---

## Checklist when you are unsure

Escalate (MUST) if any of the following applies:

- [ ] A wrong decision could cause irreversible harm
- [ ] You need an action outside your permission boundary
- [ ] Another department’s Anima would be affected
- [ ] No fix has appeared after 15+ minutes
- [ ] The same problem has happened twice or more
- [ ] Security or data safety is involved

You may (MAY) try to self-resolve if:

- [ ] You have fixed a similar issue before
- [ ] `procedures/` documents the fix
- [ ] `common_knowledge/` documents the fix
- [ ] You can finish within your permission scope
- [ ] Failure would have limited impact
