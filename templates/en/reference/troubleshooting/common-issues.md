# Common Issues and How to Resolve Them

A reference summarizing issues you may encounter during work and the steps to resolve them.
Each issue is described in the format "Symptom → Cause → Resolution steps."

If you're stuck, first read this documentation and follow the steps for the relevant item.
If the issue is not resolved here, refer to `troubleshooting/escalation-flowchart.md` and escalate appropriately.

---

## Messages Not Being Delivered

### Symptoms

- No reply to a message you sent
- The recipient says they didn't receive your message
- You executed `send_message`, but the recipient didn't respond

### Causes

1. Incorrect destination specification (Anima official name, user alias, `slack:` / `chatwork:` prefix, etc.) or a specification that doesn't match the resolution order
2. The server is down
3. The recipient's Anima is stopped or disabled, or waiting for a Provider error in the Inbox
4. An actual send error occurred, such as destination resolution, permission, or external channel delivery
5. `intent` is unspecified or invalid. For DMs, only `report` / `question` are allowed. For task delegation, use `delegate_task` (adding `intent="delegation"` to `send_message` returns a deprecation message)
6. A DM has already been sent to the same destination in the same run (only the second message to the same destination is rejected; there is no limit on the number of destinations)

### Resolution Steps

1. **Check the recipient's name and destination format**
   - Verify that the `to` parameter of `send_message` resolves to the intended recipient
   - The resolution order of the implementation (`core/messaging/outbound.py` `resolve_recipient`) is roughly as follows:
     1. **Exact match** with a known Anima name (case-sensitive) → internal
     2. **Alias** of `config.json` `external_messaging.user_aliases` (case-insensitive) → external (preferred_channel)
     3. `slack:USERID` / `chatwork:ROOMID` → external direct
     4. Bare Slack user ID (`U` + 8 or more alphanumeric characters) → Slack direct
     5. **Case-insensitive match** with a known Anima name → internal
     6. Anything else → resolution failure
   - To reliably reach an internal Anima, use the **official name** on `~/.animaworks/animas/<名前>/` or `reference/organization/structure.md`
   - How to check:
     ```
     search_memory(query="organization", scope="common_knowledge")
     ```
     Or check all Anima names in the organization via `read_memory_file(path="reference/organization/structure.md")`
   - **Note**: In chat, `send_message` cannot be used for human recipients. Reply directly with text to reach a human. To contact a human outside of chat (e.g., heartbeat), use `call_human`

2. **Check the server's operational status**
   - Since you are running, the server should be operational
   - If still concerned, report "messages are not being delivered" to your supervisor

3. **Wait for the recipient's response**
   - The recipient checks their inbox at heartbeat intervals (e.g., every 30 minutes)
   - Even if there's no immediate reply, it will be processed at the next heartbeat
   - For urgent matters, report "I need to contact them immediately" to your supervisor and request a manual startup

4. **If a send error occurs**
   - Record the error message
   - Note the situation in `state/current_state.md`
   - Report to your supervisor

### Specific Examples

```
# 名前を間違えていた場合
send_message(to="Aoi", content="...", intent="report")   # OK
send_message(to="aoi", content="...", intent="report")  # 名前が異なればエラーになる可能性あり

# DM は intent 必須（report / question のみ）。委譲は delegate_task
# 同一 run 内で同一宛先へ送れる DM は1通まで。宛先数の上限はない
send_message(
    to="aoi",
    content="了解しました。作業を開始します。",
    intent="report",           # 必須: report / question
    reply_to="msg-abc123",     # 任意: 元メッセージのID
    thread_id="thread-xyz789"  # 任意: スレッドID
)

# 確認・お礼・称賛だけのメッセージには返信しない。全体共有が必要なら post_channel（Board）を使用
```

---

## Unable to Proceed with a Task

There is no such state as "waiting until conditions are met" or "blocked" (`blocked` has been deprecated). You can only choose from three options: proceed, close, or consult.

### Symptoms

- You tried to proceed with work but lack the necessary information or permissions
- You are waiting for another Anima to complete its work
- An external service returns an error

### Causes

1. A dependent task is incomplete
2. Insufficient permissions (an operation not allowed in `permissions.json`. Even in an environment with only `permissions.md`, JSON is generated on the first `load_permissions`, and the MD is moved to `.bak`)
3. Necessary information is missing
4. An external service outage

### Resolution Steps

1. **Do not repeat the same operation**
   - Identify specifically what is missing
   - Organize "whose" "what work" is needed "by when"

2. **Determine whether you can resolve it yourself**
   - Consider whether a different approach can work around the issue
   - Search your memory to see if a similar problem occurred in the past:
     ```
     search_memory(query="error content or keywords", scope="episodes")
     search_memory(query="workaround", scope="knowledge")
     ```

3. **If you cannot resolve it, report to the requester** (see `troubleshooting/escalation-flowchart.md`). The report should include:
   - What you tried to do
   - What is missing or what you are waiting for
   - What you attempted and your recommendation
   ```
   send_message(
       to="上司の名前",
       content="【進捗報告】\nタスク: XXXの実装\n事実: YYYのAPI権限が不足\n試行: permissions.jsonを確認したが該当設定なし\n推奨: API権限の追加をお願いします",
       intent="report"
   )
   ```

4. **Decide how to handle the task**
   - If there's a chance you'll continue, you don't need to do anything (the task remains `pending`. Even if the session ends without a declaration, it automatically returns to `pending`)
   - If it's no longer needed, set it to `update_task(status="cancelled", summary="理由")`

5. **Check if there is other work you can do while waiting**
   - Persistent task queue: if tools are available, check via `list_tasks` or `Bash: animaworks-tool task list`
   - Use `list_tasks(detail=true)` to check unstarted tasks, dependencies, and reasons requiring action. Do not duplicate existing tasks or manually re-enter them
   - Start another task

---

## Memory Not Found

### Symptoms

- You can't recall something you did in the past
- A procedure document should exist but you can't find it
- Searches return no relevant results

### Causes

1. The search keywords are not appropriate
2. The search scope is too narrow
3. It hasn't been written to memory yet (first-time work)
4. The file path is incorrect

### Resolution Steps

1. **Widen the scope and search again**
   - First, search broadly with the `all` scope:
     ```
     search_memory(query="keyword to search", scope="all")
     ```
   - If there are too many results, narrow the scope:
     ```
     search_memory(query="Slack settings", scope="procedures")    # limited to procedure documents
     search_memory(query="Slack outage", scope="episodes")        # limited to past events
     search_memory(query="Slack", scope="knowledge")             # limited to learned knowledge
     ```

2. **Change the keywords and search again**
   - Try synonyms and related terms (e.g., "send," "message," "notification," "contact")
   - Try English keywords as well (e.g., "slack," "message," "send")
   - Be aware of partial matches (e.g., "Chatwork" → "chatwork," "チャットワーク")

3. **Search common knowledge**
   - If it's not in your personal memory, it may exist in common knowledge:
     ```
     search_memory(query="search keyword", scope="common_knowledge")
     ```
   - Check the table of contents of common knowledge:
     ```
     read_memory_file(path="common_knowledge/00_index.md")
     ```

4. **Check the directory directly**
   - In Mode S (Claude Agent SDK) and similar, the built-in `Glob` can list the contents under the Anima directory. In Mode A and similar, use `read_memory_file` to open known paths, or use `search_memory` to search broadly
   - If you know the file name, read it directly:
     ```
     read_memory_file(path="procedures/slack-setup.md")
     read_memory_file(path="knowledge/xxx-findings.md")
     ```

5. **If the memory does not exist**
   - It may be first-time work
   - Check whether common knowledge (`common_knowledge/`) has any related guides
   - Ask your supervisor or colleagues if they have any insights
   - After completing the work, you MUST record it as a memory (for next time)
   - Old or duplicate memories can be moved to archive/ using `archive_memory_file(path="...", reason="...")` (a move, not a deletion; `reason` is required)

### Search Scope List

| scope | Search target | Use case |
|-------|---------|------|
| `knowledge` | Learned knowledge and know-how | Response strategies, technical notes |
| `episodes` | Past action logs | Fact-checking "what was done when" |
| `procedures` | Procedure documents | Checking "how to do" procedures |
| `common_knowledge` | Knowledge shared by all Animas | Organization rules, system guides |
| `skills` | Skills and common skills (vector search) | Discovering and searching skills |
| `activity_log` | Recent action logs (tool execution results, messages, etc.) | Fact-checking recent items like "the email I just read" or "the search results from earlier" |
| `all` | All of the above (vector search + activity_log BM25 integrated via RRF) | Checking keyword existence, broad searches |

---

## Insufficient Permissions

### Symptoms

- When executing a tool, you get an error like "You don't have permission" or "Permission denied"
- You tried to read or write a file but couldn't access it
- You tried to run a command but it was rejected

### Causes

1. An operation not allowed in `permissions.json` (if only `permissions.md` is present, `load_permissions` normalizes to JSON-equivalent on read; invalid JSON may fall back to open defaults with a warning)
2. The external tool category is not enabled in the registry (`available_but_not_enabled` of `check_permissions`)
3. The file path is outside the allowed range (writing to protected files, outside `file_roots`, etc.). Additionally, **global** denials include `permissions.global.json` and framework-side patterns

### Resolution Steps

1. **Check your own permissions**
   ```
   check_permissions()
   ```
   - Returns in JSON. Check `internal_tools`, `external_tools.enabled` / `available_but_not_enabled`, and `file_access` (including read/write）・`restrictions`（ command denials)
   - Raw configuration can be checked via `read_memory_file(path="permissions.json")` (or `permissions.md` if it doesn't exist)
   - The permission description injected into the system prompt is text formatted by the runtime from the JSON

2. **Check whether the operation is permitted**
   - In principle, you can read and write within your own `anima_dir` (except protected files like `IDENTITY.md` / `SOUL.md`). Read access to your supervisor's or colleagues' `activity_log` and subordinate `state/` is reflected in `file_access` of `check_permissions` depending on the role
   - Shell commands: `commands` of `permissions.json` (follow allow/deny）. Global dangerous patterns are also blocked on the framework side)

3. **If you need additional permissions**
   - Reconsider whether the operation is truly necessary
   - Consider whether a different approach (an operation within the allowed scope) can substitute
   - If no substitute is possible, request the permission addition from your supervisor:
   ```
   send_message(
       to="上司の名前",
       content="【権限追加依頼】\n目的: XXXの作業のため\n必要な権限: /path/to/dir の読み取り\n理由: YYYの情報を参照する必要があるため",
       intent="question"
   )
   ```

4. **What you must never do**
   - Try to bypass permission checks
   - Try to execute a disallowed command through another method
   - Try to use another Anima's permissions

---

## Tools Not Usable

### Symptoms

- You called a tool but got an error like "Tool not found"
- External tools (Slack, Gmail, etc.) are unavailable

### Causes

1. The tool is not allowed in `permissions.json` (or MD-derived settings normalized on read), or gated actions are not explicitly permitted
2. The skill file is not found
3. Authentication information for the external service is not configured

### Handling Procedure

1. **Check how to use the tool in the skill**
   - Specify the path shown in the skill catalog of the system prompt with `read_memory_file` (e.g., `skills/foo/SKILL.md`, `common_skills/bar/SKILL.md`) and retrieve the full text of the procedure
   - If external tools are allowed in B-mode, they can be called with `Bash: animaworks-tool <ツール> <サブコマンド>`

2. **Check permissions**
   ```
   check_permissions()
   ```
   - `external_tools.enabled`: External tool categories listed in this Anima's tool registry (those actually passed to the session)
   - `external_tools.available_but_not_enabled`: Categories that are implemented in the framework but not in this Anima's registry. Check together with the permission, gated actions, and execution mode of `permissions.json`

3. **If not available**
   - Check with `permissions.json` whether the relevant tool/action is permitted and whether authentication information (such as `shared/credentials.json`) exists
   - If still not possible, request it from the supervisor (clearly stating "why that tool is needed")

4. **For MCP integration modes (S/C/D/G: Claude Agent SDK / Codex CLI / Cursor Agent / Gemini CLI)**
   - Built-in tools are available without a prefix (e.g., `send_message`). If not found, a process restart is required
   - For external tools, read the skill text with `read_memory_file` to check CLI usage, and execute `animaworks-tool <ツール> <サブコマンド>` via **Bash** (using the agent's Bash tool)
   - Long-running tools (image generation, local LLM, etc.) are executed asynchronously with `animaworks-tool submit`

5. **Common issues specific to D-mode (Cursor Agent)**
   - **CLI not found**: Check whether the `cursor-agent` CLI is installed on the host
   - **Authentication error**: Run `agent login` in the terminal to log in
   - **Fallback**: If not resolved, change `execution_mode` to `A`, or switch the model to LiteLLM (Mode A) for operation

6. **Common issues specific to G-mode (Gemini CLI)**
   - **CLI not found**: Check whether the `gemini` CLI is installed on the host
   - **Authentication error**: Run `gemini auth login` or set the environment variable `GEMINI_API_KEY`
   - **Fallback**: If not resolved, change `execution_mode` to `A`, or switch the model to LiteLLM (Mode A). The `gemini/` prefix may be remapped to `google/` for Google providers

7. **For A-mode (LiteLLM)**
   - For external tools, read the skill text with `read_memory_file` to check usage, and execute `animaworks-tool <ツール> <サブコマンド>` via **Bash**

8. **If the tool returns an error**
   - Record the error message accurately
   - If it is an authentication error, report it to the supervisor (setting authentication information is the administrator's responsibility)
   - If it is a temporary timeout or rate limit, wait briefly and retry (the number and interval depend on the tool implementation and server configuration)
   - If it does not improve, report the facts and what was tried to the requester

For an overview of the tool system, see `operations/tool-usage-overview.md`.

---

## Context has become too long

### Symptoms

- The session has been running for a long time
- Responses have become slower
- A notification from the system indicates "approaching the context limit"

### Causes

- The context window has been consumed by long-running work or numerous tool calls
- Large amounts of file content were loaded

### Handling Procedure

1. **Save the work status to short-term memory** (MUST)
   - Write the current work status to `shortterm/` (for chat sessions, use `shortterm/chat/`):
   ```
   write_memory_file(
       path="shortterm/chat/session_state.md",
       content="## 作業状態\n\n### 実行中のタスク\n- XXXの実装（50%完了）\n\n### 次のステップ\n1. YYYを完了する\n2. ZZZをテストする\n\n### 重要な中間結果\n- AAAの調査結果: BBB\n- CCCの設定値: DDD",
       mode="overwrite"
   )
   ```
   - For heartbeat sessions, use `shortterm/heartbeat/session_state.md`

2. **Update `state/current_state.md`** (MUST)
   ```
   write_memory_file(
       path="state/current_state.md",
       content="## 現在のタスク\n\nXXXの実装\n\n### 進捗\n- 50%完了\n- 次回はYYYから再開\n\n### メモ\n- 重要な発見事項をここに記載",
       mode="overwrite"
   )
   ```

3. **Save important insights to persistent memory** (SHOULD)
   - Save insights gained during work to `knowledge/`:
   ```
   write_memory_file(
       path="knowledge/xxx-findings.md",
       content="# XXXに関する知見\n\n## 発見事項\n...",
       mode="overwrite"
   )
   ```

4. **Wait for the session to continue**
   - The system automatically starts a new session
   - In the new session, the contents of `shortterm/chat/` (or `shortterm/heartbeat/`) are included in the context
   - Re-read `state/current_state.md` and resume the work

### Preventive Measures

- For large files, do not read the entire file; search only the necessary parts
- For long tasks, update `state/current_state.md` regularly
- Write intermediate results to memory frequently

---

## An error was returned when sending a message

### Symptoms and Causes

- Actual delivery errors are returned, such as `RecipientResolutionError`, `DeliveryFailed` for external channels, permission or company boundary errors
- When trying to send a second message to the same destination within the same run using `send_message`, it is rejected (duplicate prevention). There is no limit on the number of destinations
- When trying to post a second time to the same channel within the same run using `post_channel`, it is rejected. There is no posting cooldown between runs
- There is no sending budget per time or day, nor sending rejection based on conversation depth. Depth between internal Anima instances may be recorded in diagnostic logs

### Handling Procedure

1. **Check the error content**: Check the resolution target of `to`, the intent (`report` / `question`), channel ACL, company boundaries, and the external API response
2. **Check for duplicate sending**: Additional DMs cannot be sent to destinations already sent to within the same run. There is no need to wait based on the number of destinations
3. **Investigate delivery failures**: For external channels such as Slack / Chatwork, check the returned delivery error and the connection configuration
4. **For Inbox Provider errors**: Unread messages are reprocessed after the recovery time specified by `rate_guard`

See `communication/sending-limits.md` for details.

---

## A command was blocked

### Symptoms

- When trying to execute a command, an error such as "PermissionDenied" or "Command blocked" was returned
- Only specific commands cannot be executed

### Causes

1. Commands that match the global denial patterns of the framework / `permissions.global.json` (e.g., `rm -rf /`, etc.)
2. Commands listed in `commands.deny` of `permissions.json`

### Handling Procedure

1. **Check your own permissions**
   ```
   check_permissions()
   ```
   - Denied commands are listed in `restrictions`. Also check the configuration directly with `read_memory_file(path="permissions.json")` (in legacy environments, use `permissions.md`)

2. **Consider alternatives**
   - Consider whether an operation equivalent to the blocked command can be achieved with permitted tools
   - Example: If `rm -rf` is blocked, deleting individual files may still be permitted

3. **If permission changes are needed**
   - Request the unblocking from the supervisor
   - When requesting, clearly state "why that command is needed"

---

## The prompt was shortened

### Symptoms

- The system prompt is thinner than usual, and Priming (automatic recall) is nearly empty
- After a long conversation or a large user message, there is behavior as if the prompt was rebuilt before responding

### Causes

There are two main layers.

**1. Priming (automatic recall) tiers** — The `resolve_prompt_tier(context_window)` of `core/prompt/builder.py` determines the tier from the estimated context window. The resolution order for the window is `core/prompt/context.py` `resolve_context_window`: **`~/.animaworks/models.json` (SSoT)** → deprecated `config.json` `model_context_windows` → in-code fallback such as `MODEL_CONTEXT_WINDOWS` → default 128k.

| Tier | Condition (`context_window`) | Priming handling (`core/agent/priming.py`) |
|--------|--------------------------|---------------------------------------------|
| full | **≥ 128_000** | Format and include the normal retrieval of compact within the range of `priming.max_tokens` |
| standard | **≥ 32_000 and < 128_000** | Same compact path, but limit the retrieval budget to a maximum of 1000 tokens |
| light | **≥ 16_000 and < 32_000** | Retrieve the basic context of compact, and suppress related knowledge and episode searches |
| minimal | **< 16_000** | Maintain the basic context of compact, and suppress related knowledge and episode searches |

The query text for heartbeat / cron is the text collected from the recent `[REFLECTION]` in the activity_log (not the full long template).

**2. Contraction of the system prompt itself** — `core/agent/priming.py` `_fit_prompt_to_context_window`: When the estimated tokens of system + user plus the tool schema overhead exceed **approximately 80% of the context window**, `build_system_prompt` is rebuilt by gradually reducing the **system budget from 75% → 50% → 25%**. At the **stage at or below 25%**, the **Priming block and the human notification block are emptied** before applying. If it still does not fit, the system prompt is **hard-truncated at the byte level**.

### Handling Procedure

1. **Explicitly retrieve missing context**: Read organization, procedures, and shared knowledge with `search_memory` / `read_memory_file` (especially in `minimal` / `light`, where Priming is weak)
2. **Leave work status on disk**: Write summaries to `state/current_state.md` or `shortterm/` so that work can be resumed even if the session is interrupted
3. **Consult the supervisor or administrator**: If it is too tight in actual operation, consider `context_window` of `models.json` or a model change

---

## Other common problems

### File not found

- **Cause**: Incorrect path specification, file does not exist
- **Handling**: In Mode S, use `Glob`; otherwise, use `search_memory` or `read_memory_file` for known paths
- **Note**: `read_memory_file` can read shared directories with the `common_knowledge/`, `reference/`, and `common_skills/` prefixes in addition to Anima directory-relative paths (e.g., `knowledge/xxx.md`). For `Read` (agent built-in), the path is determined by different rules

### Cannot specify inbox with read_channel

- **Cause**: `read_channel` is for shared channels on the Board. The inbox (message box) is not a channel
- **Handling**: Messages in the inbox are processed automatically by the system. Specifying `inbox` or `inbox/` in `read_channel` will result in an error

### Command times out

- **Cause**: Processing time exceeded `timeout`
- **Handling**: Increase the `timeout` parameter when running Bash (default: 30 seconds)
- **Note**: Set an appropriate timeout value for long-running commands

### The other party's Anima does not exist

- **Cause**: Incorrect Anima name, or that Anima has not been created yet
- **Handling**: Check with the supervisor. See `reference/organization/structure.md` for the organization structure

### Frequent SEGVs after rebuilding venv

- **Cause**: When major versions of ChromaDB or PyTorch are upgraded, compatibility with existing vectordb (HNSW segments) is lost, causing SEGV (Segmentation Fault) during reads. The Rust bindings of ChromaDB 1.5.x have a known issue where they cause SEGV instead of a Python exception for corrupted indexes (chromadb #6852, #6949, #6979)
- **Handling**: After rebuilding venv, always fully rebuild vectordb as well
  1. Stop the server
  2. Delete `~/.animaworks/vectordb/` and each `~/.animaworks/animas/*/vectordb/` (or back them up and then delete)
  3. Delete `~/.animaworks/index_meta.json`
  4. Start the server (it will be automatically re-indexed at startup)
- **Prevention**: When updating packages, check version changes in chromadb, torch, and sentence-transformers, and rebuild vectordb if there are changes
