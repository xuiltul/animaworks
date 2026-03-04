# Common Knowledge Index & Keyword Reference

Index of reference documents shared by all AnimaWorks Anima.

When you are stuck or unsure of a procedure, read this file first to identify the relevant document,
then use `read_memory_file(path="common_knowledge/...")` to look up the details.

---

## When You Are Stuck, Start Here

Follow this flow to find the right document:

1. **Don't know how to send messages?**
   → Read `communication/messaging-guide.md`

1.5. **Don't know how to use Board (shared channels)?**
   → Read `communication/board-guide.md`

2. **Don't know how to give instructions or report?**
   → Read `communication/instruction-patterns.md` or `communication/reporting-guide.md`

3. **Don't know the org structure or who to contact?**
   → Read `organization/structure.md`

4. **Don't know how to use tools or call them?**
   → Read `operations/tool-usage-overview.md`

4.5. **Tools or commands don't work / getting errors?**
   → Read `troubleshooting/common-issues.md`

5. **Task is blocked / unsure what to do?**
   → Read `troubleshooting/escalation-flowchart.md`

6. **Don't know how to configure Heartbeat or cron?**
   → Read `operations/heartbeat-cron-guide.md`

6.5. **Don't know how to run long-running tools?**
   → Read `operations/background-tasks.md`

7. **Don't know how to manage tasks?**
   → Read `operations/task-management.md`

7.5. **Don't know how to set up or use voice chat?**
   → Read `operations/voice-chat-guide.md`

7.7. **Don't know how to choose or change models / want to understand execution modes?**
   → Read `operations/model-guide.md`

8. **Message sending was limited / want to know about sending limits?**
   → Read `communication/sending-limits.md`

9. **Concerned about external data reliability / prompt injection?**
   → Read `security/prompt-injection-awareness.md`

10. **None of the above applies**
   → Search with `search_memory(query="keyword", scope="common_knowledge")`

---

## Document Listing

### organization/ — Organization & Structure

| File | Description |
|------|-------------|
| `organization/structure.md` | How org structure works (hierarchy via supervisor, supervisor/subordinate/peer determination) |
| `organization/roles.md` | Roles and responsibilities (top-level / mid-level / worker Anima duties, meaning of speciality) |
| `organization/hierarchy-rules.md` | Rules across the hierarchy (communication paths, direct vs other departments, emergency exceptions) |

### communication/ — Communication

| File | Description |
|------|-------------|
| `communication/messaging-guide.md` | Full guide to sending and receiving messages (send_message params, thread management, rate limits, one-round rule) |
| `communication/board-guide.md` | Board (shared channels) guide (when to use post_channel / read_channel / read_dm_history, posting rules) |
| `communication/instruction-patterns.md` | Instruction patterns (how to write clear instructions, delegation patterns, progress checks) |
| `communication/reporting-guide.md` | How to report and escalate (timing, format, urgent vs routine) |
| `communication/sending-limits.md` | Sending limits in detail (3-layer rate limit, 30/h and 100/day caps, cascade detection, how to handle) |

### operations/ — Operations & Task Management

| File | Description |
|------|-------------|
| `operations/project-setup.md` | Project configuration (config.json structure, adding Anima, model settings, permissions) |
| `operations/task-management.md` | Task management (using current_task.md / pending.md, state transitions, priorities) |
| `operations/heartbeat-cron-guide.md` | Scheduling and running Heartbeat and cron (how Heartbeat works, cron task definitions, self-updates) |
| `operations/tool-usage-overview.md` | Tool usage overview (S/A/B mode tool sets, internal/external/supervisor tools, how to call them) |
| `operations/background-tasks.md` | Background task guide (using submit, when to use it, how to get results) |
| `operations/voice-chat-guide.md` | Voice chat guide (STT/TTS setup, WebSocket protocol, per-Anima voice settings, troubleshooting) |
| `operations/model-guide.md` | Model selection and configuration (execution modes, supported models, models.json, credential setup, context windows) |

### security/ — Security

| File | Description |
|------|-------------|
| `security/prompt-injection-awareness.md` | Prompt injection defense (trust levels, boundary tags, handling untrusted data) |

### troubleshooting/ — Troubleshooting

| File | Description |
|------|-------------|
| `troubleshooting/common-issues.md` | Common problems and fixes (undelivered messages, rate limits, blocks, memory search, permissions, tools, context) |
| `troubleshooting/escalation-flowchart.md` | Decision flowchart for when you're stuck (problem types, urgency, who to escalate to, templates) |
| `troubleshooting/gmail-credential-setup.md` | Gmail tool credential setup guide (token.json placement, pickle conversion, client_id mismatch) |

---

## Keyword Index

Find the right document from relevant keywords.

| Keywords | Reference |
|----------|-----------|
| message, send, reply, thread, inbox | `communication/messaging-guide.md` |
| send_message, reply_to, thread_id | `communication/messaging-guide.md` |
| Board, channel, shared, general, ops | `communication/board-guide.md` |
| post_channel, read_channel, read_dm_history | `communication/board-guide.md` |
| DM history, conversation, past chat | `communication/board-guide.md` |
| instruction, delegation, task request | `communication/instruction-patterns.md` |
| report, daily report, summary, completion report | `communication/reporting-guide.md` |
| escalation, consultation, mediation | `communication/reporting-guide.md`, `troubleshooting/escalation-flowchart.md` |
| organization, supervisor, subordinate, peer | `organization/structure.md` |
| role, responsibility, speciality, specialty | `organization/roles.md` |
| hierarchy, rules, permissions, communication path | `organization/hierarchy-rules.md` |
| config, hot reload, apply config | `operations/project-setup.md` |
| model, models, execution mode, model change, models.json, provider | `operations/model-guide.md` |
| model selection, context window, credential, set-model, models list, models info | `operations/model-guide.md` |
| anima info, anima list, anima set-model | `operations/model-guide.md` |
| add Anima, template, identity | `operations/project-setup.md` |
| task, progress, block, priority | `operations/task-management.md` |
| current_task, pending, state management | `operations/task-management.md` |
| Heartbeat, heartbeat, periodic check | `operations/heartbeat-cron-guide.md` |
| cron, schedule, scheduled task | `operations/heartbeat-cron-guide.md` |
| problem, error, stuck, not working | `troubleshooting/common-issues.md` |
| Gmail, gmail_unread, token.json, OAuth, credential | `troubleshooting/gmail-credential-setup.md` |
| permission, permission denied, access denied | `troubleshooting/common-issues.md` |
| tool, discover_tools, not found | `troubleshooting/common-issues.md` |
| memory, search_memory, search, not found | `troubleshooting/common-issues.md` |
| flowchart, decision, unsure, what to do | `troubleshooting/escalation-flowchart.md` |
| urgent, critical, security | `troubleshooting/escalation-flowchart.md` |
| context, limit, session continuation | `troubleshooting/common-issues.md` |
| background, submit, long-running, block | `operations/background-tasks.md` |
| animaworks-tool, external tool, Bash, CLI | `operations/tool-usage-overview.md`, `operations/background-tasks.md` |
| MCP, mcp__aw__, S-mode, tool invocation | `operations/tool-usage-overview.md` |
| skill, skill tool, procedure fetch, procedures | `operations/tool-usage-overview.md` |
| execution mode, S-mode, A-mode, B-mode | `operations/tool-usage-overview.md` |
| rate limit, sending limit, 30/hour, 100/day, outbound limit | `communication/sending-limits.md` |
| one-round rule, round-trip, loop, cascade | `communication/messaging-guide.md`, `communication/sending-limits.md` |
| prompt injection, trust, untrusted, boundary tag | `security/prompt-injection-awareness.md` |
| delegate_task, task delegation, task_tracker | `operations/task-management.md`, `organization/hierarchy-rules.md` |
| add_task, task queue, TaskQueue | `operations/task-management.md` |
| pending, TaskExec, 3-path, execution path | `operations/task-management.md`, `operations/heartbeat-cron-guide.md` |
| org_dashboard, ping_subordinate, supervisor tools | `organization/hierarchy-rules.md` |
| status.json, SSoT, model change, set-model, reload | `operations/project-setup.md` |
| blocked command, disallowed command, blocked | `operations/project-setup.md`, `troubleshooting/common-issues.md` |
| tier, tiered, prompt shortening, T1, T2, T3, T4 | `troubleshooting/common-issues.md` |
| voice, microphone, STT, TTS, voice chat | `operations/voice-chat-guide.md` |
| VOICEVOX, ElevenLabs, Style-BERT-VITS2, SBV2 | `operations/voice-chat-guide.md` |
| voice_id, voice, speaker, voice settings | `operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `operations/voice-chat-guide.md` |

---

## How to Use

### Finding documents via search

```
search_memory(query="message sending", scope="common_knowledge")
```

Use the file paths from the results and read them with `read_memory_file`.

### Specifying the path directly

```
read_memory_file(path="common_knowledge/troubleshooting/common-issues.md")
```

### Referencing this file

```
read_memory_file(path="common_knowledge/00_index.md")
```
