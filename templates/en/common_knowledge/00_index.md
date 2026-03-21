# Common Knowledge — Index & Quick Guide

Index of reference documents shared by all AnimaWorks Anima.
When you are stuck or unsure of a procedure, use this file to identify the relevant document,
then read it with `read_memory_file(path="common_knowledge/...")`.

> 💡 Detailed technical references (file specifications, model configuration, authentication setup, etc.) have been moved to `reference/`.
> Index: `reference/00_index.md`

---

## Quick Guide — When You Are Stuck

### Communication

| Problem | Reference |
|---------|-----------|
| Don't know how to send messages | `communication/messaging-guide.md` |
| Don't know how to use Board (shared channels) | `communication/board-guide.md` |
| Don't know how to give instructions or report | `communication/instruction-patterns.md` / `communication/reporting-guide.md` |
| Want to check required fields for delegation/reports/escalation | `communication/message-quality-protocol.md` |
| Message sending was limited | `communication/sending-limits.md` |
| Don't know how to notify humans | `communication/call-human-guide.md` |
| Don't know how to configure Slack bot tokens | `reference/communication/slack-bot-token-guide.md` (→ reference) |

### Organization & Hierarchy

| Problem | Reference |
|---------|-----------|
| Don't know the org structure or who to contact | `reference/organization/structure.md` (→ reference) |
| Want to check roles and responsibilities | `organization/roles.md` |
| Don't know communication rules across hierarchy | `organization/hierarchy-rules.md` |

### Tasks & Operations

| Problem | Reference |
|---------|-----------|
| Don't know how to manage tasks | `operations/task-management.md` |
| Want to use the task board (human-facing dashboard) | `operations/task-board-guide.md` |
| Don't know how to configure Heartbeat or cron | `operations/heartbeat-cron-guide.md` |
| Don't know how to run long-running tools | `operations/background-tasks.md` |
| Don't know how to register or use workspaces | `operations/workspace-guide.md` |
| Want to change project settings | `reference/operations/project-setup.md` (→ reference) |

### Tools, Models & Technical

| Problem | Reference |
|---------|-----------|
| Don't know how to use or call tools | `operations/tool-usage-overview.md` |
| Don't know how to choose or change models | `reference/operations/model-guide.md` (→ reference) |
| Want to change Mode S authentication method | `reference/operations/mode-s-auth-guide.md` (→ reference) |
| Don't know how to set up or use voice chat | `reference/operations/voice-chat-guide.md` (→ reference) |

### Understanding Yourself

| Problem | Reference |
|---------|-----------|
| Want to know what an Anima is | `anatomy/what-is-anima.md` |
| Want to understand your configuration files | `reference/anatomy/anima-anatomy.md` (→ reference) |
| Want to understand how memory works | `anatomy/memory-system.md` |

### Troubleshooting

| Problem | Reference |
|---------|-----------|
| Tools or commands don't work / getting errors | `troubleshooting/common-issues.md` |
| Task is blocked / unsure what to do | `troubleshooting/escalation-flowchart.md` |
| Gmail tool credential setup not working | `reference/troubleshooting/gmail-credential-setup.md` (→ reference) |

### Security

| Problem | Reference |
|---------|-----------|
| Concerned about external data reliability | `security/prompt-injection-awareness.md` |

### Use Cases

| Problem | Reference |
|---------|-----------|
| Want to know what AnimaWorks can do | `usecases/usecase-overview.md` |

**None of the above?** → Search with `search_memory(query="keyword", scope="common_knowledge")`

---

## Document Listing

### anatomy/ — Anima Anatomy & Components

| File | Description |
|------|-------------|
| `what-is-anima.md` | What is an Anima (concept, design philosophy, lifecycle, execution paths) |
| `anima-anatomy.md` | → Moved to `reference/anatomy/anima-anatomy.md`. Complete file reference |
| `memory-system.md` | Memory system guide (memory types, Priming, Consolidation, Forgetting, tool usage) |

### organization/ — Organization & Structure

| File | Description |
|------|-------------|
| `structure.md` | → Moved to `reference/organization/structure.md`. How org structure works |
| `roles.md` | Roles and responsibilities (top-level / mid-level / worker Anima duties) |
| `hierarchy-rules.md` | Rules across hierarchy (communication paths, supervisor tools, emergency exceptions) |

### communication/ — Communication

| File | Description |
|------|-------------|
| `messaging-guide.md` | Full guide to messaging (send_message params, thread management, one-round rule) |
| `board-guide.md` | Board (shared channels) guide (post_channel / read_channel usage, posting rules) |
| `instruction-patterns.md` | Instruction patterns (clear instructions, delegation patterns, progress checks) |
| `reporting-guide.md` | How to report and escalate (timing, format, urgent vs routine) |
| `message-quality-protocol.md` | Message quality protocol (delegation 4 fields, completion 3 fields, escalation 4 fields) |
| `sending-limits.md` | Sending limits in detail (3-layer rate limit, 30/h and 100/day caps, cascade detection) |
| `call-human-guide.md` | Human notification guide (call_human usage, receiving replies, notification channels) |
| `slack-bot-token-guide.md` | → Moved to `reference/communication/slack-bot-token-guide.md`. Slack bot token configuration |

### operations/ — Operations & Task Management

| File | Description |
|------|-------------|
| `project-setup.md` | → Moved to `reference/operations/project-setup.md`. Project configuration |
| `task-management.md` | Task management (current_state.md usage and task queue, state transitions, priorities) |
| `task-board-guide.md` | Task board (human-facing dashboard) — structure and operational guidelines |
| `heartbeat-cron-guide.md` | Scheduling and running Heartbeat and cron (how Heartbeat works, cron definitions, self-updates) |
| `tool-usage-overview.md` | Tool usage overview (S/C/D/G/A/B mode tool sets, internal/external tools, how to call them) |
| `background-tasks.md` | Background task guide (using submit, when to use it, how to get results) |
| `workspace-guide.md` | Workspace guide (concept, registration, tool usage, troubleshooting) |
| `model-guide.md` | → Moved to `reference/operations/model-guide.md`. Model selection and configuration |
| `mode-s-auth-guide.md` | → Moved to `reference/operations/mode-s-auth-guide.md`. Mode S authentication guide |
| `voice-chat-guide.md` | → Moved to `reference/operations/voice-chat-guide.md`. Voice chat guide |

### security/ — Security

| File | Description |
|------|-------------|
| `prompt-injection-awareness.md` | Prompt injection defense (trust levels, boundary tags, handling untrusted data) |

### troubleshooting/ — Troubleshooting

| File | Description |
|------|-------------|
| `common-issues.md` | Common problems and fixes (undelivered messages, rate limits, permissions, tools, context) |
| `escalation-flowchart.md` | Decision flowchart when stuck (problem types, urgency, who to escalate to) |
| `gmail-credential-setup.md` | → Moved to `reference/troubleshooting/gmail-credential-setup.md`. Gmail tool credential setup guide |

### usecases/ — Use Case Guides

| File | Description |
|------|-------------|
| `usecase-overview.md` | Use case guide overview (what AnimaWorks can do, getting started, full topic list) |
| `usecase-communication.md` | Communication automation (chat/email monitoring, escalation, scheduled notifications) |
| `usecase-development.md` | Software development support (code review, CI/CD monitoring, issue implementation) |
| `usecase-monitoring.md` | Infrastructure & service monitoring (uptime checks, resources, SSL, log analysis) |
| `usecase-secretary.md` | Secretary & admin support (scheduling, coordination, daily reports, reminders) |
| `usecase-research.md` | Research & analysis (web search, competitor analysis, market research, reports) |
| `usecase-knowledge.md` | Knowledge management & documentation (procedures, FAQ building, lessons learned) |
| `usecase-customer-support.md` | Customer support (first response, FAQ auto-reply, escalation management) |

---

## Keyword Index

| Keywords | Reference |
|----------|-----------|
| message, send_message, reply, thread, inbox | `communication/messaging-guide.md` |
| Board, channel, post_channel, read_channel | `communication/board-guide.md` |
| DM history, read_dm_history, past conversation | `communication/board-guide.md` |
| instruction, delegation, task request | `communication/instruction-patterns.md` |
| report, daily report, summary, escalation | `communication/reporting-guide.md` |
| quality protocol, required fields, verification evidence, completion criteria, delegation check | `communication/message-quality-protocol.md` |
| rate limit, sending limit, 30/hour, 100/day, one-round rule | `communication/sending-limits.md` |
| call_human, human notification, notify human | `communication/call-human-guide.md` |
| Slack, bot token, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| organization, supervisor, subordinate, peer | `reference/organization/structure.md` |
| role, responsibility, speciality, specialty | `organization/roles.md` |
| hierarchy, communication path, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, task delegation, task_tracker | `organization/hierarchy-rules.md`, `operations/task-management.md` |
| task, current_state, pending, progress, priority | `operations/task-management.md` |
| task queue, submit_tasks, update_task, TaskExec, animaworks-tool task list | `operations/task-management.md` |
| task board, dashboard, human-facing | `operations/task-board-guide.md` |
| config, status.json, SSoT, reload, settings | `reference/operations/project-setup.md` |
| Heartbeat, heartbeat, periodic check | `operations/heartbeat-cron-guide.md` |
| cron, schedule, scheduled task | `operations/heartbeat-cron-guide.md` |
| tool, animaworks-tool, MCP, skill | `operations/tool-usage-overview.md` |
| execution mode, S-mode, C-mode, D-mode, G-mode, A-mode, B-mode | `operations/tool-usage-overview.md` |
| background, submit, long-running tool | `operations/background-tasks.md` |
| workspace, working_directory, project directory | `operations/workspace-guide.md` |
| model, models.json, credential, set-model, context window | `reference/operations/model-guide.md` |
| background_model, background model, cost optimization | `reference/operations/model-guide.md` |
| Mode S, authentication, API direct, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, self, anatomy, composition, lifecycle | `anatomy/what-is-anima.md` |
| identity, injection, personality, guidelines, immutable, mutable | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, heartbeat.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| memory, episodes, knowledge, procedures, skills | `anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting | `anatomy/memory-system.md` |
| search_memory, write_memory_file, memory search | `anatomy/memory-system.md` |
| prompt injection, trust, untrusted, boundary tag | `security/prompt-injection-awareness.md` |
| error, problem, not working, permission, blocked command | `troubleshooting/common-issues.md` |
| flowchart, decision, unsure, urgent, security | `troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| tier, tiered, T1, T2, T3, T4 | `troubleshooting/common-issues.md` |
| use case, examples, what can it do | `usecases/usecase-overview.md` |

---

## How to Use

```
# Search by keyword
search_memory(query="message sending", scope="common_knowledge")

# Specify path directly
read_memory_file(path="common_knowledge/communication/messaging-guide.md")

# Read a technical reference
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# Reference this file
read_memory_file(path="common_knowledge/00_index.md")
```
