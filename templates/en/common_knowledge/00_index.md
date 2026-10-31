# Common Knowledge — Table of Contents & Quick Guide

Table of contents for the reference documentation shared by all Anima in AnimaWorks.
When you are stuck or unsure of a procedure, use this file to locate the relevant document, and refer to `read_memory_file(path="common_knowledge/...")` for details.

> 💡 Detailed technical references (configuration file specifications, model configuration, authentication configuration, etc.) have been moved to `reference/`.
> Table of contents: `reference/00_index.md`

---

## ⭐ Start Here

If you are new to AnimaWorks or want to organize the big picture, read the following single file first.
The key points of Heartbeat / Cron / team design / memory / cost optimization are consolidated into one page.

| File | Content |
|---------|------|
| **`anatomy/essentials.md`** | **AnimaWorks Essential Guide** — Overview of the big picture, 5 execution paths, Heartbeat vs Cron, team design, how to flow tasks, memory system, and cost optimization in one page |

After reading, follow the table of contents below for details on each topic.

---

## Quick Guide for When You Are Stuck

### Communication

| Issue | Reference |
|---------|--------|
| I don't know how to send a message | `reference/communication/messaging-guide.md` |
| I don't know how to use Board (shared channel) | `communication/board-guide.md` |
| I don't know how to give instructions or report | `reference/communication/instruction-patterns.md` / `reference/communication/reporting-guide.md` |
| I want to check the required items for delegation, completion reports, and escalation | `communication/message-quality-protocol.md` |
| I want to check the behavior of Inbox startup and message sending | `communication/sending-limits.md` |
| I don't know how to notify humans | `communication/call-human-guide.md` |
| I don't know how to configure the Slack bot token | `reference/communication/slack-bot-token-guide.md` ※Technical reference |

### Organization & Hierarchy

| Issue | Reference |
|---------|--------|
| I don't know the organizational structure or who to contact | `reference/organization/structure.md` ※ Technical reference |
| I want to check roles and areas of responsibility | `reference/organization/roles.md` |
| I don't know the communication rules between hierarchy levels | `organization/hierarchy-rules.md` |

### Tasks and Operations

| Issue | Reference |
|---------|--------|
| I don't know how to manage tasks | `reference/operations/task-management.md` |
| I want to use the task board (human-facing dashboard) | `operations/task-board-guide.md` |
| I don't know how to configure heartbeats or cron | `reference/operations/heartbeat-cron-guide.md` |
| I want to add confirmation rules before sending, posting, or writing to memory | `operations/action-rules-guide.md` |
| I don't know how to run long-running tools | `operations/background-tasks.md` |
| I don't know how to register or use workspaces | `operations/workspace-guide.md` |
| Where to put worktrees, temporary files, and caches | `operations/scratch-and-worktrees.md` |
| I want to learn how to create a new skill and check its metadata | `common_skills/skill-creator/SKILL.md` |
| I want to change project settings | `reference/operations/project-setup.md` ※Technical reference |

### Tools, Models & Technology

| Issue | Reference |
|---------|--------|
| I don't know how to use or call tools | `reference/operations/tool-usage-overview.md` |
| I don't know how to choose or change models | `reference/operations/model-guide.md` ※ Technical reference |
| I want to change the Mode S authentication method | `reference/operations/mode-s-auth-guide.md` ※ Technical reference |
| I don't know how to configure or use voice chat | `reference/operations/voice-chat-guide.md` ※ Technical reference |

### Understanding Yourself

| Issue | Reference |
|---------|--------|
| I want to know what an Anima is | `anatomy/what-is-anima.md` |
| I want to know the role of my configuration file | `reference/anatomy/anima-anatomy.md` ※ Technical reference |
| I want to know the mechanism and types of memory | `reference/anatomy/memory-system.md` |

### Troubleshooting

| Issue | Reference |
|---------|--------|
| Tools or commands don't work / errors occur | `reference/troubleshooting/common-issues.md` |
| A task is blocked / I'm unsure how to decide | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail tool authentication configuration isn't working | `reference/troubleshooting/gmail-credential-setup.md` ※ Technical reference |

### Security

| Issue | Reference |
|---------|--------|
| I'm concerned about the reliability of external data | `security/prompt-injection-awareness.md` |

### Use Cases

| Issue | Reference |
|---------|--------|
| I want to know what can be done with AnimaWorks | `reference/usecases/usecase-overview.md` |

**If not covered above** → Search using `search_memory(query="キーワード", scope="common_knowledge")`

---

## Document List

### anatomy/ — Anima Components

| File | Overview |
|---------|------|
| ⭐ `essentials.md` | **Essential Guide** — Overview of the entire AnimaWorks picture in one page (execution paths, Heartbeat vs Cron, team design, memory, cost optimization) |
| `what-is-anima.md` | What is an Anima (concept, design philosophy, lifecycle, execution paths) |
| `anima-anatomy.md` | → Moved to `reference/anatomy/anima-anatomy.md`. Complete configuration file guide |
| `memory-system.md` | Memory system guide (memory types, Priming, Consolidation, Forgetting, tool selection) |

### organization/ — Organization & Structure

| File | Overview |
|---------|------|
| `structure.md` | → Moved to `reference/organization/structure.md`. How the organizational structure works |
| `roles.md` | Roles and areas of responsibility (top-level / middle management / execution Anima responsibilities) |
| `hierarchy-rules.md` | Rules between hierarchy levels (communication paths, supervisor tool, exceptions in emergencies) |

### communication/ — Communication

| File | Description |
|---------|------|
| `messaging-guide.md` | Guide for sending and receiving messages (send_message parameters, thread management, reply policy) |
| `board-guide.md` | Board (shared channel) guide (how to use post_channel / read_channel, posting rules) |
| `instruction-patterns.md` | Patterns for giving instructions (how to write clear instructions, delegation patterns, progress checks) |
| `reporting-guide.md` | Reporting and escalation methods (report timing, format, urgent vs. periodic) |
| `message-quality-protocol.md` | Message quality protocol (mandatory checks: 4 items for delegation, 3 items for completion reports, 4 items for escalation) |
| `sending-limits.md` | Inbox file wake, message aggregation, duplicate prevention within runs, loop avoidance |
| `call-human-guide.md` | Guide for notifying humans (how to use call_human, receiving replies, notification channel configuration) |
| `slack-bot-token-guide.md` | → Moved to `reference/communication/slack-bot-token-guide.md`. Slack bot token configuration guide |

### operations/ — Operations and Task Management

| File | Overview |
|---------|------|
| `project-setup.md` | → Moved to `reference/operations/project-setup.md`. How to configure projects |
| `task-management.md` | Task management (how to use current_state.md, task queues, status transitions, and priorities) |
| `task-board-guide.md` | How the task board (human-facing dashboard) works and how to operate it |
| `heartbeat-cron-guide.md` | Configuring and operating scheduled execution (how heartbeats work, defining cron tasks, self-updates) |
| `action-rules-guide.md` | Action rules (`[ACTION-RULE]`, `trigger_tools`, pre-send confirmation, required `read_memory_file`) |
| `tool-usage-overview.md` | Overview of tool usage (tool systems by S/C/D/G/A/B mode, internal/external tools, how to invoke them) |
| `background-tasks.md` | Background task execution guide (how to use submit, decision criteria, how to receive results) |
| `workspace-guide.md` | Workspace guide (concepts, registration, using workspaces with tools, troubleshooting) |
| `scratch-and-worktrees.md` | Where to put work products (scratch, shared worktrees, caches, automatic deletion) |
| `model-guide.md` | → Moved to `reference/operations/model-guide.md`. Model selection and configuration guide |
| `mode-s-auth-guide.md` | → Moved to `reference/operations/mode-s-auth-guide.md`. Mode S authentication mode configuration guide |
| `voice-chat-guide.md` | → Moved to `reference/operations/voice-chat-guide.md`. Voice chat guide |

### security/ — Security

| File | Overview |
|---------|------|
| `prompt-injection-awareness.md` | Prompt injection defense guide (trust levels, boundary tags, rules for handling untrusted data) |

### troubleshooting/ — Troubleshooting

| File | Overview |
|---------|------|
| `common-issues.md` | Common problems and solutions (undelivered messages, send restrictions, permissions, tools, context) |
| `escalation-flowchart.md` | Decision flowchart for when you are stuck (problem classification, urgency assessment, escalation destination) |
| `gmail-credential-setup.md` | → Moved to `reference/troubleshooting/gmail-credential-setup.md`. Gmail Tool authentication configuration guide |

### usecases/ — Use Case Guides

| File | Overview |
|---------|------|
| `usecase-overview.md` | Use case guide overview (what you can do with AnimaWorks, how to get started, full theme list) |
| `usecase-communication.md` | Communication automation (chat, email monitoring, escalation, regular contact) |
| `usecase-development.md` | Software development support (code review, CI/CD monitoring, Issue implementation, bug investigation) |
| `usecase-monitoring.md` | Infrastructure and service monitoring (liveness monitoring, resource monitoring, SSL certificates, log analysis) |
| `usecase-secretary.md` | Secretary and administrative support (schedule management, contact coordination, daily report creation, reminders) |
| `usecase-research.md` | Research and analysis (web search, competitive analysis, market research, report creation) |
| `usecase-knowledge.md` | Knowledge management and documentation (procedure manual creation, FAQ building, accumulating lessons learned) |
| `usecase-customer-support.md` | Customer support (first response, FAQ auto-replies, escalation management) |

## Keyword Index

| Keyword | Reference |
|-----------|--------|
| fundamentals, introduction, big picture, essentials, getting started, overview | `anatomy/essentials.md` |
| message, send_message, send, reply, thread, inbox | `reference/communication/messaging-guide.md` |
| Board, channel, post_channel, read_channel | `communication/board-guide.md` |
| DM history, read_dm_history, past conversations | `communication/board-guide.md` |
| instructions, delegation, task requests, delegation | `reference/communication/instruction-patterns.md` |
| reports, daily reports, summaries, completion reports, escalation | `reference/communication/reporting-guide.md` |
| quality protocol, required items, validation evidence, completion criteria, delegation check | `communication/message-quality-protocol.md` |
| Inbox wake, message aggregation, duplicate send prevention, replying to acknowledgements, conversation loop | `communication/sending-limits.md` |
| call_human, human notifications, contact a human, notification channel | `communication/call-human-guide.md` |
| Slack, bot token, SLACK_BOT_TOKEN, not_in_channel | `reference/communication/slack-bot-token-guide.md` |
| organization, supervisor, supervisor, subordinate, colleagues | `reference/organization/structure.md` |
| roles, responsibilities, speciality, expertise | `reference/organization/roles.md` |
| hierarchy, communication paths, org_dashboard, ping_subordinate | `organization/hierarchy-rules.md` |
| delegate_task, task delegation, task_tracker | `organization/hierarchy-rules.md`, `reference/operations/task-management.md` |
| task, current_state, pending, progress, priority | `reference/operations/task-management.md` |
| task queue, submit_tasks, update_task, TaskExec, animaworks-tool task list | `reference/operations/task-management.md` |
| task board, dashboard, human-facing | `operations/task-board-guide.md` |
| configuration, config, status.json, SSoT, reload | `reference/operations/project-setup.md` |
| heartbeat, heartbeat, periodic checks | `reference/operations/heartbeat-cron-guide.md` |
| cron, schedule, scheduled tasks | `reference/operations/heartbeat-cron-guide.md` |
| tools, animaworks-tool, MCP, skill | `reference/operations/tool-usage-overview.md` |
| execution modes, S-mode, C-mode, D-mode, G-mode, A-mode, B-mode | `reference/operations/tool-usage-overview.md` |
| background, submit, long-running tools | `operations/background-tasks.md` |
| workspace, workspace, working directory, working_directory | `operations/workspace-guide.md` |
| scratch, worktree, temporary files, cache, ANIMAWORKS_SCRATCH_DIR | `operations/scratch-and-worktrees.md` |
| model, models.json, credential, set-model, context window | `reference/operations/model-guide.md` |
| background_model, background model, cost optimization | `reference/operations/model-guide.md` |
| Mode S, authentication, direct API, Bedrock, Vertex AI, Max plan | `reference/operations/mode-s-auth-guide.md` |
| audio, voice, STT, TTS, VOICEVOX, ElevenLabs | `reference/operations/voice-chat-guide.md` |
| WebSocket, /ws/voice, barge-in, VAD, PTT | `reference/operations/voice-chat-guide.md` |
| Anima, self, configuration, design, lifecycle | `anatomy/what-is-anima.md` |
| IDENTITY.md, SOUL.md, AGENTS.md, personality, profile, behavioral guidelines, immutable, variable | `reference/anatomy/anima-anatomy.md` |
| permissions.json, bootstrap, HEARTBEAT.md, cron.md | `reference/anatomy/anima-anatomy.md` |
| memory, memory, episodes, knowledge, procedures | `reference/anatomy/memory-system.md` |
| Priming, RAG, Consolidation, Forgetting, forgetting | `reference/anatomy/memory-system.md` |
| consolidation, 2-phase, multipass, error traces | `reference/anatomy/memory-system.md` |
| search_memory, write_memory_file, memory search | `reference/anatomy/memory-system.md` |
| skills, skill search, common_skills, search_memory scope="skills" | `reference/anatomy/memory-system.md`, `reference/operations/tool-usage-overview.md` |
| activity_log, BM25, RRF, recent log search | `reference/anatomy/memory-system.md`, `reference/troubleshooting/common-issues.md` |
| prompt injection, trust, untrusted, boundary tags | `security/prompt-injection-awareness.md` |
| errors, problems, not working, permissions, blocked commands | `reference/troubleshooting/common-issues.md` |
| flowcharts, decisions, uncertainty, emergencies, security | `reference/troubleshooting/escalation-flowchart.md` |
| Gmail, token.json, OAuth, pickle | `reference/troubleshooting/gmail-credential-setup.md` |
| tiers, tiered, T1, T2, T3, T4 | `reference/troubleshooting/common-issues.md` |
| use cases, examples, what you can do | `reference/usecases/usecase-overview.md` |

---

## How to Use

```
# キーワードで検索
search_memory(query="メッセージ 送信", scope="common_knowledge")

# パスを直接指定
read_memory_file(path="reference/communication/messaging-guide.md")

# 技術リファレンスを参照
read_memory_file(path="reference/anatomy/anima-anatomy.md")

# このファイル自体を参照
read_memory_file(path="common_knowledge/00_index.md")
```
