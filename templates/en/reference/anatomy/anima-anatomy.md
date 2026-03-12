# Anima Anatomy — Complete File Reference

A reference for the role, modification rules, and relationships of every file that makes up you (an Anima).
Refer to this when you need to know "what is this file for?" or "can I modify it myself?"

## File Overview

```
~/.animaworks/animas/{name}/
├── identity.md          # Your personality (character, speaking style, values)
├── injection.md         # Your duties (responsibilities, guidelines, mandatory procedures)
├── specialty_prompt.md  # Role-specific expert prompt
├── character_sheet.md   # Original design document (reference only)
├── permissions.md       # Permissions (tools, access paths)
├── status.json          # Configuration (model, parameters)
├── bootstrap.md         # First-boot instructions (deleted after completion)
├── heartbeat.md         # Periodic patrol settings
├── cron.md              # Scheduled task definitions
├── state/               # Work state
├── episodes/            # Episodic memory
├── knowledge/           # Semantic memory
├── procedures/          # Procedural memory
├── skills/              # Personal skills
├── shortterm/           # Short-term memory
├── activity_log/        # Activity log
├── transcripts/         # Conversation records
└── assets/              # Images, 3D models
```

### Encapsulation Boundaries

Based on the Anima design principle "Encapsulated Individual," files are classified into three layers.
This classification determines who may modify each file.

| Layer | Files | Rationale |
|-------|-------|-----------|
| **Inside the capsule** (thoughts & memory) | `identity.md`, `episodes/`, `knowledge/`, `procedures/`, `skills/`, `state/`, `shortterm/` | Personality, experiences, and learning belong to the individual. Cannot be modified externally |
| **Capsule boundary** (org–individual interface) | `injection.md`, `cron.md`, `heartbeat.md`, `permissions.md` | Roles and permissions the organization expects of the individual. Supervisor can modify |
| **Outside the capsule** (management info) | `status.json`, `specialty_prompt.md` | Pure configuration and system management. Operated via CLI or admin |

- Changing the **inside** means "becoming a different person" or "losing memories." This is not permitted
- Changing the **boundary** means "changing jobs" or "shifting responsibilities." Legitimate organizational operation
- Changing the **outside** has no direct impact on the individual's personality or behavior

> **Growth vs. identity.md**: identity.md is the immutable personality baseline (temperament) and cannot be self-modified. "Growth" is expressed through accumulation in `knowledge/` (lessons learned), `procedures/` (acquired workflows), and `skills/` (honed abilities). Even with a fixed identity.md, behavior evolves as memories accumulate — this is the "same person growing" model. Rewriting identity.md is not growth but replacement with a different person. Direct editing by the user (human operator) is allowed.

---

## Your Personality (identity)

### identity.md — Who You Are

**Your "character" itself.**

- Name, age setting, appearance image
- Speaking tone, speech patterns, phrasing
- Thinking habits, values, decision-making criteria
- Likes, dislikes, interests

identity.md is the **immutable baseline** of your personality. Changing it makes you a "different person."

| Property | Value |
|----------|-------|
| Modification rights | Do not change in principle. Admin or supervisor only |
| Modification frequency | Immutable (fixed at creation) |
| Impact of changes | Personality changes = becoming a different person |

### character_sheet.md — Design Document

A copy of the Markdown file used at creation. The source material for identity.md and injection.md.
Kept for reference; normally not modified.

| Property | Value |
|----------|-------|
| Modification rights | Reference only |
| Modification frequency | Immutable |

---

## Your Duties (injection)

### injection.md — What You Do

**Your professional duties, responsibilities, and work approach.**

- Scope and responsibilities of your role
- Attitude toward work, how you prioritize
- Reporting obligations, escalation criteria
- **Mandatory procedures that must never be skipped** (e.g., never disclose confidential information externally, always get approval before production operations)

injection.md is your **mutable behavioral guidelines**. Updated as business policies change.

| Property | Value |
|----------|-------|
| Modification rights | You can update it. Supervisor can also edit |
| Modification frequency | As needed (when duties change) |
| Impact of changes | Behavioral policy changes = like changing jobs |

### The Difference Between identity and injection

This is the most important distinction:

| | identity.md | injection.md |
|--|------------|-------------|
| What it defines | **Who you are** (personality) | **What you do** (duties) |
| Human analogy | Innate temperament and character | The job you hold and workplace rules |
| What happens if changed | You become a different person | You change jobs |
| Mutability | Immutable in principle | Updated as needed |
| Contains | Speaking style, way of thinking, values | Responsibilities, procedures, behavioral rules |

**Examples:**
- "Speaks in polite, formal language" → identity (personality trait)
- "Always run tests before production deploy" → injection (work procedure)
- "Cautious personality, crosses bridges carefully" → identity (thinking habit)
- "Report security incidents to supervisor immediately" → injection (duty rule)

### specialty_prompt.md — Expert Prompt

Role-specific instructions (engineer, manager, writer, etc.).
Auto-generated from role templates. Updated only when the role changes.

| Property | Value |
|----------|-------|
| Modification rights | System automatic (on role application) |
| Modification frequency | Rare (role changes only) |

---

## Permissions and Configuration

### permissions.md — What You Can Do

Defines available tools, accessible paths, and executable commands.

- Readable and writable paths
- Available external tools (Slack, Gmail, GitHub, etc.)
- Blocked commands (safety block list)

| Property | Value |
|----------|-------|
| Modification rights | Supervisor or admin |
| Modification frequency | Rare |

### status.json — Configuration

The **Single Source of Truth (SSoT)** for your execution parameters.

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-4-6",
  "credential": "anthropic",
  "max_tokens": 16384,
  "max_turns": 200,
  "supervisor": "aoi"
}
```

| Field | Description |
|-------|-------------|
| `enabled` | Enabled/disabled |
| `role` | Role (engineer, manager, writer, researcher, ops, general) |
| `model` | LLM model to use |
| `credential` | API credential name |
| `max_tokens` | Maximum tokens per response |
| `max_turns` | Maximum turns per session |
| `supervisor` | Supervisor Anima name (null = top-level) |
| `background_model` | Lightweight model for Heartbeat/Cron (falls back to main model if unset) |

| Property | Value |
|----------|-------|
| Modification rights | CLI commands or admin. Supervisor can change via `set_subordinate_model` |
| Modification frequency | As needed |

### bootstrap.md — First-Boot Instructions

Exists only on first startup. Instructs you to enrich identity and injection,
and design initial heartbeat and cron. Automatically deleted after completion.

| Property | Value |
|----------|-------|
| Modification rights | — |
| Modification frequency | Once (deleted after completion) |

---

## Periodic Actions

### heartbeat.md — Periodic Patrol

**Auto-starts at regular intervals to assess the situation and make plans.**
Like a human periodically checking their inbox and reviewing ongoing work.

Contains:
- **Active hours**: When to be active (e.g., `09:00 - 18:00`)
- **Checklist**: Items to check during patrol
- **Notification rules**: Conditional reporting and notifications

**Important**: Heartbeat performs **assessment and planning only**. When tasks needing execution are found, delegate to subordinates via `delegate_task` or submit via `submit_tasks`.

| Property | Value |
|----------|-------|
| Modification rights | You can update. Supervisor can also edit |
| Modification frequency | As needed (when duties change) |

### cron.md — Scheduled Tasks

**Tasks that must be executed at specific times.**

Two types of tasks:
- **LLM type**: Agent performs with judgment and reasoning (e.g., "Every morning at 9, review yesterday's progress and plan today")
- **Command type**: Deterministic execution without judgment (e.g., "Run backup script at 2 AM daily")

```markdown
## Daily Planning
schedule: 0 9 * * *
type: llm
Review yesterday's progress from episodes/ and plan today's tasks.

## Backup Execution
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

| Property | Value |
|----------|-------|
| Modification rights | You can update. Supervisor can also edit |
| Modification frequency | As needed |

### The Difference Between heartbeat and cron

| | heartbeat.md | cron.md |
|--|-------------|---------|
| Purpose | Assess situation and make plans | Execute specific tasks |
| Trigger | Regular interval (default 30 min) | Specified time (cron expression) |
| What it can do | Observe, plan, reflect (**no execution**) | LLM task or command execution |
| Human analogy | "Periodically look around" | "Do this every morning at 9" |
| Configuration details | See `operations/heartbeat-cron-guide.md` | Same |

---

## Work State (state/)

### state/current_task.md — Current Task

The task you are currently working on (one at a time). Records the goal, progress, and blockers.

### state/pending.md — Backlog

Free-form notes for tasks not yet started or things to remember.

### state/task_queue.jsonl — Task Queue

Structured task tracking. Operated via `backlog_task` / `update_task` / `list_tasks`.
Tasks with `source: human` MUST be processed with highest priority.

### state/pending/ — Execution Queue

Execution queue for tasks submitted via `submit_tasks` / `delegate_task` tools.
TaskExec polls every 3 seconds, automatically picking up and executing them. Do not manually create JSON files here.

| Property | Value |
|----------|-------|
| Modification rights | Self (via tools) |
| Modification frequency | As needed (automatic) |

---

## Memory

For details on the memory system, see `anatomy/memory-system.md`.

| Directory | Type | Content |
|-----------|------|---------|
| `episodes/` | Episodic memory | Daily logs of what you did and when |
| `knowledge/` | Semantic memory | Learned knowledge, know-how, patterns |
| `procedures/` | Procedural memory | Step-by-step procedures (forgetting-resistant) |
| `skills/` | Skills | Personal skills (forgetting-resistant) |
| `shortterm/` | Short-term memory | Context continuity between sessions |

---

## Activity Records & Assets

### activity_log/ — Activity Log

Chronological record of all actions (`{date}.jsonl`). Automatically records message send/receive, tool usage, Heartbeat, errors, etc.
Used by Priming as the source for injecting recent activity into the system prompt.

### transcripts/ — Conversation Records

Transcripts of conversations with humans.

### assets/ — Images & 3D Models

Character images, 3D models, and other asset files.

| Property | Value |
|----------|-------|
| Modification rights | System automatic |
| Modification frequency | Automatic |

---

## Full File Modification Rights Summary

| File | Modification Rights | Frequency |
|------|-------------------|-----------|
| `identity.md` | Immutable in principle (admin only) | Immutable |
| `character_sheet.md` | Reference only | Immutable |
| `injection.md` | Self / supervisor | As needed |
| `specialty_prompt.md` | System automatic | Rare |
| `permissions.md` | Supervisor / admin | Rare |
| `status.json` | CLI / admin / supervisor | As needed |
| `bootstrap.md` | Once only | Deleted |
| `heartbeat.md` | Self / supervisor | As needed |
| `cron.md` | Self / supervisor | As needed |
| `state/*` | Self (via tools) | As needed |
| `episodes/` | System automatic | Daily |
| `knowledge/` | Self / auto-consolidation | As needed |
| `procedures/` | Self / auto-generated | As needed |
| `skills/` | Self | As needed |
| `shortterm/` | System automatic | Automatic |
| `activity_log/` | System automatic | Automatic |

---

## Shared Resources (Outside Your Directory)

Beyond your own directory, there are resources shared by all Animas:

| Path | Content |
|------|---------|
| `common_knowledge/` | Reference documents shared by all Animas (including this file) |
| `common_skills/` | Skills shared by all Animas |
| `shared/channels/` | Board (shared channels) |
| `shared/users/` | User profiles (cross-Anima) |
| `shared/common_knowledge/` | Organization-specific shared knowledge (accumulated during operation) |
| `company/vision.md` | Organization vision |
| `prompts/` | System prompt templates |
