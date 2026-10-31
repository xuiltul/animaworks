---
name: newstaff
description: >-
  A skill for hiring/creating a new Digital Anima in an AnimaWorks organization.
  Create a character sheet (Markdown) based on an interview, then use the CLI command (animaworks anima create) to generate   IDENTITY/SOUL/AGENTS/permissions and other items all at once. After creation, it performs its own setup through bootstrap.
  “Create a new employee,” “hire someone,” “new employee,” “hiring,” “create an Anima,” “recruit,” “add a team member,” “add staff,” “create a subordinate,” “add personnel,” “hire,” “recruit,” “team member”
---


# Skill: Hiring a New Employee

## Prerequisites

- The direction of the new employee's role must be decided (if unclear, conduct an interview)

## Procedure

### 1. Interview (minimal is fine)

Gather the following information from the requester. **Only the bold items are required**; if others are not specified, they will be auto-generated:

**Required:**
- **English name** (lowercase alphanumeric only. This becomes the directory name)
- **Role/specialty**: What they will handle (e.g., research, development, communication, infrastructure monitoring)
- **Personality direction**: Not the function but the tone (e.g., energetic, gal, airheaded, passionate, ojou, older sister). If there are already 2 cool types in the organization, don't choose that
- **Face type**: Choose 1 from `{data_dir}/prompts/face_types.md`. Avoid too much overlap with existing members

**Optional (reflected if specified, auto-generated if not):**
- Japanese name
- Age
- Any other preferences

**Technical configuration (defaults used if not specified):**
- Role: `commander` (can delegate to other employees) or `worker` (receives delegation)
- supervisor: English name of the Anima who is the supervisor (required for worker; if not specified, self)

**Brain (LLM model) configuration:**

Do not show the model table to the user to choose from (since unusable models are mixed in). Only when the user specifies a model, configure it with reference to the table below. If not specified, omit the model row in the character sheet and proceed with the default model:

| Level | Execution mode | Example models | Features | credential |
|--------|-----------|-------------|------|------------|
| S | autonomous | `claude-opus-5-5`, `claude-sonnet-5-5` | Claude Agent SDK. Most capable | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | Via LiteLLM. Tool use available | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | No tools. Local execution, low cost | ollama |

※ If not specified, omit the model, execution mode, and credential rows (defaults will be used).

### 2. Character design (auto-generated)

Flesh out the character from the information gathered in the interview. **Personality comes first, function comes second.**

Always Read the following in this order:
1. `{data_dir}/prompts/face_types.md`
2. `{data_dir}/prompts/character_design_guide.md`

Don't associate a cool beauty with the role. Don't make hobbies an extension of work. Check that hair color, face type, and speech style don't overlap with existing members.

### 3. Create a character sheet and create everything at once via the CLI

Write the results of the interview and design to a file as a character sheet, following the **character sheet specification**, then create it with the CLI command:

1. Write the character sheet to a file (e.g., `/tmp/{english_name}.md`)
2. Run the following command:

```bash
animaworks anima create --from-md /tmp/{english_name}.md --name {english_name} --supervisor {supervisor_english_name}
```

**Configuring the supervisor:**
- Explicitly specify it with the `supervisor` parameter (recommended)
- If omitted: Retrieved from the character sheet’s `| 上司 |` field
- If neither is provided: You (the calling Anima) become the supervisor

**Character sheet specification:**

```markdown
# キャラクターシート: {Japanese name}

## 基本情報

| 項目 | 設定 |
|------|------|
| 英名 | {lowercase alphanumeric} |
| 日本語名 | {Japanese full name} |
| 役職/専門 | {Role description} |
| 上司 | {supervisor English name} |
| 役割 | {commander / worker} |
| モデル | {model name — ユーザーが指定した時だけ書く} |
| credential | {ユーザーがモデルを指定した時だけ書く} |

## 人格 (→ SOUL.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## 役割・行動方針 (→ AGENTS.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## 権限 (→ permissions.json) [省略可]

{If omitted: default template applied}

## 定期業務 (→ HEARTBEAT.md, cron.md) [省略可]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## 初回起動指示 (→ BOOTSTRAP.md 追加指示) [省略可]

{If omitted: standard bootstrap only}
```

**Required sections**: Basic Information, Personality, Role and Behavioral Guidelines
**Optional sections**: Permissions, Regular Duties, Initial Startup Instructions

This automatically performs:
- Creates the directory structure all at once
- Places skeleton files
- Places BOOTSTRAP.md
- Creates status.json (including supervisor)
- Registers it in config.json (model, supervisor, etc.)
- Applies defaults to omitted sections

### 4. Confirm the model configuration in config.json

`animaworks anima create` automatically registers in config.json, but confirm and supplement the following:

- `model`: The model name decided during the interview
- `credential`: The credential name to use
- `execution_mode`: autonomous or assisted
- `speciality`: Job title / specialty

### 4.5. Propose and configure heartbeat frequency

Based on the new employee's work drive type, propose a periodic heartbeat interval to the requester and configure the agreed value:

| Drive type | Recommended interval | Example |
|--------|---------|-----|
| Spontaneous situational judgment and coordination as main work | 30 minutes (default) | Team coordinator, progress supervisor |
| Message/event-driven as the main driver | 60–120 minutes | Review lead, development lead |
| Cron periodic tasks as the main driver | 120–240 minutes | Monitoring, accounting, legal |

Add `"heartbeat_interval_minutes": <分>` (1–1440) to `{data_dir}/animas/{英名}/status.json` for the configuration.
Message-triggered heartbeats and cron run independently of this configuration, so extending the interval does not reduce responsiveness. When in doubt, choose a longer interval (empty heartbeats waste tokens).

### 5. Reflect on the server

```
Bash: curl -s -X POST "${ANIMAWORKS_SERVER_URL:-http://localhost:18500}"/api/system/reload
```

### 6. Report to the requester

Report the completion of hiring:
- The new employee's name and role
- The configured technical stack (model, execution mode)

- Upon arrival, they should immediately start on the first small task, and you should report the results in a summary

⚠️ Do not report avatar image generation (the new Anima itself generates it via bootstrap)

### The new Anima autonomously performs the following thereafter:
- Enriches IDENTITY.md / SOUL.md / AGENTS.md
- Self-designs HEARTBEAT.md / cron.md
- Generates an avatar image (with supervisor reference)
- Reports its appointment to the supervisor
- Starts on its first small task (something useful to the user) and reports the results to the supervisor
