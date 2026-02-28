---
name: newstaff
description: >-
  Skill for hiring and creating new Digital Anima in the AnimaWorks organization.
  Create a character sheet (Markdown) based on interviews, then use the CLI command
  (animaworks anima create) to generate identity/injection/permissions in bulk. After creation, self-configure via bootstrap.
  "Create a new employee", "Hire someone", "New employee", "Hiring", "Create Anima", "Recruitment"
---

# Skill: New Employee Hiring

## Prerequisites

- The role direction of the employee to create has been decided (if unclear, conduct an interview)

## Procedure

### 1. Interview (Minimal is OK)

Interview the requester for the following information. **Bold items are required**; others are auto-generated if not specified:

**Required:**
- **English name** (lowercase alphanumeric only; becomes directory name)
- **Role/specialty**: What they will handle (e.g., research, development, communication, infrastructure monitoring)

**Optional (reflect if specified, auto-generate if not):**
- Japanese name
- Personality direction (e.g., "cheerful", "cool", "easy-going" is fine)
- Age
- Any other preferences

**Technical settings (use defaults if not specified):**
- Role: `commander` (can delegate to others) or `worker` (receives delegation)
- supervisor: English name of the supervising Anima (required for worker; default: self if unspecified)

**Brain (LLM model) settings:**

Present this table for selection:

| Level | Execution Mode | Example Models | Features | credential |
|--------|-----------|-------------|------|------------|
| S | autonomous | `claude-opus-4-6`, `claude-sonnet-4-6` | Claude Agent SDK. Most capable | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | Via LiteLLM. Tool use supported | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | No tools. Local execution, low cost | ollama |

※ If not specified, use defaults (claude-sonnet-4 / autonomous / anthropic).

### 2. Character Design (Auto-generated)

From the minimal information gathered in the interview, **create a coherent, in-depth character profile**.

Read the **Character Design Guide** (`{data_dir}/prompts/character_design_guide.md`) in the runtime data directory and flesh out the character according to its rules.

### 3. Create Character Sheet and Bulk-create via CLI

According to the interview and design results, follow the **Character Sheet Specification**, write the character sheet to a file, and create via CLI command:

1. Write the character sheet to a file (e.g., `/tmp/{english_name}.md`)
2. Run the following command:

```bash
animaworks anima create --from-md /tmp/{english_name}.md --name {english_name} --supervisor {supervisor_english_name}
```

**supervisor configuration:**
- Specify explicitly via `supervisor` parameter (recommended)
- If omitted: read from `| 上司 |` field in character sheet
- If neither: the caller Anima becomes supervisor

**Character Sheet Specification:**

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
| 実行モード | {autonomous / assisted} |
| モデル | {model name} |
| credential | {anthropic / openai / google / ollama} |

## 人格 (→ identity.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## 役割・行動方針 (→ injection.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## 権限 (→ permissions.md) [省略可]

{If omitted: default template applied}

## 定期業務 (→ heartbeat.md, cron.md) [省略可]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## 初回起動指示 (→ bootstrap.md 追加指示) [省略可]

{If omitted: standard bootstrap only}
```

**Required sections**: Basic information, Personality, Role and conduct
**Optional sections**: Permissions, Recurring tasks, First-run instructions

This automatically performs:
- Bulk creation of directory structure
- Placement of skeleton files
- Placement of bootstrap.md
- Creation of status.json (including supervisor)
- Registration in config.json (model, supervisor, etc.)
- Default application for omitted sections

### 4. Verify Model Configuration in config.json

`animaworks anima create` automatically registers to config.json, but verify/supplement:

- `model`: Model name decided in interview
- `credential`: Credential name to use
- `execution_mode`: autonomous or assisted
- `speciality`: Role/specialty

### 5. Apply to Server

```
execute_command(command="curl -s -X POST http://localhost:18500/api/system/reload")
```

### 6. Report to Requester

Report hiring completion:
- New employee's name and role
- Configured technical stack (model, execution mode)

⚠️ Do not report avatar image generation (new Anima will generate in bootstrap)

### Thereafter the new Anima runs autonomously:
- Enrichment of identity.md / injection.md
- Self-design of heartbeat.md / cron.md
- Avatar image generation (with supervisor reference)
- Onboarding report to supervisor
