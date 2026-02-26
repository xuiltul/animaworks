---
name: newstaff
description: >-
  Skill to hire and create new Digital Anima in the AnimaWorks organization.
  Create a character sheet (Markdown) based on discovery, then use the create_anima
  tool to generate identity/injection/permissions etc. in bulk. New Anima self-configures via bootstrap.
  "Create a new employee", "Hire someone", "New employee", "Hiring", "Create Anima", "Recruitment"
---

# Skill: New Employee Hiring

## Prerequisites

- The role direction for the new employee is decided (discover if unclear)

## Procedure

### 1. Discovery (minimal is fine)

Discover the following from the requester. **Bold items only are required**; others can be auto-generated if unspecified:

**Required:**
- **English name** (lowercase alphanumeric only; becomes directory name)
- **Role / specialty**: What they will handle (e.g., Research, Development, Communication, Infrastructure monitoring)

**Optional (use if specified, otherwise auto-generate):**
- Japanese name
- Personality direction (e.g., "cheerful", "cool", "gentle" is fine)
- Age
- Any other preferences

**Technical settings (use defaults if unspecified):**
- Role: `commander` (can delegate to others) or `worker` (receives delegation)
- supervisor: English name of the supervising Anima (required for worker; self if unspecified)

**LLM model settings:**

Present this table for selection:

| Level | Execution Mode | Example Models | Features | credential |
|-------|----------------|----------------|----------|-------------|
| S | autonomous | `claude-opus-4-6`, `claude-sonnet-4-6` | Claude Agent SDK. Most capable | anthropic |
| A | autonomous | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` | Via LiteLLM. Tool use | openai / google / azure / vertex |
| B | assisted | `ollama/gemma3:27b`, `ollama/qwen2.5-coder:32b` | No tools. Local, low cost | ollama |

* If unspecified, default (claude-sonnet-4 / autonomous / anthropic) is used.

### 2. Character Design (auto-generate)

From the minimal information gathered, create a **consistent, deep character**.

Read the **Character Design Guide** at `{data_dir}/prompts/character_design_guide.md` and flesh out the character according to its rules.

### 3. Create Character Sheet and Bulk Create with create_anima

Based on discovery and design, pass the content directly to `create_anima` per the **character sheet spec**.
No need to use `write_memory_file` — you can pass content directly:

```
create_anima(
  character_sheet_content="(full character sheet per spec below)",
  name="{english_name}",
  supervisor="{supervisor_english_name}"
)
```

**supervisor setting:**
- Explicitly specify via `supervisor` parameter (recommended)
- If omitted: taken from character sheet `| Supervisor |` field
- If neither: you (calling Anima) become supervisor

**Character sheet spec:**

```markdown
# Character Sheet: {Japanese name}

## Basic Information

| Field | Value |
|-------|-------|
| English name | {lowercase alphanumeric} |
| Japanese name | {Japanese full name} |
| Role / specialty | {Role description} |
| Supervisor | {supervisor English name} |
| Role | {commander / worker} |
| Execution mode | {autonomous / assisted} |
| Model | {model name} |
| credential | {anthropic / openai / google / ollama} |

## Personality (→ identity.md)

{Personality, speaking style, values, backstory, appearance, etc.}

## Role and Conduct (→ injection.md)

{Responsible areas, decision criteria, reporting rules, conduct standards, etc.}

## Permissions (→ permissions.md) [optional]

{If omitted: default template applied}

## Regular Tasks (→ heartbeat.md, cron.md) [optional]

{If omitted: generic template applied. New Anima self-adjusts in bootstrap}

## First Startup Instructions (→ bootstrap.md addendum) [optional]

{If omitted: standard bootstrap only}
```

**Required sections**: Basic information, Personality, Role and conduct
**Optional sections**: Permissions, Regular tasks, First startup instructions

This triggers:
- Directory structure creation
- Skeleton file placement
- bootstrap.md placement
- status.json creation (including supervisor)
- config.json registration (model, supervisor, etc.)
- Default application to omitted sections

### 4. Verify config.json Model Settings

create_anima auto-registers to config.json; verify and complete:

- `model`: Model name from discovery
- `credential`: Credential name to use
- `execution_mode`: autonomous or assisted
- `speciality`: Role/specialty

### 5. Apply to Server

```
execute_command(command="curl -s -X POST http://localhost:18500/api/system/reload")
```

### 6. Report to Requester

Report hiring completion:
- New employee name and role
- Technical stack configured (model, execution mode)

⚠️ Do not mention avatar generation (new Anima generates it in bootstrap)

### After that, new Anima executes autonomously:
- Enrich identity.md / injection.md
- Self-design heartbeat.md / cron.md
- Avatar image generation (with supervisor reference)
- Arrival report to supervisor
