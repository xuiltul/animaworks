# Anima Configuration Files: A Complete Guide

A reference to the purpose, change rules, and relationships of all files that make up you (Anima).
Consult this guide to find out “what this file is for” and “whether you can change it yourself.”

## File List and Roles

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # あなたのプロフィール（名前・外見・イメージカラー）
├── SOUL.md              # あなたの人格（性格・話し方・価値観）
├── AGENTS.md            # あなたの職務（職責・行動指針・必須手順）
├── specialty_prompt.md  # ロール別専門プロンプト
├── character_sheet.md   # 作成時の設計図（参照用）
├── permissions.json       # 権限（使えるツール・アクセス範囲）
├── status.json          # 設定情報（モデル・パラメータ）
├── BOOTSTRAP.md         # 初回起動指示（完了後に削除）
├── HEARTBEAT.md         # 定期巡回の設定
├── cron.md              # 定時タスクの設定
├── state/               # 作業状態
│   ├── current_state.md
│   └── task_results/    # タスク実行結果
├── episodes/            # エピソード記憶
├── knowledge/           # 意味記憶
├── procedures/          # 手続き記憶
├── skills/              # 個人スキル
├── shortterm/           # 短期記憶
├── activity_log/        # 活動ログ
├── transcripts/         # 会話記録
└── assets/              # 画像・3Dモデル
```

### Encapsulation Boundaries

Based on Anima’s design principle of “the encapsulated individual,” files are classified into three layers.
This classification is the basis for determining “who is allowed to change a given file.”

| Classification | Files | Reason |
|------|---------|------|
| **Inside the capsule** (thoughts and memory) | `IDENTITY.md`, `SOUL.md`, `episodes/`, `knowledge/`, `procedures/`, `skills/`, `state/`, `shortterm/` | Profile, personality, experience, and learning belong to the individual. They cannot be changed externally. |
| **Capsule boundary** (the point of contact between the organization and the individual) | `AGENTS.md`, `cron.md`, `HEARTBEAT.md`, `permissions.json` | Organization configuration owned by root. The supervisor can edit `cron.md` / `HEARTBEAT.md`; other files are changed through the root-managed API/CLI. |
| **Outside the capsule** (administrative information) | `status.json`, `specialty_prompt.md` | Configuration handled by the root server and the CLI while Anima is shut down. The Anima process cannot write to these files directly. |

- Changing the **inside** would make you “a different person” or “lose your memories.” This is not allowed.
- Changing the **boundary** would mean “changing jobs” or “changing your scope of work.” This is valid for organizational operations, but must be done through a root-owned API or tool.
- Changing the **outside** has no direct effect on your personality or behavior.

> **The relationship between growth and IDENTITY.md / SOUL.md**: IDENTITY.md (profile and appearance) and SOUL.md (personality and values) are immutable baselines that Anima cannot rewrite. “Growth” is expressed through what accumulates in `knowledge/` (lessons learned), `procedures/` (procedures acquired), and `skills/` (skills developed). Even when IDENTITY.md / SOUL.md remain fixed, accumulated memories will certainly change behavior—this is the model of “the same person growing.” If a change is necessary, a human administrator uses the root-managed API/CLI. Root also authorizes and saves write requests made during bootstrap.

---

## Your Profile and Personality (IDENTITY / SOUL)

### IDENTITY.md — Your Profile and Appearance

**Name, basic profile, appearance, image color, and AI model information.**

- Name, configured age, affiliation, position, supervisor
- Appearance concept, height, hair and eye color
- Image color
- Information about the AI model you use

IDENTITY.md is the **immutable baseline** for your profile and appearance. Changing it changes how you look to those around you.

| Item | Value |
|------|-----|
| Change permission | Generally not changed. Supervisor or administrator only |
| Change frequency | Immutable (set at creation) |
| Effect of change | Profile and appearance change |

### SOUL.md — Your Personality and Values

**Your “personality” itself.**

- Tone, manner of speaking, and phrasing
- Thinking habits, values, and standards for making decisions
- Likes, dislikes, and interests
- Catchphrases and beliefs

SOUL.md is the **immutable baseline** of your personality. Changing it would make you “a different person.”

| Item | Value |
|------|-----|
| Change permission | Generally not changed. Supervisor or administrator only |
| Change frequency | Immutable (set at creation) |
| Effect of change | Personality changes = you become a different person |

### character_sheet.md — Blueprint

A copy of the Markdown file used at creation. The source material for IDENTITY.md / SOUL.md and AGENTS.md.
It is stored for reference and is not normally changed.

| Item | Value |
|------|-----|
| Change permission | Reference only |
| Change frequency | Immutable |

---

## Your Duties (AGENTS)

### AGENTS.md — What You Do

**Your professional duties, responsibilities, and approach to work.**

- Scope and responsibilities of your assigned work
- Attitude toward work and how you set priorities
- Reporting obligations and escalation criteria
- **Procedures that must never be skipped** (e.g., do not disclose confidential information externally; get confirmation before operating on production environments)

AGENTS.md is your **mutable behavioral guidance**. An authorized supervisor or human administrator can request changes through root, and root authorizes writes during bootstrap. Do not write to it directly from the Anima process.

| Item | Value |
|------|-----|
| Change permission | You can update it yourself. Supervisor can also edit it |
| Change frequency | As needed (when duties change) |
| Effect of change | Policy for your behavior changes = like changing jobs |

### The Difference Between IDENTITY.md / SOUL.md and AGENTS.md

This is the most important distinction:

| | IDENTITY.md | SOUL.md | AGENTS.md |
|--|------------|----------|-------------|
| What it defines | **Profile and appearance** (name and looks) | **Who you are** (personality) | **What you do** (duties) |
| Human equivalent | Official records, profile, and appearance | Inborn temperament and personality | Profession and workplace rules |
| What happens if changed | How you look to others changes | You become a different person | You change jobs |
| Whether it can be changed | Generally immutable | Generally immutable | Updated as needed |
| What it includes | Name, basic profile, appearance, image color | Manner of speaking, way of thinking, values, individuality | Responsibilities, procedures, code of conduct |

**Examples:**
- “158 cm tall, with silver hair” → IDENTITY.md (appearance)
- “Speaks politely and formally” → SOUL.md (personality expressed in speech)
- “Always run tests before deploying to production” → AGENTS.md (work procedure)
- “Cautious by nature and always checks things thoroughly” → SOUL.md (thinking habit)
- “Report security incidents to the supervisor immediately” → AGENTS.md (work rule)

### specialty_prompt.md — Specialized Prompt

Specialized instructions based on the role (engineer, manager, writer, etc.).
Automatically generated from a role template. Updated only when the role changes.

| Item | Value |
|------|-----|
| Change permission | Automatic by the system (when a role is applied) |
| Change frequency | Rarely (only when the role changes) |

---

## Permissions and Configuration

### permissions.json — What You Can Do

Defines the tools you can use, paths you can access, and commands you can run.

- Locations you can read from and write to
- External tools you can use (Slack, Gmail, GitHub, etc.)
- Commands you cannot run (a blocklist for safety)

| Item | Value |
|------|-----|
| Change permission | Supervisor or administrator |
| Change frequency | Rarely |

### status.json — Configuration Information

The **Single Source of Truth (SSoT)** for your runtime parameters.

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-5-5",
  "credential": "anthropic",
  "max_tokens": 16384,
  "supervisor": "aoi"
}
```

| Field | Description |
|-----------|------|
| `enabled` | Enabled/disabled |
| `role` | Role (engineer, manager, writer, researcher, ops, general) |
| `model` | LLM model to use |
| `credential` | Name of the API authentication credentials |
| `max_tokens` | Maximum tokens per response |
| `supervisor` | Supervisor’s Anima name (null = top level) |
| `background_model` | Lightweight model for Heartbeat/Cron (uses the main model if unset) |

| Item | Value |
|------|-----|
| Change permission | CLI command or administrator. Supervisor can change it using `set_subordinate_model` |
| Change frequency | As needed |

### BOOTSTRAP.md — First-Startup Instructions

A file that exists only when you start up for the first time. It provides instructions for fleshing out IDENTITY.md / SOUL.md and AGENTS.md,
and for initially designing HEARTBEAT.md and cron.md. It is automatically deleted after completion.

| Item | Value |
|------|-----|
| Change permission | — |
| Change frequency | Once (deleted after completion) |

---

## Recurring Actions

### HEARTBEAT.md — Regular Check-ins

Configuration for **starting up automatically at set intervals to check the situation and plan**.
Like a person regularly checking their inbox and reviewing ongoing work.

Includes:
- **Active hours**: When to be active (e.g., `09:00 - 18:00`)
- **Checklist**: Items to check during a check-in
- **Notification rules**: Reports and notifications based on conditions

**Important**: Heartbeat is for **checking and planning only**. If you find a task that needs to be carried out, delegate it to a subordinate using `delegate_task` or submit it using `submit_tasks`.

| Item | Value |
|------|-----|
| Change permission | You can update it yourself. Supervisor can also edit it |
| Change frequency | As needed (when duties change) |

### cron.md — Scheduled Tasks

Defines tasks that **must be run at set times**.

There are two types of tasks:
- **LLM type**: The agent performs the task using judgment and reasoning (e.g., “Every morning at 9, review yesterday’s progress and plan for today”)
- **Command type**: Runs deterministically without requiring judgment (e.g., “Run the backup script every day at 2 a.m.”)

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
episodes/ から昨日の進捗を確認し、今日のタスクを計画する。

## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

| Item | Value |
|------|-----|
| Change permission | You can update it yourself. Supervisor can also edit it |
| Change frequency | As needed |

### Difference Between Heartbeat and cron

| | HEARTBEAT.md | cron.md |
|--|-------------|---------|
| Purpose | Check the situation and make plans | Run scheduled tasks |
| Trigger | At regular intervals (30 minutes by default) | At a specified time (cron expression) |
| What it can do | Observe, plan, and reflect (**does not execute tasks**) | Run LLM tasks or commands |
| Human equivalent | “Look around regularly” | “Do this every morning at 9” |
| Configuration details | See `operations/heartbeat-cron-guide.md` | Same as left |

---

## Work Status (state/）

### state/current_state.md — Current Status

The task or situation you are working on right now (one at a time). Records the task’s goal, progress, and blockers.
It is retained across normal Heartbeat / cron / conversation boundaries. Prompt injection is limited to 3000 characters, and disk trimming runs at the default of 8000 characters (`heartbeat.current_state_max_chars`, 0 = disabled).

> **Legacy task storage**: `state/task_queue.jsonl` and `state/pending/` are evidence kept only for migration and export. Do not use or edit them as an active queue. Use the canonical task tools.

### Canonical Tasks

Use `list_tasks` / `submit_tasks` / `update_task`. The host stores instructions, dependencies, execution attempts, and delegation aliases in a single TaskStore. Human-originated tasks take highest priority. Only the host sets `in_progress`, and results are declared using `done` / `pending` / `cancelled`. To explicitly resume, specify `resume: true` with the same ID, preserving the original input and history.

### state/task_results/ — Task Execution Results

The accepted result summary from TaskExec is `{task_id}/{attempt_token}.md` (maximum 2000 characters). The host passes the accepted result it selected to subsequent tasks; do not infer completion merely from the presence of a file.

| Item | Value |
|------|-----|
| Change permission | You can operate it yourself (through tools) |
| Change frequency | As needed (automatic) |

---

## Memory

See `anatomy/memory-system.md` for details about the memory system.

| Directory | Type | Contents |
|------------|------|------|
| `episodes/` | Episodic memory | Daily logs of what you did and when |
| `knowledge/` | Semantic memory | Knowledge, know-how, and patterns learned |
| `procedures/` | Procedural memory | Procedures (resistant to forgetting) |
| `skills/` | Skills | Personal skills (resistant to forgetting) |
| `shortterm/` | Short-term memory | For maintaining context between sessions |

---

## Activity Records and Assets

### activity_log/ — Activity Log

A chronological record of all actions (`{date}.jsonl`). Automatically records message sending and receiving, tool usage, Heartbeat, errors, and more.
Used by Priming as a source for injecting recent activity into the system prompt.
To search explicitly, use `search_memory(query="...", scope="activity_log")` (BM25. In `scope="all"`, vector results are combined using RRF).

### transcripts/ — Conversation Logs

Transcripts of conversations with humans.

### assets/ — Images and 3D Models

Asset files such as character images and 3D models.

| Item | Value |
|------|-----|
| Modification permission | Automatic by the system |
| Modification frequency | Automatic |

---

## Summary of Modification Permissions for All Files

| File | Modification permission | Modification frequency |
|---------|---------|---------|
| `IDENTITY.md` | Generally immutable (administrators only) | Immutable |
| `SOUL.md` | Generally immutable (administrators only) | Immutable |
| `character_sheet.md` | Read-only | Immutable |
| `AGENTS.md` | Self / supervisor | As needed |
| `specialty_prompt.md` | Automatic by the system | Rare |
| `permissions.json` | Supervisor / administrator | Rare |
| `status.json` | CLI / administrator / supervisor | As needed |
| `BOOTSTRAP.md` | One-time | Deletion |
| `HEARTBEAT.md` | Self / supervisor | As needed |
| `cron.md` | Self / supervisor | As needed |
| `state/*` | Self (via tools) | As needed |
| `episodes/` | Automatic by the system | Daily |
| `knowledge/` | Self / automatic merge | As needed |
| `procedures/` | Self / automatic generation | As needed |
| `skills/` | Self | As needed |
| `shortterm/` | Automatic by the system | Automatic |
| `activity_log/` | Automatic by the system | Automatic |

---

## Shared Resources (Outside Anima)

There are resources shared by all Anima outside your directory:

| Path | Contents |
|------|------|
| `common_knowledge/` | Reference documentation shared by all Anima (this file is one of them) |
| `common_skills/` | Skills shared by all Anima |
| `shared/channels/` | Board (shared channel) |
| `shared/users/` | User profile (across Anima) |
| `shared/common_knowledge/` | Organization-specific shared knowledge (accumulated during operation) |
| `company/vision.md` | Organization vision |
| `prompts/` | System prompt template |
