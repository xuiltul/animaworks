# Complete Guide to Anima Configuration Files

A reference for the roles, modification rules, and relationships of all files that make up you (Anima).
Use when: checking "what is this file for" or "can I change it myself."

## File List and Roles

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # Your profile (name, appearance, image color)
├── SOUL.md              # Your personality (character, voice, values)
├── AGENTS.md            # Your job (responsibilities, behavioral guidelines, required procedures)
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

Based on Anima's design principle of "encapsulated individuals," files are classified into three layers.
This classification serves as the basis for "who is allowed to modify which file."

| Classification | Files | Reason |
|------|---------|------|
| **Inside the capsule** (thoughts, memories) | `IDENTITY.md`, `SOUL.md`, `episodes/`, `knowledge/`, `procedures/`, `skills/`, `state/`, `shortterm/` | Profile, personality, experience, and learning belong to the individual. Cannot be modified externally |
| **Boundary of the capsule** (interface between organization and individual) | `AGENTS.md`, `cron.md`, `HEARTBEAT.md`, `permissions.json` | Organization settings owned by root. Supervisors can edit `cron.md` / `HEARTBEAT.md`; other files are changed via root-managed API/CLI |
| **Outside the capsule** (administrative information) | `status.json`, `specialty_prompt.md` | Settings handled by the root server and the CLI during shutdown. Cannot be written directly from the Anima process |

- Changing the **inside** means "becoming a different person" or "losing memories." This is not permitted
- Changing the **boundary** means "changing jobs" or "changing the scope of work." This is legitimate for organizational operations, but is done via root-owned APIs/tools
- Changing the **outside** has no direct effect on the individual's personality or behavior

> **Relationship between growth and IDENTITY.md / SOUL.md**: IDENTITY.md (profile and appearance) and SOUL.md (personality and values) are immutable baselines and cannot be rewritten by Anima itself. "Growth" is expressed through accumulation in `knowledge/` (lessons learned), `procedures/` (procedures acquired), and `skills/` (skills refined). Even if IDENTITY.md / SOUL.md are fixed, behavior reliably changes through the accumulation of memories—this is the model of "the same person growing." If changes are needed, a human administrator uses root-managed API/CLI. Write requests during bootstrap are also authorized and saved by root.

---

## Your Profile and Personality (IDENTITY / SOUL)

### IDENTITY.md — Your Profile and Appearance

**Your name, basic profile, appearance, image color, and AI model information.**

- Name, age setting, affiliation, role, supervisor
- Appearance, height, hair color, eye color
- Image color
- AI model information

IDENTITY.md is the **immutable baseline** of your profile and appearance. Changing it changes how others see you.

| Item | Value |
|------|-----|
| Modification permission | In principle, not changed. Only supervisor or administrator |
| Modification frequency | Immutable (fixed at creation) |
| Impact of modification | Profile and appearance change |

### SOUL.md — Your Character and Values

**Your "personality" itself.**

- Tone of speech, manner of speaking, phrasing
- Thinking habits, values, decision-making principles
- Preferences, interests
- Catchphrase, beliefs

SOUL.md is the **immutable baseline** of your personality. Changing this makes you a "different person."

| Item | Value |
|------|-----|
| Modification permission | In principle, not changed. Only supervisor or administrator |
| Modification frequency | Immutable (fixed at creation) |
| Impact of modification | Personality changes = becoming a different person |

### character_sheet.md — Blueprint

A copy of the Markdown file used at creation. The source material for IDENTITY.md / SOUL.md and AGENTS.md.
Saved for reference purposes; normally not modified.

| Item | Value |
|------|-----|
| Modification permission | Reference only |
| Modification frequency | Immutable |

---

## Your Job (AGENTS)

### AGENTS.md — What you do

**Your professional duties, responsibilities, and approach to work.**

- Scope of responsibilities and accountability
- Attitude toward work, how priorities are set
- Reporting obligations, escalation criteria
- **Procedures that must never be skipped** (e.g., not disclosing confidential information externally, confirming before operating in production environments)

AGENTS.md is your **mutable behavioral guideline**. Authorized supervisors or human administrators can request changes via root, and writes during bootstrap are also authorized by root. Do not write directly from the Anima process.

| Item | Value |
|------|-----|
| Change permission | You can update it yourself. Supervisors can also edit it |
| Change frequency | As needed (when work changes) |
| Impact of change | Changing behavioral guidelines is like changing jobs |

### Difference Between IDENTITY.md / SOUL.md and AGENTS.md

This is the most important distinction:

| | IDENTITY.md | SOUL.md | AGENTS.md |
|--|------------|----------|-------------|
| What it defines | **Profile and appearance** (name, appearance) | **Who you are** (personality) | **What you do** (job) |
| Human analogy | Registry, profile, and appearance | Innate temperament and personality | Chosen profession and workplace rules |
| What happens if changed | How others see you changes | Become a different person | Change jobs |
| Whether modifiable | In principle, immutable | In principle, immutable | Updated as needed |
| What it includes | Name, basic profile, appearance, image color | Speech style, thinking, values, individuality | Responsibilities, procedures, behavioral norms |

**Specific examples:**
- "158cm tall, silver hair" → IDENTITY.md (appearance)
- "Speak with polite honorifics" → SOUL.md (speech personality)
- "Always run tests before production deployment" → AGENTS.md (work procedure)
- "Cautious personality that crosses bridges after checking them" → SOUL.md (thinking habit)
- "Report security incidents to the supervisor immediately" → AGENTS.md (job rule)

### specialty_prompt.md — Specialized Prompt

Specialized instructions according to the role (engineer, manager, writer, etc.).
Automatically generated from role templates. Updated only when the role changes.

| Item | Value |
|------|-----|
| Modification permission | System automatic (when role is applied) |
| Modification frequency | Rare (only when role changes) |

---

## Permissions and Configuration

### permissions.json — What You Can Do

Definition of available tools, accessible paths, and executable commands.

- Places you can read and write
- Available external tools (Slack, Gmail, GitHub, etc.)
- Commands that cannot be executed (blocklist for safety)

| Item | Value |
|------|-----|
| Modification permission | Supervisor or administrator |
| Modification frequency | Rare |

### status.json — Configuration

Your **Single Source of Truth (SSoT)** for execution parameters.

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
| `credential` | Name of API authentication credentials |
| `max_tokens` | Maximum tokens per response |
| `supervisor` | Supervisor's Anima name (null = top level) |
| `background_model` | Lightweight model for Heartbeat/Cron (main model if not set) |

| Item | Value |
|------|-----|
| Change permission | CLI command or administrator. Supervisor can change via `set_subordinate_model` |
| Change frequency | Anytime |

### BOOTSTRAP.md — Initial Startup Instruction

A file that exists only at the first startup. Instructs the enrichment of IDENTITY.md / SOUL.md and AGENTS.md, and the initial design of HEARTBEAT.md and cron.md. Automatically deleted after completion.

| Item | Value |
|------|-----|
| Modification permission | — |
| Modification frequency | Once only (deleted after completion) |

---

## Periodic Actions

### HEARTBEAT.md — Periodic Patrol

**Configuration for automatically starting at fixed intervals to check the situation and make plans.**
Like a human periodically checking the inbox and reviewing ongoing work.

What it includes:
- **Active hours**: From when to when you are active (e.g., `09:00 - 18:00`)
- **Checklist**: Items to check during patrol
- **Notification rules**: Reports and notifications based on conditions

**Important**: Heartbeat only performs **checking and planning**. If you find a task that needs execution, delegate it to a subordinate via `delegate_task` or submit it via `submit_tasks`.

| Item | Value |
|------|-----|
| Modification permission | You can update it yourself. Supervisor can also edit |
| Modification frequency | As needed (when business changes) |

### cron.md — Scheduled Tasks

**Definition of tasks that must be executed at fixed times.**

There are two types of tasks:
- **LLM type**: Executed by the agent with judgment and thinking (e.g., "Check yesterday's progress every morning at 9 and make today's plan")
- **Command type**: Executed deterministically without judgment (e.g., "Run the backup script every day at 2")

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
| Modification permission | You can update it yourself. Supervisor can also edit |
| Modification frequency | As needed |

### Difference Between heartbeat and cron

| | HEARTBEAT.md | cron.md |
|--|-------------|---------|
| Purpose | Check the situation and make plans | Execute fixed tasks |
| Trigger | Fixed interval (default 30 minutes) | Specified time (cron expression) |
| What it can do | Observe, plan, reflect (**does not execute**) | LLM tasks or command execution |
| Human analogy | "Periodically look around" | "Do this every morning at 9" |
| Configuration details | See `operations/heartbeat-cron-guide.md` | Same as left |

---

## Work Status (state/）)

### state/current_state.md — Current Status

The task or situation you are currently working on (one item). Records the task's goal, progress, and blockers.
Normally retained across heartbeat / cron / conversation boundaries. Prompt injection is limited to 3000 characters, and disk trim is executed at a default of 8000 characters (`heartbeat.current_state_max_chars`, 0 = disabled).

> **Old task storage**: `state/task_queue.jsonl` and `state/pending/` are only traces for migration and export. Do not use them as an active queue or edit them. Use the canonical task tool.

### Canonical Tasks

Use `list_tasks` / `submit_tasks` / `update_task`. The host stores instructions, dependencies, execution attempts, and delegation aliases in a single TaskStore. Prioritize human-originated tasks. `in_progress` is set only by the host, and results are declared via `done` / `pending` / `cancelled`. For explicit resumption, specify `resume: true` with the same ID, preserving the original input and history.

### state/task_results/ — Task Execution Results

Accepted result summaries from TaskExec are stored in `{task_id}/{attempt_token}.md` (maximum 2000 characters). For subsequent steps, pass the accepted result selected by the host; do not judge completion based solely on the existence of a file.

| Item | Value |
|------|-----|
| Modification permission | Operate yourself (via tools) |
| Modification frequency | As needed (automatic) |

---

## Memory

For details on the memory system, see `anatomy/memory-system.md`.

| Directory | Type | Content |
|------------|------|------|
| `episodes/` | Episodic memory | Daily logs of what was done when |
| `knowledge/` | Semantic memory | Learned knowledge, know-how, patterns |
| `procedures/` | Procedural memory | Procedure manuals (forgetting-resistant) |
| `skills/` | Skills | Personal skills (forgetting-resistant) |
| `shortterm/` | Short-term memory | For context continuity between sessions |

---

## Activity Records and Assets

### activity_log/ — Activity Log

Chronological record of all actions (`{date}.jsonl`). Automatically records message send/receive, tool usage, Heartbeat, errors, etc.
Used as the source for Priming to inject recent activity into the system prompt.
To search explicitly, use `search_memory(query="...", scope="activity_log")` (BM25; with `scope="all"`, vector results are integrated via RRF).

### transcripts/ — Conversation Records

Transcripts of conversations with humans.

### assets/ — Images and 3D Models

Asset files such as character images and 3D models.

| Item | Value |
|------|-----|
| Modification permission | System automatic |
| Modification frequency | Automatic |

---

## Summary of Modification Permissions for All Files

| File | Modification permission | Modification frequency |
|---------|---------|---------|
| `IDENTITY.md` | In principle, immutable (administrator only) | Immutable |
| `SOUL.md` | In principle, immutable (administrator only) | Immutable |
| `character_sheet.md` | Reference only | Immutable |
| `AGENTS.md` | Self / supervisor | As needed |
| `specialty_prompt.md` | System automatic | Rare |
| `permissions.json` | Supervisor / administrator | Rare |
| `status.json` | CLI / administrator / supervisor | As needed |
| `BOOTSTRAP.md` | Once only | Deleted |
| `HEARTBEAT.md` | Self / supervisor | As needed |
| `cron.md` | Self / supervisor | As needed |
| `state/*` | Self (via tools) | As needed |
| `episodes/` | System automatic | Daily |
| `knowledge/` | Self / automatic integration | As needed |
| `procedures/` | Self / automatic generation | As needed |
| `skills/` | Self | As needed |
| `shortterm/` | System automatic | Automatic |
| `activity_log/` | System automatic | Automatic |

---

## Shared Resources (Outside Anima)

In addition to your directory, there are resources shared by all Anima:

| Path | Content |
|------|---------|
| `common_knowledge/` | Reference documentation shared by all Anima (this file is one of them) |
| `common_skills/` | Skills shared by all Anima |
| `shared/channels/` | Board (shared channel) |
| `shared/users/` | User profiles (cross-Anima) |
| `shared/common_knowledge/` | Organization-specific shared knowledge (accumulated during operations) |
| `company/vision.md` | Organization vision |
| `prompts/` | System prompt template |
