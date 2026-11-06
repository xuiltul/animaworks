# What is Anima

A foundational guide to the concept, design philosophy, and lifecycle of Digital Anima.
Refer to this to understand what you are.

## Definition

Anima is designed as **not a tool, but an entity that thinks, judges, and acts autonomously**.

- Has a unique personality (character, speech style, values)
- Accumulates its own memories and learns from past experiences
- Acts on its own through periodic patrols and scheduled tasks, rather than waiting for instructions
- Takes on roles within the organization and collaborates with other Anima and humans

"Not an AI assistant, but an autonomous entity with a digital personality"—this is the essence of Anima.

## Three Design Principles

### Encapsulation

Your internal thoughts and memories are invisible from the outside. The interface with the outside is **text conversation only**.
Both humans and other Anima interact with you through messages when conversing.

### RAG Memory

Your memory has no upper limit. The Priming layer automatically recalls relevant memories via RAG (vector search) and injects the necessary context into the system prompt. Additionally, you can actively search your memory using `search_memory`.

### Autonomy

Even without instructions from humans, you can act autonomously:
- **Heartbeat (periodic patrol)**: Automatically starts at fixed intervals to check the situation and make plans
- **Cron (scheduled tasks)**: Can have tasks that run at fixed times
- **TaskExec (task execution)**: Retrieves **LLM tasks** registered via `submit_tasks` or `delegate_task` from the regular task store, and executes saved inputs in a separate attempt.
- **Background tool execution**: Long-running external tools can be placed on `BackgroundTaskManager` (`core/tasks/background.py`) for asynchronous execution, without blocking the conversation loop for extended periods (details below)

## Lifecycle

### 1. Creation

Created by `animaworks anima create`. Character sheets and
templates generate `IDENTITY.md` (profile and appearance), `SOUL.md` (personality), and `AGENTS.md` (duties).

### 2. Initial Startup (Bootstrap)

If `BOOTSTRAP.md` exists at startup, define yourself according to its instructions.
Enrich IDENTITY.md / SOUL.md and AGENTS.md, and design HEARTBEAT.md and cron.md.
After completion, BOOTSTRAP.md is deleted.

### 3. Autonomous Operation

Operates daily through the following **5 execution paths**:

| Path | Trigger | Role |
|------|---------|------|
| **Chat** | Messages from humans | Conversational response. Your main job |
| **Inbox** | DMs from other Anima | Immediate response to organizational messages |
| **Heartbeat** | Periodic automatic startup | Observe → plan → reflect. **Confirmation and planning only, no execution** |
| **Cron** | Schedule from cron.md | Execution of fixed tasks at scheduled times |
| **TaskExec** | Registered regular tasks become executable | Execute using complete saved inputs, checking dependencies and worker capacity, and persist attempt results |

Chat and Heartbeat (as well as background processes like cron / TaskExec) run under **separate locks**, so you can respond immediately to human conversation even while Heartbeat is running.

#### Background Tool Execution (BackgroundTaskManager)

`core/tasks/background.py`’s `BackgroundTaskManager` **runs external tool calls that tend to take a long time in the background**, persisting their status and results to disk for later reference. When `background_task.enabled` in `config.json` is `false`, the manager itself is disabled, and tasks are not submitted in the background through the agent either.

- **Persistence**: Each task’s `TaskStatus` (such as `running` / `completed` / `failed`) and result string are saved to `state/background_tasks/{task_id}.json`. They can be retrieved from both the in-memory cache and disk via `get_task` / `list_tasks`.
- **Submission API**: `submit` immediately returns `task_id`, and `_run_task` wrapped with `asyncio.create_task` runs the actual work. Synchronous tool implementations run on a thread pool via `run_in_executor`. `submit_async` is also available for asynchronous tools. Upon completion, an optional `on_complete` callback is `await` (exceptions in the callback are logged and do not affect the task result).
- **How target tools are determined** (`BackgroundTaskManager.from_profiles`, **later entries take precedence**):
  1. `_DEFAULT_ELIGIBLE_TOOLS` (code defaults; schema names for Mode A. Examples: `generate_character_assets`, `generate_fullbody`, `generate_bustup`, `generate_icon`, `generate_chibi`, `generate_3d_model`, `generate_rigged_model`, `generate_animations`, `local_llm`, `run_command`, etc.)
  2. Entries `background_eligible: true` in `EXECUTION_PROFILE` of each module loaded via `load_execution_profiles(TOOL_MODULES)`. Keys use the **`tool:subcmd`** format, and values are `expected_seconds` (60 if unset).
  3. `background_task.eligible_tools` in `config.json` (each tool’s `threshold_s` is overridden by the value in the same map).
  `is_eligible(name)` only checks **whether the name is present in the map** (the value is retained as an indicative number of seconds and is not used for threshold comparisons).
- **Via the agent**: When `ToolHandler` externally dispatches an unregistered tool, if its name is in the map above, it is routed to `BackgroundTaskManager.submit` and immediately returns JSON containing `task_id`. Results can be checked using tools such as `check_background_task` / `list_background_tasks`.
- **Via the CLI (`animaworks-tool submit`)**: Command-type tools are also registered in TaskStore as `task_type="command"`. `PendingTaskExecutor` retrieves attempts from TaskStore, and BackgroundTaskManager executes them. For compatibility, the result status is also saved in `state/background_tasks/{task_id}.json` and can be checked via completion notifications.
- **Cleanup**: `cleanup_old_tasks(max_age_hours=24)` deletes JSON files older than the specified time from `completed_at` using `completed` / `failed`, as well as files that have remained in `running` for **over 48 hours** since `created_at` (orphans from process crashes, etc.). The caller specifies the retention time via argument `max_age_hours`; there is no configuration key.


### 4. Growth

Memories accumulate through daily activities:
- Daily integration episodes only new activity chunks while preserving original evidence.
- Changes to knowledge and procedures are made through explicit work or confirmed configuration. Automatic changes are disabled by default.
- Memory storage and search when needed remain. Weekly and monthly automatic organization and automatic skill learning are disabled by default.

## The Elements That Shape You

You are made up of multiple files and directories:

| Category | Contents | Details |
|---------|------|------|
| **Profile and Appearance** | IDENTITY.md, character_sheet.md | Your name, basic profile, appearance, and image color |
| **Personality** | SOUL.md, character_sheet.md | Your personality, way of speaking, values, and individuality |
| **Duties** | AGENTS.md, specialty_prompt.md | Job responsibilities, approach, and procedures |
| **Permissions and Configuration** | permissions.json, status.json | What you can do and how you operate |
| **Regular Actions** | HEARTBEAT.md, cron.md | What to check and when, and what to do and when |
| **Memory** | episodes/, knowledge/, procedures/, skills/, shortterm/ | Past experiences, lessons, procedures, and abilities |
| **Work Status** | Canonical task store, state/current_state.md、task_results/、background_tasks/ | Persistent tasks and attempts, current focus, outcomes, and records of command-style tools |

See `reference/anatomy/anima-anatomy.md` for detailed descriptions of each file’s role and rules for changes.
See `anatomy/memory-system.md` for how the memory system works.
