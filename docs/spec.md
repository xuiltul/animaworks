# Digital Anima Requirements Specification v1.0

## 1. Overview

Digital Anima is the minimal unit that encapsulates an AI agent as "a single person."

**Core Design Principles:**

- Internal state is invisible from the outside. The only external interface is text conversation
- Memory is "archive-based." The agent searches for and retrieves only the memories it needs, when it needs them
- Full context is never shared. Information is compressed and interpreted in the agent's own words before communicating
- Heartbeats enable proactive behavior rather than waiting for instructions
- Roles and principles are injected later. Digital Anima itself is an "empty vessel"

**Technical Direction:**

- Agent execution operates in **4 modes**: **A1** (Claude Agent SDK), **A1 Fallback** (Anthropic SDK direct), **A2** (LiteLLM + tool_use), **B** (one-shot assisted)
- Configuration is unified in **config.json** (Pydantic validation); memories are written in **Markdown**
- Multiple Animas operate collaboratively in a **hierarchical structure** (hierarchy defined by the `supervisor` field, synchronous delegation)

-----

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Digital Anima                      │
│                                                       │
│  Identity ──── Who I am (always resident)             │
│  Agent Core ── 4 execution modes                      │
│    ├ A1: Claude Agent SDK (Claude-only, autonomous)   │
│    ├ A1 Fallback: Anthropic SDK direct (when SDK      │
│    │  not installed)                                  │
│    ├ A2: LiteLLM + tool_use (GPT-4o, Gemini, etc.,   │
│    │  autonomous)                                     │
│    └ B:  One-shot assisted (Ollama, etc., FW-managed) │
│  Memory ───── Archive-based long-term memory          │
│    │           (autonomous search for recall)          │
│    ├ Conversation memory (rolling compression)        │
│    ├ Short-term memory (session continuity)           │
│    └ Unified activity log (JSONL timeline)            │
│  Boards ───── Slack-style shared channels             │
│  Permissions ─ Tool/file/command restrictions          │
│  Communication ─ Text + file references               │
│  Lifecycle ── Message receipt / heartbeat / cron       │
│  Injection ── Role/principles/behavior rules           │
│    │           (injected later)                        │
│                                                       │
└──────────────────────────────────────────────────────┘
        ▲                       │
   Text (incoming)         Text (outgoing)
```

-----

## 3. File Structure

```
animaworks/
├── core/
│   ├── anima.py               # DigitalAnima class
│   ├── agent.py               # AgentCore (execution mode selection, cycle management)
│   ├── anima_factory.py       # Anima creation (template/blank/MD)
│   ├── init.py                # Runtime initialization
│   ├── schemas.py             # Data models (Message, CycleResult, etc.)
│   ├── paths.py               # Path resolution
│   ├── messenger.py           # Inter-Anima message send/receive
│   ├── lifecycle.py           # Heartbeat/cron management (APScheduler)
│   ├── outbound.py            # Outbound message routing
│   ├── schedule_parser.py     # Schedule notation parser
│   ├── logging_config.py      # Log configuration
│   ├── memory/                # Memory subsystem
│   │   ├── manager.py         #   Archive-based memory search/write
│   │   ├── conversation.py    #   Conversation memory (rolling compression)
│   │   ├── shortterm.py       #   Short-term memory (session continuity)
│   │   ├── activity.py        #   Unified activity log (JSONL timeline)
│   │   ├── streaming_journal.py #  Streaming journal (WAL)
│   │   ├── priming.py         #   Automatic recall layer
│   │   ├── consolidation.py   #   Memory consolidation (daily/weekly)
│   │   └── forgetting.py      #   Active forgetting (3 stages)
│   ├── config/                # Configuration management
│   │   ├── models.py          #   Pydantic unified config model
│   │   ├── cli.py             #   config subcommand
│   │   └── migrate.py         #   Legacy config migration
│   ├── prompt/                # Prompt and context management
│   │   ├── builder.py         #   System prompt construction (24 sections)
│   │   └── context.py         #   Context window tracking
│   ├── tooling/               # Tool infrastructure
│   │   ├── handler.py         #   Tool execution dispatch, permission checks
│   │   ├── schemas.py         #   Tool schema definitions
│   │   ├── guide.py           #   Dynamic tool guide generation
│   │   └── dispatch.py        #   External tool routing
│   ├── execution/             # Execution engines
│   │   ├── base.py            #   BaseExecutor ABC
│   │   ├── agent_sdk.py       #   Mode A1: Claude Agent SDK
│   │   ├── anthropic_fallback.py # Mode A1 Fallback: Anthropic SDK direct
│   │   ├── litellm_loop.py    #   Mode A2: LiteLLM + tool_use
│   │   ├── assisted.py        #   Mode B: Framework-assisted
│   │   └── _session.py        #   Session continuity and chaining
│   └── tools/                 # External tool implementations
│       ├── web_search.py, x_search.py, slack.py
│       ├── chatwork.py, gmail.py, github.py
│       ├── transcribe.py, aws_collector.py
│       ├── image_gen.py       #   Image and 3D model generation
│       └── local_llm.py
├── cli/                       # CLI package
│   ├── parser.py              #   argparse definitions + cli_main()
│   └── commands/              #   Subcommand implementations
├── server/
│   ├── app.py                 # FastAPI application
│   ├── routes/                # API routes (split by domain)
│   │   ├── animas.py, chat.py, sessions.py
│   │   ├── memory_routes.py, logs_routes.py
│   │   ├── channels.py       #   Board/shared channel and DM history API
│   │   ├── system.py, assets.py, config_routes.py
│   │   ├── webhooks.py        #   External messaging webhooks
│   │   └── websocket_route.py
│   ├── websocket.py           # WebSocket management
│   └── static/                # Web UI
│       ├── index.html         # Dashboard
│       ├── modules/           # JS modules (activity, animas, api, app,
│       │                      #   chat, history, login, memory, router,
│       │                      #   state, status, touch, websocket)
│       └── workspace/         # Interactive Workspace
├── templates/
│   ├── prompts/               # Prompt templates
│   ├── anima_templates/       # Anima scaffolding (_blank)
│   ├── roles/                 # Role templates (engineer, researcher, manager, writer, ops, general)
│   └── company/               # Organization vision templates
├── main.py                    # CLI entry point
└── tests/                     # Test suite
```

### 3.1 Anima Directory (`~/.animaworks/animas/{name}/`)

Each Anima is composed of the following files and directories:

|File / Directory            |Description                         |
|----------------------------|------------------------------------|
|`identity.md`               |Personality and strengths (immutable baseline)|
|`injection.md`              |Role, principles, behavior rules (replaceable)|
|`permissions.md`            |Tool/file/command permissions        |
|`heartbeat.md`              |Periodic check interval and active hours|
|`cron.md`                   |Scheduled tasks (YAML)              |
|`bootstrap.md`              |Self-construction instructions on first launch|
|`status.json`               |Enabled/disabled, role, model settings|
|`specialty_prompt.md`       |Role-specific specialized prompt     |
|`assets/`                   |Character images and 3D models       |
|`transcripts/`              |Conversation transcripts             |
|`skills/`                   |Personal skills (YAML frontmatter + Markdown body)|
|`activity_log/`             |Unified activity log (daily JSONL)   |
|`state/`                    |Working memory (current_task, pending)|
|`episodes/`                 |Episodic memory (daily logs)         |
|`knowledge/`                |Semantic memory (learned knowledge)  |
|`procedures/`               |Procedural memory (runbooks)         |
|`shortterm/`                |Short-term memory (session continuity)|

### 3.2 config.json (Unified Configuration)

All settings are consolidated in `~/.animaworks/config.json`. Validated with the Pydantic `AnimaWorksConfig` model, with support for per-person overrides.

**Top-Level Structure:**

|Section                |Description                          |
|-----------------------|-------------------------------------|
|`system`               |Operation mode, log level            |
|`credentials`          |Per-provider API keys and endpoints (named map)|
|`model_modes`          |Model name to execution mode (A1/A2/B) custom mapping|
|`anima_defaults`       |Default values applied to all Animas |
|`animas`               |Per-Anima overrides (unspecified fields fall back to defaults)|
|`consolidation`        |Memory consolidation settings (daily/weekly execution times and thresholds)|
|`rag`                  |RAG settings (embedding model, graph spreading activation, etc.)|
|`priming`              |Automatic recall settings (per-message-type token budgets)|
|`image_gen`            |Image generation settings (style consistency, Vibe Transfer)|
|`human_notification`   |Human notification settings (channels: Slack/LINE/Telegram/Chatwork/ntfy)|
|`server`               |Server runtime settings (IPC, keep-alive, streaming)|
|`external_messaging`   |External messaging integration (Slack Socket Mode, Chatwork Webhook)|
|`background_task`      |Background tool execution settings (target tools, thresholds)|

**AnimaModelConfig Fields:**

|Field                                |Type            |Default                     |Description                      |
|-------------------------------------|----------------|----------------------------|---------------------------------|
|`model`                              |`str`           |`claude-sonnet-4-20250514`  |Model name to use (bare name, no provider prefix)|
|`fallback_model`                     |`str \| null`   |`null`                      |Fallback model                   |
|`max_tokens`                         |`int`           |`4096`                      |Maximum tokens per response      |
|`max_turns`                          |`int`           |`20`                        |Maximum turns per cycle          |
|`credential`                         |`str`           |`"anthropic"`               |Credential name to use           |
|`context_threshold`                  |`float`         |`0.50`                      |Threshold for short-term memory externalization (context usage ratio)|
|`max_chains`                         |`int`           |`2`                         |Maximum automatic session continuations|
|`conversation_history_threshold`     |`float`         |`0.30`                      |Compression trigger for conversation memory (context usage ratio)|
|`execution_mode`                     |`str \| null`   |`null` (auto-detect)        |`"autonomous"` or `"assisted"`   |
|`supervisor`                         |`str \| null`   |`null`                      |Name of the supervisory Anima    |
|`speciality`                         |`str \| null`   |`null`                      |Free-text area of expertise      |

**config.json Example:**

```json
{
  "version": 1,
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "", "base_url": null },
    "ollama": { "api_key": "dummy", "base_url": "http://localhost:11434/v1" }
  },
  "anima_defaults": {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "max_turns": 20,
    "credential": "anthropic",
    "context_threshold": 0.50,
    "conversation_history_threshold": 0.30
  },
  "animas": {
    "alice": {},
    "bob": { "model": "gpt-4o", "credential": "openai", "supervisor": "alice" }
  }
}
```

**Security:** config.json is saved with `0o600` permissions (owner read/write only). API key management via environment variables is also supported.

**RAGConfig Fields:**

|Field                          |Type      |Default                           |Description                        |
|-------------------------------|----------|----------------------------------|-----------------------------------|
|`enabled`                      |`bool`    |`true`                            |Enable/disable RAG functionality   |
|`embedding_model`              |`str`     |`intfloat/multilingual-e5-small`  |Embedding model to use             |
|`use_gpu`                      |`bool`    |`false`                           |Whether to use GPU                 |
|`enable_spreading_activation`  |`bool`    |`true`                            |Enable/disable graph-based spreading activation|
|`max_graph_hops`               |`int`     |`2`                               |Maximum hops for graph traversal   |
|`enable_file_watcher`          |`bool`    |`true`                            |Enable/disable file change monitoring|
|`graph_cache_enabled`          |`bool`    |`true`                            |Enable/disable graph cache         |
|`implicit_link_threshold`      |`float`   |`0.75`                            |Similarity threshold for implicit link generation|
|`spreading_memory_types`       |`list`    |`["knowledge", "episodes"]`       |Memory types targeted by spreading activation|

**PrimingConfig Fields:**

|Field                 |Type    |Default |Description                        |
|----------------------|--------|--------|-----------------------------------|
|`dynamic_budget`      |`bool`  |`true`  |Enable/disable dynamic budget allocation|
|`budget_greeting`     |`int`   |`500`   |Token budget for greeting messages |
|`budget_question`     |`int`   |`1500`  |Token budget for question messages |
|`budget_request`      |`int`   |`3000`  |Token budget for request messages  |
|`budget_heartbeat`    |`int`   |`200`   |Token budget for heartbeat         |

**ServerConfig Fields:**

|Field                          |Type     |Default |Description                        |
|-------------------------------|---------|--------|-----------------------------------|
|`ipc_stream_timeout`           |`int`    |`60`    |IPC streaming per-chunk timeout (seconds)|
|`keepalive_interval`           |`int`    |`30`    |Keep-alive send interval (seconds) |
|`max_streaming_duration`       |`int`    |`1800`  |Maximum streaming duration (seconds)|
|`stream_checkpoint_enabled`    |`bool`   |`true`  |Save tool results during streaming |
|`stream_retry_max`             |`int`    |`3`     |Maximum auto-retry count on stream disconnection|
|`stream_retry_delay_s`         |`float`  |`5.0`   |Wait time between retries (seconds)|

**ExternalMessagingConfig Fields:**

|Field                 |Type          |Default    |Description                        |
|----------------------|--------------|-----------|-----------------------------------|
|`preferred_channel`   |`str`         |`"slack"`  |Preferred send channel (`"slack"` or `"chatwork"`)|
|`user_aliases`        |`dict`        |`{}`       |User alias to contact info mapping |
|`slack`               |`object`      |           |Slack settings (enabled, mode, anima_mapping)|
|`chatwork`            |`object`      |           |Chatwork settings (enabled, mode, anima_mapping)|

**BackgroundTaskConfig Fields:**

|Field                     |Type    |Default |Description                        |
|--------------------------|--------|--------|-----------------------------------|
|`enabled`                 |`bool`  |`true`  |Enable/disable background execution|
|`eligible_tools`          |`dict`  |        |Map of target tool names to thresholds (threshold_s)|
|`result_retention_hours`  |`int`   |`24`    |Result retention period (hours)    |

-----

## 4. Memory System (Archive-Based)

### 4.1 Design Philosophy

Traditional AI agents mechanically truncate memory and pack it into prompts (truncation-based). This is equivalent to "anterograde amnesia where only the most recent memories exist."

The archive-based approach is different. Just as a person retrieves the documents they need from an archive, **Digital Anima searches for and retrieves only the memories it needs, when it needs them.** There is no upper limit on memory capacity. Only "what is needed right now" enters the context.

### 4.2 Correspondence with Neuroscience Models

```
┌─────────────────────────────────────────────────┐
│  Working Memory (Prefrontal Cortex)             │
│  = Context Window                               │
│  Limited capacity. Temporary holding of          │
│  "what I'm currently thinking about"             │
│  → Delegated to the SDK. No additional           │
│    implementation needed                         │
└──────────────────┬──────────────────────────────┘
                    │ Recall (search) / Encoding (write)
┌──────────────────┴──────────────────────────────┐
│  Long-Term Memory (Cerebral Cortex /             │
│  Hippocampal System)                             │
│                                                    │
│  episodes/   Episodic memory — what happened     │
│              and when                             │
│  knowledge/  Semantic memory — lessons and        │
│              knowledge learned                    │
│  procedures/ Procedural memory — work runbooks    │
└────────────────────────────────────────────────┘
```

### 4.3 Role of Memory Directories

|Directory                     |Brain Analog              |Contents              |Update Method                |
|------------------------------|--------------------------|----------------------|-----------------------------|
|`state/`                      |Persistent part of working memory|Current state, incomplete tasks|Overwritten each cycle       |
|`state/conversation.json`     |Conversation memory       |Rolling conversation history|LLM summarization when threshold exceeded|
|`shortterm/`                  |Short-term memory (session continuity)|Context carry-over    |Auto-externalized on session switch|
|`episodes/`                   |Episodic memory (hippocampus)|Daily action logs     |Appended to date-based files |
|`knowledge/`                  |Semantic memory (temporal cortex)|Lessons, rules, counterpart traits|Created/updated by topic     |
|`procedures/`                 |Procedural memory (basal ganglia)|Work runbooks         |Revised as needed            |

### 4.4 Memory Operations

**Recall (remembering)** --- Always search the archive before making decisions.

1. Search `knowledge/` by keywords (counterpart name, topic, etc.)
1. Search `episodes/` as needed (what happened in the past)
1. Check `procedures/` if the procedure is unclear
1. Load relevant memories before making a decision

**Encoding (writing)** --- Update memories after taking action.

1. Append an action log to `episodes/YYYY-MM-DD.md`
1. Write to `knowledge/` if something new was learned
1. Protect important lessons with the `[IMPORTANT]` tag
1. Update `state/current_task.md`

**Consolidation (reflection)** --- Transfer from episodic to semantic memory. Corresponds to memory consolidation during sleep in neuroscience.

- Extract patterns from `episodes/` logs and generalize them into `knowledge/`
- Run periodically via heartbeat or cron

### 4.5 Episodic Memory Format

```markdown
# 2026-02-12 Action Log

## 09:15 Chatwork Unreplied Check
- Trigger: Heartbeat
- Decision: Found 2 unreplied messages. Handle internal ones myself, escalate external ones
- Result: Drafted replies and obtained approval
- Lesson: None

## 14:30 Reply to Tanaka-san
- Trigger: Message received
- Decision: Recalled previous rejection for casual tone → drafted formal version
- Result: Approved
- Lesson: Reconfirmed that the approach for the construction industry is correct
```

### 4.6 Knowledge Format

```markdown
# Response Guidelines

## Communication Rules
- [IMPORTANT] Always respond with formal business language
- Casual tone is not acceptable (rejected on 2026-02-11)
- The construction industry values formal communication

## Contacts
- Primary contact: Tanaka-san
```

### 4.7 Experimental Validation Results

Archive-based memory achieved S-rank on all 5 criteria in manual testing (conducted 2026-02-12).

- **Recall**: Successfully searched for and utilized past memories not present in the prompt
- **Encoding**: Properly wrote action logs, lessons, and new knowledge to files
- **Reflexion**: Extracted lessons from rejection (failure) and changed subsequent decisions
- **Consolidation**: Extracted meta-patterns from individual episodes and generalized them as knowledge
- **Restoration**: Restored state from `state/` and `episodes/` after context was cleared

The key to success was a strong system prompt instruction: "Making decisions without searching memory is prohibited."

### 4.8 Conversation Memory (ConversationMemory)

Rolling chat history. When the accumulated size exceeds `conversation_history_threshold` (default 30%), older turns are compressed via LLM summarization while recent turns retain their original text. Saved in `state/conversation.json`.

### 4.9 Short-Term Memory (ShortTermMemory)

Externalized memory for session continuity. When the context threshold is exceeded in A2 mode, `session_state.json` (machine-readable) and `session_state.md` (for next prompt injection) are generated. Automatically archived to `shortterm/archive/` (up to 100 entries).

### 4.10 Unified Activity Log (ActivityLogger)

An append-only JSONL log that records all interactions as a single timeline.

- **File location**: `{anima_dir}/activity_log/{date}.jsonl`
- **Event types**: `message_received`, `response_sent`, `channel_read`, `channel_post`, `dm_received`, `dm_sent`, `human_notify`, `tool_use`, `heartbeat_start`, `heartbeat_end`, `cron_executed`, `memory_write`, `error`
- **Entry fields**: `ts` (ISO 8601), `type`, `content`, `summary`, `from`/`to`, `channel`, `tool`, `via`, `meta`
- **Priming integration**: Injects recent activity into the system prompt, enabling cross-session context continuity
- **Purpose**: The single data source for the Priming layer's "recent activity" channel. Unifies previously scattered transcript, dm_log, and heartbeat_history files

### 4.11 Streaming Journal (StreamingJournal)

A crash-resilient Write-Ahead Log (WAL). Incrementally persists streaming output to disk.

- **File location**: `{anima_dir}/shortterm/streaming_journal.jsonl`
- **Lifecycle**: `open()` → `write_text()` / `write_tool_*()` → `finalize()` (file deleted on normal completion)
- **Crash recovery**: On next startup, `recover()` reads orphaned journals and restores them as `JournalRecovery`
- **Buffering**: Flushed at 1-second intervals or 500 characters, with persistence guaranteed via fsync
- **Event types**: `start` (trigger and session info), `text` (text chunks), `tool_start` / `tool_end`, `done`

### 4.12 RAG Configuration

The embedding model is selectable via `rag.embedding_model` in `config.json`. Default is `intfloat/multilingual-e5-small` (384 dimensions). GPU usage and graph-based spreading activation are also configured in config.json.

-----

## 5. Identity (Self-Definition)

The information that allows a Digital Anima to recognize "who it is." Always resident in working memory.

```markdown
# Identity: Tanaka

## Personality Traits
- Cautious, considers risks first
- Detail-oriented, dislikes ambiguity

## Perspective
Prioritizes technical feasibility. "Will it actually work?" is always the starting point for decisions.

## Strengths
- Backend design, performance optimization

## Weaknesses
- UI/UX design decisions, understanding users' emotional needs
```

-----

## 6. Permissions

Restrictions on "what a Digital Anima can do." Permission restrictions create "limited visibility," which produces dependence on others --- and that dependence is what makes the organization valuable.

```markdown
# Permissions: Tanaka

## Available Tools
Read, Write, Edit, Bash, Grep, Glob

## Unavailable Tools
WebSearch, WebFetch

## Readable Locations
- Under /project/src/backend/
- Under /project/docs/
- Under /shared/reports/

## Writable Locations
- Under /project/src/backend/
- Under /workspace/Tanaka/

## Invisible Locations
- /project/.env
- Under /project/src/frontend/ (Suzuki's jurisdiction)

## Allowed Commands
npm test, npm run build, git diff, git log

## Disallowed Commands
git push (requires approval), rm -rf, docker
```

Since the frontend code is unreadable, the agent needs to ask a colleague, "What are the frontend constraints?" This "asking because I don't know" is what drives horizontal communication in the organization.

-----

## 7. Communication

### Principles

- Text and file references only. Direct sharing of internal state is prohibited
- Compress and interpret information in your own words before communicating. Never send the full context
- For lengthy content, save it as a file and communicate: "I've placed it here, please take a look"

### Message Structure

```json
{
  "id": "20260213_100000_abc",
  "thread_id": "",
  "reply_to": "",
  "from_person": "Tanaka",
  "to_person": "Suzuki",
  "type": "message",
  "content": "I've revised the auth API design. I placed it in auth-api-design.md, please review.",
  "attachments": [],
  "timestamp": "2026-02-13T10:00:00Z"
}
```

### Message Types

In the current implementation, the `type` field defaults to the single type `"message"`. The following type classifications are retained in the design for future extension.

|type (future extension) |Description          |
|------------------------|---------------------|
|request                 |Request or instruction from a superior|
|report                  |Report to a superior |
|consultation            |Consultation with a peer|
|broadcast               |Organization-wide announcement|

Suzuki sees only the design document. Tanaka's thought process and discarded alternatives are invisible. This information asymmetry is what enables fresh perspectives from different backgrounds.

-----

## 8. Lifecycle

### 8.1 Activation Triggers

A Digital Anima has its own internal clock. All three triggers belong to the "individual."

|Trigger          |Description                        |
|-----------------|-----------------------------------|
|Message received |Activated when a message arrives from another|
|Heartbeat        |Periodically checks the situation. Does nothing if there is nothing to do|
|cron             |Executes predetermined tasks at predetermined times on its own clock|

### 8.2 Heartbeat

The act of periodically "looking up and scanning the surroundings" at regular intervals. Executes while retaining the main context; does nothing if there is nothing to address.

```markdown
# Heartbeat: Tanaka

## Execution Interval
Every 30 minutes

## Active Hours
9:00 - 22:00 (JST)

## Checklist
- Are there unread messages in the inbox?
- Has a blocker arisen for any in-progress task?
- Have new files been placed in my work area?
- If nothing, do nothing (HEARTBEAT_OK)

## Notification Rules
- Notify relevant parties only when deemed urgent
- Do not repeat the same notification within 24 hours
```

### 8.3 cron

Performs predetermined tasks at predetermined times on its own clock. Unlike heartbeat, it always executes something and produces a result.

cron does not depend on external schedulers or organizational structure. **Each Digital Anima owns its own cron.** Just as a person has their own habit of writing a diary every morning.

```markdown
# Cron: Tanaka

## Morning Work Planning (Daily 9:00 JST)
Review yesterday's progress from long-term memory and plan today's tasks.
Prioritize based on principles and objectives.
Output the result to /workspace/Tanaka/daily-plan.md.

## Weekly Retrospective (Every Friday 17:00 JST)
Re-read this week's episodes/ and extract patterns, consolidating them into knowledge/.
(Memory consolidation = the neuroscience analog of memory fixation during sleep)
```

**Differences Between Heartbeat and cron:**

|Aspect      |Heartbeat              |cron                   |
|------------|-----------------------|-----------------------|
|Human analog|Occasionally checking email while working|Morning routine, weekly retrospective|
|Context     |Retained               |Not retained (new session)|
|Decision    |"Is there anything I should care about?"|Executed unconditionally|
|When idle   |Does nothing           |Always produces output |
|Ownership   |Internal to the individual|Internal to the individual|

### 8.4 Flow of a Single Cycle

```
Activation (message or heartbeat or cron)
  ↓
Recall: Search the archive for relevant memories
  ↓
Think & Act: Agent Core (A1/A2/B mode) processes
  ↓
Communicate: Summarize results and send as text or create files
  ↓
Encode: Write action logs, lessons, and knowledge
  ↓
Update state: Update state/
  ↓
Rest
```

-----

## 9. Injectable Slot (Post-Injection)

Digital Anima is an "empty vessel." Roles and principles are injected via Markdown.

```markdown
# Injection: Tanaka

## Role
Tech Lead. Responsible for technical decision-making and code review.
Area of responsibility: backend architecture.

## Principles
Solve user problems through high-quality software.

## Code of Conduct
- Never compromise on quality
- Pursue simplicity
- When in doubt, return to "What is best for the user?"

## Things Not To Do
- Direct access to the production database
- Frontend implementation (delegate to Suzuki)
- Pushing to the main branch without approval
```

-----

## 10. System Prompt Construction

The various Markdown files and templates are combined to build a single system prompt. `build_system_prompt()` in `core/prompt/builder.py` assembles 24 sections in order.

```
System Prompt =
   1. environment (guardrails, folder structure)
   2. bootstrap (first-launch instructions — conditional)
   3. company/vision.md (organization vision)
   4. identity.md (who you are)
   5. injection.md (role, principles)
   6. specialty_prompt.md (role-specific specialized prompt — conditional)
   7. permissions.md (what you can do)
   8. state/current_task.md + pending.md (in-progress and pending tasks)
   9. Recent activity summary (retrieved from ActivityLogger)
  10. priming (RAG automatic recall — per-message-type budget)
  11. memory_guide (memory directory guide + file listings)
  12. common_knowledge (shared reference hints — conditional)
  13. Matched skill full-text injection (description-based matching + within budget)
  14. Unmatched personal skills (table format)
  15. Unmatched common skills (table format)
  16. Hiring rules (only when newstaff skill is present — conditional)
  17. External tool guide (only when permitted — conditional)
  18. A2 reflection (self-correction prompt for A2 mode)
  19. emotion metadata (facial expression metadata instructions)
  20. hiring context (only when no other Animas exist — conditional)
  21. behavior_rules (search before deciding)
  22. org context (organization structure — supervisor/subordinates/peers)
  23. messaging instructions (message send/receive + peer Anima list)
  24. human notification guidance (top-level Anima + only when notifications enabled — conditional)
```

Each section is separated by `---`, and conditional sections (bootstrap, tools_guide, etc.) are injected only when applicable.

Skill injection performs description-based matching against the message content and injects the full text of matched skills within budget (trigger-based skill injection). The budget is determined by message type: greeting=1000, question=3000, request=5000, heartbeat=2000.

Including "Making decisions without searching memory is prohibited" in `behavior_rules` is the key to the success of archive-based memory (validated experimentally).

-----

## 11. Implemented Features

- **Digital Anima class** --- Encapsulation and autonomous operation. 1 Anima = 1 directory
- **4 execution modes** --- A1: Claude Agent SDK / A1 Fallback: Anthropic SDK direct / A2: LiteLLM + tool_use / B: Assisted (one-shot)
- **Archive-based memory** --- episodes (daily logs) / knowledge (lessons and knowledge) / procedures (runbooks) / state (working memory)
- **Conversation memory** --- Rolling compression. Older turns compressed via LLM summarization when threshold exceeded
- **Short-term memory** --- Session continuity. Externalized as JSON+MD when context threshold exceeded
- **Unified activity log** --- All interactions recorded as JSONL timeline. Priming integration for cross-session context continuity
- **Streaming journal (WAL)** --- Crash resilience. Incremental persistence of text and tool results
- **Board/shared channels** --- Slack-style shared channels. REST API for channel posts, mentions, and DM history
- **Unified outbound routing** --- Auto-resolves recipient names to internal Anima or external platforms (Slack/Chatwork) for delivery
- **Heartbeat and cron** --- Schedule management via APScheduler. Japanese schedule notation support
- **Inter-Anima messaging** --- Text communication via Messenger. Hierarchical delegation (supervisor → subordinate synchronous delegation)
- **Unified configuration** --- config.json + Pydantic validation. Per-person overrides
- **FastAPI server** --- REST + WebSocket + Web UI (3D office, conversation view)
- **10 external tools** --- web_search, slack, chatwork, gmail, github, x_search, transcribe, aws_collector, local_llm, image_gen
- **Anima creation** --- From template / blank (_blank) / MD file
- **Trigger-based skill injection** --- Description-based matching from message content. Full text of matched skills injected within budget
- **External messaging integration** --- Slack Socket Mode (real-time bidirectional), Chatwork Webhook (inbound)
- **Embedding model configuration** --- RAG embedding model selectable via config.json
- **A1 Fallback executor** --- When Claude Agent SDK is not installed, executes tool_use loop directly via the Anthropic SDK

-----

## 12. Design Decision Log

|Decision                               |Rationale                                                           |
|---------------------------------------|--------------------------------------------------------------------|
|Memory format: JSON → Markdown files   |Experiments showed that AI reads and writes Markdown more naturally, with better Grep search compatibility|
|Forgetting: score-based → [IMPORTANT] tag + consolidation|Simple tag-based approach is more practical. Consolidation naturally organizes importance|
|config.md → config.json                |From per-anima MD to unified JSON. Pydantic validation + per-anima overrides|
|Do not build the agent loop ourselves  |Delegate to the Claude Agent SDK. No reinventing the wheel          |
|4-way execution mode branching         |Claude SDK first priority, Anthropic SDK fallback, LiteLLM for general use, Assisted for weak models. All within the Anima capsule|
|agent.py refactoring                   |1848 lines → 465 lines. Split into execution/, tool_handler, tool_schemas|
|Permissions as "limited visibility"    |Not knowing things forces asking others. Omniscience makes organizations meaningless|
|Archive-based memory adopted           |Truncation-based (packing the last N entries into the prompt) does not scale memory. Archive-based has no upper limit on memory capacity|
|cron as the "individual's" internal clock|cron is not an organizational scheduler; each Digital Anima owns its own habits. Just as a person has their own daily routine|
