<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/lifecycle.md -->
<!-- i18n: source-sha256=d7c192cd9f539a0c0ddea724cc79608c33077c8e984c3fad725d8a2574a8a054 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Startup Triggers and Lifecycle

Anima is executed not only in response to requests from the server, but also triggered by messages and schedules. A root process remains resident to manage the scheduler and receive handling, while actual conversation and automated execution are offloaded to dedicated lanes in the task runner.

## Triggers and lanes

| Trigger | Processing performed |
|---|---|
| chat | Processes conversations with the user by thread. |
| inbox | Processes messages from other Anima or integrations in a dedicated inbox lane. |
| heartbeat | Performs periodic checks based on the instructions in `HEARTBEAT.md` using the heartbeat task runner. |
| cron | Executes scheduled tasks defined in `cron.md` using the cron task runner. |
| task | A background worker takes on executable tasks from TaskStore. |
| greet | Handles greetings after startup or at the start of a conversation using a dedicated chat task runner. |

Chat uses a per-thread conversation lock to prevent conflicts within the same conversation. Inbox and scheduled background work have separate control locks. TaskExec uses a dedicated worker slot and AgentCore, and does not share the execution lock for heartbeat/Cron. Updates to the file `state/current_state.md` use a process-safe state file lock.

## Heartbeat and cron

The default heartbeat interval is 30 minutes in the global configuration, and can be overridden per Anima in `status.json`'s `heartbeat_interval_minutes`. The valid range is 1–1440 minutes. The global `activity_level` defaults to 100%, with an allowed range of 10–400%; higher values shorten the effective interval. `activity_schedule` can be used to switch activity levels by time of day. Active hours are configured in `HEARTBEAT.md`.

`cron.md` is a definition per heading, with a standard five-field cron expression specified in `schedule`. Configure `type`, a description, and, if needed, command/tool and arguments; `core/runtime/schedule_parser.py` parses them. During setup and hot reload, only schedules that could not be parsed or registered are reported as `cron_health_*.md`. There is no periodic monitoring of the execution frequency or failures of registered cron jobs.

## First startup and one cycle

On first startup, `core/anima/bootstrap_state.py` checks the definition status of `IDENTITY.md` / `SOUL.md` and `AGENTS.md`, among others, and records bootstrap start, failure, and completion. The supervisor starts the process and begins normal processing after confirming that the IPC endpoint and dependent services are ready.

1. Map the trigger to a session type and lane, then acquire the necessary lock or worker slot.
2. Resolve `status.json` and common configuration, and prepare the prompt from memory, permissions, current status, messages, and other inputs.
3. The engine adapter appropriate to the execution mode mediates model calls and tool execution, then returns events to the root.
4. Finalize the response, conversation record, task status, and activity record, and save notifications needed for subsequent processing.
5. Release the lock / worker slot, returning the root scheduler and inbox dispatcher to a state where they can accept the next request.

## Design Decisions

- **Cron is each Anima's internal clock.** It is not an organization-wide scheduler; just as a person has their own daily routines, each Anima has its own habits.
