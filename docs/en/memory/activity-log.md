<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/activity-log.md -->
<!-- i18n: source-sha256=79f543a7c8e5b8a683d0ba6b2184927f6350431ed966bd2ffeea5a2991768cf6 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 193a5e72

# Activity Log

`core/memory/activity/` records Anima's messages, executions, memory operations, and more in chronological order. The public API is `core/memory/activity/logger.py`'s `ActivityLogger`, and records are appended to `{anima_dir}/activity_log/{date}.jsonl`.

## Entries

Each JSONL line is a single event. The basic fields are `ts` (timestamp), `type` (event type), `content`, `summary`, `from`, `to`, `channel`, `tool`, `meta`, `origin`, `origin_chain`, and `ctx`. Empty values are omitted at save time. Internally, `from_person` and `to_person` are used, and in JSON they are output as `from` and `to`, respectively.

There is no closed enum for event types; the caller records them according to the use case. Current representative examples are as follows.

| Category | Event Examples |
|---|---|
| Conversation and communication | `message_received`, `response_sent`, `message_sent`, `channel_post`, `channel_read`, `human_notify`, `human_reply` |
| Execution and status | `tool_use`, `tool_result`, `error`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `cron_executed`, `inbox_processing_start`, `inbox_processing_end` |
| Memory and integration | `memory_write`, `issue_resolved`, `knowledge_outcome`, `knowledge_reconsolidated`, `procedure_reconsolidated`, `consolidation_start`, `consolidation_end` |
| Tasks and skills | `task_created`, `task_updated`, `task_exec_start`, `task_exec_end`, `skill_auto_created`, `skill_autolearn_summary` |

Old `dm_sent` and `dm_received` are treated as `message_sent` and `message_received`, respectively, at load time.

## Streaming Journal

`core/memory/conversation/streaming_journal.py`'s `StreamingJournal` sequentially records text fragments and tool start and end events to `shortterm/streaming_journal_{session_type}.jsonl` during LLM response streaming. On normal completion, `done` is recorded to close the journal. If the process terminates abnormally, the response text and tool execution status can be recovered from the remaining records at the next startup. Writes occur when 500 characters are reached, or when 1 second has elapsed since the last flush.

## Retention and Rotation

`ActivityLogger` also provides read access for search, timeline display, and more. Log retention volume, duration, and rotation timing are managed via `activity_log` configuration, with execution handled by the supervisor. See the [configuration reference](../reference/config.md) for the list of configuration items.
