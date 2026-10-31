<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/anima-files.md -->
<!-- i18n: source-sha256=55b82aac73f8908843de4a8ce75c5ce2d5ac3323dfee2ee456625e5b71b0726d generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Commit checked: b304b7dc

# Anima Files

Each Anima has a directory at `~/.animaworks/animas/{name}/`. Its personality, behavioral policies, and runtime configuration are stored separately as text and JSON, while information updated during a session is placed in `state/`.

## Definitions and Configuration

| File | Role |
|---|---|
| `IDENTITY.md` | Describes the enduring identity presented externally, including the name, basic profile, and speaking style. |
| `SOUL.md` | Describes the foundations of behavior, including values, principles, and inner personality. |
| `AGENTS.md` | Describes expertise, additional behavioral guidelines, and work instructions, and is incorporated into the system prompt. Formerly `injection.md`. |
| `permissions.json` | Specifies permissions for tools, commands, and file access for an individual Anima. |
| `status.json` | Holds per-anima information resolved at runtime, such as enabled status, role, responsibilities, model, authentication, supervisor, and heartbeat enablement. |
| `HEARTBEAT.md` | Describes heartbeat instructions and active hours. The execution interval is determined by the global configuration or the value in `status.json`; it is not configured in the body of this file. |
| `BOOTSTRAP.md` | Describes instructions for self-introduction and initialization at first startup. |
| `cron.md` | Describes periodically executed tasks under headings. The supervisor reads the schedule and execution details. |

`status.json` is the source of truth for per-anima values related to ModelConfig. See the [configuration reference](../reference/config.md) for a list of corresponding keys and default values. `permissions.json` and the global `permissions.global.json` are used to determine permissions. See [Security](../security.md) for permission boundaries and evaluation methods.

## Workspace Files

Anima’s personality and work instructions are managed using the same standard filenames as OpenClaw / Hermes. The old names (`identity.md`, `injection.md`, `heartbeat.md`, `bootstrap.md`) are automatically migrated to the new names when the server starts; thereafter, only the new names are read and written.

| Purpose | AnimaWorks | OpenClaw | Hermes |
|---|---|---|---|
| Public identity | `IDENTITY.md` | `IDENTITY.md` | `IDENTITY.md` (import) |
| Inner values and principles | `SOUL.md` | `SOUL.md` | `SOUL.md` (when exporting, IDENTITY and then SOUL are concatenated) |
| Working directory instructions | `AGENTS.md` | `AGENTS.md` | `AGENTS.md` |
| Periodic check-in instructions | `HEARTBEAT.md` | `HEARTBEAT.md` | `HEARTBEAT.md` |
| First-startup instructions | `BOOTSTRAP.md` | — | — |

The old `identity.md` is split into `split_legacy_identity()` during migration. Headings about names, profiles, and speaking style go into `IDENTITY.md`, while headings about values, personality, and tone go into `SOUL.md`. The old `injection.md` corresponds to `AGENTS.md`, the old `heartbeat.md` to `HEARTBEAT.md`, and the old `bootstrap.md` to `BOOTSTRAP.md`.

### Migration and Rollback

When the server starts, the automatic migration in `core/infra/runtime_init.py` migrates all animas before they are detected. The old files from before the changes and the reference files from before they were rewritten are saved in each anima’s `archive/standard_files_migration/<UTC時刻>/`. `manifest.json` records the old-name-to-new-name mappings, sha256 hashes, list of new files, conflicts, execution time, and version.

```bash
# 戻す内容を確認する（何も変更しない）
animaworks migrate --rollback-standard-files --dry-run

# 全 anima を戻す
animaworks migrate --rollback-standard-files

# 特定の anima だけ戻す
animaworks migrate --rollback-standard-files --anima alice
```

Rollback first backs up the new standard files to `archive/standard_files_migration/<UTC時刻>/rolled_back_new/`, then restores the old files and removes the record of this step’s application from `migration_state.json`. A warning appears if any new files were edited after migration, but those edits remain in `rolled_back_new/`. Run this command **before starting with an older version of AnimaWorks**.

### Export and Import

```bash
# OpenClaw / Hermes 形式へ書き出す
animaworks anima export alice --format openclaw --out ./alice-openclaw
animaworks anima export alice --format hermes --out ./alice-hermes

# OpenClaw / Hermes のデータを取り込む
animaworks import openclaw --path ~/.openclaw --target-anima alice --apply
animaworks import hermes --path ~/.hermes --target-anima alice --apply
```

Exports omit `permissions.json`, `status.json`, and `cron.md` because they have no equivalents in the standard format, and record in `ANIMAWORKS_EXPORT.md` what was and was not exported. Secret information is redacted in `redact_credentials()`. If the import source contains `SOUL.md`, `IDENTITY.md`, `AGENTS.md`, or `HEARTBEAT.md`, they are applied directly if there is no same-named new file. If a file already exists, it is not overwritten unless `--replace` is specified; instead, a draft is saved in `state/migrations/drafts/`.

## Runtime Status

| Path | Role |
|---|---|
| `state/current_state.md` | Working status containing ongoing work, decisions, and next steps. It is included in the prompt as needed, and updates are protected by a lock for the status file. |
| `state/task_queue.jsonl` | Migration input used when legacy-format task data remains. The source of truth for current tasks is the shared SQLite TaskStore; this file is not treated as a separate runtime queue. |
| `state/background_tasks/` | Stores the status and results of background processes started with `animaworks-tool submit`. Completion results are notified or referenced in conversations and subsequent background runs. |
| `scratch/` | A disposable workspace. Child processes receive `ANIMAWORKS_SCRATCH_DIR` as either a task-specific `scratch/task-<id>/` or a session-type-specific `scratch/<種類>/`. Task-specific workspaces are deleted when the task completes or is canceled; other workspaces are deleted by housekeeping after 24 hours of inactivity. |

Anima’s long-term memory, conversation history, skills, and other data are separated into subdirectories according to their purpose. See the relevant chapters in `docs/ja/memory/` for details about memory.

## Design Decisions

- **Memory is kept in Markdown files.** AI can read and write them naturally, and they work well with grep. JSON is limited to configuration and status.
- **An archive-style memory is used.** A truncation-based approach that packs the most recent N items into the prompt imposes a limit on memory size. With an archive-style approach that searches for and recalls what is needed, memory can continue to grow.
- **Permissions limit visibility.** When people do not know something, they ask others. An organization loses its meaning if everyone can see everything.
