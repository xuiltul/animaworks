<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/anima-files.md -->
<!-- i18n: source-sha256=55b82aac73f8908843de4a8ce75c5ce2d5ac3323dfee2ee456625e5b71b0726d generated=2026-10-20 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Anima files

Each Anima has a directory at `~/.animaworks/animas/{name}/`. Personality, behavioral guidelines, and runtime configuration are stored separately in text and JSON, while information updated during a session is placed in `state/`.

## Definition and configuration

| File | Role |
|---|---|
| `IDENTITY.md` | Describes the persistent, externally presented identity, such as name, basic profile, and speaking style. |
| `SOUL.md` | Describes the values, principles, and inner character that form the basis of behavior. |
| `AGENTS.md` | Describes expertise, additional behavioral guidelines, and work instructions, incorporated into the system prompt. Replaces legacy `injection.md`. |
| `permissions.json` | Represents permissions for tools, commands, and file access for individual Anima instances. |
| `status.json` | Holds per-anima information resolved at runtime, such as enabled status, role, assignment, model, authentication, supervisor, and heartbeat activation. |
| `HEARTBEAT.md` | Describes heartbeat instructions and active time windows. The execution interval is determined by global configuration or the value of `status.json`, and is not set from the body of this file. |
| `BOOTSTRAP.md` | Describes self-introduction and initialization instructions for the first startup. |
| `cron.md` | Describes periodically executed tasks under each heading. The supervisor reads the schedule and execution content. |

`status.json` is the authoritative source for per-anima values related to ModelConfig. See the [configuration reference](../reference/config.md) for the list of corresponding keys and default values. `permissions.json` and the global `permissions.global.json` are used for permission decisions. See [security](../security.md) for permission boundaries and evaluation methods.

## Workspace files

Anima personality and work instructions use the same standard file names as OpenClaw and Hermes. Legacy names (`identity.md`, `injection.md`, `heartbeat.md`, `bootstrap.md`) are automatically switched to the new names by the startup migration, and only the new names are read and written afterward.

| Purpose | AnimaWorks | OpenClaw | Hermes |
|---|---|---|---|
| Public identity | `IDENTITY.md` | `IDENTITY.md` | `IDENTITY.md` (import) |
| Inner values and principles | `SOUL.md` | `SOUL.md` | `SOUL.md` (export concatenates IDENTITY then SOUL) |
| Working directory instructions | `AGENTS.md` | `AGENTS.md` | `AGENTS.md` |
| Periodic patrol instructions | `HEARTBEAT.md` | `HEARTBEAT.md` | `HEARTBEAT.md` |
| First-start instructions | `BOOTSTRAP.md` | — | — |

During migration, legacy `identity.md` is split by `split_legacy_identity()`. Headings such as name, profile, and speaking style go to `IDENTITY.md`; headings such as values, personality, and tone go to `SOUL.md`. Legacy `injection.md` maps to `AGENTS.md`, `heartbeat.md` to `HEARTBEAT.md`, and `bootstrap.md` to `BOOTSTRAP.md`.

### Migration and rollback

At server startup, the automatic migration in `core/infra/runtime_init.py` migrates every anima before anima discovery. The original legacy files and the pre-rewrite reference files are saved under each anima's `archive/standard_files_migration/<UTC timestamp>/`; `manifest.json` records the old-to-new mapping, sha256 values, created files, collisions, execution time, and version.

```bash
# Preview the rollback without changing anything
animaworks migrate --rollback-standard-files --dry-run

# Roll back every anima
animaworks migrate --rollback-standard-files

# Roll back one anima
animaworks migrate --rollback-standard-files --anima alice
```

Rollback moves the new standard files to `archive/standard_files_migration/<UTC timestamp>/rolled_back_new/`, restores the legacy files, and removes this step's entry from `migration_state.json`. If a new file was edited after migration, a warning is shown, but the edited content remains in `rolled_back_new/`. Run this command **before starting an older AnimaWorks version**.

### Export and import

```bash
# Export to OpenClaw / Hermes format
animaworks anima export alice --format openclaw --out ./alice-openclaw
animaworks anima export alice --format hermes --out ./alice-hermes

# Import OpenClaw / Hermes data
animaworks import openclaw --path ~/.openclaw --target-anima alice --apply
animaworks import hermes --path ~/.hermes --target-anima alice --apply
```

Export omits `permissions.json`, `status.json`, and `cron.md` because the standard formats have no counterpart, and records what was and was not exported in `ANIMAWORKS_EXPORT.md`. Secrets are redacted by `redact_credentials()`. When the import source contains `SOUL.md`, `IDENTITY.md`, `AGENTS.md`, or `HEARTBEAT.md`, the file is written directly if the same-name target does not exist. If it exists, it is not overwritten unless `--replace` is given; a draft is left in `state/migrations/drafts/` instead.

## Runtime Status

| Path | Role |
|---|---|
| `state/current_state.md` | Working state that holds ongoing work, decisions, next steps, and so on. It is incorporated into the prompt as needed, and updates are protected by a lock for the state file. |
| `state/task_queue.jsonl` | Migration input used when legacy task data remains. The source of truth for current tasks is the shared SQLite TaskStore; do not treat this file as a separate runtime queue. |
| `state/background_tasks/` | Stores the status and results of background processes started at `animaworks-tool submit`. Completion results are notified or referenced in conversations and by subsequent background runs. |
| `scratch/` | A disposable workspace. Child processes receive either a task-specific `scratch/task-<id>/` or a session-type-specific `scratch/<種類>/` as `ANIMAWORKS_SCRATCH_DIR`. Task-specific workspaces are deleted when the task is completed or canceled; all others are deleted by housekeeping after 24 hours of inactivity. |

Anima’s long-term memory, conversation history, skills, and other resources are organized into subdirectories by purpose. For details about memory, see the relevant sections of `docs/ja/memory/`.

## Design decisions

- **Memory is held in Markdown files.** AI can read and write them naturally, and they work well with grep. JSON is limited to configuration and status.
- **Archive-style memory is adopted.** Truncation-style memory that packs the latest N items into the prompt has a limit on memory capacity. With archive-style memory that searches and recalls what is needed, memory can continue to grow.
- **Permissions are a limitation of visibility.** Because there are things one does not know, one asks others. If everyone can see everything, the organization loses its meaning.
