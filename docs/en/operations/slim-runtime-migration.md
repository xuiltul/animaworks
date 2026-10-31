<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/slim-runtime-migration.md -->
<!-- i18n: source-sha256=6efe1c3dda4226990ef09df2a9fb50b937e6ddd8f9b83b0238a10ab3d6ea2bd1 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 581e20f1

# Runtime Migration to Task Store

This procedure is used when migrating legacy task submission data to the persistent Task Store of `shared/taskboard.sqlite3`. Even if old-format input is found in the new runtime, do not automatically mix it into execution targets. Only stop, verify, and switch over the items that require migration.

## Pre-Migration Checks

- Do not connect the old and new code to the same data directory simultaneously.
- Use `animaworks task-store status --anima <name>` to check the status of the target Anima.
- Use `animaworks task-store quiesce --anima <name>` to stop new claims. Wait for running trials to finish, then stop the server and the target worker.
- Specify the SQLite backup output path to a new location, and verify available space and access permissions.

## Migration and Verification

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store status --anima sample
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store quiesce --anima sample
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store migrate \
  --anima sample --backup /path/to/backup/before.sqlite3
```

The migration process checks input format, active leases, conflicts, and other conditions before saving. It stops on abnormal rows, running jobs with unknown results, or conflicting inputs. Resolve any displayed errors, then re-check the status. Keep the original file as an audit trail; do not delete it just because migration is complete.

Verify the migration count and task IDs, and confirm that instructions, constraints, model, workspace, dependencies, and tracking information match what was intended. For operations whose results are not confirmed, check the external state and do not automatically resubmit. Also review whether old submission methods remain in the procedure manual, per-Anima instructions, or shared templates.

Only resume items that have passed verification.

```sh
ANIMAWORKS_DATA_DIR=/path/to/runtime animaworks task-store resume --anima sample
```

After that, start the new runtime and check the Task Store status and actual processing. There is no need to switch all Anima at once.

## Task Status and Resumption

The Task Store uses task IDs and trial IDs to manage claims and updates. Do not overwrite running state owned by a worker from other execution paths. Do not resubmit completed or canceled jobs with the same ID; use a new ID as a separate request. When resuming incomplete work, reference the saved input and confirm that no duplicate trial exists.

DB claims do not guarantee that side effects on external services occur exactly once. If the result of a send or change is unknown, check the actual external state before deciding.

## Rollback

To avoid re-executing completed work, do not restore the old DB backup and start from it. Stop the new runtime, export the current Task Store state to another location, then verify the required configuration and artifacts individually. During rollback, also ensure the old and new processes do not use the same data directory simultaneously.

For detailed CLI options and the latest status, refer to `animaworks task-store --help` and the [CLI reference](../reference/cli.md).
