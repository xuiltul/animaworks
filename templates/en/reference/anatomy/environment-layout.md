## Runtime Data Directory

All runtime data is stored under `{data_dir}/`.

```
{data_dir}/
├── company/          # Company vision and policy (read-only)
├── animas/          # All Anima data
│   ├── {anima_name}/    # ← You
│   └── ...               # Other Anima
├── prompts/          # Prompt templates (character design guide, etc.)
├── vault.json        # Shared credential vault
├── shared/           # Shared area across Anima
│   ├── channels/     # Board channels (general.jsonl, ops.jsonl, etc.)
│   ├── credentials.json  # Legacy compatibility fallback
│   ├── inbox/        # Message inbox
│   └── users/        # Shared user memory (per-user subdirectories)
├── common_skills/    # Shared skills (read-only)
└── tmp/              # Working directory
    └── attachments/  # Message attachments
```

## Access Rules

1. **Your own directory** (`{data_dir}/animas/{anima_name}/`): Full read/write access
2. **Shared area** (`{data_dir}/shared/`): Read/write. Used for messaging and shared user memory
3. **Common skills** (`{data_dir}/common_skills/`): Only top-level members (no supervisor) can write. Others read-only. Skills available to all
4. **Company info** (`{data_dir}/company/`): Only top-level members can write
5. **Prompts** (`{data_dir}/prompts/`): Read-only. Templates such as character design guide
6. **Other Anima directories**: Access only as explicitly permitted in permissions.json
7. **Descendants' directories** (supervisors only — same permissions for children, grandchildren, great-grandchildren, etc.):
   - **Management files**: `injection.md`, `cron.md`, `heartbeat.md`, `status.json` are **read/write** (for organizational role assignments and configuration changes)
   - **State files**: `activity_log/` and `state/current_state.md` are **read-only**. Inspect subordinate tasks through authorized task tools; canonical task storage is host-owned and must not be directly edited.
   - **identity.md**: **read-only** (write-protected)
8. **Peers' activity_log**: You may read `activity_log/` of peers who share the same supervisor (for verification). Writing is not allowed
