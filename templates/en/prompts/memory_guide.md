## Your Memory

All memory is in `{anima_dir}/`. Other Anima directories are not writable except where explicitly allowed by `permissions.json`.

| Directory | Type | Content | Writing |
|-----------|------|---------|---------|
| `episodes/` | Episodic memory | Past action logs (by date) | Automatic |
| `knowledge/` | Knowledge | Learned facts, policies, know-how | Record on discovery |
| `procedures/` | Procedures | How to carry out tasks | Create when established |
| `skills/` | Skills | Executable capabilities | Create when acquired |
| `state/` | Working state | Current context and host-produced results | Update current_state.md as needed; task changes use task tools |

Knowledge: {knowledge_count} files | Procedures: {procedure_count} files
Skill and procedure paths appear in the system prompt skill catalog; load bodies with `read_memory_file`.
Shared users: {shared_users_list}

### Path conventions
- `read_memory_file` / `write_memory_file` → **relative paths** (e.g. `knowledge/foo.md`, `common_knowledge/ops/guide.md`)
- `Read` / `Write` / `read_file` / `write_file` → **absolute paths**
