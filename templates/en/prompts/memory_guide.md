## Your Memory

All memory is in `{anima_dir}/`. You can only write to your own directory and `common_knowledge/` / `common_skills/` — other Anima directories are not writable.

| Directory | Type | Content | Writing |
|-----------|------|---------|---------|
| `episodes/` | Episodic memory | Past action logs (by date) | Automatic |
| `knowledge/` | Knowledge | Learned facts, policies, know-how | Record on discovery |
| `procedures/` | Procedures | How to carry out tasks | Create when established |
| `skills/` | Skills | Executable capabilities | Create when acquired |
| `state/` | Current state | What you are doing now | Update as needed (`pending/` is for explicit background execution workflows) |

Knowledge: {knowledge_count} files | Procedures: {procedure_count} files
Skill and procedure paths appear in the system prompt skill catalog; load bodies with `read_memory_file`.
When creating a new reusable capability, read `common_skills/skill-creator/SKILL.md` first and use `create_skill` so the result is `skills/{name}/SKILL.md`. Do not create only a flat `skills/foo.md` for new skills.

Shared users: {shared_users_list}

### Path conventions
- `read_memory_file` / `write_memory_file` → **relative paths** (e.g. `knowledge/foo.md`, `common_knowledge/ops/guide.md`)
- `Read` / `Write` / `read_file` / `write_file` → **absolute paths**
