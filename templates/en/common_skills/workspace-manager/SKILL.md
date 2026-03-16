---
name: workspace-manager
description: >-
  Register, list, remove, and assign workspaces (project directories).
  "workspace" "working directory" "project registration"
tags: [workspace, directory, project, management]
---

# Workspace Management

Skill for managing project directories (workspaces) where Animas perform work.

## Concept

An Anima normally lives in its "home" (~/.animaworks/animas/{name}/).
When working on a project, it "goes to the workplace" (workspace) to do the job.

Workspaces are stored in a shared registry (config.json `workspaces` section) and referenced by alias#hash.

## Aliases and Hashes

- Alias: A short name assigned by humans (e.g., `aischreiber`)
- Hash: The first 8 hex digits of the path's SHA-256, auto-generated (e.g., `3af4be6e`)
- Qualified form: `aischreiber#3af4be6e` — zero collision risk
- Tool arguments accept alias only, qualified form, hash only, or absolute path

## Operations

### Register

Add to the workspaces section of config.json:

1. `read_memory_file(path="config.json")` to check current settings
2. Add alias and path to the workspaces section
3. Save with `write_memory_file`

Or run via Bash:
```bash
python3 -c "
from core.workspace import register_workspace
result = register_workspace('alias_name', '/absolute/path/to/project')
print(result)
"
```

**Note**: Registration fails if the directory does not exist.

### List

Read the workspaces section of config.json:
```
read_memory_file(path="config.json")
```

### Remove

Delete the entry from the workspaces section of config.json.

### Change Your Default Workspace

Update the `default_workspace` field in your `status.json`:
1. `read_memory_file(path="status.json")` to check current content
2. Set `default_workspace` to an alias (e.g., `aischreiber`) or qualified form (e.g., `aischreiber#3af4be6e`)
3. `write_memory_file(path="status.json", content=...)` to save

Example:
```json
{
  "default_workspace": "aischreiber#3af4be6e"
}
```

### Assign to Subordinates (for Supervisors)

1. Register the workspace (see above)
2. Update the subordinate's `status.json` `default_workspace` field:
   - `read_memory_file(path="../{subordinate}/status.json")`
   - Set `default_workspace` to the alias
   - `write_memory_file(path="../{subordinate}/status.json", content=...)`

## Tool Usage

- **machine_run**: Specify alias or qualified form in `working_directory`
- **submit_tasks**: Specify alias in each task's `workspace` field
- **delegate_task**: Specify alias in the `workspace` field

## Notes

- Directories are validated both at registration and at resolution time
- Attempting to register a non-existent directory results in an error
- Overwriting an alias changes the hash, so old hash references will fail to resolve
- Humans don't need to remember hashes — alias alone is sufficient
