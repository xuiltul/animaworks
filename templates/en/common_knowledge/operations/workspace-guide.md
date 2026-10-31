# Workspace Guide

An overview of the project directory (workspace) where Anima works and how to use it.

See `operations/scratch-and-worktrees.md` for where worktrees, builds, and temporary files are stored during work.

## What is a Workspace

### The Concept of Home and Workplace

Anima normally resides in its "home" (`~/.animaworks/animas/{name}/`).
This is where Anima-specific data such as IDENTITY.md / SOUL.md, memory, and configuration are stored.

When performing project work (code changes, research, builds, etc.),
Anima goes out to the "workplace" (workspace) to work.
A workspace is a directory containing the project's source code and deliverables.

### Registry and Aliases

Workspaces are registered in the organization-wide registry (the `workspaces` section of `config.json`).
They can be uniquely referenced by a short human-assigned name (alias) and the first 8 characters of the SHA-256 hash of the path.

| Format | Example | Use |
|------|-----|------|
| Alias only | `myproject` | Normal reference (use with hash when conflicts occur) |
| Full form | `myproject#3af4be6e` | Strict reference with zero conflicts |
| Hash only | `3af4be6e` | When the alias is unknown |
| Absolute path | `/home/user/dev/myproject` | Direct specification (works even if not registered) |

## Usage in Tools

### submit_tasks

When an alias is specified in the `workspace` field of each task,
TaskExec uses that workspace as the working directory.

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "コンパイル", "description": "...", "workspace": "myproject", "parallel": true}
])
```

### delegate_task

When an alias is specified in the `workspace` field,
the delegated subordinate works in that workspace.

```
delegate_task(name="aoi", instruction="API テストを実施して", workspace="myproject")
```

## Registration and Assignment

### Registration Procedure

Refer to the `common_skills/workspace-manager` skill for details.
Key points:

1. Add the alias and path to the `workspaces` section of `config.json`
2. Or call `core.org.workspace.register_workspace` from Python
3. The directory is checked for existence at registration time (an error occurs if it does not exist)

### Assignment to Subordinates

Supervisors update the `default_workspace` field of the subordinate's `status.json`
to assign the primary working directory. Refer to the `workspace-manager` skill.

## Common Issues

### Directory Does Not Exist

- **At registration**: An error occurs when trying to register a non-existent path
- **At use**: If the directory is deleted after registration, an error occurs during resolution
- **Remedy**: Verify the path and re-register with the correct absolute path

### Alias Not Found

- **Cause**: The alias is not registered in the registry, or there is a typo
- **Remedy**: Check the `workspaces` section with `read_memory_file(path="config.json")` and use the correct alias

### Hash Has Changed

- **Cause**: If the alias is overwritten and the path is changed, the hash also changes
- **Remedy**: If using the full form (`alias#hash`), update to the new hash. Using only the alias has no impact
