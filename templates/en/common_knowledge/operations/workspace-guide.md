# Workspace Guide

Concept and usage of project directories (workspaces) where Anima work.

## What is a Workspace

### Home vs Workplace

Anima normally "live at home" (`~/.animaworks/animas/{name}/`).
This is where identity, memory, configuration, and other Anima-specific data are stored.

When doing project work (code changes, investigation, builds, etc.),
they "go to the office" (workspace) to work.
A workspace is the directory where project source code and artifacts reside.

### Registry and Alias

Workspaces are registered in the organization-shared registry (`workspaces` section in `config.json`).
They can be uniquely referenced by a short name (alias) assigned by humans and the first 8 characters of the path's SHA-256 (hash).

| Form | Example | Use |
|------|---------|-----|
| Alias only | `aischreiber` | Normal reference (use hash form if collision) |
| Full form | `aischreiber#3af4be6e` | Strict reference with zero collision |
| Hash only | `3af4be6e` | When alias is unknown |
| Absolute path | `/home/main/dev/AI-Schreiber` | Direct specification (works even if unregistered) |

## Using with Tools

### machine_run (Machine Tool)

`working_directory` accepts alias, full form, hash, or absolute path.

```bash
animaworks-tool machine run "Refactor the code" -d aischreiber
animaworks-tool machine run "Run tests" -d aischreiber#3af4be6e
animaworks-tool machine run "Build" -d /home/main/dev/AI-Schreiber
```

### submit_tasks

Specify the alias in each task's `workspace` field.
TaskExec will use that workspace as the working directory.

```
submit_tasks(batch_id="build", tasks=[
  {"task_id": "t1", "title": "Compile", "description": "...", "workspace": "aischreiber", "parallel": true}
])
```

### delegate_task

Specify the alias in the `workspace` field.
The delegated subordinate will work in that workspace.

```
delegate_task(name="aoi", instruction="Run API test", deadline="2d", workspace="aischreiber")
```

## Registration and Assignment

### Registration Procedure

See the `common_skills/workspace-manager` skill for details.
Summary:

1. Add alias and path to the `workspaces` section in `config.json`
2. Or call `core.workspace.register_workspace` from Python
3. Directory existence is verified at registration (error if it does not exist)

### Assigning to Subordinates

Supervisors update the subordinate's `status.json` `default_workspace` field
to assign the primary working directory. See the `workspace-manager` skill.

## Common Issues

### Directory Does Not Exist

- **At registration**: Error when trying to register a non-existent path
- **At use**: Error at resolution if the directory was deleted after registration
- **Fix**: Verify the path and re-register with the correct absolute path

### Alias Not Found

- **Cause**: Alias not registered in the registry, or typo
- **Fix**: Check the `workspaces` section with `read_memory_file(path="config.json")` and use the correct alias

### Hash Changed

- **Cause**: Overwriting an alias with a different path changes the hash
- **Fix**: If using full form (`alias#hash`), update to the new hash. No impact if using alias only
