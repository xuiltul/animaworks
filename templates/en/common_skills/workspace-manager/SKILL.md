---
name: workspace-manager
description: >-
  Registers, lists, removes, and assigns workspaces (project directories) for Anima work.
  Use when: binding project paths to Anima, managing aliases, or switching workspace roots.
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

A top-level Anima with an explicit human instruction uses `grant_workspace_access`:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "make_default": true
}
```

This tool updates the shared workspace registry, adds the path to the target Anima's `permissions.json.file_roots`, and optionally updates `status.json.default_workspace`.

**Note**: Registration fails if the directory does not exist.
**Note**: `read_memory_file(path="config.json")` reads the Anima-local `config.json`. Do not use it for the shared workspace registry.

### List

Use `core.workspace.list_workspaces()` to inspect the shared registry. Do not use `read_memory_file(path="config.json")`.

### Remove

Treat removal as an administrator operation. For normal work, avoid overwriting an existing alias and register a new alias instead.

### Change Your Default Workspace

A top-level Anima sets `make_default: true` when calling `grant_workspace_access`.
Non-top-level Animas cannot grant themselves workspace access; ask the top-level Anima via a human instruction.

### Assign to Subordinates (for Supervisors)

A top-level Anima with an explicit human instruction can grant access to a subordinate or descendant by setting `target_anima`:

```json
{
  "alias": "finance-dashboard",
  "path": "/absolute/path/to/project",
  "target_anima": "ritsu",
  "make_default": true
}
```

## Tool Usage

- **machine_run**: Specify alias or qualified form in `working_directory`
- **submit_tasks**: Specify alias in each task's `workspace` field
- **delegate_task**: Specify alias in the `workspace` field

## Notes

- Directories are validated both at registration and at resolution time
- Attempting to register a non-existent directory results in an error
- Overwriting an alias changes the hash, so old hash references will fail to resolve
- Humans don't need to remember hashes — alias alone is sufficient
