---
name: subordinate-management
description: >-
  Skill for process management: pausing and resuming subordinate Anima.
  "pause", "stop", "resume", "wake", "disable", "enable",
  "pause", "resume", "process management", "stop subordinate"
---

# Skill: Subordinate Pause and Resume Management

## Available Tools

| Tool | Purpose |
|------|---------|
| `disable_subordinate` | Pause subordinate (stop process + prevent auto-resume) |
| `enable_subordinate` | Resume paused subordinate |

## Important: disable_subordinate vs send_message

- **disable_subordinate**: Sets status.json to `enabled: false`. Reconciliation does not auto-resume. **Use this one**
- Sending "take a break" via send_message alone does **not** stop the process. Reconciliation will restart even after messaging

## Usage

### Pausing Multiple Subordinates

Call `disable_subordinate` for each one:

```
disable_subordinate(name="hinata", reason="Temporary pause due to reduced workload")
disable_subordinate(name="natsume", reason="Temporary pause due to reduced workload")
```

### Pausing All but One Subordinate

Pause everyone except the one to keep, using `disable_subordinate` for each.

### Resuming Paused Subordinates

```
enable_subordinate(name="hinata")
```

## Permissions

- Only **your direct subordinates** can be operated
- Subordinates of subordinates (grandchildren) cannot be operated directly. Ask their supervisor
- You cannot pause yourself
