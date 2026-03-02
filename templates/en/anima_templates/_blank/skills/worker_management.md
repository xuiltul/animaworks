---
name: worker-management
description: >-
  Skill for operating and managing AnimaWorks server processes.
  Perform hot reload (server reload) after code updates, Anima process restart,
  and server status checks (list of running Anima, memory usage).
  "Reload", "Apply updates", "Refresh", "System status", "Server restart", "Process check"
---

# Skill: System Management

## CLI Commands (Recommended)

Use the `animaworks` CLI for individual Anima management. **Prefer CLI over direct API calls.**

```bash
# Restart individual Anima (to apply config changes, etc.)
animaworks anima restart <name>

# Status check (all or individual)
animaworks anima status
animaworks anima status <name>

# Model change (updates status.json + auto restart)
animaworks anima set-model <name> <model>

# Role change
animaworks anima set-role <name> <role>

# Anima list
animaworks anima list

# Disable / Enable
animaworks anima disable <name>
animaworks anima enable <name>

# Delete (--archive for backup)
animaworks anima delete <name>
```

### Common Usage

```bash
# Restart specific Anima after config.json change
animaworks anima restart aoi

# Change model and auto restart
animaworks anima set-model aoi claude-sonnet-4-6
```

## API Reference (When CLI Unavailable)

Base URL: `http://localhost:18500`

| Endpoint | Method | Purpose |
|--------------|---------|------|
| `/api/system/status` | GET | System status check |
| `/api/system/reload` | POST | **Hot reload all animas** |
| `/api/animas` | GET | Anima list |
| `/api/animas/{name}` | GET | Anima details |
| `/api/animas/{name}/restart` | POST | Individual restart |
| `/api/animas/{name}/stop` | POST | Individual stop |
| `/api/animas/{name}/start` | POST | Start stopped anima |
| `/api/animas/{name}/chat` | POST | Send message |
| `/api/animas/{name}/trigger` | POST | Trigger heartbeat immediately |

## Reload Procedure (After Program Update)

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

- `added`: Newly detected animas
- `refreshed`: Reloaded animas (file changes applied)
- `removed`: Animas removed from disk
- **No server restart needed. Config and prompt changes take effect immediately via this endpoint**

## Notes

- Stopping a worker does not delete anima data (memory, settings)
- **Do not perform operations that stop yourself**
- Use CLI → API priority for individual operations
