---
name: worker-management
description: >-
  AnimaWorks server process operations and management skill.
  Hot reload after code updates (server reload), Anima process restart,
  server status check (running Anima list, memory usage).
  "Reload", "Apply updates", "Reload config", "System status", "Server restart", "Check processes"
---

# Skill: System Management

## CLI Commands (recommended)

Use the `animaworks` CLI for per-Anima management. **Prefer CLI over direct API calls.**

```bash
# Restart individual Anima (e.g., after config changes)
animaworks anima restart <name>

# Status check (all or individual)
animaworks anima status
animaworks anima status <name>

# Change model (updates status.json + auto restart)
animaworks anima set-model <name> <model>

# Change role
animaworks anima set-role <name> <role>

# List Anima
animaworks anima list

# Disable / Enable
animaworks anima disable <name>
animaworks anima enable <name>

# Delete (--archive for backup)
animaworks anima delete <name>
```

### Common usage

```bash
# Restart specific Anima after config.json change
animaworks anima restart tsumugi

# Change model and auto restart
animaworks anima set-model tsumugi claude-sonnet-4-6
```

## API Reference (when CLI is unavailable)

Base URL: `http://localhost:18500`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/system/status` | GET | System status |
| `/api/system/reload` | POST | **Hot reload all anima** |
| `/api/animas` | GET | Anima list |
| `/api/animas/{name}` | GET | Anima details |
| `/api/animas/{name}/restart` | POST | Restart individual Anima |
| `/api/animas/{name}/stop` | POST | Stop individual Anima |
| `/api/animas/{name}/start` | POST | Start stopped Anima |
| `/api/animas/{name}/chat` | POST | Send message |
| `/api/animas/{name}/trigger` | POST | Trigger heartbeat immediately |

## Reload procedure (after code updates)

```bash
curl -s -X POST http://localhost:18500/api/system/reload | python3 -m json.tool
```

- `added`: Newly detected anima
- `refreshed`: Reloaded anima (file changes applied)
- `removed`: Anima removed from disk
- **No server restart needed. This endpoint reflects config and prompt changes immediately**

## Notes

- Stopping an Anima does not remove its data (memory, settings)
- **Do not stop yourself**
- Prefer CLI → API for per-Anima operations
