# Organization Structure

Organization structure in AnimaWorks is built dynamically from `config.json`.
This doc explains how it is defined, interpreted, and displayed.

## Hierarchy via supervisor

Hierarchy is defined only by each Anima's `supervisor` field.

- `supervisor: null` or unset → Top-level Anima
- `supervisor: "alice"` → alice is the supervisor

Example in config.json:

```json
{
  "animas": {
    "alice": {
      "supervisor": null,
      "speciality": "Strategy & overall"
    },
    "bob": {
      "supervisor": "alice",
      "speciality": "Dev lead"
    },
    "carol": {
      "supervisor": "alice",
      "speciality": "Design & UX"
    },
    "dave": {
      "supervisor": "bob",
      "speciality": "Backend dev"
    }
  }
}
```

Resulting structure:

```
alice (Strategy & overall)
├── bob (Dev lead)
│   └── dave (Backend dev)
└── carol (Design & UX)
```

Rules:
- `supervisor` must be an Anima in `animas`
- No cycles (e.g. alice → bob → alice)
- Each Anima has at most one supervisor

## How Org Context Is Built

`core/prompt/builder.py` `_build_org_context()` derives from config.json:

1. **Supervisor**: Your `supervisor` value. If unset: "You are top-level"
2. **Subordinates**: Anima whose `supervisor` is your name
3. **Peers**: Anima with the same `supervisor` (excluding yourself)

This appears in the system prompt as your org position:

```
## Your Org Position

Your speciality: Dev lead

Supervisor: alice (Strategy & overall)
Subordinates: dave (Backend dev)
Peers (same supervisor): carol (Design & UX)
```

## How to Read Your Position

From the "Your org position" section:

| Item | Meaning | Impact |
|------|---------|--------|
| Your speciality | `speciality` | Questions and decisions in this area are your responsibility |
| Supervisor | Your report target | Reports, escalations go here |
| Subordinates | Your reports | Tasks, status checks |
| Peers | Same supervisor | Direct coordination on shared work |

### What to Check

- If supervisor is "(none — you are top-level)", you are top-level
- If subordinates is "(none)", you do execution work yourself
- If peers exist, you can coordinate directly with them

## Changes and Reload

To change org structure:

1. Edit `config.json` `animas` (supervisor, speciality)
2. Restart the server (`animaworks start`)
3. New org context is built on next start (message, Heartbeat, cron)

Notes:
- Changes apply only after restart (old context until then)
- Adding/removing Anima also requires restart
- SHOULD notify affected Anima of org changes

## Example Org Patterns

### Pattern 1: Flat Org

Everyone top-level:

```json
{
  "animas": {
    "alice": { "supervisor": null, "speciality": "Planning" },
    "bob":   { "supervisor": null, "speciality": "Dev" },
    "carol": { "supervisor": null, "speciality": "Design" }
  }
}
```

```
alice (Planning)
bob (Dev)
carol (Design)
```

- Everyone is equal and can message directly
- Good for small teams or independent roles
- Peers: "(none)"

### Pattern 2: Hierarchical

Clear hierarchy:

```json
{
  "animas": {
    "alice": { "supervisor": null,    "speciality": "CEO" },
    "bob":   { "supervisor": "alice", "speciality": "Dev lead" },
    "carol": { "supervisor": "alice", "speciality": "Sales lead" },
    "dave":  { "supervisor": "bob",   "speciality": "Backend" },
    "eve":   { "supervisor": "bob",   "speciality": "Frontend" },
    "frank": { "supervisor": "carol", "speciality": "Customer support" }
  }
}
```

```
alice (CEO)
├── bob (Dev lead)
│   ├── dave (Backend)
│   └── eve (Frontend)
└── carol (Sales lead)
    └── frank (Customer support)
```

- bob and carol are peers (same supervisor = alice)
- dave and eve are peers (same supervisor = bob)
- dave to frank: dave → bob → alice → carol → frank (cross-dept rule)

### Pattern 3: Manager + Specialists

One manager, many specialists:

```json
{
  "animas": {
    "manager": { "supervisor": null,      "speciality": "Project mgmt" },
    "dev1":    { "supervisor": "manager", "speciality": "API dev" },
    "dev2":    { "supervisor": "manager", "speciality": "DB design" },
    "dev3":    { "supervisor": "manager", "speciality": "Infra" },
    "qa":      { "supervisor": "manager", "speciality": "QA" }
  }
}
```

```
manager (Project mgmt)
├── dev1 (API dev)
├── dev2 (DB design)
├── dev3 (Infra)
└── qa (QA)
```

- All peers, easy coordination
- manager owns task allocation and status
- Good for startups and project teams

## speciality Field

`speciality` is free text describing an Anima's focus.

- Shown next to the name in org context (e.g. `bob (Dev lead)`)
- Helps others decide who to contact or delegate to
- Unset shows as "(unset)"

Guidelines:
- Short and concrete: `Backend dev`, `Customer support`, `Data analysis`
- Avoid vague: `Various` → `Planning, coordination`
- Multiple areas: `UI design · Frontend dev`
