# Development Team Templates

**[日本語版](dev-team.ja.md)**

AnimaWorks ships a set of anima templates for running an in-house development team backed by
GitHub pull-request workflows. They are injected directly through the existing
`animaworks create <name> --template <template>` mechanism — no framework code changes are needed.

> These templates are a **starting point**. Memory, knowledge, and procedures grow through
> real operation. Treat them as scaffolding, not as a copy of anyone's operating history.

## Recommended team shape

A pragmatic default is **1 lead + 2–3 engineers + 1 researcher**:

| Template | Role | Purpose |
|----------|------|---------|
| `dev-lead` | manager | Product manager. Delegates to engineers/researchers, patrols open PRs, owns the quality gate and escalations |
| `dev-engineer` | engineer | Implementation in isolated worktrees, PR creation, monitoring and fixing own PR CI, completion reports |
| `dev-researcher` | researcher | Read-only investigation, evidence-based reports (conclusion → evidence → unverified) |

All roles are **optional**. If you prefer to start from a blank anima and cultivate it by hand,
use the default (`_blank`) template and grow the role incrementally — that remains the default choice.

## Creating a team

```bash
animaworks create alice --template dev-lead
animaworks create bob   --template dev-engineer
animaworks create carol --template dev-engineer
animaworks create dave  --template dev-researcher
```

Each command creates an anima with `identity.md` / `heartbeat.md` / `cron.md` /
`injection.md` / `permissions.json` / `status.json`. Individual names are given at creation time;
the template itself is role-based and engine-neutral (no model or engine name is hardcoded).

## Connecting to a GitHub pipeline

Development-team animas are meant to sit in front of a pull-request pipeline. The main
connection points are:

- **Webhook gateway** — inbound GitHub events (PR opened, review comment, CI result) enter via
  the webhook endpoints and become tasks for the team. See the [API reference](api-reference.md)
  for the webhook endpoints.
- **`gh-ci` / `gh-review` / `gh-comment` tasks** — the gateway dispatches each inbound event as
  one of these task types, serialized per pull request so two animas never work on the same PR at
  once. The lead's cron patrol acts as the safety net for anything the event flow misses.

Details of the pipeline itself are described in the existing reference docs; this page only
points to them and does not duplicate them.

## Notes

- The templates define discipline (delegation-first, worktree isolation, read-only research)
  but no domain knowledge. What each team actually learns lives in its own `knowledge/` and
  `episodes/`, grown during operation.
- Role guidelines referenced via `read_memory_file(...)` point into `common_knowledge/`, which
  is shipped with the templates.
