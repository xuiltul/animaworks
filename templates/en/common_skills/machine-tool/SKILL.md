---
name: machine-tool
description: >-
  Delegate tasks to external agent CLIs (machine tools). Offload heavy work like
  code changes, investigation, and analysis to claude/codex/cursor-agent/gemini.
  "machine" "machine tool" "external agent"
tags: [machine, agent, external, delegation]
---

# Machine Tool

Delegate tasks to external agent CLIs (claude, codex, cursor-agent, gemini).
Offload heavy work like code changes, investigation, and analysis to external agents.

## Design Philosophy

You are the **craftsperson**. The machine is a **machine tool** (CNC, laser cutter, etc.).
A machine tool can cut with incredible precision, but it doesn't decide what to build.
It has no memory, no communication, no identity.
**Your job is to provide precise blueprints (instructions).**

## CLI Usage (Mode S)

```bash
# Basic
animaworks-tool machine run "detailed instruction" -d /path/to/workdir

# Specify engine
animaworks-tool machine run -e cursor-agent "instruction" -d /path/to/workdir
animaworks-tool machine run -e claude "instruction" -d /path/to/workdir
animaworks-tool machine run -e gemini "instruction" -d /path/to/workdir

# Override model
animaworks-tool machine run -e claude -m claude-sonnet-4-6 "instruction" -d /path/to/workdir

# Background execution (results retrieved at next heartbeat)
animaworks-tool machine run --background "instruction" -d /path/to/workdir

# Timeout in seconds
animaworks-tool machine run -t 300 "instruction" -d /path/to/workdir
```

## use_tool Invocation (Mode A/B)

```json
{
  "tool": "use_tool",
  "arguments": {
    "tool_name": "machine",
    "action": "run",
    "args": {
      "engine": "cursor-agent",
      "instruction": "detailed instruction",
      "working_directory": "/path/to/workdir"
    }
  }
}
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| engine | YES | Engine name (cursor-agent, claude, codex, gemini) |
| instruction | YES | Task instruction. Specify goal, target, constraints, expected output |
| working_directory | YES | Absolute path to working directory |
| background | no | true for async execution (default: false) |
| model | no | Model override (engine default if omitted) |
| timeout | no | Timeout in seconds (sync: 600, async: 1800) |

## List Available Engines

To check available engines and priority:

```json
{"tool": "use_tool", "arguments": {"tool_name": "machine", "action": "run", "args": {"engine": "__list__", "instruction": "", "working_directory": ""}}}
```

## Writing Good Instructions (Important)

Vague instructions lead to poor results. Always include:

1. **Goal** — What to accomplish
2. **Target files/modules** — What to modify
3. **Constraints** — Coding conventions, API compatibility, etc.
4. **Expected output** — Code, report, diff, etc.

## When to Use

| Scenario | Suitable? |
|----------|-----------|
| Multi-file code changes | YES |
| Bug investigation / root cause analysis | YES |
| Test code generation | YES |
| Refactoring | YES |
| Short questions | NO (answer yourself) |
| Work requiring memory/messaging | NO (do it yourself) |

## Notes

- Machine tools have NO access to AnimaWorks infrastructure (no memory, messaging, or org info)
- Rate limited (5 per session, 2 per heartbeat)
- background=true results are written to `state/background_notifications/` and retrieved at next heartbeat
