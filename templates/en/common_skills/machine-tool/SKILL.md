---
name: machine-tool
description: >-
  Delegates work to external agent CLIs (machine tools) for large code changes, investigation, or analysis.
  Use when: offloading implementation via the machine command, heavy refactors, or batched agent runs.
tags: [machine, agent, external, delegation]
---

# Machine Tool

Delegate tasks to external agent CLIs.
Offload heavy work like code changes, investigation, and analysis to external agents.

## Design Philosophy

You are the **craftsperson**. The machine is a **machine tool** (CNC, laser cutter, etc.).
A machine tool can cut with incredible precision, but it doesn't decide what to build.
It has no memory, no communication, no identity.
**Your job is to provide precise blueprints (instructions).**

## CLI Usage

```bash
animaworks-tool machine run [options] "instruction" -d /path/to/workdir
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-e ENGINE` | Engine selection (omit for auto-selected default; use `-h` to list available engines) |
| `-d PATH` | Working directory (default: current directory) |
| `-t SECONDS` | Timeout in seconds (default: 600s sync, 1800s background) |
| `-m MODEL` | Model override (default: engine's default) |
| `--background` | Background execution (1800s timeout; output streams to `state/cmd_output/`) |
| `-j / --json` | Output result as JSON |

### Basic Examples

```bash
# Minimal (default engine, current directory)
animaworks-tool machine run "detailed instruction"

# Specify engine and directory
animaworks-tool machine run -e cursor-agent "instruction" -d /path/to/workdir

# Background execution
animaworks-tool machine run --background "instruction" -d /path/to/workdir

# Custom timeout
animaworks-tool machine run -t 300 "instruction" -d /path/to/workdir
```

## Writing Good Instructions (Important)

Vague instructions lead to poor results. Always include:

1. **Goal** — What to accomplish
2. **Target files/modules** — What to modify
3. **Constraints** — Coding conventions, API compatibility, etc.
4. **Expected output** — Code, report, diff, etc.

### Pass Long Instructions via File

Instructions containing Bash special characters (`|`, `` ` ``, `$`) will cause shell errors
if passed directly. Write to a file first:

```bash
# Write instruction to file
cat > /tmp/instruction.txt << 'INSTRUCTION'
## Task: PR #2087 Code Review

| Aspect | Check |
|--------|-------|
| Correctness | Meets issue requirements |
| Maintainability | Readability, tests, SRP |

Target file: `app/Services/Movacal/MovacalApiClient.php`
INSTRUCTION

# Read from file and execute
animaworks-tool machine run "$(cat /tmp/instruction.txt)" -d /path/to/workdir
```

## Parallel Execution (`--background`)

Use `--background` to run multiple machines simultaneously.
Output streams in real-time to `state/cmd_output/`.

### 3-Parallel Review Example

```bash
# Launch 3 review perspectives in parallel
animaworks-tool machine run --background "Correctness review..." -d /path &
animaworks-tool machine run --background "Maintainability review..." -d /path &
animaworks-tool machine run --background "Consistency review..." -d /path &
wait

# Check results (Read files from state/cmd_output/)
```

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
- Background output streams to `state/cmd_output/`, check with Read/Glob
