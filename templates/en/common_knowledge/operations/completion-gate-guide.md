# completion_gate — Pre-Completion Verification Guide

A self-verification mechanism to ensure final answer quality.
Available across all execution modes (S/A/B/C/D/G).

## Overview

`completion_gate` makes the Anima execute the following checklist before delivering the final answer:

- [ ] Re-read the original instructions and confirmed each requirement is addressed
- [ ] Evidence exists from THIS session — not just assumption
- [ ] Nothing was simplified or omitted from what was requested
- [ ] An independent reviewer would accept this as complete

## Behavior by Mode

| Mode | Implementation | Enforcement |
|------|---------------|-------------|
| **S** (Agent SDK) | Stop hook blocks the first stop attempt and injects the checklist as `reason`. Second stop passes through | Automatic (no tool call needed) |
| **A** (LiteLLM) | `completion_gate` tool + marker file. One retry forced if not called | Automatic (1 retry) |
| **B** (Assisted) | Provided as a tool. No forced retry | Guidance only |
| **C/D/G** (Codex/Cursor/Gemini) | Tool provided via MCP | Guidance only |

## Trigger Applicability

| Trigger | Applied | Reason |
|---------|---------|--------|
| `chat` | ✅ | Ensure quality of human interaction |
| `task:*` | ✅ | Ensure task execution completion quality |
| `cron:*` | ✅ | Ensure scheduled task quality |
| `heartbeat` | ❌ | Observe/Plan/Reflect only; no final answer produced |
| `inbox:*` | ❌ | Lightweight replies only; verification overhead unnecessary |
| `consolidation:*` | ❌ | Excluded from tool list entirely |

## Usage in Mode A

In Mode A (LiteLLM), you are expected to call the `completion_gate` tool just before your final answer. If forgotten, a one-time reminder is shown.

```
completion_gate()
```

The checklist is returned. If no issues, proceed with the answer as-is. If issues are found, output only the corrected answer.

## Notes

- Do **not** narrate the verification process. Check internally; if no issues, stop without additional output
- In Mode S, no `completion_gate` tool call is needed (the Stop hook handles it automatically)
- Trust level is `trusted` (not subject to prompt injection defense)
