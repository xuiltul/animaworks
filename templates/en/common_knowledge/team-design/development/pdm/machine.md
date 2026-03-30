# PdM — Machine Usage Patterns

## Base rules

1. **Write the plan first** — Running on inline short strings only is forbidden. Pass a plan file.
2. **Output is draft** — Always verify machine output yourself and set `status: approved` before the next phase.
3. **Storage**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Put memory, messaging, and org context in the plan.

---

## Overview

PdM **uses machine for investigation “hands and feet”; planning decisions stay with PdM**.

- Investigation / information gathering → delegate to machine
- Interpreting results, prioritization, implementation approach → PdM’s judgment
- Writing plan.md → PdM writes it

---

## Phase 1: Investigation

### Step 1: Write an investigation plan (PdM)

Create a plan that states clearly what machine should investigate.

```bash
write_memory_file(path="state/plans/{date}_{summary}.investigation.md", content="...")
```

### Step 2: Run investigation on machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{investigation_plan})" \
  -d /path/to/worktree
```

Machine returns findings as text. Save to `state/plans/{date}_{summary}.investigation.md` (`status: draft`).

### Step 3: Verify investigation results

PdM reads investigation.md and checks:

- [ ] The investigation goal is answered
- [ ] Facts vs assumptions are distinguished
- [ ] No important omissions
- [ ] Whether more investigation is needed

If issues exist, PdM fixes or supplements and sets `status: approved`.

## Phase 2: Plan creation

### Step 4: Create plan.md (PdM’s judgment)

From investigation.md, **PdM writes** plan.md.

“Implementation approach”, “priorities”, and “constraints” in plan.md are core PdM judgments — **never** have machine write them (NEVER).

### Step 5: Delegate

Confirm `status: approved` on plan.md, then pass to Engineer with `delegate_task`.

---

## Investigation plan template

```markdown
# Investigation plan: {problem / topic summary}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: investigation

## Goal

{What to clarify — 1–3 sentences}

## Scope

{Narrow scope — e.g. `core/memory/*.py` RAG rather than “whole repo”}

- {item1}
- {item2}

## Procedure

{Concrete steps for machine}

1. {step1}
2. {step2}
3. {step3}

## Out of scope

{What not to investigate — so machine does not drift}

- {exclude1}
- {exclude2}

## Decision criteria

{When to count as “issue” or “action needed”}

## Expected output

{Section structure and format — vague output is hard to reuse downstream}

Output in this form:
- Findings table: item / fact or assumption / impact
- Impact assessment
- Recommended actions
```

## Plan template (plan.md)

```markdown
# Plan: {task name}

status: draft
author: {anima_name}
date: {YYYY-MM-DD}
type: plan

## Goal

{1–3 sentences}

## Background / investigation summary

{Why this work — from investigation.md}

## Target files

- {file1}
- {file2}

## Current analysis

{Current code / problem structure}

## Implementation approach

{Step-by-step — PdM’s judgment}

1. {approach1}
2. {approach2}
3. {approach3}

## Constraints

{Coding standards, API compatibility, performance, etc.}

- {c1}
- {c2}

## Completion criteria

{Objectively verifiable — not “make it better” but “change X to Y”}

- {criterion1}
- {criterion2}

## Testing approach

{What to test, at high level}

## Risks

| Risk | Mitigation |
|------|------------|
| {r1} | {m1} |
| {r2} | {m2} |
```

---

## Constraints

- Investigation plan MUST be written by PdM.
- Judgment sections of plan.md (implementation approach, priorities, constraints) MUST be written by PdM.
- investigation.md may be machine-generated but stays draft until PdM verifies and approves.
- NEVER pass plan.md to Engineer without `status: approved`.
