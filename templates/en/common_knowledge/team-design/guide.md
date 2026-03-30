# Team Design Guide — Core Principles

## Overview

This document defines core principles for designing Anima teams by purpose.
Separate from the framework’s organization mechanics (`organization/roles.md`, `organization/hierarchy-rules.md`),
it guides **what role mix to use for which goals**.

---

## Why Separate Functions

Role separation in AI agents differs from human teams for specific reasons:

| Reason | Explanation |
|--------|----------------|
| **Context isolation** | One agent doing everything inflates the context window and hurts judgment. Roles bound context. |
| **Deeper specialization** | Role-specific guidance, checklists, and memory raise quality over a generalist. |
| **Parallel execution** | Independent roles (e.g. review vs test) can run concurrently and increase throughput. |
| **Structural quality** | Splitting “do” and “verify” reduces self-review blind spots. |

---

## Design Principles

### 1. Single responsibility

Each role has one clear duty. Ambiguous roles produce unclear judgment and lower quality.

**Good**: “Run code review and judge quality” (Reviewer)  
**Bad**: “Implement, review, and test” (everything in one)

### 2. Separate execution and verification

Structure so the same Anima that uses machine also verifies its output. Never pass machine output downstream unverified.

```
Anima writes the brief → machine executes → Anima verifies and approves
```

### 3. Document-driven handoffs

Hand off between roles with status-bearing Markdown. Avoid handoffs in chat alone — information is lost.

```
plan.md (status: approved) → delegate_task to Engineer
```

### 4. Design for parallelism

Where roles are independent, design for parallel work. Do not serialize independent roles unnecessarily.

---

## Team Design Process

### Step 1: Define the goal

State in one sentence what the team must achieve.

Examples:
- “Deliver a software project end-to-end from planning through implementation and verification.”
- “Provide 24/7 first-line customer support.”

### Step 2: Decompose roles

List capabilities needed and split into roles using single responsibility.

Criteria:
- **Requires judgment?** → Often worth a dedicated role.
- **Independent of other work?** → Split to gain parallelism.
- **Needs deep expertise?** → Use a specialist role for quality.

### Step 3: Define responsibility boundaries

For each role, define MUST / SHOULD / MAY. Make adjacent boundaries explicit (“mine until here, theirs from there”).

### Step 4: Design the handoff chain

Document document flow between roles and where parallel execution is possible.

### Step 5: Pick role templates

Choose the closest framework `--role` (engineer, manager, writer, researcher, ops, general) and override with team design in `injection.md`.

---

## When to Combine Roles

Small teams or tight resources may have one Anima cover multiple roles.

### Combining is OK when

- **Task is small** — One person can keep quality across the flow.
- **Perspectives are close** — e.g. PdM + Engineer on a tightly coupled small change.
- **Cost** — Not enough work to justify dedicated Anima.

### Prefer separation when

- **Same person executes and verifies** — Avoid Engineer reviewing their own code.
- **Context clashes** — Frequent switching between review and implementation mindsets.
- **Parallelism pays** — You want review and test in parallel.

### If combining

Still **switch roles consciously**: “Now I am Reviewer”, “Now I am Engineer”.

---

## Scaling

| Scale | Composition | Typical use |
|-------|-------------|-------------|
| **Solo** | One Anima, all roles | Small tasks, prototypes |
| **Pair** | PdM + Engineer (review by Engineer) | Medium routine work |
| **Full team** | PdM + Engineer + Reviewer + Tester | Serious projects |
| **Scaled** | PdM + multiple Engineers + multiple Reviewers/Testers | Large / multi-module |

When to scale up:
- **High cost of failure** → More role separation.
- **Many parallelizable modules** → More Engineers.
- **High quality bar** → Independent Reviewer and Tester.

---

## Team Templates

| Template | Path | Summary |
|----------|------|---------|
| Development full team | `team-design/development/team.md` | Four roles: PdM + Engineer + Reviewer + Tester |

> To add a template, use the same layout (`team.md` + per-role directories) under `team-design/{team_name}/`.
