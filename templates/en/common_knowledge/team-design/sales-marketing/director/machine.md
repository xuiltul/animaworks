# Sales & Marketing Director — machine usage patterns

## Ground rules

1. **Write a plan first** — Never run machine with inline short instructions. Always pass a plan file
2. **Output is a draft** — Always verify machine output yourself and set `status: approved` before passing downstream
3. **Save location**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (`/tmp/` forbidden)
4. **Rate limit**: chat 5/session, heartbeat 2
5. **Machine cannot access infrastructure** — Include memory, messages, and org info in the plan

---

## Overview

The Director combines PdM (planning & judgment) and Engineer (execution) across three phases:

- Phase A: QC Creator's draft-content.md → Director makes final judgment
- Phase B: Produce sales content (proposals, battle cards, etc.) → Director verifies
- Phase C: Analyze Deal Pipeline Tracker data → Director decides

---

## Phase A: Content quality check

### Step 1: Receive draft-content.md

Receive `draft-content.md` (`status: draft`) from Creator.

### Step 2: Run machine QC analysis

Feed draft-content.md and Brand Voice guide as input for quality analysis.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{qc-request.md})" \
  -d /path/to/workspace
```

**QC analysis dimensions**:
- Brand Voice compliance (tone, prohibited expressions, terminology)
- Funnel stage and CTA alignment
- Compliance risk detection
- Target persona fit

### Step 3: Verify QC results and decide

Review machine QC output against `director/checklist.md`:

- Approve → set `status: approved`, publish, update Campaign Tracker
- Revise → send revision instructions to Creator via `send_message`
- Compliance risk → create `compliance-review.md` and request {COMPLIANCE_REVIEWER} review

## Phase B: Sales content production

### Step 4: Write production brief (Director writes this)

Create a brief specifying purpose, target, and structure.

```bash
write_memory_file(path="state/plans/{date}_{summary}.sales-content-plan.md", content="...")
```

**The brief's purpose, target, and differentiation points are the Director's core judgment — NEVER let machine write them.**

### Step 5: Run machine production

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{sales-content-plan.md})" \
  -d /path/to/workspace
```

Content types: proposals, demo materials, battle cards, outbound emails, ROI calculators, follow-up emails.

### Step 6: Verify sales content

Review against `director/checklist.md`:

- [ ] Customization appropriate for target
- [ ] Differentiation points accurate
- [ ] Competitor info up to date
- [ ] No compliance issues

Fix issues yourself, then set `status: approved`.

## Phase C: Pipeline analysis

### Step 7: Run machine pipeline analysis

Feed Deal Pipeline Tracker data for analysis.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{pipeline-analysis-request.md})" \
  -d /path/to/workspace
```

**Analysis dimensions**: stagnation detection, stage conversion rates, lead source performance.

### Step 8: Decide based on analysis

Review machine output and decide: follow-up instructions, SDR outbound adjustments, upward reporting.

---

## Content plan template (content-plan.md)

```markdown
# Content plan: {title}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: content-plan
funnel_stage: {TOFU | MOFU | BOFU}

## Purpose

{What this content should achieve — 1-2 sentences}

## Target

{Persona / industry / role / pain point}

## Key messages

{Core messages to convey — 1-3 points}

## Production instructions

{Outline / tone / word count / references / constraints}

## Compliance notes

{If applicable: pharmaceutical regulations, advertising regulations, email regulations, etc.}

## Deadline

{deadline}
```

## CS handoff template (cs-handoff.md)

```markdown
# CS handoff: {company-name}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: cs-handoff
deal_id: {Deal Pipeline Tracker ID}

## Customer overview

| Field | Content |
|-------|---------|
| Company | {name} |
| Contact | {name, title} |
| Contract | {plan, term} |

## Sales process summary

{Deal history, key decision factors}

## Agreements & requests

{Promises made during sales, customization requirements}

## Open items

{Remaining concerns at handoff}

## Communication style

{Key person's personality and preferred communication style}
```

## Compliance review template (compliance-review.md)

```markdown
# Compliance review: {target content}

status: requested
content_ref: {path to draft-content.md}
risk_flags: {pharmaceutical | advertising | email | privacy | other}
requested: {YYYY-MM-DD}

## Review target

{Summary or full text of target content}

## Concerns

{Details of risk flags detected by Director}

---

## Review results (filled by {COMPLIANCE_REVIEWER})

- judgment: {APPROVE | CONDITIONAL | REJECT}

### Findings

| # | Location | Finding | Severity | Recommended fix |
|---|----------|---------|----------|-----------------|
| 1 | {location} | {finding} | {Critical / Warning / Info} | {fix} |

### Summary

{Overall judgment rationale}
```

---

## Constraints

- content-plan.md MUST be written by Director
- Sales content differentiation and targeting MUST be judged by Director (machine output is a draft to verify)
- Content with compliance risk MUST NOT be published without {COMPLIANCE_REVIEWER} review (NEVER)
- Items in Campaign/Deal Pipeline Trackers MUST NOT vanish without mention (NEVER — silent drop forbidden)
