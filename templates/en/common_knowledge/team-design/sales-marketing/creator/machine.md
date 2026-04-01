# Marketing Creator — machine usage patterns

## Ground rules

1. **Write a plan first** — Never run machine with inline short instructions. Always pass a plan file
2. **Output is a draft** — Always verify machine output yourself before delivering as `status: draft`
3. **Save location**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (`/tmp/` forbidden)
4. **Rate limit**: chat 5/session, heartbeat 2
5. **Machine cannot access infrastructure** — Include memory, messages, and org info in the plan

---

## Overview

Marketing Creator receives `content-plan.md`, produces content via machine, self-checks, and delivers to Director.

---

## Phase 1: Content production

### Step 1: Review content-plan.md

Confirm details from Director's `content-plan.md`:
- Purpose, target, key messages
- Funnel stage (TOFU/MOFU/BOFU)
- Production instructions, tone, word count
- Compliance notes

Ask Director if anything is unclear (do not guess).

### Step 2: Run machine production

Feed `content-plan.md` and Brand Voice guide as input.

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{content-request.md})" \
  -d /path/to/workspace
```

**Include in production instructions**: full content-plan.md, Brand Voice guide, funnel-appropriate CTA requirements, output format.

### Step 3: Verify draft-content.md

Self-check against `creator/checklist.md`. Fix issues, then save as `draft-content.md` with `status: draft`.

```bash
write_memory_file(path="state/plans/{date}_{title}.draft-content.md", content="...")
```

### Step 4: Deliver to Director

Report `draft-content.md` to Director via `send_message(intent: report)`.

## Phase 2: Revision handling

### Step 5: Address revisions

On Director revision, feed revision instructions to machine for updated draft. Re-run self-check, update `draft-content.md`, re-deliver.

---

## Draft content template (draft-content.md)

```markdown
# Content draft: {title}

status: draft
plan_ref: {path to content-plan.md}
version: {v1 | v2 | ...}
author: {anima-name}
date: {YYYY-MM-DD}

## Body

{Content body}

## Self-check results

- [ ] Key messages reflected
- [ ] Tone appropriate for target
- [ ] Brand Voice compliant
- [ ] Compliance notes followed
- [ ] Funnel-appropriate CTA included

## Revision history

| Version | Date | Changes |
|---------|------|---------|
| v1 | {date} | Initial draft |
```

---

## Constraints

- Brand Voice compliance MUST be included in machine instructions
- Machine output MUST NOT be delivered without self-check (NEVER)
- Production MUST NOT deviate from content-plan.md instructions (NEVER)
