# SDR — machine usage patterns

## Ground rules

1. **Write a plan first** — Never run machine with inline short instructions. Always pass a plan file
2. **Output is a draft** — Always verify machine output yourself before sending
3. **Save location**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (`/tmp/` forbidden)
4. **Rate limit**: chat 5/session, heartbeat 2
5. **Machine cannot access infrastructure** — Include lead info and nurturing context in the plan

---

## Overview

SDR uses machine in two scenarios:
- Drafting initial contact messages when leads are discovered
- Drafting follow-up emails for nurturing targets

---

## Phase 1: Lead outreach draft

### Step 1: Organize lead information

Compile discovered lead info: discovery source (SNS / inbound / event), BANT evaluation, profile and interests.

### Step 2: Run machine draft

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{outreach-request.md})" \
  -d /path/to/workspace
```

### Step 3: Verify and send

Self-check against `sdr/checklist.md`: tone, compliance (email regulation opt-in), personalization. Fix issues, then send.

## Phase 2: Nurturing email

### Step 4: Organize nurturing context

Compile: previous interactions, BANT changes, lead's response patterns.

### Step 5: Run machine draft

Feed nurturing context for follow-up email draft.

### Step 6: Verify and send

Self-check against `sdr/checklist.md`, then send.

---

## Lead report template (lead-report.md)

```markdown
# Lead report: {company or individual}

status: {new | qualified | disqualified | nurturing}
source: {inbound | outbound | sns | referral | event | other}
author: {anima-name}
discovered: {YYYY-MM-DD}

## BANT evaluation

| Item | Rating | Evidence |
|------|--------|----------|
| Budget | {yes / unknown / no} | {evidence} |
| Authority | {yes / unknown / no} | {evidence} |
| Need | {clear / latent / none} | {evidence} |
| Timeline | {specific / undecided / none} | {evidence} |

## Custom fields

{Add team-specific evaluation criteria on deployment}

## Lead summary

{Discovery context, interaction summary, notable points}

## Recommended action

{Proposal to Director: convert to deal / continue nurturing / pass / investigate further}
```

---

## Constraints

- Machine output MUST NOT be sent without self-check (NEVER)
- Messages with compliance concerns MUST be confirmed with Director before sending
- Emails without opt-out mechanism are forbidden (NEVER)
