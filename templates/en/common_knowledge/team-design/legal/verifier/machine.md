# Legal Verifier — machine usage patterns

## Ground rules

1. **Write the plan document first** — Do not run on a short inline prompt string. Pass a plan file.
2. **Output is draft** — Always validate machine output yourself before `status: approved` for the next step.
3. **Save location**: `state/plans/{YYYY-MM-DD}_{summary}.{type}.md` (no `/tmp/`)
4. **Rate limits**: chat 5/session, heartbeat 2
5. **Machine has no infra access** — Put memory, messaging, and org context into the plan document.

---

## Overview

Verifier **delegates verification scans to machine and validates that those results are sound (meta-review)**.

- Verification lens design → Verifier decides
- Diff detection and carry-forward cross-check → machine
- Soundness of findings → Verifier decides
- Worst-case scenarios → Verifier decides

Machine can speed up clause diffs, mechanical carry-forward checks, and prior-round risk comparisons; judging optimism bias and building worst cases stays with Verifier.

---

## Workflow

### Step 1: Write verification plan (Verifier)

Create a plan that states lenses, subject, and criteria.

```bash
write_memory_file(path="state/plans/{date}_{matter-name}.verification.md", content="...")
```

Before writing, assemble on the Anima side:
- Director’s `audit-report.md` and `analysis-plan.md`
- carry-forward tracker (for comparison to prior version)
- Contract text (before and after if amended)

### Step 2: Run verification scan on machine

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{date}_{matter-name}.verification.md)" \
  -d /path/to/workspace
```

Append or overwrite `state/plans/{date}_{matter-name}.verification.md` (`status: draft`).

### Step 3: Meta-review verification output

Read `verification.md` and check:

- [ ] Findings are factual (no false positives)
- [ ] carry-forward cross-check has no gaps
- [ ] Worst-case analysis exists for Director “acceptable” calls
- [ ] Prior-round risk comparison is accurate
- [ ] Anything else you would flag from your own lenses

Edit and supplement yourself, then set `status: approved`.

### Step 4: Feedback

Send approved `verification-report.md` to Director.
If you request changes, state concrete rationale and suggested fixes.

---

## Template: verification plan (`verification.plan.md`)

```markdown
# Verification plan: {subject summary}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: verification-plan

## Verification lenses

- [ ] Optimism bias: re-check every “acceptable” item
- [ ] carry-forward trace: every prior finding appears in this report
- [ ] Prior-round risk comparison: downgrades have enough support
- [ ] Legal accuracy: cited statutes and cases
- [ ] Clause coverage: every contract section was analyzed

## Subject

- audit-report.md: {path}
- analysis-plan.md: {path}
- carry-forward tracker: {path}
- Contract text: {path / location}

## Diff summary

{Main changes before/after — summary, not full text}

## Required output format

Use the structure below. **Output that does not follow this format is invalid.**

- **Critical**: Must-fix (optimism bias, silent drop, legal error)
- **Warning**: Should-fix (thin rationale, wording risk)
- **Info**: FYI / improvement ideas
```

## Template: verification report (`verification-report.md`)

```markdown
# Verification report: {matter-name}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: verification-report

## Overall verdict

{APPROVE / REQUEST_CHANGES / COMMENT}

## Optimism bias findings

| # | Clause | Director call | Verifier view | Worst case | Recommendation |
|---|--------|---------------|---------------|------------|----------------|
| 1 | {clause} | {acceptable} | {concern} | {if exploited} | {fix} |

## Carry-forward gaps

| # | Prior finding | Prior risk | In this report | Verdict |
|---|---------------|------------|----------------|---------|
| 1 | {finding} | {risk} | {mentioned / silent drop} | {OK / NG} |

## Prior-round risk comparison

| # | Clause | Prior risk | This risk | Sound rationale? |
|---|--------|------------|-----------|------------------|
| 1 | {clause} | {prior} | {now} | {sound / thin} |

## Legal accuracy

| # | Finding | Severity |
|---|---------|----------|
| 1 | {content} | {Critical/Warning/Info} |

## Verifier notes

{Your analysis, extra observations, recommendations}
```

---

## Constraints

- Verification plan (what to check) MUST be written by Verifier
- NEVER forward machine output to Director unchanged — Verifier must meta-review
- NEVER send `verification-report.md` without `status: approved`
- MUST build worst cases for “acceptable” items yourself — do not delegate that to machine
