# Sales & Marketing Full Team — Overview

## Four-role layout

| Role | Responsibility | Suggested `--role` | Example `speciality` | Details |
|------|----------------|---------------------|----------------------|---------|
| **Sales & Marketing Director** | Strategy, sales execution [machine], pipeline management, content QC, final sign-off | `manager` | `sales-marketing-director` | `sales-marketing/director/` |
| **Marketing Creator** | Marketing content production [machine], Brand Voice compliance | `writer` | `marketing-creator` | `sales-marketing/creator/` |
| **SDR (Sales Development)** | Lead development, nurturing, engagement, inbound handling | `general` | `sales-development` | `sales-marketing/sdr/` |
| **Market Researcher** | Market research, competitor analysis, prospect profiling | `researcher` | `market-researcher` | `sales-marketing/researcher/` |

Putting every step in one Anima invites content self-review bias, lax lead qualification, and context pollution from competing marketing vs. sales priorities.

Each role directory has `injection.template.md` (injection skeleton), `machine.md` (machine usage patterns for that role only), and `checklist.md` (quality checklist).

> Core principles: `team-design/guide.md`

## Two execution modes

### Campaign mode (plan-driven)

```
Director → content-plan.md (approved) → delegate_task
  → Marketing Creator → machine production → draft-content.md (draft)
    → Director → checklist + machine QC → approve / revise / {COMPLIANCE_REVIEWER} compliance review
```

Standard flow for content marketing. Director plans, Creator produces, Director verifies.

### Engagement mode (SDR autonomous patrol)

```
SDR → SNS/email/inbound monitoring (cron)
  → Lead found → report to Director + lead-report.md
  → Nurturing target → machine draft → SDR verifies & sends
  → CS inquiry → escalate to CS team
```

SDR autonomously patrols channels via cron, finding and nurturing leads.

## Handoff chain

```
Director → content-plan.md (approved)
  → delegate_task
    → Creator: content production (machine)
    → Researcher: market research (direct tool use)
      → draft-content.md / research-report.md (reviewed)
        → Director → machine QC + checklist → approve / revise
          └─ compliance risk → request review from {COMPLIANCE_REVIEWER} (cross-team)
          └─ approved → publish → Campaign Tracker update

SDR → autonomous patrol (cron)
  → lead-report.md → Director → Deal Tracker update
  → deal won → cs-handoff.md → CS team
```

### Handoff documents

| From → To | Document | Condition |
|-----------|----------|-----------|
| Director → Creator | `content-plan.md` | `status: approved` |
| Creator → Director | `draft-content.md` | `status: draft` |
| Director → SDR | Outbound instructions | `delegate_task` |
| SDR → Director | `lead-report.md` | On lead discovery |
| Director → {COMPLIANCE_REVIEWER} | `compliance-review.md` | Compliance risk flag |
| {COMPLIANCE_REVIEWER} → Director | Same file with review results appended | `status: reviewed` |
| Director → Researcher | Research request | `delegate_task` |
| Researcher → Director | `research-report.md` | `status: approved` |
| Director → CS team | `cs-handoff.md` | On deal close |

### Operating rules

- **Fix cycle**: Critical → full re-production / Warning → diff-only fix / Still unresolved after 3 rounds → escalate to humans
- **Campaign Pipeline Tracker**: Track content production stages. Silent drop forbidden
- **Deal Pipeline Tracker**: Track deal stages. Silent drop forbidden
- **Compliance escalation**: Creator/SDR first filter → Director second judgment → {COMPLIANCE_REVIEWER} cross-team review
- **Product marketing**: New feature info comes through upper management (COO, etc.) to Director
- **Machine failure**: Record in `current_state.md` → reassess on next heartbeat

## Scaling

| Scale | Composition | Notes |
|-------|-------------|-------|
| Solo | Director covers all roles (quality via checklists) | SNS posts, simple research |
| Pair | Director + Creator | Content marketing focused |
| Trio | Director + Creator + SDR | Including outbound sales |
| Full team | Four roles as in this template | Full-funnel marketing + sales |

## Mapping to other teams

| Development role | Legal role | Sales & MKT role | Why it maps |
|------------------|------------|-------------------|-------------|
| PdM (plan, decide) | Director (analysis plan, judgment) | Director (strategy, sales execution) | Sets what to do |
| Engineer (implement) | Director + machine | Director + machine (sales content) | Machine executes production |
| Reviewer (static verification) | Verifier (independent verification) | {COMPLIANCE_REVIEWER} (compliance) | Independent verification, cross-team |
| Tester (dynamic verification) | Researcher (evidence verification) | Researcher (market research) | External data backing |
| — | — | Creator (content production) | Marketing-specific: high-volume content |
| — | — | SDR (lead development) | Sales-specific: real-time monitoring & engagement |

## Campaign Pipeline Tracker — content tracking table

Tracks content/campaign production stages. Structurally prevents silent drop.

### Tracking rules

- Register new content plans in this table when created
- Update all items' stages on each Heartbeat / review
- Report stagnation (no stage change for 2+ weeks) to Director
- Silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Campaign tracking table: {team-name}

| # | Plan name | Type | Funnel | Stage | Owner | Start | Deadline | Notes |
|---|-----------|------|--------|-------|-------|-------|----------|-------|
| CP-1 | {name} | {blog/email/...} | {TOFU/MOFU/BOFU} | {stage} | {Creator/Director} | {date} | {date} | {notes} |

Stage legend:
- Planning: content-plan.md in progress
- Research: research request sent to Researcher
- Production: Creator producing via machine
- QC: Director quality check in progress
- Compliance: {COMPLIANCE_REVIEWER} review in progress
- Approved: ready to publish
- Published: delivered
- Measuring: performance analysis
```

## Deal Pipeline Tracker — deal tracking table

Tracks individual deal sales stages. Structurally prevents silent drop.

### Tracking rules

- Register leads in this table when SDR discovers them
- Update all items' stages on each Heartbeat / review
- Analyze stagnation (no stage change for 2+ weeks)
- Silent drop (disappearance without mention) is forbidden

### Template

```markdown
# Deal tracking table: {team-name}

| # | Company | Source | Stage | Owner | Start | Updated | Notes |
|---|---------|--------|-------|-------|-------|---------|-------|
| D-1 | {name} | {inbound/outbound/...} | {stage} | {SDR/Director} | {date} | {date} | {notes} |

Stage legend:
- Lead: acquired (unqualified)
- Qualified: BANT evaluation passed
- Discovery: deep-diving needs
- Proposal: proposal submitted
- Negotiation: terms under negotiation
- Won: deal closed
- Lost: deal lost (record reason in notes)
- CS Handoff: handed off to CS team
```
