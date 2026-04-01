# Precedent Researcher — quality checklist

Checklist for research collection and reporting quality.

---

## A. Collection quality

### Source completeness

- [ ] Every item has a URL or bibliographic citation
- [ ] Statute references are accurate (e.g. Civil Code Art. 415)
- [ ] Case citations are complete (court, date, docket)
- [ ] Guidelines/circulars show issuer, date, document number
- [ ] Retrieval date recorded (for tracking law and case changes over time)

### Reliability

- [ ] Prefer primary sources (statute text, judgment, official guidance)
- [ ] For secondary sources (articles, blogs), attempt to trace to primary
- [ ] “Unverified” / “needs more research” called out explicitly
- [ ] No reliance on repealed law or superseded guidance

---

## B. Coverage

### Mapping to `audit-report.md`

- [ ] Every “industry standard,” “generally,” “typically,” etc. claim is covered
- [ ] Each claim has status: supported / not supported / partial
- [ ] “Not supported” items say “no support — recommend revisiting risk rating”

### Statute and case breadth

- [ ] Checked latest amendments for relevant statutes
- [ ] If several cases apply, major ones are covered
- [ ] If cases conflict, both sides are recorded
- [ ] Research matches applicable governing law (domestic vs foreign)

---

## C. Reporting quality to Director

### `precedent-report.md` structure

- [ ] Organized by research topic / domain
- [ ] Lead with conclusion (support status) per topic
- [ ] Open items and follow-up research called out
- [ ] Major findings that affect Director risk are summarized

### Timing

- [ ] Material findings (case that changes risk, new law) reported immediately
- [ ] Completion report includes file path, main findings, and open items

---

## Template: `precedent-report.md`

```markdown
# Statute and case research report: {matter-name}

status: draft
author: {anima-name}
date: {YYYY-MM-DD}
type: precedent-report
source: {path to source audit-report.md}

## Research summary

- Topics: {N}
- Supported: {N} / Not supported: {N} / Partial: {N} / Unverified: {N}

## Findings

### {Topic 1: e.g. indemnity industry practice}

- **Claim in audit-report**: {quote}
- **Support status**: {yes / no / partial}
- **Basis**:
  - Statute: {number and text}
  - Case: {court, date, docket, holding}
  - Practice: {source and content}
- **Notes**: {implications}

### {Topic 2}

(Repeat for each topic.)

## Material findings

{Summary of what most affects Director risk rating}

## Open items

| # | Item | Why open | Next step |
|---|------|----------|-----------|
| 1 | {item} | {reason} | {step} |
```
