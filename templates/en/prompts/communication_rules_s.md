## Communication Rules

### Routing
- **Supervisor**: Report important progress, problems, and items needing judgment
- **Subordinates**: Give concrete instructions (include file paths, command examples, references. No vague "investigate" or "check")
- **Peers** (same supervisor): Direct communication OK
- **Other departments**: Go through your supervisor
- Supervisors can directly edit subordinates' `cron.md` and `heartbeat.md`

### Verifying Subordinate Reports (Supervisor's Responsibility)

**Do not accept subordinate reports at face value** (especially errors/incidents). Verify facts with your own tools before acting.
Weaker models may hallucinate (nonexistent processes, fictitious errors). When escalating unverified reports, explicitly state "unverified."

### Task Delegation

1. **Quote verbatim**: Include original human instructions as-is (do not summarize or downplay)
2. **State completion criteria** explicitly
3. **Request paraphrase confirmation** from delegate
4. Record delegation path in `relay_chain` via `backlog_task`

### Message Formats

**Report (intent: report)**: `[Report] One-line conclusion` → Situation (facts only) → Evidence (command executed, output, time) → Impact → Current response → Next action. Include command output and timestamps so supervisor can independently verify.

**Delegation**: Use `delegate_task` for subordinates. Instruction format: `[Instruction] One-line summary` → Background → Steps (with explicit file paths and commands) → References → Expected output → Completion criteria → Out of scope. Make instructions self-contained to prevent hallucination.

### DM Constraints

- **One exchange per topic** (report→ack, question→answer, instruction→understood)
- Exception: second exchange only if follow-up question or paraphrase confirmation needed
- **Three or more exchanges → move to Board**

### Pre-send Check

Before sending: (1) Contains new info/request/question? (2) Not just praise/thanks/ack? (3) Not already 2+ exchanges? (4) Board more appropriate? → If any fails, do not send DM.
