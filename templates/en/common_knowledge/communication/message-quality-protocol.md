# Message Quality Protocol

Mandatory checklist before sending messages.
For detailed examples, see `communication/reporting-guide.md` / `communication/instruction-patterns.md`.

---

## 1. Delegation — 4 Required Fields

Always include when using `delegate_task` or sending a request:

| # | Field | Content |
|---|-------|---------|
| 1 | Task description | What to do (1-2 line summary with context) |
| 2 | Completion criteria | What "done" looks like |
| 3 | References | File paths / Issue・PR URLs / related resources |
| 4 | Deadline & report target | When to finish, who to report to |

**Pre-delegation check**: Verify no open delegation exists for the same Issue/PR via `list_tasks`.

---

## 2. Completion Report — 3 Required Fields

Always include in completion reports (intent="report"):

| # | Field | Content |
|---|-------|---------|
| 1 | Result | What was completed (1 line) |
| 2 | Artifacts | File paths / PR URLs / numeric results |
| 3 | Verification evidence | How verified, count of items checked, timestamp |

**Required even for "all clear"**: State what was checked, how many, and when.

- Bad: "No issues found"
- Good: "Checked Slack 3ch + Chatwork 2rooms, ERROR 0, last checked 14:52 JST"

---

## 3. Escalation — 4 Required Fields

Always include in problem reports or decision requests:

| # | Field | Content |
|---|-------|---------|
| 1 | Facts | What happened (timestamp + observed facts only — no evaluation or emotion) |
| 2 | Impact | Who/what is blocked |
| 3 | Attempted fixes | What you tried and the result |
| 4 | Options | 2+ options with recommendation |

**Prohibited**: Using evaluation words instead of facts ("terrible state", "completely failed").
State facts and timestamps first; add evaluation as a single sentence at the end if needed.
