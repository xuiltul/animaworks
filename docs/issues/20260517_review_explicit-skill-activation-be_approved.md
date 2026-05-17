# Code Review: Explicit Skill Activation BE - Approved

**Review Date**: 2026-05-17
**Original Issue**: `docs/issues/20260517_explicit-skill-activation-be.md`
**Worktree**: `/home/main/dev/animaworks-bak-issue-20260517-180522`
**Status**: APPROVED

## Summary

The backend implementation satisfies the issue requirements. It adds thread-scoped active skill state, UI-facing APIs, chat-only prompt injection, safety rejection/warnings, usage recording, focused unit tests, server route tests, E2E prompt tests, and API documentation.

## Findings

### Requirement Alignment

Status: PASS

- `GET /api/animas/{name}/skills` lists visible skills and marks active state for the requested thread.
- `PUT /api/animas/{name}/skills/active` validates refs, persists accepted canonical refs, clears on `refs: []`, and returns accepted/rejected/warning details.
- `## Active Skills` is injected only for chat prompts and is isolated by `thread_id`.
- Active skill rendering records `SkillUsageEventType.use` with `active_skill` notes.
- `dangerous`, `blocked`, and `quarantine` remain blocked; `warn`, `caution`, and risky skills require `confirm_risk=true`.
- Existing cron skill injection and catalog behavior remain separate.

### Test Coverage

Status: PASS

- Focused coverage command passed at 94.04% for `core.skills.activation` and `server.routes.skills`.
- New focused tests: 13 passed.
- Existing skill/prompt regression set: 90 passed.
- Existing server route regression set: 52 passed.
- Marked E2E suite: 169 passed, 2 skipped, 14160 deselected.

### Code Quality and SRP

Status: PASS

- New behavior is concentrated in `core/skills/activation.py` and `server/routes/skills.py`.
- Prompt builder changes are limited to injecting rendered active skill context for chat sessions.
- Agent cycle and priming changes only pass `thread_id` through existing prompt rebuild paths.
- No unrelated feature behavior was changed.

### File Size

Status: PASS

- New implementation file `core/skills/activation.py` is 495 lines.
- New route and test files are below 500 lines.
- Existing large files touched by small integration points were already above 500 lines before this change.

### Regression

Status: PASS WITH KNOWN ENVIRONMENT/BASELINE FAILURES

`python3 -m pytest --tb=short -q` completed with 14247 passed, 48 skipped, 6 failed, and 30 errors. The failures are not attributable to this change:

- Playwright browser executable is missing for `test_ipad_viewport_e2e.py` and `test_responsive_layout_e2e.py`.
- `watchdog` is not installed for `test_common_knowledge.py` and `test_watcher.py`; clean main HEAD has the same failure.
- `test_mode_b.py::TestModeBSkillInjection::test_common_skill_in_system_prompt` fails on clean main HEAD as well.
- `test_bootstrap_protection.py::TestBootstrapRetryLimit::test_retry_count_reset_on_success` fails on clean main HEAD as well.

### Independent Reviews

Status: PARTIAL

- Cursor Agent was launched with `claude-4.6-opus-high-thinking`, exited status 0, but produced an empty review file. Treated as unavailable.
- Codex subagent review was skipped because this session only permits subagents when the user explicitly requests delegation/subagents.

## Decision

APPROVED. No revision required for the explicit skill activation backend implementation.
