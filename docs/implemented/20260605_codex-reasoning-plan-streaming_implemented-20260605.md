# Mode C Codex Reasoning and Plan Streaming — Expose SDK reasoning summaries and plan progress as fine-grained GUI events

## Overview

Mode C already streams Codex agent text and tool progress through the new `openai-codex` SDK path, but the GUI still rarely shows thinking/progress content because the executor does not request Codex reasoning summaries and does not map Codex plan notifications. This issue makes Codex emit the public reasoning summary surface by default and maps Codex plan/progress notifications into the existing AnimaWorks `thinking_*` stream contract.

This builds on the prior Codex App Server SDK migration. It does not expose hidden model reasoning.

## Problem / Background

### Current State

- `CodexSDKExecutor._codex_turn_kwargs()` passes approval/cwd/model/sandbox only; it does not pass `summary=` or `effort=` to the SDK — `core/execution/codex_sdk.py:994`.
- The executor already maps `item/reasoning/textDelta` and `item/reasoning/summaryTextDelta` to `thinking_delta`, but Codex is not asked to produce summaries by default — `core/execution/codex_sdk.py:1536`.
- The executor does not handle `item/plan/delta` or `turn/plan/updated`, even though the installed SDK exposes both notification types.
- The frontend expects a `thinking_start` before animated thinking display; Codex currently emits only `thinking_delta` from reasoning items — `server/static/pages/chat/streaming-controller.js:494`.
- The SSE route and frontend already support `thinking_start`, `thinking_delta`, and `thinking_end` — `server/routes/chat_chunk_handler.py:123`, `server/static/shared/chat-stream.js:306`.

### Root Cause

1. Codex SDK turn options do not request a reasoning summary, so summary delta notifications may never appear — `core/execution/codex_sdk.py:994`.
2. Codex plan notifications are ignored, so visible progress is lost even when the app server emits it — `core/execution/codex_sdk.py:1726`.
3. The Codex executor emits `thinking_delta` without a surrounding `thinking_start` / `thinking_end` lifecycle, so the chat UI may not initialize its thinking animator — `server/static/pages/chat/streaming-controller.js:494`.

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `core/execution/codex_sdk.py` | Direct | Needs Codex reasoning summary turn kwargs and plan notification mapping. |
| `tests/unit/test_codex_sdk_executor.py` | Direct | Needs coverage for summary config, plan delta, plan updated, and thinking lifecycle. |
| `tests/e2e/test_codex_mode_e2e.py` | Indirect | Existing Mode C mocks must tolerate new turn kwargs. |
| `server/routes/chat_chunk_handler.py` | No change expected | Already forwards `thinking_start`, `thinking_delta`, and `thinking_end`. |
| `server/static/shared/chat-stream.js` | No change expected | Already handles thinking SSE events. |
| `server/static/pages/chat/streaming-controller.js` | No change expected | Already starts and stops the thinking animator when lifecycle events arrive. |

## Decided Approach / 確定方針

### Design Decision

確定: Mode C will request public Codex reasoning summaries by default using `summary=ReasoningSummary(root=ReasoningSummaryValue.concise)` on SDK turns, allow per-credential override through `extra_keys.codex_reasoning_summary`, and map Codex plan notifications into the existing `thinking_*` streaming lifecycle. The executor must emit `thinking_start` before the first Codex reasoning/plan delta and `thinking_end` before final `done` when thinking was opened. Hidden/raw model reasoning must not be synthesized or exposed.

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Frontend redesign | Could make a custom Codex progress panel | Existing SSE/frontend already supports thinking lifecycle; larger blast radius | **Rejected**: Backend chunk mapping is sufficient. |
| Use SDK `stream_text()` | Simple text-only streaming | Drops plan, reasoning summary, tool, and file-change events | **Rejected**: Conflicts with fine-grained GUI goal. |
| Emit hidden reasoning or synthesize thinking | Would make UI look more active | Not a public SDK surface; can misrepresent model behavior | **Rejected**: Only SDK-provided public summary/plan text is allowed. |
| Always use `detailed` summaries | Maximum visible reasoning summary | More verbose, higher token/UI noise | **Rejected**: Default should be useful but restrained. |
| **Use `concise` public summary + plan mapping (Adopted)** | Fine-grained, safe, low blast radius | Some models/turns may still emit little summary text | **Adopted**: Best match for GUI granularity without exposing hidden reasoning. |

### Key Decisions from Discussion

1. **Hidden thinking stays hidden**: Only SDK-provided reasoning summary and plan/progress text may be shown — Reason: maintain OpenAI reasoning boundaries and avoid invented text.
2. **Default summary is `concise`**: Mode C requests concise reasoning summaries by default — Reason: visible progress without excessive noise.
3. **Config key is `extra_keys.codex_reasoning_summary`**: Accepted values are `auto`, `concise`, `detailed`, and `none` — Reason: follows existing provider-specific credential key pattern such as `codex_model` and `api_version`.
4. **Plan notifications map to thinking**: `item/plan/delta` and `turn/plan/updated` become GUI-visible thinking/progress chunks — Reason: plan updates are public progress signals and improve granularity even when reasoning summaries are sparse.
5. **Thinking lifecycle is explicit**: Emit `thinking_start` once before first reasoning/plan delta and `thinking_end` once before `done` — Reason: frontend animator expects lifecycle events.

### Changes by Module

| Module | Change Type | Description |
|--------|-------------|-------------|
| `core/execution/codex_sdk.py` | Modify | Add summary option resolver, pass `summary` to turn kwargs, map plan notifications, and wrap reasoning/plan output with thinking lifecycle chunks. |
| `tests/unit/test_codex_sdk_executor.py` | Modify | Add tests for summary default/override/invalid values, plan delta mapping, turn plan mapping, and thinking lifecycle ordering. |
| `tests/e2e/test_codex_mode_e2e.py` | Verify/update if needed | Ensure Mode C mocks still pass with additional SDK turn kwargs. |
| `server/routes/chat_chunk_handler.py` | No change expected | Existing thinking SSE mapping remains sufficient. |
| `server/static/shared/chat-stream.js` | No change expected | Existing thinking event handling remains sufficient. |

#### Change 1: Reasoning summary turn kwargs

**Target**: `core/execution/codex_sdk.py`

Before:

```python
return {
    "approval_mode": self._sdk_approval_mode(),
    "cwd": str(self._task_cwd or self._anima_dir),
    "model": provider_config.model,
    "sandbox": self._sdk_sandbox(),
}
```

After:

```python
kwargs = {
    "approval_mode": self._sdk_approval_mode(),
    "cwd": str(self._task_cwd or self._anima_dir),
    "model": provider_config.model,
    "sandbox": self._sdk_sandbox(),
}
summary = self._sdk_reasoning_summary()
if summary is not None:
    kwargs["summary"] = summary
return kwargs
```

### Edge Cases

| Case | Handling |
|------|----------|
| `codex_reasoning_summary` missing | Use `concise`. |
| `codex_reasoning_summary=none` | Do not pass `summary`; reasoning/plan notifications are still mapped if the SDK emits them. |
| Invalid summary value | Log a warning and fall back to `concise`. |
| SDK does not emit reasoning summary | No fabricated thinking text; plan/tool/text streaming still works. |
| First event is a reasoning/plan delta | Emit `thinking_start` immediately before the delta. |
| Multiple reasoning/plan deltas | Emit `thinking_start` only once. |
| Turn completes after thinking opened | Emit `thinking_end` before final `done`. |
| Stream errors after thinking opened | Emit `thinking_end` before propagating error when practical; never suppress the error. |
| `codex exec --json` fallback | No change required; fallback may remain coarse and may not emit thinking. |

## Implementation Plan

### Phase 1: Codex summary option resolver

| # | Task | Target |
|---|------|--------|
| 1-1 | Add helper to resolve `extra_keys.codex_reasoning_summary` to SDK `ReasoningSummary` or `None` | `core/execution/codex_sdk.py` |
| 1-2 | Pass resolved summary in `_codex_turn_kwargs()` | `core/execution/codex_sdk.py` |
| 1-3 | Add unit tests for default, `none`, and invalid values | `tests/unit/test_codex_sdk_executor.py` |

**Completion condition**: SDK turn kwargs request `concise` reasoning summaries by default and can disable them.

### Phase 2: Plan and thinking lifecycle mapping

| # | Task | Target |
|---|------|--------|
| 2-1 | Add local `thinking_started` state and `thinking_start`/`thinking_end` helpers in streaming path | `core/execution/codex_sdk.py` |
| 2-2 | Use helpers for existing reasoning delta and reasoning item snapshot paths | `core/execution/codex_sdk.py` |
| 2-3 | Map `item/plan/delta` to `thinking_delta` | `core/execution/codex_sdk.py` |
| 2-4 | Map `turn/plan/updated` to concise plan text suitable for GUI | `core/execution/codex_sdk.py` |

**Completion condition**: Reasoning and plan events produce a coherent thinking lifecycle in the stream.

### Phase 3: Verification

| # | Task | Target |
|---|------|--------|
| 3-1 | Add unit tests for plan delta and turn plan update mapping | `tests/unit/test_codex_sdk_executor.py` |
| 3-2 | Re-run Mode C unit coverage and e2e mocks | test commands |
| 3-3 | Run ruff and diff checks | touched files |

**Completion condition**: Relevant tests pass and executor coverage remains at or above the existing 80% target.

## Scope

### In Scope

- Codex SDK public reasoning summary request for Mode C.
- Plan notification mapping to the existing GUI thinking/progress stream.
- Explicit thinking lifecycle events around Codex reasoning/plan chunks.
- Unit/e2e tests for the new behavior.

### Out of Scope

- Exposing hidden/raw model chain-of-thought — Reason: not a public SDK surface.
- Frontend redesign — Reason: existing UI already supports thinking lifecycle.
- Changing Mode S/A/B/D/G behavior — Reason: request targets Codex Mode C.
- Live Codex credential validation — Reason: tests should remain deterministic and credential-free.
- Changing default Codex model selection — Reason: unrelated to streaming granularity.

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Some Codex models/turns emit no summary even when requested | GUI still shows little thinking text | Also map plan notifications and avoid fabricating content. |
| `detailed` summary can be noisy | GUI clutter and token overhead | Default to `concise`; make value configurable. |
| SDK enum/API changes | Runtime import or kwargs failures | Isolate resolver, test against installed SDK, keep invalid config fallback. |
| Thinking lifecycle ordering bugs | Frontend animator may not render or may stay open | Unit-test ordering: start before deltas, end before done. |
| Plan text can repeat | Duplicate visual noise | For `turn/plan/updated`, render current plan snapshot as concise lines; do not accumulate duplicate full snapshots in executor state beyond emitted chunks. |

## Acceptance Criteria

- [ ] Mode C SDK turn kwargs include `summary=concise` by default.
- [ ] `extra_keys.codex_reasoning_summary=none` disables passing `summary`.
- [ ] Invalid `codex_reasoning_summary` logs/falls back to `concise` rather than failing.
- [ ] `item/reasoning/summaryTextDelta` and `item/reasoning/textDelta` emit `thinking_start` before first delta and `thinking_delta` for each delta.
- [ ] `item/plan/delta` emits GUI-visible thinking/progress text.
- [ ] `turn/plan/updated` emits GUI-visible plan snapshot text.
- [ ] `thinking_end` is emitted before final `done` when thinking was opened.
- [ ] Updated tests pass: `uv run pytest tests/unit/test_codex_sdk_executor.py --cov=core.execution.codex_sdk --cov-report=term-missing --cov-fail-under=80 -q`.
- [ ] Updated tests pass: `uv run pytest tests/e2e/test_codex_mode_e2e.py tests/e2e/test_non_chat_clean_session_isolation_e2e.py tests/unit/test_session_compactor.py tests/unit/core/memory/test_llm_utils.py -q`.
- [ ] Ruff check passes for touched files.

## References

- `core/execution/codex_sdk.py:994` — current SDK turn kwargs omit `summary`.
- `core/execution/codex_sdk.py:1536` — current reasoning delta mapping.
- `core/execution/codex_sdk.py:1726` — unhandled event fallback where plan notifications currently land.
- `server/routes/chat_chunk_handler.py:123` — SSE support for thinking lifecycle.
- `server/static/shared/chat-stream.js:306` — frontend SSE thinking event handling.
- `server/static/pages/chat/streaming-controller.js:494` — thinking animator initialization on `thinking_start`.
- Installed `openai_codex.generated.v2_all.ReasoningSummary` — `auto`, `concise`, `detailed`, `none` summary values.
- Installed `openai_codex.generated.notification_registry.NOTIFICATION_MODELS` — includes `item/plan/delta`, `turn/plan/updated`, and reasoning summary notifications.
