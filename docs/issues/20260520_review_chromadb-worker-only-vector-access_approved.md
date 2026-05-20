# Review: ChromaDB worker-only vector access

Status: APPROVED
Date: 2026-05-20
Issue: `docs/issues/20260520_chromadb-worker-only-vector-access.md`
Worktree: `/home/main/dev/animaworks-bak-issue-20260520-104848`
Branch: `issue-20260520-104848`
Implementation commit: `5e9695e4 fix: route ChromaDB access through vector worker`

## Summary

The implementation satisfies the issue requirement: production ChromaDB access is now worker-only. Direct native Chroma construction is guarded, the server process routes RAG through the vector worker URL, internal vector routes fail closed with 503 instead of falling back to direct Chroma, and offline/preflight/repair paths use worker-backed access.

No critical or high-priority findings remain.

## Requirement Alignment

Pass.

- `ChromaVectorStore` construction is guarded by `ANIMAWORKS_ALLOW_DIRECT_CHROMA`.
- `get_vector_store()` returns `HttpVectorStore` when `ANIMAWORKS_VECTOR_URL` is set and otherwise does not fall back to direct Chroma unless the explicit allow flag is present.
- The vector worker process sets the direct-Chroma allow flag and removes inherited recursive vector URLs.
- Server lifespan sets and restores the server process vector URL around worker lifecycle.
- `/api/internal/vector/*` routes proxy only to the worker or return 503.
- Startup preflight, repair, offline indexing, rename cleanup, scripts, and LOCOMO setup now use worker-backed vector access.
- Static regression coverage prevents new production `ChromaVectorStore(` and `chromadb.PersistentClient(` call sites outside allowed files.

## Automated Checks

Pass with noted environment limitations.

- `python3 -m py_compile ...`: passed for changed Python modules and scripts.
- `git diff --check`: passed.
- Targeted vector route/static tests: `6 passed, 1 warning`.
- Related regression suite: `2123 passed, 19 skipped, 7 warnings`.
- Full suite excluding two Playwright-browser E2E files: `14292 passed, 48 skipped, 167 warnings`.
- Full unfiltered suite was also run; the only remaining errors after unit fixes were Playwright browser setup errors in `tests/e2e/test_ipad_viewport_e2e.py` and `tests/e2e/test_responsive_layout_e2e.py` because Chromium is not installed under `~/.cache/ms-playwright`.

The bundled coverage checker was attempted and timed out after 5 minutes, reporting `Coverage: 0.0%`. This was treated as tool/environment limitation because the relevant targeted and broad test suites passed.

The bundled file-size checker reports many pre-existing oversized files across the repository. The new files in this implementation are small (`31`, `33`, and `42` lines), and reviewed changed core route/helper files remain within 500 lines after the route fallback removal.

## Manual Review

Pass.

- Fail-closed behavior is intentional and matches the issue: worker unavailable means 503/degraded RAG, not native fallback in the parent process.
- Environment mutations are scoped and restored for server lifespan and temporary worker handles.
- The HTTP store cache is keyed by both base URL and anima name, avoiding stale worker URL reuse.
- Existing low-level store tests remain possible through the explicit test allow fixture without weakening production defaults.
- The E2E smoke exercises a real temporary worker and `HttpVectorStore` round trip.

## Independent Reviews

Skipped.

The worktree-review skill normally requests external/Codex agent reviews. This session did not spawn subagents because the active developer instruction only allows subagent spawning when the user explicitly asks for subagents or delegated parallel agent work. Review evidence is therefore from automated checks plus self-review.

## Findings

None requiring revision.

Residual risk: Playwright-dependent viewport E2E tests were not validated in this environment because the Playwright Chromium binary is missing. This is unrelated to the vector worker change and was isolated by running the rest of the suite successfully.
