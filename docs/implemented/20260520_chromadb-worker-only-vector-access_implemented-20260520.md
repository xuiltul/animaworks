# ChromaDB worker-only vector access — Route every ChromaDB operation through the vector worker

## Overview

AnimaWorks already has a process-isolated vector worker, but not every ChromaDB path is forced through it. On 2026-05-20 03:50:31 JST the main server process segfaulted in `chromadb_rust_bindings.abi3.so` while running server-side synaptic downscaling, proving that direct ChromaDB fallback paths still exist. This issue makes worker-only ChromaDB access the production invariant: if native Chroma crashes, only the vector worker process may die.

## Problem / Background

### Current State

- `animaworks start` has been restored, and `mei` is back to `running`, but the recovery only repaired this incident's vector stores.
- The kernel logged `tokio-rt-worker ... segfault ... in chromadb_rust_bindings.abi3.so` at 2026-05-20 03:50:31 JST.
- The daemon log ended after `Daily consolidation for natsume` and `Starting synaptic downscaling for anima=natsume`, with no graceful FastAPI shutdown.
- The server-side downscaling path calls `ForgettingEngine.synaptic_downscaling()` from the supervisor scheduler, then scans Chroma collections.
- A vector worker subprocess exists and is process-isolated, but the server, startup preflight, and offline CLI commands can still instantiate `ChromaVectorStore` directly.

Relevant code:

- `core/supervisor/_mgr_scheduler.py:363` — server process runs post-consolidation synaptic downscaling.
- `core/memory/forgetting.py:255` — downscaling obtains a vector store and scans collections.
- `core/memory/rag/store.py:381` — Chroma `coll.get()` native call that can segfault below Python.
- `core/memory/rag/singleton.py:71` — `ANIMAWORKS_VECTOR_URL` uses HTTP, otherwise direct Chroma fallback is used.
- `server/app.py:548` — vector URLs are intentionally only passed to child processes, not server `os.environ`.
- `server/routes/internal.py:205` — internal vector routes proxy to the worker but can fall back to direct Chroma.
- `cli/commands/server.py:374` — startup RAG preflight runs before app/vector worker startup.
- `cli/commands/index_cmd.py:242` — index command delegates only when a running server is detected.
- `cli/commands/anima_mgmt.py:1252` — rename cleanup directly constructs `ChromaVectorStore`.

### Root Cause

1. Direct Chroma fallback remains the default when `ANIMAWORKS_VECTOR_URL` is absent — `core/memory/rag/singleton.py:71`.
2. The server process starts a vector worker but does not set its own `ANIMAWORKS_VECTOR_URL`, so server-side scheduler, repair, and memory code can load native Chroma — `server/app.py:548`.
3. Internal vector endpoints still contain direct fallback branches after `_try_vector_worker()` returns `None` — `server/routes/internal.py:225`.
4. Startup preflight repairs suspected RAG DBs before the worker exists, so repair/reindex can run native Chroma in the startup process — `cli/commands/server.py:374`.
5. Offline CLI paths use direct Chroma when the server is not running — `cli/commands/index_cmd.py:242`, `cli/commands/anima_mgmt.py:1252`.

### Impact

| Component | Impact | Description |
|-----------|--------|-------------|
| `core/memory/rag/singleton.py` | Direct | Allows production code to instantiate native Chroma when no vector URL is set. |
| `server/app.py` | Direct | Leaves server-owned RAG operations outside vector worker isolation. |
| `server/routes/internal.py` | Direct | Reintroduces parent-process crash risk when worker is unavailable or disabled. |
| `cli/commands/server.py` | Direct | Startup preflight can crash before the HTTP server is available. |
| `cli/commands/index_cmd.py` | Direct | Offline indexing can load native Chroma in the CLI process. |
| `cli/commands/anima_mgmt.py` | Direct | Rename cleanup directly opens vector DB collections. |
| `core/memory/rag/vector_worker.py` | Direct | Must become the only production owner of `ChromaVectorStore`. |

## Decided Approach / 確定方針

### Design Decision

確定: Production ChromaDB access is worker-only. The vector worker remains the sole process allowed to instantiate `ChromaVectorStore` or `chromadb.PersistentClient`; server, supervisor, repair, indexing, and CLI paths must use `HttpVectorStore` through either the long-lived server vector worker or a temporary worker context. Direct Chroma fallback is removed from production paths and guarded by an explicit allow flag used only by the worker and tests.

### Rejected Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Catch exceptions around synaptic downscaling | Small patch | Native segfault is below Python and cannot be caught | **Rejected**: does not contain process death. |
| Disable synaptic downscaling | Avoids the observed trigger | Other RAG/search/index paths can still hit Chroma segfaults | **Rejected**: treats one symptom only. |
| Keep direct fallback when worker is down | Preserves degraded functionality | Reintroduces the exact server-crash path this issue removes | **Rejected**: fallback must fail closed with 503/skip, not crash the parent. |
| Repair corrupt DBs only | Helps current data state | Does not prevent future native Chroma faults from killing server | **Rejected**: repair is recovery, not isolation. |
| Replace ChromaDB now | Could remove the native crash class | Larger backend migration with unrelated ranking/storage decisions | **Rejected**: out of scope for this containment fix. |
| Worker-only access | Contains native crashes to worker, matches current architecture | Requires touching root singleton, server lifecycle, routes, CLI, and tests | **Adopted**: only approach that enforces process isolation across all Chroma paths. |

### Key Decisions from Discussion

1. **Vector worker is already process-isolated**: The worker runs as a separate PID and speaks HTTP — Reason: keep this architecture and close the remaining direct-access gaps.
2. **Direct Chroma is not a production fallback**: If the worker is unavailable, server vector API returns `503` and non-critical background jobs skip/degrade — Reason: native segfault must not kill the server process.
3. **Server-owned RAG code uses the worker directly**: After worker start, server `os.environ["ANIMAWORKS_VECTOR_URL"]` points to the worker base URL — Reason: scheduler and repair calls should not call the server's own internal FastAPI route.
4. **Child processes keep internal server URLs**: Existing child `ANIMAWORKS_VECTOR_URL=http://127.0.0.1:<server>/api/internal/vector` remains — Reason: child traffic should still go through server auth/lifecycle and be proxied to the worker.
5. **Startup preflight no longer opens Chroma before worker isolation exists**: It runs after worker startup or inside a temporary worker context — Reason: restart recovery must not reproduce the crash before the server is available.
6. **Offline CLI operations start a temporary vector worker**: Commands that need Chroma and do not detect a running server use a managed temporary worker plus `HttpVectorStore` — Reason: CLI must follow the same isolation rule.
7. **Tests may use direct Chroma only with an explicit test/worker allow flag** — Reason: unit and E2E tests still need low-level store coverage without weakening production invariants.

### Changes by Module

| Module | Change Type | Description |
|--------|-------------|-------------|
| `core/memory/rag/singleton.py` | Modify | Add direct-Chroma guard. Return `HttpVectorStore` when `ANIMAWORKS_VECTOR_URL` is set. Without URL and without explicit allow flag, return `None` and log a warning instead of constructing `ChromaVectorStore`. |
| `core/memory/rag/store.py` | Modify | Add `ChromaVectorStore` constructor guard so production direct construction raises unless `ANIMAWORKS_ALLOW_DIRECT_CHROMA=1` is present. |
| `core/memory/rag/vector_worker_client.py` | Modify | Set `ANIMAWORKS_ALLOW_DIRECT_CHROMA=1` only for the worker subprocess; keep recursive `ANIMAWORKS_VECTOR_URL` removed from worker env. Add a temporary worker context for offline CLI/preflight use. |
| `core/memory/rag/vector_worker.py` | Modify | Ensure worker startup sets the direct-Chroma allow flag before loading vector stores. Add missing worker API parity for any operation still needed by CLI, including cosine migration checks if retained. |
| `server/app.py` | Modify | After long-lived vector worker starts, set server process `ANIMAWORKS_VECTOR_URL` to the worker URL and clear/reset it on shutdown. Keep `app.state.child_env_urls` pointing at server internal endpoints. |
| `server/routes/internal.py` | Modify | Remove direct fallback from all `/api/internal/vector/*` handlers. If worker is disabled/unavailable, return `503` for vector operations and never instantiate Chroma in the server process. |
| `cli/commands/server.py` | Modify | Move startup RAG preflight under worker isolation: either after worker startup in app lifespan or inside a temporary vector worker context before repair/reindex. |
| `core/memory/rag/repair_rebuild.py` | Modify | Keep behavior but require callers to provide worker-backed vector access; raise a clear error if no worker URL is available. |
| `cli/commands/index_cmd.py` | Modify | When server delegation is unavailable, start a temporary vector worker and route `get_vector_store()` through HTTP. Remove direct `ChromaVectorStore` type checks or expose needed checks via worker endpoint. |
| `cli/commands/anima_mgmt.py` | Modify | Replace direct rename cleanup `ChromaVectorStore` construction with worker-backed collection deletion or metadata reset plus reindex. |
| `tests/unit/core/memory/` | Modify/New | Cover singleton guard, store guard, worker allow env, and HTTP path selection. |
| `tests/unit/server/` | Modify/New | Cover internal vector route 503 behavior and no direct fallback. |
| `tests/unit/cli/` | Modify/New | Cover offline index/temp-worker behavior and rename cleanup no-direct behavior. |
| `tests/e2e/` | Modify/New | Add smoke coverage proving worker-only vector API path and server survival after worker death. |

#### Change 1: Guard the root vector store singleton

**Target**: `core/memory/rag/singleton.py`

```python
# Before
vector_url = os.environ.get("ANIMAWORKS_VECTOR_URL")
if vector_url:
    return _get_http_store(vector_url, anima_name)
store = ChromaVectorStore(persist_dir=persist_dir)

# After
vector_url = os.environ.get("ANIMAWORKS_VECTOR_URL")
if vector_url:
    return _get_http_store(vector_url, anima_name)
if os.environ.get("ANIMAWORKS_ALLOW_DIRECT_CHROMA") != "1":
    logger.warning("Vector store unavailable: direct ChromaDB access is disabled outside vector worker")
    return None
store = ChromaVectorStore(persist_dir=persist_dir)
```

#### Change 2: Fail closed in internal vector routes

**Target**: `server/routes/internal.py`

```python
# Before
proxied = await _try_vector_worker(request, "/query", body)
if proxied is not None:
    return proxied
store = get_vector_store(body.anima_name)

# After
proxied = await _require_vector_worker(request, "/query", body)
return proxied
```

`_require_vector_worker()` must return the proxied response or a `503` JSON response. It must not return `None`.

### Edge Cases

| Case | Handling |
|------|----------|
| Worker disabled in config | Server vector endpoints return `503 {"detail": "Vector worker unavailable"}`. Background jobs log and skip RAG work. |
| Worker crashes during request | Current request returns `503`; manager restarts worker and records existing native crash repair evidence. Server process remains alive. |
| Startup preflight finds suspect DBs | Repair/reindex runs through a worker-backed store. Startup logs per-anima results. |
| Startup preflight worker crashes | Server startup continues with RAG degraded for affected animas and repair evidence recorded; parent process must not segfault. |
| CLI index while server is running | Continue using server delegation to `/api/internal/vector` and `/api/internal/embed`. |
| CLI index while server is stopped | Start temporary vector worker and set `ANIMAWORKS_VECTOR_URL` for command duration. |
| Rename cleanup with missing worker | Reset `index_meta.json`, attempt worker-backed collection deletion, and continue with clear warning if vector cleanup is unavailable. |
| Unit tests that instantiate `ChromaVectorStore` | Set explicit direct-Chroma allow flag in fixtures or targeted test scopes only. |
| Existing callers expect `get_vector_store()` can return `None` | Preserve `None` behavior and keep existing skip/degrade paths where possible. |

## Implementation Plan

### Phase 1: Enforce worker-only Chroma construction

| # | Task | Target |
|---|------|--------|
| 1-1 | Add direct-Chroma allow helper and guard singleton fallback | `core/memory/rag/singleton.py` |
| 1-2 | Add constructor guard to `ChromaVectorStore` | `core/memory/rag/store.py` |
| 1-3 | Set allow flag in vector worker process and worker manager launch env | `core/memory/rag/vector_worker.py`, `core/memory/rag/vector_worker_client.py` |

**Completion condition**: Production code without `ANIMAWORKS_VECTOR_URL` and without `ANIMAWORKS_ALLOW_DIRECT_CHROMA=1` cannot instantiate native Chroma.

### Phase 2: Route server and internal APIs only through worker

| # | Task | Target |
|---|------|--------|
| 2-1 | Set server process `ANIMAWORKS_VECTOR_URL` to worker URL after worker startup | `server/app.py` |
| 2-2 | Remove direct fallback from every internal vector endpoint | `server/routes/internal.py` |
| 2-3 | Ensure worker unavailable means 503/degraded behavior, not direct Chroma | `server/routes/internal.py`, scheduler callers via singleton behavior |

**Completion condition**: Server process RAG operations use `HttpVectorStore`, and `/api/internal/vector/*` never imports/constructs `ChromaVectorStore`.

### Phase 3: Move preflight and offline CLI under worker isolation

| # | Task | Target |
|---|------|--------|
| 3-1 | Add temporary vector worker context for short-lived commands | `core/memory/rag/vector_worker_client.py` |
| 3-2 | Run startup preflight with worker-backed vector store | `cli/commands/server.py` or `server/app.py` |
| 3-3 | Route offline `index` through temporary worker | `cli/commands/index_cmd.py` |
| 3-4 | Replace direct rename cleanup with worker-backed deletion | `cli/commands/anima_mgmt.py` |

**Completion condition**: Startup repair, offline indexing, and rename cleanup no longer open Chroma in their own process.

### Phase 4: Tests and regression protection

| # | Task | Target |
|---|------|--------|
| 4-1 | Add unit tests for direct guard and worker env | `tests/unit/core/memory/` |
| 4-2 | Add route tests proving no direct fallback and 503 on worker unavailable | `tests/unit/server/` |
| 4-3 | Add CLI tests for server delegation and temporary worker path | `tests/unit/cli/` |
| 4-4 | Add E2E/smoke test for worker-only vector route and server survival after worker exit | `tests/e2e/` |
| 4-5 | Add static regression test for forbidden production `ChromaVectorStore(` usage outside allowed files | `tests/unit/` |

**Completion condition**: Targeted tests pass, and static guard prevents reintroducing production direct Chroma access.

## Scope

### In Scope

- Production worker-only invariant for all ChromaDB operations.
- Direct construction guard for `ChromaVectorStore`.
- Server process `ANIMAWORKS_VECTOR_URL` wiring to the vector worker.
- Removal of server internal vector direct fallback.
- Startup preflight under vector worker isolation.
- Offline CLI vector operations through server delegation or temporary worker.
- Unit and E2E/smoke tests proving the invariant.

### Out of Scope

- Replacing ChromaDB with a different vector database — Reason: backend migration is broader than crash containment.
- Changing RAG ranking/retrieval semantics — Reason: worker-only routing must preserve behavior.
- Rewriting repair policy, cooldowns, or quarantine retention — Reason: existing repair system remains the recovery layer.
- Fixing unrelated bootstrap artifact `needs_repair` states for `fuji`, `shion`, or `sae` — Reason: separate bootstrap lifecycle issue.
- Removing the vector worker architecture — Reason: user confirmed worker process isolation is the intended foundation.

## Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Worker unavailable makes RAG temporarily return 503/empty results | RAG degradation instead of full crash | Fail closed, log clearly, keep server alive, rely on worker restart/repair. |
| Startup preflight timing changes delay or skip repair | Corrupt DB may remain degraded longer | Keep preflight blocking when worker succeeds; continue startup only when repair cannot run safely. |
| Tests relying on direct Chroma fail | Test suite churn | Add explicit allow fixture for low-level store tests and static tests for production paths. |
| Temporary worker adds CLI startup cost | Slower offline `index` / repair commands | Only start temp worker when no running server delegation exists. |
| Server env mutation leaks past shutdown in embedded tests | Cross-test interference | Save/restore previous `ANIMAWORKS_VECTOR_URL` during app lifespan shutdown. |

## Acceptance Criteria

- [ ] `ChromaVectorStore` cannot be constructed in production unless `ANIMAWORKS_ALLOW_DIRECT_CHROMA=1` is set.
- [ ] `get_vector_store()` returns `HttpVectorStore` when `ANIMAWORKS_VECTOR_URL` is set and does not fall back to direct Chroma otherwise.
- [ ] The vector worker subprocess sets the direct-Chroma allow flag and remains the only production path that opens native Chroma.
- [ ] Server startup sets the server process vector URL to the worker URL before scheduler/anima background work can call RAG.
- [ ] Every `/api/internal/vector/*` endpoint proxies to the vector worker or returns `503`; no route body directly calls `get_vector_store()`.
- [ ] Startup RAG preflight and repair/reindex run through worker-backed vector access.
- [ ] `animaworks index` uses server delegation when available and a temporary vector worker when the server is absent.
- [ ] Anima rename RAG cleanup no longer constructs `ChromaVectorStore` directly.
- [ ] Simulated worker death does not terminate the server process and results in worker-unavailable/repair behavior.
- [ ] Static test prevents new production direct `ChromaVectorStore(` or `chromadb.PersistentClient` call sites outside allowed worker/store/test code.

## References

- `core/supervisor/_mgr_scheduler.py:363` — post-consolidation synaptic downscaling ran in server process before the crash.
- `core/memory/forgetting.py:255` — downscaling obtains vector store and scans Chroma collections.
- `core/memory/rag/store.py:381` — native Chroma `coll.get()` collection scan.
- `core/memory/rag/singleton.py:71` — HTTP vector URL branch followed by direct fallback.
- `server/app.py:548` — child-only vector URL setup leaves server environment unset.
- `server/routes/internal.py:205` — worker proxy can return `None` and allow direct fallback.
- `cli/commands/server.py:374` — startup preflight before worker/app startup.
- `cli/commands/index_cmd.py:242` — offline index path lacks worker delegation when server is down.
- `cli/commands/anima_mgmt.py:1252` — direct rename cleanup Chroma usage.
- `docs/implemented/20260518_rag-crash-preflight-vector-worker_implemented-20260518.md` — prior vector worker isolation and crash preflight work.
