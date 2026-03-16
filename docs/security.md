# Security Architecture

AnimaWorks runs autonomous AI agents with tool access, persistent memory, and inter-agent communication. This creates a fundamentally different threat surface than stateless LLM wrappers — agents can read files, execute commands, send messages, and operate on schedules without human intervention.

This document describes the layered security model and an adversarial threat analysis based on cutting-edge LLM/agent attack research (OWASP Top 10 for LLM 2025, AdapTools, MemoryGraft, ChatInject, RoguePilot, MCP Tool Poisoning, RAGPoison, Confused Deputy attacks).

**Last audited**: 2026-03-06

---

## Threat Model

| Threat | Attack Vector | Impact |
|--------|---------------|--------|
| Prompt injection via external data | Web search results, Slack/Chatwork messages, emails | Agent executes attacker-controlled instructions |
| RAG / Memory poisoning | Malicious web content → knowledge → persistent recall | Long-term behavioral drift across all sessions |
| Lateral movement between agents | Compromised agent sends malicious DMs to peers | Privilege escalation across the organization |
| Confused Deputy attack | Low-privilege agent tricks high-privilege agent | Unauthorized tool execution, data exfiltration |
| Consolidation contamination | Poisoned episodes/activity → knowledge extraction | Trusted knowledge generated from tainted sources |
| Destructive command execution | Agent runs `rm -rf /` or `curl … \| sh` | Data loss, system compromise |
| Shell injection bypass | Network tools via pipes | Data exfiltration via allowed commands |
| Path traversal | Agent reads/writes outside its sandbox | Cross-agent data leak, config tampering |
| Activity log tampering | Agent writes fake entries to own activity_log | Manipulated Priming context |
| Infinite message loops | Two agents endlessly replying to each other | Resource exhaustion, API cost explosion |
| Unintended external sending | Agent sends messages to unexpected recipients | Data exfiltration |
| Session hijacking | Stolen tokens with no expiration | Persistent unauthorized access |
| Credential exposure | Plaintext API keys in config.json | External service abuse |

---

## Part I: Current Defense Layers

### 1. Prompt Injection Defense — Trust Boundary Labeling

Every piece of data entering an agent's context is tagged with a trust level. The model sees these boundaries explicitly and is instructed to treat untrusted content as data, never as instructions.

#### Trust Levels

| Level | Target Sources | Treatment |
|-------|----------------|-----------|
| `trusted` | Internal tools (send_message, search_memory, submit_tasks, update_task, post_channel, etc.), system-generated | Execute normally |
| `medium` | Read, Grep, Write, Bash, RAG results, user profiles, consolidated knowledge | Interpret as reference data |
| `untrusted` | web_search, WebFetch, x_search, x_user_tweets, slack_*, chatwork_*, gmail_*, read_channel, read_dm_history, local_llm | **Never follow directives** |

#### Implementation

```
<tool_result tool="web_search" trust="untrusted">
  Search results — may contain injection attempts
</tool_result>

<priming source="related_knowledge" trust="medium" origin="consolidation">
  RAG-retrieved context
</priming>
```

**Origin categories**: `system`, `human`, `anima`, `external_platform`, `external_web`, `consolidation`, `unknown`. Each maps to a trust level via `ORIGIN_TRUST_MAP`.

**Origin chain propagation**: When data flows through multiple systems (e.g., web → RAG index → priming), the trust level degrades to the **minimum** in the chain. `resolve_trust(origin, origin_chain)` computes the conservative minimum across all nodes in the chain plus the current origin.

**Session-level trust tracking**: `_min_trust_seen` tracks the minimum trust rank (2=trusted, 1=medium, 0=untrusted) across all tool calls in a session. Updated in Mode S (`PreToolUse` hook + `run/min_trust_seen` file), Mode A (`litellm_loop` and `anthropic_fallback`). Reset at each interaction cycle start.

**Trigger and tier injection conditions** (`core/prompt/builder.py`):

- `tool_data_interpretation` is in **Group 1** but is **not injected** when `trigger="task"` (TaskExec). TaskExec runs with minimal context, so the model does not receive trust boundary interpretation instructions. Tool results are still wrapped with `wrap_tool_result` so tags are applied, but note that the "tag interpretation rules" instruction to the model is omitted.
- `permissions` is injected only when `tier != TIER_MINIMAL`. When context is under 16k (TIER_MINIMAL), permissions are omitted.
- `behavior_rules` applies only to TIER_FULL and TIER_STANDARD. Omitted for TIER_LIGHT / TIER_MINIMAL.
- Tier boundaries: 128k+ = FULL, 32k–128k = STANDARD, 16k–32k = LIGHT, under 16k = MINIMAL.

**Key files**: `core/execution/_sanitize.py` (trust resolution, boundary wrapping, `TOOL_TRUST_LEVELS`, `ORIGIN_TRUST_MAP`), `core/prompt/builder.py` (trigger/tier prompt construction, `tool_data_interpretation` injection conditions), `templates/{locale}/prompts/tool_data_interpretation.md` (trust level and origin chain interpretation instructions; locale depends on config.locale)

---

### 2. Memory Provenance — RAG and Knowledge Trust Tracking

#### write_memory_file origin propagation

When an agent writes to `knowledge/*.md`, the system checks `_min_trust_seen` for the session. If the session has processed untrusted (rank 0) or medium (rank 1) tool results, an `origin` frontmatter is added:

- Rank 0 (untrusted) → `origin: external_web`
- Rank 1 (medium) → `origin: mixed`
- Rank 2 (trusted) → no origin tag (clean knowledge)

The origin is passed to the RAG indexer and stored in ChromaDB chunk metadata.

#### RAG indexer origin tracking

`index_file()` accepts an `origin` parameter and stores it as `metadata["origin"]` in chunk metadata.

#### Priming Channel C trust splitting

When Priming retrieves related knowledge via RAG, each chunk's `origin` metadata is evaluated with `resolve_trust()`. Chunks are split into:

- **trusted/medium** → `related_knowledge` (wrapped with `trust="medium"`)
- **untrusted** → `related_knowledge_external` (wrapped with `trust="untrusted"`, `origin="external_platform"`)

Budget prioritizes trusted/medium content first; untrusted content fills remaining budget.

#### Consolidation origin tracking

Daily consolidation reads YAML frontmatter `origin:` from source knowledge files. If any source has external origin (`external_web`, `mixed`, `consolidation_external`), the consolidated output is downgraded to `origin: consolidation_external` (resolves to `untrusted`).

**Key files**: `core/tooling/handler_memory.py` (write_memory_file origin propagation), `core/memory/rag/indexer.py` (origin in chunk metadata), `core/memory/priming.py` (Channel C trust splitting), `core/memory/consolidation.py` (origin chain tracking)

---

### 3. Command Execution Security — 5-Layer Defense

Agents can execute shell commands. Five independent layers prevent abuse:

#### Layer 1: Shell Injection Detection

Blocks shell metacharacters that could chain or inject commands:

- Semicolons (`;`), backticks (`` ` ``), newlines (`\n`)
- Command substitution (`$()`, `${}`, `$VAR`)

#### Layer 2: Hardcoded Blocklist

Pattern-matched commands that are **always** blocked regardless of permissions:

| Pattern | Reason |
|---------|--------|
| `rm -rf`, `rm -r` | Recursive deletion |
| `mkfs` | Filesystem creation |
| `dd of=/dev/` | Direct disk write |
| `curl\|sh`, `wget\|sh` | Remote code execution |
| `\| sh`, `\| bash`, `\| python`, `\| perl`, `\| ruby`, `\| node` | Pipe to interpreter |
| `nc`, `ncat`, `socat`, `telnet` | Network exfiltration tools |
| `curl -d/-F/-T`, `curl --data`, `wget --post` | Data upload / exfiltration |
| `chmod *7*` | World-writable permissions |
| `shutdown`, `reboot` | System shutdown |
| `> /dev/sd*`, `> /dev/nvme*`, `> /etc/` | Device/system file redirect |

#### Layer 2.5: Per-Agent Denied Commands

Each agent's `permissions.json` can define `commands.denied_commands` for additional blocked commands.

#### Layer 3: Command Permissions Model

`permissions.json` uses an "Open by Default, Deny by Exception" model. When `commands.allow_all` is true (default), all commands are permitted except those in `denied_commands` and the hardcoded blocklist. When false, only commands in `commands.allowlist` are permitted.

#### Layer 4: Per-Agent Allowlist

When `commands.allow_all` is false, only commands matching the agent's allowlist are permitted.

#### Layer 5: Path Traversal Detection

Command arguments are checked for path traversal patterns (`../`).

**Pipeline segment checking**: Each segment of piped commands is checked independently.

**Key files**: `core/tooling/handler_base.py` (`_BLOCKED_CMD_PATTERNS`, `_INJECTION_RE`), `core/tooling/handler_perms.py` (`_check_command_permission`)

---

### 4. File Access Control — Sandboxed by Default

Each agent operates within its own directory (`~/.animaworks/animas/{name}/`). File access is controlled by `file_roots` in `permissions.json` — default `["/"]` grants full access within the anima directory; restricted roots limit writable paths. Mode C (Codex) sandbox adapts dynamically: `file_roots: ["/"]` → `danger-full-access`; restricted → `workspace-write` with dynamic `writable_roots`.

#### Protected Files and Directories (Immutable)

These cannot be written by the agent that owns them:

- `permissions.json` — Pydantic-validated tool, file, and command permissions (replaces legacy `permissions.md`)
- `permissions.md` — Legacy file; protected when present (auto-migrated to JSON)
- `identity.md` — Core personality (immutable baseline)
- `bootstrap.md` — First-run instructions
- `activity_log/` — Activity log directory; only `ActivityLogger` (code-level) may append entries

#### Supervisor Access Matrix

| Path | Direct Report | All Descendants |
|------|:---:|:---:|
| `activity_log/` | Read | Read |
| `state/current_state.md`, `pending.md` | — | Read |
| `state/task_queue.jsonl`, `pending/` | — | Read |
| `status.json` | Read/Write | Read |
| `identity.md` | — | Read |
| `injection.md` | Read/Write | Read |
| `cron.md`, `heartbeat.md` | Read/Write | — |

Descendant resolution uses BFS with cycle detection. Peers (same supervisor) can read each other's `activity_log/`.

**Key files**: `core/tooling/handler_base.py` (`_PROTECTED_FILES`, `_PROTECTED_DIRS`, `_is_protected_write`), `core/tooling/handler_perms.py` (`_check_file_permission`)

---

### 5. Process Isolation

Each agent runs as an independent OS process:

- **Process isolation**: Crash in one agent doesn't affect others
- **Unix Domain Socket IPC**: Inter-process communication over filesystem sockets (no TCP)
- **Independent locks**: Chat, Inbox, and background tasks use separate asyncio locks
- **Socket directory**: `~/.animaworks/run/sockets/{name}.sock` with stale socket cleanup on startup

**Key files**: `core/supervisor/manager.py`, `core/supervisor/ipc.py`, `core/supervisor/runner.py`

---

### 6. Rate Limiting — 3-Layer Outbound Control

#### Layer 1: Per-Run (Session-Scoped)

- No duplicate DM to the same recipient
- Max 2 distinct DM recipients per execution
- One channel post per channel per session
- Cross-session channel post cooldown (`channel_post_cooldown_s`)
- Persisted to `run/replied_to.jsonl`

#### Layer 2: Cross-Run (Persistent)

- **Configurable per-agent send limits** (hourly and daily)
- Computed from `activity_log` sliding window
- `ack`, `error`, `system_alert` messages are exempt

#### Layer 3: Behavior Awareness (Self-Regulation)

Recent outbound messages (last 2 hours, max 3) are injected into the system prompt via Priming.

#### Cascade Prevention

- **Conversation depth limiter**: Configurable max turns within `depth_window_s`
- **Inbox rate limiter**: Cooldown, cascade detection, per-sender rate limit
- **Fail-closed**: Returns `False` on activity log read failure

**Key files**: `core/tooling/handler_comms.py`, `core/cascade_limiter.py`, `core/supervisor/inbox_rate_limiter.py`, `core/memory/priming.py`

---

### 7. Authentication & Session Management

#### Auth Modes

| Mode | Use Case |
|------|----------|
| `local_trust` | Development — localhost requests bypass auth |
| `password` | Single-user password protection |
| `multi_user` | Multiple users with individual accounts |

#### Session Security

- **Argon2id** password hashing (memory-hard, side-channel resistant)
- **48-byte URL-safe tokens** (cryptographically random)
- **Max 10 sessions per user** — oldest evicted on overflow
- **Session TTL** — `config.server.session_ttl_days` (default: 7 days). Expired sessions are rejected and removed in `validate_session()`.
- **Password change revokes sessions** — `change_password()` calls `revoke_all_sessions()` to invalidate all sessions
- **Cookie-based** session transport with middleware guard on `/api/` and `/ws` routes
- Config files saved with **0600 permissions**

#### Localhost Trust

When `trust_localhost` is enabled, requests from loopback addresses are authenticated automatically. Origin and Host header checks mitigate CSRF.

**Key files**: `core/auth/manager.py`, `server/app.py`, `server/localhost.py`

---

### 8. Webhook Verification

| Platform | Method | Replay Protection |
|----------|--------|-------------------|
| Slack | HMAC-SHA256 (signing secret) | Timestamp check (5-minute window) |
| Chatwork | HMAC-SHA256 (webhook token) | — |

Both use constant-time comparison (`hmac.compare_digest`).

**Key file**: `server/routes/webhooks.py`

---

### 9. SSRF Mitigation — Media Proxy

The media proxy (`/api/media/proxy`) fetches external images for display in the UI:

- **HTTPS only**
- **Domain allowlist or open-with-scan** — configurable via `MediaProxyConfig.mode`
- **Private IP blocking** — localhost, RFC 1918, link-local, multicast, reserved
- **DNS resolution check** — prevents DNS rebinding
- **Content-Type validation** — only `image/jpeg`, `image/png`, `image/gif`, `image/webp`; SVG blocked
- **Magic bytes verification** — validates actual file format matches declared content-type
- **Size limit** — `max_bytes` (default 5 MB)
- **Redirect validation** — redirect targets re-validated; max redirect count enforced
- **Per-IP rate limiting** — configurable (default 30 req/min)
- **Security headers** — `X-Content-Type-Options: nosniff`

**Key file**: `server/routes/media_proxy.py`

---

### 10. Mode S (Agent SDK) Security

When running on Claude Agent SDK (Mode S), additional guardrails apply via `PreToolUse` hooks:

- **Bash command filtering**: Separate blocklist for SDK (includes Chatwork CLI bypass prevention, network exfiltration tools, data upload patterns)
- **File write protection**: Validates against protected file list and sandbox
- **File read restriction**: Blocks access to other agents' directories (except subordinate/peer activity_log, subordinate management files)
- **Output truncation**: Bash output capped at 10KB; file reads default-limited to 500 lines; grep/glob also limited
- **Trust tracking**: `_SDK_TOOL_TRUST` mapping; persisted to `run/min_trust_seen`

**Key files**: `core/execution/_sdk_security.py`, `core/execution/_sdk_hooks.py`

---

### 11. Outbound Routing Security

`resolve_recipient()` prevents agents from sending to unintended recipients:

1. Exact match against known agent names (case-sensitive)
2. User alias lookup (case-insensitive)
3. Platform-prefixed recipients
4. Slack User ID pattern match
5. Case-insensitive agent name match
6. **Unknown recipients → RecipientNotFoundError** (fail-closed)

**Key file**: `core/outbound.py`

---

### 12. Inter-Agent Message Security

#### Message Origin Chain

DMs carry `origin_chain` metadata, built by `build_outgoing_origin_chain()`. Receivers can evaluate the trust lineage of messages.

#### Inbox from_person Validation

`Messenger.receive()` validates `from_person` against `known_animas` (`config.animas`). Unknown `from_person` is rejected and logged.

#### Inbox Directory Permissions

Inbox directories are created with `0o700` permissions.

#### Channel Name Validation

`_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")` prevents path traversal.

#### Board Channel Content Limits

Channel posts are limited to `max_length=10000` via Pydantic.

**Key files**: `core/messenger.py`, `core/tooling/handler_comms.py`, `core/tooling/handler_base.py`

---

## Part II: Adversarial Threat Analysis

### Resolved Vulnerabilities

Vulnerabilities identified in the initial audit that have been addressed:

| ID | Severity | Title | Resolution |
|----|----------|-------|------------|
| RAG-1 | Critical → Mitigated | Web → Knowledge → RAG Persistent Poisoning | `write_memory_file` propagates `_min_trust_seen` as origin frontmatter; RAG indexer stores origin in chunk metadata; Priming Channel C splits trusted/untrusted |
| CON-1 | High → Mitigated | Consolidation Pipeline Contamination | `_has_external_origin_in_files()` checks source file origins; output downgraded to `consolidation_external` when external origin present |
| MSG-1 | High → Mitigated | Inbox File-Level Spoofing | `from_person` validated against `known_animas`; inbox dirs set to `0o700` |
| BOARD-1 | High → Mitigated | Board Channel Broadcast Poisoning | Auth middleware protects channel POST; content limited to 10,000 chars; channel name regex validation |
| ALOG-1 | High → Resolved | Activity Log Tampering | `activity_log/` in `_PROTECTED_DIRS`; writes blocked via `_is_protected_write` |
| CMD-1 | High → Resolved | Shell Mode Network Exfiltration | `nc`, `ncat`, `socat`, `telnet`, `curl -d/--data`, `wget --post` added to blocklist |
| AUTH-1 | High → Resolved | Perpetual Session Tokens | TTL check in `validate_session()` (default 7 days); `change_password()` calls `revoke_all_sessions()` |
| DEPUTY-1 | Medium → Mitigated | Confused Deputy Privilege Escalation | `origin_chain` metadata in messages; `from_person` validation; trust boundary instructions in `tool_data_interpretation` |

---

### Remaining Vulnerabilities

#### High

| ID | Category | Title | Status |
|----|----------|-------|--------|
| CFG-1 | Config | Plaintext Credential Storage | Partial (per-tool env_var fallback exists; no env-only mode in CredentialConfig) |

#### Medium

| ID | Category | Title | Status |
|----|----------|-------|--------|
| IPC-1 | Network | Socket File Permissions | Not implemented (no `chmod 0o700` on Unix sockets) |
| WS-1 | Network | Voice WebSocket Audio Injection | Partial (60s buffer max; no max frame size or PCM format validation) |
| OB-1 | Rate Limit | Multi-Agent Distributed Spam | Not implemented (per-sender rate limit only; no per-recipient aggregate) |
| PR-1 | Memory | PageRank Graph Manipulation | Not implemented (no trust-weighted PageRank) |
| SKILL-1 | Memory | Skill Description Keyword Stuffing | Not implemented (no mitigation in 3-tier matching) |
| PI-1 | Prompt | Tool Trust Level Registration Gap | Not implemented (unlisted tools fall back to untrusted; no CI check) |
| CMD-2 | Execution | Denied List Partial Match Bypass | Not implemented (substring matching; no `shutil.which()` resolution) |
| EXT-1 | External | Indirect Injection via External Sources | Mitigated by trust labeling; no additional regex filter |
| LEAK-1 | Info Disclosure | System Prompt Leakage | Partial (trust rules exist; no explicit anti-leak instruction) |

#### Low

| ID | Category | Title | Status |
|----|----------|-------|--------|
| AUTH-2 | Auth | Localhost Trust Over-Permission | Not implemented (no `X-Forwarded-For` support) |
| FILE-1 | File | Symlink Following in allowed_dirs | Not implemented (uses `resolve()`; no strict symlink rejection) |
| WS-2 | Network | WebSocket JSON Schema Laxity | Not implemented (no Pydantic validation for voice WebSocket JSON) |
| OB-2 | Rate Limit | Activity Log Write Bypass | Not implemented (send does not depend on activity log success) |
| ACCESS-1 | Memory | RAG Access Count Inflation | Not implemented (no access_count cap) |

---

## Part III: Defense-in-Depth Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    External Data                        │
│          (Web, Slack, email, Board, DM, etc.)            │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Trust Boundary     │  ← untrusted/medium/trusted tags
              │  Labeling           │  ← origin chain propagation
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Auth & Session     │  ← Argon2id, TTL-enforced sessions
              │  Management         │  ← Webhook HMAC verification
              └──────────┬──────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
│ Command │      │ File Access  │     │  Outbound   │
│ Security│      │   Control    │     │  Rate Limit │
│ (5-layer│      │ (sandbox +   │     │  (3-layer + │
│  check) │      │  ACL)        │     │   cascade)  │
└────┬────┘      └──────┬──────┘     └──────┬──────┘
     │                  │                   │
     └───────────────┐  │  ┌────────────────┘
                     │  │  │
              ┌──────▼──▼──▼────────┐
              │  Memory Provenance  │  ← origin tracking in RAG/knowledge
              │  (trust chain)      │  ← Channel C trust splitting
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Process Isolation  │  ← per-agent OS process
              │  (Unix sockets)     │  ← independent locks
              └─────────────────────┘
```

Each layer operates independently. A failure in one layer is caught by others.

---

## Part IV: Remediation Roadmap

### Phase 1: Quick Wins (XS effort)

| Priority | ID | Action | Effort |
|:---:|------|--------|:---:|
| 1 | IPC-1 | `chmod 0o700` on socket files and `run/` directory | XS |
| 2 | PI-1 | CI check for tool trust level registration completeness | XS |
| 3 | ACCESS-1 | Access count cap + per-session deduplication | XS |

### Phase 2: Hardening (S–M effort)

| Priority | ID | Action | Effort |
|:---:|------|--------|:---:|
| 4 | CFG-1 | Env-var-only credential mode; agent-unreadable paths for `config.json` | M |
| 5 | WS-1 | Max frame size + PCM format validation | S |
| 6 | OB-1 | Per-recipient rate limit across all agents | S |
| 7 | LEAK-1 | Anti-leak instruction in system prompt; output monitoring | S |
| 8 | CMD-2 | `shutil.which()` resolution + basename comparison | S |

### Phase 3: Defense-in-Depth (long-term)

| Priority | ID | Action | Effort |
|:---:|------|--------|:---:|
| 9 | PR-1 | Trust-weighted PageRank | M |
| 10 | EXT-1 | Injection pattern regex filter on external data | M |
| 11 | AUTH-2 | Reverse proxy guidance; `X-Forwarded-For` support | S |
| 12 | ALOG+ | Append-only hash chain for activity log | M |
| 13 | MSG+ | HMAC message signing between agents | L |

Effort scale: XS = less than 1 hour, S = 1–4 hours, M = 4–16 hours, L = more than 16 hours

---

## Related Documents

| Document | Description |
|----------|-------------|
| [Provenance Foundation](specs/20260228_provenance-1-foundation.md) | Trust resolution and origin categories |
| [Input Boundary Labeling](specs/20260228_provenance-2-input-boundary.md) | Tool result and priming trust tagging |
| [Trust Propagation](specs/20260228_provenance-3-propagation.md) | Origin chain across data flows |
| [RAG Provenance](specs/20260228_provenance-4-rag-provenance.md) | Trust tracking in vector search |
| [Mode S Trust](specs/20260228_provenance-5-mode-s-trust.md) | Agent SDK security hooks |
| [Command Injection Fix](specs/20260228_security-command-injection-fix.md) | Pipe and newline injection |
| [Path Traversal Fix](specs/20260228_security-path-traversal-fix.md) | common_knowledge and create_anima path validation |
| [Memory Write Security](specs/20260215_memory-write-security-20260216.md) | Protected files and cross-mode hardening |
