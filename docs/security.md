# Security Architecture

AnimaWorks runs autonomous AI agents with tool access, persistent memory, and inter-agent communication. This creates a fundamentally different threat surface than stateless LLM wrappers — agents can read files, execute commands, send messages, and operate on schedules without human intervention.

This document describes the layered security model and an adversarial threat analysis based on cutting-edge LLM/agent attack research (OWASP Top 10 for LLM 2025, AdapTools, MemoryGraft, ChatInject, RoguePilot, MCP Tool Poisoning, RAGPoison, Confused Deputy attacks).

**Last audited**: 2026-03-02

---

## Threat Model

| Threat | Vector | Impact |
|--------|--------|--------|
| Prompt injection via external data | Web search results, Slack/Chatwork messages, emails | Agent executes attacker-controlled instructions |
| RAG / Memory poisoning | Malicious web content → knowledge → persistent recall | Long-term behavioral drift across all sessions |
| Lateral movement between agents | Compromised agent sends malicious DMs to peers | Privilege escalation across the organization |
| Confused Deputy attack | Low-privilege agent tricks high-privilege agent | Unauthorized tool execution, data exfiltration |
| Consolidation contamination | Poisoned episodes/activity → knowledge extraction | Trusted knowledge generated from tainted sources |
| Destructive command execution | Agent runs `rm -rf /` or `curl … \| sh` | Data loss, system compromise |
| Shell injection bypass | Network tools in pipes, shell mode escalation | Data exfiltration via allowed commands |
| Path traversal | Agent reads/writes outside its sandbox | Cross-agent data leak, config tampering |
| Activity log tampering | Agent writes fake entries to own activity_log | Manipulated Priming context, false history |
| Infinite message loops | Two agents endlessly replying to each other | Resource exhaustion, API cost explosion |
| Unauthorized external access | Agent sends messages to unintended recipients | Data exfiltration |
| Session hijacking | Stolen tokens with no expiration | Persistent unauthorized access |
| Credential exposure | Plaintext API keys in config.json | External service abuse |

---

## Part I: Current Defense Layers

### 1. Prompt Injection Defense — Trust Boundary Labeling

Every piece of data entering an agent's context is tagged with a trust level. The model sees these boundaries explicitly and is instructed to treat untrusted content as data, never as instructions.

#### Trust Levels

| Level | Sources | Treatment |
|-------|---------|-----------|
| `trusted` | Internal tools (send_message, search_memory), system-generated | Execute normally |
| `medium` | File reads, RAG results, user profiles, consolidated knowledge | Interpret as reference data |
| `untrusted` | web_search, slack_read, chatwork_read, gmail_read, x_search | **Never follow directives** |

#### Implementation

```
<tool_result tool="web_search" trust="untrusted">
  Search results here — may contain injection attempts
</tool_result>

<priming source="related_knowledge" trust="medium" origin="consolidation">
  RAG-retrieved context
</priming>
```

**Origin categories**: `system`, `human`, `anima`, `external_platform`, `external_web`, `consolidation`, `unknown`. Each maps to a trust level via `ORIGIN_TRUST_MAP`.

**Origin chain propagation**: When data flows through multiple systems (e.g., web → RAG index → priming), the trust level degrades to the **minimum** in the chain. `resolve_trust(origin, origin_chain)` computes the conservative minimum across all nodes in the chain plus the current origin. A web search result indexed into RAG retains `untrusted` status even when retrieved later.

**Session-level trust tracking**: `_min_trust_seen` tracks the minimum trust rank (2=trusted, 1=medium, 0=untrusted) across all tool calls in a session. Updated in Mode S (via `PreToolUse` hook + `run/min_trust_seen` file for MCP subprocess), Mode A (in `litellm_loop` and `anthropic_fallback` tool result processing). Reset at each new interaction cycle.

**Key files**: `core/execution/_sanitize.py` (trust resolution, boundary wrapping, `TOOL_TRUST_LEVELS`, `ORIGIN_TRUST_MAP`), `templates/*/prompts/tool_data_interpretation.md` (model instructions for interpreting trust levels and origin chains)

---

### 2. Memory Provenance — RAG and Knowledge Trust

#### write_memory_file origin propagation

When an agent writes to `knowledge/*.md`, the system checks `_min_trust_seen` for the session. If the session has encountered untrusted (rank 0) or medium (rank 1) tool results, an `origin` frontmatter is prepended:

- Rank 0 (untrusted) → `origin: external_web`
- Rank 1 (medium) → `origin: mixed`
- Rank 2 (trusted) → no origin tag (clean knowledge)

The origin is also passed to the RAG indexer, which stores it in ChromaDB chunk metadata.

For Mode S, `_min_trust_seen` is persisted to `run/min_trust_seen` so the MCP server subprocess can read it.

#### RAG indexer origin tracking

`index_file()` accepts an `origin` parameter. Chunk metadata in ChromaDB includes `metadata["origin"]` when set.

#### Priming Channel C trust splitting

When Priming retrieves related knowledge via RAG, each chunk's `origin` metadata is checked via `resolve_trust()`. Chunks are split into:

- **Trusted/medium** → `related_knowledge` (wrapped with `trust="medium"`)
- **Untrusted** → `related_knowledge_external` (wrapped with `trust="untrusted"`, `origin="external_platform"`)

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

Each agent's `permissions.md` can define a `## 実行できないコマンド` section listing additional blocked commands specific to that agent's role.

#### Layer 3: Per-Agent Section Required

A `## コマンド実行` or `## 実行できるコマンド` section must exist in `permissions.md` — default-deny for agents without explicit command permissions.

#### Layer 4: Per-Agent Allowlist

Only commands matching the agent's allowlist (from `permissions.md`) are permitted.

#### Layer 5: Path Traversal Detection

Command arguments are checked for path traversal patterns (`../`) that would escape the agent's sandbox.

**Pipeline segment checking**: Each segment of piped commands is checked independently — `safe_cmd | dangerous_cmd` is still blocked.

**Key files**: `core/tooling/handler_base.py` (blocklist `_BLOCKED_CMD_PATTERNS`, injection regex `_INJECTION_RE`), `core/tooling/handler_perms.py` (5-layer check pipeline `_check_command_permission`)

---

### 4. File Access Control — Sandboxed by Default

Each agent operates within its own directory (`~/.animaworks/animas/{name}/`). File access outside this sandbox requires explicit permission.

#### Protected Files and Directories (Immutable)

These cannot be written by the agent that owns them, preventing self-modification of security-critical settings:

- `permissions.md` — Tool and command allowlists
- `identity.md` — Core personality (immutable baseline)
- `bootstrap.md` — First-run instructions
- `activity_log/` — Activity log directory; only `ActivityLogger` (code-level) may append entries

#### Supervisor Access Matrix

Supervisors (managers) can access subordinate data with scoped permissions:

| Path | Direct Report | All Descendants |
|------|:---:|:---:|
| `activity_log/` | Read | Read |
| `state/current_task.md`, `pending.md` | — | Read |
| `state/task_queue.jsonl`, `pending/` | — | Read |
| `status.json` | Read/Write | Read |
| `identity.md` | — | Read |
| `injection.md` | Read/Write | Read |
| `cron.md`, `heartbeat.md` | Read/Write | — |

Descendant resolution uses BFS with cycle detection to prevent circular supervisor chains from causing infinite loops. Peers (same supervisor) can read each other's `activity_log/`.

**Key files**: `core/tooling/handler_base.py` (`_PROTECTED_FILES`, `_PROTECTED_DIRS`, `_is_protected_write`), `core/tooling/handler_perms.py` (`_check_file_permission`), `core/tooling/handler_memory.py` (memory read/write guards), `core/tooling/handler_org.py` (hierarchy checks)

---

### 5. Process Isolation

Each agent runs as an independent OS process managed by `ProcessSupervisor`:

- **Separate processes**: Crash in one agent doesn't affect others
- **Unix Domain Socket IPC**: Inter-process communication over filesystem sockets (not TCP), limiting network exposure
- **Independent locks**: Chat, inbox, and background tasks use separate asyncio locks — concurrent paths don't block each other
- **Socket directory**: `~/.animaworks/run/sockets/{name}.sock` with stale socket cleanup on startup

**Key files**: `core/supervisor/manager.py`, `core/supervisor/ipc.py`, `core/supervisor/runner.py`

---

### 6. Rate Limiting — 3-Layer Outbound Control

Autonomous agents must not spam. Three independent layers enforce message limits:

#### Layer 1: Per-Run (Session-Scoped)

- No duplicate DM to the same recipient within one execution session
- Max 2 distinct DM recipients per run
- One channel post per channel per session
- Cross-session channel post cooldown (configurable `channel_post_cooldown_s`)
- Tracked via in-memory sets (`_replied_to`, `_posted_channels`) and persisted to `run/replied_to.jsonl`

#### Layer 2: Cross-Run (Persistent)

- **Configurable messages per hour** per agent (`max_messages_per_hour`)
- **Configurable messages per day** per agent (`max_messages_per_day`)
- Computed from `activity_log` sliding window — survives process restarts
- `ack`, `error`, `system_alert` messages are exempt

#### Layer 3: Behavior Awareness (Self-Regulation)

Recent outbound messages (last 2 hours, max 3) are injected into the agent's system prompt via Priming. The agent can see its own recent sending pattern and self-regulate.

#### Cascade Prevention

- **Conversation depth limiter**: Configurable max turns between any agent pair within `depth_window_s`
- **Inbox rate limiter**: Cooldown period (`msg_heartbeat_cooldown_s`), cascade detection within `cascade_window_s`, per-sender rate limit during heartbeat
- **Fail-closed**: Depth check returns `False` on activity log read failure

**Key files**: `core/tooling/handler_comms.py` (per-run), `core/cascade_limiter.py` (cross-run, depth), `core/supervisor/inbox_rate_limiter.py` (inbox), `core/memory/priming.py` (`_collect_recent_outbound`)

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
- **48-byte URL-safe tokens** for sessions (cryptographically random)
- **Max 10 sessions per user** — oldest evicted on overflow
- **Session TTL** — configurable via `config.server.session_ttl_days` (default: 7). Expired sessions are rejected and removed in `validate_session()`.
- **Password change revokes sessions** — `change_password()` calls `revoke_all_sessions()` for the affected user
- **Cookie-based** session transport with middleware guard on `/api/` and `/ws` routes
- Config files (`config.json`, `auth.json`) saved with **0600 permissions**

#### Localhost Trust

When `trust_localhost` is enabled, requests from loopback addresses are authenticated automatically. Origin and Host header checks mitigate CSRF from browser-based attacks against localhost.

**Key files**: `core/auth/manager.py`, `server/app.py` (auth_guard middleware), `server/localhost.py`

---

### 8. Webhook Verification

Inbound webhooks from external platforms are cryptographically verified:

| Platform | Method | Replay Protection |
|----------|--------|-------------------|
| Slack | HMAC-SHA256 with signing secret | Timestamp check (5-minute window) |
| Chatwork | HMAC-SHA256 with webhook token | — |

Both use constant-time comparison (`hmac.compare_digest`).

**Key file**: `server/routes/webhooks.py`

---

### 9. SSRF Mitigation — Media Proxy

The media proxy (`/api/media/proxy`) fetches external images for display in the UI. It enforces:

- **HTTPS only** — no plaintext HTTP
- **Domain allowlist or open-with-scan** — configurable via `MediaProxyConfig.mode`
- **Private IP blocking** — blocks localhost, private ranges (RFC 1918), link-local, multicast, reserved
- **DNS resolution check** — resolves hostname and verifies the IP isn't private (prevents DNS rebinding)
- **Content-type validation** — only `image/jpeg`, `image/png`, `image/gif`, `image/webp`; SVG blocked
- **Magic bytes verification** — validates actual file format matches declared content-type
- **Size limit** — configurable `max_bytes` (default 5 MB)
- **Redirect validation** — redirect targets are re-validated; max redirects enforced
- **Per-IP rate limiting** — configurable per-client rate limit (default 30 req/min)
- **Security headers** — `X-Content-Type-Options: nosniff`

**Key file**: `server/routes/media_proxy.py`

---

### 10. Mode S (Agent SDK) Security

When running on Claude Agent SDK (Mode S), additional guardrails apply via `PreToolUse` hooks:

- **Bash command filtering**: Separate blocklist for SDK-executed commands (includes Chatwork CLI bypass prevention, network exfiltration tools, data upload patterns)
- **File write protection**: Validates write targets against protected file list and agent sandbox
- **File read restriction**: Blocks access to other agents' directories (except subordinate/peer activity_log, subordinate management files)
- **Output truncation**: Bash output capped at 10KB (head+tail), file reads default-limited to 500 lines, grep/glob results capped
- **Trust tracking**: `_SDK_TOOL_TRUST` mapping for SDK tool names (Read/Write/Edit/Bash → medium, WebFetch/WebSearch → untrusted), persisted to `run/min_trust_seen` for MCP subprocess access

**Key file**: `core/execution/_sdk_security.py`, `core/execution/_sdk_hooks.py`

---

### 11. Outbound Routing Security

The unified outbound router (`resolve_recipient()`) prevents agents from sending messages to unintended recipients:

1. Exact match against known agent names (case-sensitive)
2. User alias lookup (case-insensitive) from explicit config
3. Platform-prefixed recipients (`slack:USERID`, `chatwork:ROOMID`)
4. Slack User ID pattern match
5. Fallback case-insensitive agent match
6. **Unknown recipients → RecipientNotFoundError** (fail-closed)

Agents cannot send to arbitrary external addresses without explicit configuration.

**Key file**: `core/outbound.py`

---

### 12. Inter-Agent Message Security

#### Origin Chain in Messages

DMs between agents carry `origin_chain` metadata, built by `build_outgoing_origin_chain()`. This appends the session's origin and `ORIGIN_ANIMA` to the chain, enabling the receiver to assess the trust lineage of a message.

#### Inbox from_person Validation

`Messenger.receive()` validates `from_person` against `known_animas` (from `config.animas`). Messages with unknown `from_person` are rejected and logged as warnings, preventing spoofing as another agent.

#### Inbox Directory Permissions

Inbox directories are created with `0o700` permissions, restricting access to the owning user.

#### Channel Name Validation

Channel and peer names are validated against `_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")`, preventing path traversal in channel operations.

#### Board Channel Content Limits

Channel posts are limited to 10,000 characters via Pydantic validation (`max_length=10000`).

**Key files**: `core/messenger.py`, `core/tooling/handler_comms.py`, `core/tooling/handler_base.py` (`build_outgoing_origin_chain`)

---

## Part II: Adversarial Threat Analysis

This section documents an offensive security audit applying cutting-edge LLM/agent attack research to AnimaWorks' architecture. Each vulnerability is assessed from an attacker's perspective with concrete exploitation scenarios.

### Research Basis

| Source | Key Finding |
|--------|-------------|
| OWASP Top 10 for LLM 2025 | 10 vulnerability categories: prompt injection, sensitive info disclosure, supply chain, data poisoning, improper output handling, excessive agency, system prompt leakage, vector/embedding weaknesses, misinformation, unbounded consumption |
| AdapTools (arXiv 2602.20720) | Adaptive indirect prompt injection achieving 2.13x improvement in attack success rates against state-of-the-art defenses |
| MemoryGraft (arXiv 2512.16962) | Persistent agent compromise via poisoned experience retrieval |
| ChatInject (arXiv 2509.22830) | Role-based message manipulation achieving 32-52% attack success rates on agent frameworks |
| Confused Deputy (Quarkslab, promptfoo) | Low-privilege agents tricking high-privilege agents in multi-agent systems |

---

### Resolved Vulnerabilities

These vulnerabilities identified in the initial audit have been addressed:

| ID | Severity | Title | Resolution |
|----|----------|-------|------------|
| RAG-1 | Critical → Mitigated | Web → Knowledge → RAG Persistent Poisoning | `write_memory_file` propagates `_min_trust_seen` as origin frontmatter; RAG indexer stores origin in chunk metadata; Priming Channel C splits trusted/untrusted chunks |
| CON-1 | High → Mitigated | Consolidation Pipeline Contamination | `_has_external_origin_in_files()` checks source file origins; consolidation output downgraded to `consolidation_external` when inputs are external |
| MSG-1 | High → Mitigated | Inbox File-Level Spoofing | `from_person` validated against `known_animas`; inbox dirs set to `0o700` |
| BOARD-1 | High → Mitigated | Board Channel Broadcast Poisoning | Auth middleware protects channel POST; content limited to 10,000 chars; channel name regex validation |
| ALOG-1 | High → Resolved | Activity Log Tampering | `activity_log/` in `_PROTECTED_DIRS`; writes blocked via `_is_protected_write` |
| CMD-1 | High → Resolved | Shell Mode Network Exfiltration | `nc`, `ncat`, `socat`, `telnet`, `curl -d/--data`, `wget --post` in blocklist |
| AUTH-1 | High → Resolved | Perpetual Session Tokens | TTL check in `validate_session()` (default 7 days); `change_password()` calls `revoke_all_sessions()` |
| DEPUTY-1 | Medium → Mitigated | Confused Deputy Privilege Escalation | `origin_chain` metadata in inter-agent messages; `from_person` validation; `tool_data_interpretation` instructions for trust boundaries |

---

### Remaining Vulnerabilities

#### High

| ID | Category | Title | Status |
|----|----------|-------|--------|
| CFG-1 | Config | Plaintext Credential Storage | Partial mitigation (per-tool env_var fallback exists, but no first-class env-only mode) |

#### Medium

| ID | Category | Title | Status |
|----|----------|-------|--------|
| IPC-1 | Network | Socket File Permission Exposure | Not implemented (no `chmod 0o700` on Unix socket files) |
| WS-1 | Network | Voice WebSocket Audio Injection | Partial (60s buffer max, but no explicit max frame size or PCM format validation) |
| OB-1 | Rate Limit | Multi-Agent Distributed Spam | Not implemented (per-sender rate, no per-recipient aggregate) |
| PR-1 | Memory | PageRank Graph Manipulation | Not implemented (no trust-weighted PageRank) |
| SKILL-1 | Memory | Skill Description Keyword Stuffing | Not implemented (no anti-stuffing in 3-tier matching) |
| PI-1 | Prompt | New Tool Trust Registration Gap | Not implemented (unlisted tools fall back to `untrusted`, but no CI assertion) |
| CMD-2 | Execution | Denied List Partial Match Bypass | Not implemented (substring matching; no `shutil.which()` resolution) |
| EXT-1 | External | Indirect Prompt Injection via External Sources | Mitigated by trust labeling; no additional regex pattern filter |
| LEAK-1 | Info Disclosure | System Prompt Leakage | Partial (`tool_data_interpretation` has trust rules but no explicit anti-leak instruction) |

#### Low

| ID | Category | Title | Status |
|----|----------|-------|--------|
| AUTH-2 | Auth | Localhost Trust Over-Permission | Not implemented (no `X-Forwarded-For` awareness) |
| FILE-1 | File | Symlink Following in allowed_dirs | Not implemented (uses `resolve()` without strict symlink rejection) |
| WS-2 | Network | WebSocket JSON Schema Laxity | Not implemented (no Pydantic validation for voice WebSocket JSON) |
| OB-2 | Rate Limit | Activity Log Write Bypass | Not implemented (send does not depend on activity log success) |
| ACCESS-1 | Memory | RAG Access Count Inflation | Not implemented (no access_count cap) |

---

### Vulnerability Details (Remaining)

#### CFG-1: Plaintext Credential Storage

`CredentialConfig` stores API keys in `config.json`. File has `0600` permissions, but backup tools, NFS mounts, or same-user processes can read it. Per-tool `env_var` fallback exists (e.g., `ANTHROPIC_API_KEY`), but `CredentialConfig` has no first-class env-only mode.

**Recommendation**: Add env-only credential mode; add `config.json` to agent-unreadable paths.

#### IPC-1: Socket File Permission Exposure

Unix sockets created by `asyncio.start_unix_server()` without explicit `chmod`. On multi-user systems, another user could connect.

**Recommendation**: `os.chmod(socket_path, 0o700)` after creation.

#### WS-1: Voice WebSocket Audio Injection

Audio buffer is capped at 60 seconds (`MAX_AUDIO_BUFFER_BYTES`), clearing on overflow. However, no explicit max frame size for individual WebSocket binary messages, and no PCM format validation before STT processing.

**Recommendation**: Max frame size enforcement; PCM format validation.

#### OB-1: Multi-Agent Distributed Spam

Rate limiting is per-agent (hour/day) and per-pair (depth). Multiple agents can independently target the same external recipient.

**Recommendation**: Global per-recipient rate limit across all agents.

---

## Part III: Defense-in-Depth Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    External Data                        │
│          (web, slack, email, board, DM, etc.)            │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Trust Boundary     │  ← untrusted/medium/trusted tags
              │  Labeling           │  ← origin chain propagation
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Auth & Session     │  ← Argon2id, TTL-enforced sessions
              │  Management         │  ← webhook HMAC verification
              └──────────┬──────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
┌────▼────┐      ┌──────▼──────┐     ┌──────▼──────┐
│ Command │      │ File Access │     │  Outbound   │
│ Security│      │   Control   │     │  Rate Limit │
│ (5-layer│      │ (sandbox +  │     │  (3-layer + │
│  check) │      │  ACL matrix)│     │   cascade)  │
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

Each layer operates independently. A failure in one layer is caught by others — prompt injection that bypasses trust labeling still faces command blocklists, file sandboxing, rate limits, and memory provenance tracking.

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
| 4 | CFG-1 | Support env-var-only credential mode; add `config.json` to agent-unreadable paths | M |
| 5 | WS-1 | Maximum frame size + PCM format validation for voice WebSocket | S |
| 6 | OB-1 | Global per-recipient rate limit across all agents | S |
| 7 | LEAK-1 | Anti-leak instruction in system prompt; output monitoring for prompt fragments | S |
| 8 | CMD-2 | `shutil.which()` resolution + basename comparison for denied commands | S |

### Phase 3: Defense-in-Depth (long-term)

| Priority | ID | Action | Effort |
|:---:|------|--------|:---:|
| 9 | PR-1 | Trust-weighted PageRank (untrusted-origin nodes get reduced activation) | M |
| 10 | EXT-1 | Injection pattern regex filter on external data before LLM ingestion | M |
| 11 | AUTH-2 | Document reverse proxy guidance; add `X-Forwarded-For` support | S |
| 12 | ALOG+ | Append-only hash chain for activity log integrity | M |
| 13 | MSG+ | HMAC message signing between agents (cryptographic spoofing prevention) | L |

Effort scale: XS = less than 1 hour, S = 1-4 hours, M = 4-16 hours, L = more than 16 hours

---

## Related Documents

| Document | Description |
|----------|-------------|
| [Provenance Foundation](implemented/20260228_provenance-1-foundation.md) | Trust resolution and origin categories |
| [Input Boundary Labeling](implemented/20260228_provenance-2-input-boundary.md) | Tool result and priming trust tagging |
| [Trust Propagation](implemented/20260228_provenance-3-propagation.md) | Origin chain across data flows |
| [RAG Provenance](implemented/20260228_provenance-4-rag-provenance.md) | Trust tracking in vector search |
| [Mode S Trust](implemented/20260228_provenance-5-mode-s-trust.md) | Agent SDK security hooks |
| [Command Injection Fix](implemented/20260228_security-command-injection-fix.md) | Pipe-to-interpreter and newline injection |
| [Path Traversal Fix](implemented/20260228_security-path-traversal-fix.md) | common_knowledge and create_anima path validation |
| [Memory Write Security](implemented/20260215_memory-write-security-20260216.md) | Protected files and cross-mode hardening |
