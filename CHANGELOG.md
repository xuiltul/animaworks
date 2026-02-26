# Changelog

All notable changes to AnimaWorks will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- voice chat bubble integration, thinking delta, mobile touch improvements
- add Codex SDK execution mode C for OpenAI Codex CLI integration
- thinking UI integration — all frontends + persistence + voice indicator
- add thinking streaming events and collapsible UI component
- integrate adaptive thinking into S/A execution engines
- add thinking_effort schema, resolve_max_tokens, adaptive thinking helpers
- add prompt i18n (ja/en) — locale-aware template system
- add status.json hot-reload via IPC without process restart
- enable extended thinking for Bedrock Claude models
- add business theme UI with CSS design tokens and workspace org dashboard
- add voice chat system (STT + TTS + WebSocket)
- add AWS Bedrock provider support for LiteLLM execution
- expand supervisor file access permissions for subordinate management
- enable WebFetch/WebSearch native tools in S mode
- add bustup image overlay on avatar click in chat page

### Fixed
- HTTPS reverse proxy auth loop and asset reconciliation infinite retry
- preserve thinking_blocks in LiteLLM tool-call iterations
- resolve frontend regressions and improve workspace org dashboard
- sanitize thinking text rendering to prevent XSS
- resolve review revision findings for prompt i18n
- use output_config instead of reasoning_effort for Anthropic SDK
- resolve all 98 test regressions from i18n restructuring + main rebase
- update assisted.py adaptive thinking + fix config_reader test mocks
- resolve review findings — cli.py path regression + DRY _get_locale
- 91件の失敗テストを現行実装に追従させる
- address review findings — disposeOffice on view switch, Lucide createIcons timing
- raise compaction threshold ceiling from 0.95 to 0.98
- web_search dispatch bug and context threshold auto-scaling
- add builder.py to silent-pass allowlist for org tree status.json fallback
- board mobile scroll and channel switching
- add missing tool result for ToolExecutionError in serial path
- improve error handling with custom exception hierarchy
- resolve type safety issues across codebase

### Changed
- move outbound section to PrimingEngine

## [0.4.1] - 2026-02-25

### Added

- Cron sessions now use heartbeat-equivalent context (full identity + memory + org); removed separate heartbeat trigger from cron
- Chatwork outbound messaging re-enabled with `chatwork send` support
- Subagent CLI execution skill (`common_skills/subagent-cli.md`) for delegating shell tasks to Claude Code subprocess

### Fixed

- Chatwork send rejects with clear error when WRITE token is not configured (previously silent failure)
- Skill/procedure invocation instruction in `memory_guide` updated to reference `skill` tool (progressive disclosure)

## [0.4.0] - 2026-02-25

### Added

#### Execution & Architecture
- 3-path execution separation — Heartbeat/Inbox/TaskExec with independent locks and trigger-based prompt filtering
- Tiered System Prompt — 4-tier progressive reduction (T1 Full → T4 Minimal) based on context window size
- Prompt injection defense with boundary labeling (`trusted`/`medium`/`untrusted` trust levels on tool results and Priming data)
- `debug_superuser` flag for unrestricted file/command access bypass (debug Anima support)

#### Supervisor & Organization
- Supervisor tools expansion — 6 new tools for manager Animas (`org_dashboard`, `ping_subordinate`, `read_subordinate_state`, `delegate_task`, `task_tracker`, `restart_subordinate`)
- Per-anima denied command enforcement from `permissions.md`

#### Memory & Knowledge
- Tool result consolidation — persist tool results to long-term memory via daily consolidation
- Procedures/knowledge separated injection into system prompt (distinct budget allocation)
- `write_memory_file` enables `common_knowledge/` writes with improved hints
- `read_file` hardening — dynamic line limits, line numbers, code block formatting, safety notes, partial reads

#### Communication & UI
- Messaging data model unification — DM/Message event names consolidated, `dm_logs` deprecated (activity_log is primary)
- Board reverse pagination — newest messages first with infinite scroll
- Board DM list UI improvements — mini avatars, sorting, layout optimization
- Activity timeline — per-anima selection fix and trigger-based grouping
- Chatwork `files`/`download` subcommands

#### Configuration
- `status.json` as Single Source of Truth for Anima model configuration (2-layer resolution: status.json → anima_defaults)
- DM log rotation registered as daily system cron in LifecycleManager
- Orphan Anima archival before auto-deletion

### Fixed

- Context window exceeded: automatic tier downgrade with hard truncation fallback
- `max_tokens` default raised from 4096 → 8192
- LiteLLM streaming empty response diagnostics and ContextVar reset safety
- MCP integer type validation auto-relaxation and rate limit error messages
- Setup page password configuration failure
- Activity timeline grouping — per-anima tracking eliminates cross-anima orphans
- Command meta-character blanket rejection relaxed to blocklist approach
- Final iteration tool exclusion to force final text answer
- DM display issues — dedup, garbage pair filter, arrow notation
- Board offset calculation for 3+ page channels
- Board infinite scroll completion
- Skill descriptions included in system prompt `memory_guide` section
- Heartbeat tool instruction and org context directory scan
- `check_permissions` external tool enumeration bug and `task_tracker` private method usage
- `permissions.md` section header inconsistency and DM question intent
- 3-path execution review fixes — `state_file_lock`, inbox status, wake signal
- 24 failing tests updated to match current source code

### Changed

- `injection.md` model info abolished — `status.json` is now the sole model config source
- `identity.md`/`injection.md` placed immediately after Group 1 to guarantee personality resolution before any context
- Remove system prompt duplicates and unify legacy terminology

## [0.3.1] - 2026-02-25

### Changed

- Update all default models to current generation: Opus 4 → Opus 4.6 ($5/$25, 67% cheaper), Sonnet 4 → Sonnet 4.6 (same price, 1M context), Haiku 4.5 → GPT-4.1-mini / Gemini 2.5 Flash
- Role templates updated: engineer/manager use claude-opus-4-6, writer/general/researcher use claude-sonnet-4-6
- Context window map: add 1M entries for Opus 4.6, Sonnet 4.6, GPT-4.1 family
- Setup wizard: update provider model lists to current generation
- Centralize default model name into `DEFAULT_ANIMA_MODEL` constant — future model updates only need 1 line change + role templates

### Fixed

- Orphan directory prevention: 3-layer defense (pre-creation validation, automatic cleanup, config consistency check)
- `read_memory_file` returns directory listing on File not found instead of bare error
- ToolCallRecord dataclass JSON serialization error
- ARG_MAX exceeded: oversized system prompts passed via temp file

### Added

- Azure OpenAI `api_version` / Vertex AI credential passthrough to LiteLLM
- `CHANGELOG.md` with auto-generation script (`scripts/generate_changelog.py`)
- SSE/IPC layer separation — producer task decoupled architecture
- Unified outbound rate limiting for `send_message` / `post_channel`

### Performance

- Reduce system prompt bloat: remove s_builtin section, add knowledge budget cap



## [0.3.0] - 2026-02-25

First official release. AnimaWorks is a framework that treats AI agents not as
tools but as autonomous individuals ("Anima"), each with their own identity,
memory, and decision-making criteria.

### Added

#### Core Framework
- Anima lifecycle management — create, delete, disable, enable with CLI and API
- Process Supervisor — each Anima runs as an isolated child process with Unix Domain Socket IPC
- Hierarchical organization — `supervisor` field defines reporting structure, messaging-based communication
- Role templates — 6 preset roles (engineer, manager, writer, researcher, ops, general) with model/parameter defaults
- 3-layer config resolution — per-anima override > role template > global defaults
- Unified credential management with config.json cascade

#### Execution Engine
- Mode S (SDK) — Claude Agent SDK with Claude Code subprocess, streaming, PreCompact hooks
- Mode A (Autonomous) — LiteLLM + tool_use loop for GPT-4o, Gemini Pro, Ollama models with tool support
- Mode B (Basic) — framework-mediated I/O for lightweight/tool-less models
- Automatic mode resolution via wildcard pattern matching on model names
- Session chaining — automatic context overflow detection and new session creation
- Streaming support across all execution modes with SSE relay
- ARG_MAX protection — oversized system prompts passed via temp file
- Context tracking with message_start event parsing (S mode)

#### Memory System
- RAG engine — ChromaDB + intfloat/multilingual-e5-small (384-dim) with incremental indexing
- Knowledge graph — NetworkX-based spreading activation with Personalized PageRank
- Priming layer — 5-channel parallel automatic recall injected into system prompt
  - A: Sender profile, B: Recent activity, C: Related knowledge, D: Skill match, E: Pending tasks
- Dynamic budget allocation by message type (greeting/question/request/heartbeat)
- Consolidation — daily episode→knowledge synthesis (NREM sleep analog) + weekly merge/compression
- Active forgetting — 3-stage synaptic homeostasis (downscaling → reorganization → complete forgetting)
- Unified activity log — all interactions recorded as JSONL timeline per Anima
- Streaming journal — crash-resistant Write-Ahead Log for streaming output recovery
- Conversation memory with automatic compression (display 20 / trigger 50 / retain 20)
- Shared user memory — cross-Anima user profiles in shared/users/
- Atomic file writes with fsync and two-stage recovery

#### Communication
- Internal messaging via Messenger (send_message) with async delivery
- Board — shared channels (append-only JSONL) with mentions and DM history
- External messaging — Slack (Socket Mode + Webhook) and Chatwork integration
- Unified outbound routing — auto-detect internal Anima vs external platform
- Human notification — call_human with multi-channel support (Slack, Chatwork, LINE, Telegram, ntfy)
- Outbound rate limiting — 3-layer cascade prevention for message storms
- DM gratitude loop and board pollution suppression

#### Autonomy
- Heartbeat — periodic self-check with customizable checklist
- Cron — scheduled tasks with YAML definition, async parallel execution
- Background task submission and execution
- Task queue with persistence, deadline enforcement, and delegation prompt injection
- Heartbeat/conversation parallelization with lock separation

#### System Prompt
- 6-group structured prompt builder (environment → identity → situation → memory → organization → meta)
- Distilled knowledge injection (knowledge/ + procedures/, 10% of context budget)
- Dynamic tool guide generation per execution mode
- Behavior rules with MUST constraints

#### Web UI
- FastAPI server with WebSocket real-time updates
- SPA dashboard with activity timeline, status panels, and memory viewer
- 3D office workspace with pathfinding, idle behaviors, and desk layout
- Visual novel-style conversation screen with expression variants
- Board UI — channel/DM browsing and posting
- Setup wizard (GUI) with language selection (17 languages)
- Multi-user authentication (password + localhost trust)
- Mobile responsive design (iPad Safari viewport fix, touch support)
- SSE streaming for chat and heartbeat responses
- Infinite scroll pagination for conversation history
- Multimodal image input in chat

#### Tools
- External tool framework — auto-discovery, creation, hot-reload, unified dispatch
- Web search and X (Twitter) search
- Slack and Chatwork messaging
- Gmail integration
- GitHub and AWS integration
- Image generation — NovelAI API, fal.ai (Flux), Meshy (3D models)
- Transcription (Whisper) and local LLM (Ollama)
- Supervisor model control — parent Anima can change child's model and restart

#### Asset System
- Character image generation with bust-up expression variants
- Vibe Transfer for style consistency across team
- 3D model generation via Meshy API with FBX→glTF conversion
- Asset reconciler — periodic batch generation for missing assets
- 3-layer cache (API / delivery / compression) for 3D models

#### CLI
- `animaworks init` — workspace initialization with safe merge and full reset modes
- `animaworks server` — start/stop/restart with PID file management
- `animaworks chat` / `animaworks send` — interactive and one-shot messaging
- `animaworks create-anima` / `animaworks delete-anima` — Anima management from character sheets
- `animaworks index` — RAG index management
- `animaworks config` — configuration management with export-sections
- `animaworks board` — Board channel management
- One-liner setup script for quick installation

#### DevOps
- Publish script — private→public repo sync with rsync, PII scan (Claude Code), and changelog
- Apache-2.0 license with SPDX headers
- Memory evaluation framework — 3 ablation experiments with synthetic datasets
- Comprehensive test suite (unit + E2E)

### Changed
- Migrated from distributed architecture (Gateway + Worker + Redis) to monolithic FastAPI server
- Renamed execution modes from A1/A2/B to S/A/B
- Switched license from AGPL-3.0 to Apache-2.0
- Moved model mode patterns from config.json to models.json
- Tool permissions changed from whitelist to default-allow (blacklist) model

[Unreleased]: https://github.com/xuiltul/animaworks/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/xuiltul/animaworks/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/xuiltul/animaworks/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/xuiltul/animaworks/compare/v0.3.0...v0.3.1
