# Changelog

All notable changes to AnimaWorks will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.10] - 2026-03-04

### Added
- DK removal Phase 1+2 — full Channel C search + budget expansion
- add Priming Channel F (episodes) and search_memory episodes support
- add check_background_task / list_background_tasks MCP tools
- add audit_subordinate supervisor tool for monitoring subordinate activity
- expose plan_tasks via MCP and update heartbeat prompts to recommend it
- add model info CLI commands, anima info, and comprehensive docs
- add think tag strip filter for Qwen3.5 content-embedded reasoning
- improve task awareness — origin_chain human bonus + heartbeat add_task guidance
- implement live tool activity streaming — tool_detail SSE + subordinate activity broadcast
- improve heartbeat effectiveness — raise tool limit, filter activity noise, enforce STALE tracking
- implement live tool activity streaming with real-time UI updates
- implement Board channel ACL (access control for shared channels)
- add Qwen 3.5 model support (Mode A + 64K context window)
- implement credential vault encryption with PyNaCl SealedBox
- per-Anima Chatwork write token + fix streaming-controller container ref
- idle conversation pre-compression, Claude auto-update, server PID detection fix
- update system prompts for S-mode Task tool auto-routing
- resolve avatar URLs for chat history from_person and workspace
- add anima avatar display to chat bubbles
- persist per-pane anima/thread selection across reloads
- demo README & Quick Start — English/Japanese guides with Docker demo link
- demo fictional runtime data — 3-day activity logs and state files
- demo asset infrastructure — directory structure, generation and optimization scripts
- add chatwork_delete tool for self-message deletion
- NovaCraft world-building — 4 presets × 3 characters with full personality
- demo Docker infrastructure with preset selection
- Task tool delegation + SDK subagent for S-mode
- add frontend image resize & cache module for avatar thumbnails
- add token usage tracking and cost estimation

### Fixed
- auto-convert anime prompts to realistic in asset generation
- adjust episode budget to 500 tokens per Issue spec
- skip memory_eval e2e tests when experiments/ unavailable in CI
- address review findings — ElevenLabs TTSSynthesisError, exception separation, tests
- replace silent except-pass with debug logging in audit_subordinate
- patch path for MeshyClient credential mock in asset optimization tests
- per-thread interrupt event via ContextVar for parallel streams
- await missing on neurogenesis_reorganize + catch-up missed consolidation jobs
- transcribe tool broken via submit — composite name + subcommand mismatch
- address review findings for audit_subordinate
- update dispatch dict test to include audit_subordinate tool
- token usage input/output_tokens always 0 — use dict.get() instead of getattr()
- add missing tool category flags to Mode B AssistedExecutor
- RC-1 無音嚥下防止 + RC-6 interrupt時response_done保証
- TTS P0 — stop swallowing synthesis errors (RC-1) and guarantee response_done on interrupt (RC-6)
- filter subordinate tool activity by org hierarchy on frontend
- avoid buffering non-think content in StreamingThinkFilter
- filter subordinate activity to prevent global tool_use event leakage
- priming Channel C keyword extraction and search accuracy
- auto-inject frontmatter for knowledge/procedures, repair Priming pipeline
- prevent subprocess leak by re-raising GeneratorExit in async generators
- unify started_at to milliseconds in stream_registry.py
- prevent keepOnlyStreaming from clearing completed chat messages
- address Critical/Important review findings for Board ACL
- replace fragile split("---", 2) frontmatter parsing with line-based parser
- voice TTS playback failure on reconnect — AudioContext state management and server robustness
- update credential resolver test to expect vault.json in error message
- guard app.js init() against double execution on Settings navigation
- prevent mic button presence from shifting send button layout
- prevent pane auto-focus on stream end, add flash notification
- モバイル情報パネルが右にズレて崩れる問題を修正
- address review findings for token usage tracking
- voice chat UI improvements — layout, VAD loading, TTS sanitization, duplicate response prevention

### Changed
- simplify streaming-controller and session-manager
- remove obsolete UI test scripts and re-enable realistic animations
- separate streaming indicator animation for active vs inactive tabs

### Performance
- paginate activity API — cap per-Anima loading instead of O(N) full scan
- zone-based partial DOM updates for streaming chat bubbles

### Other
- Revert "refactor: simplify streaming-controller and session-manager"


## [0.4.9] - 2026-03-02

### Added
- Chat マルチペイン分割 — VS Code 風に複数チャットインスタンスを横並び表示（分割/閉じるボタン付き）
- gmail_draft に添付ファイル対応とスレッド返信自動解決を追加
- アイコンのみ表示時にアニマ名ツールチップをポップアップ表示（デスクトップ hover / モバイル tap）

### Fixed
- アイコンのみ表示時（サイドバー折りたたみ / モバイル）にアニマタブの閉じるボタンが非表示で操作不能だった問題を修正
- モバイルでドロップダウンがペインの overflow:hidden にクリップされる問題を修正
- Priming レイヤー5件のバグ修正（CUDA OOM, Channel B スコアリング, 表示肥大, Channel D フォールバック, outbound 切り詰め）
- SDK セッション再開時に即座にコンパクションが発生する問題を10分タイムアウトで防止
- gmail_draft attachments パラメータの文字列→リスト変換を追加
- 37件のテスト関数を現行実装に合わせて更新

## [0.4.8] - 2026-03-02

### Added
- アセットRemakeで画像スタイル選択UI追加、configデフォルト参照に変更
- style-aware prompt separation for realistic image generation
- asset remake UX improvements — scratch generation, preview history, expression grid
- unified anime/realistic avatar display and remake support
- split display mode from color theme, add Settings page
- smart scroll with floating scroll-to-bottom button
- image generation pipeline enhancements and asset reconciler updates
- mobile chat UX improvements, dashboard, and setup wizard enhancements
- theme system with 10 color presets and dropdown selector
- DAG scheduler for parallel task execution
- restore transcript writing and add shared conversation log
- thread tab styling improvements — subtle active state, streaming pulse, completion indicator

### Fixed
- add locale-based ethnicity to realistic prompt conversion
- replace hardcoded anima names with generic placeholders in templates
- sidebar active menu text invisible on dark themes
- thinking inline preview text invisible on dark themes
- dark theme input text color - use token instead of hardcoded black
- streaming stop button and per-thread concurrency
- update tests for asset reconciler default, streaming indicator path, and deferred trigger
- update streaming indicator test for refactored chat JS
- prevent Codex SDK LimitOverrunError on thread resume with large prompts
- use JST date in tool_result_log and channels tests
- handle additional Codex SDK event types (text.delta, response.completed)
- use JST date in heartbeat history test for CI timezone compat
- ensure chat queue auto-drains after streaming completes
- use JST date for activity_log paths in more test files
- preserve full message content in heartbeat dedup consolidation
- use JST date in activity spec test to match now_iso() timezone
- resolve CI test failures and add activity group_type filter

### Changed
- align private repo tracking with public, merge PR #1
- rename dashboard anima list label to Org Chart


## [0.4.7] - 2026-03-01

### Added
- publish.sh --release で自動バージョンインクリメント
- cron中のinbox抑制 + per-anima flockによる多重起動防止
- add origin metadata to RAG chunks and trust-separated priming output (provenance phase 4)
- propagate origin_chain in Anima-to-Anima messaging (provenance phase 3)
- add origin tracking at external data entry points (provenance phase 2)

### Fixed
- session chaining時にSDK session IDをクリアしてfreshセッションを開始
- compaction空白地帯の解消 — context window是正・閾値スケール修正・Mode Sチェイニング有効化
- Mode S streaming compaction failure — 2 root causes
- prevent repeated content in session-chained responses
- make status.json the SSoT for supervisor/speciality fields
- detect stale cron/heartbeat schedules via file mtime reconciliation
- detect stale cron/heartbeat schedules via file mtime reconciliation
- load mode_s_auth from status.json into resolved config


## [0.4.6] - 2026-02-28

### Fixed
- Mode C (Codex SDK) session-chaining: add CodexResultMessage adapter for num_turns/session_id interface
- Mode C prompt selection unified with Mode S via `_is_mcp_mode()` helper (communication_rules, messaging, hiring_rules, tool guides)

## [0.4.5] - 2026-02-28

### Added
- Mode S (Agent SDK) multimodal image input support
- text artifact popup viewer for code blocks (file: cards)
- ConversationMemory provider-specific credential injection (Bedrock/Azure/Vertex)
- toggle for chat right-side status pane
- Mode C (Codex SDK) interrupt_event support

### Fixed
- create-anima name parsing bug: Japanese headings no longer captured as anima name
- create-anima documentation updated from deprecated `create-anima` to `anima create`
- chat message duplication and cross-anima display contamination
- concurrent per-anima streaming in chat UI
- chat stream error recovery and reconnection handling
- recovered chat content preservation on reload

### Changed
- **major backend refactoring**: split 6 God-class modules into focused Mixin files
  - `handler.py` (3298 → 8 files)
  - `agent_sdk.py` (1716 → 5 files)
  - `image_gen.py` (2434 → 6 files)
  - `activity.py` (1471 → 6 files)
  - `manager.py` (1419 → 4 files)
  - `litellm_loop.py` (1392 → 4 files)
- **frontend refactoring**: split large JS modules
  - `chat.js` (2490 → 12 controllers)
  - `character.js` (1396 → 5 modules)
  - `office3d.js` (1208 → 5 modules)
  - `chat-controller.js` (1009 → 5 modules)
  - `app.js` (716 → 4 modules)
  - `timeline.js` (576 → 4 modules)
- extract shared chat logic to `shared/chat/` for Dashboard/Workspace reuse

## [0.4.4] - 2026-02-27

### Added
- expose permitted external tools as native MCP tools in Mode S

### Fixed
- harden media proxy with extracted secure module
- harden parsing paths and align e2e expectations
- switch chat controls to SVG and add desktop sidebar toggle
- improve workspace chat tab stream indicators
- auto-refresh chat view every 5 seconds
- enrich cron job parsing with next and last run data
- streamline memory tabs and support scheduler job fallbacks
- load scheduler jobs from split API fields
- tighten chat input box height on chat and workspace
- reduce workspace chat action button sizes
- harden log viewer path handling and polish UI behaviors
- resolve markdown image paths and add attachment fallback
- scale down images in chat bubbles rendered via markdown
- voice TTS not switching when changing anima tab
- resume active stream on reload regardless of process status
- streaming stop button now scoped to current anima+thread


## [0.4.3] - 2026-02-27

### Added
- surface assistant image artifacts in chat and history
- add persistent dashboard chat tabs with unread stars
- multi-thread chat backend + frontend
- add frontend i18n with shared i18n.js module and locale JSON files (L6)
- add core/i18n.py and externalize hardcoded Japanese strings in Python (L5)
- add English translations for knowledge, skills, and roles templates (L4)
- message queue with management UI and queue-anytime support
- pending message queue, icon-only buttons, dynamic button state
- add pending message queue with interrupt-and-send for chat UI
- implement call_human reply routing from Slack (Issue #1)
- intercept SDK Task tool → pending LLM task for background execution
- add voice output sanitization and voice-mode suffix for TTS
- migrate skill format from flat files to directory structure
- Mode B補助輪強化 — ツール仕様テキスト改善 + インテント検出リプロンプト
- add process control buttons and LLM session interrupt to WebUI
- add web_fetch internal tool for Anima URL content retrieval

### Fixed
- improve add-conversation anima menu UX
- refine chat tab UI behavior and responsive layout
- preserve streaming partial responses during heartbeat relay
- polish chat input UX and refresh static asset versions
- make VAD auto mode fully hands-free
- close remaining assistant-image review gaps
- isolate thread histories in chat view and tighten voice reply length
- improve workspace voice input UX and metadata handling
- restore VAD auto mode and suppress silent STT hallucination
- prevent duplicated voice stream updates on UI reinit
- improve chat UI behaviors and thinking stream visibility
- show avatar icons in anima chat tabs
- restore old chat threads and improve tab UX
- improve workspace streaming UX and sync docs navigation
- improve workspace stream resume and thinking preview behavior
- avoid task-intercept blocked misclassification
- resume dashboard chat stream after page return
- clear chat display immediately when switching threads
- clear chat input immediately after send submission
- make channel ID regex case-insensitive for consistency
- thread_id validation and conversation view filtering
- close prior activity group on next trigger
- update review docs and chat sidebar responsiveness
- persist chat drafts until successful send
- resolve review revision findings for i18n L4-L6
- update tests for i18n string changes and adapt evaluation docs
- remove bottom whitespace in chat sidebar memory section
- avoid no-response when stream events are missing
- make send/queue buttons perfect circles with explicit width+height
- prevent mobile auto-zoom on chat input focus
- catch StopAsyncIteration on Agent SDK session resume
- add voice chat balloon callbacks to standalone chat page
- prevent duplicate process spawn via _starting/_restarting guards
- add per-anima mode_s_auth to prevent shared API key rate limiting
- address review findings — missing interrupt_event paths, CSS :active parity

### Changed
- address round-2 review findings for reply routing
- address review findings for call_human reply routing
- Cursor-style input — embed buttons inside textarea container
- use English persona-aware message for Task intercept deny reason

### Performance
- parallelize anima startup and make web server start first


## [0.4.2] - 2026-02-26

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

[Unreleased]: https://github.com/xuiltul/animaworks/compare/v0.4.10...HEAD
[0.4.10]: https://github.com/xuiltul/animaworks/compare/v0.4.9...v0.4.10[0.4.8]: https://github.com/xuiltul/animaworks/compare/v0.4.7...v0.4.8
[0.4.3]: https://github.com/xuiltul/animaworks/compare/v0.4.2...v0.4.3
[0.4.0]: https://github.com/xuiltul/animaworks/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/xuiltul/animaworks/compare/v0.3.0...v0.3.1
