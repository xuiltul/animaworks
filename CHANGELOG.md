# Changelog

All notable changes to AnimaWorks will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/xuiltul/animaworks/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/xuiltul/animaworks/releases/tag/v0.3.0
