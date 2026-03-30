# Changelog

All notable changes to AnimaWorks will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-03-30

### Added

#### Memory & Search
- `search_memory` に `activity_log` スコープを追加 — 直近3日間の行動ログをBM25キーワード検索可能に。`scope="all"` 使用時はベクトル検索結果とRRF（Reciprocal Rank Fusion）で統合
- `search_memory` に `skills` スコープを追加 — スキル・共通スキルをベクトル検索可能に
- 人間フィードバック・好み情報の `write_memory_file` 自動保存を強制化

#### Completion & Quality
- `completion_gate` ツールを全実行モード（S/A/B/C/D/G）に追加 — 最終回答前の完了チェックリスト検証
- Mode S: Agent SDK Stop hook による直接チェックリスト注入
- Mode A: マーカーファイル方式で未呼び出し時に1回リトライ強制

#### Consolidation
- 2-phase multipass consolidation — tool_result全文を活用した高精度エピソード抽出 + エラートレース分析
- Phase A: activity_logを時間チャンク分割 → LLM one-shotでエピソード抽出 → マージ・重複除去
- Phase B: エラートレース分析（error + failed tool_result 収集・要約）→ 知識抽出・procedure自動生成

#### Organization & Delegation
- 委譲タスクのPriming表示 + 部下からの自動同期（`sync_delegated`）
- スーパーバイザーの `state/plans/` 読み取りアクセスを許可
- cross-Anima 書き込み境界ガイダンスをプロンプト・common_knowledgeに追加

#### External Integrations
- Discord連携ツール追加（REST API v10）— メッセージ送受信・チャネル操作
- Slackユーザーメンション解決をインジェスト時に実行 + システムアノテーション付与 + intent="question" 即時トリガー最適化
- `call_human` スレッド返信時の intent='question' 自動設定

#### Assets & UI
- Animaアセットにアイコン追加・個別再生成・fullbodyアップロード対応 (#141)
- セッション境界での `current_state.md` 自動アーカイブ (#143)

#### Templates & Documentation
- team-designテンプレート追加・machine関連ドキュメントを `operations/machine/` に統合
- machine-tool-usageをロールベースワークフローガイドに再構成
- skill-creatorを Use-when ガイド + `lint_skill.py` で刷新

### Changed
- `skill` ツールを廃止し `read_memory_file` に統合 — スキルカタログはシステムプロンプトに直接表示、`read_memory_file(path="common_skills/.../SKILL.md")` で全文取得
- Priming Channel D（スキルマッチ）を削除 — スキル情報はシステムプロンプト内カタログに移行
- `common_skills/` 全スキル説明を Use-when パターンに移行（ja/en/ko）
- `read_memory_file` に `common_skills/` パス解決を追加
- `delegate_task` スキーマに自己完結型の指示ルールをインライン化
- Claude組込みの CronCreate/Delete/List ツールを拒否リストに追加
- アセットルート強化・パイプラインドキュメント整合・コーディング規約修正

### Fixed
- POSIX cron `day_of_week` 番号（0=日曜）とAPScheduler（0=月曜）の互換性問題を修正
- `cron.md` パーサーのフォーマット許容性向上（よくある書式ミスに耐性）
- ローカル `file://` 画像パスおよび絶対パスの解決を修正
- ChromaDB `InternalError` を `query()` でキャッチしHTTP 500を防止
- RAGメタデータの非整数 `version`/`count` フィールドに対するガード追加
- テーブル要素のフォスターペアレンティングによるチャットレイアウト崩れを修正
- `data-*` 属性のエスケープを属性安全方式に変更しDOM破損を防止
- `reply_routing` のサイレント例外をデバッグログに置換
- 古い `current_task.md` が `current_state.md` 存在時に自動削除されない問題を修正
- 韓国語プロンプトのフォーマット修正

### Security
- `litellm>=1.82.6` をピン留め — サプライチェーン攻撃バージョンの排除

### Migration
- `v062_skill_removal_and_activity_log`: テンプレート全同期（common_knowledge, prompts, reference, common_skills）+ DB tool_descriptions/guides再同期 + 旧 `skill` ツール記述削除

## [0.6.1] - 2026-03-21

### Fixed
- fix ruff lint violations: bare f-string (F541), missing `raise from` (B904), and formatting
- add missing `_is_port_listening` and `_get_daemon_log_path` mocks to restart CLI tests
- load `GlobalPermissionsCache` in E2E blocklist test after blocklist refactoring

## [0.6.0] - 2026-03-21

### Added

#### New Execution Engines
- Mode D (Cursor Agent) execution engine — Cursor Agent CLI subprocess with MCP integration agent loop
- Mode D session continuity via `cursor-agent --resume` for cross-turn context preservation
- Mode D system prompt optimization with A+C hybrid turn rotation
- Mode G (Gemini CLI) execution engine — Gemini CLI subprocess with stream-JSON parse
- 6 execution modes (S/C/D/G/A/B) supported across docs, setup wizard, and settings UI (#139)

#### Meeting & Communication
- Meeting Room mode for multi-Anima conferences with facilitator-driven discussion
- block communication tools during meetings and add context summarization (#130)
- message quality protocol for Anima-to-Anima communication — structured format, noise reduction (#137)
- stream activity report generation via SSE

#### Security & Permissions
- `permissions.global.json` — unified global command security config with startup verification
- Mode S `bypassPermissions` — full built-in tool access without explicit allowed_tools list

#### Planning & Monitoring
- `todo_write` session-scoped planning tool for Mode A agents
- cron health check with `cron.md` parse validation and periodic monitoring
- heartbeat quality improvement — observe evidence requirements, plan-outcome tracking, OK gate

#### Platform & i18n
- Korean locale (ko) — full i18n support including prompt templates, common_knowledge, common_skills, reference docs, and web UI strings (#124)
- Windows-native supervisor support (#136)
- Codex login support in setup wizard and settings

#### Other
- migration step for task_delegation_rules → common_knowledge move
- token usage pricing correction and cache token tracking across all execution paths

### Fixed
- prevent restart helper from being killed by process scanner during shutdown
- prevent activity log bloat from unbounded `tool_result` content
- prevent `delegate_task` dual-trigger duplicate execution (#129)
- clarify `submit_tasks` vs `delegate_task` tool descriptions to prevent misuse
- enable real-time streaming for meeting mode chat bubbles
- eliminate false positives in global permissions deny patterns
- copy auth credentials and settings to Gemini CLI per-Anima workspace
- map `message:*` trigger to chat session type for Mode D resume
- add None guard to SDK stream cache token accumulation

### Changed
- move task_delegation_rules to common_knowledge and unify access
- loopback host validation and i18n error messages


## [0.5.5] - 2026-03-18

### Added
- unified `animaworks migrate` command — 22 migration steps across 5 categories (structural, per-anima, template sync, DB sync, version tracking) with `--dry-run`, `--list`, `--force`, `--resync-db` options
- auto-migration on server startup via `ensure_runtime_dir()` — no manual action needed for most upgrades
- `migration_state.json` for idempotent version tracking of applied migrations


## [0.5.4] - 2026-03-18

### Added
- overhaul MessageDeduplicator with overflow_inbox individual file approach
- fully separate ChromaDB from runner processes — server-only ownership
- enforce MUST rules for memory verification and file reading before acting

### Fixed
- apply ruff format to dedup.py and handler_skills.py
- replace silent except with logger.warning in http_store.close()
- harden RAG process separation — bool returns, ABC bug, CLI safety
- eliminate remaining ChromaDB ABC bypass and direct instantiation
- use dedicated COMPACT_TIMEOUT_SEC (300s) for SDK idle compaction
- save shortterm in idle compaction (Mode A/B/S fallback) and preserve in Mode A blocking threshold path
- add extra_mcp_servers mock in test_loads_from_config_json


## [0.5.3] - 2026-03-17

### Added
- improve priming query quality for chat/inbox/heartbeat
- compress role specialty prompts and add report-formats common knowledge
- improve priming quality — Channel B/C/E noise removal, knowledge injection, task priority
- cap frequency_boost and implement per-anima access counting for shared collections
- unify resolve_context_window with models.json as SSoT (#115)
- vector-primary search_memory with rich results and episode chunking fix
- align read_file budget with Claude Code Read tool constraints
- Mode A context compaction — align tool output limits with Mode S + LLM one-shot compaction
- replace permissions.md templates with JSON, add CLI permissions command
- replace MD parsers with JSON loader, unify execution mode security
- add PermissionsConfig model, loader, and MD→JSON migration
- enable progressive streaming for Codex mode (C) over IPC
- add --background CLI flag to machine tool & improve SKILL docs
- restrict delegate_task to direct subordinates only
- unify descendant permissions with direct subordinate (#108)
- add replay time range selector UI (1h/3h/6h/12h/24h)
- reinforce memory consolidation merge pipeline
- add agent-browser skill and browser automation guide
- prevent injection.md bloat — [IMPORTANT] always-prime, size governance, workspace separation
- read-before-write guard and knowledge dedup hints for write_memory_file
- add background parameter to Bash tool for async command execution with streaming output
- centralize embedding inference into server process
- SWE-bench multi-agent evaluation infrastructure
- add replay mode to Dashboard (Business 2D) org-dashboard
- workspace registry and alias resolution for Anima working directories
- add real-time context window usage ring indicator to chat UI
- graceful interrupt for Mode S session preservation
- inject machine tool MUST directive into TaskExec prompt
- enhance Dashboard (Business 2D) real-time visualization
- add fuzzy CJK-Latin spacing tolerance to Edit tool
- Phase 4 — CLI subcommands for supervisor/vault/internal tools + test fixes
- unify tool schema — 18-tool Claude Code-compatible architecture
- add machine-tool common skill and fix hint reference
- show machine tool hint in system prompt for non-heartbeat triggers
- machine tool engine priority with __list__ discovery
- unified CLI fallback routing — both entry points resolve all commands
- dynamically hide machine tool engines based on CLI availability
- add machine tool — external agent CLI as stateless power tools for Animas
- Support prefers-color-scheme for default theme (fixes #56) (#87)
- add housekeeping rotation for task_results/ and pending/failed/
- add --since HH:MM filter to audit_subordinate tool and CLI
- add conceptual integration of [IMPORTANT] memories (amygdala→semantic consolidation)
- consolidation retry + PostToolUse knowledge frontmatter hook (#73)
- task queue 2-layer sync — plan_tasks Layer 2 registration + PendingExec completion/failure sync
- add task-architecture.md to common_knowledge anatomy
- add AST-based hardcoded Japanese string detection test
- add 2-stage heartbeat timeout (soft reminder + hard cutoff)
- prevent current_task.md bloat with HB cleanup instruction and auto-pruning
- task_queue compact with archival + list_tasks output optimization
- deprecate send_message(intent="delegation"), enforce delegate_task usage
- delegate_task writes to subordinate state/pending for immediate execution
- enrich call_human thread replies with notification context
- add advanced agent benchmark (Sonnet 4.6 vs Qwen3.5-35B)
- add importance boost to RAG retriever (amygdala model)
- add 4-model agent benchmark results and Qwen3.5-35B recommendation
- add AnimaWorks agent benchmark runner for hina evaluation
- add Team Presets API with industry x purpose templates
- add AI Brainstorm feature with multi-character perspective generation
- add Team Builder UI, External Tasks widget, fd_limits utility and watchdog graceful import
- add common_knowledge access paths reference, update index files and tests
- add reference/ shared directory infrastructure
- move 8 detailed reference docs from common_knowledge to reference/
- add streaming repetition detection as safety net
- add penalty parameter support in models.json and _build_llm_kwargs
- add Bedrock Kimi K2.5 thinking support via reasoning_config
- pass enable_thinking via extra_body for openai/* models (vLLM)
- unify thinking display across all execution modes
- support thinking mode for Qwen models on AWS Bedrock
- collapsible background sessions in chat main tab

### Fixed
- tighten session summary task extraction criteria to reduce false positives
- use dict access in compressed summary keyword search test
- use dict keys instead of tuple indices in conversation summary tests
- use dict key instead of tuple index in procedures search test
- align e2e activity log test with chat noise filtering
- prevent race condition in restart helper when PID file is missing
- update test assertions for compressed communication_rules template
- address review feedback (iteration 1)
- prevent thread context from truncating user messages in inbox
- remove hiring_context from init.py, cli.py, compare_prompt_db.py
- replace silent except-pass with logged debug message in rag_search
- e2e test_hybrid_search_common_knowledge failing on CI
- e2e test_search_memory_text_scope failing due to missing ChromaDB collection
- remove dimension param from create_collection to prevent GPU model loading in runners
- resolve ruff lint and format issues
- update test mock paths after slack/chatwork submodule refactoring (9033eab8)
- update test mock path for _call_compression_llm after refactoring
- restore direct retriever injection in PrimingEngine._get_or_create_retriever
- resolve ruff lint errors in core/ — I001 import sort + F401 unused import
- replace parse_permitted_tools() with load_permissions() + get_permitted_tools()
- inter-anima boundary check and test updates for permissions.json
- eliminate SSE streaming race condition in _sse_tail event delivery
- update E2E test to check task_queue.jsonl instead of current_state.md
- ruff format core/memory/task_queue.py
- address review feedback — remove remaining pending.md refs, batch task lookups
- update budget constant test for Channel E (300 → 500)
- Issue #114 — update templates and handler for current_state/pending separation
- address review feedback (iteration 1)
- replay now fetches all events for selected time range
- set reasoning_content=None in mock to prevent MagicMock thinking injection
- replace silent except Exception: pass with except OSError in pipe cleanup
- update test assertions for openai/ thinking default and i18n baseline
- initialize _read_paths in _make_handler for test_path_traversal
- initialize _read_paths in _FakeWriteHandler and pre-populate in overwrite test
- add archive_memory_file to MCP tools and strengthen consolidation prompts
- block delegation tools during memory consolidation
- exclude .archive/ and _archived/ from merge candidates and RAG indexing
- ruff lint and format for core/ cli/ server/
- replay time range now covers full requested hours
- use raw vector similarity for merge candidates, improve archive exclusion
- improve replay feature — event mapping, 24h range, 200x speed
- address review feedback (iteration 2)
- address review feedback (iteration 1)
- default enable_thinking=True for openai/ models and detect untagged thinking
- use polling for heartbeat intervals that don't divide evenly into 60
- broaden _get_locale() exception handling to include ConfigError
- remove unused imports (F401) in tests and scripts
- remove extraneous f-prefix from f-strings without placeholders (F541)
- remove unused imports and variables (F401/F841)
- make sync tests async to match module-level pytestmark
- implement strip_untagged_thinking for vLLM tag-free thinking detection
- explicitly disable thinking for openai/ models when thinking=None
- address review feedback — correct env var and cleanup
- isolate SWE runtime from production ~/.animaworks/
- mock load_auth and slack_socket_manager in server unit tests
- correct invalid loop variable in schemas.py (F821)
- export const チェックを test_websocket_imports_match_org_exports に追加
- use visible tool in dashboard viz test
- combine if-elif branches in activity.py (SIM114)
- expand onDone search window from 1000 to 1200 chars
- add sync comments for VISIBLE_TOOL_NAMES across FE/BE
- address review feedback — buffer all WS events during replay
- filter tool_use from Dashboard card streams
- multi-layer defense against <think> tag leakage into chat responses
- repair org-dashboard crash + feat: add grid snap for card placement
- isolate streaming updates per thread to prevent cross-thread bleed
- restore context ring on Anima tab switch
- wrap entire interrupt+receive in timeout guard
- normalize readable locations in permission templates
- add shared/ write permission to all role templates
- address review feedback (iteration 1)
- use horizontal whitespace only in fuzzy CJK-Latin pattern
- align E2E tests with unified 18-tool architecture
- map WebSearch 'limit' param to 'count' in dispatch layer
- address review findings — _WRITE_TOOLS, path traversal, budget mapping
- replace silent except-pass with logged exceptions in machine.py
- add injection.md/status.json to Mode S subordinate file access
- inject credentials from AnimaWorks config into machine subprocess
- apply ruff format to machine.py and update i18n hardcode baseline
- address review findings for machine tool
- prevent thinking preview scroll-jump by patching DOM in-place
- prevent thinking preview scroll-jump by patching DOM in-place
- recalculate resolved_mode in _resolve_background_config
- fall back to CPU when CUDA OOM during embedding model load
- extend t() locale allowlist to include zh and ko (#78)
- preserve </think> on early-exit path for safety-net
- remove misleading offset/limit hint from read_memory_file truncation message (#80)
- map legacy 'ts' field to timestamp in Message validator
- inboxの不正ファイルがメッセージ処理全体を停止させる問題を修正
- replace IntervalTrigger with polling-based heartbeat for interval > 60min
- correct claude_agent_sdk import in compact_session
- grace IPC errors when process is alive to prevent false SIGTERM
- IntervalTrigger の end_date によるハートビート停止バグを修正
- add 2000-line truncation to read_memory_file to prevent prompt too large
- resolve ruff lint/format errors (import sort, formatting)
- resolve ruff lint errors in _parse_since (F821, UP037)
- update remaining plan_tasks references in task_queue.py docstrings
- skip parallel tasks with failed dependencies (mirror serial check)
- make `animaworks restart` survive caller death via detached helper
- ensure [IMPORTANT] tag is preserved and discoverable across memory lifecycle
- depends_onを持つタスクの初期statusをpendingに修正
- add trailing slash to knowledge_dir_str startswith check to prevent false matches
- strip orphan </think> tags from Qwen3.5 streaming output
- address review feedback (iteration 1)
- rename activity filter "タスク" → "タスク管理" to distinguish from "タスク実行"
- add anatomy/task-architecture.md to expected file list
- add soft < hard timeout validation to HeartbeatConfig
- remove unused UTC import after datetime cleanup in handler_skills
- remove redundant local datetime import shadowing now_iso in _handle_plan_tasks
- update test_task_metrics mock for split list_tasks() calls
- update audit.py for new list_tasks() default behavior
- strip residual </think> tags when multiple think blocks emitted
- isolate TestSchedulerManagerE2E from system config
- update skill-creator test assertion and restore tags field in ja template
- use MagicMock for synchronous get_pid in restart_race tests
- address review feedback (iteration 2)
- update remaining en templates to deprecate send_message delegation intent
- update remaining ja templates to deprecate send_message delegation intent
- replace silent except-pass with logger.debug in call_human
- save notification mapping in CLI call_human (Mode S reply routing)
- update existing tests for delegation intent deprecation and SDK hook changes
- use detail length for tool_detail events in debug log
- multiple small fixes — enable_thinking=False on Bedrock + chunk counter off-by-one
- use actual event name in debug log (was hardcoded as text_delta)
- repair test_stream_exception_handling mock as proper async generator
- set data-theme attribute on body for dark themes (closes #53)
- add HEALTHCHECK to Dockerfile and docker-compose files (closes #52)
- add .dockerignore for smaller Docker builds (closes #51)
- downgrade debug-labeled stream logs from INFO to DEBUG
- repair malformed tool-call JSON in non-streaming execute() path
- update RepetitionDetector tests for n=10/threshold=10 defaults
- relax RepetitionDetector thresholds to reduce false positives
- repair malformed tool-call JSON from GLM-4.7 thinking mode
- simplify StreamingThinkFilter and add Bedrock GLM thinking support
- support vLLM reasoning parser that strips <think> opening tag
- resolve CI failures — ruff format + CSS hover/active parity
- text-format tool call IDをイテレーションごとにユニーク化
- Bedrock tool calling — keep toolConfig when history has toolUse/toolResult
- Llama 4 Maverickのテキスト形式ツールコールをパースして実行する
- update stale references to files moved to reference/
- fall back to non-streaming for models without streaming tool use
- update test expectations for moved reference files
- set litellm.modify_params=True in all executors for Bedrock compatibility
- immortalize SDK sessions — remove TTL, preserve on compaction failure
- sanitize tool_use_id for Bedrock Converse API compatibility
- address review feedback (iteration 2)
- rename unused loop variables to underscore prefix (B007, 10件)
- restrict synthetic thinking_blocks injection to Anthropic models only
- use PID-unique temp file in save_config to prevent concurrent rename race
- replace blind Exception with ValidationError in pytest.raises (B017/F841)
- rename unused loop variables to _ prefix (B007/I001)
- bind loop variable in lambda to prevent B023 closure bug
- ruff format _litellm_streaming.py
- add enable_thinking to _thinking_enabled check in streaming
- ruff format core/execution/base.py
- resolve UP032/UP015/SIM118/UP012/F401 lint warnings (5 files)
- apply same Qwen Bedrock routing to assisted.py (Mode B)
- resolve SIM110/SIM114/SIM103/B011/F841 lint warnings
- add # noqa: F401 to try/except availability-check imports
- replace invalid noqa directives and remove trailing whitespace
- remove unnecessary f-prefix and unused imports (ruff F401/F541)
- mock Agent SDK fallback in test_llm_failure_returns_none
- untrack private-only files and harden .gitignore
- prevent auto-scroll to bottom during thinking zone updates
- patch Agent SDK transport to allow graceful CLI shutdown
- pass thread_id to ConversationMemory and ShortTermMemory
- pass source param to process_message_stream in streaming handler
- update Message.source docstring to include googlechat
- add source parameter to process_message for external platform awareness
- add text_delta to mock_stream so archive_paths is called
- add text_delta to mock_stream so archive_paths is called
- move overflow:hidden to base .chat-bubble to cover all bubble types
- wrap long lines in code blocks to prevent bubble overflow
- infinite scroll not working for tool-heavy animas (hinata)
- prevent chat bubble content overflow causing page-level scroll
- task exec markdown rendering, compact header, italic muted body
- session splitting on trigger change, subtle bg-session styling
- don't archive inbox messages when LLM returns empty response
- address review feedback (iteration 1)
- bootstrapループ防止（3つの構造的バグ修正）
- prevent bootstrap infinite loop (3 structural bugs)
- update session_tool_uses tests for deferred-chaining design
- remove stale inject_shortterm patches from litellm_loop test
- SDK session complete isolation — chat-only resume, fresh for background

### Changed
- compress communication/messaging prompt templates
- remove hiring_rules/hiring_context system prompt injection
- remove per-line truncation and align Mode S Read default with CC
- extract magic numbers to named constants in compaction logic
- split _image_clients.py into image/ package
- split handler_org.py into focused Mixin submodules
- decompose PrimingEngine, config/models, and ConversationMemory god classes
- split slack.py and chatwork.py into focused submodules
- split lifecycle.py, builder.py, chat.py into submodules (WT-3)
- split i18n.py and schemas.py into domain-based packages (Phase A)
- extract agent_sdk.py into _sdk_interrupt, _sdk_options, extended _sdk_stream/_sdk_session
- fix ambiguous test assertion for jira args
- rename current_task.md to current_state.md and AnimaStatus.current_task to active_label
- add missing trust levels, fix lint and spec typo
- restrict emotion_instruction to chat trigger only
- streamline EN/KO/ZH READMEs to match JA structure
- fix indentation, remove redundant imports, update exception catches and tests
- replace generic RuntimeError with domain-specific exceptions
- update tests, docs, and scripts for task tool rename
- trim conceptual integration prompt — remove PII examples, reduce verbosity
- rename task tools — plan_tasks→submit_tasks, add_task→backlog_task
- extract _handle_hard_timeout helper to stay within 85-line budget
- remove all intent="delegation" references from templates
- SDK hook delegates only when subordinate explicitly named
- replace str+Enum with StrEnum in evaluation framework (UP042)
- restructure common_knowledge 00_index.md for clarity
- improve bootstrap loop fix

### Performance
- open output file once per stream thread instead of per line
- optimize Board page — lazy load, caching, incremental polling

### Other
- フィードバックに対応しました
- symple化
- Anima個別のMCP設定を追加
- GoogleTasks対応
- Revert "fix: explicitly disable thinking for openai/ models when thinking=None"
- Revert "fix: prevent bootstrap infinite loop (3 structural bugs)"
- Revert "style: apply ruff format to bootstrap loop fix files"


## [0.5.2] - 2026-03-09

### Added
- add md_to_chatwork() Markdown sanitizer for Chatwork messages
- add encapsulation boundary classification to anima-anatomy docs
- add Heartbeat and Cron tabs to chat sidebar
- slack_channel_post / slack_channel_update as gated external tool actions
- gated external tool actions — default-deny safety valve for dangerous sub-actions
- add TextAnimator for smooth FE streaming text display
- internationalize timezone handling — auto-detect system TZ + configurable override
- add "Enter to send" toggle in Settings page
- dual-query RAG strategy + language-agnostic keyword extraction
- add slack_react tool for emoji reactions (#22)
- add daily RAG indexing to ProcessSupervisor and fix per-anima vectordb
- add per-anima RAG index builder script
- add GitHub Release workflow with LLM-generated release notes
- recursive directory indexing for all memory types (#20)
- change ChromaDB distance metric to cosine similarity (#19)
- normalize system prompt heading hierarchy with XML tags
- add .ragignore support and retriever min_score threshold (#18)
- DK summary injection — replace full-text with title+description lists
- restrict recent_tool_results injection to Mode B only
- budget-aware timeline thinning for LLM activity report input
- i18n audit timeline strings and send plain text to Activity Report LLM
- unified timeline audit with cross-anima merged view
- update LLM prompt for key_activities and add CLI audit --mode report
- add qualitative fields (key_activities, top_tools) to AnimaAuditEntry
- redesign audit report mode with priority-based category display
- rewrite audit_subordinate to match Issue specification
- add Activity Report page (#17)
- add one_shot_completion() with LiteLLM → Agent SDK fallback
- add time-based activity schedule (night mode)
- add global Activity Level slider for heartbeat cost control
- add generic Notion external tool
- add profile subcommand for multi-instance management
- unified outbound budget with role defaults + status.json override
- add demo interactive onboarding with 3-layer experience
- migrate ~480 hardcoded Japanese strings to i18n t() system

### Fixed
- remove stale server.pid before starting in Docker container
- address review feedback (iteration 2)
- protect identity.md from self-modification in Mode S
- update environment.md rule 7 to document subordinate management file permissions
- replace get_event_loop() with get_running_loop() in async contexts
- skip malformed activity log entries instead of failing entire file
- remap 'event' key to 'type' in activity log loader
- use resp is not None instead of isinstance(resp, dict) for SlackResponse
- correct indentation error in test_rag_e2e.py
- assign haiku model to general-role animas in demo
- add missing demo examples and adjust_dates.sh to Dockerfile
- correct indentation error in test_heartbeat_decomposition_e2e.py
- replace date.today() with today_local() in all e2e and housekeeping tests
- force-reset _app_tz in conftest to prevent timezone state leakage
- replace date.today() with today_local() across all test files
- use today_local() in conversation transcript tests to match JST implementation
- use today_local() instead of date.today() in activity log rotation tests
- patch get_credential in x_search tests to isolate from shared/credentials.json
- patch get_credential in web_search tests to isolate from shared/credentials.json
- replace silent except-pass with debug log in asset reconciliation config load
- replace silent except-pass with debug log in gated action permission check
- update template file count test for usecases/ directory
- use safe .get() for threadId in send_message response
- propagate context and reply_to fields in plan_tasks batch handler
- improve Channel D skill matching — configurable threshold & personal-first sort
- add explicit None checks for remaining get_vector_store() callers
- plan_tasks _wake() callback receives unwanted self argument
- graceful degradation when RAG/ChromaDB fails to initialize
- update scheduler E2E test for daily indexing cron job
- robust indexing_time/enabled extraction from consolidation config
- TextAnimator accumulator-based timing and rate calculation
- widen onDone handler scan window in streaming indicator test
- update org context E2E test assertions for refactored comm rules
- align Mode B skill injection test with priming-based matching
- fix TextAnimator timing bug — reset _lastStepTime on idle ticks
- address review feedback (iteration 1)
- update record_access unit tests for DB-read access_count
- address review feedback (iteration 2)
- read access_count from DB in record_access to prevent stale increment
- remove CHUNK-DEBUG temporary logging from chat streaming
- add missing mock attrs for indexing cron + resolve lint violations
- replace silent except-Exception-pass with debug logging
- resolve ruff lint and format violations
- reset ToolPromptStore singleton between tests to prevent cache leak
- unify anima_factory locale fallback to locale→en→ja
- remove unnecessary noqa comment from error handler
- suppress Slack Bolt 404 for unhandled events
- align tests with communication-rules refactor and remove date-sensitive timestamps
- increase asyncio StreamReader buffer for Codex subprocess pipes (#21)
- address review feedback (iteration 2)
- align skills hash pattern with SKILL.md-only indexing
- address review feedback (iteration 1)
- address review feedback (iteration 1)
- update activity report tests for generate_org_timeline mock
- exclude self-addressed messages from outbound limit count
- preserve LLM utils fixes and initial truncation safety net
- remove in-place meta mutation and dead meta_copy code in audit
- add backward compatibility for legacy 'days' parameter
- address review findings (C1/H1-H3/M1/M5)
- add localStorage fallback for activity settings and fix demo splash link
- raise on LLM failure in conversation _call_llm to preserve compression behavior
- add error handling to settings page API calls (night mode save/load)
- replace broad 'except Exception: pass' with KeyError in scheduler_manager
- address review feedback (iteration 1)
- resolve ruff lint (I001 import sort) and format violations
- replace undefined CSS variables in demo splash and suggest cards
- downgrade demo models from opus/sonnet to sonnet/haiku for cost savings
- reuse httpx.Client, add CLI tests (coverage 65% → 92%)
- address review feedback (iteration 1)
- address review feedback (iteration 1)
- centralize consolidation model for all internal LLM calls
- prevent zombie process accumulation via explicit wait() and periodic reaper
- reset _last_progress_at on lock acquisition to prevent busy-hang false positives
- update _run_priming mock return value to match new tuple signature
- correct expected Slack user ID in test_slack_prefix_case_insensitive
- skip Group 6 header for task trigger in system prompt builder
- improve chat renderer and chat styling
- resolve flaky tests caused by hardcoded dates and dedup cache pollution
- restore docs/images/ needed by README

### Changed
- compress communication rules & messaging prompts (~1,000 tokens saved)
- compress environment.md prompt — remove Claude-redundant instructions
- simplify memory_guide — remove skill list, use counts
- move public design docs from docs/implemented/ to docs/specs/
- i18n hardcoded Japanese strings in skill_creator and related modules

### Performance
- truncate tool fields in conversation API and optimize history poll diff

### Other
- bug fix indexer.py
- modify debug systemprompt


## [0.5.1] - 2026-03-06




## [0.5.0] - 2026-03-06

### Added
- update demo defaults to en-business preset with real avatar assets
- add Anima identity (username/icon_url) to slack_send tool
- add Slack notification icons for all Animas
- workspace dashboard live status, KPI polling, and activity streams
- replace tier-based prompt scaling with linear budget allocation
- implement live activity cards for workspace org-dashboard
- add message lines & avatar variants to org dashboard
- replace org-dashboard with canvas node graph layout
- progress-aware busy hang detection replacing counter-based kill
- unified hot-reload system for config and connections
- add copy/download action buttons to assistant chat bubbles
- eliminate Any types with Protocol/TypedDict replacements
- add `animaworks anima rename` CLI command
- fix silent failures in task lifecycle and memory I/O
- unify tool visibility across all execution modes (S/A/B)
- unified housekeeping engine for disk rotation
- auto-inject reply instruction metadata for external platform inbox messages
- enforce MUST task creation in heartbeat/inbox prompts
- config thread-safety and cleanup fixes
- notification channel vault/shared credential support and robustness fixes
- use os.replace for atomic writes and atomic truncation in cron_logger
- handle Slack app_mention events with ts-based dedup and thread reply context
- auto-assign intent on Slack mention/DM for immediate inbox processing
- align skill-creator with Agent Skills spec and fix path resolution
- document external message reception in messaging-guide (ja/en)
- add call_human guide to common_knowledge (ja/en)
- resync prompt DB sections migration + behavior_rules memory-backed advice rule
- add --force flag to stop/restart commands
- validate/repair broken knowledge frontmatter on write and startup
- improve _fetch_node_content fallback and update_graph_incremental type inference
- add LLM API retry for 429/5xx/network errors
- write recovery_note on heartbeat process crash and fix unread_count
- pending task failure safety — file move lifecycle and failure notifications
- per-Anima Slack bot token resolution
- add background_model override for heartbeat/inbox/cron cost reduction
- expose credential vault tools (vault_get/vault_store/vault_list) to Animas
- shared common_knowledge per-anima index with hash-based change detection
- search_memory OR-split and priming keyword fallback
- add AI-speed task deadline guidelines to environment.md
- fix Agent tool intercept and add heartbeat task results visibility
- spreading activation repair + episodes support
- expose all supervisor tools in Mode S MCP + add CLI audit subcommand
- consolidation quality improvements — frontmatter repair, REFLECTION extraction, smart activity filtering
- incremental sync of common_skills/common_knowledge on startup
- add gmail inbox, sent, and search subcommands

### Fixed
- regenerate female business avatars with correct gender prompt
- add gender field to demo character sheets and fix prompt generation
- add missing bubble-actions and voice-controls-slot to workspace CSS/HTML
- add missing chat avatar styles to workspace CSS
- always show amber spinner for any running stream entry
- detect active groups on init so workspace spinners reflect ongoing tasks
- replace silent except-pass with debug logging in reconciliation
- org-dashboard spinner only spins during active tasks, not idle Running
- address review feedback (iteration 1)
- resolve ruff lint errors and time-dependent test failures
- repair 3 broken unit tests
- pass execution mode explicitly to _estimate_tool_overhead
- add :active parity for .org-card:hover selector
- address review feedback (iteration 1)
- prevent error handler double-fault, fix exception hierarchy and restore fallbacks
- address review feedback (iteration 1)
- 7 bugfix batch — error handling, activity log, health check, client lifecycle
- update regression tests for new exception propagation
- resolve per-anima Slack token in CLI path
- MCP supervisor safe fallback + executor context window overrides
- address review feedback (iteration 2)
- remove stale lastChunkTime reference in chat-stream.js
- address review feedback (iteration 1)
- replace broad except Exception with specific custom exceptions
- address review feedback (iteration 1)
- address review feedback (iteration 2)
- replace inbox file creation with activity log for external sends
- address review feedback (iteration 1)
- use _thread.LockType for Python 3.12 isinstance compatibility
- resolve all 8 failing tests on main
- rename E2E test to clarify regression guard intent
- address review feedback (iteration 1)
- correct test mocking targets for CLI staleness tests
- CLI staleness — log path, --local deprecation, exit code, env var unification
- use getattr for record.result_summary robustness
- cron logger data integrity — KeyError on command entries and timezone mismatch
- use record.result_summary for tool_end completed_tools summary
- prevent broadcast() race condition with list snapshot
- use Date comparison for liveIsNewer to handle UTC/JST timezone mismatch
- route call_human replies back to originating Anima
- add num_retries to Mode B LLM calls for transient error resilience
- align repair metadata with ensure_knowledge_frontmatter
- address review feedback (iteration 1)
- address review feedback (iteration 2)
- add defensive type check for reply_to in failure notification path
- update base tool count 27→30 for vault tools addition
- update E2E outbound test for per-Anima slack _send_via_slack signature
- add vault tools to dispatch dict test expectations
- resolve CI test failures — vault MCP schemas and vibe reference realistic path
- disable spreading activation in RAGFilter tests
- resolve review regressions — test alignment, silent exception, code quality
- address review findings — result.id bug, C-method, config params, threading lock
- remove Chatwork-specific Bash blocks and update S-mode tool docs
- address review findings — remove dead imports, fix parse-fail path, include E2E test updates

### Changed
- apply ruff format to server/routes/system.py
- apply ruff format to 3 remaining files
- introduce ruff linting/formatting and improve CI pipeline
- remove deprecated modules and simplify architecture
- unify external platform source constants and deduplicate _detect_slack_intent
- unify claude-opus-4-20250514 refs to claude-opus-4-6
- simplify gmail tool — extract _fetch_emails, Email.to_dict, constants

### Performance
- reduce markdown re-render interval to 30ms / 10 chars
- fix chat streaming stutter — incremental markdown, ASGI middleware, log reduction
- fix streaming chat display jank — RAF batching, log reduction, SSE flush


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

[Unreleased]: https://github.com/xuiltul/animaworks/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/xuiltul/animaworks/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/xuiltul/animaworks/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/xuiltul/animaworks/compare/v0.5.5...v0.6.0
[0.4.3]: https://github.com/xuiltul/animaworks/compare/v0.4.2...v0.4.3
[0.4.0]: https://github.com/xuiltul/animaworks/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/xuiltul/animaworks/compare/v0.3.0...v0.3.1
