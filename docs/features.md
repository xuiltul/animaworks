# AnimaWorks Feature Index

**[日本語版](features.ja.md)**

> Last updated: 2026-02-18
> Related: [spec.md](spec.md), [memory.md](memory.md), [vision.md](vision.md)

An index of implemented features in the AnimaWorks framework, organized across 18 categories. "Design" links point to design and implementation documents; "Review" links point to code review reports.

---

## 1. Core Architecture

Framework foundation changes including agent.py refactoring, hierarchy design, process isolation, and renaming.

- **agent.py Refactoring Plan** (2026-02-14) — Separation of concerns and class design overhaul for the agent core
  [Design](implemented/20260214_agent-refactoring_implementation.md)
- **Design Document vs. Implementation Gap Notes** (2026-02-14) — Identifying and resolving divergences between original design docs and implemented code
  [Design](implemented/20260214_design-implementation-gap_issue.md)
- **Dynamic System Prompt Injection Architecture** (2026-02-14) — Dynamic construction of system prompts with an 18-section structure
  [Design](implemented/20260214_dynamic-prompt-injection_implementation.md)
- **Hierarchy, Async Communication, and Multi-Model Design** (2026-02-14) — Supervisor hierarchy, A1/A2/B modes, and multi-provider architecture
  [Design](implemented/20260214_hierarchy-and-delegation_design.md)
- **Design Conformance Fix Plan Notes** (2026-02-14) — Correction plan to bridge the gap between design docs and implementation
  [Design](implemented/20260214_plan-gap-fix_implementation.md)
- **API Critical Refactoring for Scaling** (2026-02-15) — FastAPI route splitting and scaling infrastructure
  [Design](implemented/20260215_api-critical-refactoring-for-scaling_implemented-20260215.md) | [Review](implemented/20260215_review_api-critical-refactoring-for-scaling_approved-20260215.md)
- **System Reference Document Creation** (2026-02-15) — Shared knowledge base for autonomous Anima reference
  [Design](implemented/20260215_system-reference-documents_implemented-20260215.md)
- **Person Lifecycle CLI Command Integration** (2026-02-16) — Reorganization of CLI command structure for Anima support
  [Design](implemented/20260216_person-lifecycle-cli-commands-20260217.md) | [Review](implemented/20260217_review_person-lifecycle-cli-commands_approved-20260217.md)
- **Full Rename: Person to Anima** (2026-02-16) — Codebase-wide rename from Person to Anima
  [Design](implemented/20260216_rename-person-to-anima_implemented-20260216.md) | [Review](implemented/20260216_review_rename-person-to-anima_approved-20260216.md)
- **Comprehensive Documentation Overhaul** (2026-02-17) — Synchronizing CLAUDE.md, README, and spec with the current codebase
  [Design](implemented/20260217_documentation-overhaul-implemented-20260217.md) | [Review](implemented/20260217_review_documentation-overhaul_approved-20260217.md)

---

## 2. Execution Engine

Improvements to A1/A2/B modes, Agent SDK crash recovery, SSE enhancements, and more.

- **A2 Agentic Loop Enhancement** (2026-02-15) — Reliability and feature improvements for the LiteLLM + tool_use loop
  [Design](implemented/20260215_a2-agentic-loop-enhancement_implemented-20260215.md) | [Review](implemented/20260215_review_a2-agentic-loop-enhancement_approved-20260215.md)
- **Conversation Data Loss on Error** (2026-02-15) — Context preservation when execution errors occur
  [Design](implemented/20260215_error-conversation-data-loss_implemented-20260215.md) | [Review](implemented/20260215_review_error-conversation-data-loss_approved-20260215.md)
- **Checkpoint Retry on Stream Disconnect** (2026-02-16) — Automatic recovery from SSE stream disconnections
  [Design](implemented/20260216_checkpoint-retry-on-stream-disconnect-implemented-20260216.md) | [Review](implemented/20260216_review_checkpoint-retry-on-stream-disconnect_approved-20260216.md)
- **create_anima Tool Unavailable in Mode A1** (2026-02-16) — Fix for Anima hiring chain failure in Agent SDK mode
  [Design](implemented/20260216_mode-a1-create-anima-unavailable.md) | [Review](implemented/20260216_review_mode-a1-create-anima-unavailable_approved.md)
- **SSE Stream Returns Empty Response After Anima Restart** (2026-02-16) — SSE stream recovery after process restart
  [Design](implemented/20260216_sse-stream-empty-after-anima-restart_implemented-20260216.md) | [Review](implemented/20260216_review_sse-stream-empty-after-anima-restart_approved-20260216.md)
- **Uvicorn Timeout Misconfiguration and Agent SDK Hook Race Condition** (2026-02-16) — Server initialization stability improvements
  [Design](implemented/20260216_uvicorn-timeout-and-agent-sdk-hook-errors_implemented-20260216.md) | [Review](implemented/20260216_review_uvicorn-timeout-and-agent-sdk-hook-errors_approved-20260216.md)
- **Automatic Process Recovery on Agent SDK Crash** (2026-02-17) — Crash detection and automatic restart mechanism
  [Design](implemented/20260217_agent-sdk-crash-recovery_implemented-20260217.md) | [Review](implemented/20260217_review_agent-sdk-crash-recovery_approved-20260217.md)
- **Mode B: Text-Based Pseudo Tool Call Loop** (2026-02-17) — Text-based tool execution for models without native tool_use support
  [Design](implemented/20260217_mode-b-text-based-tool-loop_implemented-20260217.md) | [Review](implemented/20260217_review_mode-b-text-based-tool-loop_approved-20260217.md)
- **SSE Chat Streaming: Consolidation of 3 Duplicate Code Paths** (2026-02-17) — DRY refactoring of SSE streaming code
  [Design](implemented/20260217_sse-chat-code-deduplication_implemented-20260217.md) | [Review](implemented/20260217_review_sse-chat-code-deduplication_approved-20260217.md)

---

## 3. Memory System

Priming, RAG, memory consolidation/forgetting, activity log, streaming journal, and more.

- **Priming Layer Implementation Plan** (2026-02-14) — Overall plan including RAG design and consolidation architecture
  [Design](implemented/20260214_priming-layer_design.md)
- **Priming Layer Phase 1 Implementation** (2026-02-14) — Initial implementation of 4-channel parallel priming
  [Design](implemented/20260214_priming-layer-phase1_implementation.md)
- **Priming Layer Phase 2: Daily Consolidation** (2026-02-14) — NREM sleep-analog daily memory consolidation
  [Design](implemented/20260214_priming-layer-phase2-consolidation_implementation.md)
- **Memory Performance Evaluation Phase 2: Dataset Generation** (2026-02-14) — Automated generation of datasets for memory retrieval accuracy evaluation
  [Design](implemented/20260214_memory-eval-phase2_dataset-generation.md)
- **common_knowledge RAG Infrastructure** (2026-02-15) — Vector search support for the shared knowledge base
  [Design](implemented/20260215_common-knowledge-rag-infrastructure-implemented-20260215.md) | [Review](implemented/20260215_common-knowledge-rag-infrastructure_review-approved-20260215.md)
- **Memory Activation Strength and Active Forgetting Mechanism** (2026-02-15) — Hebbian LTP + 3-stage forgetting implementation
  [Design](implemented/20260215_memory-access-frequency-and-forgetting-implemented-20260215.md) | [Review](implemented/20260215_review_memory-access-frequency-and-forgetting_approved-20260215.md)
- **Neuroscience-Inspired Memory Retrieval Enhancements: 3 Proposals** (2026-02-15) — Spreading activation, temporal decay, and access frequency improvements
  [Design](implemented/20260215_memory-retrieval-neuroscience-enhancements.md)
- **Priming/hiring_context Conflict Fix** (2026-02-15) — Resolving conflicts between priming and hiring context
  [Design](implemented/20260215_priming-hiring-context-fix.md) | [Review](implemented/20260215_review_priming-hiring-context-fix_approved.md)
- **RAG: Performance Degradation from Multiple Embedding Model Initializations** (2026-02-15) — Startup time reduction via singleton pattern
  [Design](implemented/20260215_rag-embedding-model-multi-init_implemented-20260215.md)
- **Migration to Dense Vector Search + Full Knowledge Graph Implementation** (2026-02-15) — BM25 removal, dense vector search unification + NetworkX PageRank
  [Design](implemented/20260215_simplify-rag-to-dense-vector-only_implemented-20260215.md) | [Review](implemented/20260215_review_simplify-rag-to-dense-vector-only_approved-20260215.md)
- **Chunk ID Collision Bug in _chunk_by_markdown_headings** (2026-02-16) — Fix for duplicate IDs in preamble sections
  [Design](implemented/20260216_chunk-id-preamble-collision-20260216.md) | [Review](implemented/20260216_review_chunk-id-preamble-collision_approved-20260216.md)
- **Consolidation Prompt Improvements and LLM Response Logging** (2026-02-16) — Memory consolidation quality improvements and debugging support
  [Design](implemented/20260216_consolidation-prompt-and-logging-improvement-20260216.md)
- **RAG Indexer Chunk ID Duplication Bug & Missing shared_common_knowledge Initialization** (2026-02-16) — RAG index path bug fixes
  [Design](implemented/20260216_rag-indexer-path-bug-and-shared-knowledge-init-20260216.md) | [Review](implemented/20260216_review_rag-indexer-path-bug-and-shared-knowledge-init_approved-20260216.md)
- **Inter-Anima Messages Not Recorded in Episode Memory** (2026-02-17) — Fix for missing DM/channel message memory recording
  [Design](implemented/20260217_anima-message-episode-recording_implemented-20260217.md) | [Review](implemented/20260217_review_anima-message-episode-recording_approved-20260217.md)
- **Daily Consolidation Engine Ignores Episode Files with Suffixes** (2026-02-17) — Fix for filename pattern matching
  [Design](implemented/20260217_consolidation-episode-filename-pattern_implemented-20260217.md)
- **Consolidation Not Running for Stopped Animas** (2026-02-17) — Ensuring consolidation runs for all initialized Animas
  [Design](implemented/20260217_consolidation-run-for-all-animas_implemented-20260217.md) | [Review](implemented/20260217_review_consolidation-run-for-all-animas_approved-20260217.md)
- **Embedding Model Comparison Test Results** (2026-02-17) — Accuracy comparison of multilingual-e5-small vs. other models
  [Design](implemented/20260217_embedding-model-comparison.md)
- **Add episodes/ Path Validation to write_memory_file Tool** (2026-02-17) — Prevention of memory writes to invalid paths
  [Design](implemented/20260217_write-memory-file-episode-path-validation_implemented-20260217.md)
- **Unified Activity Log: Spec Compliance Fixes (6 Items)** (2026-02-18) — Improving activity log compliance with specification
  [Design](implemented/20260218_activity-log-spec-compliance-fixes-implemented-20260218.md) | [Review](implemented/20260218_review_activity-log-spec-compliance_approved-20260218.md)
- **Streaming Journal: Crash-Resilient Response Output Persistence** (2026-02-18) — Fault-tolerant streaming output via Write-Ahead Log
  [Design](implemented/20260218_streaming-journal-implemented-20260218.md) | [Review](implemented/20260218_review_streaming-journal_approved-20260218.md)
- **Unified Activity Log: Single Timeline Recording of All Interactions** (2026-02-18) — Consolidation of transcript/dm_log/heartbeat_history
  [Design](implemented/20260218_unified-activity-log-implemented-20260218.md) | [Review](implemented/20260218_review_unified-activity-log_approved-20260218.md)

---

## 4. Communication & Messaging

Board/shared channels, external messaging, outbound routing, and more.

- **A1 Mode Inter-Anima Messaging Integration** (2026-02-15) — Internal messaging implementation for Agent SDK mode
  [Design](implemented/20260215_a1-messaging-integration_implemented-20260215.md) | [Review](implemented/20260215_review_a1-messaging-integration_approved-20260215.md)
- **Greet Duplicate Invocation and Orphaned Inbox Files** (2026-02-16) — Fix for messaging duplication and orphaned file issues
  [Design](implemented/20260216_greet-duplicate-and-orphaned-inbox-files_implemented-20260216.md) | [Review](implemented/20260216_review_greet-duplicate-and-orphaned-inbox-files_approved-20260216.md)
- **External Messaging Webhook Integration (Slack & Chatwork)** (2026-02-17) — Webhook message reception from external services
  [Design](implemented/20260217_external-messaging-integration_implemented-20260217.md) | [Review](implemented/20260217_review_external-messaging-integration_approved-20260217.md)
- **send_message Unified Outbound Routing** (2026-02-17) — Automatic routing to Slack/Chatwork/internal based on recipient
  [Design](implemented/20260217_send-message-unified-outbound-routing.md)
- **Slack Socket Mode for Real-Time Message Reception** (2026-02-17) — Real-time Slack integration via WebSocket
  [Design](implemented/20260217_slack-socket-mode-integration_implemented-20260217.md) | [Review](implemented/20260217_review_slack-socket-mode-integration_approved-20260217.md)
- **Board Web UI: Dashboard + Workspace** (2026-02-18) — Web UI frontend for shared channels
  [Design](implemented/20260218_channel-webui-and-onboarding_implemented-20260218.md) | [Review](implemented/20260218_review_channel-webui-and-onboarding_approved-20260218.md)
- **Slack-Style Shared Channels + Unified Message Log Architecture** (2026-02-18) — Shared channel infrastructure (#general, #ops, etc.)
  [Design](implemented/20260218_shared-channel-messaging_implemented-20260218.md) | [Review](implemented/20260218_review_shared-channel-messaging_approved-20260218.md)

---

## 5. Scheduling & Lifecycle

Heartbeat, cron, bootstrap, reconciliation, and more.

- **bootstrap.md Unconditionally Deployed to All Animas** (2026-02-14) — Conditional deployment of first-launch instructions
  [Design](implemented/20260214_bootstrap-unconditional-placement_issue.md)
- **Cron Command-Type Task Design** (2026-02-14) — Command-type execution for cron tasks
  [Design](implemented/20260214_cron-command-type_issue.md)
- **Heartbeat Cascade Issue: Fix Investigation Notes** (2026-02-14) — Prevention of cascading heartbeat runaway
  [Design](implemented/20260214_heartbeat-cascade-fix_notes.md)
- **Heartbeat Cascade Issue: Incident Report** (2026-02-14) — Analysis of the 2026-02-13 cascade failure
  [Design](implemented/20260214_heartbeat-cascade_incident.md)
- **Background Bootstrap Execution + Timeout Improvements** (2026-02-15) — Asynchronous first-launch execution and timeout control
  [Design](implemented/20260215_bootstrap-background-execution-and-timeout_implemented-20260215.md) | [Review](implemented/20260215_review_bootstrap-background-execution_approved-20260215.md)
- **Fix Heartbeat Interval to 30 Minutes** (2026-02-15) — Removal of interval parsing in favor of fixed interval
  [Design](implemented/20260215_fix-heartbeat-interval-to-30min_implemented-20260215.md)
- **Per-Anima Autonomous Scheduler** (2026-02-15) — In-process heartbeat/cron execution within child processes
  [Design](implemented/20260215_person-autonomous-scheduler_implemented-20260215.md) | [Review](implemented/20260215_review_person-autonomous-scheduler_approved-20260215.md)
- **Autonomous Task Management: Self-Updating Heartbeat/Cron** (2026-02-15) — Anima self-modification of their own schedules
  [Design](implemented/20260215_self-modify-heartbeat-cron-implemented-20260215.md) | [Review](implemented/20260215_review_self-modify-heartbeat-cron_approved-20260215.md)
- **Supervisor Periodic Reconciliation** (2026-02-15) — Automatic detection and startup of unrunning Animas
  [Design](implemented/20260215_supervisor-person-reconciliation_implemented-20260215.md) | [Review](implemented/20260215_review_supervisor-person-reconciliation_approved-20260215.md)
- **Heartbeat Collision Crash + Orphan Directory Detection + Org Tree Visualization** (2026-02-16) — Heartbeat mutual exclusion and orphan process detection
  [Design](implemented/20260216_heartbeat-collision-orphan-detection-org-tree-implemented-20260216.md) | [Review](implemented/20260216_review_heartbeat-collision-orphan-detection-org-tree_approved-20260216.md)
- **Scheduler Regression and Activity Tab Display Issues** (2026-02-16) — Fix for scheduler regression bug
  [Design](implemented/20260216_scheduler-regression-and-activity-tab-20260216.md) | [Review](implemented/20260216_review_scheduler-regression-and-activity-tab_approved-20260216.md)
- **cron.md Template Improvements + Mode B Skill Injection Fix** (2026-02-17) — Quality improvements for cron templates
  [Design](implemented/20260217_cron-template-and-mode-b-skill-fix_implemented-20260217.md) | [Review](implemented/20260217_review_cron-template-and-mode-b-skill-fix_approved-20260217.md)
- **Context Gap Between Heartbeat and Dialogue + Messaging Improvements** (2026-02-17) — Context sharing between heartbeat and conversations
  [Design](implemented/20260217_heartbeat-dialogue-context-gap-and-messaging_implemented-20260217.md) | [Review](implemented/20260217_review_heartbeat-dialogue-context-gap-and-messaging_approved-20260217.md)
- **SSE Relay of LLM Responses During Heartbeat Processing** (2026-02-17) — Real-time delivery of responses during heartbeat
  [Design](implemented/20260217_heartbeat-sse-relay-implemented-20260217.md)
- **Reconciliation Kills Bootstrapping Animas, Causing Infinite Loop** (2026-02-17) — Bootstrap protection
  [Design](implemented/20260217_protect-bootstrapping-from-reconciliation-implemented-20260217.md) | [Review](implemented/20260217_review_protect-bootstrapping-from-reconciliation_approved-20260217.md)
- **Reconciliation Kills Animas Missing status.json Every 30 Seconds** (2026-02-17) — Protection for Animas before status.json generation
  [Design](implemented/20260217_reconciliation-kills-animas-missing-status-json_implemented-20260217.md) | [Review](implemented/20260217_review_reconciliation-kills-animas-missing-status-json_revision-20260217.md)

---

## 6. Skills & Tools

Trigger-based injection, auto-discovery, background task execution, and more.

- **Tool Call Timeout Improvements and Background Execution** (2026-02-15) — Asynchronous execution of long-running tools
  [Design](implemented/20260215_tool-call-timeout-background-execution_implemented-20260216.md) | [Review](implemented/20260216_review_tool-call-timeout-background-execution_approved-20260216.md)
- **Long-Running Tool Execution Blocks Chat** (2026-02-16) — Non-blocking tool execution
  [Design](implemented/20260216_long-running-tool-chat-blocking_implemented-20260216.md) | [Review](implemented/20260216_review_long-running-tool-chat-blocking_approved-20260216.md)
- **Tool Auto-Discovery, Anima Creation, Hot Reload, and Unified Dispatch** (2026-02-16) — Tool plugin auto-detection and unified dispatching
  [Design](implemented/20260216_tool-auto-discovery-creation-hotreload-implemented-20260216.md) | [Review](implemented/20260216_review_tool-auto-discovery-creation-hotreload_approved-20260216.md)
- **Background Task Submission System (Mode A1 Long-Running Tool Solution)** (2026-02-17) — Introduction of async task queue
  [Design](implemented/20260217_background-task-submission_implemented-20260217.md) | [Review](implemented/20260217_review_background-task-submission_approved-20260217.md)
- **3-Stage Skill Matching + Skill Creator** (2026-02-17) — Description/trigger/keyword 3-stage skill matching
  [Design](implemented/20260217_skill-matching-enhancement-and-skill-creator_implemented-20260218.md) | [Review](implemented/20260217_review_skill-matching-enhancement-and-skill-creator_approved-20260218.md)
- **Switch External Tool Permissions to Default-Allow (Blocklist Mode)** (2026-02-17) — Transition from allowlist to blocklist approach
  [Design](implemented/20260217_tool-permissions-default-all-implemented-20260217.md)
- **Claude Code-Compliant Skill Format + Description-Based Auto-Injection** (2026-02-17) — Trigger-based skill injection architecture
  [Design](implemented/20260217_trigger-based-skill-injection_implemented-20260217.md) | [Review](implemented/20260217_review_trigger-based-skill-injection_approved-20260217.md)

---

## 7. Configuration & Authentication

Credential centralization, role templates, embedding model selection, and more.

- **Unified Credential Management: config.json Priority Cascade and Generic Schema** (2026-02-15) — 3-layer merge management for authentication credentials
  [Design](implemented/20260215_unified-credential-management_implemented-20260215.md) | [Review](implemented/20260215_review_unified-credential-management_approved-20260215.md)
- **Organization Structure (supervisor) Not Synced to config.json** (2026-02-16) — Creation pipeline fix + periodic sync mechanism
  [Design](implemented/20260216_org-structure-config-sync_implemented-20260216.md) | [Review](implemented/20260216_review_org-structure-config-sync_approved-20260216.md)
- **Centralize Credentials to shared/credentials.json** (2026-02-17) — Consolidation of scattered authentication information
  [Design](implemented/20260217_centralize-credentials-to-shared-file_implemented-20260217.md) | [Review](implemented/20260217_review_centralize-credentials-to-shared-file_approved-20260217.md)
- **Make Embedding Model Selectable from config.json** (2026-02-17) — Dynamic RAG model switching
  [Review](implemented/20260217_review_embedding-model-config_approved.md)
- **load_config() mtime-based cache reload** (2026-02-17) — Automatic configuration file reload
  [Review](implemented/20260217_review_config-mtime-reload_approved-20260217.md)
- **Role Template Introduction: Anima "Job Title + Ability Score" System** (2026-02-17) — Templatization of 6 roles
  [Design](implemented/20260217_role-templates-and-ability-scores-implemented-20260217.md) | [Review](implemented/20260217_review_role-templates-and-ability-scores_approved-20260217.md)

---

## 8. Web UI: Dashboard

SPA migration, activity timeline, scheduler tab, and more.

- **Dashboard GUI: Migration to SPA Architecture** (2026-02-15) — Transition from static HTML to Single Page Application
  [Design](implemented/20260215_dashboard-gui-spa-migration_implemented-20260215.md) | [Review](implemented/20260215_review_dashboard-gui-spa-migration_approved-20260215.md)
- **Visual Representation Improvements for Anima State** (2026-02-15) — State display for Sleeping/Bootstrapping/Active
  [Design](implemented/20260215_bootstrap-ui-during-creation_implemented-20260215.md) | [Review](implemented/20260215_review_bootstrap-ui-during-creation_approved-20260215.md)
- **Batch Fix for Dashboard UI Display Issues** (2026-02-15) — Comprehensive fix for multiple UI display problems
  [Design](implemented/20260215_fix-dashboard-ui-display-issues_implemented-20260215.md) | [Review](implemented/20260215_review_fix-dashboard-ui-display-issues_approved-20260215.md)
- **Web UI: Bootstrap Messages Not Displayed on Frontend** (2026-02-15) — Real-time display of bootstrap progress
  [Design](implemented/20260215_webui-bootstrap-message-invisible_implemented-20260215.md)
- **Web UI: System Status Display Broken** (2026-02-15) — Fix for missing scheduler_running field
  [Design](implemented/20260215_webui-status-display-broken_implemented-20260215.md)
- **Activity Timeline: Message Display Loss, Cron Reliability, and Format Consistency** (2026-02-16) — Timeline display quality improvements
  [Design](implemented/20260216_activity-timeline-message-cron-reliability_implemented-20260216.md) | [Review](implemented/20260216_review_activity-timeline-message-cron-reliability_approved-20260216.md)
- **Activity Timeline: Inter-Anima Message Display, Pagination, and Filter UI** (2026-02-16) — Feature enhancements for the timeline
  [Design](implemented/20260216_activity-timeline-message-visibility-and-pagination-implemented-20260216.md) | [Review](implemented/20260216_review_activity-timeline-message-visibility-and-pagination_revision-20260216.md)
- **crypto.randomUUID Crashes All Frontend Features in Non-Secure Context** (2026-02-17) — Fix for UUID generation in HTTP context
  [Design](implemented/20260217_fix-crypto-randomuuid-crash_implemented-20260217.md)
- **Activity Timeline Message Detail Popup** (2026-02-17) — Click-to-view detail display for timeline entries
  [Design](implemented/20260217_timeline-message-detail-popup_implemented-20260217.md) | [Review](implemented/20260217_review_timeline-message-detail-popup_approved-20260217.md)

---

## 9. Web UI: Workspace

3D office, character display, responsive design, iPad support, and more.

- **Frontend (Workspace) Design** (2026-02-14) — Initial design for Three.js + WebSocket workspace
  [Design](implemented/20260214_frontend-viewer_implementation.md)
- **3D Office Character Simulation: Detailed Implementation Spec** (2026-02-14) — Specification for Three.js-based 3D office
  [Design](implemented/20260214_office-simulation_issue.md)
- **Workspace Chat: Emotion Tag Display & Status Notification Duplication** (2026-02-15) — UI fixes for emotion expression and status notifications
  [Design](implemented/20260215_fix-workspace-chat-emotion-tag-and-status-notifications_implemented-20260215.md) | [Review](implemented/20260215_review_fix-workspace-chat-emotion-tag-and-status-notifications_approved-20260215.md)
- **Workspace: Anima Birth Reveal Animation** (2026-02-15) — Animation effects for new Anima creation
  [Design](implemented/20260215_person-birth-reveal-animation_implemented-20260215.md) | [Review](implemented/20260215_review_person-birth-reveal-animation_approved-20260215.md)
- **Remove Live2D Canvas Procedural Rendering: Simplification to Static Illustrations** (2026-02-15) — Migration from Live2D to static image display
  [Design](implemented/20260215_remove-live2d-canvas-rendering_implemented-20260215.md) | [Review](implemented/20260215_review_remove-live2d-canvas-rendering_approved-20260215.md)
- **Workspace Chat Bubble Text Cutoff** (2026-02-15) — Fix for chat bubble text display truncation
  [Design](implemented/20260215_workspace-chat-bubble-cutoff_implemented-20260215.md) | [Review](implemented/20260215_review_workspace-chat-bubble-cutoff_approved-20260215.md)
- **Workspace 3D Character Click-to-Greet** (2026-02-15) — Interaction with 3D characters
  [Design](implemented/20260215_workspace_character_greeting_implemented-20260215.md) | [Review](implemented/20260215_review_workspace_character_greeting_approved-20260215.md)
- **GLB Cache scene.clone(true) Breaks Skeleton Binding** (2026-02-16) — Fix for SkinnedMesh cloning 3D model issues
  [Design](implemented/20260216_glb-skinnedmesh-clone-skeleton-binding-broken-20260216.md) | [Review](implemented/20260216_review_glb-skinnedmesh-clone-skeleton-binding-broken_approved-20260216.md)
- **Responsive Design: Dashboard & Workspace Mobile UX** (2026-02-16) — Mobile and tablet support
  [Design](implemented/20260216_responsive-design-mobile-ux_implemented-20260216.md) | [Review](implemented/20260216_review_responsive-design-mobile-ux_approved-20260216.md)
- **Workspace 3D Office: Character Scale Anomaly + Org Hierarchy Tree Bug** (2026-02-16) — Fixes for 3D scaling and tree layout
  [Design](implemented/20260216_workspace-character-scale-and-hierarchy-bug_implemented-20260216.md) | [Review](implemented/20260216_review_workspace-character-scale-and-hierarchy-bug_approved-20260216.md)
- **Workspace iPad Display Fix** (2026-02-16) — Viewport, timeline placement, and responsive adjustments
  [Design](implemented/20260216_workspace-ipad-viewport-fix_implemented-20260216.md) | [Review](implemented/20260216_review_workspace-ipad-viewport-fix_approved-20260216.md)
- **Workspace 3D Office: Tree Layout Not Working, All Characters in a Row** (2026-02-16) — Layout fix based on organizational hierarchy
  [Design](implemented/20260216_workspace-tree-layout-broken_implemented-20260216.md)
- **Workspace 3D Character Scaling Fix** (2026-02-17) — Fix for character display size
  [Design](implemented/20260217_workspace-character-scaling-fix_implemented-20260217.md)

---

## 10. Web UI: Chat

Infinite scroll, multimodal image input, SSE reconnection, and more.

- **Chat Message Duplicate Display** (2026-02-15) — Fix for duplicate messages from WebSocket and SSE
  [Design](implemented/20260215_chat-duplicate-message_implemented-20260215.md) | [Review](implemented/20260215_review_chat-duplicate-message_approved-20260215.md)
- **Greeting Message Duplication and Prompt Improvement** (2026-02-15) — Deduplication of greeting messages
  [Design](implemented/20260215_greeting-duplicate-and-prompt-improvement-20260216.md) | [Review](implemented/20260216_review_greeting-duplicate-and-prompt-improvement_approved.md)
- **Loading Indicator Improvements During Streaming** (2026-02-15) — UI improvements during tool call execution
  [Design](implemented/20260215_tool-call-loading-indicator-visibility-implemented-20260215.md) | [Review](implemented/20260215_review_tool-call-loading-indicator-visibility_approved-20260215.md)
- **Chat History Infinite Scroll (Pagination)** (2026-02-17) — Dynamic loading of past chat history
  [Design](implemented/20260217_chat-history-infinite-scroll_implemented-20260217.md) | [Review](implemented/20260217_review_chat-history-infinite-scroll_approved-20260217.md)
- **Web UI Chat Multimodal Image Input** (2026-02-17) — Image attachment and sending in chat
  [Design](implemented/20260217_multimodal-image-input-for-chat_implemented-20260217.md) | [Review](implemented/20260217_review_multimodal-image-input-for-chat_approved-20260217.md)
- **SSE Stream Reconnection and Progress State Recovery** (2026-02-17) — Automatic reconnection and state restoration on SSE disconnect
  [Design](implemented/20260217_sse-reconnection-and-progress-recovery_implemented-20260217.md) | [Review](implemented/20260217_review_sse-reconnection-and-progress-recovery_approved-20260217.md)

---

## 11. Asset Generation

Image generation pipeline, expression variants, 3D model caching, NovelAI V4, and more.

- **Character Image Art Style Consistency** (2026-02-14) — Techniques for unifying art style across multiple Animas
  [Design](implemented/20260214_avatar-style-consistency_issue.md) | [Review](implemented/20260214_review_avatar-style-consistency_revision.md)
- **Bust-Up Expression Variant System** (2026-02-14) — Generating expression variations based on emotions
  [Design](implemented/20260214_bustup-expression-system_implemented-20260215.md) | [Review](implemented/20260214_review_bustup-expression-system_approved-20260215.md)
- **Character Image & 3D Model Generation Pipeline** (2026-02-14) — Unified pipeline integrating NovelAI/Flux/Meshy
  [Design](implemented/20260214_image-gen-pipeline_issue.md)
- **Auto-Apply Supervisor Image as Vibe Transfer Reference on Subordinate Creation** (2026-02-15) — Automation of art style inheritance
  [Design](implemented/20260215_supervisor-image-as-vibe-reference_implemented-20260215.md) | [Review](implemented/20260215_review_supervisor-image-as-vibe-reference_approved-20260215.md)
- **3-Layer Optimization for 3D Model Downloads: Caching, Compression, and Reduction** (2026-02-16) — GLB model performance improvements
  [Design](implemented/20260216_3d-model-cache-and-optimization_implemented-20260216.md) | [Review](implemented/20260216_review_3d-model-cache-and-optimization_approved-20260216.md)
- **Auto-Generation Fallback Pipeline for Missing Assets** (2026-02-16) — Automatic fallback when images are not generated
  [Design](implemented/20260216_asset-generation-fallback-pipeline_implemented-20260216.md) | [Review](implemented/20260216_review_asset-generation-fallback-pipeline_approved-20260216.md)
- **Asset Reconciler: LLM-Based Automatic Image Prompt Synthesis** (2026-02-16) — Image prompt generation from character information
  [Design](implemented/20260216_asset-reconciler-llm-prompt-synthesis_implemented-20260216.md) | [Review](implemented/20260216_review_asset-reconciler-llm-prompt-synthesis_approved-20260216.md)
- **Character Asset Remake (Vibe Transfer + Web UI Preview)** (2026-02-16) — Style transfer for existing images
  [Design](implemented/20260216_character-asset-remake-with-style-transfer-implemented-20260216.md) | [Review](implemented/20260216_review_character-asset-remake-with-style-transfer_revision-20260216.md)
- **NovelAI V4/V4.5 Vibe Transfer: 500 Error Due to Unused encode-vibe** (2026-02-16) — Vibe Transfer API fix
  [Design](implemented/20260216_novelai-v4-vibe-transfer-encode-fix_implemented-20260216.md)

---

## 12. Process Management & IPC

Zombie detection, keepalive, buffer overflow fixes, and more.

- **Process Isolation Architecture** (2026-02-14) — Anima isolation via Unix Domain Socket + child processes
  [Design](implemented/20260214_process-isolation_issue.md) | [Review](implemented/20260214_review_process-isolation-design_revision.md)
- **IPC Layer Fails to JSON-Serialize datetime Objects** (2026-02-15) — Fix for IPC communication serialization
  [Design](implemented/20260215_fix-ipc-datetime-serialization_implemented-20260215.md) | [Review](implemented/20260215_review_fix-ipc-datetime-serialization_approved-20260215.md)
- **Individual Anima Process Management API** (2026-02-16) — REST API-based process control
  [Design](implemented/20260216_individual-anima-restart-api_implemented-20260216.md) | [Review](implemented/20260216_review_individual-anima-restart-api_approved-20260216.md)
- **ping() Does Not Increment Counter in FAILED State, Causing Zombie Processes** (2026-02-16) — Fix for zombie process detection
  [Design](implemented/20260216_ipc-ping-counter-zombie-state_implemented-20260216.md)
- **IPC readline() Buffer Limit 64KB Causes Crash on Large Messages** (2026-02-16) — Buffer overflow mitigation
  [Design](implemented/20260216_ipc-readline-buffer-overflow_implemented-20260216.md)
- **Introduce Keep-Alive and Inter-Chunk Timeout for IPC/SSE Streams** (2026-02-16) — Connection maintenance and timeout control
  [Design](implemented/20260216_ipc-stream-keepalive-and-chunk-timeout-20260216.md) | [Review](implemented/20260216_review_ipc-stream-keepalive-and-chunk-timeout_approved-20260216.md)
- **is_alive() Cannot Detect IPC Connection Death** (2026-02-16) — Health check accuracy improvement
  [Design](implemented/20260216_is-alive-ipc-death-detection_implemented-20260216.md)
- **Server Stop/Start Resilience on PID File Loss** (2026-02-17) — PID file robustness improvement
  [Review](implemented/20260217_review_pid-file-resilience_approved-20260217.md)
- **Comprehensive WebSocket Connection Stability Improvements** (2026-02-16) — WebSocket communication reliability enhancement
  [Review](implemented/20260216_review_websocket-stability-improvements_approved-20260216.md)

---

## 13. Human Notification

call_human integration, org structure prompt injection, and more.

- **Top-Level Anima Human Notification: A1 Messaging Integration** (2026-02-15) — Human notification from supervisor-less Animas
  [Design](implemented/20260215_a1-messaging-integration_human-notification_implemented-20260215.md) | [Review](implemented/20260215_review_a1-messaging-integration_human-notification_approved-20260215.md)
- **Notify Human Chat Window Integration Verification** (2026-02-15) — Chat UI integration for human notifications
  [Design](implemented/20260215_notify-human-chat-window-integration-implemented-20260216.md) | [Review](implemented/20260216_review_notify-human-chat-window-integration_approved-20260216.md)
- **Org Structure Information Injection into System Prompt** (2026-02-15) — Auto-injection of supervisor/subordinates/peers
  [Design](implemented/20260215_org-structure-prompt-injection_implemented-20260215.md) | [Review](implemented/20260215_review_org-structure-prompt-injection_approved-20260215.md)
- **Unified call_human: Consolidating Human Contact into a Single Function** (2026-02-17) — Integration of Slack/Chatwork/LINE/Telegram/ntfy
  [Design](implemented/20260217_unify-call-human-notification_implemented-20260217.md) | [Review](implemented/20260217_review_unify-call-human-notification_approved-20260217.md)

---

## 14. Setup & Onboarding

Setup wizard, i18n, auto-start, and more.

- **Animas Fail to Start After Setup (RAG Initialization Timeout)** (2026-02-15) — Fix for initialization flow
  [Design](implemented/20260215_person-startup-timeout-after-setup_implemented-20260215.md)
- **Expand Setup Screen Language Selector to 17-Language Dropdown** (2026-02-15) — Enhanced multilingual support
  [Design](implemented/20260215_setup-language-selector-expansion_implemented-20260215.md)
- **Setup Wizard: Add User Info Step & Anima Auto-Start** (2026-02-15) — Expanded initial setup flow
  [Design](implemented/20260215_setup-user-info-and-person-autostart-20260215.md) | [Review](implemented/20260215_review_setup-user-info-and-person-autostart_approved-20260215.md)
- **Setup Wizard: i18n Not Applied & Browser Cache Issues** (2026-02-15) — Internationalization and display fixes
  [Design](implemented/20260215_setup-wizard-i18n-and-cache-fix-20260215.md) | [Review](implemented/20260215_review_setup-wizard-i18n-and-cache-fix_approved-20260215.md)
- **Setup Wizard: Simplify Character Creation Step** (2026-02-15) — Change to leader creation step
  [Design](implemented/20260215_setup-wizard-simplify-character-step_implemented-20260215.md)
- **GUI Setup Wizard: Web UI-Based Initial Configuration on First Launch** (2026-02-15) — Web UI initial setup flow
  [Design](implemented/20260215_setup-wizard_implemented-20260215.md) | [Review](implemented/20260215_review_setup-wizard_approved-20260215.md)

---

## 15. Logging & Observability

Logging enhancements, frontend log delivery, and more.

- **Comprehensive Logging Infrastructure Enhancement** (2026-02-17) — Frontend server logging + structlog + backend traceability
  [Design](implemented/20260217_comprehensive-logging-enhancement_implemented-20260217.md) | [Review](implemented/20260217_review_comprehensive-logging-enhancement_approved-20260217.md)
- **Fix Frontend Logs Not Reaching Server** (2026-02-17) — Fix for log delivery pipeline
  [Design](implemented/20260217_fix-frontend-log-delivery_implemented-20260217.md) | [Review](implemented/20260217_review_fix-frontend-log-delivery_approved-20260217.md)

---

## 16. Testing & Quality

Test fixes, 500 error root cause analysis, and more.

- **500 Server Error Root Cause and Comprehensive Error Handling Improvements** (2026-02-15) — Fix for app.state.animas KeyError
  [Design](implemented/20260215_500-server-error-root-cause-implemented-20260216.md) | [Review](implemented/20260216_review_500-server-error-root-cause_approved-20260216.md)
- **Test Fixes Following Phase 3 ProcessSupervisor Refactoring** (2026-02-15) — Test adaptation after Supervisor migration
  [Design](implemented/20260215_fix-failing-unit-tests-after-supervisor-refactor_implemented-20260215.md) | [Review](implemented/20260215_review_fix-failing-unit-tests-after-supervisor-refactor_approved-20260215.md)
- **Fix Remaining 23 Test Failures** (2026-02-15) — Test suite stabilization
  [Design](implemented/20260215_fix-remaining-23-test-failures_implemented-20260215.md) | [Review](implemented/20260215_review_fix-remaining-23-test-failures_approved-20260215.md)
- **Fix app.state.animas KeyError (500 Server Error)** (2026-02-15) — Server state initialization fix
  [Review](implemented/20260215_review_fix-state-persons-keyerror_approved-20260215.md)
- **Test Suite: Fix 135 Failures and Errors** (2026-02-17) — Large-scale test repair
  [Design](implemented/20260217_test-suite-135-failures-cleanup.md)

---

## 17. Security

Memory write security, licensing, and more.

- **Licensing Strategy Design** (2026-02-14) — Apache-2.0 licensing strategy
  [Design](implemented/20260214_licensing-strategy_design.md)
- **Memory Write Security: Protected Files and Path Traversal Prevention Across All Execution Modes** (2026-02-15) — Security hardening for memory writes
  [Design](implemented/20260215_memory-write-security-20260216.md) | [Review](implemented/20260216_review_memory-write-security_approved-20260216.md)

---

## 18. Anima Creation

Hybrid creation, dynamic prompt injection, and more.

- **Hybrid Anima Creation: Unified create_anima Tool + Character Sheet Specification** (2026-02-16) — Unified creation via tool invocation + MD character sheet
  [Design](implemented/20260216_person-creation-hybrid-and-create-tool_implemented-20260216.md) | [Review](implemented/20260216_review_person-creation-hybrid-and-create-tool_approved-20260216.md)
