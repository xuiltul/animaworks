<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/execution.md -->
<!-- i18n: source-sha256=dc706c803aa27d2b8b296c056e63e9b928e46ad6c645d16d576bde53c3d44278 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: b304b7dc

# Model Execution

Resolve the execution mode from the model name and per-anima configuration, then pass it to the SDK, CLI, or API call for each engine. S, C, X, D, and G use dedicated engines, while A runs API models through LiteLLM.

## Execution Modes

| Mode | Primary model name patterns | Engine | Authentication | Tool execution |
|---|---|---|---|---|
| S | `claude-*` | Claude Agent SDK | Claude subscription, API key, or Bedrock or Vertex AI authentication | Uses the SDK's built-in features and MCP, connecting to AnimaWorks' tool handler. |
| C | `codex/*`, `openai-codex/*` | Codex SDK / CLI | Codex CLI login information or configured provider credentials | Uses Codex's execution features, converting to common events and tool records. |
| X | `grok/*` | Grok Build CLI | Grok CLI authentication | Standardizes events and execution results via the CLI's ACP stream. |
| D | `cursor/*` | Cursor Agent CLI | Cursor CLI login information | Uses the CLI's session and tool execution. |
| G | `gemini/*` | Gemini CLI | Google-side CLI authentication | Treats CLI execution results as common events. |
| A | `openai/*`, `azure/*`, `bedrock/*`, `google/*`, `vertex_ai/*`, etc. | LiteLLM loop | Provider API key or cloud credentials | Executes LiteLLM's function/tool call via AnimaWorks' handler. |

The resolution order from the model name is: explicit per-anima specification in `status.json`, model-specific fallback in `models.json` and `config.json`, and the default pattern in `core/config/model_mode.py`. Models that match none of these resolve to A. See the [configuration reference](../reference/config.md) for individual configuration items.

## Common Execution Layer and Failure Behavior

Each engine implementation is placed in `core/execution/engines/`. The common layer's `core/execution/events.py`, `core/execution/session/session_store.py`, `process_runner.py`, `watchdog.py`, `tool_evidence.py`, `cli_stream.py` standardize events, session persistence, process control, and tool evidence. Error classification, process termination, binary search, `clear_session`, and response/error determination are also handled in the intermediate layer. D and G are also included in this configuration.

The watchdog determines that the engine is idle if no engine event arrives within 1200 seconds. It is not a timeout measuring total execution time. Rate limit and provider overload information is shared per provider family in `llm_rate_guard`, preventing other Anima instances from making consecutive requests to the same provider. This guard is designed so that even if reading or writing fails, normal execution is not interrupted.

In addition to the main model, `fallback_model` or `fallback_models` can be specified. Based on classification results such as authentication and rate limit, along with fallback configuration, the system switches to the next candidate. For heartbeat and cron, `background_model`, `background_credential`, and `background_thinking_effort` can be specified.

## Context Management

Context compression methods differ by engine. S saves a baseline amount of conversation to track changes, using the SDK's compression features and idle-time compression. C discards the current thread at a threshold and starts a new thread. X and D record the number of resume turns, rotating sessions every 10 turns. A summarizes the conversation when approaching input limits or on overflow, then retries with shortened history. G follows the dedicated CLI's session behavior and does not perform common session persistence with turn limits.

The default per-anima context threshold is 0.50, and the absolute ceiling is 0.75. The default compaction token threshold for task execution is 0, with a default maximum count of 6. Model-specific compaction thresholds can also be specified in `models.json`.

## Design Decisions

- **Do not build a custom agent loop.** Let the engine (Claude Agent SDK, Codex, each CLI, LiteLLM) handle tool-use iteration; AnimaWorks only handles the common processing around it (prompt assembly, permissions, sessions, error classification, and logging).
