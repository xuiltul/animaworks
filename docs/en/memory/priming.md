<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/priming.md -->
<!-- i18n: source-sha256=d5a04a5c993d7ed7bc3ff3758c23f89175249339ec84279d9c33e9ffd1b807f8 generated=2026-11-06 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 193a5e72

# Automatic Recall (Priming)

Automatic recall is the process of retrieving relevant memory cues from the execution path that received input and adding them to the LLM context. The configuration is defined in `core/memory/priming/engine.py` at `PrimingEngine`, and the default profile is defined in `core/config/schemas.py` at `PrimingConfig`.

## Profiles and Channels

The default for `PrimingConfig.profile` is `compact`. You can override an Anima’s configuration by specifying `priming_profile` as `compact` or `full` in its `status.json`. The channels are summarized below.

| Channel | What it retrieves | `compact` | `full` |
|---|---|---:|---:|
| A | Sender profile | ○ | ○ |
| B | Activity log and recent activity in shared channels | Conditional on background triggers | ○ |
| C0 | Pointers to persistent candidates and important knowledge | ○ | ○ |
| C | Search for relevant knowledge | Conditional | ○ |
| E | Incomplete tasks | ○ | ○ |
| F | Past episodes | — | ○ |
| G | Not an independent channel; graph-derived candidates accompanying episode search | — | ○ (within F search) |
| Auxiliary | Recent sends and notifications to humans that have not been sent | ○ | ○ |

At `compact`, channels A, E, and C0 and auxiliary information are gathered in parallel. A search of C is performed when `channel` is `chat` or `task`, or when `intent` is `question`, `request`, or `delegation`. B is fetched on background triggers `heartbeat`・`inbox`・`cron` when `compact_background_recall_enabled` is enabled and the limits on the number of recent activities and tokens are positive. F is also fetched when there is a background trigger and a query, and the episode count and token limits are positive. G is not an independent Priming channel; it may be used within the F search.

## C0 and `always_prime`

When `always_prime: true` is set in a knowledge file, it is treated as a resident candidate for automatic recall on the RAG index. `compact` calls C0 with `resident_only=True` and selects up to 3 normal knowledge pointers from these explicit opt-in candidates. The current `compact` path does not check for matches with the input query and selects resident candidates in order of update date and time. Candidates treated as ACTION-RULE may be injected in the body. Being `[IMPORTANT]` alone is not a reason to inject unrelated knowledge as resident content. `full` searches not only resident candidates but also important knowledge related to the query.

## Search Policy

The search policy is defined in `core/memory/retrieval/unified_search.py` at `TRIGGER_POLICIES`. In the search for `scope="all"`, the scope, number of candidates, and whether reranking is applied vary depending on the trigger. Since C and F specify individual scopes, the candidate count and reranking policy for each trigger are applied while using those scopes.

| Trigger | Main search scope | Rerank |
|---|---|---:|
| `chat` | facts, episodes, knowledge, procedures, activity log | Enabled |
| `inbox` | facts, episodes, activity log | Enabled |
| `heartbeat` | episodes, activity log | Disabled |
| `task` | facts, procedures, knowledge | Enabled |
| `cron` | facts, episodes, knowledge, activity log | Enabled |
| `tool` | Scope for tool search | Enabled |

For individual search stages and the scope of `search_memory`, refer to [Intentional Recall and Search](retrieval.md).

## Limits and Timeouts

The default value of `priming.max_tokens` is 2,000, and it is used as the upper limit for recall results passed to the profile. It is not a mechanism for dynamically allocating this value according to message type. Unsent human notifications are handled under a separate contract and are not excluded to satisfy the normal recall budget.

The default value of `priming.channel_timeout_seconds` is 60 seconds. Limits are set for each channel, and only channels that time out are treated as empty while other retrieval results continue. For all configuration items, refer to the [Configuration Reference](../reference/config.md).
