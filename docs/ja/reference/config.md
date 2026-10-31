<!-- 自動生成ファイル・編集禁止。再生成: uv run python scripts/gen_reference.py config -->
<!-- generator: gen_reference/1  kind: config  source-sha256: 34b4266f4b3eb98b0050e05b06a3821bc9b1435dc191ec4fec13001546c8b2a7 -->

# 設定リファレンス

`AnimaWorksConfig`、per-anima `ModelConfig`、`models.json` の定義から生成しています。

## `config.json`

### `version`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `version` | `int` | `1` | 設定ファイル形式のバージョン。 |

### `setup_complete`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `setup_complete` | `bool` | `false` | 初期セットアップが完了したかどうか。 |

### `locale`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `locale` | `str` | `"ja"` | システム全体で使用する既定の言語。 |

### `system`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `system` | `SystemConfig` | `{SystemConfig}` | 実行環境、タイムゾーン、基本動作に関する設定。 |
| `system.mode` | `str` | `"server"` | — |
| `system.timezone` | `str` | `""` | IANA TZ name; empty = auto-detect from system |

### `credentials`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `credentials` | `dict[str, CredentialConfig]` | `{"anthropic":{"model":"CredentialConfig"}}` | 外部モデル・サービスの認証情報。 |
| `credentials.type` | `str` | `"api_key"` | — |
| `credentials.api_key` | `str` | `""` | — |
| `credentials.keys` | `dict[str, str]` | `{}` | — |
| `credentials.base_url` | `str \| None` | `null` | — |

### `model_modes`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `model_modes` | `dict[str, str]` | `{}` | モデル名パターンと実行モードの対応。 |

### `model_context_windows`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `model_context_windows` | `dict[str, int]` | `{}` | モデルごとのコンテキスト長の互換設定。models.json の使用を推奨。 |

### `model_max_tokens`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `model_max_tokens` | `dict[str, int]` | `{}` | モデル名パターンごとの既定出力トークン数。 |

### `anima_defaults`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `anima_defaults` | `AnimaDefaults` | `{AnimaDefaults}` | 各 anima に適用するモデル・実行設定の既定値。 |
| `anima_defaults.model` | `str` | `"claude-sonnet-5-5"` | — |
| `anima_defaults.fallback_model` | `str \| None` | `null` | — |
| `anima_defaults.fallback_models` | `list[str]` | `[]` | — |
| `anima_defaults.background_model` | `str \| None` | `null` | — |
| `anima_defaults.background_credential` | `str \| None` | `null` | — |
| `anima_defaults.background_thinking_effort` | `str \| None` | `null` | heartbeat/cron thinking effort override |
| `anima_defaults.voice_thinking_effort` | `str \| None` | `null` | voice chat thinking effort override |
| `anima_defaults.max_tokens` | `int` | `8192` | — |
| `anima_defaults.credential` | `str` | `"anthropic"` | — |
| `anima_defaults.context_threshold` | `float` | `0.5` | — |
| `anima_defaults.context_absolute_ceiling` | `float` | `0.75` | — |
| `anima_defaults.task_compaction_tokens` | `int` | `0` | — |
| `anima_defaults.task_compaction_max` | `int` | `6` | — |
| `anima_defaults.max_session_age_hours` | `float` | `24.0` | — |
| `anima_defaults.conversation_history_threshold` | `float` | `0.3` | — |
| `anima_defaults.execution_mode` | `str \| None` | `null` | None = auto-detect from model |
| `anima_defaults.supervisor` | `str \| None` | `null` | — |
| `anima_defaults.speciality` | `str \| None` | `null` | — |
| `anima_defaults.extra_mcp_servers` | `dict[str, dict]` | `{}` | — |
| `anima_defaults.thinking` | `bool \| None` | `null` | Extended thinking (Bedrock: reasoning_effort, Ollama: think) |
| `anima_defaults.thinking_effort` | `str \| None` | `null` | "low"/"medium"/"high"/"max" (default: "high") |
| `anima_defaults.mode_s_auth` | `str \| None` | `null` | Mode S auth: "max"\|"api"\|"bedrock"\|"vertex"\|None(=max) |
| `anima_defaults.default_workspace` | `str` | `""` | — |
| `anima_defaults.consolidation_enabled` | `bool` | `true` | — |
| `anima_defaults.heartbeat_enabled` | `bool` | `true` | 既定true。falseで定期heartbeatのみ無効化。メッセージ起因HB・cronは影響なし |
| `anima_defaults.token_budget_monthly` | `int \| None` | `null` | None = monthly token usage is unlimited |

### `animas`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `animas` | `dict[str, AnimaModelConfig]` | `{}` | anima ごとのモデル設定上書き。 |
| `animas.supervisor` | `str \| None` | `null` | — |
| `animas.company` | `str \| None` | `null` | — |
| `animas.speciality` | `str \| None` | `null` | — |
| `animas.model` | `str \| None` | `null` | — |
| `animas.heartbeat_enabled` | `bool \| None` | `null` | — |
| `animas.background_review_enabled` | `bool \| None` | `null` | — |
| `animas.token_budget_monthly` | `int \| None` | `null` | — |
| `animas.aliases` | `list[str]` | `[]` | — |

### `consolidation`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `consolidation` | `ConsolidationConfig` | `{ConsolidationConfig}` | 記憶統合の動作とスケジュール。 |
| `consolidation.daily_enabled` | `bool` | `true` | — |
| `consolidation.weekly_distillation_enabled` | `bool` | `true` | — |
| `consolidation.synaptic_downscaling_enabled` | `bool` | `true` | — |
| `consolidation.skill_autolearn_enabled` | `bool` | `true` | — |
| `consolidation.curator_auto_apply_enabled` | `bool` | `false` | — |
| `consolidation.daily_time` | `str` | `"02:00"` | Format: HH:MM |
| `consolidation.min_episodes_threshold` | `int` | `1` | — |
| `consolidation.llm_model` | `str` | `"claude-sonnet-5-5"` | — |
| `consolidation.llm_credential` | `str` | `""` | — |
| `consolidation.weekly_llm_model` | `str \| None` | `null` | — |
| `consolidation.weekly_llm_credential` | `str \| None` | `null` | — |
| `consolidation.llm_fallback_model` | `str \| None` | `null` | — |
| `consolidation.llm_fallback_credential` | `str \| None` | `null` | — |
| `consolidation.fact_reconcile_model` | `str \| None` | `null` | — |
| `consolidation.fact_reconcile_credential` | `str \| None` | `null` | — |
| `consolidation.episode_summary_max_input_bytes` | `int` | `204800` | Maximum UTF-8 prompt size for each daily episode-summary LLM call. |
| `consolidation.episode_summary_backfill_days` | `int` | `7` | Look back this many local days for unprocessed daily episode activity. |
| `consolidation.episode_summary_backfill_max_days_per_run` | `int` | `1` | Maximum older days to backfill during one daily consolidation (yesterday is separate). |
| `consolidation.episode_summary_exclude_noop_cron` | `bool` | `true` | Exclude 'did nothing' cron executions from daily episode-summary input. |
| `consolidation.ipc_timeout_base_seconds` | `int` | `1800` | — |
| `consolidation.ipc_timeout_per_activity_entry_seconds` | `float` | `4.0` | — |
| `consolidation.ipc_timeout_per_episode_seconds` | `float` | `120.0` | — |
| `consolidation.ipc_timeout_max_seconds` | `int` | `7200` | — |
| `consolidation.weekly_ipc_timeout_seconds` | `int` | `3600` | — |
| `consolidation.max_concurrent_animas` | `int` | `3` | Maximum number of Anima daily/weekly consolidations to run concurrently. |
| `consolidation.weekly_enabled` | `bool` | `false` | — |
| `consolidation.weekly_time` | `str` | `"sun:03:00"` | Format: day:HH:MM |
| `consolidation.indexing_enabled` | `bool` | `true` | Daily RAG indexing toggle |
| `consolidation.indexing_time` | `str` | `"04:00"` | Format: HH:MM |
| `consolidation.knowledge_self_correction_enabled` | `bool` | `true` | — |
| `consolidation.knowledge_self_correction_max_reconsolidation_files` | `int` | `5` | — |
| `consolidation.knowledge_self_correction_timeout_seconds` | `int` | `300` | — |
| `consolidation.fact_extraction_chunk_chars` | `int` | `12000` | Maximum characters per atomic-fact-extraction chunk during consolidation. 0 (or negative) disables splitting. |
| `consolidation.post_processing_cooldown_seconds` | `int` | `30` | — |
| `consolidation.inactivity_skip_enabled` | `bool` | `true` | — |
| `consolidation.inactivity_days` | `int` | `7` | — |
| `consolidation.episode_summary_input_profile` | `Literal['full', 'compact']` | `"compact"` | — |
| `consolidation.episode_summary_cron_digest_min_runs` | `int` | `6` | — |
| `consolidation.episode_summary_cron_digest_max_notable_runs` | `int` | `5` | — |
| `consolidation.episode_summary_tool_use_max_bytes` | `int` | `300` | — |
| `consolidation.episode_summary_bash_tool_use_max_bytes` | `int` | `120` | — |
| `consolidation.episode_summary_error_tail_bytes` | `int` | `300` | — |
| `consolidation.episode_summary_max_output_tokens` | `int` | `4096` | — |
| `consolidation.live_fact_extraction_enabled` | `bool` | `true` | — |
| `consolidation.live_fact_model` | `str \| None` | `null` | — |
| `consolidation.live_fact_credential` | `str \| None` | `null` | — |
| `consolidation.live_fact_min_input_chars` | `int` | `200` | — |
| `consolidation.live_fact_max_input_chars` | `int` | `24000` | — |
| `consolidation.live_fact_debounce_seconds` | `int` | `120` | — |

### `helper_models`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `helper_models` | `HelperModelsConfig` | `{HelperModelsConfig}` | 補助LLM呼び出しごとのモデル、認証情報、明示的な代替候補、one-shot方針。 |
| `helper_models.episode_summary` | `HelperModelRole \| None` | `null` | — |
| `helper_models.episode_summary.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.episode_summary.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.episode_summary.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.episode_summary.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.episode_summary.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.episode_summary.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.episode_summary.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.fact_extraction` | `HelperModelRole \| None` | `null` | — |
| `helper_models.fact_extraction.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.fact_extraction.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.fact_extraction.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.fact_extraction.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.fact_extraction.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.fact_extraction.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.fact_extraction.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.fact_reconcile` | `HelperModelRole \| None` | `null` | — |
| `helper_models.fact_reconcile.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.fact_reconcile.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.fact_reconcile.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.fact_reconcile.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.fact_reconcile.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.fact_reconcile.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.fact_reconcile.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.weekly_consolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.weekly_consolidation.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.weekly_consolidation.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.weekly_consolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.weekly_consolidation.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.weekly_consolidation.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.weekly_consolidation.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.weekly_consolidation.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.project_consolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.project_consolidation.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.project_consolidation.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.project_consolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.project_consolidation.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.project_consolidation.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.project_consolidation.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.project_consolidation.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.conversation_compression` | `HelperModelRole \| None` | `null` | — |
| `helper_models.conversation_compression.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.conversation_compression.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.conversation_compression.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.conversation_compression.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.conversation_compression.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.conversation_compression.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.conversation_compression.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.distillation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.distillation.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.distillation.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.distillation.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.distillation.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.distillation.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.distillation.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.distillation.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.reconsolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.reconsolidation.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.reconsolidation.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.reconsolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.reconsolidation.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.reconsolidation.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.reconsolidation.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.reconsolidation.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.asset_reconcile` | `HelperModelRole \| None` | `null` | — |
| `helper_models.asset_reconcile.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.asset_reconcile.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.asset_reconcile.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.asset_reconcile.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.asset_reconcile.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.asset_reconcile.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.asset_reconcile.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.meeting_summary` | `HelperModelRole \| None` | `null` | — |
| `helper_models.meeting_summary.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.meeting_summary.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.meeting_summary.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.meeting_summary.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.meeting_summary.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.meeting_summary.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.meeting_summary.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |
| `helper_models.default` | `HelperModelRole` | `{HelperModelRole}` | — |
| `helper_models.default.model` | `str \| None` | `null` | Primary helper model identifier. |
| `helper_models.default.credential` | `str \| None` | `null` | Credential name for the primary model. |
| `helper_models.default.fallbacks` | `list[HelperModelFallback]` | `[]` | Ordered explicit fallback models. |
| `helper_models.default.fallbacks.model` | `str` | `"—"` | Fallback model identifier. |
| `helper_models.default.fallbacks.credential` | `str \| None` | `null` | Credential name for the fallback model. |
| `helper_models.default.allow_agent_sdk_fallback` | `bool` | `false` | Allow one-shot calls for this role to fall back to the Claude Agent SDK. |
| `helper_models.default.max_output_tokens` | `int \| None` | `null` | Maximum output tokens for one-shot calls made by this role. |

### `background_review`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `background_review` | `BackgroundReviewConfig` | `{BackgroundReviewConfig}` | セッション後の非同期振り返りと人物像更新の設定。 |
| `background_review.enabled` | `bool` | `true` | — |
| `background_review.chat_every_user_turns` | `int` | `10` | — |
| `background_review.min_interval_minutes` | `int` | `10` | — |
| `background_review.max_per_day` | `int` | `24` | — |
| `background_review.max_input_bytes` | `int` | `61440` | — |
| `background_review.max_writes` | `int` | `3` | — |
| `background_review.peer_profile_max_chars` | `int` | `1500` | — |

### `rag`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `rag` | `RAGConfig` | `{RAGConfig}` | 検索拡張生成と記憶検索の設定。 |
| `rag.enabled` | `bool` | `true` | — |
| `rag.embedding_model` | `str` | `"intfloat/multilingual-e5-small"` | — |
| `rag.embedding_e5_prefix_enabled` | `bool` | `false` | When enabled, prefix query embeddings with embedding_query_prefix and indexed document embeddings with embedding_document_prefix. This is intended for E5-family ablations and requires re-indexing for document-prefix changes to take effect. |
| `rag.embedding_query_prefix` | `str` | `"query: "` | — |
| `rag.embedding_document_prefix` | `str` | `"passage: "` | — |
| `rag.embedding_max_seq_length` | `int` | `2048` | Cap on the embedding model's max sequence length (tokens). Long-context models like ruri-v3 default to 8192, which blows up GPU activation memory during bulk encode. 0 = use model default. |
| `rag.use_gpu` | `bool` | `false` | — |
| `rag.min_retrieval_score` | `float` | `0.3` | — |
| `rag.skill_match_min_score` | `float` | `0.75` | — |
| `rag.repair_enabled` | `bool` | `true` | — |
| `rag.repair_error_threshold` | `int` | `2` | — |
| `rag.repair_window_minutes` | `int` | `5` | — |
| `rag.repair_cooldown_minutes` | `int` | `60` | — |
| `rag.repair_max_consecutive_failures` | `int` | `2` | — |
| `rag.repair_timeout_seconds` | `int` | `1800` | — |
| `rag.repair_poll_interval_seconds` | `int` | `5` | — |
| `rag.repair_max_concurrent` | `int` | `1` | — |
| `rag.upsert_quarantine_failure_threshold` | `int` | `3` | — |
| `rag.shared_check_ttl_seconds` | `float` | `30.0` | — |
| `rag.shared_check_backoff_initial_seconds` | `float` | `5.0` | — |
| `rag.shared_check_backoff_max_seconds` | `float` | `300.0` | — |
| `rag.rerank_enabled` | `bool` | `true` | — |
| `rag.rerank_candidate_pool` | `int` | `50` | — |
| `rag.cross_encoder_model` | `str` | `"cross-encoder/ms-marco-MiniLM-L-12-v2"` | — |
| `rag.confidence_threshold` | `float` | `0.35` | — |
| `rag.rrf_confidence_threshold` | `float` | `0.02` | — |
| `rag.facts_extraction_enabled` | `bool` | `true` | — |
| `rag.fact_extraction_timeout_seconds` | `int` | `120` | Default timeout in seconds for LLM calls used by legacy atomic fact extraction; per-Anima status.json extraction_timeout overrides this value. |
| `rag.fact_extraction_max_tokens` | `int` | `8192` | Maximum output tokens for legacy atomic fact extraction LLM calls. |
| `rag.facts_extraction_single_call` | `bool` | `true` | Extract legacy atomic facts and their entities in one LLM call when supported. |
| `rag.facts_reconcile_enabled` | `bool` | `true` | Enable legacy atomic fact reconciliation before append; failures fall back to ADD. |
| `rag.facts_reconcile_similarity_threshold` | `float` | `0.82` | Minimum facts vector similarity before strict LLM duplicate/contradiction/complement labeling. |
| `rag.facts_reconcile_top_k` | `int` | `5` | Maximum similar active facts considered during legacy fact reconciliation. |
| `rag.entity_registry_enabled` | `bool` | `true` | — |

### `gpu`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `gpu` | `GPUConfig` | `{GPUConfig}` | GPU 利用とデバイス選択の設定。 |
| `gpu.embedding_device` | `Literal['auto', 'cuda', 'cpu']` | `"auto"` | — |
| `gpu.reranker_device` | `Literal['auto', 'cuda', 'cpu']` | `"cpu"` | — |
| `gpu.embedding_batch_size` | `int` | `32` | — |
| `gpu.embedding_bulk_yield_batches` | `int` | `5` | Maximum consecutive embedding batches bulk work may yield to waiting interactive work. |

### `memory`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `memory` | `MemoryConfig` | `{MemoryConfig}` | 記憶保存・検索の共通設定。 |
| `memory.fact_edge_types` | `list[FactEdgeTypeConfig]` | `[]` | — |
| `memory.fact_edge_types.name` | `str` | `"—"` | Upper snake case semantic edge type name |
| `memory.fact_edge_types.description` | `str` | `"—"` | Short explanation shown in extraction prompts |

### `skills`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `skills` | `SkillsConfig` | `{SkillsConfig}` | スキル読み込みと管理の設定。 |
| `skills.promotion` | `SkillPromotionConfig` | `{SkillPromotionConfig}` | — |
| `skills.promotion.success_count_threshold` | `int` | `3` | — |
| `skills.promotion.confidence_threshold` | `float` | `0.8` | — |
| `skills.promotion.failure_count_max` | `int` | `1` | — |
| `skills.promotion.last_used_within_days` | `int` | `180` | — |
| `skills.cron` | `SkillCronConfig` | `{SkillCronConfig}` | — |
| `skills.cron.max_skill_chars` | `int` | `6000` | — |
| `skills.cron.max_total_chars` | `int` | `12000` | — |
| `skills.cron.allow_warn_caution` | `bool` | `false` | — |
| `skills.cron.allow_destructive` | `bool` | `false` | — |
| `skills.cron.allow_external_send` | `bool` | `false` | — |
| `skills.external_roots` | `list[ExternalSkillRoot]` | `…` | — |
| `skills.external_roots.path` | `str` | `"—"` | may contain ``~``; expanded before use |
| `skills.external_roots.engine` | `str` | `"—"` | matches ^[a-z][a-z0-9-]*$ |
| `skills.external_roots.trust_level` | `str` | `"trusted"` | — |
| `skills.external_roots.enabled` | `bool` | `true` | — |

### `chatwork_tool`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `chatwork_tool` | `ChatworkToolConfig` | `{ChatworkToolConfig}` | Chatwork ツールの権限設定。 |
| `chatwork_tool.grants` | `dict[str, dict[str, str]]` | `{}` | — |

### `prompt`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `prompt` | `PromptConfig` | `{PromptConfig}` | システムプロンプトとプロンプト構築の設定。 |
| `prompt.injection_size_warning_chars` | `int` | `2000` | — |
| `prompt.identity_business_exclude_headings` | `list[str]` | `["外見","基本プロフィール","Appearance","Basic Profile"]` | — |
| `prompt.system_prompt_target_tokens` | `int` | `6000` | — |
| `prompt.system_prompt_ceiling_pct` | `float` | `0.35` | — |
| `prompt.skill_catalog_router_enabled` | `bool` | `true` | — |
| `prompt.skill_catalog_router_top_k` | `int` | `5` | — |
| `prompt.skill_catalog_router_min_score` | `float` | `1.15` | — |
| `prompt.skill_catalog_router_include_body` | `bool` | `true` | — |
| `prompt.skill_catalog_router_dense_enabled` | `bool` | `true` | — |
| `prompt.skill_catalog_router_dense_weight` | `float` | `8.0` | — |
| `prompt.skill_catalog_max_items` | `int` | `3` | — |

### `priming`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `priming` | `PrimingConfig` | `{PrimingConfig}` | anima の起動時に読み込む情報の設定。 |
| `priming.max_tokens` | `int` | `2000` | — |
| `priming.channel_timeout_seconds` | `float` | `60.0` | — |
| `priming.recent_facts_enabled` | `bool` | `true` | — |
| `priming.recent_facts_max_tokens` | `int` | `500` | — |
| `priming.compact_background_recall_enabled` | `bool` | `true` | — |
| `priming.compact_background_recall` | `CompactBackgroundRecallConfig` | `{CompactBackgroundRecallConfig}` | — |
| `priming.compact_background_recall.related_knowledge_max_items` | `int` | `3` | — |
| `priming.compact_background_recall.related_knowledge_max_tokens` | `int` | `180` | — |
| `priming.compact_background_recall.episodes_max_items` | `int` | `2` | — |
| `priming.compact_background_recall.episodes_max_tokens` | `int` | `400` | — |
| `priming.compact_background_recall.recent_activity_max_items` | `int` | `5` | — |
| `priming.compact_background_recall.recent_activity_max_tokens` | `int` | `300` | — |

### `image_gen`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `image_gen` | `ImageGenConfig` | `{ImageGenConfig}` | 画像生成プロバイダーと既定パラメーター。 |
| `image_gen.backend` | `Literal['api', 'diffusers', 'atlascloud']` | `"api"` | — |
| `image_gen.image_style` | `Literal['anime', 'realistic']` | `"realistic"` | — |
| `image_gen.prefer_codex` | `bool` | `true` | codex CLIがあれば画像生成に最優先で使う |
| `image_gen.style_reference` | `str \| None` | `null` | Path to organization-wide style reference image |
| `image_gen.style_prefix` | `str` | `""` | Common style tags prepended to character prompt |
| `image_gen.style_suffix` | `str` | `""` | Common style tags appended to character prompt |
| `image_gen.negative_prompt_extra` | `str` | `""` | Extra tags added to negative prompt |
| `image_gen.vibe_strength` | `float` | `0.6` | Vibe Transfer strength (0.0-1.0) |
| `image_gen.vibe_info_extracted` | `float` | `0.8` | Vibe Transfer information extraction (0.0-1.0) |
| `image_gen.enable_3d` | `bool` | `true` | Enable 3D model generation (Meshy API) |
| `image_gen.pixel_base_sheet` | `str \| None` | `null` | Base pixel sheet; None uses the bundled sample_01.png |
| `image_gen.pixel_concurrency` | `int` | `4` | Parallel codex image_gen calls for pixel sheet rows |
| `image_gen.diffusers_text2img_model` | `str` | `"auto"` | — |
| `image_gen.diffusers_img2img_model` | `str` | `"auto"` | — |
| `image_gen.diffusers_text2img_model_realistic` | `str` | `""` | Override for realistic style |
| `image_gen.diffusers_text2img_model_anime` | `str` | `""` | Override for anime style |
| `image_gen.diffusers_device` | `Literal['auto', 'cuda', 'cpu']` | `"auto"` | — |
| `image_gen.diffusers_torch_dtype` | `Literal['auto', 'float16', 'float32', 'bfloat16']` | `"auto"` | — |
| `image_gen.diffusers_local_files_only` | `bool` | `true` | — |
| `image_gen.diffusers_num_inference_steps` | `int` | `28` | — |
| `image_gen.diffusers_img2img_strength` | `float` | `0.55` | — |
| `image_gen.ip_adapter_model` | `str` | `"h94/IP-Adapter"` | — |
| `image_gen.ip_adapter_scale` | `float` | `0.6` | IP-Adapter face reference blend weight (0.0-1.0) |

### `human_notification`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `human_notification` | `HumanNotificationConfig` | `{HumanNotificationConfig}` | 人間への通知方法と通知先。 |
| `human_notification.enabled` | `bool` | `false` | — |
| `human_notification.channels` | `list[NotificationChannelConfig]` | `[]` | — |
| `human_notification.channels.type` | `str` | `"—"` | "slack", "line", "telegram", "chatwork", "ntfy" |
| `human_notification.channels.enabled` | `bool` | `true` | — |
| `human_notification.channels.config` | `dict[str, Any]` | `{}` | — |
| `human_notification.web_ui` | `bool` | `true` | built-in channel: push call_human into the Web UI chat |

### `interaction`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `interaction` | `InteractionConfig` | `{InteractionConfig}` | anima 間の対話とメッセージ処理。 |
| `interaction.default_approver_ids` | `list[str]` | `[]` | Default Slack user IDs merged with per-call call_human allowed_users. |
| `interaction.web_base_url` | `str` | `""` | — |

### `server`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `server` | `ServerConfig` | `{ServerConfig}` | HTTP サーバー、認証、利用量制御の設定。 |
| `server.session_ttl_days` | `int \| None` | `90` | None = unlimited |
| `server.ipc_stream_timeout` | `int` | `60` | per-chunk timeout in seconds |
| `server.keepalive_interval` | `int` | `30` | keep-alive emission interval in seconds |
| `server.runner_liveness_timeout` | `int` | `900` | — |
| `server.chat_max_concurrency` | `int` | `1` | — |
| `server.anima_startup_ready_timeout` | `int` | `120` | — |
| `server.anima_stop_timeout` | `float` | `60.0` | — |
| `server.restart_drain_timeout_seconds` | `int` | `600` | — |
| `server.health_check_warmup_seconds` | `int` | `300` | — |
| `server.runner_warmup_seconds` | `int` | `180` | — |
| `server.spawn_timeout` | `int` | `300` | — |
| `server.supervisor_respawn_max_retries` | `int` | `3` | — |
| `server.supervisor_respawn_retry_interval_seconds` | `float` | `30.0` | base backoff interval (seconds) |
| `server.supervisor_respawn_backoff_max_seconds` | `float` | `1800.0` | max backoff (seconds) |
| `server.stream_checkpoint_enabled` | `bool` | `true` | save tool results during streaming |
| `server.stream_retry_max` | `int` | `3` | max automatic retries on stream disconnect |
| `server.stream_retry_delay_s` | `float` | `5.0` | delay between retries (seconds) |
| `server.llm_num_retries` | `int` | `3` | retries for LLM API calls (429/5xx/network) |
| `server.ollama_keep_alive` | `str` | `""` | — |
| `server.ollama_total_timeout` | `int` | `0` | Hard upper bound (seconds) on a single Ollama generation call; 0 = unlimited |
| `server.media_proxy` | `MediaProxyConfig` | `{MediaProxyConfig}` | — |
| `server.media_proxy.mode` | `Literal['allowlist', 'open_with_scan']` | `"open_with_scan"` | — |
| `server.media_proxy.allowed_domains` | `list[str]` | `["cdn.search.brave.com","images.unsplash.com","images.pexels.com","upload.wikimedia.org"]` | — |
| `server.media_proxy.max_bytes` | `int` | `5242880` | — |
| `server.media_proxy.max_redirects` | `int` | `3` | — |
| `server.media_proxy.timeout_connect_s` | `float` | `5.0` | — |
| `server.media_proxy.timeout_read_s` | `float` | `10.0` | — |
| `server.media_proxy.rate_limit_requests` | `int` | `30` | — |
| `server.media_proxy.rate_limit_window_s` | `int` | `60` | — |
| `server.base_path` | `str` | `""` | Reverse proxy sub-path (e.g. "/app"); empty = root deploy |
| `server.internal_api_auth` | `Literal['off', 'log', 'enforce']` | `"log"` | /api/internal/* caller verification |

### `llm_rate_guard`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `llm_rate_guard` | `LlmRateGuardConfig` | `{LlmRateGuardConfig}` | LLM 呼び出し頻度と同時実行数の制御。 |
| `llm_rate_guard.enabled` | `bool` | `true` | — |
| `llm_rate_guard.default_block_seconds` | `int` | `60` | — |
| `llm_rate_guard.max_block_seconds` | `int` | `600` | — |
| `llm_rate_guard.quota_block_seconds` | `int` | `1800` | — |
| `llm_rate_guard.max_quota_block_seconds` | `int` | `14400` | — |

### `mcp`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `mcp` | `MCPConfig` | `{MCPConfig}` | MCP サーバー接続とツール公開の設定。 |
| `mcp.trigger_scoped_tools` | `bool` | `true` | — |

### `external_messaging`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `external_messaging` | `ExternalMessagingConfig` | `{ExternalMessagingConfig}` | Slack など外部メッセージング連携。 |
| `external_messaging.preferred_channel` | `str` | `"slack"` | "slack" \| "chatwork" \| "discord" |
| `external_messaging.user_aliases` | `dict[str, UserAliasConfig]` | `{}` | alias → contact info |
| `external_messaging.user_aliases.slack_user_id` | `str` | `""` | — |
| `external_messaging.user_aliases.chatwork_room_id` | `str` | `""` | — |
| `external_messaging.user_aliases.discord_user_id` | `str` | `""` | — |
| `external_messaging.user_aliases.outbound_dm` | `bool` | `false` | — |
| `external_messaging.chat_dm_redirect` | `bool` | `false` | — |
| `external_messaging.slack` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.slack.enabled` | `bool` | `false` | — |
| `external_messaging.slack.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.slack.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = ignore this channel) |
| `external_messaging.slack.default_anima` | `str` | `""` | fallback anima for unmapped channels |
| `external_messaging.slack.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (per-Anima webhook routing) |
| `external_messaging.slack.auto_response` | `bool` | `false` | auto-post LLM responses back to originating platform |
| `external_messaging.slack.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (auto-populated) |
| `external_messaging.slack.board_outbound_sync` | `list[str]` | `[]` | board names to sync outbound to this platform (whitelist) |
| `external_messaging.slack.board_outbound_sync_all` | `bool` | `false` | when whitelist is empty, opt in to syncing all mapped boards |
| `external_messaging.slack.guild_id` | `str` | `""` | Discord guild snowflake ID (Discord only) |
| `external_messaging.slack.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord only) |
| `external_messaging.slack.default_channel_company` | `str` | `""` | company for auto-created boards (empty = no attribution) |
| `external_messaging.chatwork` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.chatwork.enabled` | `bool` | `false` | — |
| `external_messaging.chatwork.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.chatwork.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = ignore this channel) |
| `external_messaging.chatwork.default_anima` | `str` | `""` | fallback anima for unmapped channels |
| `external_messaging.chatwork.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (per-Anima webhook routing) |
| `external_messaging.chatwork.auto_response` | `bool` | `false` | auto-post LLM responses back to originating platform |
| `external_messaging.chatwork.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (auto-populated) |
| `external_messaging.chatwork.board_outbound_sync` | `list[str]` | `[]` | board names to sync outbound to this platform (whitelist) |
| `external_messaging.chatwork.board_outbound_sync_all` | `bool` | `false` | when whitelist is empty, opt in to syncing all mapped boards |
| `external_messaging.chatwork.guild_id` | `str` | `""` | Discord guild snowflake ID (Discord only) |
| `external_messaging.chatwork.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord only) |
| `external_messaging.chatwork.default_channel_company` | `str` | `""` | company for auto-created boards (empty = no attribution) |
| `external_messaging.discord` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.discord.enabled` | `bool` | `false` | — |
| `external_messaging.discord.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.discord.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = ignore this channel) |
| `external_messaging.discord.default_anima` | `str` | `""` | fallback anima for unmapped channels |
| `external_messaging.discord.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (per-Anima webhook routing) |
| `external_messaging.discord.auto_response` | `bool` | `false` | auto-post LLM responses back to originating platform |
| `external_messaging.discord.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (auto-populated) |
| `external_messaging.discord.board_outbound_sync` | `list[str]` | `[]` | board names to sync outbound to this platform (whitelist) |
| `external_messaging.discord.board_outbound_sync_all` | `bool` | `false` | when whitelist is empty, opt in to syncing all mapped boards |
| `external_messaging.discord.guild_id` | `str` | `""` | Discord guild snowflake ID (Discord only) |
| `external_messaging.discord.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord only) |
| `external_messaging.discord.default_channel_company` | `str` | `""` | company for auto-created boards (empty = no attribution) |
| `external_messaging.zoom` | `ZoomRTMSConfig` | `{ZoomRTMSConfig}` | — |
| `external_messaging.zoom.enabled` | `bool` | `false` | — |
| `external_messaging.zoom.default_anima` | `str` | `""` | fallback anima for unmapped meetings |
| `external_messaging.zoom.meeting_mapping` | `dict[str, str]` | `{}` | meeting_id → anima_name |
| `external_messaging.zoom.chunk_interval_seconds` | `int` | `300` | flush interval for buffered transcript |
| `external_messaging.zoom.chunk_max_chars` | `int` | `4000` | max chars per chunk (flush on whichever comes first) |

### `external_tasks`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `external_tasks` | `ExternalTasksConfig` | `{ExternalTasksConfig}` | 外部タスクシステムとの連携。 |
| `external_tasks.enabled` | `bool` | `true` | — |
| `external_tasks.interval_minutes` | `int` | `5` | — |
| `external_tasks.sources` | `ExternalTasksSourcesConfig` | `{ExternalTasksSourcesConfig}` | — |
| `external_tasks.sources.github` | `bool` | `true` | — |
| `external_tasks.sources.slack` | `bool` | `true` | — |
| `external_tasks.sources.chatwork` | `bool` | `true` | — |
| `external_tasks.sources.gmail` | `bool` | `true` | — |

### `github_webhook`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `github_webhook` | `GitHubWebhookConfig` | `{GitHubWebhookConfig}` | GitHub webhook の受信と検証。 |
| `github_webhook.enabled` | `bool` | `false` | — |
| `github_webhook.repos` | `list[str]` | `[]` | — |
| `github_webhook.dispatcher_anima` | `str` | `"rin"` | — |
| `github_webhook.bot_login` | `str` | `""` | — |
| `github_webhook.reviewer_login` | `str` | `""` | — |
| `github_webhook.quiet_seconds` | `float` | `180` | — |
| `github_webhook.drop_bot_noise` | `bool` | `true` | — |

### `event_export`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `event_export` | `EventExportConfig` | `{EventExportConfig}` | イベントデータのエクスポート。 |
| `event_export.url` | `str \| None` | `null` | — |
| `event_export.headers` | `dict[str, str]` | `{}` | — |
| `event_export.event_types` | `list[str] \| None` | `null` | — |
| `event_export.include_token_usage` | `bool` | `true` | — |
| `event_export.max_retries` | `int` | `8` | — |
| `event_export.backoff_base_seconds` | `float` | `2.0` | — |
| `event_export.spool_max_mb` | `int` | `64` | — |

### `background_task`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `background_task` | `BackgroundTaskConfig` | `{BackgroundTaskConfig}` | バックグラウンド実行対象ツールの設定。 |
| `background_task.enabled` | `bool` | `true` | — |
| `background_task.shutdown_drain_seconds` | `float` | `600.0` | — |
| `background_task.eligible_tools` | `dict[str, BackgroundToolConfig]` | `…` | — |
| `background_task.eligible_tools.threshold_s` | `int` | `30` | — |
| `background_task.result_memory_retention_minutes` | `int` | `60` | in-process result cache |
| `background_task.max_completed_tasks_in_memory` | `int` | `200` | — |
| `background_task.worker_pool_size` | `int` | `1` | — |

### `activity_log`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `activity_log` | `ActivityLogConfig` | `{ActivityLogConfig}` | 操作・活動ログの保存設定。 |
| `activity_log.rotation_enabled` | `bool` | `true` | — |
| `activity_log.rotation_mode` | `Literal['size', 'time', 'both']` | `"size"` | — |
| `activity_log.max_size_mb` | `int` | `1024` | per-anima total, default 1GB |
| `activity_log.max_file_size_mb` | `int` | `100` | per-file bloat trigger; 0 disables |
| `activity_log.max_age_days` | `int` | `7` | mode="time"\|"both" で使用 |
| `activity_log.rotation_time` | `str` | `"05:00"` | 実行時刻 (configured TZ) |

### `logging`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `logging` | `LoggingConfig` | `{LoggingConfig}` | ログレベル、出力先、機密情報のマスキング。 |
| `logging.redaction_enabled` | `bool` | `true` | Mask secrets in log output; disable for raw-log debugging. |

### `heartbeat`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `heartbeat` | `HeartbeatConfig` | `{HeartbeatConfig}` | 定期的な heartbeat の実行設定。 |
| `heartbeat.interval_minutes` | `int` | `30` | — |
| `heartbeat.current_state_max_chars` | `int` | `8000` | Max chars for current_state.md before trim; 0 = disabled |
| `heartbeat.current_state_cleanup_chars` | `int` | `2000` | Soft cleanup threshold for current_state.md; 0 = 80% of current_state_max_chars |
| `heartbeat.heartbeat_md_max_bytes` | `int` | `8000` | Max bytes of heartbeat.md before a compaction instruction is injected into the heartbeat prompt; 0 = disabled |
| `heartbeat.recent_dialogue_max_age_hours` | `int` | `6` | Include recent chat dialogue in heartbeat context only when the last turn is younger than this many hours; 0 = always include |
| `heartbeat.soft_timeout_seconds` | `int` | `300` | Seconds before injecting a wrap-up system-reminder into the HB session |
| `heartbeat.hard_timeout_seconds` | `int` | `0` | Seconds before forcefully terminating the HB session; 0 = disabled |
| `heartbeat.cron_followup_repeat_suppress_hours` | `float` | `6.0` | Suppress a command cron's LLM follow-up when its output is identical to the previous follow-up for this many hours; 0 disables suppression |
| `heartbeat.default_model` | `str \| None` | `null` | global background model for heartbeat/cron (None = use main model) |
| `heartbeat.enable_read_ack` | `bool` | `false` | — |
| `heartbeat.delegation_dm_enabled` | `bool` | `true` | delegate_task already writes the pending descriptor for the target; the DM only wakes an extra inbox run. Set false to skip it. |
| `heartbeat.idle_compaction_minutes` | `float` | `10.0` | Minutes after last stream end to trigger idle auto-compaction |
| `heartbeat.resolved_interaction_reminder_hours` | `int` | `48` | Hours to inject resolved-approval reminders into the system prompt; 0 disables the reminder section |

### `voice`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `voice` | `VoiceConfig` | `{VoiceConfig}` | 音声入出力と音声プロバイダー。 |
| `voice.stt_model` | `str` | `"large-v3-turbo"` | — |
| `voice.stt_device` | `str` | `"auto"` | — |
| `voice.stt_compute_type` | `str` | `"default"` | — |
| `voice.stt_language` | `str \| None` | `null` | — |
| `voice.stt_refine_enabled` | `bool` | `false` | — |
| `voice.default_tts_provider` | `str` | `"voicevox"` | — |
| `voice.front_model` | `str \| None` | `null` | — |
| `voice.front_api_base` | `str \| None` | `null` | — |
| `voice.proactive_enabled` | `bool` | `true` | — |
| `voice.proactive_initial_delay_sec` | `float` | `10.0` | — |
| `voice.proactive_lead_sec` | `float` | `5.0` | — |
| `voice.notify_delegations_on_web_disconnect` | `bool` | `false` | — |
| `voice.voicevox` | `VoicevoxConfig` | `{VoicevoxConfig}` | — |
| `voice.voicevox.base_url` | `str` | `"http://localhost:50021"` | — |
| `voice.elevenlabs` | `ElevenLabsVoiceConfig` | `{ElevenLabsVoiceConfig}` | — |
| `voice.elevenlabs.api_key_env` | `str` | `"ELEVENLABS_API_KEY"` | — |
| `voice.elevenlabs.model_id` | `str` | `"eleven_flash_v2_5"` | — |
| `voice.style_bert_vits2` | `StyleBertVits2Config` | `{StyleBertVits2Config}` | — |
| `voice.style_bert_vits2.base_url` | `str` | `"http://localhost:5000"` | — |
| `voice.irodori` | `IrodoriConfig` | `{IrodoriConfig}` | — |
| `voice.irodori.base_url` | `str` | `"http://localhost:7861"` | — |
| `voice.gemini` | `GeminiTTSVoiceConfig` | `{GeminiTTSVoiceConfig}` | — |
| `voice.gemini.model` | `str` | `"gemini-3.8-flash-tts"` | — |
| `voice.gemini.api_key_env` | `str` | `"GEMINI_API_KEY"` | — |
| `voice.gemini.vault_key` | `str` | `"GEMINI_API_KEY"` | — |
| `voice.gemini.chunk_seconds` | `float` | `1.0` | — |

### `phone`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `phone` | `PhoneConfig` | `{PhoneConfig}` | Twilio を利用した Anima 専用の電話会話と緊急通知。 |
| `phone.enabled` | `bool` | `false` | — |
| `phone.anima` | `str` | `"aoi"` | — |
| `phone.public_base_url` | `str` | `"https://zoomhook.kk-a.jp"` | — |
| `phone.from_number` | `str` | `"+16073257655"` | — |
| `phone.owner_numbers` | `list[str]` | `["+819060943763"]` | — |
| `phone.from_person` | `str` | `"human"` | — |
| `phone.thread_id` | `str` | `"phone"` | — |
| `phone.pin_vault_key` | `str` | `"AOI_PHONE_PIN"` | — |
| `phone.account_sid_vault_key` | `str` | `"TWILIO_ACCOUNT_SID"` | — |
| `phone.auth_token_vault_key` | `str` | `"TWILIO_AUTH_TOKEN"` | — |
| `phone.alert_max_attempts` | `int` | `3` | — |
| `phone.alert_retry_interval_sec` | `float` | `120` | — |
| `phone.turn_timeout_sec` | `float` | `300` | Legacy Gather/poll timeout; unused by the Media Streams phone conversation. |
| `phone.turn_end_silence_ms` | `int` | `1200` | — |

### `housekeeping`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `housekeeping` | `HousekeepingConfig` | `{HousekeepingConfig}` | 実行時データの定期整理。 |
| `housekeeping.enabled` | `bool` | `true` | — |
| `housekeeping.run_time` | `str` | `"05:30"` | — |
| `housekeeping.prompt_log_retention_days` | `int` | `3` | — |
| `housekeeping.daemon_log_max_size_mb` | `int` | `50` | — |
| `housekeeping.daemon_log_keep_generations` | `int` | `5` | — |
| `housekeeping.anima_log_retention_days` | `int` | `30` | — |
| `housekeeping.anima_log_total_max_size_mb` | `int` | `200` | — |
| `housekeeping.frontend_log_backup_count` | `int` | `7` | — |
| `housekeeping.dm_log_archive_retention_days` | `int` | `30` | — |
| `housekeeping.cron_log_retention_days` | `int` | `14` | — |
| `housekeeping.shortterm_retention_days` | `int` | `7` | — |
| `housekeeping.shortterm_archive_retention_days` | `int` | `30` | — |
| `housekeeping.shortterm_thread_gc_days` | `int` | `30` | — |
| `housekeeping.facts_lock_stale_hours` | `int` | `24` | — |
| `housekeeping.curator_report_retention_days` | `int` | `30` | — |
| `housekeeping.task_results_retention_days` | `int` | `7` | — |
| `housekeeping.corrupt_vectordb_keep_generations` | `int` | `2` | — |
| `housekeeping.tmp_retention_days` | `int` | `14` | — |
| `housekeeping.backup_retention_days` | `int` | `90` | — |
| `housekeeping.codex_log_max_size_mb` | `int` | `200` | — |
| `housekeeping.codex_tmp_retention_hours` | `int` | `12` | — |
| `housekeeping.codex_sessions_retention_days` | `int` | `7` | — |
| `housekeeping.anima_scratch_retention_hours` | `int` | `24` | — |
| `housekeeping.shared_worktrees_idle_hours` | `int` | `24` | — |
| `housekeeping.workspace_sweep_enabled` | `bool` | `true` | — |
| `housekeeping.anima_tmp_gitdirs_retention_days` | `int` | `14` | — |
| `housekeeping.anima_local_log_retention_days` | `int` | `30` | — |
| `housekeeping.pending_processing_stale_hours` | `int` | `24` | — |
| `housekeeping.background_running_stale_hours` | `int` | `48` | — |
| `housekeeping.current_state_stale_hours` | `int` | `24` | — |
| `housekeeping.suppressed_messages_max_size_mb` | `int` | `10` | — |
| `housekeeping.suppressed_messages_keep_generations` | `int` | `5` | — |
| `housekeeping.sdk_bash_injection_max_size_mb` | `int` | `10` | — |
| `housekeeping.archive_superseded_retention_days` | `int` | `7` | — |
| `housekeeping.archive_versions_keep_per_file` | `int` | `5` | — |

### `inbox`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `inbox` | `InboxConfig` | `{InboxConfig}` | anima の受信箱と通知表示。 |
| `inbox.inbox_quiet_seconds` | `float` | `90.0` | — |
| `inbox.inbox_quiet_max_wait_seconds` | `float` | `300.0` | — |
| `inbox.ttl_hours` | `float` | `24.0` | — |
| `inbox.expired_retention_days` | `int` | `7` | — |
| `inbox.processed_retention_days` | `int` | `30` | — |
| `inbox.quarantine_retention_days` | `int` | `30` | — |

### `local_llm`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `local_llm` | `LocalLLMConfig` | `{LocalLLMConfig}` | ローカル LLM エンドポイントとモデル設定。 |
| `local_llm.base_url` | `str` | `"http://127.0.0.1:11434"` | — |
| `local_llm.default_model` | `str` | `"ollama/qwen2.5-coder:14b"` | — |
| `local_llm.credential` | `str` | `""` | — |
| `local_llm.auto_apply_presets` | `bool` | `false` | — |
| `local_llm.presets` | `dict[str, str]` | `{"coding":"ollama/qwen2.5-coder:14b","general":"ollama/glm4:9b","reasoning":"ollama/deepseek-r1:8b"}` | — |
| `local_llm.role_presets` | `dict[str, str]` | `…` | — |

### `workspaces`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `workspaces` | `dict[str, str]` | `{}` | 名前付き workspace パスの対応。 |

### `github_identities`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `github_identities` | `dict[str, str]` | `{}` | 会社 slug と GitHub アカウントの対応。 |

### `activity_level`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `activity_level` | `int` | `100` | 全体の活動頻度を調整する倍率。 |

### `activity_schedule`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `activity_schedule` | `list[ActivityScheduleEntry]` | `[]` | 時刻帯ごとの活動頻度。 |
| `activity_schedule.start` | `str` | `"—"` | Start time in HH:MM format |
| `activity_schedule.end` | `str` | `"—"` | End time in HH:MM format (may wrap past midnight) |
| `activity_schedule.level` | `int` | `"—"` | Activity level percentage for this period |

### `icon_url_template`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `icon_url_template` | `str` | `""` | anima アイコン URL のテンプレート。 |

### `ui`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `ui` | `UIConfig` | `{UIConfig}` | Web UI の表示設定。 |
| `ui.theme` | `str` | `"default"` | — |
| `ui.demo_mode` | `bool` | `false` | — |

### `cli`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `cli` | `CLIConfig` | `{CLIConfig}` | CLI クライアントの設定。default_anima は引数なしの animaworks / animaworks chat で開く anima（空なら上司なしの有効な anima が1体だけの時それを使う）。 |
| `cli.default_anima` | `str` | `""` | — |

### `enclave`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `enclave` | `EnclaveConfig` | `{EnclaveConfig}` | 隔離された enclave モードのサーバー設定。有効時は起動時にセキュリティガードを検査する。 |
| `enclave.enabled` | `bool` | `false` | この実行環境を隔離 enclave として起動する。 |
| `enclave.name` | `str` | `""` | enclave gateway が返す識別名。 |
| `enclave.socket_path` | `str` | `""` | gateway が待ち受ける Unix ドメインソケットのパス。 |
| `enclave.socket_group` | `str` | `""` | gateway ソケットに設定する OS グループ。 |
| `enclave.entry_anima` | `str` | `""` | enclave 内で質問を処理する entry anima。 |
| `enclave.allowed_peer_uids` | `list[int]` | `[]` | gateway への接続を許可する OS ユーザー UID。 |
| `enclave.max_concurrency` | `int` | `2` | gateway が同時に処理する質問の最大数。 |
| `enclave.request_timeout_s` | `int` | `0` | 質問処理と gateway 応答のタイムアウト秒数。 |
| `enclave.allowed_llm_credentials` | `list[str]` | `[]` | enclave 内の anima が利用できる認証情報名。 |
| `enclave.model_profiles` | `dict[str, str]` | `{}` | — |
| `enclave.default_model_profile` | `str` | `""` | — |
| `enclave.datasets` | `dict[str, EnclaveDatasetConfig]` | `{}` | enclave 内の anima が参照できる JSONL データセット。 |
| `enclave.datasets.path` | `str` | `"—"` | data directory 内にある JSONL ファイルの相対パス。 |
| `enclave.datasets.id_field` | `str` | `"—"` | レコードを一意に取得するための ID フィールド。 |
| `enclave.datasets.searchable_fields` | `list[str]` | `[]` | 部分一致検索を許可するフィールド。 |
| `enclave.sql_sources` | `dict[str, EnclaveSqlSourceConfig]` | `{}` | — |
| `enclave.sql_sources.driver` | `Literal['mysql']` | `"mysql"` | — |
| `enclave.sql_sources.host` | `str` | `"—"` | Tunnel target (or direct) host |
| `enclave.sql_sources.port` | `int` | `3306` | — |
| `enclave.sql_sources.database` | `str` | `"—"` | — |
| `enclave.sql_sources.user` | `str` | `"—"` | — |
| `enclave.sql_sources.password_secret` | `str` | `"—"` | — |
| `enclave.sql_sources.ssl` | `bool` | `true` | — |
| `enclave.sql_sources.ssl_ca` | `str \| None` | `null` | Optional CA bundle path for TLS verification |
| `enclave.sql_sources.ssl_verify_identity` | `bool` | `false` | — |
| `enclave.sql_sources.tunnel` | `EnclaveSsmTunnelConfig \| None` | `null` | — |
| `enclave.sql_sources.tunnel.type` | `Literal['ssm_port_forward']` | `"ssm_port_forward"` | — |
| `enclave.sql_sources.tunnel.region` | `str` | `"—"` | AWS region for SSM and target resolution |
| `enclave.sql_sources.tunnel.target_instance_id` | `str \| None` | `null` | — |
| `enclave.sql_sources.tunnel.target_tag_name` | `str \| None` | `null` | — |
| `enclave.sql_sources.tunnel.aws_secret` | `str` | `"—"` | Secret name holding AWS credential JSON |
| `enclave.sql_sources.tunnel.plugin_path` | `str` | `"/usr/local/bin/session-manager-plugin"` | — |
| `enclave.sql_sources.tunnel.idle_shutdown_s` | `int` | `600` | — |
| `enclave.sql_sources.tunnel.local_port` | `int \| None` | `null` | Fixed loopback port for the tunnel (lets a host firewall allow it); random when unset |
| `enclave.sql_sources.tunnel.local_port_count` | `int` | `1` | Number of consecutive loopback ports starting at local_port; concurrent processes each reserve a different one |
| `enclave.sql_sources.max_rows` | `int` | `200` | — |
| `enclave.sql_sources.timeout_s` | `int` | `30` | — |
| `enclave.sql_sources.cell_max_chars` | `int` | `2000` | — |
| `enclave.sql_sources.app_key_secret` | `str \| None` | `null` | — |
| `enclave.sql_sources.decrypt_columns` | `list[str]` | `[]` | — |
| `enclave.sql_sources.session_variables` | `dict[str, str]` | `{}` | User variables set on every connection (``SET @name = 'value'``), e.g. for row-level views that filter on a session variable |
| `enclave.aws_sources` | `dict[str, EnclaveAwsSourceConfig]` | `{}` | — |
| `enclave.aws_sources.region` | `str` | `"—"` | — |
| `enclave.aws_sources.aws_secret` | `str` | `"—"` | Secret name holding AWS credential JSON |
| `enclave.aws_sources.log_groups` | `list[str]` | `[]` | — |
| `enclave.aws_sources.pi_resource_id` | `str \| None` | `null` | — |
| `enclave.aws_sources.rds_instance_id` | `str \| None` | `null` | — |
| `enclave.aws_sources.s3_buckets` | `list[str]` | `[]` | — |
| `enclave.aws_sources.max_bytes` | `int` | `200000` | — |
| `enclave.secrets_dir` | `str \| None` | `null` | — |
| `enclave.raw_dir` | `str` | `"raw"` | — |
| `enclave.egress` | `dict[str, Any]` | `{}` | 外部へ返す facts を処理する egress ステージ設定。 |
| `enclave.egress_probe` | `list[str]` | `[]` | — |
| `enclave.egress_probe_timeout_s` | `float` | `5.0` | — |

### `enclaves`

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `enclaves` | `dict[str, EnclaveClientConfig]` | `{}` | 外部の enclave インスタンスに接続するクライアント側の設定。 |
| `enclaves.socket_path` | `str` | `"—"` | 接続先 enclave の Unix ドメインソケット。 |
| `enclaves.allowed_animas` | `list[str]` | `[]` | その enclave への質問を許可する本体側 anima。 |
| `enclaves.timeout_s` | `int` | `900` | enclave gateway への接続タイムアウト秒数。 |

## anima ごとの `status.json`

| キー | ModelConfig 型 | 既定値 | 説明 |
|---|---|---|---|
| `background_credential` | `str \| None` | `null` | — |
| `background_model` | `str \| None` | `null` | heartbeat や cron で使うモデル上書き。 |
| `background_thinking_effort` | `str \| None` | `null` | — |
| `consolidation_enabled` | `—` | `—` | — |
| `context_absolute_ceiling` | `float` | `0.75` | — |
| `context_threshold` | `float` | `0.5` | — |
| `conversation_history_threshold` | `float` | `0.3` | — |
| `credential` | `str \| None` | `null` | anima に関連付ける認証情報名。 |
| `default_workspace` | `—` | `—` | — |
| `execution_mode` | `str \| None` | `null` | anima の実行モード。 |
| `extra_mcp_servers` | `dict[str, dict]` | `{}` | — |
| `fallback_model` | `str \| None` | `null` | 主モデル失敗時に使う代替モデル。 |
| `fallback_models` | `list[str]` | `[]` | — |
| `heartbeat_enabled` | `bool` | `true` | 定期 heartbeat の有効・無効。 |
| `max_session_age_hours` | `float` | `24.0` | — |
| `max_tokens` | `int` | `8192` | モデルの最大出力トークン数。 |
| `mode_s_auth` | `str \| None` | `null` | — |
| `model` | `str` | `"claude-sonnet-5-5"` | anima の主モデル。 |
| `speciality` | `str \| None` | `null` | — |
| `supervisor` | `str \| None` | `null` | 上位 supervisor anima の名前。 |
| `task_compaction_max` | `int` | `6` | — |
| `task_compaction_tokens` | `int` | `0` | — |
| `thinking` | `bool \| None` | `null` | — |
| `thinking_effort` | `str \| None` | `null` | — |
| `token_budget_monthly` | `int \| None` | `null` | — |
| `voice_thinking_effort` | `str \| None` | `null` | — |
| `bootstrap_state` | `—` | `—` | 初期化処理の状態。 |
| `company` | `—` | `—` | 所属する会社。 |
| `department` | `—` | `—` | 所属する部署。 |
| `enabled` | `—` | `—` | anima の有効・無効。 |
| `needs_user_input` | `—` | `—` | ユーザー入力が必要な状態かどうか。 |
| `role` | `—` | `—` | anima の役割テンプレート。 |

## `models.json`

モデル名パターンごとの実行モードとコンテキストウィンドウです。

| モデル名パターン | モード | コンテキストウィンドウ |
|---|---|---:|
| `agy/*` | `Y` | 1000000 |
| `anthropic/claude-*` | `A` | 200000 |
| `azure/*` | `A` | 128000 |
| `azure/gpt-4.1*` | `A` | 1000000 |
| `bedrock/*` | `A` | 200000 |
| `bedrock/meta.llama4*` | `A` | 1000000 |
| `bedrock/qwen.*` | `A` | 131072 |
| `claude-*` | `S` | 200000 |
| `claude-opus-4-6` | `S` | 200000 |
| `claude-opus-5-5` | `S` | 200000 |
| `claude-sonnet-4-6` | `S` | 200000 |
| `claude-sonnet-5-5` | `S` | 200000 |
| `codex/*` | `C` | 128000 |
| `codex/gpt-4.1` | `C` | 1000000 |
| `codex/gpt-5.4` | `C` | 272000 |
| `codex/o3` | `C` | 200000 |
| `codex/o4-mini` | `C` | 200000 |
| `cursor/*` | `D` | 1000000 |
| `deepseek/deepseek-chat` | `A` | 128000 |
| `gemini/*` | `G` | 1000000 |
| `google/*` | `A` | 1000000 |
| `grok/*` | `X` | 500000 |
| `grok/grok-4.5` | `X` | 500000 |
| `mistral/*` | `A` | 256000 |
| `nanogpt/*` | `A` | 128000 |
| `ollama/*` | `A` | 8192 |
| `ollama/gemma3*` | `A` | 8192 |
| `ollama/glm-4.7*` | `A` | 32768 |
| `ollama/glm-5*` | `A` | 32768 |
| `ollama/kimi-k2*` | `A` | 131072 |
| `ollama/llama4:*` | `A` | 131072 |
| `ollama/phi4*` | `A` | 16384 |
| `ollama/qwen3-coder:*` | `A` | 40960 |
| `ollama/qwen3:14b` | `A` | 40960 |
| `ollama/qwen3:30b` | `A` | 40960 |
| `ollama/qwen3:32b` | `A` | 40960 |
| `openai/*` | `A` | 128000 |
| `openai/deepseek-v4-flash` | `A` | 1048576 |
| `openai/deepseek-v4-flash-0731` | `A` | 1048576 |
| `openai/gpt-4.1*` | `A` | 1000000 |
| `openai/gpt-5.4*` | `A` | 272000 |
| `openai/qwen3.5*` | `A` | 64000 |
| `openai/qwen3.6-27b*` | `A` | 32768 |
| `openai/qwen3.6-35b*` | `A` | 32768 |
| `openai/zai.glm*` | `A` | 131072 |
| `vertex_ai/*` | `A` | 1000000 |
| `xai/*` | `A` | 256000 |
