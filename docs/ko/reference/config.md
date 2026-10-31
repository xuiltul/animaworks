<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/reference/config.md -->
<!-- i18n: source-sha256=544ced8ff2ba6f5174edc608d237ec9e3acd24a0488af75602ca3e769eb57110 generated=2026-10-31 engine=luna model=gpt-6-luna translator=2 -->

# 설정 참조

`AnimaWorksConfig`, per-anima `ModelConfig`, `models.json`의 정의에서 생성하고 있습니다.

## `config.json`

### `version`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `version` | `int` | `1` | 설정 파일 형식의 버전. |

### `setup_complete`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `setup_complete` | `bool` | `false` | 초기 설정이 완료되었는지 여부. |

### `locale`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `locale` | `str` | `"ja"` | 시스템 전체에서 사용할 기본 언어. |

### `system`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `system` | `SystemConfig` | `{SystemConfig}` | 실행 환경, 시간대, 기본 동작에 관한 설정. |
| `system.mode` | `str` | `"server"` | — |
| `system.timezone` | `str` | `""` | IANA TZ name; empty = auto-detect from system |

### `credentials`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `credentials` | `dict[str, CredentialConfig]` | `{"anthropic":{"model":"CredentialConfig"}}` | 외부 모델 서비스의 인증 정보. |
| `credentials.type` | `str` | `"api_key"` | — |
| `credentials.api_key` | `str` | `""` | — |
| `credentials.keys` | `dict[str, str]` | `{}` | — |
| `credentials.base_url` | `str \| None` | `null` | — |

### `model_modes`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `model_modes` | `dict[str, str]` | `{}` | 모델명 패턴과 실행 모드의 대응 관계. |

### `model_context_windows`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `model_context_windows` | `dict[str, int]` | `{}` | 모델별 컨텍스트 길이 호환 설정. models.json 사용을 권장. |

### `model_max_tokens`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `model_max_tokens` | `dict[str, int]` | `{}` | 모델명 패턴별 기본 출력 토큰 수. |

### `anima_defaults`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `anima_defaults` | `AnimaDefaults` | `{AnimaDefaults}` | 각 anima에 적용할 모델·실행 설정의 기본값. |
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
| `anima_defaults.execution_mode` | `str \| None` | `null` | None = 모델에서 자동 감지 |
| `anima_defaults.supervisor` | `str \| None` | `null` | — |
| `anima_defaults.speciality` | `str \| None` | `null` | — |
| `anima_defaults.extra_mcp_servers` | `dict[str, dict]` | `{}` | — |
| `anima_defaults.thinking` | `bool \| None` | `null` | 확장 사고 (Bedrock: reasoning_effort, Ollama: think) |
| `anima_defaults.thinking_effort` | `str \| None` | `null` | "low"/"medium"/"high"/"max" (기본값: "high") |
| `anima_defaults.mode_s_auth` | `str \| None` | `null` | Mode S 인증: "max"\|"api"\|"bedrock"\|"vertex"\|None(=max) |
| `anima_defaults.default_workspace` | `str` | `""` | — |
| `anima_defaults.consolidation_enabled` | `bool` | `true` | — |
| `anima_defaults.heartbeat_enabled` | `bool` | `true` | 기본 true. false로 정기 heartbeat만 비활성화. 메시지 기반 HB·cron은 영향 없음 |
| `anima_defaults.token_budget_monthly` | `int \| None` | `null` | None = 월간 토큰 사용량 무제한 |

### `animas`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `animas` | `dict[str, AnimaModelConfig]` | `{}` | Anima별 모델 설정 재정의. |
| `animas.supervisor` | `str \| None` | `null` | — |
| `animas.company` | `str \| None` | `null` | — |
| `animas.speciality` | `str \| None` | `null` | — |
| `animas.model` | `str \| None` | `null` | — |
| `animas.heartbeat_enabled` | `bool \| None` | `null` | — |
| `animas.background_review_enabled` | `bool \| None` | `null` | — |
| `animas.token_budget_monthly` | `int \| None` | `null` | — |
| `animas.aliases` | `list[str]` | `[]` | — |

### `consolidation`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `consolidation` | `ConsolidationConfig` | `{ConsolidationConfig}` | 기억 통합의 동작과 일정. |
| `consolidation.daily_enabled` | `bool` | `true` | — |
| `consolidation.weekly_distillation_enabled` | `bool` | `true` | — |
| `consolidation.synaptic_downscaling_enabled` | `bool` | `true` | — |
| `consolidation.skill_autolearn_enabled` | `bool` | `true` | — |
| `consolidation.curator_auto_apply_enabled` | `bool` | `false` | — |
| `consolidation.daily_time` | `str` | `"02:00"` | 형식: HH:MM |
| `consolidation.min_episodes_threshold` | `int` | `1` | — |
| `consolidation.llm_model` | `str` | `"claude-sonnet-5-5"` | — |
| `consolidation.llm_credential` | `str` | `""` | — |
| `consolidation.weekly_llm_model` | `str \| None` | `null` | — |
| `consolidation.weekly_llm_credential` | `str \| None` | `null` | — |
| `consolidation.llm_fallback_model` | `str \| None` | `null` | — |
| `consolidation.llm_fallback_credential` | `str \| None` | `null` | — |
| `consolidation.fact_reconcile_model` | `str \| None` | `null` | — |
| `consolidation.fact_reconcile_credential` | `str \| None` | `null` | — |
| `consolidation.episode_summary_max_input_bytes` | `int` | `204800` | 각 일일 에피소드 요약 LLM 호출의 최대 UTF-8 프롬프트 크기. |
| `consolidation.episode_summary_backfill_days` | `int` | `7` | 처리되지 않은 일일 에피소드 활동을 위해 이만큼의 로컬 일수를 되돌아봄. |
| `consolidation.episode_summary_backfill_max_days_per_run` | `int` | `1` | 한 번의 일일 통합 중 백필할 최대 이전 일수 (어제는 별도). |
| `consolidation.episode_summary_exclude_noop_cron` | `bool` | `true` | 일일 에피소드 요약 입력에서 '아무것도 안 함' cron 실행 제외. |
| `consolidation.ipc_timeout_base_seconds` | `int` | `1800` | — |
| `consolidation.ipc_timeout_per_activity_entry_seconds` | `float` | `4.0` | — |
| `consolidation.ipc_timeout_per_episode_seconds` | `float` | `120.0` | — |
| `consolidation.ipc_timeout_max_seconds` | `int` | `7200` | — |
| `consolidation.weekly_ipc_timeout_seconds` | `int` | `3600` | — |
| `consolidation.max_concurrent_animas` | `int` | `3` | 동시에 실행할 최대 Anima daily/weekly 통합 수. |
| `consolidation.weekly_enabled` | `bool` | `false` | — |
| `consolidation.weekly_time` | `str` | `"sun:03:00"` | 형식: day:HH:MM |
| `consolidation.indexing_enabled` | `bool` | `true` | 일일 RAG 인덱싱 토글 |
| `consolidation.indexing_time` | `str` | `"04:00"` | 형식: HH:MM |
| `consolidation.knowledge_self_correction_enabled` | `bool` | `true` | — |
| `consolidation.knowledge_self_correction_max_reconsolidation_files` | `int` | `5` | — |
| `consolidation.knowledge_self_correction_timeout_seconds` | `int` | `300` | — |
| `consolidation.fact_extraction_chunk_chars` | `int` | `12000` | 통합 중 원자적 사실 추출 청크당 최대 문자 수. 0(또는 음수)은 분할을 비활성화. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `helper_models` | `HelperModelsConfig` | `{HelperModelsConfig}` | 보조 LLM 호출별 모델, 인증 정보, 명시적 대체 후보 및 원샷 정책. |
| `helper_models.episode_summary` | `HelperModelRole \| None` | `null` | — |
| `helper_models.episode_summary.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.episode_summary.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.episode_summary.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.episode_summary.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.episode_summary.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.episode_summary.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.episode_summary.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.fact_extraction` | `HelperModelRole \| None` | `null` | — |
| `helper_models.fact_extraction.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.fact_extraction.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.fact_extraction.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.fact_extraction.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.fact_extraction.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.fact_extraction.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.fact_extraction.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.fact_reconcile` | `HelperModelRole \| None` | `null` | — |
| `helper_models.fact_reconcile.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.fact_reconcile.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.fact_reconcile.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.fact_reconcile.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.fact_reconcile.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.fact_reconcile.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.fact_reconcile.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.weekly_consolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.weekly_consolidation.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.weekly_consolidation.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.weekly_consolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.weekly_consolidation.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.weekly_consolidation.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.weekly_consolidation.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.weekly_consolidation.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.project_consolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.project_consolidation.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.project_consolidation.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.project_consolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.project_consolidation.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.project_consolidation.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.project_consolidation.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.project_consolidation.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.conversation_compression` | `HelperModelRole \| None` | `null` | — |
| `helper_models.conversation_compression.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.conversation_compression.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.conversation_compression.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.conversation_compression.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.conversation_compression.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.conversation_compression.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.conversation_compression.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.distillation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.distillation.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.distillation.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.distillation.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.distillation.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.distillation.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.distillation.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.distillation.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.reconsolidation` | `HelperModelRole \| None` | `null` | — |
| `helper_models.reconsolidation.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.reconsolidation.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.reconsolidation.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.reconsolidation.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.reconsolidation.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.reconsolidation.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.reconsolidation.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.asset_reconcile` | `HelperModelRole \| None` | `null` | — |
| `helper_models.asset_reconcile.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.asset_reconcile.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.asset_reconcile.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.asset_reconcile.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.asset_reconcile.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.asset_reconcile.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.asset_reconcile.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.meeting_summary` | `HelperModelRole \| None` | `null` | — |
| `helper_models.meeting_summary.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.meeting_summary.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.meeting_summary.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.meeting_summary.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.meeting_summary.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.meeting_summary.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.meeting_summary.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |
| `helper_models.default` | `HelperModelRole` | `{HelperModelRole}` | — |
| `helper_models.default.model` | `str \| None` | `null` | 기본 보조 모델 식별자. |
| `helper_models.default.credential` | `str \| None` | `null` | 기본 모델의 인증 정보 이름. |
| `helper_models.default.fallbacks` | `list[HelperModelFallback]` | `[]` | 순서가 지정된 명시적 대체 모델 목록. |
| `helper_models.default.fallbacks.model` | `str` | `"—"` | 대체 모델 식별자. |
| `helper_models.default.fallbacks.credential` | `str \| None` | `null` | 대체 모델의 인증 정보 이름. |
| `helper_models.default.allow_agent_sdk_fallback` | `bool` | `false` | 이 역할의 원샷 호출이 Claude Agent SDK로 대체되도록 허용합니다. |
| `helper_models.default.max_output_tokens` | `int \| None` | `null` | 이 역할에서 수행하는 원샷 호출의 최대 출력 토큰 수. |

### `background_review`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `background_review` | `BackgroundReviewConfig` | `{BackgroundReviewConfig}` | 세션 후 비동기 회고 및 인물상 업데이트 설정. |
| `background_review.enabled` | `bool` | `true` | — |
| `background_review.chat_every_user_turns` | `int` | `10` | — |
| `background_review.min_interval_minutes` | `int` | `10` | — |
| `background_review.max_per_day` | `int` | `24` | — |
| `background_review.max_input_bytes` | `int` | `61440` | — |
| `background_review.max_writes` | `int` | `3` | — |
| `background_review.peer_profile_max_chars` | `int` | `1500` | — |

### `rag`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `rag` | `RAGConfig` | `{RAGConfig}` | 검색 증강 생성 및 기억 검색 설정. |
| `rag.enabled` | `bool` | `true` | — |
| `rag.embedding_model` | `str` | `"intfloat/multilingual-e5-small"` | — |
| `rag.embedding_e5_prefix_enabled` | `bool` | `false` | 활성화하면 쿼리 임베딩 앞에 embedding_query_prefix를, 인덱싱된 문서 임베딩 앞에 embedding_document_prefix를 추가합니다. E5 계열 절제 실험을 위해 마련된 기능이며, 문서 접두사 변경 사항을 적용하려면 다시 인덱싱해야 합니다. |
| `rag.embedding_query_prefix` | `str` | `"query: "` | — |
| `rag.embedding_document_prefix` | `str` | `"passage: "` | — |
| `rag.embedding_max_seq_length` | `int` | `2048` | 임베딩 모델의 최대 시퀀스 길이(토큰) 제한입니다. ruri-v3와 같은 긴 컨텍스트 모델의 기본값은 8192이며, 대량 인코딩 중 GPU 활성화 메모리 사용량이 급증합니다. 0 = 모델 기본값 사용. |
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
| `rag.fact_extraction_timeout_seconds` | `int` | `120` | 기존 원자 사실 추출에 사용하는 LLM 호출의 기본 시간 제한(초)입니다. Anima별 status.json extraction_timeout 값이 이 값을 재정의합니다. |
| `rag.fact_extraction_max_tokens` | `int` | `8192` | 기존 원자 사실 추출 LLM 호출의 최대 출력 토큰 수입니다. |
| `rag.facts_extraction_single_call` | `bool` | `true` | 지원되는 경우 단일 LLM 호출로 기존 원자 사실과 해당 엔터티를 추출합니다. |
| `rag.facts_reconcile_enabled` | `bool` | `true` | 추가하기 전에 기존 원자 사실 조정을 활성화합니다. 실패하면 ADD로 대체합니다. |
| `rag.facts_reconcile_similarity_threshold` | `float` | `0.82` | 엄격한 LLM duplicate/contradiction/complement 라벨링을 적용하기 위한 사실 벡터 유사도 최소값입니다. |
| `rag.facts_reconcile_top_k` | `int` | `5` | 기존 사실 조정 중 고려할 유사한 활성 사실의 최대 수입니다. |
| `rag.entity_registry_enabled` | `bool` | `true` | — |

### `gpu`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `gpu` | `GPUConfig` | `{GPUConfig}` | GPU 사용 및 장치 선택 설정. |
| `gpu.embedding_device` | `Literal['auto', 'cuda', 'cpu']` | `"auto"` | — |
| `gpu.reranker_device` | `Literal['auto', 'cuda', 'cpu']` | `"cpu"` | — |
| `gpu.embedding_batch_size` | `int` | `32` | — |
| `gpu.embedding_bulk_yield_batches` | `int` | `5` | 대량 작업이 대기 중인 대화형 작업에 양보할 수 있는 임베딩 배치의 연속 처리 최대 횟수입니다. |

### `memory`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `memory` | `MemoryConfig` | `{MemoryConfig}` | 기억 저장 및 검색 공통 설정. |
| `memory.fact_edge_types` | `list[FactEdgeTypeConfig]` | `[]` | — |
| `memory.fact_edge_types.name` | `str` | `"—"` | 대문자 스네이크 케이스 형식의 의미적 에지 유형 이름 |
| `memory.fact_edge_types.description` | `str` | `"—"` | 추출 프롬프트에 표시되는 짧은 설명 |

### `skills`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `skills` | `SkillsConfig` | `{SkillsConfig}` | 스킬 로드 및 관리 설정. |
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
| `skills.external_roots.path` | `str` | `"—"` | ``~``를 포함할 수 있으며, 사용 전에 확장됩니다. |
| `skills.external_roots.engine` | `str` | `"—"` | ^[a-z][a-z0-9-]*$와 일치합니다. |
| `skills.external_roots.trust_level` | `str` | `"trusted"` | — |
| `skills.external_roots.enabled` | `bool` | `true` | — |

### `chatwork_tool`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `chatwork_tool` | `ChatworkToolConfig` | `{ChatworkToolConfig}` | Chatwork 도구 권한 설정. |
| `chatwork_tool.grants` | `dict[str, dict[str, str]]` | `{}` | — |

### `prompt`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `prompt` | `PromptConfig` | `{PromptConfig}` | 시스템 프롬프트 및 프롬프트 구성 설정. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `priming` | `PrimingConfig` | `{PrimingConfig}` | Anima 시작 시 로드할 정보 설정. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `image_gen` | `ImageGenConfig` | `{ImageGenConfig}` | 이미지 생성 공급자 및 기본 매개변수. |
| `image_gen.backend` | `Literal['api', 'diffusers', 'atlascloud']` | `"api"` | — |
| `image_gen.image_style` | `Literal['anime', 'realistic']` | `"realistic"` | — |
| `image_gen.prefer_codex` | `bool` | `true` | codex CLI가 있으면 이미지 생성에 최우선으로 사용 |
| `image_gen.style_reference` | `str \| None` | `null` | 조직 전체에서 사용할 스타일 참조 이미지 경로 |
| `image_gen.style_prefix` | `str` | `""` | 캐릭터 프롬프트 앞에 추가되는 공통 스타일 태그 |
| `image_gen.style_suffix` | `str` | `""` | 캐릭터 프롬프트 뒤에 추가되는 공통 스타일 태그 |
| `image_gen.negative_prompt_extra` | `str` | `""` | 네거티브 프롬프트에 추가되는 태그 |
| `image_gen.vibe_strength` | `float` | `0.6` | 바이브 전송 강도 (0.0-1.0) |
| `image_gen.vibe_info_extracted` | `float` | `0.8` | 바이브 전송 정보 추출 (0.0-1.0) |
| `image_gen.enable_3d` | `bool` | `true` | 3D 모델 생성 활성화 (Meshy API) |
| `image_gen.pixel_base_sheet` | `str \| None` | `null` | 기본 픽셀 시트. None이면 번들로 제공되는 sample_01.png 사용 |
| `image_gen.pixel_concurrency` | `int` | `4` | 픽셀 시트 행별 codex image_gen 호출 병렬 실행 |
| `image_gen.diffusers_text2img_model` | `str` | `"auto"` | — |
| `image_gen.diffusers_img2img_model` | `str` | `"auto"` | — |
| `image_gen.diffusers_text2img_model_realistic` | `str` | `""` | 사실적인 스타일에 대한 재정의 |
| `image_gen.diffusers_text2img_model_anime` | `str` | `""` | 애니메이션 스타일에 대한 재정의 |
| `image_gen.diffusers_device` | `Literal['auto', 'cuda', 'cpu']` | `"auto"` | — |
| `image_gen.diffusers_torch_dtype` | `Literal['auto', 'float16', 'float32', 'bfloat16']` | `"auto"` | — |
| `image_gen.diffusers_local_files_only` | `bool` | `true` | — |
| `image_gen.diffusers_num_inference_steps` | `int` | `28` | — |
| `image_gen.diffusers_img2img_strength` | `float` | `0.55` | — |
| `image_gen.ip_adapter_model` | `str` | `"h94/IP-Adapter"` | — |
| `image_gen.ip_adapter_scale` | `float` | `0.6` | IP-Adapter 얼굴 참조 혼합 가중치 (0.0-1.0) |

### `human_notification`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `human_notification` | `HumanNotificationConfig` | `{HumanNotificationConfig}` | 인간에게 알리는 방법과 알림 대상. |
| `human_notification.enabled` | `bool` | `false` | — |
| `human_notification.channels` | `list[NotificationChannelConfig]` | `[]` | — |
| `human_notification.channels.type` | `str` | `"—"` | "slack", "line", "telegram", "chatwork", "ntfy" |
| `human_notification.channels.enabled` | `bool` | `true` | — |
| `human_notification.channels.config` | `dict[str, Any]` | `{}` | — |
| `human_notification.web_ui` | `bool` | `true` | 내장 채널: call_human을 Web UI 채팅에 푸시 |

### `interaction`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `interaction` | `InteractionConfig` | `{InteractionConfig}` | 애니마 간 대화 및 메시지 처리. |
| `interaction.default_approver_ids` | `list[str]` | `[]` | 기본 Slack 사용자 ID와 호출별 call_human 허용 사용자 ID를 합칩니다. |
| `interaction.web_base_url` | `str` | `""` | — |

### `server`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `server` | `ServerConfig` | `{ServerConfig}` | HTTP 서버, 인증, 사용량 제어 설정. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `llm_rate_guard` | `LlmRateGuardConfig` | `{LlmRateGuardConfig}` | LLM 호출 빈도 및 동시 실행 수 제어. |
| `llm_rate_guard.enabled` | `bool` | `true` | — |
| `llm_rate_guard.default_block_seconds` | `int` | `60` | — |
| `llm_rate_guard.max_block_seconds` | `int` | `600` | — |
| `llm_rate_guard.quota_block_seconds` | `int` | `1800` | — |
| `llm_rate_guard.max_quota_block_seconds` | `int` | `14400` | — |

### `mcp`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `mcp` | `MCPConfig` | `{MCPConfig}` | MCP 서버 연결 및 도구 공개 설정. |
| `mcp.trigger_scoped_tools` | `bool` | `true` | — |

### `external_messaging`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `external_messaging` | `ExternalMessagingConfig` | `{ExternalMessagingConfig}` | Slack 등 외부 메시징 연동. |
| `external_messaging.preferred_channel` | `str` | `"slack"` | "slack" \| "chatwork" \| "discord" |
| `external_messaging.user_aliases` | `dict[str, UserAliasConfig]` | `{}` | 별칭 → 연락처 정보 |
| `external_messaging.user_aliases.slack_user_id` | `str` | `""` | — |
| `external_messaging.user_aliases.chatwork_room_id` | `str` | `""` | — |
| `external_messaging.user_aliases.discord_user_id` | `str` | `""` | — |
| `external_messaging.user_aliases.outbound_dm` | `bool` | `false` | — |
| `external_messaging.chat_dm_redirect` | `bool` | `false` | — |
| `external_messaging.slack` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.slack.enabled` | `bool` | `false` | — |
| `external_messaging.slack.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.slack.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = 이 채널 무시) |
| `external_messaging.slack.default_anima` | `str` | `""` | 매핑되지 않은 채널의 대체 Anima |
| `external_messaging.slack.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (Anima별 웹훅 라우팅) |
| `external_messaging.slack.auto_response` | `bool` | `false` | LLM 응답을 원래 플랫폼에 자동 게시 |
| `external_messaging.slack.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (자동 입력) |
| `external_messaging.slack.board_outbound_sync` | `list[str]` | `[]` | 이 플랫폼으로 내보낼 동기화 대상 보드 이름(허용 목록) |
| `external_messaging.slack.board_outbound_sync_all` | `bool` | `false` | 허용 목록이 비어 있으면 매핑된 모든 보드의 동기화를 허용 |
| `external_messaging.slack.guild_id` | `str` | `""` | Discord 길드 스노플레이크 ID (Discord 전용) |
| `external_messaging.slack.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord 전용) |
| `external_messaging.slack.default_channel_company` | `str` | `""` | 자동 생성된 보드의 회사 (비어 있으면 귀속 없음) |
| `external_messaging.chatwork` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.chatwork.enabled` | `bool` | `false` | — |
| `external_messaging.chatwork.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.chatwork.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = 이 채널 무시) |
| `external_messaging.chatwork.default_anima` | `str` | `""` | 매핑되지 않은 채널의 대체 Anima |
| `external_messaging.chatwork.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (Anima별 웹훅 라우팅) |
| `external_messaging.chatwork.auto_response` | `bool` | `false` | LLM 응답을 원래 플랫폼에 자동 게시 |
| `external_messaging.chatwork.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (자동 입력) |
| `external_messaging.chatwork.board_outbound_sync` | `list[str]` | `[]` | 이 플랫폼으로 내보낼 동기화 대상 보드 이름(허용 목록) |
| `external_messaging.chatwork.board_outbound_sync_all` | `bool` | `false` | 허용 목록이 비어 있으면 매핑된 모든 보드의 동기화를 허용 |
| `external_messaging.chatwork.guild_id` | `str` | `""` | Discord 길드 스노플레이크 ID (Discord 전용) |
| `external_messaging.chatwork.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord 전용) |
| `external_messaging.chatwork.default_channel_company` | `str` | `""` | 자동 생성된 보드의 회사 (비어 있으면 귀속 없음) |
| `external_messaging.discord` | `ExternalMessagingChannelConfig` | `{ExternalMessagingChannelConfig}` | — |
| `external_messaging.discord.enabled` | `bool` | `false` | — |
| `external_messaging.discord.mode` | `str` | `"socket"` | "socket" \| "webhook" |
| `external_messaging.discord.anima_mapping` | `dict[str, str]` | `{}` | channel_id → anima_name ("" = 이 채널 무시) |
| `external_messaging.discord.default_anima` | `str` | `""` | 매핑되지 않은 채널의 대체 Anima |
| `external_messaging.discord.app_id_mapping` | `dict[str, str]` | `{}` | api_app_id → anima_name (Anima별 웹훅 라우팅) |
| `external_messaging.discord.auto_response` | `bool` | `false` | LLM 응답을 원래 플랫폼에 자동 게시 |
| `external_messaging.discord.board_mapping` | `dict[str, str]` | `{}` | channel_id → animaworks_board_name (자동 입력) |
| `external_messaging.discord.board_outbound_sync` | `list[str]` | `[]` | 이 플랫폼으로 내보낼 동기화 대상 보드 이름(허용 목록) |
| `external_messaging.discord.board_outbound_sync_all` | `bool` | `false` | 허용 목록이 비어 있으면 매핑된 모든 보드의 동기화를 허용 |
| `external_messaging.discord.guild_id` | `str` | `""` | Discord 길드 스노플레이크 ID (Discord 전용) |
| `external_messaging.discord.channel_members` | `dict[str, list[str]]` | `{}` | channel_id → [anima_name, ...] (Discord 전용) |
| `external_messaging.discord.default_channel_company` | `str` | `""` | 자동 생성된 보드의 회사 (비어 있으면 귀속 없음) |
| `external_messaging.zoom` | `ZoomRTMSConfig` | `{ZoomRTMSConfig}` | — |
| `external_messaging.zoom.enabled` | `bool` | `false` | — |
| `external_messaging.zoom.default_anima` | `str` | `""` | 매핑되지 않은 회의의 대체 Anima |
| `external_messaging.zoom.meeting_mapping` | `dict[str, str]` | `{}` | meeting_id → anima_name |
| `external_messaging.zoom.chunk_interval_seconds` | `int` | `300` | 버퍼링된 대화록의 플러시 간격 |
| `external_messaging.zoom.chunk_max_chars` | `int` | `4000` | 청크당 최대 문자 수 (먼저 도달하는 기준으로 플러시) |

### `external_tasks`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `external_tasks` | `ExternalTasksConfig` | `{ExternalTasksConfig}` | 외부 작업 시스템과의 연동. |
| `external_tasks.enabled` | `bool` | `true` | — |
| `external_tasks.interval_minutes` | `int` | `5` | — |
| `external_tasks.sources` | `ExternalTasksSourcesConfig` | `{ExternalTasksSourcesConfig}` | — |
| `external_tasks.sources.github` | `bool` | `true` | — |
| `external_tasks.sources.slack` | `bool` | `true` | — |
| `external_tasks.sources.chatwork` | `bool` | `true` | — |
| `external_tasks.sources.gmail` | `bool` | `true` | — |

### `github_webhook`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `github_webhook` | `GitHubWebhookConfig` | `{GitHubWebhookConfig}` | GitHub 웹훅 수신 및 검증. |
| `github_webhook.enabled` | `bool` | `false` | — |
| `github_webhook.repos` | `list[str]` | `[]` | — |
| `github_webhook.dispatcher_anima` | `str` | `"rin"` | — |
| `github_webhook.bot_login` | `str` | `""` | — |
| `github_webhook.reviewer_login` | `str` | `""` | — |
| `github_webhook.quiet_seconds` | `float` | `180` | — |
| `github_webhook.drop_bot_noise` | `bool` | `true` | — |

### `event_export`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `event_export` | `EventExportConfig` | `{EventExportConfig}` | 이벤트 데이터 내보내기. |
| `event_export.url` | `str \| None` | `null` | — |
| `event_export.headers` | `dict[str, str]` | `{}` | — |
| `event_export.event_types` | `list[str] \| None` | `null` | — |
| `event_export.include_token_usage` | `bool` | `true` | — |
| `event_export.max_retries` | `int` | `8` | — |
| `event_export.backoff_base_seconds` | `float` | `2.0` | — |
| `event_export.spool_max_mb` | `int` | `64` | — |

### `background_task`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `background_task` | `BackgroundTaskConfig` | `{BackgroundTaskConfig}` | 백그라운드 실행 대상 도구 설정. |
| `background_task.enabled` | `bool` | `true` | — |
| `background_task.shutdown_drain_seconds` | `float` | `600.0` | — |
| `background_task.eligible_tools` | `dict[str, BackgroundToolConfig]` | `…` | — |
| `background_task.eligible_tools.threshold_s` | `int` | `30` | — |
| `background_task.result_memory_retention_minutes` | `int` | `60` | 프로세스 내 결과 캐시 |
| `background_task.max_completed_tasks_in_memory` | `int` | `200` | — |
| `background_task.worker_pool_size` | `int` | `1` | — |

### `activity_log`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `activity_log` | `ActivityLogConfig` | `{ActivityLogConfig}` | 작업 및 활동 로그 저장 설정. |
| `activity_log.rotation_enabled` | `bool` | `true` | — |
| `activity_log.rotation_mode` | `Literal['size', 'time', 'both']` | `"size"` | — |
| `activity_log.max_size_mb` | `int` | `1024` | per-anima total, default 1GB |
| `activity_log.max_file_size_mb` | `int` | `100` | per-file bloat trigger; 0 disables |
| `activity_log.max_age_days` | `int` | `7` | mode="time"\|"both"에서 사용 |
| `activity_log.rotation_time` | `str` | `"05:00"` | 실행 시각 (configured TZ) |

### `logging`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `logging` | `LoggingConfig` | `{LoggingConfig}` | 로그 수준, 출력 대상, 민감 정보 마스킹. |
| `logging.redaction_enabled` | `bool` | `true` | Mask secrets in log output; disable for raw-log debugging. |

### `heartbeat`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `heartbeat` | `HeartbeatConfig` | `{HeartbeatConfig}` | 정기 heartbeat 실행 설정. |
| `heartbeat.interval_minutes` | `int` | `30` | — |
| `heartbeat.current_state_max_chars` | `int` | `8000` | Max chars for current_state.md before trim; 0 = disabled |
| `heartbeat.current_state_cleanup_chars` | `int` | `2000` | Soft cleanup threshold for current_state.md; 0 = 80% of current_state_max_chars |
| `heartbeat.heartbeat_md_max_bytes` | `int` | `8000` | Max bytes of HEARTBEAT.md before a compaction instruction is injected into the heartbeat prompt; 0 = disabled |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `voice` | `VoiceConfig` | `{VoiceConfig}` | 음성 입출력 및 음성 공급자. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `phone` | `PhoneConfig` | `{PhoneConfig}` | Twilio를 이용한 Anima 전용 전화 대화와 긴급 알림. |
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
| `phone.turn_timeout_sec` | `float` | `300` | 레거시 Gather/poll 타임아웃; Media Streams 전화 대화에서는 사용되지 않음. |
| `phone.turn_end_silence_ms` | `int` | `1200` | — |

### `housekeeping`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `housekeeping` | `HousekeepingConfig` | `{HousekeepingConfig}` | 런타임 데이터 정기 정리. |
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

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `inbox` | `InboxConfig` | `{InboxConfig}` | anima의 받은 편지함 및 알림 표시. |
| `inbox.inbox_quiet_seconds` | `float` | `90.0` | — |
| `inbox.inbox_quiet_max_wait_seconds` | `float` | `300.0` | — |
| `inbox.ttl_hours` | `float` | `24.0` | — |
| `inbox.expired_retention_days` | `int` | `7` | — |
| `inbox.processed_retention_days` | `int` | `30` | — |
| `inbox.quarantine_retention_days` | `int` | `30` | — |

### `local_llm`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `local_llm` | `LocalLLMConfig` | `{LocalLLMConfig}` | 로컬 LLM 엔드포인트 및 모델 설정. |
| `local_llm.base_url` | `str` | `"http://127.0.0.1:11434"` | — |
| `local_llm.default_model` | `str` | `"ollama/qwen2.5-coder:14b"` | — |
| `local_llm.credential` | `str` | `""` | — |
| `local_llm.auto_apply_presets` | `bool` | `false` | — |
| `local_llm.presets` | `dict[str, str]` | `{"coding":"ollama/qwen2.5-coder:14b","general":"ollama/glm4:9b","reasoning":"ollama/deepseek-r1:8b"}` | — |
| `local_llm.role_presets` | `dict[str, str]` | `…` | — |

### `workspaces`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `workspaces` | `dict[str, str]` | `{}` | 이름이 지정된 워크스페이스 경로 매핑. |

### `github_identities`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `github_identities` | `dict[str, str]` | `{}` | 회사 slug와 GitHub 계정 매핑. |

### `activity_level`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `activity_level` | `int` | `100` | 전체 활동 빈도를 조정하는 배율. |

### `activity_schedule`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `activity_schedule` | `list[ActivityScheduleEntry]` | `[]` | 시간대별 활동 빈도. |
| `activity_schedule.start` | `str` | `"—"` | HH:MM 형식의 시작 시간 |
| `activity_schedule.end` | `str` | `"—"` | HH:MM 형식의 종료 시간 (자정을 넘어갈 수 있음) |
| `activity_schedule.level` | `int` | `"—"` | 이 기간의 활동 수준 백분율 |

### `icon_url_template`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `icon_url_template` | `str` | `""` | anima 아이콘 URL 템플릿. |

### `ui`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `ui` | `UIConfig` | `{UIConfig}` | 웹 UI 표시 설정. |
| `ui.theme` | `str` | `"default"` | — |
| `ui.demo_mode` | `bool` | `false` | — |

### `cli`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `cli` | `CLIConfig` | `{CLIConfig}` | CLI 클라이언트 설정. default_anima는 인수 없이 animaworks / animaworks chat을 실행할 때 열리는 anima를 지정합니다(비어 있으면 상급자 없이 활성화된 anima가 하나뿐일 때 해당 anima를 사용). |
| `cli.default_anima` | `str` | `""` | — |

### `enclave`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `enclave` | `EnclaveConfig` | `{EnclaveConfig}` | 격리된 엔클레이브 모드의 서버 설정. 활성화하면 시작 시 보안 가드를 검사합니다. |
| `enclave.enabled` | `bool` | `false` | 이 실행 환경을 격리된 엔클레이브로 시작합니다. |
| `enclave.name` | `str` | `""` | 엔클레이브 게이트웨이가 반환하는 식별 이름입니다. |
| `enclave.socket_path` | `str` | `""` | 게이트웨이가 수신 대기하는 Unix 도메인 소켓 경로입니다. |
| `enclave.socket_group` | `str` | `""` | 게이트웨이 소켓에 설정할 OS 그룹입니다. |
| `enclave.entry_anima` | `str` | `""` | 엔클레이브 내에서 질문을 처리하는 엔트리 anima입니다. |
| `enclave.allowed_peer_uids` | `list[int]` | `[]` | 게이트웨이 연결을 허용할 OS 사용자 UID입니다. |
| `enclave.max_concurrency` | `int` | `2` | 게이트웨이가 동시에 처리할 질문의 최대 개수입니다. |
| `enclave.request_timeout_s` | `int` | `0` | 질문 처리 및 게이트웨이 응답의 제한 시간(초)입니다. |
| `enclave.allowed_llm_credentials` | `list[str]` | `[]` | 엔클레이브 내 anima가 사용할 수 있는 인증 정보 이름입니다. |
| `enclave.model_profiles` | `dict[str, str]` | `{}` | — |
| `enclave.default_model_profile` | `str` | `""` | — |
| `enclave.datasets` | `dict[str, EnclaveDatasetConfig]` | `{}` | 엔클레이브 내 anima가 참조할 수 있는 JSONL 데이터 세트입니다. |
| `enclave.datasets.path` | `str` | `"—"` | 데이터 디렉터리에 있는 JSONL 파일의 상대 경로입니다. |
| `enclave.datasets.id_field` | `str` | `"—"` | 레코드를 고유하게 가져오기 위한 ID 필드입니다. |
| `enclave.datasets.searchable_fields` | `list[str]` | `[]` | 부분 일치 검색을 허용할 필드입니다. |
| `enclave.sql_sources` | `dict[str, EnclaveSqlSourceConfig]` | `{}` | — |
| `enclave.sql_sources.driver` | `Literal['mysql']` | `"mysql"` | — |
| `enclave.sql_sources.host` | `str` | `"—"` | 터널 대상(또는 직접 연결) 호스트 |
| `enclave.sql_sources.port` | `int` | `3306` | — |
| `enclave.sql_sources.database` | `str` | `"—"` | — |
| `enclave.sql_sources.user` | `str` | `"—"` | — |
| `enclave.sql_sources.password_secret` | `str` | `"—"` | — |
| `enclave.sql_sources.ssl` | `bool` | `true` | — |
| `enclave.sql_sources.ssl_ca` | `str \| None` | `null` | TLS 검증에 사용할 선택적 CA 번들 경로 |
| `enclave.sql_sources.ssl_verify_identity` | `bool` | `false` | — |
| `enclave.sql_sources.tunnel` | `EnclaveSsmTunnelConfig \| None` | `null` | — |
| `enclave.sql_sources.tunnel.type` | `Literal['ssm_port_forward']` | `"ssm_port_forward"` | — |
| `enclave.sql_sources.tunnel.region` | `str` | `"—"` | SSM 및 대상 확인에 사용할 AWS 리전 |
| `enclave.sql_sources.tunnel.target_instance_id` | `str \| None` | `null` | — |
| `enclave.sql_sources.tunnel.target_tag_name` | `str \| None` | `null` | — |
| `enclave.sql_sources.tunnel.aws_secret` | `str` | `"—"` | AWS 인증 정보 JSON이 저장된 시크릿 이름 |
| `enclave.sql_sources.tunnel.plugin_path` | `str` | `"/usr/local/bin/session-manager-plugin"` | — |
| `enclave.sql_sources.tunnel.idle_shutdown_s` | `int` | `600` | — |
| `enclave.sql_sources.tunnel.local_port` | `int \| None` | `null` | 터널의 고정 루프백 포트(호스트 방화벽에서 허용할 수 있음). 설정하지 않으면 무작위 포트 사용 |
| `enclave.sql_sources.tunnel.local_port_count` | `int` | `1` | local_port에서 시작하는 연속 루프백 포트 수입니다. 동시 실행 프로세스는 각각 다른 포트를 예약합니다. |
| `enclave.sql_sources.max_rows` | `int` | `200` | — |
| `enclave.sql_sources.timeout_s` | `int` | `30` | — |
| `enclave.sql_sources.cell_max_chars` | `int` | `2000` | — |
| `enclave.sql_sources.app_key_secret` | `str \| None` | `null` | — |
| `enclave.sql_sources.decrypt_columns` | `list[str]` | `[]` | — |
| `enclave.sql_sources.session_variables` | `dict[str, str]` | `{}` | 모든 연결에서 설정되는 사용자 변수(``SET @name = 'value'``)입니다. 예를 들어 세션 변수를 기준으로 필터링하는 행 수준 뷰에 사용할 수 있습니다. |
| `enclave.aws_sources` | `dict[str, EnclaveAwsSourceConfig]` | `{}` | — |
| `enclave.aws_sources.region` | `str` | `"—"` | — |
| `enclave.aws_sources.aws_secret` | `str` | `"—"` | AWS 인증 정보 JSON이 저장된 시크릿 이름 |
| `enclave.aws_sources.log_groups` | `list[str]` | `[]` | — |
| `enclave.aws_sources.pi_resource_id` | `str \| None` | `null` | — |
| `enclave.aws_sources.rds_instance_id` | `str \| None` | `null` | — |
| `enclave.aws_sources.s3_buckets` | `list[str]` | `[]` | — |
| `enclave.aws_sources.max_bytes` | `int` | `200000` | — |
| `enclave.secrets_dir` | `str \| None` | `null` | — |
| `enclave.raw_dir` | `str` | `"raw"` | — |
| `enclave.egress` | `dict[str, Any]` | `{}` | 외부로 반환할 facts를 처리하는 egress 단계 설정입니다. |
| `enclave.egress_probe` | `list[str]` | `[]` | — |
| `enclave.egress_probe_timeout_s` | `float` | `5.0` | — |

### `enclaves`

| 키 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `enclaves` | `dict[str, EnclaveClientConfig]` | `{}` | 외부 enclave 인스턴스에 연결하는 클라이언트 측 설정. |
| `enclaves.socket_path` | `str` | `"—"` | 연결할 enclave의 Unix 도메인 소켓. |
| `enclaves.allowed_animas` | `list[str]` | `[]` | 해당 enclave에 질문하는 것이 허용된 본체 측 anima. |
| `enclaves.timeout_s` | `int` | `900` | enclave gateway 연결 시간 초과(초). |

## anima별 `status.json`

| 키 | ModelConfig 유형 | 기본값 | 설명 |
|---|---|---|---|
| `background_credential` | `str \| None` | `null` | — |
| `background_model` | `str \| None` | `null` | heartbeat나 cron에서 사용하는 모델 오버라이드. |
| `background_thinking_effort` | `str \| None` | `null` | — |
| `consolidation_enabled` | `—` | `—` | — |
| `context_absolute_ceiling` | `float` | `0.75` | — |
| `context_threshold` | `float` | `0.5` | — |
| `conversation_history_threshold` | `float` | `0.3` | — |
| `credential` | `str \| None` | `null` | anima에 연결할 인증 정보 이름. |
| `default_workspace` | `—` | `—` | — |
| `execution_mode` | `str \| None` | `null` | anima의 실행 모드. |
| `extra_mcp_servers` | `dict[str, dict]` | `{}` | — |
| `fallback_model` | `str \| None` | `null` | 주 모델 실패 시 사용할 대체 모델. |
| `fallback_models` | `list[str]` | `[]` | — |
| `heartbeat_enabled` | `bool` | `true` | 정기 heartbeat의 활성·비활성. |
| `max_session_age_hours` | `float` | `24.0` | — |
| `max_tokens` | `int` | `8192` | 모델의 최대 출력 토큰 수. |
| `mode_s_auth` | `str \| None` | `null` | — |
| `model` | `str` | `"claude-sonnet-5-5"` | anima의 주 모델. |
| `speciality` | `str \| None` | `null` | — |
| `supervisor` | `str \| None` | `null` | 상위 supervisor anima의 이름. |
| `task_compaction_max` | `int` | `6` | — |
| `task_compaction_tokens` | `int` | `0` | — |
| `thinking` | `bool \| None` | `null` | — |
| `thinking_effort` | `str \| None` | `null` | — |
| `token_budget_monthly` | `int \| None` | `null` | — |
| `voice_thinking_effort` | `str \| None` | `null` | — |
| `bootstrap_state` | `—` | `—` | 초기화 처리의 상태. |
| `company` | `—` | `—` | 소속 회사. |
| `department` | `—` | `—` | 소속 부서. |
| `enabled` | `—` | `—` | anima의 활성·비활성. |
| `needs_user_input` | `—` | `—` | 사용자 입력이 필요한 상태인지 여부. |
| `role` | `—` | `—` | anima의 역할 템플릿. |

## `models.json`

모델명 패턴별 실행 모드와 컨텍스트 윈도우입니다.

| 모델명 패턴 | 모드 | 컨텍스트 윈도우 |
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
