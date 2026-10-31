<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/reference/modules.md -->
<!-- i18n: source-sha256=5f183e62f463963e2ac2b6f15b16291e5e6376e527435d0132b42afff2ece880 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

# 모듈 목록

`git ls-files core cli server`에서 추적 대상인 Python 파일을 나열합니다. 비공개 모듈에는 표시가 있습니다.

## `cli`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `cli` | 9 | — |
| `cli.__main__（非公開）` | 9 | — |
| `cli._anima_tool（非公開）` | 33 | — |
| `cli._gateway（非公開）` | 93 | — |
| `cli.codex_command_hook` | 64 | Codex의 ``PreToolUse`` 명령 정책 훅용 CLI 어댑터. |
| `cli.demo` | 407 | 네이티브 ``animaworks demo`` 명령. |
| `cli.parser` | 867 | — |
| `cli.tool_dispatch` | 414 | 외부 도구, 작업 제출 및 명령 별칭을 위한 CLI 디스패치. |

## `cli.commands`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `cli.commands` | 5 | — |
| `cli.commands.anima` | 303 | — |
| `cli.commands.anima_mgmt` | 1173 | 아니마 프로세스 관리를 위한 CLI 명령. |
| `cli.commands.board` | 144 | — |
| `cli.commands.company_cmd` | 272 | — |
| `cli.commands.config_cmd` | 238 | ``animaworks config``용 CLI 핸들러 및 대화형 마법사. |
| `cli.commands.cost_cmd` | 326 | — |
| `cli.commands.enclave_cmd` | 323 | 엔클레이브 런타임 운영 명령. |
| `cli.commands.import_cmd` | 88 | — |
| `cli.commands.index_cmd` | 376 | — |
| `cli.commands.init_cmd` | 155 | — |
| `cli.commands.internal_cmd` | 156 | — |
| `cli.commands.logs` | 209 | 아니마 로그를 보기 위한 CLI 명령. |
| `cli.commands.mcp_cmd` | 66 | — |
| `cli.commands.memory_cmd` | 56 | — |
| `cli.commands.messaging` | 86 | — |
| `cli.commands.migrate_cmd` | 114 | — |
| `cli.commands.models_cmd` | 304 | 모델 정보 및 관리를 위한 CLI 명령. |
| `cli.commands.optimize_assets` | 189 | — |
| `cli.commands.profile` | 332 | — |
| `cli.commands.rag_repair_status` | 147 | 영구 RAG 복구 상태 보고. |
| `cli.commands.remake_cmd` | 279 | — |
| `cli.commands.repair_rag_cmd` | 135 | — |
| `cli.commands.server` | 868 | — |
| `cli.commands.skills` | 211 | — |
| `cli.commands.supervisor_cmd` | 91 | — |
| `cli.commands.task_cmd` | 667 | — |
| `cli.commands.task_store_cmd` | 258 | 운영자 전용, 코호트 단위 작업 마이그레이션 및 현재 상태 내보내기. |
| `cli.commands.tmp_cmd` | 173 | — |
| `cli.commands.vault_cmd` | 309 | — |

## `cli.tui`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `cli.tui` | 207 | — |
| `cli.tui.app` | 1936 | — |
| `cli.tui.client` | 394 | — |
| `cli.tui.commands` | 251 | — |
| `cli.tui.keybindings` | 90 | TUI용 키 바인딩 설정. |
| `cli.tui.markdown` | 535 | 터미널 대화 기록에 맞게 조정한 Markdown → Rich 렌더링 객체. |
| `cli.tui.session` | 216 | TUI 세션 유지 기능. |
| `cli.tui.sse` | 98 | — |
| `cli.tui.state` | 280 | TUI 사이드바, 활동 피드 및 팔레트의 클라이언트 전용 UI 상태. |
| `cli.tui.widgets.call_human` | 85 | 대화 기록에 표시되는 call_human 알림 카드. |
| `cli.tui.widgets.chat_input` | 152 | — |
| `cli.tui.widgets.palette` | 115 | 슬래시 명령 자동 완성 팔레트. |
| `cli.tui.widgets.response_status` | 63 | — |
| `cli.tui.widgets.sidebar` | 248 | 사이드바 위젯: Anima 목록 및 활동 피드. |
| `cli.tui.widgets.status_bar` | 160 | — |
| `cli.tui.widgets.thinking` | 98 | — |
| `cli.tui.widgets.tool_card` | 140 | — |
| `cli.tui.widgets.transcript` | 293 | — |

## `cli.tui.widgets`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `cli.tui.widgets` | 32 | — |

## `core`

—

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core` | 7 | — |
| `core.credentials` | 276 | — |
| `core.exceptions` | 157 | — |
| `core.host_api` | 88 | — |
| `core.internal_api` | 51 | — |
| `core.paths` | 218 | AnimaWorks의 중앙 집중식 경로 확인. |
| `core.schemas` | 242 | — |
| `core.time_utils` | 103 | — |
| `core.trust` | 434 | — |

## `core.activity`

활동 로그의 기록, 재생, 타임라인 표시.

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.activity` | 24 | — |
| `core.activity.audit` | 290 | — |
| `core.activity.conversation` | 1414 | — |
| `core.activity.event_export` | 33 | — |
| `core.activity.format` | 571 | — |
| `core.activity.logger` | 584 | — |
| `core.activity.models` | 202 | — |
| `core.activity.replay` | 537 | — |
| `core.activity.rotation` | 187 | — |
| `core.activity.runtime_context` | 30 | — |
| `core.activity.slack_mirror` | 601 | — |
| `core.activity.timeline` | 348 | — |

## `core.agent`

LLM 에이전트 실행, 대화 제어, 엔진 연동.

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.agent` | 23 | — |
| `core.agent.agent_core` | 325 | — |
| `core.agent.cycle` | 1524 | — |
| `core.agent.executor_factory` | 180 | — |
| `core.agent.priming` | 504 | — |
| `core.agent.prompt_log` | 151 | — |
| `core.agent.session_compactor` | 566 | 애니마별 × thread_id별 유휴 압축 타이머 관리. |

## `core.anima`

디지털 Anima의 수명 주기와 런타임 객체.

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.anima` | 23 | — |
| `core.anima._mixin_protocols（非公開）` | 130 | 구성 믹스인을 위한 구조적 호스트 프로토콜. |
| `core.anima.admin` | 170 | — |
| `core.anima.asset_reconciler` | 786 | — |
| `core.anima.bootstrap_state` | 591 | — |
| `core.anima.chat_attachments` | 253 | — |
| `core.anima.digital_anima` | 702 | — |
| `core.anima.emotion_tag` | 84 | LLM 응답을 위한 공통 감정 태그 추출. |
| `core.anima.factory` | 842 | Anima 생성 팩토리: 템플릿, 빈 상태 또는 MD 파일에서 새 디지털 Anima를 생성합니다. |
| `core.anima.heartbeat` | 1022 | — |
| `core.anima.image_artifacts` | 219 | — |
| `core.anima.inbox` | 1001 | — |
| `core.anima.inbox_overflow` | 100 | — |
| `core.anima.lifecycle` | 1662 | — |
| `core.anima.messaging` | 1426 | — |
| `core.anima.response_normalize` | 141 | — |
| `core.anima.roster` | 83 | — |
| `core.anima.settings_store` | 98 | — |
| `core.anima.skills_check` | 15 | — |

## `core.auth`

사용자 인증, 세션 및 인증 정보 관리.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.auth` | 7 | — |
| `core.auth.manager` | 192 | — |
| `core.auth.models` | 45 | — |

## `core.channels`

Slack, Discord, Chatwork의 공통 전송 클라이언트 및 토큰 해석.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.channels` | 1 | 외부 통신 채널을 위한 공유 클라이언트 및 토큰 해석. |
| `core.channels.chatwork` | 33 | 중앙 Chatwork 메시지 전송 클라이언트. |
| `core.channels.discord` | 346 | 채널 메시징을 위한 중앙 Discord REST API v10 클라이언트. |
| `core.channels.slack` | 97 | 중앙 Slack Web API 및 인커밍 웹훅 전송 클라이언트. |
| `core.channels.tokens` | 102 | Anima별 외부 채널 토큰 해석. |

## `core.config`

애플리케이션 설정 스키마, 로드, 검증, 마이그레이션.

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.config` | 42 | — |
| `core.config.anima_registry` | 313 | config.json에서 Anima 등록: 등록, 등록 해제, 이름 변경. |
| `core.config.env_slots` | 92 | — |
| `core.config.file_access_policy` | 569 | — |
| `core.config.global_permissions` | 259 | — |
| `core.config.helper_models` | 499 | — |
| `core.config.io` | 317 | 설정 I/O: 싱글턴 캐시, 로드 및 저장. |
| `core.config.local_llm` | 69 | 로컬 Ollama 기반 모델 기본값 및 역할 프리셋을 위한 헬퍼. |
| `core.config.migrate` | 201 | — |
| `core.config.model_catalog` | 197 | 정적 모델 카탈로그 및 요청별 모델 재정의 검증. |
| `core.config.model_config` | 885 | 모델 설정 해석: load_model_config, 패널티, max_tokens. |
| `core.config.model_discovery` | 581 | 설치된 CLI에서 “모드 + 모델” 카탈로그 동적 탐색. |
| `core.config.model_mode` | 458 | 표준 S/C/D/G/X/Y/A 모드에 대한 모델 실행 모드 해석. |
| `core.config.models` | 100 | 중앙 설정 모듈 — 분할된 모듈을 다시 내보내는 파사드. |
| `core.config.ops` | 203 | AnimaWorks 설정을 읽고 업데이트하는 애플리케이션 작업. |
| `core.config.resolver` | 160 | 설정 해석: status.json와 anima_defaults 병합. |
| `core.config.schemas` | 1540 | AnimaWorks용 Pydantic 설정 스키마. |
| `core.config.vault` | 461 | PyNaCl SealedBox 암호화를 사용하는 자격 증명 보관소. |

## `core.enclave`

격리된 엔클레이브 모드의 설정 모델과 시작 시 보안 가드.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.enclave` | 54 | 전용 소켓에 바인딩되는 격리된 런타임 인스턴스인 엔클레이브 모드. |
| `core.enclave.config` | 207 | 엔클레이브 모드의 설정 모델. |
| `core.enclave.egress.audit` | 62 | 이그레스 파이프라인의 감사 로그. |
| `core.enclave.egress.config` | 113 | 이그레스 파이프라인의 설정 모델. |
| `core.enclave.egress.fs` | 38 | 엔클레이브 file/directory 권한을 적용하는 간단한 파일 시스템 도우미. |
| `core.enclave.egress.masker.dispatch` | 40 | 내장 마스커의 프로필 디스패치. |
| `core.enclave.egress.masker.facts` | 162 | 레코드 사실의 규칙 기반 마스킹. |
| `core.enclave.egress.masker.log_pii` | 115 | log/audit PII 마스킹. |
| `core.enclave.egress.masker.ner` | 97 | MeCab(fugashi + IPADIC)을 사용한 개체명 인식 마스킹. |
| `core.enclave.egress.models` | 57 | 이그레스 파이프라인의 데이터 구조. |
| `core.enclave.egress.pipeline` | 96 | 이그레스 파이프라인: 설정된 단계를 적용하고 오류가 발생하면 모두 차단합니다. |
| `core.enclave.egress.stages` | 236 | 이그레스 파이프라인의 단계 구현. |
| `core.enclave.gateway` | 521 | 게이트웨이: 엔클레이브 인스턴스의 인그레스 지점. |
| `core.enclave.gateway_server` | 216 | 엔클레이브 게이트웨이의 수명 주기 및 Unix 소켓 연결. |
| `core.enclave.guards` | 273 | 엔클레이브 모드의 시작 시 가드. |
| `core.enclave.jobs` | 254 | 엔클레이브 게이트웨이의 영구 쿼리 작업. |
| `core.enclave.laravel_crypt` | 115 | Laravel ``Crypt::encryptString`` 페이로드 복호화 도우미. |
| `core.enclave.ops` | 78 | 엔클레이브 상태 확인 및 감사 요약을 위한 운영 도우미. |
| `core.enclave.raw_store` | 150 | 엔클레이브 도구가 읽는 수정되지 않은 결과를 위한 비공개 저장소. |
| `core.enclave.secrets` | 128 | 엔클레이브 자격 증명 저장소에서 비밀 정보 읽기. |
| `core.enclave.ssm_tunnel` | 371 | 엔클레이브 SQL 소스를 위한 SSM Session Manager 포트 포워딩 터널. |

## `core.enclave.egress`

—

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.enclave.egress` | 29 | 이그레스 파이프라인: 엔클레이브를 벗어나기 전에 발신 답변을 마스킹. |

## `core.enclave.egress.masker`

—

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.enclave.egress.masker` | 19 | 이그레스 파이프라인용 기본 마스커. |

## `core.execution`

도구 실행, 명령 실행, 안전 제어.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution` | 60 | — |
| `core.execution._shortterm_handoff（非公開）` | 145 | — |
| `core.execution._streaming（非公開）` | 303 | — |
| `core.execution._tool_summary（非公開）` | 99 | — |
| `core.execution.base` | 915 | — |
| `core.execution.busy_probe` | 68 | Congestion probe for self-hosted fallback models (vLLM ``/metrics``). |
| `core.execution.cli_stream` | 328 | — |
| `core.execution.engine_base` | 76 | — |
| `core.execution.engines.agy.executor` | 795 | — |
| `core.execution.engines.claude._sdk_hooks（非公開）` | 652 | — |
| `core.execution.engines.claude._sdk_interrupt（非公開）` | 112 | — |
| `core.execution.engines.claude._sdk_options（非公開）` | 567 | — |
| `core.execution.engines.claude._sdk_patch（非公開）` | 261 | — |
| `core.execution.engines.claude._sdk_security（非公開）` | 304 | — |
| `core.execution.engines.claude._sdk_session（非公開）` | 586 | — |
| `core.execution.engines.claude._sdk_stream（非公開）` | 503 | — |
| `core.execution.engines.claude.executor` | 867 | — |
| `core.execution.engines.codex.events` | 664 | — |
| `core.execution.engines.codex.executor` | 844 | — |
| `core.execution.engines.codex.setup` | 946 | — |
| `core.execution.engines.cursor.executor` | 655 | — |
| `core.execution.engines.gemini.executor` | 461 | — |
| `core.execution.engines.grok.executor` | 1033 | — |
| `core.execution.engines.litellm._litellm_context（非公開）` | 526 | — |
| `core.execution.engines.litellm._litellm_tools（非公開）` | 404 | — |
| `core.execution.engines.litellm._llm_call（非公開）` | 326 | — |
| `core.execution.engines.litellm.executor` | 1068 | — |
| `core.execution.events` | 255 | — |
| `core.execution.fallback_activity` | 290 | Activity-log integration for ephemeral runtime model fallback. |
| `core.execution.github_identity` | 162 | GitHub identity resolution for executor environments. |
| `core.execution.loop_guards` | 411 | In-loop guard mechanisms for the self-hosted execution loops (Mode A/B). |
| `core.execution.mcp_env` | 39 | — |
| `core.execution.process_runner` | 184 | — |
| `core.execution.reminder` | 128 | — |
| `core.execution.session.engine_session` | 160 | — |
| `core.execution.session.session_context` | 114 | — |
| `core.execution.session.session_ids` | 116 | — |
| `core.execution.session.session_store` | 139 | — |
| `core.execution.session.session_types` | 73 | — |
| `core.execution.tool_evidence` | 182 | — |
| `core.execution.watchdog` | 139 | — |
| `core.execution.workspace_env` | 71 | — |

## `core.execution.engines`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines` | 6 | — |

## `core.execution.engines.agy`

—

| 모듈 | 행 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.execution.engines.agy` | 6 | — |

## `core.execution.engines.claude`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.claude` | 18 | — |

## `core.execution.engines.codex`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.codex` | 6 | — |

## `core.execution.engines.cursor`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.cursor` | 6 | — |

## `core.execution.engines.gemini`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.gemini` | 6 | — |

## `core.execution.engines.grok`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.grok` | 6 | — |

## `core.execution.engines.litellm`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.engines.litellm` | 6 | — |

## `core.execution.session`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.execution.session` | 7 | — |

## `core.i18n`

번역 카탈로그와 언어 선택.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.i18n` | 135 | 런타임 문자열을 위한 경량 i18n 지원. |
| `core.i18n.strings.communication` | 46 | 도메인별 i18n 문자열. |
| `core.i18n.strings.company` | 14 | 회사 관리를 위한 현지화된 문자열. |
| `core.i18n.strings.config` | 840 | 도메인별 i18n 문자열. |
| `core.i18n.strings.discord` | 28 | — |
| `core.i18n.strings.execution` | 221 | 도메인별 i18n 문자열. |
| `core.i18n.strings.handler` | 416 | 도메인별 i18n 문자열(핸들러 1부). |
| `core.i18n.strings.handler_ext` | 393 | 도메인별 i18n 문자열(핸들러 2부). |
| `core.i18n.strings.lifecycle` | 104 | 도메인별 i18n 문자열. |
| `core.i18n.strings.memory` | 418 | 도메인별 i18n 문자열. |
| `core.i18n.strings.migrate` | 99 | — |
| `core.i18n.strings.misc` | 527 | 도메인별 i18n 문자열. |
| `core.i18n.strings.misc_routes` | 21 | 도메인별 i18n 문자열(레거시 경로 모듈). |
| `core.i18n.strings.models` | 70 | 모델 검사를 위한 현지화된 CLI 메시지. |
| `core.i18n.strings.phone` | 127 | — |
| `core.i18n.strings.room_manager` | 29 | 회의실 관리자를 위한 i18n 문자열. |
| `core.i18n.strings.server` | 341 | 도메인별 i18n 문자열. |
| `core.i18n.strings.supervisor` | 140 | 도메인별 i18n 문자열. |
| `core.i18n.strings.tmp` | 74 | — |
| `core.i18n.strings.tooling` | 175 | 도메인별 i18n 문자열(도구 프롬프트 및 도구 기능). |
| `core.i18n.strings.tooling_schema` | 540 | 도메인별 i18n 문자열(schema.*). |
| `core.i18n.strings.tooling_schema_ext` | 167 | 도메인별 i18n 문자열(schema.* 2부). |
| `core.i18n.strings.zoom` | 26 | — |

## `core.i18n.strings`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.i18n.strings` | 65 | 모든 도메인 문자열 모듈을 하나의 딕셔너리로 병합. |

## `core.infra`

로그, 데이터베이스, 캐시 등의 기반 기능.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.infra` | 6 | — |
| `core.infra.event_export` | 389 | — |
| `core.infra.execution_sdk_preflight` | 126 | — |
| `core.infra.gpu` | 173 | — |
| `core.infra.logging_config` | 530 | AnimaWorks의 중앙 집중식 로그 설정 |
| `core.infra.runtime_init` | 429 | 최초 실행 초기화: 템플릿을 런타임 데이터 디렉터리에 복사 |
| `core.infra.startup_progress` | 191 | — |
| `core.infra.tmp_cleanup` | 254 | — |

## `core.integrations`

외부 서비스 연동 및 animaworks-tool 구현.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.integrations` | 79 | 핵심 연동 도구 검색 및 레지스트리. |
| `core.integrations._anima_icon_url（非公開）` | 322 | Anima 아이콘 URL 확인 — 대시보드, 발신, Slack, 알림, 도구 등. |
| `core.integrations._async_compat（非公開）` | 41 | 동기식 HTTP 클라이언트를 사용하는 도구를 위한 비동기 호환성 헬퍼. |
| `core.integrations._base（非公開）` | 200 | — |
| `core.integrations._cache（非公開）` | 172 | 통신 도구용 공유 SQLite 메시지 캐시 기본 클래스. |
| `core.integrations._chatwork_cache（非公開）` | 324 | Chatwork 오프라인 검색 및 미응답 감지를 위한 SQLite 메시지 캐시. |
| `core.integrations._chatwork_client（非公開）` | 235 | Chatwork v2 API용 HTTP 클라이언트. |
| `core.integrations._chatwork_cli（非公開）` | 633 | Chatwork 도구용 독립 실행형 CLI 진입점. |
| `core.integrations._chatwork_identity（非公開）` | 76 | Chatwork 신원 및 위임 확인. |
| `core.integrations._chatwork_markdown（非公開）` | 162 | Markdown을 Chatwork 형식으로 변환하는 유틸리티. |
| `core.integrations._comm_cli（非公開）` | 64 | — |
| `core.integrations._discord_cache（非公開）` | 293 | Discord용 SQLite 메시지 캐시(오프라인 검색, 동기화 상태). |
| `core.integrations._discord_client（非公開）` | 47 | 하위 호환성을 위한 Discord 클라이언트 가져오기 경로. |
| `core.integrations._discord_cli（非公開）` | 297 | Discord 도구용 독립 실행형 CLI 진입점. |
| `core.integrations._discord_markdown（非公開）` | 138 | Discord 마크업 헬퍼: 일반 텍스트 정리 및 길이 제한. |
| `core.integrations._google_auth（非公開）` | 175 | Google 연동용 공유 OAuth2 자격 증명 처리. |
| `core.integrations._image_clients（非公開）` | 93 | image/3D 생성을 위한 API 클라이언트 및 공유 상수. |
| `core.integrations._image_cli（非公開）` | 371 | ``animaworks-tool image_gen``용 CLI 진입점. |
| `core.integrations._image_glb（非公開）` | 473 | GLB/FBX 에셋 변환, 최적화 및 압축. |
| `core.integrations._image_pipeline（非公開）` | 939 | ImageGenPipeline – 전체 캐릭터 에셋 생성을 조율합니다. |
| `core.integrations._image_pixel（非公開）` | 624 | 승인된 기본 시트에서 Pixel Workspace 캐릭터 시트를 생성합니다. |
| `core.integrations._image_schemas（非公開）` | 42 | 이미지 생성을 위한 도구 스키마 및 CLI 안내. |
| `core.integrations._retry（非公開）` | 170 | AnimaWorks 도구용 공유 retry/backoff 유틸리티. |
| `core.integrations._slack_cache（非公開）` | 435 | Slack용 SQLite 메시지 캐시(오프라인 검색, 미응답 감지). |
| `core.integrations._slack_client（非公開）` | 308 | 속도 제한 재시도 및 페이지네이션을 지원하는 Slack Web API 클라이언트. |
| `core.integrations._slack_cli（非公開）` | 308 | Slack 도구용 독립 실행형 CLI 진입점. |
| `core.integrations._slack_markdown（非公開）` | 240 | Slack 마크다운 변환 및 서식 지정 유틸리티. |
| `core.integrations.aws_collector` | 408 | AnimaWorks AWS 수집기 도구 — ECS 상태, CloudWatch 로그 및 메트릭. |
| `core.integrations.call_human` | 113 | — |
| `core.integrations.chatwork` | 281 | AnimaWorks용 Chatwork 연동. |
| `core.integrations.discord` | 284 | AnimaWorks용 Discord 연동. |
| `core.integrations.enclave` | 554 | enclave ask 작업 제출 및 관리를 위한 호스트 측 도구. |
| `core.integrations.enclave_aws` | 866 | enclave 런타임용 읽기 전용 AWS 데이터 도구. |
| `core.integrations.enclave_progress` | 111 | anima가 enclave ask 작업에 중간 진행 상황을 기록하는 도구. |
| `core.integrations.enclave_records` | 222 | enclave 런타임 내부에서 설정된 JSONL 데이터 세트를 읽습니다. |
| `core.integrations.enclave_sql` | 397 | enclave 런타임용 읽기 전용 MySQL 쿼리 도구. |
| `core.integrations.github` | 418 | AnimaWorks GitHub 도구 — gh CLI 래퍼. |
| `core.integrations.gmail` | 1254 | AnimaWorks Gmail 도구 -- Gmail API 직접 액세스. |
| `core.integrations.google_calendar` | 615 | — |
| `core.integrations.google_sheets` | 470 | — |
| `core.integrations.google_tasks` | 445 | AnimaWorks Google Tasks 도구 -- Google Tasks API 액세스. |
| `core.integrations.image.atlascloud` | 156 | 캐릭터 이미지 및 참조 편집을 위한 선택적 Atlas Cloud 백엔드. |
| `core.integrations.image.codex` | 370 | Codex CLI 이미지 생성 클라이언트(로컬 codex를 통한 image_gen 도구). |
| `core.integrations.image.constants` | 57 | image/3D 생성을 위한 URL 상수, 시간 초과 및 실행 프로필. |
| `core.integrations.image.diffusers_local` | 905 | 로컬 Diffusers 기반 이미지 생성 헬퍼. |
| `core.integrations.image.fal` | 243 | Fal.ai Flux Kontext 및 Flux Pro 텍스트-이미지 API 클라이언트. |
| `core.integrations.image.meshy` | 311 | Meshy 이미지-3D 변환, 리깅 및 애니메이션 API 클라이언트. |
| `core.integrations.image.novelai` | 193 | 애니메이션 전신 이미지 생성을 위한 NovelAI V4.5 API 클라이언트. |
| `core.integrations.image.prompts` | 279 | 상반신, 치비 및 표정 변형을 위한 프롬프트 상수. |
| `core.integrations.image.utils` | 169 | image/3D 생성 클라이언트용 공유 유틸리티. |
| `core.integrations.image_gen` | 425 | AnimaWorks용 캐릭터 이미지 및 3D 모델 생성 도구. |
| `core.integrations.local_llm` | 552 | AnimaWorks 로컬 LLM 도구 -- Ollama API 클라이언트. |
| `core.integrations.notion` | 862 | AnimaWorks용 Notion 연동. |
| `core.integrations.slack` | 340 | AnimaWorks용 Slack 연동. |
| `core.integrations.transcribe` | 429 | AnimaWorks 음성 전사 도구 -- LLM 보정 기능을 갖춘 Whisper 음성-텍스트 변환. |
| `core.integrations.web_search` | 408 | AnimaWorks용 웹 검색 도구. |
| `core.integrations.x_search` | 348 | AnimaWorks용 X (Twitter) 검색 도구. |

## `core.integrations.image`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.integrations.image` | 85 | 이미지 및 3D 생성 API 클라이언트와 공통 상수. |

## `core.lifecycle`

anima의 시작, 종료, 초기화 라이프사이클.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.lifecycle` | 17 | — |
| `core.lifecycle.knowledge_correction` | 127 | — |
| `core.lifecycle.system_consolidation` | 308 | — |

## `core.llm`

LLM 오류 분류, 속도 제한, 재시도 공통 기능.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.llm` | 1 | LLM 관련 핵심 유틸리티. |
| `core.llm.guard.backoff` | 42 | 협력형 LLM 재시도를 위한 백오프 타이밍 도우미. |
| `core.llm.guard.error_classifier` | 805 | 협력형 복구를 위한 중앙 집중식 LLM API 오류 분류. |
| `core.llm.guard.rate_guard` | 341 | 프로세스 간 LLM 속도 제한 가드(전체 시스템 회로 차단기). |
| `core.llm.helper_completion` | 145 | — |
| `core.llm.oneshot` | 943 | 메모리 관리 모듈을 위한 공유 LLM 도우미 유틸리티. |

## `core.llm.guard`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.llm.guard` | 1 | Provider-independent LLM retry and failure guards. |

## `core.mcp`

Model Context Protocol 서버 및 클라이언트.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.mcp` | 0 | — |
| `core.mcp.server` | 736 | — |

## `core.memory`

대화 및 에피소드 기억의 저장, 검색, 정리.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.memory` | 20 | — |
| `core.memory._llm_parse（非公開）` | 185 | 메모리 파이프라인에서 공유하는 LLM 출력 파싱 헬퍼. |
| `core.memory.config_reader` | 31 | — |
| `core.memory.conversation.compression` | 322 | 대화 기억의 압축 로직. |
| `core.memory.conversation.finalize` | 493 | 대화 기억의 세션 마무리 처리. |
| `core.memory.conversation.memory` | 349 | 대화 기억(대화 기억 / 작업 메모리) 관리. |
| `core.memory.conversation.models` | 149 | 대화 기억용 데이터 클래스 및 상수. |
| `core.memory.conversation.prompt` | 270 | 대화 기억용 프롬프트 빌드 함수. |
| `core.memory.conversation.shortterm` | 313 | 단기 기억 관리. |
| `core.memory.conversation.state_update` | 49 | 대화 기억 마무리 처리를 위한 상태 업데이트 함수. |
| `core.memory.conversation.streaming_journal` | 473 | — |
| `core.memory.facts.chunking` | 91 | — |
| `core.memory.facts.config` | 85 | — |
| `core.memory.facts.connections` | 544 | — |
| `core.memory.facts.entity_index` | 452 | — |
| `core.memory.facts.extraction` | 529 | — |
| `core.memory.facts.extractor` | 431 | LLM 기반 엔터티 및 사실 추출 파이프라인. |
| `core.memory.facts.invalidation` | 515 | — |
| `core.memory.facts.invalidation_llm` | 154 | — |
| `core.memory.facts.live` | 578 | — |
| `core.memory.facts.observability` | 41 | — |
| `core.memory.facts.ontology` | 247 | 엔터티 / 사실 추출 결과용 Pydantic 모델. |
| `core.memory.facts.prompts.en` | 133 | 엔터티 / 사실 추출용 영어 프롬프트. |
| `core.memory.facts.prompts.ja` | 135 | 엔터티 / 사실 추출용 일본어 프롬프트. |
| `core.memory.facts.store` | 460 | — |
| `core.memory.frontmatter` | 430 | — |
| `core.memory.io` | 59 | — |
| `core.memory.maintenance.activity_compaction` | 479 | — |
| `core.memory.maintenance.background_review` | 565 | — |
| `core.memory.maintenance.consolidation` | 1190 | — |
| `core.memory.maintenance.cron_logger` | 159 | — |
| `core.memory.maintenance.cron_noop` | 241 | — |
| `core.memory.maintenance.distillation` | 548 | — |
| `core.memory.maintenance.forgetting` | 588 | — |
| `core.memory.maintenance.housekeeping` | 1964 | — |
| `core.memory.maintenance.hygiene` | 75 | — |
| `core.memory.maintenance.reconsolidation` | 669 | — |
| `core.memory.maintenance.resolution_tracker` | 60 | — |
| `core.memory.manager` | 683 | — |
| `core.memory.peer_profiles` | 37 | — |
| `core.memory.priming.channel_a` | 70 | — |
| `core.memory.priming.channel_b` | 534 | — |
| `core.memory.priming.channel_c` | 610 | — |
| `core.memory.priming.channel_e` | 205 | — |
| `core.memory.priming.channel_f` | 215 | — |
| `core.memory.priming.channel_g` | 113 | — |
| `core.memory.priming.constants` | 92 | — |
| `core.memory.priming.engine` | 503 | — |
| `core.memory.priming.format` | 124 | — |
| `core.memory.priming.items` | 59 | — |
| `core.memory.priming.outbound` | 142 | — |
| `core.memory.priming.policy` | 25 | — |
| `core.memory.priming.result` | 76 | — |
| `core.memory.priming.utils` | 284 | — |
| `core.memory.rag.cli_access` | 228 | 활성 소유자 또는 서버를 통해 phase3 벡터 저장소에 접근하는 CLI 기능. |
| `core.memory.rag.contextual_header` | 163 | — |
| `core.memory.rag.direct_access` | 24 | — |
| `core.memory.rag.embedding` | 393 | — |
| `core.memory.rag.endpoints` | 79 | — |
| `core.memory.rag.episode_time` | 46 | — |
| `core.memory.rag.exclusion` | 38 | — |
| `core.memory.rag.facts_chunker` | 100 | — |
| `core.memory.rag.index_signature` | 27 | 기존 임베딩 인덱스 서명에 대한 호환성 진단. |
| `core.memory.rag.indexer` | 1593 | — |
| `core.memory.rag.indexer_delete` | 135 | — |
| `core.memory.rag.owner_lock` | 84 | 아니마의 기본 벡터 데이터베이스에 대한 독점 소유권 잠금. |
| `core.memory.rag.repair.detect` | 743 | — |
| `core.memory.rag.repair.rebuild` | 401 | — |
| `core.memory.rag.repair.state` | 191 | RAG 자동 복구를 위한 영구 복구 상태 헬퍼. |
| `core.memory.rag.repair.types` | 27 | — |
| `core.memory.rag.retriever` | 809 | — |
| `core.memory.rag.shared_check_registry` | 140 | — |
| `core.memory.rag.shared_meta` | 162 | — |
| `core.memory.rag.sqlite_health` | 359 | — |
| `core.memory.rag.store` | 783 | — |
| `core.memory.rag.vector_client` | 497 | — |
| `core.memory.rag.vector_ops` | 88 | 벡터 API 요청과 MemoryService 와이어 형식 간 변환. |
| `core.memory.rag.vector_registry` | 160 | — |
| `core.memory.retrieval.bm25` | 1325 | — |
| `core.memory.retrieval.code_index` | 221 | — |
| `core.memory.retrieval.confidence_gate` | 39 | — |
| `core.memory.retrieval.entity` | 192 | — |
| `core.memory.retrieval.pipeline` | 92 | — |
| `core.memory.retrieval.query_expansion` | 496 | — |
| `core.memory.retrieval.rag_search` | 1149 | — |
| `core.memory.retrieval.reranker` | 262 | — |
| `core.memory.retrieval.rrf` | 107 | — |
| `core.memory.retrieval.search_metadata` | 108 | — |
| `core.memory.retrieval.time_expr` | 230 | — |
| `core.memory.retrieval.types` | 38 | — |
| `core.memory.retrieval.unified_search` | 722 | — |
| `core.memory.skill_metadata` | 49 | — |
| `core.memory.state_lock` | 96 | ``state/current_state.md`` 업데이트를 위한 프로세스 안전 잠금. |

## `core.memory.conversation`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.conversation` | 23 | — |

## `core.memory.facts`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.facts` | 37 | — |

## `core.memory.facts.prompts`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.facts.prompts` | 3 | — |

## `core.memory.maintenance`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.maintenance` | 6 | — |

## `core.memory.priming`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.memory.priming` | 54 | 프라이밍 계층 - 자동 기억 회상. |

## `core.memory.rag`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.rag` | 23 | — |

## `core.memory.rag.repair`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.memory.rag.repair` | 31 | — |

## `core.memory.retrieval`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.memory.retrieval` | 1 | 메모리 검색 파이프라인. 모듈을 직접 지정해 임포트하세요. |

## `core.messaging`

Anima 간 및 외부와의 메시지 전달.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.messaging` | 6 | — |
| `core.messaging.board_fanout` | 95 | — |
| `core.messaging.discord_webhooks` | 296 | — |
| `core.messaging.inbox_quiet_window` | 38 | — |
| `core.messaging.meeting_room_store` | 130 | — |
| `core.messaging.messenger` | 1082 | — |
| `core.messaging.outbound` | 408 | — |
| `core.messaging.outbound_auto` | 377 | — |
| `core.messaging.recipient_status` | 79 | — |
| `core.messaging.reply_grants` | 253 | — |
| `core.messaging.sender` | 32 | — |

## `core.migrations`

런타임 데이터 형식의 단계적 마이그레이션.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.migrations` | 1 | Runtime migration framework; import specific modules directly. |
| `core.migrations.registry` | 149 | — |
| `core.migrations.steps` | 1300 | Migration step implementations for AnimaWorks runtime data. |
| `core.migrations.template_sync` | 137 | — |
| `core.migrations.tracker` | 143 | — |

## `core.notification`

알림 생성 및 전달.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.notification` | 40 | — |
| `core.notification.channels.chatwork` | 79 | — |
| `core.notification.channels.discord` | 251 | — |
| `core.notification.channels.line` | 86 | — |
| `core.notification.channels.ntfy` | 87 | — |
| `core.notification.channels.slack` | 261 | — |
| `core.notification.channels.telegram` | 91 | — |
| `core.notification.channels.web` | 65 | — |
| `core.notification.interactive` | 612 | — |
| `core.notification.notifier` | 238 | — |
| `core.notification.reply_routing` | 500 | — |
| `core.notification.slack_names` | 67 | — |

## `core.notification.channels`

—

| 모듈 | 줄 수 | 독스트링 첫 줄 |
|---|---:|---|
| `core.notification.channels` | 7 | — |

## `core.org`

회사, 부서, 역할 등의 조직 모델.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.org` | 6 | — |
| `core.org.company` | 1212 | Company membership and cross-company boundary helpers. |
| `core.org.company_resources` | 86 | — |
| `core.org.hierarchy` | 39 | — |
| `core.org.org_sync` | 482 | — |
| `core.org.workspace` | 232 | — |

## `core.phone`

Twilio 전화 채널의 음성 합성, 통화 상태 및 웹훅 관리.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.phone` | 7 | — |
| `core.phone.alert` | 218 | — |
| `core.phone.audio_store` | 81 | — |
| `core.phone.session` | 66 | — |
| `core.phone.speech` | 83 | — |
| `core.phone.stream_tokens` | 101 | — |
| `core.phone.stream_transport` | 120 | — |
| `core.phone.twilio_client` | 194 | — |
| `core.phone.urgent` | 46 | — |

## `core.platform`

실행 엔진이나 OS별 차이를 흡수하는 연동 계층.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.platform` | 4 | — |
| `core.platform.agy` | 32 | — |
| `core.platform.atomic_io` | 207 | — |
| `core.platform.claude_code` | 194 | — |
| `core.platform.codex` | 245 | — |
| `core.platform.cursor` | 60 | — |
| `core.platform.env` | 91 | — |
| `core.platform.fd_limits` | 60 | — |
| `core.platform.gemini` | 42 | — |
| `core.platform.grok` | 40 | — |
| `core.platform.locks` | 125 | — |
| `core.platform.pid` | 42 | — |
| `core.platform.process` | 327 | — |
| `core.platform.process_role` | 66 | 프로세스 역할 메타데이터를 위한 저수준 환경 접근. |
| `core.platform.processing_lease` | 344 | — |
| `core.platform.scratch` | 99 | — |
| `core.platform.state_writer` | 1014 | — |
| `core.platform.status_store` | 54 | — |
| `core.platform.subprocess_entries` | 23 | — |
| `core.platform.tasks` | 35 | — |

## `core.prompt`

시스템 프롬프트와 컨텍스트 구성.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.prompt` | 1 | Prompt construction package; import specific modules directly. |
| `core.prompt.assembler` | 307 | — |
| `core.prompt.builder` | 1289 | — |
| `core.prompt.context` | 490 | Context window usage tracker. |
| `core.prompt.messaging` | 147 | — |
| `core.prompt.org_context` | 378 | — |
| `core.prompt.sections` | 52 | — |

## `core.runtime`

Anima 메인의 런타임 구성 요소, 프로세스 간 통신 및 작업 실행.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.runtime` | 1 | 개별 Anima 프로세스가 소유하는 런타임 구성 요소. |
| `core.runtime.cron_diagnostics` | 85 | LLM 실행을 낭비하는 cron 일정에 대한 등록 시점 진단. |
| `core.runtime.cron_followup` | 169 | 레거시 및 격리 실행기에서 공유하는 명령 cron 후속 처리 정책. |
| `core.runtime.event_bus` | 88 | Anima 메인 실행기가 내보낸 이벤트를 위한 프로세스 내 버퍼. |
| `core.runtime.inbox_rate_limiter` | 371 | 이벤트 기반 받은 편지함 깨우기 및 지연된 트리거 관리. |
| `core.runtime.ipc` | 508 | 플랫폼별 전송 방식을 통해 JSON Lines를 사용하는 IPC 통신 계층. |
| `core.runtime.ipc_v2` | 414 | Anima 메인과 작업 실행기 사이에서 사용하는 영구 양방향 IPC v2. |
| `core.runtime.memory_service` | 774 | Anima 메인 소유 벡터 메모리 서비스. |
| `core.runtime.process_role` | 25 | AnimaWorks 프로세스 진입점에서 공유하는 프로세스 역할 메타데이터. |
| `core.runtime.runner` | 1409 | Anima 하위 프로세스의 자식 프로세스 진입점. |
| `core.runtime.schedule_parser` | 484 | — |
| `core.runtime.scheduler_manager` | 1058 | 하트비트 및 cron 작업을 위한 APScheduler 관리. |
| `core.runtime.state_writer` | 27 | — |
| `core.runtime.streaming_handler` | 449 | 스트리밍 IPC 메시지 처리기. |
| `core.runtime.task_runner` | 1056 | 일회성 작업 실행기 진입점. |
| `core.runtime.task_runner_supervisor` | 1446 | 일회성 작업 실행기 프로세스의 Anima 메인 측 수명 주기 관리자. |
| `core.runtime.transport` | 240 | IPC server/client 통신용 전송 헬퍼. |

## `core.skills`

스킬 검색, 로드 및 실행 지원.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.skills` | 20 | — |
| `core.skills.activation` | 459 | — |
| `core.skills.activation_render` | 56 | — |
| `core.skills.activation_state` | 89 | — |
| `core.skills.autolearn` | 210 | — |
| `core.skills.autolearn_lifecycle` | 52 | — |
| `core.skills.cron_context` | 251 | — |
| `core.skills.curator` | 555 | — |
| `core.skills.dense` | 226 | — |
| `core.skills.guard` | 430 | — |
| `core.skills.hub` | 551 | — |
| `core.skills.hub_storage` | 63 | — |
| `core.skills.index` | 613 | — |
| `core.skills.ledger` | 404 | — |
| `core.skills.loader` | 176 | — |
| `core.skills.migration._common（非公開）` | 179 | — |
| `core.skills.migration.hermes` | 513 | — |
| `core.skills.migration.hermes_format` | 170 | — |
| `core.skills.migration.openclaw` | 220 | — |
| `core.skills.migration.report` | 125 | — |
| `core.skills.models` | 384 | — |
| `core.skills.pointer_rewriter` | 221 | — |
| `core.skills.policy` | 84 | — |
| `core.skills.probation_promotion` | 163 | — |
| `core.skills.promotion` | 604 | — |
| `core.skills.promotion_approval` | 85 | — |
| `core.skills.promotion_utils` | 148 | — |
| `core.skills.reference_rewriter` | 422 | — |
| `core.skills.router` | 504 | — |
| `core.skills.router_metadata` | 85 | — |
| `core.skills.sources.github` | 152 | — |
| `core.skills.sources.local` | 81 | — |
| `core.skills.sources.url` | 51 | — |
| `core.skills.trust` | 143 | — |
| `core.skills.trust_gate` | 22 | — |
| `core.skills.usage` | 202 | — |

## `core.skills.migration`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.skills.migration` | 14 | — |

## `core.skills.sources`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.skills.sources` | 1 | Skill source adapters; import specific modules directly. |

## `core.tasks`

작업 등록, 상태 관리, 실행 제어.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tasks` | 1 | Task queue, task board, delegated/background task execution and external task sources. |
| `core.tasks.autonomy_report` | 475 | — |
| `core.tasks.background` | 618 | — |
| `core.tasks.board.board_actions` | 260 | — |
| `core.tasks.board.housekeeping` | 188 | — |
| `core.tasks.board.models` | 39 | Pydantic models for the single TaskBoard view (read straight from TaskStore). |
| `core.tasks.board.notices` | 120 | — |
| `core.tasks.board.readiness` | 31 | Read-only boundary between legacy task files and canonical execution. |
| `core.tasks.board.tasks` | 1619 | Durable execution records; the single source of truth for the TaskBoard. |
| `core.tasks.board.view` | 119 | Single TaskBoard view built directly from the canonical TaskStore. |
| `core.tasks.dispatch` | 521 | — |
| `core.tasks.external.collector` | 211 | Multi-source external tasks collector with per-source fault isolation. |
| `core.tasks.external.models` | 47 | Data models for the external tasks snapshot store. |
| `core.tasks.external.sources.chatwork` | 228 | Chatwork external tasks collector (open my-tasks + unreplied To). |
| `core.tasks.external.sources.github` | 182 | GitHub external tasks collector via ``gh`` CLI. |
| `core.tasks.external.sources.gmail` | 124 | Gmail external tasks collector (unread inbox, last 7 days). |
| `core.tasks.external.sources.slack` | 184 | Slack external tasks collector (unreplied mentions via message cache). |
| `core.tasks.external.store` | 51 | Atomic JSON snapshot store for external tasks. |
| `core.tasks.pending_executor` | 1708 | Execute claimed TaskStore work in background lanes. |
| `core.tasks.queue` | 516 | — |
| `core.tasks.waiting` | 81 | — |
| `core.tasks.wake` | 70 | Cross-process wake fan-out for the PendingTaskExecutor. |

## `core.tasks.board`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tasks.board` | 1 | Task board package; import specific modules directly. |

## `core.tasks.external`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tasks.external` | 1 | External task collection package; import specific modules directly. |

## `core.tasks.external.sources`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tasks.external.sources` | 8 | Per-source collectors for external tasks (stubs in this phase). |

## `core.text`

텍스트의 토큰 예상과 예산 내 잘라내기.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.text` | 1 | Shared text processing utilities. |
| `core.text.tokens` | 92 | — |

## `core.tooling`

도구 스키마, 권한, 실행 기반.

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tooling` | 1 | Tooling package; import specific modules to avoid eager handler loading. |
| `core.tooling._handler_protocols（非公開）` | 233 | Structural host protocols for the compositional mixins. |
| `core.tooling.codex_command_hook` | 31 | Core command-policy decision for Codex's PreToolUse hook. |
| `core.tooling.dispatch` | 256 | — |
| `core.tooling.handler` | 911 | — |
| `core.tooling.handler_base` | 335 | — |
| `core.tooling.handler_chat_files` | 87 | — |
| `core.tooling.handler_comms` | 1027 | — |
| `core.tooling.handler_create_anima` | 101 | — |
| `core.tooling.handler_delegation` | 310 | — |
| `core.tooling.handler_exec` | 374 | — |
| `core.tooling.handler_files` | 899 | — |
| `core.tooling.handler_memory` | 1434 | — |
| `core.tooling.handler_org` | 39 | — |
| `core.tooling.handler_org_dashboard` | 203 | — |
| `core.tooling.handler_perms` | 446 | — |
| `core.tooling.handler_skills` | 864 | — |
| `core.tooling.handler_subordinate_control` | 457 | — |
| `core.tooling.handler_workspace` | 279 | — |
| `core.tooling.org_helpers` | 158 | — |
| `core.tooling.permissions` | 344 | — |
| `core.tooling.policy.action_gate` | 188 | — |
| `core.tooling.policy.command_policy` | 496 | — |
| `core.tooling.policy.registry` | 65 | — |
| `core.tooling.policy.schemas.admin` | 221 | — |
| `core.tooling.policy.schemas.builder` | 79 | — |
| `core.tooling.policy.schemas.channel` | 118 | — |
| `core.tooling.policy.schemas.chat_files` | 42 | — |
| `core.tooling.policy.schemas.converters` | 27 | — |
| `core.tooling.policy.schemas.loader` | 101 | — |
| `core.tooling.policy.schemas.memory` | 228 | — |
| `core.tooling.policy.schemas.notification` | 71 | — |
| `core.tooling.policy.schemas.session_todo` | 62 | — |
| `core.tooling.policy.schemas.skill` | 306 | — |
| `core.tooling.policy.schemas.supervisor` | 348 | — |
| `core.tooling.policy.schemas.task` | 204 | — |
| `core.tooling.policy.schemas.workspace` | 46 | — |
| `core.tooling.policy.submit_tasks` | 91 | — |
| `core.tooling.policy.surface` | 259 | — |
| `core.tooling.policy.tool_content` | 42 | — |
| `core.tooling.skill_creator` | 120 | — |
| `core.tooling.skill_promotion_tool` | 176 | — |
| `core.tooling.standalone` | 322 | — |
| `core.tooling.tool_context` | 17 | Shared runtime state passed to ToolHandler mixin delegates. |

## `core.tooling.policy`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tooling.policy` | 1 | Tool schemas and policy primitives shared by execution and tooling. |

## `core.tooling.policy.schemas`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `core.tooling.policy.schemas` | 73 | Canonical tool schema definitions and format converters. |

## `core.usage`

LLM 사용량 및 비용 기록·집계.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.usage` | 1 | 토큰 사용량 계산 및 Anima별 토큰 예산. |
| `core.usage.token_budget` | 73 | — |
| `core.usage.token_usage` | 605 | — |

## `core.voice`

음성 입출력 및 음성 대화.

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `core.voice` | 7 | Voice chat subsystem — STT, TTS, and session orchestration. |
| `core.voice.audio_codec` | 161 | — |
| `core.voice.emotion_style` | 114 | — |
| `core.voice.front` | 385 | Voice front lane — lightweight speech-first chat path via a local LLM. |
| `core.voice.front_conversation` | 694 | Transport-agnostic front-lane conversation and delegation handling. |
| `core.voice.sentence_splitter` | 73 | Japanese-aware sentence splitting for streaming TTS. |
| `core.voice.session` | 1843 | Voice session — STT -> Chat -> TTS orchestration. |
| `core.voice.session_factory` | 66 | — |
| `core.voice.speech_text` | 332 | — |
| `core.voice.stt` | 145 | Voice STT — in-memory PCM transcription via faster-whisper. |
| `core.voice.stt_stream` | 300 | Streaming STT — rolling buffer + LocalAgreement-2 prefix commitment. |
| `core.voice.transport` | 21 | Transport protocol for voice-session output. |
| `core.voice.tts_base` | 63 | TTS abstract base — provider interface and config. |
| `core.voice.tts_elevenlabs` | 133 | ElevenLabs TTS provider — REST API streaming. |
| `core.voice.tts_factory` | 47 | TTS provider factory. |
| `core.voice.tts_gemini` | 220 | Gemini TTS provider — Gemini API ``streamGenerateContent`` (SSE streaming). |
| `core.voice.tts_irodori` | 79 | Irodori-TTS provider — HTTP API. |
| `core.voice.tts_sbv2` | 112 | Style-BERT-VITS2 / AivisSpeech TTS provider. |
| `core.voice.tts_voicevox` | 110 | VOICEVOX TTS provider — Engine HTTP API. |
| `core.voice.turn_detector` | 398 | — |
| `core.voice.voice_config` | 76 | — |

## `server`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `server` | 7 | — |
| `server.app` | 1517 | — |
| `server.enclave_delivery` | 327 | Collect completed enclave ask jobs and deliver them to host inboxes. |
| `server.events` | 56 | — |
| `server.internal_auth` | 232 | — |
| `server.localhost` | 86 | — |
| `server.reload_manager` | 115 | — |
| `server.room_manager` | 542 | Meeting room lifecycle, orchestration, and minutes generation. |
| `server.stream_registry` | 489 | — |
| `server.websocket` | 165 | — |

## `server.gateways`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `server.gateways` | 1 | Inbound chat/meeting gateways (Slack, Discord, Zoom, GitHub) run by the server. |
| `server.gateways.discord_channel_sync` | 278 | — |
| `server.gateways.discord_gateway` | 727 | — |
| `server.gateways.github_gateway` | 534 | — |
| `server.gateways.slack_channel_sync` | 505 | — |
| `server.gateways.slack_interactive` | 239 | — |
| `server.gateways.slack_socket` | 1145 | — |
| `server.gateways.zoom_gateway` | 700 | — |

## `server.routes`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `server.routes` | 63 | — |
| `server.routes.animas` | 1268 | — |
| `server.routes.approve` | 89 | — |
| `server.routes.assets` | 1737 | — |
| `server.routes.attachment_preview` | 741 | — |
| `server.routes.auth` | 133 | — |
| `server.routes.channels` | 496 | — |
| `server.routes.chat` | 459 | — |
| `server.routes.chat_chunk_handler` | 289 | — |
| `server.routes.chat_emotion` | 8 | — |
| `server.routes.chat_files` | 99 | — |
| `server.routes.chat_images` | 80 | — |
| `server.routes.chat_models` | 60 | — |
| `server.routes.chat_producer` | 346 | — |
| `server.routes.chat_resume` | 114 | — |
| `server.routes.chat_ui_state` | 101 | — |
| `server.routes.chat_ws_effects` | 55 | — |
| `server.routes.config_routes` | 450 | — |
| `server.routes.external_tasks` | 261 | — |
| `server.routes.internal` | 1537 | — |
| `server.routes.logs_routes` | 217 | — |
| `server.routes.media_proxy` | 186 | — |
| `server.routes.memory_routes` | 531 | — |
| `server.routes.phone` | 404 | — |
| `server.routes.phone_stream` | 258 | — |
| `server.routes.room` | 443 | Meeting room API routes with SSE streaming. |
| `server.routes.sessions` | 734 | — |
| `server.routes.setup` | 603 | — |
| `server.routes.skills` | 132 | — |
| `server.routes.system` | 1217 | — |
| `server.routes.taskboard` | 237 | — |
| `server.routes.usage_routes` | 844 | — |
| `server.routes.users` | 280 | — |
| `server.routes.voice` | 223 | Voice chat WebSocket endpoint; per-Anima status.json settings use shared helpers. |
| `server.routes.webhooks` | 503 | — |
| `server.routes.websocket_route` | 46 | — |

## `server.services`

—

| 모듈 | 줄 수 | docstring 첫 줄 |
|---|---:|---|
| `server.services` | 5 | Server application services. |
| `server.services.anima_admin` | 7 | — |

## `server.supervisor`

—

| 모듈 | 행 수 | docstring 첫 줄 |
|---|---:|---|
| `server.supervisor` | 23 | Server-level supervision APIs for managing Anima processes. |
| `server.supervisor._manager_protocols（非公開）` | 85 | Structural host protocols for the compositional mixins. |
| `server.supervisor._mgr_health（非公開）` | 470 | Health check mixin for ProcessSupervisor. |
| `server.supervisor._mgr_rag_repair（非公開）` | 268 | Supervised RAG repair mixin for ProcessSupervisor. |
| `server.supervisor._mgr_reconcile（非公開）` | 320 | Reconciliation mixin for ProcessSupervisor. |
| `server.supervisor._mgr_scheduler（非公開）` | 1281 | System scheduler mixin for ProcessSupervisor. |
| `server.supervisor.activity_schedule` | 100 | — |
| `server.supervisor.auto_updater` | 198 | — |
| `server.supervisor.manager` | 1275 | Process Supervisor - Manages lifecycle of Anima child processes. |
| `server.supervisor.process_handle` | 768 | Process handle for managing child Anima processes. |
| `server.supervisor.restart_state` | 169 | Unified restart state machine for ProcessSupervisor. |
