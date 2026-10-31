<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/execution.md -->
<!-- i18n: source-sha256=dc706c803aa27d2b8b296c056e63e9b928e46ad6c645d16d576bde53c3d44278 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: b304b7dc

# 모델 실행

모델 이름과 per-anima 설정에서 실행 모드를 해결하고, 엔진별 SDK, CLI 또는 API 호출에 전달한다. S, C, X, D, G는 전용 엔진을 사용하고, A는 LiteLLM을 통해 API 모델을 실행한다.

## 실행 모드

| 모드 | 주요 모델 이름 패턴 | 엔진 | 인증 | 도구 실행 |
|---|---|---|---|---|
| S | `claude-*` | Claude Agent SDK | Claude 구독, API key, Bedrock 또는 Vertex AI 인증 | SDK의 내장 기능과 MCP를 활용하고, AnimaWorks의 tool handler와 연결한다. |
| C | `codex/*`, `openai-codex/*` | Codex SDK / CLI | Codex CLI 로그인 정보 또는 설정된 provider credential | Codex 측 실행 기능을 사용하고, 공통 이벤트와 tool 기록으로 변환한다. |
| X | `grok/*` | Grok Build CLI | Grok CLI 인증 | CLI의 ACP stream을 통해 이벤트와 실행 결과를 공통화한다. |
| D | `cursor/*` | Cursor Agent CLI | Cursor CLI 로그인 정보 | CLI의 session과 tool 실행을 활용한다. |
| G | `gemini/*` | Gemini CLI | Google 측 CLI 인증 | CLI 실행 결과를 공통 이벤트로 처리한다. |
| A | `openai/*`, `azure/*`, `bedrock/*`, `google/*`, `vertex_ai/*` 등 | LiteLLM loop | provider의 API key 또는 cloud credential | LiteLLM의 function/tool call을 AnimaWorks의 handler로 실행한다. |

모델 이름에서의 해결 순서는 `status.json`의 명시적 per-anima 지정, `models.json`, `config.json`의 모델별 fallback, `core/config/model_mode.py`의 기본 패턴이다. 어느 것에도 일치하지 않는 모델은 A로 해결된다. 개별 설정 항목은 [설정 참조](../reference/config.md)를 참조한다.

## 공통 실행 계층과 장애 시 동작

각 엔진의 구현은 `core/execution/engines/`에 위치한다. 공통 계층의 `core/execution/events.py`, `core/execution/session/session_store.py`, `process_runner.py`, `watchdog.py`, `tool_evidence.py`, `cli_stream.py`은 이벤트, 세션 저장, 프로세스 제어, tool evidence를 공통화한다. 오류 분류, 프로세스 종료, 바이너리 탐색, `clear_session`, 응답·오류 판정도 중간 계층에서 처리한다. D와 G도 이 구성에 포함된다.

watchdog은 engine event가 1200초 동안 도착하지 않으면 idle로 판정한다. 전체 실행 시간을 재는 timeout이 아니다. rate limit이나 provider 과부하 정보는 `llm_rate_guard`에 provider family별로 공유되며, 다른 Anima가 같은 provider로 연속 요청하는 것을 억제한다. 이 guard는 읽기·쓰기에 실패해도 일반적인 실행을 멈추지 않는 설계이다.

주 모델에 더해 `fallback_model` 또는 `fallback_models`을 지정할 수 있다. 인증·rate limit 등의 분류 결과와 fallback 설정에 기반하여 다음 후보로 전환한다. heartbeat와 cron에는 `background_model`, `background_credential`, `background_thinking_effort`을 지정할 수 있다.

## 컨텍스트 관리

컨텍스트 압축 방법은 엔진마다 다르다. S는 대화의 기준량을 저장하여 변화를 추적하고, SDK의 압축 기능과 idle 시 압축을 활용한다. C는 임계값에서 현재 thread를 폐기하고 새 thread를 시작한다. X와 D는 재개 turn 수를 기록하고, 10 turn에서 session을 로테이션한다. A는 입력 상한 접근이나 overflow 시 대화를 요약하고, 단축된 기록으로 재시도한다. G는 전용 CLI의 session 동작을 따르고, 공통의 turn 제한 session 저장은 수행하지 않는다.

per-anima의 context threshold 기본값은 0.50, absolute ceiling은 0.75이다. 작업 실행용 compaction token threshold는 0이 기본이고, 상한 횟수 기본값은 6이다. 모델별 compaction threshold는 `models.json`에서도 지정할 수 있다.

## 설계 판단

- **에이전트 루프는 직접 만들지 않는다.** tool use의 반복은 엔진(Claude Agent SDK, Codex, 각 CLI, LiteLLM)에 맡기고, AnimaWorks는 그 전후의 공통 처리(prompt 조립, 권한, 세션, 오류 분류, 기록)만 담당한다.
