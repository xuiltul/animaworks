<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/priming.md -->
<!-- i18n: source-sha256=d5a04a5c993d7ed7bc3ff3758c23f89175249339ec84279d9c33e9ffd1b807f8 generated=2026-11-06 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 자동 회상 (Priming)

자동 회상은 입력을 받은 실행 경로에서 관련된 기억의 단서를 획득하여 LLM의 컨텍스트에 추가하는 처리이다. 구성은 `core/memory/priming/engine.py`의 `PrimingEngine`, 기본 프로필은 `core/config/schemas.py`의 `PrimingConfig`에서 정의된다.

## 프로필 및 채널

`PrimingConfig.profile`의 기본값은 `compact`이다. Anima별 `status.json`에 `priming_profile`을 `compact` 또는 `full`로 지정하면 해당 Anima의 설정을 덮어쓸 수 있다. 채널의 개요는 다음과 같다.

| 채널 | 가져오는 항목 | `compact` | `full` |
|---|---|---:|---:|
| A | 발신자 프로필 | ○ | ○ |
| B | 활동 로그와 공유 채널의 최근 동향 | 백그라운드 트리거에서 조건부 | ○ |
| C0 | 상주 후보·중요 지식 포인터 | ○ | ○ |
| C | 관련 지식 검색 | 조건부 | ○ |
| E | 미완료 작업 | ○ | ○ |
| F | 과거 에피소드 | — | ○ |
| G | 독립 채널이 아님. 에피소드 검색에 수반되는 그래프 유래 후보 | — | ○ (F 검색 내) |
| 보조 | 최근 전송 내용, 사람에게 전송되지 않은 알림 | ○ | ○ |

`compact`에서는 채널 A·E·C0와 보조 정보를 병렬로 수집한다. C 검색은 `channel`가 `chat` 또는 `task`이거나, `intent`가 `question`, `request`, `delegation`인 경우에 수행한다. B는 `heartbeat`·`inbox`·`cron`의 백그라운드 트리거에서 `compact_background_recall_enabled`가 활성화되어 있고 최근 활동의 건수·토큰 상한이 양수인 경우 가져온다. F도 백그라운드 트리거에서 쿼리가 있고 에피소드의 건수·토큰 상한이 양수인 경우 가져온다. G는 독립적인 Priming 채널이 아니라 F 검색 내에서 활용될 수 있다.

## C0와 `always_prime`

지식 파일에 `always_prime: true`을 설정하면 RAG 인덱스에서 자동 회상의 상주 후보로 취급된다. `compact`는 C0를 `resident_only=True`으로 호출하고, 이 명시적 opt-in 후보에서 일반 지식 포인터를 최대 3건 선택한다. 현재의 `compact` 경로는 입력 쿼리와의 일치를 확인하지 않고, 상주 후보를 업데이트 날짜순으로 선택한다. ACTION-RULE로 취급되는 후보는 본문에서 주입되는 경우가 있다. `[IMPORTANT]`라는 이유만으로 무관한 지식을 상주 주입하는 것은 아니다. `full`은 상주 후보에 더해 쿼리와 관련된 중요 지식도 검색한다.

## 검색 방침

검색 방침은 `core/memory/retrieval/unified_search.py`의 `TRIGGER_POLICIES`에서 정의된다. `scope="all"`의 검색에서는 trigger에 따라 scope, 후보 수, rerank의 유무가 달라진다. C와 F는 개별 scope를 지정하므로, 그 scope를 사용하면서 trigger별 후보 수와 rerank 방침이 적용된다.

| trigger | 주요 검색 scope | rerank |
|---|---|---:|
| `chat` | facts, episodes, knowledge, procedures, activity log | 유효 |
| `inbox` | facts, episodes, activity log | 유효 |
| `heartbeat` | episodes, activity log | 무효 |
| `task` | facts, procedures, knowledge | 유효 |
| `cron` | facts, episodes, knowledge, activity log | 유효 |
| `tool` | 도구 검색용 scope | 유효 |

개별 검색 단계와 `search_memory`의 scope는 [의도적 회상과 검색](retrieval.md)을 참조한다.

## 상한과 타임아웃

`priming.max_tokens`의 기본값은 2,000이며, 프로필에 전달하는 회상 결과의 상한으로 사용된다. 메시지 유형에 따라 이 값을 동적으로 배분하는 메커니즘은 아니다. 미전송 인간 알림은 별도의 계약으로 처리되며, 일반 회상 예산을 충족하기 위해 제외되지 않는다.

`priming.channel_timeout_seconds`의 기본값은 60초이다. 채널별로 상한을 두고, 시간 초과된 채널만 비어 있는 것으로 처리하며, 다른 획득 결과는 계속 진행한다. 설정의 전체 항목은 [설정 참조](../reference/config.md)를 참조한다.
