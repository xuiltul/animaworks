<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/reference/cli.md -->
<!-- i18n: source-sha256=d381aa269e104fc13b7413556212b1ae09c5385a36a4e4c9da2af55d9140c099 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

# CLI 참조: `animaworks`

`animaworks` 명령어의 argparse 정의에서 생성되었습니다.

## 전역 옵션

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --gateway-url | option | — | — | Gateway URL |
| --data-dir | option | — | — | 런타임 데이터 디렉터리 재정의 (기본값: ~/.animaworks 또는 ANIMAWORKS_DATA_DIR) |

## `anima`

Anima 프로세스 관리

`usage: animaworks anima [-h]
                        {restart,status,create,export,delete,disable,enable,list,info,permissions,set-model,set-background-model,reload,set-role,rename,audit}
                        ...`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `anima audit`

하위 애니마의 최근 활동 감사

`usage: animaworks anima audit [-h] [--all] [--days DAYS] [--since SINCE]
                              [--date DATE]
                              [anima]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 대상 애니마 이름 (--all 사용 시 생략) |
| --all | flag | false | — | 모든 애니마 감사 |
| --days | option | 1 | — | 감사할 일수 (기본값: 1, 최대: 30) |
| --since | option | — | — | HH:MM 형식의 시작 시간 (오늘, JST). 지정 시 --days를 재정의 |
| --date | option | — | — | 특정 날짜 (YYYY-MM-DD, 'today', 또는 'yesterday'). 해당 날짜의 활동만 표시 |

## `anima create`

새 애니마 생성

`usage: animaworks anima create [-h] [--name NAME] [--template TEMPLATE]
                               [--from-md PATH] [--supervisor SUPERVISOR]
                               [--role {engineer,researcher,manager,writer,ops,general}]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --name | option | — | — | 애니마 이름 (빈칸일 때 필수, template/md일 때 선택) |
| --template | option | — | — | 이름이 있는 템플릿에서 생성 |
| --from-md | option | — | — | MD 파일에서 생성 |
| --supervisor | option | — | — | 상위 애니마 이름 (캐릭터 시트 재정의) |
| --role | option | — | engineer, researcher, manager, writer, ops, general | 적용할 역할 템플릿 (기본값: general) |

## `anima delete`

애니마 삭제 (선택적 아카이브 포함)

`usage: animaworks anima delete [-h] [--no-archive] [--force] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 삭제할 애니마 이름 |
| --no-archive | flag | false | — | 삭제 전 ZIP 아카이브 생성 건너뛰기 |
| --force | flag | false | — | 확인 프롬프트 건너뛰기 |

## `anima disable`

애니마 비활성화 (휴양)

`usage: animaworks anima disable [-h] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 비활성화할 애니마 이름 |

## `anima enable`

애니마 활성화 (복귀)

`usage: animaworks anima enable [-h] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 활성화할 애니마 이름 |

## `anima export`

Anima의 작업 공간을 OpenClaw / Hermes Agent 형식으로 내보내기

`usage: animaworks anima export [-h] --format {openclaw,hermes} --out DIR anima`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인수 | — | — | 내보낼 Anima 이름 |
| --format | 옵션 | — | openclaw, hermes | 대상 작업 공간 형식 |
| --out | 옵션 | — | — | 출력 디렉터리 |

## `anima info`

애니마의 상세 설정 표시

`usage: animaworks anima info [-h] [--json] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 |
| --json | flag | false | — | JSON으로 출력 |

## `anima list`

모든 애니마를 상태와 함께 나열

`usage: animaworks anima list [-h] [--local]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --local | flag | false | — | 파일시스템을 직접 스캔 |

## `anima permissions`

애니마의 권한 설정 표시

`usage: animaworks anima permissions [-h] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 |

## `anima reload`

status.json에서 애니마 설정 핫 리로드

`usage: animaworks anima reload [-h] [--all] [anima]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 (--all 사용 시 불필요) |
| --all | flag | false | — | 실행 중인 모든 애니마의 설정 리로드 |

## `anima rename`

애니마 이름 변경

`usage: animaworks anima rename [-h] [--force] old_name new_name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| old_name | positional | — | — | 현재 애니마 이름 |
| new_name | positional | — | — | 새 애니마 이름 |
| --force | flag | false | — | 확인 프롬프트 건너뛰기 |

## `anima restart`

애니마 프로세스 재시작

`usage: animaworks anima restart [-h] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 |

## `anima set-background-model`

heartbeat/cron 모델 설정

`usage: animaworks anima set-background-model [-h] [--credential CREDENTIAL]
                                             [--all] [--clear]
                                             [anima] [model]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 |
| model | positional | — | — | 배경 모델 이름 |
| --credential | option | — | — | 자격 증명 이름 |
| --all | flag | false | — | 활성화된 모든 애니마에 적용 |
| --clear | flag | false | — | 배경 모델 재정의 제거 |

## `anima set-model`

지정된 애니마의 주 모델을 변경합니다.

`usage: animaworks anima set-model [-h] [--credential CREDENTIAL] [--all]
                                  [anima] [model]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 (--all 사용 시 불필요) |
| model | positional | — | — | 모델 이름 (예: azure/gpt-4.1-mini) |
| --credential | option | — | — | 자격 증명 이름 |
| --all | flag | false | — | 활성화된 모든 애니마에 적용 |

## `anima set-role`

애니마의 역할 변경

`usage: animaworks anima set-role [-h] [--status-only] [--no-restart]
                                 anima
                                 {engineer,researcher,manager,writer,ops,general}`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 |
| role | positional | — | engineer, researcher, manager, writer, ops, general | 할당할 새 역할 |
| --status-only | flag | false | — | status.json 역할 필드만 업데이트; 템플릿 파일 재적용 건너뛰기 |
| --no-restart | flag | false | — | 역할 변경 후 자동 재시작 건너뛰기 |

## `anima status`

애니마 프로세스 상태 표시

`usage: animaworks anima status [-h] [anima]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | 애니마 이름 (모든 애니마는 생략) |

## `board`

보드 공유 채널 작업

`usage: animaworks board [-h] {read,post,dm-history} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `board dm-history`

상대와의 DM 기록 읽기

`usage: animaworks board dm-history [-h] [--limit LIMIT]
                                   [--direction {sent,received,both}]
                                   [--hours HOURS] [--keyword KEYWORD]
                                   from_anima peer`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| from_anima | positional | — | — | 자신의 아니마 이름 |
| peer | positional | — | — | 상대 아니마 이름 |
| --limit | option | 20 | — | 최대 메시지 수 |
| --direction | option | "both" | sent, received, both | 메시지 방향으로 필터링 (기본값: both) |
| --hours | option | — | — | 최근 N시간 동안의 메시지만 포함 |
| --keyword | option | — | — | 이 키워드가 포함된 메시지만 포함 |

## `board post`

채널에 게시

`usage: animaworks board post [-h] from_anima channel text`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| from_anima | 위치 인자 | — | — | 보내는 애니마 이름 |
| channel | 위치 인자 | — | — | 채널 이름 |
| text | 위치 인자 | — | — | 메시지 내용 |

## `board read`

채널 메시지 읽기

`usage: animaworks board read [-h] [--limit LIMIT] [--human-only] channel`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| channel | 위치 인자 | — | — | 채널 이름 (예: general, ops) |
| --limit | 옵션 | 20 | — | 최대 메시지 수 |
| --human-only | 플래그 | false | — | 사람 메시지만 표시 |

## `chat`

애니마와 대화

`usage: animaworks chat [-h] [--local] [--from FROM_PERSON]
                       [--thread THREAD_ID] [--no-tui] [--resume [SESSION_ID]]
                       [--sessions] [--user USER] [--password PASSWORD]
                       [--no-reattach]
                       [anima] [message]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인자 | — | — | 애니마 이름 |
| message | 위치 인자 | — | — | 보낼 메시지 (생략하면 대화형 TUI가 열림) |
| --local | 플래그 | false | — | (더 이상 사용되지 않음) 직접 모드 (게이트웨이 없음) |
| --from, --as | 옵션 | "human" | — | 보내는 사람 이름 (기본값: human) |
| --thread | 옵션 | "default" | — | 스레드 ID (기본값: default) |
| --no-tui | 플래그 | false | — | 메시지가 없을 때 TUI를 열지 않고 stdin을 읽음 |
| --resume | 옵션 | — | — | 이전 TUI 세션 재개 (선택적 SESSION_ID; 기본값은 최신) |
| --sessions | 플래그 | false | — | 저장된 TUI 세션 목록을 표시하고 종료 |
| --user | 옵션 | — | — | 인증된 게이트웨이용 사용자 이름 |
| --password | 옵션 | — | — | 인증된 게이트웨이용 비밀번호 (프로세스 목록에 표시됨) |
| --no-reattach | 플래그 | false | — | 시작 시 진행 중인 스트림에 다시 연결하지 않음 |

## `company`

회사 워크스페이스, 애니마 소속, 회사 소유 자산을 관리합니다.

`usage: animaworks company [-h] {create,list,assign,adopt,split,export} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `company adopt`

데이터 디렉터리 자산을 회사 아래로 이동하고, 먼저 백업한 후 일반적으로 이전 경로에 상대 심볼릭 링크를 남깁니다.

`usage: animaworks company adopt [-h] --to NAME [--dest SUBDIR] [--no-symlink]
                                path [path ...]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| path | 위치 인자 | — | — | 데이터 디렉터리 상대 또는 절대 자산 경로 |
| --to | 옵션 | — | — | 대상 회사 |
| --dest | 옵션 | — | shared, knowledge, skills, credentials, . | 대상 하위 디렉터리 (기본값: 각 소스에서 추론) |
| --no-symlink | 플래그 | false | — | 이전 경로에 심볼릭 링크를 남기지 않음 |

## `company assign`

하나 이상의 애니마를 회사에 배정하거나 회사 배정을 해제합니다.

`usage: animaworks company assign [-h] (--to NAME | --unassign)
                                 anima [anima ...]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인자 | — | — | 애니마 이름 |
| --to | 옵션 | — | — | 대상 회사 |
| --unassign | 플래그 | false | — | 회사 배정 해제 |

## `company create`

회사 워크스페이스를 생성하거나 기존 워크스페이스에 누락된 스캐폴드를 추가합니다.

`usage: animaworks company create [-h] [--display-name DISPLAY_NAME] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 회사 이름 ([a-z0-9][a-z0-9_-]*) |
| --display-name | 옵션 | — | — | 사람이 읽을 수 있는 회사 이름 (기본값: 회사 이름) |

## `company export`

회사의 구성원과 자산을 휴대용 마이그레이션 번들로 수집하며, 비밀값은 삭제되고 남은 마이그레이션 작업은 문서화됩니다.

`usage: animaworks company export [-h] --out DIR name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 회사 이름 |
| --out | 옵션 | — | — | 출력 디렉터리 (이미 파일이 있으면 안 됨) |

## `company list`

모든 회사, 표시 이름, 구성원 수, 그리고 미배정 애니마를 나열합니다.

`usage: animaworks company list [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `company split`

매니페스트에서 회사 생성, 애니마 배정, 자산 채택을 계획하거나 실행합니다.

`usage: animaworks company split [-h] --manifest FILE [--execute]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --manifest | 옵션 | — | — | YAML 또는 JSON 분할 매니페스트 |
| --execute | 플래그 | false | — | 계획 실행 (기본값: 드라이런만) |

## `config`

구성 관리

`usage: animaworks config [-h] [--interactive] {get,set,list} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --interactive, -i | 플래그 | false | — | 대화형 설정 마법사 |

## `config get`

구성 값 가져오기

`usage: animaworks config get [-h] [--show-secrets] key`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| key | 위치 인자 | — | — | 점 표기법 키 (예: system.timezone) |
| --show-secrets | 플래그 | false | — | API 키 값 표시 |

## `config list`

모든 구성 값 나열

`usage: animaworks config list [-h] [--section SECTION] [--show-secrets]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --section | 옵션 | — | — | 섹션별 필터 |
| --show-secrets | 플래그 | false | — | API 키 값 표시 |

## `config set`

구성 값 설정

`usage: animaworks config set [-h] key value`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| key | 위치 인자 | — | — | 점 표기법 키 |
| value | 위치 인자 | — | — | 설정할 값 |

## `cost`

cli.cost_help

`usage: animaworks cost [-h] [--days DAYS] [--today] [--json] [anima]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인자 | — | — | 애니마 이름 (생략하면 모든 애니마) |
| --days | 옵션 | 30 | — | 집계할 일 수 (기본값: 30) |
| --today | 플래그 | false | — | 오늘만 표시 |
| --json | 플래그 | false | — | JSON으로 출력 |

## `demo`

3-에이전트 데모 팀 실행 (Claude Code 또는 Codex에 로그인되어 있으면 API 키 불필요)

`usage: animaworks demo [-h]
                       [--preset {en-anime,en-business,ja-anime,ja-business}]
                       [--data-dir DATA_DIR] [--port PORT] [--host HOST]
                       [--reset]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --preset | 옵션 | "en-business" | en-anime, en-business, ja-anime, ja-business | — |
| --data-dir | 옵션 | "~/.animaworks-demo" | — | — |
| --port | 옵션 | 18501 | — | — |
| --host | 옵션 | "0.0.0.0" | — | — |
| --reset | 플래그 | false | — | — |

## `enclave`

격리된 엔클레이브의 상태 확인 및 감사 집계를 수행합니다.

`usage: animaworks enclave [-h] {doctor,status} ...`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `enclave doctor`

엔클레이브의 시작 가드, 소켓, 이그레스 설정을 점검합니다.

`usage: animaworks enclave doctor [-h] [--json]`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --json | 플래그 | false | — | 기계 판독 가능한 JSON 형식으로 출력합니다 |

## `enclave status`

당일 이그레스 성공 및 차단 건수를 표시합니다.

`usage: animaworks enclave status [-h]`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `heartbeat`

하트비트 트리거

`usage: animaworks heartbeat [-h] [--local] anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인자 | — | — | Anima 이름 |
| --local | 플래그 | false | — | (더 이상 사용되지 않음) 직접 모드 (게이트웨이 없음) |

## `import`

Hermes 또는 OpenClaw 데이터를 AnimaWorks로 가져오기

`usage: animaworks import [-h] {hermes,openclaw} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `import hermes`

Hermes Agent 데이터 가져오기

`usage: animaworks import hermes [-h] --path PATH [--dry-run | --apply]
                                [--replace] [--json] [--common-skills]
                                [--target-anima TARGET_ANIMA]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --path | 옵션 | — | — | 소스 디렉터리, 예: ~/.hermes 또는 ~/.openclaw |
| --dry-run | 플래그 | false | — | 런타임 파일시스템을 변경하지 않고 미리보기 |
| --apply | 플래그 | false | — | 가져올 수 있는 항목을 적용하고 마이그레이션 보고서 작성 |
| --replace | 플래그 | false | — | 백업 매니페스트 후 기존 생성 대상 교체 |
| --json | 플래그 | false | — | Markdown 대신 JSON 출력 |
| --common-skills | 플래그 | false | — | 스킬을 common_skills/community로 가져오기 |
| --target-anima | 옵션 | — | — | 개인 스킬, 사용법, 작업, 초안의 대상 anima |

## `import openclaw`

OpenClaw 데이터 가져오기

`usage: animaworks import openclaw [-h] --path PATH [--dry-run | --apply]
                                  [--replace] [--json] --target-anima
                                  TARGET_ANIMA`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --path | 옵션 | — | — | 소스 디렉터리, 예: ~/.hermes 또는 ~/.openclaw |
| --dry-run | 플래그 | false | — | 런타임 파일시스템을 변경하지 않고 미리보기 |
| --apply | 플래그 | false | — | 가져올 수 있는 항목을 적용하고 마이그레이션 보고서 작성 |
| --replace | 플래그 | false | — | 백업 매니페스트 후 기존 생성 대상 교체 |
| --json | 플래그 | false | — | Markdown 대신 JSON 출력 |
| --target-anima | 옵션 | — | — | 생성된 초안의 대상 anima |

## `index`

하이브리드 검색을 위해 메모리 파일을 벡터 데이터베이스에 인덱싱합니다.

`usage: animaworks index [-h] [--anima ANIMA] [--full] [--shared] [--dry-run]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | 옵션 | — | — | 이 anima의 메모리만 인덱싱 (기본값: 모든 anima) |
| --full | 플래그 | false | — | 전체 재인덱싱 강제 (기존 인덱스 삭제 후 재구축) |
| --shared | 플래그 | false | — | 공유 컬렉션(common_knowledge + common_skills)을 각 활성 anima의 DB에 인덱싱 |
| --dry-run | 플래그 | false | — | 실제 인덱싱 없이 인덱싱될 항목만 표시 |

## `init`

런타임 디렉터리를 초기화하고, 필요에 따라 anima를 생성합니다.

`usage: animaworks init [-h]
                       [--force | --template NAME | --from-md PATH | --blank NAME | --skip-anima]
                       [--name NAME]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --force | 플래그 | false | — | 기존 런타임에 누락된 템플릿 파일 병합 |
| --template | 옵션 | — | — | 비대화형: 이름이 지정된 템플릿에서 anima 생성 |
| --from-md | 옵션 | — | — | 비대화형: MD 파일에서 anima 생성 |
| --blank | 옵션 | — | — | 비대화형: 지정된 이름으로 빈 anima 생성 |
| --skip-anima | 플래그 | false | — | 인프라만 초기화하고 anima 생성 건너뛰기 |
| --name | 옵션 | — | — | anima 이름 재정의 (--from-md와 함께 사용) |

## `internal`

Anima 사용을 위한 내부 도구

`usage: animaworks internal [-h]
                           {archive-memory,check-permissions,create-skill,manage-channel,list-background-tasks,check-background-task}
                           ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `internal archive-memory`

기억 파일 보관

`usage: animaworks internal archive-memory [-h] [--reason REASON] path`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| path | positional | — | — | archive_memory_file에서 허용하는 상대 경로 |
| --reason | option | "archived via CLI" | — | 보관 사유 |

## `internal check-background-task`

특정 작업 확인

`usage: animaworks internal check-background-task [-h] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | 작업 ID |

## `internal check-permissions`

도구 권한 확인

`usage: animaworks internal check-permissions [-h] tool_name [action]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| tool_name | 위치 인자 | — | — | 도구 이름 |
| action | 위치 인자 | "" | — | 선택적 작업 |

## `internal create-skill`

스킬 파일 생성

`usage: animaworks internal create-skill [-h] [--content CONTENT]
                                        [--location {personal,common}]
                                        name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | positional | — | — | 스킬 이름 (또는 name.md) |
| --content | option | — | — | SKILL.md 콘텐츠 (기본값: stdin) |
| --location | option | "personal" | personal, common | — |

## `internal list-background-tasks`

백그라운드 작업 목록 조회

`usage: animaworks internal list-background-tasks [-h]
                                                 [--status {running,completed,failed,pending}]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --status | option | — | running, completed, failed, pending | 상태로 필터링 |

## `internal manage-channel`

채널 및 구성원 관리

`usage: animaworks internal manage-channel [-h] [--member MEMBERS]
                                          [--description DESCRIPTION]
                                          {create,archive,add_member,remove_member,info}
                                          channel`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| action | positional | — | create, archive, add_member, remove_member, info | 작업 |
| channel | positional | — | — | 채널 이름 |
| --member | option | [] | — | 구성원 이름 (반복 지정 가능) |
| --description | option | — | — | 새 채널에 대한 설명 |

## `logs`

anima 로그 보기

`usage: animaworks logs [-h] [--all] [--lines LINES] [--date DATE] [anima]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | 위치 인자 | — | — | Anima 이름 (--all이 아니면 필수) |
| --all | 플래그 | false | — | 모든 로그 표시 (서버 + 모든 anima) |
| --lines | 옵션 | 50 | — | 표시할 줄 수 (기본값: 50) |
| --date | 옵션 | — | — | 특정 날짜 (YYYYMMDD 형식) |

## `mcp`

anima용 stdio MCP 서버 실행

`usage: animaworks mcp [-h] --anima ANIMA [--project PROJECT] [--tools TOOLS]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | 옵션 | — | — | Anima 이름 |
| --project | 옵션 | — | — | 기본 프로젝트 아카이브 |
| --tools | 옵션 | "search_memory,read_memory_file,write_memory_file" | — | 쉼표로 구분된 노출 도구 |

## `memory`

메모리 확인 및 유지보수

`usage: animaworks memory [-h] {forgetting-dry-run} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `memory forgetting-dry-run`

업데이트 없이 저활성화 및 완전 망각 대상 건수를 표시합니다

`usage: animaworks memory forgetting-dry-run [-h] --anima ANIMA`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | 옵션 | — | — | 확인할 Anima 이름 |

## `migrate`

런타임 데이터에 필요한 마이그레이션을 실행합니다.

`usage: animaworks migrate [-h] [--dry-run] [--verbose] [--list] [--force]
                          [--rollback-standard-files] [--anima NAME]`

| 이름 | 종류 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --dry-run | 플래그 | false | — | 아무것도 수정하지 않고 변경 사항 미리보기 |
| --verbose | 플래그 | false | — | 파일별 상세 변경 사항 표시 |
| --list | 플래그 | false | — | 모든 마이그레이션 단계와 상태 표시 |
| --force | 플래그 | false | — | 상태와 관계없이 모든 마이그레이션 다시 적용 |
| --rollback-standard-files | 플래그 | false | — | 최신 standard-files 아카이브에서 레거시 작업 공간 파일 복원 |
| --anima | 옵션 | — | — | 롤백을 하나의 Anima로 제한 |

## `models`

모델 정보 및 카탈로그

`usage: animaworks models [-h] {list,info,helpers,show} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `models helpers`

보조 모델의 역할별로 결정된 모델, 인증 정보 이름, 대체 후보, 결정 출처를 표시합니다.

`usage: animaworks models helpers [-h] [--anima ANIMA] [--json]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | 옵션 | — | — | 이 Anima에 대한 애니마별 보조 모델 재정의를 확인합니다 |
| --json | 플래그 | false | — | JSON 형식으로 출력 |

## `models info`

모델의 해결 모드와 컨텍스트 표시

`usage: animaworks models info [-h] model`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| model | 위치 인자 | — | — | 모델 이름 (예: claude-sonnet-5-5) |

## `models list`

List known models

`usage: animaworks models list [-h] [--mode {A,C,D,G,S,X,Y,a,c,d,g,s,x,y}]
                              [--json]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --mode | option | — | A, C, D, G, S, X, Y, a, c, d, g, s, x, y | Filter by execution mode |
| --json | flag | false | — | Output as JSON |

## `models show`

현재 models.json 내용 표시

`usage: animaworks models show [-h] [--json]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --json | 플래그 | false | — | 원시 JSON 출력 |

## `optimize-assets`

기존 3D 에셋 최적화 (메시 제거, 압축, 단순화)

`usage: animaworks optimize-assets [-h] [--anima ANIMA] [--dry-run]
                                  [--simplify [RATIO]] [--texture-compress]
                                  [--texture-resize RES] [--all]
                                  [--skip-backup]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima, -a | 옵션 | — | — | 특정 anima의 에셋만 최적화 |
| --dry-run | 플래그 | false | — | 변경 없이 수행될 작업만 표시 |
| --simplify | 옵션 | — | — | 메시 단순화 (기본 비율: 0.27 ≈ 30K→8K 폴리곤) |
| --texture-compress | 플래그 | false | — | 텍스처를 WebP 형식으로 변환 |
| --texture-resize | 옵션 | — | — | 텍스처를 RES×RES로 크기 조정 (기본값: --texture-compress 설정 시 1024) |
| --all | 플래그 | false | — | 모든 최적화 적용: 스트립 + 단순화 + 텍스처 + draco |
| --skip-backup | 플래그 | false | — | 원본 에셋 백업 생성 건너뛰기 |

## `profile`

여러 AnimaWorks 인스턴스 관리 (멀티테넌트)

`usage: animaworks profile [-h]
                          {list,add,remove,start,stop,start-all,stop-all} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `profile add`

새 프로필 등록

`usage: animaworks profile add [-h] [--data-dir DATA_DIR] [--port PORT] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 프로필 이름 |
| --data-dir | 옵션 | — | — | 데이터 디렉터리 (기본값: ~/.animaworks/<name>) |
| --port | 옵션 | — | — | 포트 (기본값: 18500부터 자동 할당, 간격 10) |

## `profile list`

상태와 함께 모든 프로필 목록 표시

`usage: animaworks profile list [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `profile remove`

프로필 등록 제거

`usage: animaworks profile remove [-h] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 프로필 이름 |

## `profile start`

프로필에 대한 서버 시작

`usage: animaworks profile start [-h] [--host HOST] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 프로필 이름 |
| --host | 옵션 | — | — | 호스트 (기본값: 0.0.0.0) |

## `profile start-all`

모든 프로필 시작

`usage: animaworks profile start-all [-h] [--host HOST]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --host | 옵션 | — | — | 호스트 (기본값: 0.0.0.0) |

## `profile stop`

프로필에 대한 서버 중지

`usage: animaworks profile stop [-h] [--force] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 프로필 이름 |
| --force | 플래그 | false | — | 강제 중지 (타임아웃 후 SIGKILL) |

## `profile stop-all`

실행 중인 모든 프로필 중지

`usage: animaworks profile stop-all [-h] [--force]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --force | 플래그 | false | — | 강제 중지 (타임아웃 후 SIGKILL) |

## `rag-repair-status`

모든 anima의 RAG 복구 상태 표시

`usage: animaworks rag-repair-status [-h] [--json]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --json | 플래그 | false | — | 구조화된 JSON 출력 |

## `remake-assets`

Regenerate character assets using Vibe Transfer to match the art style of a reference anima. Supports selective step execution and automatic backup.

`usage: animaworks remake-assets [-h] --style-from STYLE_FROM [--steps STEPS]
                                [--prompt PROMPT]
                                [--vibe-strength VIBE_STRENGTH]
                                [--vibe-info-extracted VIBE_INFO_EXTRACTED]
                                [--seed SEED]
                                [--image-style {anime,realistic}]
                                [--no-backup] [--dry-run]
                                anima`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| anima | positional | — | — | Name of the anima whose assets to remake |
| --style-from | option | — | — | Anima name to use as style reference (their fullbody image) |
| --steps | option | — | — | Comma-separated list of steps to run (choices: fullbody, bustup, icon, chibi, pixel, 3d, rigging, animations). Default: all steps |
| --prompt | option | — | — | Override character prompt (default: read from prompt.txt) |
| --vibe-strength | option | 0.6 | — | Vibe Transfer strength 0.0-1.0 (default: 0.6) |
| --vibe-info-extracted | option | 0.8 | — | Vibe Transfer information extraction 0.0-1.0 (default: 0.8) |
| --seed | option | — | — | Seed for reproducibility (fullbody generation only) |
| --image-style | option | — | anime, realistic | Image style (default: from config.json image_gen.image_style) |
| --no-backup | flag | false | — | Skip automatic backup of existing assets |
| --dry-run | flag | false | — | Show what would be done without making API calls |

## `repair-rag`

활성 phase3 벡터 소유자를 통해 RAG를 재구축합니다.

`usage: animaworks repair-rag [-h]
                             [--anima ANIMA | --all | --suspect-only | --list-suspects]
                             [--full] [--shared]
                             [--window-minutes WINDOW_MINUTES]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 복구할 Anima 이름 |
| --all | flag | false | — | 활성화된 모든 anima 복구 |
| --suspect-only | flag | false | — | 최근 RAG 손상 증거가 있는 anima만 복구 |
| --list-suspects | flag | false | — | 복구하지 않고 의심되는 RAG DB 목록만 표시 |
| --full | flag | false | — | 파괴적 격리 및 전체 재구축에 대한 확인 필요 |
| --shared | flag | false | — | 호환성을 위해 허용됨; phase3는 공유 컬렉션도 항상 재구축함 |
| --window-minutes | option | — | — | --suspect-only/--list-suspects 조회 기간 (기본값: 복구 설정 창) |
| --reason | option | "manual_repair_rag_cli" | — | ==SUPPRESS== |

## `reset`

서버를 중지하고, 런타임 디렉터리를 삭제한 후 다시 초기화합니다

`usage: animaworks reset [-h] [--restart]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --restart | flag | false | — | 리셋 후 서버 시작 |

## `restart`

Restart the server (stop then start)

`usage: animaworks restart [-h] [--host HOST] [--port PORT] [--foreground]
                          [--force]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --host | option | "0.0.0.0" | — | — |
| --port | option | 18500 | — | — |
| --foreground, -f | flag | false | — | Run in foreground with log output (default: daemonize) |
| --force | flag | false | — | Immediately send SIGKILL without draining active work |

## `send`

anima에게 메시지 전송 (발신자는 anima 또는 인간 사용자일 수 있음)

`usage: animaworks send [-h] [--thread-id THREAD_ID] [--reply-to REPLY_TO]
                       [--intent INTENT]
                       from_person to_person message`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| from_person | positional | — | — | 발신자 이름 (anima가 아닌 이름은 인간으로 전송) |
| to_person | positional | — | — | 수신자 이름 |
| message | positional | — | — | 메시지 내용 |
| --thread-id | option | — | — | 스레드 ID |
| --reply-to | option | — | — | 답장할 메시지 ID |
| --intent | option | "" | — | 메시지 의도: delegation, report, question 또는 빈 값 |

## `serve`

서버 시작 (start의 별칭)

`usage: animaworks serve [-h] [--host HOST] [--port PORT] [--foreground]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --host | option | "0.0.0.0" | — | — |
| --port | option | 18500 | — | — |
| --foreground, -f | flag | false | — | 로그 출력과 함께 포그라운드로 실행 (기본값: 데몬화) |

## `skills`

Skill Hub 가져오기 설치 및 관리

`usage: animaworks skills [-h]
                         {install,list,inspect,remove,ledger,rollback,quarantine}
                         ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `skills inspect`

설치되었거나 격리된 스킬 검사

`usage: animaworks skills inspect [-h] [--target {personal,common}]
                                 [--anima ANIMA]
                                 skill_name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| skill_name | positional | — | — | 스킬 이름 |
| --target | option | "personal" | personal, common | 설치 대상 |
| --anima | option | — | — | personal 대상의 Anima 이름 |

## `skills install`

로컬 경로, URL 또는 GitHub 소스에서 스킬 설치

`usage: animaworks skills install [-h] [--target {personal,common}]
                                 [--anima ANIMA] [--dry-run] [--replace]
                                 [--force] [--quarantine]
                                 [--trust-level {community,untrusted}]
                                 source`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| source | positional | — | — | 로컬 경로, 직접 URL 또는 github:owner/repo/path |
| --target | option | "personal" | personal, common | 설치 대상 |
| --anima | option | — | — | personal 대상의 Anima 이름 |
| --dry-run | flag | false | — | 설치 없이 스테이징 및 스캔만 수행 |
| --replace | flag | false | — | 백업 생성 후 기존 스킬 교체 |
| --force | flag | false | — | 호환성을 위해 허용됨; 가져오기 정책은 여전히 적용됨 |
| --quarantine | flag | false | — | 활성 카탈로그 대신 격리 상태로 설치 |
| --trust-level | option | "community" | community, untrusted | 활성 설치에 적용할 신뢰 수준 |

## `skills ledger`

스킬 콘텐츠 변경 내역 표시

`usage: animaworks skills ledger [-h] [--anima ANIMA] [skill_name]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| skill_name | positional | — | — | 스킬 이름으로 필터링 |
| --anima | option | — | — | 개인 기록을 하나의 Anima로 필터링 |

## `skills list`

설치된 스킬 목록 표시

`usage: animaworks skills list [-h] [--target {personal,common}]
                              [--anima ANIMA] [--quarantine]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --target | option | "personal" | personal, common | 설치 대상 |
| --anima | option | — | — | personal 대상의 Anima 이름 |
| --quarantine | flag | false | — | 격리 항목 표시 |

## `skills quarantine`

격리된 스킬 관리

`usage: animaworks skills quarantine [-h] {list,promote} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `skills quarantine list`

격리된 스킬 목록 표시

`usage: animaworks skills quarantine list [-h] [--target {personal,common}]
                                         [--anima ANIMA]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --target | option | "personal" | personal, common | 설치 대상 |
| --anima | option | — | — | personal 대상의 Anima 이름 |

## `skills quarantine promote`

승인 후 격리된 스킬 승격

`usage: animaworks skills quarantine promote [-h] --approval-id APPROVAL_ID
                                            [--replace]
                                            [--trust-level {community,untrusted}]
                                            [--target {personal,common}]
                                            [--anima ANIMA]
                                            skill_name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| skill_name | 위치 인자 | — | — | 스킬 이름 |
| --approval-id | 옵션 | — | — | 인간 승인 식별자 |
| --replace | 플래그 | false | — | 기존 활성 스킬을 백업으로 대체 |
| --trust-level | 옵션 | "community" | community, untrusted | 승격 후 적용할 신뢰 수준 |
| --target | 옵션 | "personal" | personal, common | 설치 대상 |
| --anima | 옵션 | — | — | 개인 대상용 Anima 이름 |

## `skills remove`

설치되었거나 격리된 스킬 제거

`usage: animaworks skills remove [-h] [--target {personal,common}]
                                [--anima ANIMA]
                                skill_name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| skill_name | 위치 인자 | — | — | 스킬 이름 |
| --target | 옵션 | "personal" | personal, common | 설치 대상 |
| --anima | 옵션 | — | — | 개인 대상용 Anima 이름 |

## `skills rollback`

원장 ID로 스킬 내용 변경 하나를 롤백

`usage: animaworks skills rollback [-h] --anima ANIMA id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| id | 위치 인자 | — | — | 원장 항목 ID |
| --anima | 옵션 | — | — | 롤백 대상 Anima context/owner |

## `start`

AnimaWorks 서버 시작

`usage: animaworks start [-h] [--host HOST] [--port PORT] [--foreground]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --host | 옵션 | "0.0.0.0" | — | — |
| --port | 옵션 | 18500 | — | — |
| --foreground, -f | 플래그 | false | — | 로그 출력과 함께 포그라운드로 실행 (기본값: 데몬화) |

## `status`

게이트웨이에서 시스템 상태 표시

`usage: animaworks status [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `stop`

Stop the running server

`usage: animaworks stop [-h] [--force]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --force | flag | false | — | Immediately send SIGKILL without draining active work |

## `supervisor`

Anima용 슈퍼바이저 도구

`usage: animaworks supervisor [-h]
                             {org-dashboard,ping,read-state,task-tracker} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `supervisor org-dashboard`

상태와 작업이 포함된 조직 트리 표시

`usage: animaworks supervisor org-dashboard [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `supervisor ping`

하위 Anima가 살아있는지 확인

`usage: animaworks supervisor ping [-h] [--name NAME]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --name | 옵션 | — | — | 특정 Anima 이름 (생략 시 모든 하위 대상) |

## `supervisor read-state`

하위 상태 읽기 (current_state.md, 대기 중)

`usage: animaworks supervisor read-state [-h] name`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| name | 위치 인자 | — | — | 대상 Anima 이름 |

## `supervisor task-tracker`

위임된 작업 추적

`usage: animaworks supervisor task-tracker [-h] [--status STATUS]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --status | 옵션 | "delegated" | — | 상태별 필터 (기본값: delegated) |

## `task`

Manage persistent task queue

`usage: animaworks task [-h]
                       {board,show,claim,release,done,cancel,note,add,update,resume,list,autonomy}
                       ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `task add`

실행 가능한 TaskExec 작업을 게시합니다. backlog_task는 추적 작업을 대기열에 추가하기만 합니다.

`usage: animaworks task add [-h] [--source {human,anima}] --instruction
                           INSTRUCTION --assignee ASSIGNEE [--summary SUMMARY]
                           [--relay-chain RELAY_CHAIN] [--workspace WORKSPACE]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --source | option | "anima" | human, anima | — |
| --instruction | option | — | — | 원래 지시문 텍스트 |
| --assignee | option | — | — | 담당 아니마 이름 |
| --summary | option | — | — | 한 줄 요약 (기본값: instruction[:100]) |
| --relay-chain | option | — | — | 쉼표로 구분된 릴레이 체인 |
| --workspace | option | — | — | 작업의 working_directory에 사용할 워크스페이스 별칭 또는 경로 |

## `task autonomy`

위임 관련 자율성 지표를 집계한다(읽기 전용)

`usage: animaworks task autonomy [-h] [--days DAYS] [--json] [--anima ANIMA]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --days | option | 7 | — | 집계 기간(일수, 기본값: 7) |
| --json | flag | false | — | JSON 형식으로 출력한다 |
| --anima | option | — | — | 이 Anima만 표시한다 |

## `task board`

작업 보드 항목을 확인하고 조작합니다.

`usage: animaworks task board [-h] [--anima ANIMA | --all] [--stale STALE]
                             [--limit LIMIT] [--json]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | 옵션 | — | — | 소유자의 작업과 위임된 작업 표시 |
| --all | 플래그 | false | — | 모든 소유자 표시 |
| --stale | 옵션 | — | — | DAYS보다 오래된 작업만 표시 |
| --limit | 옵션 | 50 | — | 최대 행 수 (기본값: 50) |
| --json | 플래그 | false | — | JSON 출력 |

## `task cancel`

작업 취소

`usage: animaworks task cancel [-h] --reason REASON [--json] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | — |
| --reason | 옵션 | — | — | — |
| --json | 플래그 | false | — | JSON 출력 |

## `task claim`

시간 제한 작업 임대 획득

`usage: animaworks task claim [-h] [--ttl TTL] [--json] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | — |
| --ttl | 옵션 | "30m" | — | 최대 4시간 임대 기간 (기본값: 30m) |
| --json | 플래그 | false | — | JSON 출력 |

## `task done`

작업 완료 표시

`usage: animaworks task done [-h] --note NOTE [--json] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | — |
| --note | 옵션 | — | — | — |
| --json | 플래그 | false | — | JSON 출력 |

## `task list`

작업 목록

`usage: animaworks task list [-h]
                            [--status {pending,in_progress,delegated,done,cancelled}]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --status | 옵션 | — | pending, in_progress, delegated, done, cancelled | — |

## `task note`

작업에 메모 추가

`usage: animaworks task note [-h] [--json] task_id text`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | — |
| text | 위치 인자 | — | — | — |
| --json | 플래그 | false | — | JSON 출력 |

## `task release`

작업 임대 해제

`usage: animaworks task release [-h] [--json] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | 위치 인자 | — | — | — |
| --json | 플래그 | false | — | JSON 출력 |

## `task resume`

저장된 실행 입력으로 작업을 다시 대기열에 넣기

`usage: animaworks task resume [-h] --task-id TASK_ID`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --task-id | option | — | — | 작업 ID |

## `task show`

전체 작업 세부 정보 표시

`usage: animaworks task show [-h] [--anima ANIMA] [--json] task_id`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| task_id | positional | — | — | 작업 ID 또는 위임자 별칭 |
| --anima | option | — | — | 작업 소유자로 구분 |
| --json | flag | false | — | JSON 출력 |

## `task update`

Update task status

`usage: animaworks task update [-h] --task-id TASK_ID [--status STATUS]
                              [--summary SUMMARY]
                              [--waiting-for-kind {task,time,message,external}]
                              [--waiting-for-ref WAITING_FOR_REF]
                              [--waiting-for-until WAITING_FOR_UNTIL]
                              [--scope-note SCOPE_NOTE]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --task-id | option | — | — | Task ID |
| --status | option | — | — | 선언할 상태(선택 사항. --scope-note가 지정된 경우에만 생략 가능하며, 폐기된 상태는 거부됨) |
| --summary | option | — | — | 업데이트된 요약 |
| --waiting-for-kind | option | — | task, time, message, external | 대기 중인 작업이 기다리는 대상 |
| --waiting-for-ref | option | — | — | 작업 ID, 사람 또는 자유 형식의 대기 대상 |
| --waiting-for-until | option | — | — | kind=time일 때 사용할 ISO8601 타임스탬프 |
| --scope-note | option | — | — | 범위 관련 메모를 추가하고 요청자에게 알림 |

## `task-store`

담당자 단위 작업 원본의 유지보수 및 이관

`usage: animaworks task-store [-h]
                             {status,quiesce,resume,migrate,backup,export} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `task-store backup`

WAL 포함 DB 백업 새로 생성

`usage: animaworks task-store backup [-h] --anima ANIMA --destination
                                    DESTINATION`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |
| --destination | option | — | — | 사용되지 않은 출력 경로 |

## `task-store export`

종료 중인 현재 상태를 새 디렉터리로 내보내기

`usage: animaworks task-store export [-h] --anima ANIMA --destination
                                    DESTINATION`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |
| --destination | option | — | — | 사용되지 않은 출력 경로 |

## `task-store migrate`

종료 중인 이전 원장 가져오기 (백업 필수)

`usage: animaworks task-store migrate [-h] --anima ANIMA --backup BACKUP`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |
| --backup | option | — | — | WAL 포함 DB 백업 새로 생성 |

## `task-store quiesce`

새 실행 가져오기를 영구 종료

`usage: animaworks task-store quiesce [-h] --anima ANIMA`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |

## `task-store resume`

새 실행 가져오기 재개

`usage: animaworks task-store resume [-h] --anima ANIMA`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |

## `task-store status`

종료 게이트와 실행 수 표시

`usage: animaworks task-store status [-h] --anima ANIMA`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --anima | option | — | — | 대상 담당자 이름 |

## `tmp`

AnimaWorks 임시 디렉터리 검사 및 정리

`usage: animaworks tmp [-h] {list,clean} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `tmp clean`

오래되었거나 큰 임시 파일 제거

`usage: animaworks tmp clean [-h] [--older-than DAYS] [--min-size SIZE] [--all]
                            [--force] [--project] [--dry-run]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --older-than | option | 7 | — | DAYS보다 오래된 항목 제거 (기본값: 7) |
| --min-size | option | — | — | SIZE 이상 항목도 제거 (예: 100M, 1G) |
| --all | flag | false | — | tmp 아래 모든 항목 제거 (--force 필요) |
| --force | flag | false | — | 전체 정리를 위해 --all과 함께 필요 |
| --project | flag | false | — | 저장소 tmp/도 정리 |
| --dry-run | flag | false | — | 삭제하지 않고 제거될 항목만 표시 |

## `tmp list`

tmp 사용량 요약 표시

`usage: animaworks tmp list [-h] [--project] [--top TOP]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --project | flag | false | — | 런타임 tmp 외에 저장소 tmp/도 포함 |
| --top | option | 20 | — | 루트당 최대 표시 항목 수 (기본값: 20) |

## `vault`

암호화된 볼트 값 관리

`usage: animaworks vault [-h] {status,init,get,store,list,delete} ...`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `vault delete`

한 섹션에서 키 제거

`usage: animaworks vault delete [-h] [--shared] key`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| key | positional | — | — | 제거할 키 |
| --shared | flag | false | — | 운영자 전용: 공유 섹션에서 삭제 (연쇄 삭제하지 않음) |

## `vault get`

키로 값 가져오기

`usage: animaworks vault get [-h] [--shared] key`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| key | positional | — | — | 가져올 키 |
| --shared | flag | false | — | 공유 섹션에서만 검색 (기본값: 자체 네임스페이스를 먼저 검색한 뒤 공유 섹션 검색) |

## `vault init`

볼트 키가 없으면 생성

`usage: animaworks vault init [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `vault list`

자체 및 공유 네임스페이스의 키 목록 조회

`usage: animaworks vault list [-h] [--shared]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| --shared | flag | false | — | 공유 섹션만 조회 (ANIMAWORKS_ANIMA_DIR가 필요하지 않음) |

## `vault status`

값 없이 키와 암호화 상태 표시

`usage: animaworks vault status [-h]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| — | — | — | — | — |

## `vault store`

키-값 쌍 저장

`usage: animaworks vault store [-h] [--shared] key [value]`

| 이름 | 유형 | 기본값 | 선택지 | 설명 |
|---|---|---|---|---|
| key | positional | — | — | 저장할 키 |
| value | positional | — | — | 아니마 네임스페이스에 저장할 값 |
| --shared | flag | false | — | 운영자 전용: stdin 또는 숨겨진 프롬프트를 통해 공유 섹션에 저장 |
