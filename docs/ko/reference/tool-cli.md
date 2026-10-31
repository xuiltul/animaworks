<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/reference/tool-cli.md -->
<!-- i18n: source-sha256=658df3f85c8666e262c051000087778fe185fc827a30ab7bc1a82f8131c61935 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

# 도구 CLI 참조: `animaworks-tool`

`animaworks-tool`은 core의 외부 도구 스키마에서 생성됩니다. common / personal 도구는 런타임에 따라 달라지므로 대상에서 제외됩니다.

## `submit`을 통한 백그라운드 실행

`animaworks-tool submit <tool_name> [args...]`은 장시간 실행되는 도구를 대기 작업으로 등록하고, 완료 결과를 받은 편지함으로 전달합니다. 실행하려면 `ANIMAWORKS_ANIMA_DIR`가 필요합니다.

## `discord_channel_post`

Discord 텍스트 채널에 메시지를 게시합니다. 메시지는 Anima ID(이름 + 아바타)로 표시됩니다. 나중에 참조할 수 있도록 메시지 ID를 반환합니다.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `channel_id` | string | 예 | Discord 채널 ID |
| `text` | string | 예 | 메시지 텍스트(Markdown 지원, 최대 2000자) |

## `discord_unreplied`

내 이름이 언급되었지만 아직 답변하지 않은 Discord 메시지를 찾습니다. 하트비트 중 대기 중인 요청을 확인할 때 유용합니다.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `channel_id` | string | 아니요 | 검색할 Discord 채널 ID(선택 사항 — 생략하면 캐시된 모든 채널을 검색합니다) |
| `limit` | integer | 아니요 | 최대 결과 수(기본값: 10) |

## `enclave_ask`

격리된 enclave에 질문을 보내고 job_id를 즉시 받습니다. 상태를 확인할 수 있으며, 완료 결과는 inbox에 도착합니다.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `case_id` | string | 아니요 | 선택적 케이스 식별자(생략하면 anima 이름과 날짜를 바탕으로 생성) |
| `enclave` | string | 예 | 연결할 enclave의 설정 이름 |
| `model` | string | 아니요 | 선택적 모델 지정(enclave 측 model_profiles의 이름. 예: luna-max / opus-high. 생략하면 enclave의 기본값) |
| `question` | string | 예 | 격리된 인스턴스에 보낼 질문 |
| `wait_s` | integer | 아니요 | 결과 확인을 위해 대기할 최대 시간(초)(기본값: 0. 0이면 즉시 반환) |

## `enclave_cancel`

실행 중이거나 대기 중인 enclave 작업을 취소합니다.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `enclave` | string | 예 | 연결할 enclave의 설정 이름 |
| `job_id` | string | 예 | enclave ask가 반환한 작업 ID |

## `enclave_status`

enclave 작업의 상태, 경과 시간, 진행 상황 및 완료 시 확인된 사실을 확인합니다.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `enclave` | string | 예 | 연결할 enclave의 설정 이름 |
| `job_id` | string | 예 | enclave ask가 반환한 작업 ID |

## `google_sheets_append_values`

스프레드시트 range/table(append_values)의 마지막 데이터 뒤에 행을 추가합니다. 덮어쓰기에 주의하세요. 기존 데이터를 확인하려면 먼저 read_values를 사용하세요.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `range` | string | 예 | A1 범위. 시트 이름을 지정할 수도 있습니다(예: 'Sheet1!A1:C10') |
| `spreadsheet_id` | string | 예 | 스프레드시트 ID 또는 전체 docs.google.com URL |
| `value_input_option` | string | 아니요 | 입력 해석 방식: USER_ENTERED(기본값) 또는 RAW |
| `values` | array | 예 | 셀 값의 2차원 배열(행 목록) |

## `google_sheets_read`

스프레드시트 범위의 셀 값을 읽습니다(read_values).

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `range` | string | 아니요 | A1 범위. 시트 이름을 지정할 수도 있습니다(예: 'Sheet1!A1:C10'). 기본값: A1:Z1000 |
| `spreadsheet_id` | string | 예 | 스프레드시트 ID 또는 전체 docs.google.com URL |

## `google_sheets_tabs`

스프레드시트의 시트 탭과 기본 메타데이터를 나열합니다(list_tabs).

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `spreadsheet_id` | string | 예 | 스프레드시트 ID 또는 전체 docs.google.com URL |

## `google_sheets_write_values`

스프레드시트 범위의 셀 값을 덮어씁니다(write_values). 덮어쓰기에 주의하세요. 기존 데이터를 확인하려면 먼저 read_values를 사용하세요.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `range` | string | 예 | A1 범위. 시트 이름을 지정할 수도 있습니다(예: 'Sheet1!A1:C10') |
| `spreadsheet_id` | string | 예 | 스프레드시트 ID 또는 전체 docs.google.com URL |
| `value_input_option` | string | 아니요 | 입력 해석 방식: USER_ENTERED(기본값) 또는 RAW |
| `values` | array | 예 | 셀 값의 2차원 배열(행 목록) |

## `slack_channel_post`

Bot Token API를 통해 실제 Slack 채널에 메시지를 게시합니다. slack_channel_update로 나중에 업데이트할 수 있도록 메시지 ts를 반환합니다. 내부 Board가 아닌 외부 Slack 채널에 사용하세요.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `channel_id` | string | 예 | Slack 채널 ID(예: C0AJ4J5KK46) |
| `text` | string | 예 | 메시지 텍스트(Markdown은 Slack mrkdwn으로 변환됩니다) |
| `thread_ts` | string | 아니요 | 스레드에 답글을 달 때 사용할 상위 메시지 ts(선택 사항) |

## `slack_channel_update`

ts를 기준으로 기존 Slack 메시지를 업데이트합니다. 메시지는 알림 없이 교체됩니다. task-board와 같은 실시간 대시보드에 사용하세요.

| 인수 | 유형 | 필수 | 설명 |
|---|---|---|---|
| `channel_id` | string | 예 | Slack 채널 ID |
| `text` | string | 예 | 새 메시지 텍스트 |
| `ts` | string | 예 | 업데이트할 메시지 타임스탬프(slack_channel_post 결과에서 확인) |
