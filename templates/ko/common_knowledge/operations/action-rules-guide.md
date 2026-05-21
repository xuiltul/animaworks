# 액션 규칙 (Action Rules)

## 개요

액션 규칙은 전송, 게시, 알림, 메모리 쓰기처럼 부작용이 있는 작업 직전에 확인을 넣는 지식 파일입니다. `knowledge/action-rule-*.md`에 `[ACTION-RULE]`과 `trigger_tools:`를 쓰면 대상 도구 실행 전에 검색됩니다.

## 기본 형식

```markdown
## [ACTION-RULE] 규칙 이름
trigger_tools: gmail_draft, gmail_send
keywords: 이메일, 초안, 중복 확인
---
실행 전에 반드시 read_memory_file(path="procedures/gmail-draft-check.md") 를 읽는다.
필요한 확인을 마친 뒤 같은 도구를 다시 실행한다.
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `trigger_tools` | 필수 | 대상 도구 이름. 여러 개는 쉼표로 구분 |
| `keywords` | 선택 | 검색 정확도를 높이는 단어 |
| 본문 | 필수 | 일시 정지 시 표시되는 확인 내용. 반드시 읽어야 하는 파일은 `read_memory_file(path="...")`로 작성 |

## ToolHandler 대상 도구 이름

- `call_human`
- `send_message`
- `post_channel`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI 대응

| CLI | 액션 규칙 이름 |
|-----|----------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...`은 액션 규칙 대상이 아닙니다. 큐에 들어간 실제 서브커맨드는 실행될 때 다시 확인됩니다.

## 게이트 동작

- 관련도 점수 `0.80` 미만의 규칙은 차단하지 않습니다.
- 검색 실패, vector store 부재, 일치 규칙 없음은 fail-open으로 실행을 막지 않습니다.
- 본문에 `read_memory_file(path="...")`가 있으면 같은 action-gate 세션에서 모든 경로를 읽을 때까지 차단합니다.
- 필수 읽기 파일이 없는 검토 전용 규칙은 같은 action-gate 세션의 `tool:rule`마다 한 번만 차단합니다.
- 전역 “최대 2회 정지” 제한은 없습니다.
- 정지되면 표시된 규칙을 읽고 필요한 `read_memory_file` 또는 확인을 수행한 뒤 같은 작업을 다시 실행합니다.

## 예시

```markdown
## [ACTION-RULE] Gmail 초안 전 중복 확인
trigger_tools: gmail_draft, gmail_send
keywords: Gmail, 초안, 중복, thread
---
Gmail 초안 작성 또는 전송 전에 반드시 read_memory_file(path="procedures/gmail-draft-check.md") 를 읽는다.
기존 스레드와 기존 초안의 중복을 확인한 뒤 실행한다.
```

```markdown
## [ACTION-RULE] 고객 메모리 업데이트 전 확인
trigger_tools: write_memory_file
keywords: 고객, customer, profile
---
고객 관련 `knowledge/`를 업데이트하기 전에 관련 기존 파일을 읽고 모순이 없는지 확인한다.
```

## 위치

일반적으로 `knowledge/action-rule-{topic}.md`에 생성합니다. 새 규칙을 만들기 전에 `search_memory(scope="knowledge")`로 유사 규칙을 찾고, 기존 규칙이 있으면 업데이트를 우선하세요.
