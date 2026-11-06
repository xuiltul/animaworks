# 액션 규칙 (Action Rules)

## 개요

액션 규칙은 전송, 게시, 알림, 메모리 기록 등 부작용이 있는 작업 직전에 확인을 넣기 위한 지식입니다. `knowledge/action-rule-*.md`에 `[ACTION-RULE]`과 `trigger_tools:`를 작성하면, 해당 도구 실행 전에 검색됩니다.

## 기본 형식

```markdown
## [ACTION-RULE] ルール名
trigger_tools: gmail_draft, gmail_send
keywords: メール, 下書き, 重複確認
---
実行前に必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
必要な確認が終わってから同じツールを再実行する。
```

| 필드 | 필수 | 설명 |
|-----------|------|------|
| `trigger_tools` | 필수 | 대상 도구 이름. 여러 개는 쉼표로 구분 |
| `keywords` | 선택 | 검색 정확도를 높이는 단어 |
| 본문 | 필수 | 종료 시 표시되는 확인 내용. 필독 파일은 `read_memory_file(path="...")`로 작성 |

## ToolHandler의 대상 도구 이름

- `call_human`
- `send_message`
- `animaworks-tool board post`
- `write_memory_file`
- `gmail_draft`
- `gmail_send`
- `chatwork_send`
- `slack_send`
- `discord_send`

## CLI 대응

| CLI | 액션 규칙상의 이름 |
|-----|--------------------------|
| `animaworks-tool gmail draft` | `gmail_draft` |
| `animaworks-tool gmail send` | `gmail_send` |
| `animaworks-tool chatwork send` | `chatwork_send` |
| `animaworks-tool chatwork upload` | `chatwork_send` |
| `animaworks-tool slack send` | `slack_send` |
| `animaworks-tool discord send` | `discord_send` |
| `animaworks-tool call_human` | `call_human` |

`animaworks-tool submit ...`는 액션 규칙 대상 외입니다. 백그라운드 투입처의 런타임에 대상 서브커맨드가 다시 판정됩니다.

## 게이트 동작

- 관련도 점수 `0.80` 미만의 규칙은 종료하지 않습니다.
- 검색 실패, vector store 부재, 일치 규칙 없음의 경우 fail-open으로 실행을 방해하지 않습니다.
- 본문에 `read_memory_file(path="...")`이 포함된 경우, 같은 action-gate 세션 내에서 모든 경로를 읽을 때까지 종료합니다.
- 필독 파일이 없는 리뷰 전용 규칙은 같은 action-gate 세션의 `tool:rule`마다 1회만 종료합니다.
- 전역적인 "최대 2회 종료" 제한은 없습니다.
- 종료되면 표시된 규칙을 읽고 필요한 `read_memory_file`이나 확인을 실행한 후 같은 작업을 재시도합니다.

## 작성 예

```markdown
## [ACTION-RULE] Gmail下書き前の重複確認
trigger_tools: gmail_draft, gmail_send
keywords: Gmail, 下書き, 重複, thread
---
Gmail下書きや送信の前に、必ず read_memory_file(path="procedures/gmail-draft-check.md") を読む。
既存スレッドと既存下書きの重複を確認してから実行する。
```

```markdown
## [ACTION-RULE] 顧客メモ更新前の確認
trigger_tools: write_memory_file
keywords: 顧客, customer, profile
---
顧客関連の `knowledge/` を更新する前に、関連する既存ファイルを読んで矛盾がないか確認する。
```

## 배치 위치

보통 `knowledge/action-rule-{topic}.md`에 작성합니다. 작성 전에 `search_memory(scope="knowledge")`로 유사 규칙을 찾고, 기존 규칙이 있으면 업데이트를 우선하세요.
