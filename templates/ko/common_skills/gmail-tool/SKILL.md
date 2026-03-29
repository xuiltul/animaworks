---
name: gmail-tool
description: >-
  Gmail 연동 도구. 미읽음 확인·본문 읽기·임시보관 작성을 OAuth2 Gmail API로 수행한다.
  Use when: 받은편지함 확인, 본문 조회, 임시보관 작성, 라벨 메일 처리가 필요할 때.
tags: [communication, gmail, email, external]
---

# Gmail 도구

Gmail을 OAuth2로 직접 조작하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool gmail <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### unread — 읽지 않은 메일 목록
```bash
animaworks-tool gmail unread [-n 20]
```

### read_body — 메일 본문 읽기
```bash
animaworks-tool gmail read MESSAGE_ID
```

### draft — 임시 보관함 작성
```bash
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## CLI 사용법

```bash
animaworks-tool gmail unread [-n 20]
animaworks-tool gmail read MESSAGE_ID
animaworks-tool gmail draft --to ADDR --subject SUBJ --body BODY [--thread-id TID]
```

## 주의사항

- 최초 사용 시 OAuth2 인증 흐름이 필요합니다
- credentials.json과 token.json이 ~/.animaworks/에 배치되어야 합니다
