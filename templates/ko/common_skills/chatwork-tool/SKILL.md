---
name: chatwork-tool
description: >-
  Chatwork 연동 도구. 메시지 송수신·검색·미회신 확인·룸 목록을 수행한다.
  Use when: Chatwork로 메시지 전송, 룸 목록, 미회신 확인, 채팅 검색, 멘션 대응이 필요할 때.
tags: [communication, chatwork, external]
---

# Chatwork 도구

Chatwork의 메시지 송수신, 검색, 관리를 수행하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool chatwork <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### send — 메시지 송신
```bash
animaworks-tool chatwork send ROOM MESSAGE
```

### messages — 메시지 취득
```bash
animaworks-tool chatwork messages ROOM [-n 20]
```

### search — 메시지 검색
```bash
animaworks-tool chatwork search KEYWORD [-r ROOM] [-n 50]
```

### unreplied — 미회신 메시지 확인
```bash
animaworks-tool chatwork unreplied [--json]
```
- `include_toall` (선택, 기본값: false): 전체 대상 메시지 포함 여부

### rooms — 룸 목록
```bash
animaworks-tool chatwork rooms
```

### mentions — 멘션 취득
```bash
animaworks-tool chatwork mentions [--json]
```
- `include_toall` (선택, 기본값: false): 전체 대상 메시지 포함 여부

### delete — 메시지 삭제 (본인 메시지만)
```bash
animaworks-tool chatwork delete ROOM MESSAGE_ID
```

### sync — 메시지 동기화 (캐시 갱신)
```bash
animaworks-tool chatwork sync [ROOM]
```

## CLI 사용법

```bash
animaworks-tool chatwork send ROOM MESSAGE
animaworks-tool chatwork messages ROOM [-n 20]
animaworks-tool chatwork search KEYWORD [-r ROOM] [-n 50]
animaworks-tool chatwork unreplied [--json]
animaworks-tool chatwork rooms
animaworks-tool chatwork mentions [--json]
animaworks-tool chatwork delete ROOM MESSAGE_ID
animaworks-tool chatwork sync [ROOM]
```

## 주의사항

- API Token이 credentials에 사전 설정되어 있어야 합니다
- room은 룸 이름 또는 룸 ID로 지정 가능합니다
- 메시지 송신 시 쓰기 전용 토큰이 필요할 수 있습니다
