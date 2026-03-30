---
name: notion-tool
description: >-
  Notion 연동 도구. 페이지·데이터베이스 검색·조회·생성·갱신을 API로 수행한다.
  Use when: Notion 페이지 편집, 데이터베이스 행 추가, 워크스페이스 검색이 필요할 때.
tags: [productivity, notion, external]
---

# Notion 도구

Notion API를 통해 페이지 및 데이터베이스를 검색, 조회, 생성, 업데이트하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool notion <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### search — 워크스페이스 검색
```bash
animaworks-tool notion search [검색어] -j
```

### get_page — 페이지 메타데이터 조회
```bash
animaworks-tool notion get-page PAGE_ID -j
```

### get_page_content — 페이지 본문 조회
```bash
animaworks-tool notion get-page-content PAGE_ID -j
```

### get_database — 데이터베이스 메타데이터 조회
```bash
animaworks-tool notion get-database DATABASE_ID -j
```

### query — 데이터베이스 쿼리
```bash
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
```
- `filter`: Notion API 필터 JSON (선택)
- `sorts`: 정렬 조건 배열 (선택)

### create_page — 페이지 생성
```bash
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
```
- `parent_page_id` 또는 `parent_database_id` 중 하나가 필수
- `children`: 페이지 본문 블록 배열 (선택)

### update_page — 페이지 업데이트
```bash
animaworks-tool notion update-page PAGE_ID --properties JSON -j
```

### create_database — 데이터베이스 생성
```bash
animaworks-tool notion create-database --parent-page-id ID --title "이름" --properties JSON -j
```

## CLI 사용법

```bash
animaworks-tool notion search [검색어] -j
animaworks-tool notion get-page PAGE_ID -j
animaworks-tool notion get-page-content PAGE_ID -j
animaworks-tool notion get-database DATABASE_ID -j
animaworks-tool notion query DATABASE_ID [--filter JSON] [--sorts JSON] [-n 10] -j
animaworks-tool notion create-page --parent-page-id ID --properties JSON -j
animaworks-tool notion update-page PAGE_ID --properties JSON -j
animaworks-tool notion create-database --parent-page-id ID --title "이름" --properties JSON -j
```

## 주의사항

- Notion API Token이 credentials에 사전 설정되어 있어야 합니다
- 페이지/데이터베이스 ID는 하이픈 있는 형식과 없는 형식 모두 가능합니다
- properties 구조는 Notion API 스키마를 따릅니다
