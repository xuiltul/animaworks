# common_knowledge 참조 경로

Anima가 common_knowledge에 접근하는 5가지 경로와 백그라운드 RAG 인덱스 구축 메커니즘입니다.

---

## 참조 경로 전체 개요

| # | 경로 | 유형 | Anima의 인식 |
|---|------|------|-------------|
| 1 | 시스템 프롬프트 힌트 | 자동 | 힌트를 보고 능동적으로 접근 |
| 2 | Priming Channel C | 자동 | 관련 지식으로 자동 표시 |
| 3 | `search_memory` 도구 | 능동 | scope 지정으로 명시적 검색 |
| 4 | `read_memory_file` / `write_memory_file` | 능동 | 경로 지정으로 직접 접근 |
| 5 | Claude Code 직접 파일 I/O (Mode S) | 능동 | Read/Write 등으로 직접 접근 |

---

## 경로 1: 시스템 프롬프트 힌트 주입

`builder.py`가 시스템 프롬프트 구축 시 `~/.animaworks/common_knowledge/`에 파일이 존재하면 **힌트 텍스트**를 Group 4(기억과 능력)에 주입합니다.

- **주입 타이밍**: 프롬프트 구축 시 (자동)
- **내용**: common_knowledge의 존재와 사용 방법에 대한 힌트 (파일 내용은 포함하지 않음)
- **제외 조건**: `is_task=True` (TaskExec)인 경우 생략
- **Anima의 행동**: 힌트를 보고 `search_memory`나 `read_memory_file`로 능동적으로 접근

## 경로 2: Priming Channel C — RAG 벡터 검색

`PrimingEngine`이 메시지의 키워드에서 자동으로 벡터 검색을 수행하고, 개인 knowledge와 공유 common_knowledge를 통합하여 시스템 프롬프트에 주입합니다.

- **예산**: 700토큰 (Channel C 할당)
- **검색 대상**: `shared_common_knowledge` 컬렉션 (ChromaDB)
- **병합 방법**: 개인 knowledge 검색 결과와 점수로 병합 및 정렬
- **Anima의 행동**: 관련 common_knowledge 단편이 Priming 섹션에 자동 표시

### 주의 사항
- 700토큰 제약이 있어 전문이 아닌 관련 단편만 표시
- common_knowledge 문서 수가 증가하면 개인 knowledge 청크가 밀려나는 리스크가 있음

## 경로 3: `search_memory` 도구

Anima가 `search_memory(query="...", scope="common_knowledge")`를 호출하면, 키워드 검색과 벡터 검색의 하이브리드로 common_knowledge를 검색합니다.

- **키워드 검색**: `~/.animaworks/common_knowledge/` 내의 .md 파일을 텍스트 스캔
- **벡터 검색**: `shared_common_knowledge` 컬렉션을 검색
- **scope 지정**: `"common_knowledge"`로 한정 검색, `"all"` (기본값)로도 포함됨

### 사용 예시
```
search_memory(query="메시지 전송", scope="common_knowledge")
search_memory(query="rate limit", scope="all")
```

## 경로 4: `read_memory_file` / `write_memory_file`

Anima가 `read_memory_file(path="common_knowledge/...")`를 호출하면 경로 프리픽스를 검출하여 `~/.animaworks/common_knowledge/`로 해석합니다.

- **읽기**: 모든 Anima가 접근 가능
- **쓰기**: 모든 Anima가 접근 가능 (공유 지식의 축적용)
- **경로 순회 방어**: `is_relative_to` 검사로 common_knowledge 외부 접근을 방지

### 사용 예시
```
read_memory_file(path="common_knowledge/00_index.md")
write_memory_file(path="common_knowledge/operations/new-guide.md", content="...")
```

## 경로 5: Claude Code 직접 파일 I/O (Mode S 전용)

Mode S에서는 Claude Code의 내장 도구(Read, Write, Grep, Glob 등)로 `~/.animaworks/common_knowledge/`에 직접 접근할 수 있습니다.

- **권한**: `handler_perms.py`에서 공유 읽기 전용 디렉토리로 허용
- **대상 모드**: Mode S (Agent SDK) 전용

---

## 백그라운드: RAG 인덱스 구축

common_knowledge가 벡터 검색(경로 2 및 3)에서 발견되려면 ChromaDB에 인덱스되어 있어야 합니다.

### 인덱스 타이밍

1. **Anima 기동 시**: `MemoryManager` 초기화 시 `_ensure_shared_knowledge_indexed()`가 호출되며, SHA-256 해시로 변경을 감지합니다. 변경이 있으면 `shared_common_knowledge` 컬렉션에 재인덱스
2. **매일 04:00**: `_run_daily_indexing()`으로 모든 Anima의 벡터 DB를 증분 인덱스 업데이트. common_knowledge도 이 타이밍에 재인덱스

### 청킹 전략

`memory_type="common_knowledge"`의 경우, knowledge와 동일한 **Markdown 헤딩 분할** 방식으로 청킹됩니다.

### 컬렉션 이름

`shared_common_knowledge` (모든 Anima가 공유하는 단일 컬렉션)

---

## reference/와의 차이

| 항목 | common_knowledge | reference |
|------|-----------------|-----------|
| RAG 인덱스 | 대상 (`shared_common_knowledge`) | **비대상** |
| `search_memory` | scope="common_knowledge" / "all"로 검색 가능 | 검색 불가 |
| Priming Channel C | 단편이 자동 표시됨 | 표시되지 않음 |
| `read_memory_file` | 읽기/쓰기 가능 | **읽기 전용** |
| 용도 | 일상적인 실용 가이드 및 판단 기준 | 상세 기술 레퍼런스 |
| 시스템 프롬프트 | 힌트 주입 있음 | 힌트 주입 있음 (별도 섹션) |
