<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/retrieval.md -->
<!-- i18n: source-sha256=cd9eb191d085ca1974053e1c223707420169fadf846459b8690a8f5b40695a92 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 의도적 회상과 검색

Anima는 자동 회상으로 부족한 정보를 `search_memory`로 명시적으로 찾고, 필요에 따라 `read_memory_file`로 원본 파일을 읽는다. 도구 정의는 `core/tooling/schemas/memory.py`, 인자 처리는 `core/tooling/handler_memory.py`에 있다.

## `search_memory`의 인자

| 인자 | 용도 |
|---|---|
| `query` | 자연어 검색어. 필수이다. |
| `scope` | `knowledge`, `episodes`, `procedures`, `facts`, `common_knowledge`, `skills`, `activity_log`, `code`, `all` 중에서 검색 대상을 선택한다. 생략 시 기본값은 `all`. |
| `offset` | 결과의 페이지 위치. 1페이지는 10건이며, 최대 offset은 50. |
| `project` | 등록된 project archive로 결과를 제한한다. |
| `time_range` | `after`과 `before`에 ISO 형식의 날짜 또는 시간을 지정하고, 기간을 좁힌다. |

인자의 상세와 CLI 사용 방법은 [도구 CLI 참조](../reference/tool-cli.md)를 참조한다. 코드 검색에는 `scope="code"`과 `project`가 필요하다.

## 검색 파이프라인

`core/memory/retrieval/unified_search.py`은 trigger와 scope에 기반하여 후보를 모은다. 벡터 검색은 의미적 유사도를, BM25는 어구 일치를 사용한다. 두 경로 등에서 얻은 순위 목록을 RRF로 융합하고, 설정과 후보 수에 따라 rerank한다. 실제 주요 순서는 다음과 같다.

1. 쿼리를 확장하고, 벡터 검색용과 어구 검색용 입력을 준비한다.
2. 벡터와 BM25 후보를 획득한다. `activity_log`은 활동 기록용 BM25 검색을 사용한다.
3. `core/memory/retrieval/pipeline.py`로 순위 목록을 RRF 융합한다.
4. 기간과 entity에 기반한 보정을 수행하고, 조건을 충족하면 cross-encoder로 rerank한다.
5. rerank 후 기간·entity 보정을 반영하고, 접근 이력 보정을 적용한다.
6. confidence gate는 결과의 확실성을 판단하지만, 임계값을 밑도는 후보도 버리지 않고 낮은 신뢰도로 표시한다. 후보 자체가 없는 경우에는 검색 결과를 반환하지 않는다.

기간 지정이 없는 경우에도, 쿼리 중의 날짜나 시간 표현이 검색 후보 보정에 사용될 수 있다. trigger별 scope와 rerank 방침은 [자동 회상](priming.md)에 기재한다.

## Atomic facts와 entity index

`facts/`에는 `core/memory/facts/store.py`의 `FactRecord`가 날짜별 JSONL로 저장된다. 레코드는 문장, source/target entity, 관계 종류, 유효 시점·유효 마감일, 출처 에피소드, 신뢰도 등을 가진다. 에피소드 등에서 추출되며, 일반적인 의미 기억과 동일하게 검색 대상으로 할 수 있다.

`core/memory/facts/entity_index.py`은 facts에서 `state/entity_registry.json`를 구성하고, entity 이름이나 별명을 검색 시 후보 보정에 활용한다. RAG 색인화는 `core/memory/rag/indexer.py`가 Markdown이나 facts를 청크화하고, 메타데이터와 함께 벡터 색인에 등록한다. 색인은 파일 내용을 원본으로 하는 파생 데이터이며, 정기 처리가 변경분을 반영한다.

## 벡터 경로

Chroma의 영구 저장소는 Anima별 root 프로세스의 `MemoryService`이 소유한다. `core/memory/rag/vector_client.py`의 공통 client와 `vector_registry.py`가 연결을 선택하고, root 자신은 in-process bridge를 사용한다. server 측 내부 vector API는 root로 IPC 전송하며, 다른 프로세스는 HTTP client 경유로 접근한다. 임베딩 처리도 server 측에 집약된다.

## RAG 복구

`core/memory/rag/repair/detect.py`는 손상 징후를 받아 복구를 요청한다. root의 `MemoryService`는 원본 메모리 파일에서 staging 인덱스를 만들고, 검증한 후에 전환한다. 운영 중 명시적인 복구가 필요한 경우에는 [CLI 참조의 `repair-rag`](../reference/cli.md)를 참조한다.
