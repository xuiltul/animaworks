# Priming 채널 기술 레퍼런스

PrimingEngine이 실행하는 전체 채널의 상세 사양입니다.
버짓, 검색 소스, 필터링, 동적 조정을 포함합니다.

---

## 채널 일람

| 채널 | 버짓 (토큰) | 소스 | trust |
|------|-------------|------|-------|
| A: sender_profile | 500 | `shared/users/{sender}/index.md` | medium |
| B: recent_activity | 1300 | `activity_log/` + shared channels | trusted |
| C: related_knowledge | 1200 | RAG 벡터 검색 (knowledge + common_knowledge) | medium / untrusted |
| C0: important_knowledge | 300 | `[IMPORTANT]` 태그가 지정된 청크 | medium |
| D: skill_match | 200 | 3단계 매칭 (keyword → vocab → vector) | trusted |
| E: pending_tasks | 500 | `task_queue.jsonl` + `task_results/` | trusted |
| F: episodes | 500 | RAG 벡터 검색 (episodes/) | medium |

추가 주입:

| 항목 | 버짓 | 소스 | trust |
|------|------|------|-------|
| Recent outbound | 제한 없음 (최대 3건) | activity_log (최근 2시간, `channel_post` / `message_sent`) | trusted |
| Pending human notifications | 500 | `human_notify` 이벤트 (최근 24시간) | trusted |

---

## Channel A: sender_profile

발신자의 사용자 프로필을 주입합니다.

- **소스**: `shared/users/{sender}/index.md` 직접 읽기
- **버짓**: 500 토큰
- **발신자 불명 시**: 건너뜀

---

## Channel B: recent_activity

최근 활동 타임라인을 주입합니다.

- **소스**: `activity_log/{date}.jsonl` + 공유 채널의 최신 게시물
- **버짓**: 1300 토큰

### 트리거별 필터링

| 트리거 | 제외되는 이벤트 타입 |
|--------|----------------------|
| `heartbeat` / `cron:*` | `tool_use`, `tool_result`, `heartbeat_start`, `heartbeat_end`, `heartbeat_reflection`, `inbox_processing_start`, `inbox_processing_end` |
| `chat` | `cron_executed` |

---

## Channel C: related_knowledge

RAG 벡터 검색으로 관련 지식을 주입합니다.

- **버짓**: 1200 토큰
- **검색 방식**: Dual-query (메시지 컨텍스트 + 키워드만)
- **검색 대상**: 개인 `knowledge/` + `shared_common_knowledge` 컬렉션
- **최소 스코어**: `config.json`의 `rag.min_retrieval_score` (기본값 0.3)

### trust 분리

검색 결과는 청크의 `origin`에 따라 trust 레벨로 분리됩니다:

| trust | 대상 | 처리 |
|-------|------|------|
| `medium` | 개인 knowledge, common_knowledge | 우선적으로 버짓 소비 |
| `untrusted` | 외부 플랫폼 유래 (`origin_chain`에 `external_platform` 포함) | 잔여 버짓으로 주입. `origin=ORIGIN_EXTERNAL_PLATFORM` 태그 부착 |

---

## Channel C0: important_knowledge

`[IMPORTANT]` 태그가 지정된 청크의 요약 포인터를 항상 주입합니다.

- **버짓**: 300 토큰
- **대상**: `knowledge/` 내 `[IMPORTANT]` 태그가 지정된 청크
- **주입 형식**: 요약 포인터만 (전문 아님). 상세는 `read_memory_file`로 조회
- **용도**: 중요한 비즈니스 규칙과 판단 기준의 확실한 회상

---

## Channel D: skill_match

메시지와 관련된 스킬 이름을 주입합니다.

- **버짓**: 200 토큰
- **매칭**: 3단계 (keyword → vocab → vector)
- **반환**: 스킬 이름만 (최대 5건). 전문은 `skill` 도구로 조회 (단계적 공개)
- **대상**: 개인 `skills/` + `common_skills/`

---

## Channel E: pending_tasks

태스크 큐 요약을 주입합니다.

- **버짓**: 500 토큰
- **소스**: `TaskQueueManager.format_for_priming()`
- **내용**:
  - `pending` / `in_progress` 태스크의 목록과 요약
  - `source: human` 태스크에 🔴 HIGH 마커
  - 30분 이상 업데이트 없는 태스크에 ⚠️ STALE 마커
  - 기한 초과 태스크에 🔴 OVERDUE 마커
  - 활성 병렬 태스크 (submit_tasks 배치)의 진행 상황
  - `task_results/`의 완료 태스크 결과
  - `status: failed` + `meta.executor == "taskexec"`인 실패 태스크

---

## Channel F: episodes

RAG 벡터 검색으로 관련 에피소드를 주입합니다.

- **버짓**: 500 토큰
- **검색 대상**: `episodes/` 컬렉션 (ChromaDB)
- **최소 스코어**: Channel C와 동일 (`rag.min_retrieval_score`)

---

## 동적 버짓 조정

`config.json`의 `priming.dynamic_budget: true` (기본값)로 활성화됩니다.

### 메시지 타입별 버짓

| 메시지 타입 | 버짓 | 설정 키 |
|-------------|------|---------|
| greeting | 500 | `priming.budget_greeting` |
| question | 1500 | `priming.budget_question` |
| request | 3000 | `priming.budget_request` |
| heartbeat (폴백) | 200 | `priming.budget_heartbeat` |

### Heartbeat 버짓 계산

```
heartbeat_budget = max(budget_heartbeat, context_window × heartbeat_context_pct)
```

- `heartbeat_context_pct`: 기본값 0.05 (컨텍스트 윈도우의 5%)
- 예: context_window=200000 → `max(200, 200000 × 0.05)` = 10000

---

## Hebbian LTP (장기 강화)

Priming에서 검색 및 표시된 청크는 `record_access()`를 통해 활성도가 업데이트됩니다. 이를 통해 자주 회상되는 기억의 망각이 방지됩니다 (Forgetting 엔진과의 연동).
