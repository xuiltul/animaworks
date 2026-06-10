# 기억 시스템 가이드

Anima 기억의 구조, 종류, 활용법에 대한 레퍼런스입니다.
기억의 검색, 기록, 정리 방법을 확인할 때 참조하세요.

## 기억의 전체 구조

당신의 기억은 인간 뇌의 기억 모델에 대응하는 여러 종류로 구성됩니다:

| 기억 유형 | 디렉토리 | 인간 비유 | 내용 |
|-----------|----------|----------|------|
| **단기 기억** | `shortterm/` | 작업 기억 | 최근 대화의 맥락 |
| **일화 기억** | `episodes/` | 경험 기억 | 언제 무엇을 했는지 |
| **의미 기억** | `knowledge/` | 지식 | 배운 것, 노하우 |
| **절차 기억** | `procedures/` | 체득한 절차 | 단계별 수행 절차 |
| **스킬** | `skills/` | 특기/전문 기술 | 실행 가능한 절차서 |

추가로 모든 Anima가 공유하는 기억도 있습니다:

| 공유 기억 | 경로 | 내용 |
|-----------|------|------|
| **공유 지식** | `common_knowledge/` | 프레임워크 레퍼런스 (이 파일 자체 포함) |
| **공통 스킬** | `common_skills/` | 모든 Anima가 사용할 수 있는 스킬 |
| **조직 공유 지식** | `shared/common_knowledge/` | 조직이 운영 중 축적한 지식 |
| **사용자 프로필** | `shared/users/` | Anima 간 공유 사용자 정보 |

---

## 단기 기억 (shortterm/)

**최근 대화와 세션의 맥락**을 보유합니다. 인간의 작업 기억에 해당합니다.

- Chat용 (`shortterm/chat/`)과 Heartbeat용 (`shortterm/heartbeat/`)으로 분리
- 컨텍스트 윈도우 사용률이 임계값을 초과하면 오래된 부분이 자동으로 외부화됨
- 세션 간 맥락 연속성을 위해 사용됨

단기 기억은 직접 조작할 필요가 없습니다. 프레임워크가 자동으로 관리합니다.

---

## 일화 기억 (episodes/)

**"언제 무엇을 했는지"의 일별 로그**입니다. 인간의 경험 기억에 해당합니다.

- 날짜별 파일 (예: `2026-03-09.md`)에 자동 기록됨
- "지난주에 뭘 했지?", "이 문제에 전에 대응한 적 있나?"를 떠올릴 때 사용
- Consolidation을 통해 패턴과 교훈이 `knowledge/`로 정제됨

### 기억 기록

```
write_memory_file(path="episodes/2026-03-09.md", content="...")
```

### 기억 검색

```
search_memory(query="Slack API 접속 테스트", scope="episodes")
```

---

## 의미 기억 (knowledge/)

**배운 지식, 노하우, 패턴**입니다. 인간이 "알고 있는 것"에 해당합니다.

- 일화에서 추출된 교훈과 패턴
- 기술 메모, 대응 방침, 판단 기준
- 일별 Consolidation으로 자동 축적되며, 직접 능동적으로 기록할 수도 있음
- **재고정화**: front matter에서 `failure_count >= 2`이고 `confidence < 0.6`인 knowledge는 절차와 마찬가지로 LLM 개정 대상이 될 수 있음

예:
- "Slack API 레이트 제한은 Tier 1에서 1req/sec"
- "이 고객은 월요일에 연락이 많은 편"
- "배포 전 확인 항목 체크리스트"

### 기억 기록

```
write_memory_file(path="knowledge/slack-api-notes.md", content="...")
```

### 기억 검색

```
search_memory(query="Slack API 레이트 제한", scope="knowledge")
```

---

## 절차 기억 (procedures/)

**"어떻게 하는지"의 단계별 절차서**입니다. 인간의 "몸이 기억한 절차"에 해당합니다.

- 문제 해결 절차, 정형 작업 흐름
- `issue_resolved` 이벤트에서 자동 생성될 수 있음 (confidence 0.4)
- **스킬만큼 전면 보호되지는 않음**: 메타데이터에 따라 망각 파이프라인 대상이 될 수 있음
- **재고정화**: front matter에서 `failure_count >= 2`이고 `confidence < 0.6`이면 LLM이 절차를 개정할 수 있음. 개정 후 카운터를 리셋하고 버전을 올리며 기존 버전은 `archive/`로 이동함

예:
- "SSL 인증서 갱신 절차"
- "신규 Anima 온보딩 절차"
- "운영 장애 시 에스컬레이션 절차"

### 기억 기록

```
write_memory_file(path="procedures/ssl-renewal.md", content="...")
```

### 기억 검색

```
search_memory(query="SSL 인증서 갱신", scope="procedures")
```

---

## 스킬 (skills/)

**실행 가능한 절차서 및 도구 사용 가이드**입니다. "특기"에 해당합니다.

- 개인 스킬 (`skills/`)과 공통 스킬 (`common_skills/`)이 있음
- 필요한 스킬은 active skill context, Skill Router, Skill Hub 또는 `read_memory_file(path="...")`로 읽음
- 스킬 본문을 항상 전부 읽을 필요는 없음. 먼저 이름, 설명, 포인터를 보고 필요할 때만 전문을 읽음
- 실적이 있는 `procedures/`는 probation skill 또는 quarantine skill로 승격될 수 있음
- **벡터 저장소에서는 항상 망각 대상에서 제외됨** (`skills` / `shared_users` 유형 보호)

### 스킬 확인

```
read_memory_file(path="skills/newstaff/SKILL.md")  # 스킬 전문 가져오기
```

### 스킬 생성

```
create_skill(skill_name="deploy-procedure", description="운영 배포 절차", body="...")
```

---

## 기억의 자동 프로세스

### Priming (자동 상기)

대화를 시작할 때마다 Priming 엔진이 관련 기억을 병렬 검색하여, 지금 문맥에 필요한 내용만 시스템 프롬프트에 주입합니다. 모든 기억을 매번 읽는 방식이 아닙니다.

| 정보원 | 검색 대상 | 기본 예산 가이드 |
|--------|----------|----------------|
| 발신자 프로필 | 상대방의 사용자 정보 | 500 |
| 최근 활동 | 통합 activity log 타임라인 + 공유 채널 | 1300 |
| 중요 지식 | `[IMPORTANT]` knowledge의 요약 포인터 | 500 |
| 관련 지식 | 개인 `knowledge` + 공유 `common_knowledge` RAG | 1000 |
| 보류 태스크 | TaskBoard + 태스크 큐 + 실행 중 태스크 + overflow inbox + task_results | 500 |
| 일화 | `episodes` RAG 검색 | 800 |
| 그래프 문맥 | memory backend의 community context와 recent facts | 500 |

스킬과 절차는 필요할 때 active skill context, Skill Hub, `read_memory_file`, 또는 `search_memory(scope="skills")`로 읽습니다. 스킬 본문 전체가 항상 자동 주입되는 설계는 아닙니다.

일반 경로에서는 인사, 질문, 요청, heartbeat 등 메시지 종류에 따라 전체 상한이 달라집니다. `config.json`의 `priming.budget_*`와 `heartbeat_context_pct`로 조정할 수 있습니다.

**[IMPORTANT]** `[IMPORTANT]` 태그가 붙은 knowledge는 요약 포인터로 표시됩니다. 요약만 표시되며, 상세 내용은 `read_memory_file`로 확인하세요. 중요한 업무 규칙을 knowledge/로 이동할 때는 `[IMPORTANT]`를 붙여 주세요.

Priming은 자동으로 동작하므로 명시적인 조작은 불필요합니다.

### 행동 전 기억 확인

외부 발신, 채널 게시, 사람에게 알림, 기억 쓰기처럼 부작용이 있는 작업에서는 관련 `[ACTION-RULE]`과 필요한 기억을 읽었는지 확인될 수 있습니다. 중단되면 안내된 기억을 `read_memory_file`로 읽은 뒤 다시 실행하세요.

### Consolidation (기억 통합)

기억을 자동으로 정리하고 정제하는 프로세스:

| 빈도 | 처리 |
|------|------|
| **일별** | 일화 → 지식 (패턴 및 교훈 추출) |
| **일별** | 문제 해결 → 절차 (수정 피드백에서 절차서 자동 생성) |
| **주별** | 지식 병합 + 일화 압축 |

### Forgetting (능동적 망각)

기억을 무한히 쌓으면 검색 정확도가 떨어지므로 3단계로 능동적으로 망각합니다:

| 단계 | 빈도 | 처리 |
|------|------|------|
| Synaptic downscaling | 일별 | 90일간 미접근, 3회 미만 참조된 청크를 마킹 |
| Neurogenesis reorganization | 주별 | 유사도 0.80 이상의 저활성 청크를 병합 |
| Complete forgetting | 월별 | 90일 이상 저활성 청크를 아카이브 및 삭제 |

**보호 대상**: 중요한 knowledge/procedures, 성숙한 절차, 보호 지정된 절차, `skills/`, `shared/users/`

---

## 기억 도구의 활용

| 하고 싶은 것 | 도구 | 예시 |
|-------------|------|------|
| 키워드로 기억 검색 | `search_memory` | `search_memory(query="API 설정", scope="all")` |
| 특정 파일 읽기 | `read_memory_file` | `read_memory_file(path="knowledge/api-notes.md")` |
| 기억 기록 | `write_memory_file` | `write_memory_file(path="knowledge/new-insight.md", content="...")` |
| 불필요한 기억 정리 | `archive_memory_file` | `archive_memory_file(path="knowledge/outdated.md")` |

### scope (검색 범위) 선택

| scope | 검색 대상 | 사용 시점 |
|-------|----------|----------|
| `knowledge` | 지식, 노하우 | "이것에 대해 알고 있는 게 있나?" |
| `episodes` | 과거 행동 로그 | "전에 이걸 한 적이 있나?" |
| `procedures` | 절차서 | "이 작업의 절차는?" |
| `common_knowledge` | 공유 레퍼런스 | "프레임워크 사양은?" |
| `activity_log` | 최근 활동 로그 (도구 실행 결과, 메시지 등) | "방금 읽은 이메일 내용", "이전 검색 결과" |
| `all` | 위의 모든 항목 (벡터 검색 + activity_log BM25를 RRF로 통합) | 폭넓은 검색 |

---

## RAG (벡터 검색)의 구조

기억 검색에는 RAG (Retrieval-Augmented Generation)가 사용됩니다:

1. **인덱싱**: 모든 기억 파일이 embedding 벡터로 변환되어 ChromaDB에 저장됨
2. **검색**: 쿼리 텍스트를 벡터화하여 유사도가 높은 기억 청크를 조회
3. **그래프 확산**: NetworkX 그래프 기반 확산 활성화로 관련 주변 기억도 추출
4. **증분 갱신**: 변경된 파일만 재인덱싱하므로 대량의 기억이 있어도 빠른 처리 가능
5. **복구**: ChromaDB 또는 벡터 검색 불일치가 발생하면 RAG repair가 `vectordb`를 격리하고 기억 파일에서 인덱스를 재구축할 수 있음

RAG는 `search_memory`를 호출하면 자동으로 사용됩니다. 구조를 의식할 필요는 없지만, **검색 정확도를 높이는 팁**:
- 구체적인 키워드를 포함한 쿼리 사용
- 기억을 기록할 때 제목과 내용을 명확하게 작성
- 관련 정보는 같은 파일에 모아서 정리
