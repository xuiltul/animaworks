# 기억 통합 태스크 (일일)

{anima_name}, 기억을 정리할 시간입니다. 아래 절차를 따라 주세요.

## 오늘의 에피소드

{episodes_summary}

## 해결된 이벤트

{resolved_events_summary}

## 오늘의 액티비티 로그 (행동 기록)
{activity_log_summary}

※ 액티비티 로그는 행동의 기록이며 추론 과정은 포함되지 않습니다.
여기서 knowledge를 추출할 때는 다음 사항에 주의하세요:
- 확실히 사실이라고 판단할 수 있는 것만 knowledge/에 기록
- 추측이나 해석이 필요한 항목은 confidence: 0.5로 기록
- frontmatter에 `source: "activity_log"` 추가

{reflections_summary}

## 기존 knowledge 파일 목록

{knowledge_files_list}

## 병합 후보 (유사 파일 쌍)

{merge_candidates}

## 에러 패턴 (지난 24시간)

{error_patterns_summary}

---

## 작업 절차

### Step 1: 중복 파일 통합 (MUST — 최우선)

**병합 후보가 제시된 경우, 모든 쌍에 대해 통합을 수행하세요.**
또한, 위 파일 목록을 확인하여 같은 주제를 다루는 중복 파일을 직접 찾으세요.

통합 절차:
1. `read_memory_file`로 양쪽 내용을 확인
2. 정보를 합쳐서 `write_memory_file`로 한쪽에 기록
3. 불필요해진 쪽을 `archive_memory_file`로 아카이브
4. `[IMPORTANT]` 태그가 있으면 통합 대상 파일에도 유지

- "나중에 통합" 또는 "복잡해서 보류"는 금지. 지금 여기서 완료하세요
- 새 파일 생성보다 기존 파일로의 통합을 항상 우선하세요

### Step 2: 에피소드에서 knowledge 추출

오늘의 에피소드를 확인하고, 실질적인 정보가 있으면:
1. `search_memory`로 관련 기존 knowledge/ 및 procedures/ 검색
2. 관련 파일이 있으면 `read_memory_file`로 확인하고 `write_memory_file`로 추가·업데이트
3. 해당하는 기존 파일이 없는 경우에만 새 파일 생성

### Step 2.5: 에러 패턴 분석

위 "에러 패턴" 섹션을 확인하고, 반복적으로 발생하는 패턴이 있으면:
1. `search_memory`로 관련 기존 procedures/ 검색
2. 기존 절차가 있으면 `read_memory_file`로 확인하고 `write_memory_file`로 추가·업데이트
3. 해당하는 기존 파일이 없는 경우에만 `procedures/`에 새로 생성
4. 1회성 에러는 기록 불필요 (노이즈)

새로 생성 시 frontmatter:
```
---
created_at: "YYYY-MM-DDTHH:MM:SS"
confidence: 0.4
auto_consolidated: true
source: "error_trace_analysis"
version: 1
---
```

### Step 3: 품질 점검
- 업데이트하거나 생성한 내용이 에피소드의 사실과 모순되지 않는지 확인
- 파일명은 주제를 명확히 나타내는 이름을 사용하세요

## 추출해야 할 정보
- 구체적인 설정 값, 인증 정보 저장 위치
- 사용자 및 시스템 식별 정보
- 절차, 워크플로우, 프로세스 기록
- 팀 구성, 역할 분담, 지휘 체계
- 기술적 결정과 그 근거
- 해결된 이벤트에서 얻은 교훈과 절차

## 중요한 제약 사항
- **이 작업은 반드시 직접 수행하세요(MUST)**. `delegate_task`, `submit_tasks`, `send_message`를 사용하지 마세요. 기억 조작 도구만으로 작업을 완료하세요
- **Step 1 통합을 생략하지 마세요**. 중복 파일이 존재하는데 통합하지 않으면 실패로 간주합니다

## 참고 사항
- 인사만 포함된 대화나 실질적 정보가 없는 교환은 knowledge화하지 마세요
- [REFLECTION] 태그가 붙은 항목은 knowledge 추출을 우선적으로 검토하세요
- `[IMPORTANT]` 태그가 붙은 항목은 **반드시** knowledge/에 추출하세요(MUST). 기존 파일과 중복되면 추가 병합하세요. **본문에도 `[IMPORTANT]` 태그를 유지하세요**
- knowledge/를 새로 생성할 때는 YAML frontmatter를 추가하세요:
  ```
  ---
  created_at: "YYYY-MM-DDTHH:MM:SS"
  confidence: 0.7
  auto_consolidated: true
  success_count: 0
  failure_count: 0
  version: 1
  last_used: ""
  ---
  ```
- 완료 후 실시 내용의 요약을 출력하세요 (통합한 쌍 수와 아카이브한 파일 수 포함)
