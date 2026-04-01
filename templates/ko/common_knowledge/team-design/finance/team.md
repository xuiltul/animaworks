# 재무 풀 팀 — 팀 개요

## 4개 역할 구성

| 역할 | 책무 | 권장 `--role` | `speciality` 예 | 상세 |
|------|------|--------------|-----------------|------|
| **Finance Director** | 분석 계획·재무 판단·수치 검증·최종 승인 | `manager` | `cfo`, `finance-director` | `finance/director/` |
| **Financial Auditor** | 독립 검증·가정 검증·Data Lineage 검증 | `researcher` | `financial-auditor` | `finance/auditor/` |
| **Data Analyst** | 소스 데이터 추출·구조화·다단계 검증 | `general` | `data-analyst` | `finance/analyst/` |
| **Market Data Collector** | 외부 시장 데이터·벤치마크·참조 가격 수집 | `general` | `market-data` | `finance/collector/` |

한 Anima에 전 과정을 몰아두면 셀프 리뷰의 사각(해석의 낙관적 편향)·차이의 소실(silent drop)·컨텍스트 비대화가 발생한다.

각 역할 디렉터리에 `injection.template.md`(injection.md 초안), `machine.md`(machine 활용 패턴, 해당 역할만), `checklist.md`(품질 체크리스트)가 있다.

> 기본 원칙 상세: `team-design/guide.md`

## 핸드오프 체인

```
Analyst (소스 데이터 추출) + Collector (외부 데이터 수집) ← 병렬 실행 가능
  → Director → analysis-plan.md (approved) → machine로 분석 실행
    → analysis-report.md (reviewed)
      → Auditor (독립 검증)
        └─ 지적 있음 → Director에 반려
        └─ APPROVE → Director → Variance Tracker 갱신 → call_human → 사람이 최종 확인
```

### 인계 문서

| 송신 → 수신 | 문서 | 조건 |
|------------|------|------|
| Analyst/Collector → Director | 소스 데이터 + 추출 검증 결과 | 검증 완료 |
| Director → Auditor | `analysis-report.md` + `analysis-plan.md` | `status: reviewed` |
| Auditor → Director | `audit-report.md` | `status: approved` |

### 운영 규칙

- **수정 사이클**: Critical → 전체 재검증(Auditor에 재의뢰) / Warning → 차분 확인만 / 3회 왕복으로 해소되지 않으면 → 사람에게 에스컬레이션
- **Variance Tracker**: 분석에서 검출한 중요 차이를 월을 넘어 추적한다. 전회에 플래그한 차이가 다음 보고서에서 언급 없이 사라지는 것(silent drop)은 금지
- **Data Lineage Rule**: analysis-report.md 내 모든 수치는 소스 데이터까지 소급할 수 있어야 한다. 추정값에는 「추정」 마커 필수
- **machine 실패 시**: `current_state.md`에 기록 → 다음 heartbeat에서 재평가

## 스케일링

| 규모 | 구성 | 비고 |
|------|------|------|
| 솔로 | Director가 전 역할 겸임(checklist로 품질 보장) | 정형 월간 보고, 단일 법인 분석 |
| 페어 | Director + Auditor | 중요한 판단을 포함하는 분석, 복수 법인 비교 |
| 트리오 | Director + Auditor + Analyst(Collector 겸임) | 데이터량이 많은 건 |
| 풀 팀 | 본 템플릿대로 4명 | 연결 분석, 대형 건, 포트폴리오 평가 |

## 개발 팀·법무 팀과의 대응 관계

| 개발 팀 역할 | 법무 팀 역할 | 재무 팀 역할 | 대응하는 이유 |
|-------------|-------------|---------------|--------------|
| PdM(조사·계획·판단) | Director(분석 계획·판단) | Director(분석 계획·판단) | 「무엇을 분석할지」를 결정하는 사령탑 |
| Engineer(구현) | Director + machine | Director + machine | Director가 machine로 분석을 실행. 독립 Anima 불필요 |
| Reviewer(정적 검증) | Verifier(독립 검증) | Auditor(독립 검증) | 「실행과 검증의 분리」의 핵. 가장 중요한 분리 지점 |
| Tester(동적 검증) | Researcher(근거 검증) | Collector(외부 데이터 수집) | 외부 정보로 뒷받침 |
| — | — | Analyst(데이터 추출) | 재무 고유. 소스 데이터의 정확한 추출·구조화 |

## Monthly Variance Tracker — 월간 차이 추적표

분석에서 검출한 중요 차이를 월을 넘어 추적한다. 전회에 플래그한 차이가 다음 보고서에서 언급 없이 사라지는 것(silent drop)을 구조적으로 방지한다.

### 추적 규칙

- 중요 차이(임계값 초과 변동)를 검출하면 이 표에 등록한다
- 다음 분석 시 전 항목의 상태를 갱신한다
- 「해소」가 아닌 항목은 다음 보고서에서 반드시 언급한다
- silent drop(언급 없이 소멸)은 금지

### 템플릿

```markdown
# 월간 차이 추적표: {대상명}

| # | 최초 검출 월 | 계정과목 | 최초 차이율 | M월 상태 | M+1월 상태 | M+2월 상태 | 현재 잔존 리스크 |
|---|---------|---------|----------|------------|-------------|-------------|--------------|
| V-1 | {월} | {과목} | {차이율} | {대응 상황} | {대응 상황} | — | {리스크 평가} |
| V-2 | {월} | {과목} | {차이율} | {대응 상황} | {대응 상황} | — | {리스크 평가} |

상태 범례:
- 해소: 원인 특정 완료·조치 완료, 리스크가 제거됨
- 지속 모니터링: 원인은 파악되었으나 재발 리스크 있음(추적 기간과 판단 기준 병기)
- 조사 중: 원인 미특정
- 악화: 차이가 확대되었거나 새로운 리스크가 발생
```
