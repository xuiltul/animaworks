# 법무 풀 팀 — 팀 개요

## 3개 역할 구성

| 역할 | 책임 | 권장 `--role` | `speciality` 예 | 상세 |
|--------|------|--------------|-----------------|------|
| **Legal Director** | 분석 계획·계약서 스캔·판단·최종 승인 | `manager` | `legal-director` | `legal/director/` |
| **Legal Verifier** | 독립 검증·낙관적 편향 검출·carry-forward 검증 | `researcher` | `legal-verifier` | `legal/verifier/` |
| **Precedent Researcher** | 법령·판례·업계 표준 수집·근거 뒷받침 | `general` | `legal-researcher` | `legal/researcher/` |

한 Anima에 전 과정을 몰아넣으면 자기 검토의 사각지대(낙관적 편향)·지적 사항의 소실(silent drop)·컨텍스트 비대화가 발생한다.

각 역할 디렉터리에는 `injection.template.md`(injection.md 초안), `machine.md`(machine 활용 패턴, 해당 역할만), `checklist.md`(품질 체크리스트)가 있다.

> 기본 원칙 상세: `team-design/guide.md`

## 핸드오프 체인

```
Director → analysis-plan.md (approved) + carry-forward tracker 참조
  → machine으로 계약서 전문 스캔 → Director가 검증
    → audit-report.md (reviewed)
      → Verifier (독립 검증) ─┐
      → Researcher (근거 검증) ─┤ ← 병렬 실행 가능
        └─ 지적 있음 → Director에게 반려
        └─ 둘 다 APPROVE → Director → carry-forward tracker 갱신 → call_human → 사람이 최종 확인
```

### 인수인계 문서

| 송신 → 수신 | 문서 | 조건 |
|----------------|------------|------|
| Director → Verifier/Researcher | `audit-report.md` + `analysis-plan.md` | `status: reviewed` |
| Verifier → Director | `verification-report.md` | `status: approved` |
| Researcher → Director | `precedent-report.md` | `status: approved` |

### 운영 규칙

- **수정 사이클**: Critical → 전면 재검증(Verifier·Researcher 모두에 재의뢰) / Warning → 차분 확인만 / 3왕복으로 해소되지 않음 → 사람에게 에스컬레이션
- **carry-forward tracker**: 사건의 모든 버전에 걸쳐 지적 사항을 추적한다. 전번 감사의 지적이 언급 없이 사라지는 것(silent drop)은 금지
- **machine 실패 시**: `current_state.md`에 기록 → 다음 heartbeat에서 재평가

## 스케일링

| 규모 | 구성 | 비고 |
|------|------|------|
| 솔로 | Director가 전 역할 겸임(checklist로 품질 보증) | NDA 확인, 정형 계약 검토 |
| 페어 | Director + Verifier | SPA 수정판 검토, 중위험 계약 |
| 풀 팀 | 본 템플릿대로 3명 | SPA 최초 감사, M&A DD, 고위험 사건 |

## 개발 팀과의 대응 관계

| 개발 팀 역할 | 법무 팀 역할 | 대응하는 이유 |
|----------------|----------------|-------------|
| PdM(조사·계획·판단) | Director(분석 계획·판단) | 「무엇을 분석할지」를 결정하는 사령탑 |
| Engineer(구현) | Director + machine(계약서 스캔) | Director가 machine으로 분석을 실행. 독립 Anima 불필요 |
| Reviewer(정적 검증) | Verifier(독립 검증) | 「실행과 검증의 분리」의 핵. 가장 중요한 분리 포인트 |
| Tester(동적 검증) | Researcher(근거 검증) | 「업계 표준」「판례」 등의 주장을 실제 데이터로 뒷받침 |

## Carry-forward Tracker — 지적 사항 추적표

계약 버전이 갱신될 때마다 이 표를 갱신한다. 전번 감사의 모든 지적 사항을 관리하고, silent drop을 구조적으로 방지한다.

### 추적 규칙

- 전번 감사의 모든 지적 사항을 이 표로 관리한다
- 신판 수령 시 모든 항목의 상태를 갱신한다
- 「해소」가 아닌 항목은 다음 검토에서 반드시 언급한다
- silent drop(언급 없이 소멸)은 금지

### 템플릿

```markdown
# 지적 사항 추적표: {사건명}

| # | 최초 발생일 | 항목 | 최초 리스크 | v1 상태 | v2 상태 | v3 상태 | 현재 잔존 리스크 |
|---|--------|------|----------|------------|------------|------------|--------------|
| C-1 | {날짜} | {지적 내용} | Critical | {대응 상황} | {대응 상황} | — | {현재 리스크} |
| C-2 | {날짜} | {지적 내용} | Critical | {대응 상황} | {대응 상황} | — | {현재 리스크} |
| H-1 | {날짜} | {지적 내용} | High | {대응 상황} | {대응 상황} | — | {현재 리스크} |
| M-1 | {날짜} | {지적 내용} | Medium | {대응 상황} | {대응 상황} | — | {현재 리스크} |

상태 범례:
- 미수정: 전번과 변경 없음
- 해소: 수정되어 리스크가 제거됨
- 부분 해소: 수정되었으나 잔존 리스크 있음(잔존 리스크란에 상세)
- 악화: 수정으로 새로운 리스크가 발생하거나 기존 리스크가 증대
```
