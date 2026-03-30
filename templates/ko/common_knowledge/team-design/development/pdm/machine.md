# PdM — machine 활용 패턴

## 기본 규칙

1. **계획서를 먼저 쓴다** — 짧은 인라인 지시만으로 실행 금지. 계획서 파일을 넘긴다.
2. **출력은 초안** — machine 출력은 반드시 검증하고 다음 단계 전에 `status: approved`로 둔다.
3. **저장 위치**: `state/plans/{YYYY-MM-DD}_{요약}.{type}.md`(`/tmp/` 금지)
4. **레이트**: chat 세션당 5회, heartbeat당 2회
5. **machine은 인프라 접근 불가** — 기억·메시지·조직 정보는 계획서에 포함한다.

---

## 개요

PdM은 **조사의 수발은 machine에 맡기고, 계획 판단은 본인이 한다**.

- 조사·정보 수집 → machine 위임
- 결과 해석·우선순위·구현 방침 결정 → PdM 판단
- 계획서(plan.md) 작성 → PdM이 직접 작성

---

## 1단계: 조사

### 1: 조사 계획서 작성(PdM)

machine에 무엇을 조사할지 명확히 한 계획서를 만든다.

```bash
write_memory_file(path="state/plans/{date}_{요약}.investigation.md", content="...")
```

### 2: machine에 조사 실행

```bash
animaworks-tool machine run \
  "$(cat $ANIMAWORKS_ANIMA_DIR/state/plans/{조사계획서})" \
  -d /path/to/worktree
```

결과를 `state/plans/{date}_{요약}.investigation.md`로 저장(`status: draft`).

### 3: 조사 결과 검증

- [ ] 조사 목적에 답이 있는가
- [ ] 사실과 추측이 구분되는가
- [ ] 중요한 누락이 없는가
- [ ] 추가 조사가 필요한가

수정·보완 후 `status: approved`.

## 2단계: 계획서 작성

### 4: plan.md 작성(PdM 판단)

investigation.md를 바탕으로 **PdM이** plan.md를 쓴다.  
「구현 방침」「우선순위」「제약」은 PdM 판단의 핵이며 machine에 쓰게 하면 안 된다(NEVER).

### 5: 위임

`status: approved`를 확인한 뒤 Engineer에게 `delegate_task`로 넘긴다.

---

## 조사 계획서·plan.md 템플릿

구조는 `common_knowledge/operations/machine/workflow-pdm.md`(영문) 및 `templates/en/team-design/development/pdm/machine.md`의 마크다운 블록과 동일하게 맞춘다.  
세부 문장 예시는 해당 파일을 참고한다.

---

## 제약

- 조사 계획서는 MUST: PdM이 작성
- plan.md의 판단 섹션(구현 방침·우선순위·제약)은 MUST: PdM이 작성
- investigation.md는 machine이 생성해도 PdM 검증·승인 전까지 초안
- `status: approved` 없는 plan.md를 Engineer에게 넘기면 안 됨(NEVER)
