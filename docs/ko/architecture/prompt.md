<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/prompt.md -->
<!-- i18n: source-sha256=4efd5e6044c341500cee440726a5e0ae86fc499c7ff7c73f6109d91d97138208 generated=2026-10-16 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: b304b7dc

# System prompt 구축

`core/prompt/builder.py`이 기억, 실행 모드, trigger, 현재 요청을 바탕으로 system prompt를 구축하고, `core/prompt/assembler.py`이 section을 XML boundary tag와 함께 조립하여 budget에 맞춘다. 고정 부분을 앞부분에 모으고, turn마다 달라지는 상태 정보를 뒤에 배치한다.

## section의 구성과 순서

builder는 6개의 논리 그룹을 만든다. 조립 순서는 그룹 1, 2, 4, 5, 6, 마지막으로 동적인 그룹 3이다.

1. **환경과 행동 규칙** — runtime 정보, workspace, Anima의 `IDENTITY.md` / `SOUL.md`, `AGENTS.md`, behavior rules.
2. **Anima 자신의 정보** — bootstrap 상태, 회사의 vision, speciality, permissions.
3. **기억과 능력** — memory guide, trigger / mode에 맞춘 tool guide, 사용 가능한 외부 tool, active skill context, skill catalog.
4. **조직과 커뮤니케이션** — 조직 관계, 내부 메시지, 인간에게 알리는 방법.
5. **메타 설정** — chat 시의 emotion 지침이나 특정 engine용 응답 규칙.
6. **현재 상황** — 현재 시각, `current_state.md`, 최근 resolution, priming, 해당하는 인간 알림, 단기 기억.

그룹 3에는 turn마다 달라지는 정보가 모인다. 앞선 그룹을 안정적인 prefix로 취급함으로써 provider의 prompt cache를 재사용하기 쉽게 한다. context window가 작은 경우 prompt tier에 따라 일부 환경 정보나 optional section을 생략한다.

## 스킬과 tool guide

스킬 목록은 Anima 고유, 공통, procedure 등을 인덱싱한 후 조립한다. chat에서는 요청문을 skill router에 전달하여 관련 후보를 우선하고, 적절한 후보가 없는 경우 일반 catalog를 사용한다. catalog의 건수는 기본적으로 최대 3건으로 제한한다. chat에서는 활성화된 스킬의 본문도 context에 추가된다. 자동 실행에서는 인간의 승인이 필요한 스킬을 목록에서 제외한다.

tool guide는 heartbeat, MCP를 사용하는 engine, 그 외의 engine에 따라 전환된다. 스킬 본문이나 tool의 실제 입출력은 요청에 따라 로드하고, system prompt에는 사용 가능한 진입점과 필요한 규칙을 배치한다.

## budget과 비대화 대처

assembler는 section의 우선순위를 바탕으로 예산을 배분하고, elastic section은 Markdown의 문단 단위로 줄인다. 개별 문단을 중간에 끊지 않고, hard ceiling을 초과하는 경우 우선순위가 낮은 rigid section도 제외한다. 기본 일반 target은 6,000 token, 상한은 context window의 35%이며, 설정으로 변경할 수 있다.

`AGENTS.md`의 크기 경고는 consolidation 중에만, 설정된 문자 수 threshold를 초과한 경우에 추가된다. 스킬 catalog에도 건수 상한이 있다. 전체·개별 설정 목록은 [설정 참조](../reference/config.md)를 참조한다.

## template 해결

Markdown template은 `templates/{locale}/` 아래에 배치한다. `core/paths.py`의 `resolve_template_path`는 지정 locale, `en`, `ja`, `_shared` 순서로 탐색한다. 배포 형태에 따른 runtime template path는 이 탐색 이후의 호환 fallback으로 취급된다.
