<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/memory/index.md -->
<!-- i18n: source-sha256=ff2e049713f751fce015b910c2fab2c4ee9c63821df9794c5156bdf2c5dbe844 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 193a5e72

# 기억 시스템

AnimaWorks의 기억은 필요한 것만 런타임 컨텍스트로 꺼내는 파일 기반 아카이브이다. Anima는 기억을 읽고 쓰며, 기록의 통합이나 정리를 판단한다. 프레임워크는 검색용 인덱스나 후보를 준비하지만, 오래된 기록을 일률적인 기계 규칙으로 이동·삭제하는 설계는 아니다. 자동 인덱스의 역할은 [의도적 회상과 검색](retrieval.md)을 참조한다.

## 인간의 기억 모델과의 대응

| 인간의 기억 | AnimaWorks | 역할 |
|---|---|---|
| 작업 기억 | LLM의 컨텍스트 | 현재 판단에 필요한 정보를 일시적으로 유지한다. |
| 에피소드 기억 | `episodes/` | 언제 무슨 일이 있었는지를 시계열로 기록한다. |
| 의미 기억 | `knowledge/` | 경험에서 얻은 지식, 교훈, 방침을 유지한다. |
| 절차 기억 | `procedures/`, `skills/` | 작업 진행 방식이나 재사용 가능한 절차를 유지한다. |
| 구조화된 사실 | `facts/` | 대화나 기록에서 추출한, entity와 시점을 수반하는 atomic fact를 저장한다. |

장기 기억의 원본은 파일이며, 검색용 벡터 인덱스나 BM25 인덱스는 재생성 가능한 파생 데이터이다. 자동 회상과 명시적 검색의 구분은 [자동 회상](priming.md)과 [검색](retrieval.md)을 참조한다.

## 기억 디렉터리

Anima별 기억 영역은 `~/.animaworks/animas/{name}/` 아래에 있다.

| 디렉터리 | 역할 |
|---|---|
| `episodes/` | 매일의 사건이나 작업의 에피소드 기록. |
| `knowledge/` | 의미 기억. 지식, 교훈, 방침 등을 Markdown으로 유지한다. |
| `procedures/` | 절차를 Markdown으로 유지하고, 실행 결과를 사용해 신뢰도나 버전을 추적한다. |
| `facts/` | 날짜 단위의 JSONL에 atomic fact를 저장한다. |
| `skills/` | 절차서나 도구 활용의 재사용 가능한 스킬 문서를 유지한다. |
| `state/` | 현재 상태나, fact에서 재구성 가능한 entity registry 등의 상태 데이터를 유지한다. |
| `shortterm/` | 세션 중의 단기 데이터와 스트리밍 저널을 유지한다. |

대인 프로필은 공유 영역의 `shared/users/`에 두며, Anima 고유의 장기 기억과는 별도로 참조된다.

## Frontmatter

`knowledge/`과 `procedures/`의 Markdown에는 YAML frontmatter를 붙일 수 있다. `core/memory/frontmatter.py`는 공통의 읽기·쓰기, 복구, 기본값 보완을 담당한다. 키를 일률적으로 필수로 하는 닫힌 스키마가 아니라, 내용이나 용도에 따른 메타데이터이다.

`knowledge/`에서는 `created_at`, `updated_at`, `confidence`, `source_episodes`, `auto_consolidated`, `version`가 작성·유지 시에 사용된다. `success_count`, `failure_count`은 `confidence`와 함께 보고된 유용성을 추적하고, `valid_from`·`valid_until`는 사실의 유효 기간, `description`는 검색 결과의 요약에 사용할 수 있다. `always_prime: true`은 자동 회상에 대한 명시적 opt-in이다. 의미와 적용 범위는 [자동 회상](priming.md)을 참조한다.

`procedures/`에서는 `description`, `confidence`, `success_count`, `failure_count`, `version`, `created_at`, `updated_at`가 절차의 설명과 이용 결과 추적에 사용된다. `auto_distilled`는 자동 증류로 만들어진 절차를 나타내고, `protected`은 보호 지정이다. `core/memory/skill_metadata.py`은 절차의 메타데이터를 skill catalog와 공통 형식으로 다룬다. `description`이 없는 경우에는 문서의 `## 概要`에서 보완할 수 있다. 스킬 문서의 메타데이터 읽기는 `core/skills/loader.py`이 담당한다.

설정 항목의 목록과 기본값은 [설정 참조](../reference/config.md)를 참조한다. 매일의 통합과 절차의 이용 추적은 [통합과 망각](consolidation.md)에 기재한다.
