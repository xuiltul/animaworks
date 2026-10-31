<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/operations/dev-team.md -->
<!-- i18n: source-sha256=33c335399f699ae80e7322151309867c19a8b99faaf967e288e2a3e7f863b69f generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인된 커밋: 581e20f1

# 개발 팀 구성

AnimaWorks에는 개발 팀을 위한 역할 템플릿이 포함되어 있다. 템플릿은 팀을 시작하기 위한 출발점이며, 구체적인 개발 절차나 지식은 각 팀의 운영에 맞게 정비한다.

## 역할 예시

| 템플릿 | 주요 역할 |
|---|---|
| `dev-lead` | 작업을 분해하고, 구현 담당이나 조사 담당에게 위임하며, 품질 확인과 에스컬레이션을 수행한다. |
| `dev-engineer` | 격리된 작업 환경에서 구현하고, 테스트와 변경 내용을 보고한다. |
| `dev-researcher` | 읽기 전용으로 조사하고, 근거와 미확인 사항을 구분하여 보고한다. |

역할의 인원수나 계층은 고정이 아니다. 소규모 팀에서는 한 사람이 여러 책임을 맡지 않고, 필요한 담당만 생성한다.

## 생성

```sh
animaworks anima create --name lead --template dev-lead
animaworks anima create --name engineer --template dev-engineer --supervisor lead
animaworks anima create --name researcher --template dev-researcher --supervisor lead
```

이름은 예시이다. 생성 후에는 조직도, 모델, 권한, heartbeat/cron 설정을 확인하고, 템플릿의 지침이 팀의 승인·리뷰 기준과 일치하도록 조정한다. Anima 생성 방법은 [CLI 참조](../reference/cli.md)를 참조한다.

## 개발 파이프라인과의 연결

GitHub webhook을 활성화하면 대상 이벤트를 지정한 dispatcher Anima로 알림을 보낼 수 있다. 알림 처리와 구현·리뷰 단계는 별개의 책임이며, 수신 이벤트가 자동으로 안전한 변경이나 병합을 보장하는 것은 아니다. 리포지토리의 권한, CI, 리뷰 승인, 브랜치 보호는 GitHub 쪽에서도 구성한다.

팀이 다루는 작업에서는 원칙적으로 작업 트리를 분리하고, 변경 대상을 명확히 하여 테스트를 실행한다. 조사 보고에는 참조한 경로와 심볼을 포함하고, 구현 결과에는 검증 내용과 남은 제약을 덧붙인다.
