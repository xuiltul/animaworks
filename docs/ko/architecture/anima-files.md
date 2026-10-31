<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/architecture/anima-files.md -->
<!-- i18n: source-sha256=55b82aac73f8908843de4a8ce75c5ce2d5ac3323dfee2ee456625e5b71b0726d generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> 확인한 커밋: b304b7dc

# Anima 파일

각 Anima는 `~/.animaworks/animas/{name}/`에 디렉터리를 가진다. 개성과 행동 방침, 실행 설정은 텍스트와 JSON으로 나누어 저장하고, 세션 중 업데이트되는 정보는 `state/`에 둔다.

## 정의와 설정

| 파일 | 역할 |
|---|---|
| `IDENTITY.md` | 이름, 기본 프로필, 말투 등 외부에 제시하는 지속적인 정체성을 기술한다. |
| `SOUL.md` | 가치관, 원칙, 내면의 성격 등 행동의 기반을 기술한다. |
| `AGENTS.md` | 전문성, 추가 행동 지침, 작업 지침을 기술하고 system prompt에 포함한다. 구 `injection.md`. |
| `permissions.json` | 개별 Anima의 도구, 명령어, 파일 접근에 관한 권한을 나타낸다. |
| `status.json` | 활성 상태, role, 담당 업무, 모델, 인증, supervisor, heartbeat 활성화 등 런타임에 결정되는 per-anima 정보를 보유한다. |
| `HEARTBEAT.md` | heartbeat 지침과 활동 시간대를 기술한다. 실행 간격은 전체 설정 또는 `status.json`의 값에 따라 결정되며, 이 파일의 본문에서는 설정하지 않는다. |
| `BOOTSTRAP.md` | 최초 시작 시 자기소개와 초기화 지침을 기술한다. |
| `cron.md` | 정기 실행할 작업을 제목별로 기술한다. 일정과 실행 내용은 supervisor가 읽는다. |

`status.json`은 ModelConfig와 관련된 per-anima 값의 기준 데이터다. 대응 키와 기본값 목록은 [설정 참조](../reference/config.md)를 참조한다. `permissions.json`과 전체 `permissions.global.json`를 권한 판단에 사용한다. 권한 경계와 평가 방법은 [보안](../security.md)을 참조한다.

## 작업 공간 파일

Anima의 인격과 작업 지침은 OpenClaw / Hermes와 동일한 표준 파일명으로 관리한다. 이전 이름(`identity.md`, `injection.md`, `heartbeat.md`, `bootstrap.md`)은 서버 시작 시 migration을 통해 자동으로 새 이름으로 전환되며, 이후에는 새 이름만 읽고 쓴다.

| 용도 | AnimaWorks | OpenClaw | Hermes |
|---|---|---|---|
| 공개 정체성 | `IDENTITY.md` | `IDENTITY.md` | `IDENTITY.md`(가져오기) |
| 내면의 가치관·원칙 | `SOUL.md` | `SOUL.md` | `SOUL.md`(내보낼 때 IDENTITY → SOUL 순으로 연결) |
| 작업 디렉터리 지침 | `AGENTS.md` | `AGENTS.md` | `AGENTS.md` |
| 정기 순회 지침 | `HEARTBEAT.md` | `HEARTBEAT.md` | `HEARTBEAT.md` |
| 최초 시작 지침 | `BOOTSTRAP.md` | — | — |

이전 `identity.md`은 migration 시 `split_legacy_identity()`로 분할된다. 이름·프로필·말투 등의 제목은 `IDENTITY.md`에, 가치관·성격·어조 등의 제목은 `SOUL.md`에 들어간다. 이전 `injection.md`은 `AGENTS.md`, 이전 `heartbeat.md`은 `HEARTBEAT.md`, 이전 `bootstrap.md`은 `BOOTSTRAP.md`에 해당한다.

### 마이그레이션과 롤백

서버 시작 시 `core/infra/runtime_init.py`의 자동 migration이 anima 감지보다 먼저 모든 anima를 마이그레이션한다. 변경 전의 이전 파일과 수정 전의 참조 파일은 anima별 `archive/standard_files_migration/<UTC時刻>/`에 저장되며, `manifest.json`에 이전 이름→새 이름 대응, sha256, 새 파일 목록, 충돌, 실행 시각과 버전이 기록된다.

```bash
# 戻す内容を確認する（何も変更しない）
animaworks migrate --rollback-standard-files --dry-run

# 全 anima を戻す
animaworks migrate --rollback-standard-files

# 特定の anima だけ戻す
animaworks migrate --rollback-standard-files --anima alice
```

롤백은 새 표준 파일을 `archive/standard_files_migration/<UTC時刻>/rolled_back_new/`에 옮겨 보관한 뒤 이전 파일을 복원하고, `migration_state.json`에서 이 단계의 적용 기록을 제거한다. 마이그레이션 후 새 파일을 편집했다면 경고가 표시되지만, 편집 내용은 `rolled_back_new/`에 남는다. **이전 버전의 AnimaWorks를 시작하기 전에** 이 명령을 실행할 것.

### 내보내기와 가져오기

```bash
# OpenClaw / Hermes 形式へ書き出す
animaworks anima export alice --format openclaw --out ./alice-openclaw
animaworks anima export alice --format hermes --out ./alice-hermes

# OpenClaw / Hermes のデータを取り込む
animaworks import openclaw --path ~/.openclaw --target-anima alice --apply
animaworks import hermes --path ~/.hermes --target-anima alice --apply
```

내보내기에는 표준 형식에 대응 항목이 없는 `permissions.json`, `status.json`, `cron.md`을 포함하지 않으며, `ANIMAWORKS_EXPORT.md`에 내보낸 항목과 내보내지 않은 항목을 기록한다. 비밀 정보는 `redact_credentials()`에서 마스킹한다. 가져오기 원본에 `SOUL.md`, `IDENTITY.md`, `AGENTS.md`, `HEARTBEAT.md`가 있는 경우, 같은 이름의 새 파일이 없으면 바로 반영한다. 기존 파일이 있으면 `--replace`을 지정하지 않는 한 덮어쓰지 않고, `state/migrations/drafts/`에 초안을 남긴다.

## 실행 상태

| 경로 | 역할 |
|---|---|
| `state/current_state.md` | 진행 중인 작업, 판단, 다음 단계 등을 보유하는 작업 상태다. prompt에 필요한 범위만 포함하며, 업데이트는 상태 파일용 lock을 통해 보호된다. |
| `state/task_queue.jsonl` | 이전 형식의 작업 데이터가 남아 있는 경우 이를 마이그레이션하기 위한 입력이다. 현재 작업의 기준 데이터는 공유 SQLite TaskStore이며, 이 파일을 런타임의 별도 큐로 취급하지 않는다. |
| `state/background_tasks/` | `animaworks-tool submit`에서 시작한 백그라운드 처리의 상태와 결과를 저장한다. 완료 결과는 대화나 후속 백그라운드 실행에서 알림을 통해 전달되거나 참조된다. |
| `scratch/` | 일회용 작업 공간이다. 자식 프로세스에는 `ANIMAWORKS_SCRATCH_DIR`로 작업용 `scratch/task-<id>/` 또는 세션 종류별 `scratch/<種類>/`가 전달된다. 작업용 공간은 작업 완료·취소 시 삭제되고, 그 외의 공간은 24시간 동안 방치되면 housekeeping이 삭제한다. |

Anima의 장기 기억, 대화 기록, 스킬 등은 용도별 하위 디렉터리로 나뉜다. 기억에 관한 자세한 내용은 `docs/ja/memory/`의 각 장을 참조한다.

## 설계 판단

- **기억은 Markdown 파일로 보관한다.** AI가 자연스럽게 읽고 쓸 수 있으며 grep과도 잘 맞는다. JSON은 설정과 상태에만 사용한다.
- **아카이브형 기억을 채택한다.** 최근 N건을 prompt에 넣는 절삭형 방식은 기억량에 한계가 생긴다. 필요한 내용을 검색해 떠올리는 아카이브형이라면 기억이 계속 늘어나도 괜찮다.
- **권한은 시야를 제한하는 것이다.** 모르는 것이 있기에 다른 사람에게 묻는다. 모두가 모든 것을 볼 수 있다면 조직은 의미를 잃는다.
