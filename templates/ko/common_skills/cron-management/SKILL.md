---
name: cron-management
description: >-
  cron.md를 올바른 형식으로 읽고 쓰는 스킬. 정기 작업의 추가·업데이트·삭제 절차를 제공한다.
  Use when: cron.md의 편집, cron식 추가, LLM형·커맨드형 작업의 추가·삭제, 정기 작업의 유지보수가 필요할 때.
---


## 프레임워크 측 구현 (참조용)

### cron（정기 작업）

cron의 **파싱**은 `core/runtime/schedule_parser.py`（`parse_cron_md` / `parse_schedule`), **등록·실행·리로드**는 `core/runtime/scheduler_manager.py`（APScheduler, `AsyncIOScheduler(timezone=get_app_timezone())`)이 담당한다.

### `core/tasks/background.py` (cron과는 다른 계열)

이 모듈은 **cron 스케줄링을 하지 않는다**. 장시간 도구 호출의 백그라운드 실행과 DM 로그의 로테이션을 담당한다. cron과 혼동하지 말 것.

**`BackgroundTaskManager`**

- **역할**: 대상 도구를 `asyncio` 상에서 비동기 실행하고, 상태를 디스크에 저장한다.
- **영속화 위치**: `state/background_tasks/{task_id}.json`(`pending` / `running` / `completed` / `failed`, 결과 문자열·오류·타임스탬프).
- **공개 API (개요)**:
  - `submit(tool_name, tool_args, execute_fn)` — 동기 실행 함수를 스레드 풀에서 실행한다.
  - `submit_async(...)` — 비동기 `execute_fn` 버전.
  - `get_task` / `list_tasks` / `active_count` — 조회.
  - `cleanup_old_tasks(max_age_hours=24)` — 완료·실패 작업을 **24시간** 초과 시 삭제. 추가로 `running` 상태로 **48시간** 초과 경과한 JSON(크래시 고아)도 삭제.
- **`from_profiles(anima_dir, ..., profiles, config_eligible)`**: 백그라운드 대상 도구 집합은 다음 **3계층 병합**(나중 것이 우선).
  1. 코드 내 기본값 `_DEFAULT_ELIGIBLE_TOOLS`(Mode A 호환의 고정 목록)
  2. 각 도구 모듈의 `EXECUTION_PROFILE`에서 `background_eligible: true`의 엔트리(`core.integrations._base.get_eligible_tools_from_profiles`). 키는 `"{tool}:{subcommand}"` 형식(Mode S의 `submit`와 정합)
  3. `config.json` 등에서 전달되는 `config_eligible`(명시적 덮어쓰기)
- **`is_eligible(tool_name)`**: 다음 두 이름 중 어느 것으로도 매칭 가능 — 스키마 이름(예: `generate_3d_model`), 프로필 키(예: `image_gen:3d`).
- **기본적으로 `_DEFAULT_ELIGIBLE_TOOLS`에 포함되는 예**(값은 대략적인 초): `generate_character_assets` / `generate_fullbody` / `generate_bustup` / `generate_icon` / `generate_chibi` / `generate_3d_model` / `generate_rigged_model` / `generate_animations`(각 30), `local_llm` / `run_command`(각 60).
- **완료 훅**: `on_complete`에 `Callable[[BackgroundTask], Awaitable[None]]`을 설정 가능. 앱 측에서 여기서 `state/background_notifications/`로의 Markdown 알림 등을 작성한다(`background.py` 자체는 알림 디렉터리를 건드리지 않는다).

**`rotate_dm_logs`**

- `shared/dm_logs/*.jsonl`의 엔트리를 `max_age_days`(기본 **7일**)로 잘라, 오래된 줄을 `{stem}.{YYYYMMDD}.archive.jsonl`에 추가 아카이브한다. cron 작업은 아니지만, 백그라운드 계열 유지보수와 같은 파일에 정의되어 있어 참조용으로 기재.

---

## cron.md의 구조

### 전체 구성

```markdown
# Cron: {自分の名前}

## タスク名1
schedule: 0 9 * * *
type: llm
タスクの説明文...

## タスク名2
schedule: */5 * * * *
type: command
command: /path/to/script.sh
```

### 반드시 지켜야 할 규칙

1. **각 작업은 `## タスク名`로 시작한다**(H2 제목. H3나 H1은 사용하지 않음)
2. **`schedule:` 줄은 필수**(`schedule:`라는 키워드로 시작. `###` 제목으로 하지 않음)
3. **스케줄은 표준 5필드 cron식만**(`09:00`나 `毎週金曜 17:00`는 불가)
4. **`type:` 줄은 필수**(`llm` 또는 `command`)
5. **작업 사이에 빈 줄을 넣는다**(가독성을 위해)

### 하면 안 되는 작성법

```markdown
❌ ### */5 * * * *           ← H3見出しにcron式を書いてはいけない
❌ ### 09:00                 ← 自然言語の時刻表記は不可
❌ ### 毎週金曜 17:00         ← 日本語のスケジュール表記は不可
❌ cron: 0 9 * * *           ← キー名は "schedule:" であること（"cron:" ではない）
❌ interval: 5m              ← interval形式は不可
❌ schedule: 0 9 * * * *     ← 6フィールドは不可（5フィールドのみ）
```

### 올바른 작성법

```markdown
✅ schedule: 0 9 * * *       ← "schedule:" + 半角スペース + 5フィールドcron式
✅ schedule: */5 * * * *
✅ schedule: 30 21 * * *
✅ schedule: 0 17 * * 5
```

---

## 5필드 cron식 참조

### 필드 구성

```
schedule: 分 時 日 月 曜日
```

| 필드 | 위치 | 범위 | 설명 |
|-----------|------|------|------|
| 분 | 1 | 0-59 | 몇 분에 실행할지 |
| 시 | 2 | 0-23 | 몇 시에 실행할지 (24시간제) |
| 일 | 3 | 1-31 | 며칠에 실행할지 |
| 월 | 4 | 1-12 | 몇 월에 실행할지 |
| 요일 | 5 | 0-6 | 무슨 요일에 실행할지 (0=월, 6=일) |

**주의**: 요일은 **0=월요일, 6=일요일**(APScheduler의 사양. 일반적인 cron의 0=일요일과 다름)

### 특수 문자

| 문자 | 의미 | 예 |
|------|------|-----|
| `*` | 모든 값 | `* * * * *` = 매분 |
| `*/n` | n 간격 | `*/5 * * * *` = 5분마다 |
| `n-m` | 범위 | `0 9-17 * * *` = 9시~17시의 매 정시 |
| `n,m` | 리스트 | `0 9,12,18 * * *` = 9시·12시·18시 |
| `n-m/s` | 범위+간격 | `0 9-17/2 * * *` = 9시~17시의 2시간마다 |

### 자주 쓰는 스케줄 예

#### 매일 계열

| 하고 싶은 일 | cron식 | 해설 |
|-------------|--------|------|
| 매일 아침 9:00 | `0 9 * * *` | 분=0, 시=9 |
| 매일 아침 9:30 | `30 9 * * *` | 분=30, 시=9 |
| 매일 12:00 (정오) | `0 12 * * *` | 분=0, 시=12 |
| 매일 18:00 | `0 18 * * *` | 분=0, 시=18 |
| 매일 밤 21:30 | `30 21 * * *` | 분=30, 시=21 |
| 매일 새벽 2:00 | `0 2 * * *` | 분=0, 시=2 |

#### 간격 계열

| 하고 싶은 일 | cron식 | 해설 |
|-------------|--------|------|
| 5분마다 | `*/5 * * * *` | 분=*/5（0,5,10,...,55） |
| 10분마다 | `*/10 * * * *` | 분=*/10（0,10,20,...,50） |
| 15분마다 | `*/15 * * * *` | 분=*/15（0,15,30,45） |
| 30분마다 | `*/30 * * * *` | 분=*/30（0,30） |
| 1시간마다 | `0 * * * *` | 매시 0분 |
| 2시간마다 | `0 */2 * * *` | 0시,2시,4시,...,22시의 0분 |
| 업무 시간 중에만 5분마다 | `*/5 9-17 * * *` | 9:00~17:55의 5분 간격 |
| 업무 시간 중에만 1시간마다 | `0 9-17 * * *` | 9:00~17:00의 매 정시 |

#### 요일 계열

| 하고 싶은 일 | cron식 | 해설 |
|-------------|--------|------|
| 평일 매일 아침 9:00 | `0 9 * * 0-4` | 요일=0-4 (월~금) |
| 매주 월요일 9:00 | `0 9 * * 0` | 요일=0 (월) |
| 매주 금요일 17:00 | `0 17 * * 4` | 요일=4 (금) |
| 매주 금요일 18:00 | `0 18 * * 4` | 요일=4 (금) |
| 평일 업무 시간 중 30분마다 | `*/30 9-17 * * 0-4` | 평일 9:00~17:30 |
| 주말 매일 아침 10:00 | `0 10 * * 5,6` | 요일=5,6 (토일) |

#### 월간 계열

| 하고 싶은 일 | cron식 | 해설 |
|-------------|--------|------|
| 매월 1일 9:00 | `0 9 1 * *` | 일=1 |
| 매월 15일 12:00 | `0 12 15 * *` | 일=15 |
| 매월 마지막 영업일 근처 (28일) | `0 17 28 * *` | 일=28 (근사) |
| 분기 첫날 9:00 (1,4,7,10월) | `0 9 1 1,4,7,10 *` | 월=1,4,7,10 |

---

## 작업 유형 상세

### type: llm — LLM 판단 작업

사고·판단이 필요한 작업. `schedule:`와 `type: llm` 뒤에 자유 기술로 작업 내용을 쓴다.

```markdown
## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。
```

- 설명문은 그대로 LLM에 대한 프롬프트로 전달된다
- 구체적인 아웃풋(무엇을 작성할지)을 명시하면 효과적
- 여러 줄 가능
- 본문에 펜스가 있는 코드 블록(\`\`\`）が含まれると、パーサーが警告ログを出す（確定コマンドなら `type: command`를 검토)

### type: command — 커맨드 실행 작업

결정적으로 실행하는 bash 커맨드나 도구 호출.

#### 패턴 A: bash 커맨드

```markdown
## バックアップ実行
schedule: 0 2 * * *
type: command
command: /usr/local/bin/backup.sh
```

- `command:`에 실행할 커맨드를 한 줄로 쓴다
- 셸 리다이렉트(`>`, `>>`, `|`)는 사용 가능
- 여러 줄 커맨드는 비권장 (한 줄로 정리하거나 스크립트 파일로)

#### 패턴 B: 도구 호출

```markdown
## Slack朝の通知
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！"
```

- `tool:`에 도구 이름(`permissions.json`의 허용 설정에 포함되는 스키마 이름. 예: Slack 게시는 `slack_channel_post` 등)
- `args:` 이후는 YAML 블록 형식으로 들여쓰기 2칸
- `ToolHandler.handle(tool, args)`로 실행되고, 결과 문자열이 stdout 상당으로 취급된다

### 옵션: skip_pattern

`type: command`로 커맨드가 **성공**(`exit_code == 0`)하고, 표준 출력이 비어 있지 않을 때만 실행되는 **후속 cron LLM**(heartbeat 동등 컨텍스트의 분석 세션)을, stdout이 이 정규식에 매치할 때 **억제**한다.

```markdown
## Chatwork未返信チェック
schedule: */5 * * * *
type: command
command: chatwork_cli.py unreplied --json
skip_pattern: "^\[\]$"
```

- `skip_pattern:`에는 정규식을 쓴다 (실행 시에는 `re.search(skip_pattern, stdout)`)
- 정규식에 `[]` 등의 YAML 특수 문자가 포함되면 따옴표(`"..."` 또는 `'...'`)로 감싼다. 파서는 바깥쪽 따옴표를 자동 제거한다
- **파싱 시**에 유효하지 않은 정규식 → 경고 로그 후 `skip_pattern`는 미설정 취급 (후속 억제 없음)
- **실행 시**에 `re.search`가 예외(유효하지 않은 패턴의 잔존 등) → 경고 로그 후 **억제하지 않고** 후속을 실행
- 위 예에서는 미회신이 0건(`[]`)인 경우에 후속을 스킵한다

### 옵션: trigger_heartbeat

커맨드 성공 시 후속 cron LLM을 실행할지 작업 단위로 제어한다(`SchedulerManager._run_cron_task`의 평가 순서를 따름).

```markdown
## Chatwork未返信チェック
schedule: */15 * * * *
type: command
command: animaworks-tool chatwork unreplied
skip_pattern: "^\[\]$"
trigger_heartbeat: false
```

- **후속이 검토되는 조건**(모두 충족할 때): `exit_code == 0` 그리고 stdout(`.strip()` 후)이 비어 있지 않음
- **stdout의 길이**: `run_cron_command`가 스케줄러에 반환하는 `stdout`는 앞부분 **1000자**로 잘린다. 후속 LLM에 전달되는 것도 이 프리뷰. 전체 로그는 `state/cron_logs/`(JSONL) 쪽을 참조
- `trigger_heartbeat: false` — 위 조건을 충족해도 후속 cron LLM을 **실행하지 않음**(`skip_pattern`보다 먼저 평가)
- `trigger_heartbeat: true`(기본값) — 조건을 충족하면 후속을 실행(다음에 `skip_pattern`을 평가)
- `false`, `no`, `0`을 지정하면 억제. 그 외에는 true 취급
- 후속 cron LLM은 heartbeat 동등의 프롬프트 필터(백그라운드용 컨텍스트)로 동작
- `exit_code != 0`나 stdout이 비어 있을 때는 후속이 **애초에 시작되지 않음**(`skip_pattern` / `trigger_heartbeat`는 무관)

---

## 유형 구분 판단 기준

### type: command를 사용해야 하는 경우
- 실행할 커맨드가 완전히 확정되어 있음
- 파라미터가 고정 (리전, 클러스터 이름, 프로필 등)
- 결과 판단은 cron LLM 세션에 맡김

### type: llm을 사용해야 하는 경우
- 상황에 따라 실행 내용을 바꿔야 함
- 여러 도구를 조합한 조사가 필요
- 인간적인 판단·분석이 실행 단계에서 필요

### 금지 패턴
- type: llm에 코드 블록(확정 커맨드)을 포함
  → 그 커맨드는 type: command로 해야 함
- 「이대로 실행할 것」이라고 쓰면서 type: llm
  → LLM은 커맨드를 정확히 재현할 수 없다. type: command를 사용하라

### Anima는 「커맨드를 암기하는 사람」이 아니다
type: command는 인간이 스크립트를 저장하는 것과 같다.
Anima의 가치는 결과를 보고 판단하는 힘에 있다.
결정적 실행은 프레임워크에 맡기고,
Anima는 판단·분석·보고에 집중하게 할 것.

---

## cron.md 분석 알림 (자동)

cron.md 최초 등록 또는 `reload_schedule` 시점에 분석·등록 오류를 감지하면, 스케줄러는 `state/background_notifications/cron_health_{タイムスタンプ}.md`을 생성한다. 알림은 다음 heartbeat 또는 cron 실행 컨텍스트에 포함된다. 등록된 cron의 실행 상태를 주기적으로 모니터링하는 기능은 없다.

**감지되는 문제**:

- 작업은 정의되어 있지만 **유효한 스케줄이 1건도 등록되지 않음** (모든 표현식이 유효하지 않은 경우 등)
- 주석이나 코드 펜스 외부에, 줄 시작 부분의 공백을 포함한 `schedule:` 줄이 있음 (`schedule:`은 줄 시작 부분에 작성)
- 유효한 설정 텍스트에 `schedule:`가 있는데, 파서가 **작업을 1건도 반환하지 않음**

---

## cron.md 조작 절차

### 대상 경로와 하위 편집

- 자신의 `cron.md` 는 `read_memory_file(path="cron.md")` 로 읽고, `write_memory_file(path="cron.md", ...)` 로 업데이트한다.
- 상급자는 부하의 `cron.md` / `HEARTBEAT.md` 를 write memory 도구로 `../{anima_name}/cron.md` 처럼 지정하여 편집할 수 있다 (자식·손자 이하의 모든 부하가 대상).
- `status.json` / `IDENTITY.md` / `SOUL.md` / `permissions.json` 는 root 소유이며, 기억·파일 도구에서 쓸 수 없다. 허가된 상급자의 `AGENTS.md` 변경과 bootstrap 중의 IDENTITY.md / SOUL.md 생성 요청은 root로 전송·인가된다. 직접 파일 조작은 하지 않는다.

### 새 작업 추가

1. 자신의 `cron.md`을 읽는다
2. 파일 끝에 새 섹션을 추가한다
3. **쓰기 전에 형식을 확인한다**(아래 체크리스트 참조)
4. 파일을 쓴다

```markdown
## 新しいタスク名
schedule: <5フィールドcron式>
type: llm|command
<説明またはcommand/tool行>
```

### 기존 작업 변경

1. `cron.md`을 읽는다
2. 해당 섹션(`## タスク名`부터 다음 `##` 직전까지)을 식별
3. 변경할 줄(`schedule:`, `type:`, 설명문 등)을 편집
4. 파일을 쓴다

### 작업 삭제

1. `cron.md`을 읽는다
2. 해당 섹션 전체(`## タスク名`부터 다음 `##` 직전까지)를 삭제
3. 파일을 쓴다

### 작업의 임시 비활성화

HTML 주석으로 감싸면 파서가 건너뛴다:

```markdown
<!--
## 一時停止中のタスク
schedule: 0 9 * * *
type: llm
このタスクは一時的に停止中。
-->
```

---

## 쓰기 전 체크리스트

cron.md을 업데이트하기 전에 다음을 **반드시** 확인할 것:

- [ ] 각 작업이 `## タスク名`으로 시작하는가(`###`이나 `#`가 아님)
- [ ] `schedule:` 줄이 있는가(`###` 제목이나 자연어가 아님)
- [ ] 스케줄이 5필드 cron식인가(`分 時 日 月 曜日`)
- [ ] 각 필드의 값이 유효 범위 내인가(분: 0-59, 시: 0-23, 일: 1-31, 월: 1-12, 요일: 0-6)
- [ ] `type:` 줄이 있는가(`llm` 또는 `command`)
- [ ] command형인 경우, `command:` 또는 `tool:`이 있는가
- [ ] tool형인 경우, `args:`의 들여쓰기가 올바른가(2칸)
- [ ] 작업 사이에 빈 줄이 있는가
- [ ] `schedule:`를 코드 블록 안에 쓰지 않았는가(상태 경고의 원인이 될 수 있음)

### 검증 방법

작성 후, 아래 명령어로 올바르게 파싱되는지 확인할 수 있다:

```bash
# プロジェクトルートで実行。ANIMAWORKS_ANIMA_DIR 未設定時は ~/.animaworks/animas/default を使用
python -c "
from core.runtime.schedule_parser import parse_cron_md, parse_schedule
import os
from pathlib import Path

cron_path = Path(os.environ.get('ANIMAWORKS_ANIMA_DIR', '~/.animaworks/animas/default')) / 'cron.md'
content = cron_path.expanduser().read_text()
tasks = parse_cron_md(content)
for t in tasks:
    trigger = parse_schedule(t.schedule)
    status = '✅' if trigger else '❌ パース失敗'
    print(f'{status} {t.name}: schedule=\"{t.schedule}\" type={t.type}')
"
```

모든 작업에 ✅가 표시되면 정상이다. ❌가 나온 경우에는 스케줄 식을 수정할 것.

---

## 완전한 기술 예

```markdown
# Cron: example_anima

## 毎朝の業務計画
schedule: 0 9 * * *
type: llm
長期記憶から昨日の進捗を確認し、今日のタスクを計画する。
理念と目標に照らして優先順位を判断する。
結果は state/current_state.md に書き出す。

## Chatwork未返信チェック
schedule: */5 9-18 * * 0-4
type: command
command: chatwork-cli unreplied --json > $ANIMAWORKS_ANIMA_DIR/state/chatwork_unreplied.json
skip_pattern: "^\[\]$"
trigger_heartbeat: false

## Slack朝の挨拶
schedule: 0 9 * * 0-4
type: command
tool: slack_channel_post
args:
  channel_id: "C0123456789"
  text: "おはようございます！今日もよろしくお願いします。"

## 週次振り返り
schedule: 0 17 * * 4
type: llm
今週のepisodes/を読み返し、パターンを抽出してknowledge/に統合する。
改善点があれば procedures/ に手順を追記する。

## 月次レポート
schedule: 0 10 1 * *
type: llm
先月のepisodes/とknowledge/を分析し、月次サマリーレポートを作成する。
レポートは knowledge/monthly_report_YYYY-MM.md として保存する。
```

---

## 흔한 실수와 수정

| 실수 | 올바른 작성법 | 원인 |
|--------|------------|------|
| `### */5 * * * *` | `schedule: */5 * * * *` | H3 제목은 작업 구분에 사용할 수 없음 |
| `### 09:00` | `schedule: 0 9 * * *` | 자연어 시간은 파싱 불가 |
| `### 毎週金曜 17:00` | `schedule: 0 17 * * 4` | 일본어 표기는 파싱 불가 |
| `schedule: 9:00` | `schedule: 0 9 * * *` | HH:MM 형식은 5필드 cron식이 아님 |
| `schedule: every 5 minutes` | `schedule: */5 * * * *` | 영어 표기는 파싱 불가 |
| `schedule: 0 9 * * 7` | `schedule: 0 9 * * 6` | 요일 7은 범위 밖(0-6) |
| `schedule: 0 9 * * SUN` | `schedule: 0 9 * * 6` | 요일 이름은 사용할 수 없음(숫자만) |
| `schedule: 0 25 * * *` | `schedule: 0 23 * * *` | 시는 0-23의 범위 |
| `schedule: 60 * * * *` | `schedule: 0 * * * *` | 분은 0-59의 범위 |
| `skip_pattern: ^\[\]$`(따옴표 없음) | `skip_pattern: "^\[\]$"` | `[]`은 YAML에서 빈 목록으로 해석됨. 따옴표로 감쌀 것 |

---

## 주의사항

- **핫 리로드**: `cron.md` / `HEARTBEAT.md`을 Anima의 `write_memory_file` 등으로 업데이트하면, 스케줄 변경 콜백으로 `reload_schedule`이 실행되어 즉시 재등록된다. 하위 관리 파일의 편집도 `../{anima_name}/cron.md`처럼 write memory 도구로 수행한다. 외부의 직접 편집은 다음 **heartbeat 또는 임의의 cron이 발화하기 직전**의 mtime 체크(`_check_schedule_freshness`)로도 검출되지만, 상급자의 하위 편집 표준 절차로는 하지 않는다
- **리로드 직후의 오래된 작업**: mtime 변화를 감지한 시점에 스케줄러가 재구축되면, **직전 정의에 연결된 cron 발화는 「stale」으로서 스킵**될 수 있다(의도적인 이중 실행 방지)
- **타임존**: APScheduler는 `get_app_timezone()`을 사용. `config.json`의 `system.timezone`에 IANA 이름(예: `Asia/Tokyo`)을 설정 가능. **빈 문자열**일 때는 OS 타임존을 자동 검출하고, 실패 시 `Asia/Tokyo`으로 폴백
- **동시 실행**: **작업 이름이 다르면**, 같은 분에 여러 작업이 겹쳐도 `asyncio.create_task`에 의해 병렬 실행될 수 있다. 같은 작업 이름은 실행 중에 재진입하지 않는다(`_cron_running`로 스킵). 각 작업은 `max_instances=1`
- command형은 실패해도 프로세스를 멈추지 않는다(`cron_executed`가 기록되고, `stderr` / exit_code가 로그에 남는다). **후속 cron LLM은 성공かつ stdout이 있을 때만**(앞서 설명)
- `type: llm`의 정기 실행에는 **백그라운드용 모델**(`status.json`의 `background_model` 등, 미설정 시 메인 모델)이 사용된다
- command의 상세 출력은 `state/cron_logs/`(일일 JSONL)에 축적되고, 최대 14일간 하우스키핑된다
- **다른 Anima의 tools/ 디렉터리에 접근하는 경우, 권한을 사전에 확인할 것**
