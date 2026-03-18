# 프로젝트 설정 방법

AnimaWorks의 설정 구조와 Anima 추가 절차에 대한 레퍼런스입니다.
설정 변경이 필요한 상황에서 검색하여 참조하세요.

## 런타임 디렉토리 초기화

런타임 데이터는 `~/.animaworks/` (또는 `ANIMAWORKS_DATA_DIR`)에 배치됩니다.
최초 설정 시 `animaworks init`으로 템플릿에서 초기화합니다.

### init 명령어

| 명령어 | 설명 |
|--------|------|
| `animaworks init` | 런타임 디렉토리를 초기화 (비대화형). 이미 존재하면 아무것도 하지 않음 |
| `animaworks init --force` | 기존 데이터에 템플릿 차이를 머지. 새 파일만 추가하고 `prompts/`는 덮어씀 |
| `animaworks init --skip-anima` | 인프라만 초기화. Anima 생성을 건너뜀 |
| `animaworks init --template NAME` | 템플릿에서 Anima를 비대화형으로 생성 |
| `animaworks init --from-md PATH` | MD 파일에서 Anima를 비대화형으로 생성 |
| `animaworks init --blank NAME` | 빈 Anima를 비대화형으로 생성 |

**권장 플로우**: `animaworks init` 실행 후, `animaworks start`로 서버를 기동하고, 웹 UI의 설정 마법사로 Anima를 추가합니다.

### 초기화 시 생성되는 디렉토리

`ensure_runtime_dir` (`core/init.py`)에 의해 다음이 생성됩니다:

- `animas/` — Anima 디렉토리
- `shared/inbox/` — 수신 메시지 큐
- `shared/users/` — 사용자 프로필
- `shared/channels/` — 공유 채널 (general, ops 초기 파일)
- `shared/dm_logs/` — DM 이력 (폴백용)
- `tmp/attachments/` — 첨부 파일 임시 저장
- `common_skills/` / `common_knowledge/` — 공통 스킬 및 지식
- `prompts/` / `company/` — 프롬프트 및 조직 템플릿
- `tool_prompts.sqlite3` — 도구 프롬프트 DB
- `models.json` — 모델 이름 → 실행 모드 매핑 (`config_defaults/`에서 복사)

기동할 때마다 `common_skills`와 `common_knowledge`는 템플릿에서 증분 동기화되며, 새 항목만 추가됩니다 (기존 파일은 유지).

## config.json 전체 구조

AnimaWorks의 통합 설정 파일은 `~/.animaworks/config.json`에 위치합니다.
모든 설정은 `AnimaWorksConfig` 모델로 정의되며, 다음의 최상위 필드를 갖습니다.

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server", "log_level": "INFO" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-4-6", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "dynamic_budget": true },
  "image_gen": {}
}
```

**참고**: `animas` 섹션은 조직 레이아웃(`supervisor`, `speciality`)만 보유합니다. 모델 이름, credential, max_turns 등의 모델 설정은 각 Anima의 `status.json`에 기록됩니다 (아래 "Anima 설정 해석" 참조).

각 섹션의 역할:

| 섹션 | 설명 |
|------|------|
| `version` | 설정 스키마 버전 (현재 `1`) |
| `setup_complete` | 최초 설정 완료 플래그 |
| `locale` | UI 언어 (`"ja"` / `"en"`) |
| `system` | 서버 모드, 로그 레벨 |
| `credentials` | API 키 및 엔드포인트 (이름 기반) |
| `model_modes` | 모델 이름 → 실행 모드 오버라이드 맵 |
| `anima_defaults` | 전체 Anima 공통 기본 설정 |
| `animas` | Anima의 조직 레이아웃 (supervisor, speciality). 모델 설정은 status.json |
| `consolidation` | 기억 통합 (일별/주별) 설정 |
| `rag` | RAG (임베딩 벡터 검색) 설정 |
| `priming` | 프라이밍 (자동 기억 호출) 토큰 예산 |
| `image_gen` | 이미지 생성 스타일 설정 |

<!-- AUTO-GENERATED:START config_fields -->
### 설정항목 레퍼런스 (자동 생성)

#### Anima설정 (per-anima overrides)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |

#### 기본값 (anima_defaults)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model` | `str` | `"claude-sonnet-4-6"` |  |
| `fallback_model` | `str | None` | None |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `max_turns` | `int` | `20` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `max_chains` | `int` | `2` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `llm_timeout` | `int` | `600` |  |
| `mode_s_auth` | `str | None` | None |  |
| `max_outbound_per_hour` | `int | None` | None |  |
| `max_outbound_per_day` | `int | None` | None |  |
| `max_recipients_per_run` | `int | None` | None |  |

#### AnimaWorksConfig 최상위

| 섹션 | 설명 |
|------|------|
| `version` | 설정 파일 버전 |
| `setup_complete` | 설정 완료 플래그 |
| `locale` | 로케일 설정 |
| `system` | 시스템 설정 (모드, 로그 레벨) |
| `credentials` | API 인증 정보 |
| `model_modes` | 모델 이름 → 실행 모드 매핑 |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima 설정 기본값 |
| `animas` | Anima별 설정 오버라이드 |
| `consolidation` | 기억 통합 설정 |
| `rag` | RAG (검색 증강 생성) 설정 |
| `priming` | 프라이밍 (자동 기억 호출) 설정 |
| `image_gen` | 이미지 생성 설정 |
| `human_notification` |  |
| `server` |  |
| `external_messaging` |  |
| `background_task` |  |
| `activity_log` |  |
| `heartbeat` |  |
| `voice` |  |
| `housekeeping` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `ui` |  |

<!-- AUTO-GENERATED:END -->

## 새로운 Anima 추가 방법

Anima 추가 방법은 3가지입니다. 모두 `animaworks anima create` 또는 `animaworks init`으로 실행합니다.
`animaworks anima create`는 `--role`과 `--supervisor` 옵션을 지원하며 권장됩니다.

### 방법 1: 템플릿에서 생성

사전 정의된 템플릿(`templates/ja/anima_templates/` 또는 `templates/en/anima_templates/` 하위)을 사용하는 방법입니다.
템플릿에는 identity.md, injection.md, permissions.json, 스킬 등이 포함되어 있습니다.

```bash
# 템플릿에서 생성 (템플릿 이름은 디렉토리 이름)
animaworks anima create --template <template_name>

# 이름을 변경하여 생성
animaworks anima create --template <template_name> --name <anima_name>
```

템플릿이 가장 권장되는 방법입니다. 이미 캐릭터 설정이 갖추어져 있어 bootstrap(최초 기동 자기 정의)을 건너뛸 수 있습니다.

### 방법 2: Markdown 파일에서 생성

캐릭터 시트(Markdown)를 준비하여 이를 기반으로 Anima를 생성합니다.

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima 이름 (생략 시 시트에서 추출)
- `--role`: 역할 템플릿 (engineer, researcher, manager, writer, ops, general). 기본값: general
- `--supervisor`: supervisor Anima 이름 (캐릭터 시트의 "상사"를 오버라이드)

Markdown 파일은 `character_sheet.md`로서 Anima 디렉토리에 복사됩니다.
시트의 "인격"과 "역할 및 행동 방침" 섹션이 identity.md와 injection.md에 반영됩니다.
역할 템플릿에서 permissions.json와 specialty_prompt.md가 적용됩니다.

Markdown 파일에는 다음을 포함하는 것이 좋습니다(SHOULD):
- `# Character: 이름` 형식의 제목, 또는 기본 정보 테이블의 "영문 이름" 행 (이름 자동 추출에 사용)
- `## 기본 정보` — 영문 이름, 상사, 모델 등의 테이블
- `## 인격` — identity.md에 반영
- `## 역할 및 행동 방침` — injection.md에 반영

### 방법 3: 빈 Anima 생성

최소한의 스켈레톤 파일로 Anima를 생성합니다.

```bash
animaworks anima create --name aoi
```

`--name`은 필수입니다. 빈 Anima 생성에서는 `{name}` 플레이스홀더가 실제 이름으로 치환된 스켈레톤 파일이 생성됩니다.
최초 기동의 bootstrap에서 에이전트가 사용자와의 대화를 통해 캐릭터를 스스로 정의합니다.

### 생성 후 디렉토리 구성

어떤 방법이든 다음의 디렉토리와 파일이 생성됩니다:

```
~/.animaworks/animas/{name}/
├── identity.md          # 인격 정의 (불변 베이스라인)
├── injection.md         # 역할 및 행동 지침 (가변)
├── bootstrap.md         # 최초 기동 지시 (완료 후 삭제)
├── permissions.json       # 도구 및 명령어 권한
├── heartbeat.md         # heartbeat 설정
├── cron.md              # 정기 작업 설정
├── episodes/            # 에피소드 기억 (일별 로그)
├── knowledge/            # 의미 기억 (배운 지식)
├── procedures/          # 절차 기억 (절차서)
├── skills/              # 개인 스킬
├── state/               # 워킹 메모리
│   ├── current_state.md  # 현재 작업
│   └── task_queue.jsonl # 영구 태스크 큐 (미착수 태스크 등)
└── shortterm/           # 단기 기억 (세션 연속용)
    └── archive/
```

### Anima 이름 규칙

Anima 이름은 다음 규칙을 따라야 합니다 (MUST):
- 소문자 영숫자, 하이픈(`-`), 언더스코어(`_`)만 사용 가능
- 첫 문자는 영문자(`a-z`)여야 함
- 언더스코어로 시작 불가 (템플릿 예약)
- 예시: `aoi`, `taro-dev`, `worker01`

## 실행 모드 (S / A / B)

AnimaWorks는 3가지 실행 모드를 가집니다. 모델 이름에서 자동 판별되지만, 수동으로 오버라이드도 가능합니다.

### Mode S (SDK): Claude Agent SDK

Claude 모델 전용입니다. Claude Code 서브프로세스를 사용하여 가장 풍부한 도구 실행이 가능합니다.

- **대상 모델**: `claude-*` (예: `claude-sonnet-4-6`, `claude-opus-4-6`)
- **특징**: 파일 조작, Bash 실행, 기억의 자율 검색을 모두 Claude Agent SDK 경유로 수행
- **credential**: `anthropic`을 사용 (MUST)

### Mode A (Autonomous): LiteLLM + tool_use 루프

tool_use를 지원하는 비 Claude 모델용입니다. LiteLLM으로 프로바이더를 통합합니다.

- **대상 모델**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b` 등
- **특징**: LiteLLM 경유로 tool_use 루프를 실행. 도구 실행은 프레임워크가 디스패치
- **credential**: 각 프로바이더에 대응하는 credential을 지정

### Mode B (Basic): Assisted (LLM은 사고만)

tool_use 미지원 모델용입니다. LLM은 사고만 수행하고, 기억 I/O는 프레임워크가 대행합니다.

- **대상 모델**: `ollama/gemma3*`, `ollama/phi4*`, 소규모 Ollama 모델 등
- **특징**: 1샷으로 응답을 생성. 도구 실행 불가
- **credential**: 보통 `ollama` 등의 로컬 credential

### 모드 자동 판별 방법

`~/.animaworks/models.json`에서 명시적으로 매핑을 추가할 수 있습니다.
미지정인 경우 코드 내의 기본 패턴(fnmatch 형식)으로 매칭됩니다.

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "B"
  }
}
```

판별 우선순위:
1. Anima의 `execution_mode` 필드 (per-Anima override)
2. `~/.animaworks/models.json` (완전 일치 → 와일드카드)
3. `config.json`의 `model_modes` (비권장 폴백)
4. 코드의 기본 패턴 (완전 일치 → 와일드카드)
5. 어디에도 매칭되지 않으면 `B` (안전 측으로 폴백)

## credential 설정

API 키는 `credentials` 섹션에서 이름으로 관리합니다.

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-...",
      "base_url": null
    },
    "openai": {
      "api_key": "sk-...",
      "base_url": null
    },
    "ollama": {
      "api_key": "",
      "base_url": "http://localhost:11434"
    }
  }
}
```

각 Anima는 `credential` 필드로 어떤 credential을 사용할지 지정합니다.

- `api_key` — API 키 문자열. 빈 문자열이면 환경 변수에서 폴백 시도
- `base_url` — 커스텀 엔드포인트. Ollama나 프록시 이용 시 설정. `null`이면 기본값

**보안**: config.json은 파일 퍼미션 `0600`으로 저장됩니다 (MUST). API 키를 포함하므로 다른 사용자의 읽기를 방지하세요.

## 권한 설정 (permissions.json)

각 Anima의 `permissions.json`에서 사용 가능한 도구, 접근 가능한 경로, 실행 가능한 명령어를 정의합니다.

```markdown
# Permissions: aoi

## 사용 가능한 도구
Read, Write, Edit, Bash, Grep, Glob

## 읽기 가능한 경로
- 자신의 디렉토리 하위 전체
- /shared/

## 쓰기 가능한 경로
- 자신의 디렉토리 하위 전체

## 실행 가능한 명령어
일반적인 명령어

## 실행 불가능한 명령어
rm -rf, 시스템 설정 변경

## 외부 도구
- image_gen: yes
- web_search: yes
- slack: no
```

권한 관련 규칙:
- 각 Anima는 자신의 `permissions.json`를 기동 시 읽습니다 (MUST)
- ToolHandler가 권한 검사를 수행하고 허용되지 않은 조작을 차단합니다
- 외부 도구 (Slack, Gmail, GitHub 등)는 `외부 도구` 섹션에서 개별적으로 허용/거부합니다
- `읽기 가능한 경로` / `쓰기 가능한 경로`는 자연어로 기술하며, ToolHandler가 해석합니다

### 차단 명령어

`permissions.json`에 `## 실행 불가능한 명령어` 섹션을 기재하면 지정된 명령어의 실행이 차단됩니다.
시스템 전체의 하드코딩된 차단 목록(`rm -rf /` 등의 위험한 명령어)에 더해, Anima 개별 차단 목록이 적용됩니다.

```markdown
## 실행 불가능한 명령어
rm -rf, docker rm, git push --force
```

파이프라인 내 명령어도 개별적으로 검사됩니다 (예: `cat file | rm -rf`에서 `rm -rf`가 차단됨).

## Anima 설정 해석 (2계층 머지)

Anima의 모델 설정은 **`status.json`이 Single Source of Truth (SSoT)**입니다.

### 설정 해석의 2계층 구조

| 우선도 | 소스 | 설명 |
|--------|------|------|
| 1 (최우선) | `status.json` | 각 Anima 디렉토리에 위치. 모델 및 실행 파라미터의 전체 설정을 보유 |
| 2 (폴백) | `anima_defaults` | `config.json`의 전체 기본값. `status.json`에 미설정인 필드에 적용 |

`config.json`의 `animas` 섹션은 **조직 레이아웃** (`supervisor`, `speciality`)만 보유합니다.
모델 이름, credential, max_turns 등의 모델 설정은 `status.json`에 기록됩니다.

### status.json 구조

파일 경로: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-4-6",
  "credential": "anthropic",
  "max_tokens": 16384,
  "max_turns": 200,
  "max_chains": 10,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### 모델 변경

CLI 명령어로 모델을 변경합니다:

```bash
animaworks anima set-model <anima_name> <model_name> [--credential <credential_name>]

# 전체 Anima의 모델을 일괄 변경
animaworks anima set-model --all <model_name>
```

supervisor가 부하의 모델을 변경하는 경우 `set_subordinate_model` 도구를 사용합니다.

### 설정 리로드

`status.json` 변경 후 프로세스를 재시작하지 않고 설정을 반영하려면 `reload` 명령어를 사용합니다:

```bash
# 단일 Anima 리로드
animaworks anima reload <anima_name>

# 전체 Anima 리로드
animaworks anima reload --all
```

리로드는 IPC 경유로 즉시 반영됩니다 (다운타임 없음). 실행 중인 세션은 이전 설정으로 완료되고, 다음 세션부터 새 설정이 적용됩니다.

**일반적인 설정 변경 워크플로우**:

1. `animaworks anima set-model <name> <model>`로 모델 변경
2. `animaworks anima reload <name>`로 즉시 반영

수동으로 `status.json`을 편집한 경우에도 동일하게 `reload`로 반영할 수 있습니다.

### 기본값 목록 (anima_defaults)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `model` | `claude-sonnet-4-6` | 사용할 LLM 모델 |
| `max_tokens` | `8192` | 1회 응답의 최대 토큰 수 |
| `max_turns` | `20` | 1세션의 최대 턴 수 |
| `credential` | `"anthropic"` | 사용할 credential 이름 |
| `context_threshold` | `0.50` | 컨텍스트 사용률이 이 임계값을 초과하면 단기 기억을 외부화 |
| `max_chains` | `2` | 자동 세션 연속의 최대 횟수 |

### 계층 구조

`supervisor` 필드만으로 조직 계층을 정의합니다 (`config.json`의 `animas` 섹션에 기재).

- `supervisor: null` — 최상위 Anima (지휘 체계의 최상위)
- `supervisor: "aoi"` — aoi의 부하로 동작

계층은 메시징을 통한 지시 및 보고로 기능합니다. 상사는 부하에게 작업을 위임할 수 있고, 부하는 상사에게 결과를 보고합니다.

## Anima 관리 명령어 목록

일상적인 Anima 조작 및 관리에 사용하는 CLI 명령어입니다.
서버가 기동 중(`animaworks start` 완료)인 상태에서 실행합니다.

| 명령어 | 설명 | 다운타임 |
|--------|------|----------|
| `animaworks anima list` | 전체 Anima의 목록과 상태 표시 | 없음 |
| `animaworks anima status [name]` | 지정 Anima (생략 시 전체)의 프로세스 상태 표시 | 없음 |
| `animaworks anima reload <name>` | status.json을 다시 읽어 모델 설정을 즉시 반영 (프로세스 재시작 없음) | 없음 |
| `animaworks anima reload --all` | 전체 Anima 설정 일괄 리로드 | 없음 |
| `animaworks anima restart <name>` | Anima 프로세스를 완전히 재시작 (코드 변경 반영 시 사용) | 15-30초 |
| `animaworks anima set-model <name> <model>` | 모델 변경 (status.json 업데이트. 반영에는 `reload` 필요) | 없음 |
| `animaworks anima set-model --all <model>` | 전체 Anima의 모델 일괄 변경 | 없음 |
| `animaworks anima enable <name>` | 비활성화된 Anima를 활성화하고 프로세스 기동 | — |
| `animaworks anima disable <name>` | Anima 비활성화 (프로세스 중지, status.json의 enabled=false) | — |
| `animaworks anima create` | 새 Anima 생성 (`--from-md`, `--template`, `--blank`) | — |
| `animaworks anima delete <name>` | Anima 삭제 (기본적으로 아카이브 보존) | — |

### 서버 관리 명령어

| 명령어 | 설명 |
|--------|------|
| `animaworks start` | 서버 기동 |
| `animaworks stop` | 서버 정지 |
| `animaworks restart` | 서버 완전 재시작 (전체 프로세스 재생성) |
| `animaworks status` | 시스템 전체 상태 표시 |
| `animaworks reset` | 서버 정지 후 런타임 디렉토리를 삭제하고 재초기화 (파괴적) |
| `animaworks reset --restart` | 위 작업 후 서버 재시작 |

### reload / restart / 서버 재시작의 구분

| 명령어 | 동작 | 다운타임 | 유스케이스 |
|--------|------|----------|-----------|
| `anima reload` | IPC로 ModelConfig 스왑 | 없음 | status.json의 모델/파라미터 변경 |
| `anima restart` | 프로세스 kill → 재생성 | 15-30초 | 코드 변경 반영, 메모리 누수 대응 |
| 서버 restart | 전체 Anima 재시작 + 신규 감지 | 15-30초 | Anima 추가/삭제 반영 |
