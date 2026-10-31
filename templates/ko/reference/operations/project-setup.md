# 프로젝트 설정 방법

AnimaWorks의 설정 구조와 Anima 추가 절차에 대한 참조.
설정 변경이 필요한 상황에서 검색하고 참조할 것.

## 런타임 디렉터리 초기화

런타임 데이터는 `~/.animaworks/`(또는 `ANIMAWORKS_DATA_DIR`)에 배치된다.
최초 설정 시 `animaworks init`로 템플릿에서 초기화한다.

### init 명령어

| 명령어 | 설명 |
|---------|------|
| `animaworks init` | 런타임 디렉터리 초기화(비대화형). 이미 존재하면 아무 작업도 하지 않음 |
| `animaworks init --force` | 기존 데이터에 템플릿 차이를 병합. 새 파일만 추가, `prompts/`는 덮어쓰기 |
| `animaworks init --skip-anima` | 인프라만 초기화. Anima 생성을 건너뜀 |
| `animaworks init --template NAME` | 템플릿에서 Anima를 비대화형으로 생성 |
| `animaworks init --from-md PATH` | MD 파일에서 Anima를 비대화형으로 생성 |
| `animaworks init --blank NAME` | 빈 Anima를 비대화형으로 생성 |

**권장 흐름**: `animaworks init` 실행 후, `animaworks start`로 서버를 시작하고 웹 UI의 설정 마법사에서 Anima를 추가한다.

### 초기화 시 생성되는 디렉터리

`ensure_runtime_dir`(`core/infra/runtime_init.py`)에 의해 다음이 생성됨:

- `animas/` — Anima 디렉터리
- `shared/inbox/` — 수신 메시지 큐
- `shared/users/` — 사용자 프로필
- `shared/channels/` — 공유 채널(general, ops의 초기 파일)
- `shared/dm_logs/` — DM 이력(폴백용)
- `tmp/attachments/` — 첨부 파일 임시 저장
- `common_skills/` / `common_knowledge/` — 공통 스킬·지식
- `prompts/` / `company/` — 프롬프트·조직 템플릿
- `tool_prompts.sqlite3` — 도구 프롬프트 DB
- `models.json` — 모델명→실행 모드 매핑(`config_defaults/`에서 복사)

시작할 때마다 `common_skills`와 `common_knowledge`은 템플릿에서 증분 동기화되며, 새 항목만 추가된다(기존 파일은 유지).

## config.json의 전체 구조

AnimaWorks의 통합 설정 파일은 `~/.animaworks/config.json`에 배치된다.
모든 설정은 `AnimaWorksConfig` 모델로 정의되어 있으며, 다음과 같은 최상위 필드를 가진다.

```json
{
  "version": 1,
  "setup_complete": true,
  "locale": "ja",
  "system": { "mode": "server" },
  "credentials": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openai": { "api_key": "sk-..." }
  },
  "model_modes": {},
  "anima_defaults": { "model": "claude-sonnet-5-5", "max_tokens": 8192 },
  "animas": {
    "aoi": { "supervisor": null, "speciality": null },
    "taro": { "supervisor": "aoi", "speciality": null }
  },
  "consolidation": { "daily_enabled": true, "daily_time": "02:00" },
  "rag": { "enabled": true },
  "priming": { "max_tokens": 2000 },
  "image_gen": {}
}
```

**주의**: `animas` 섹션은 조직 레이아웃(`supervisor`, `speciality`)만 보유한다. 모델 이름·credential 등의 모델 설정은 각 Anima의 `status.json`에 기록된다(후술의 「Anima 설정의 해결」 참조).

각 섹션의 역할:

| 섹션 | 설명 |
|-----------|------|
| `version` | 설정 스키마 버전(현재 `1`) |
| `setup_complete` | 최초 설정 완료 플래그 |
| `locale` | UI 언어(`"ja"` / `"en"`) |
| `system` | 서버 모드·타임존 |
| `credentials` | API 키·엔드포인트(이름 지정) |
| `model_modes` | 모델 이름→실행 모드의 오버라이드 맵 |
| `anima_defaults` | 모든 Anima 공통의 기본 설정 |
| `animas` | Anima의 조직 레이아웃(supervisor, speciality). 모델 설정은 status.json |
| `consolidation` | 기억 통합(일일/주간)의 설정 |
| `rag` | RAG(임베딩 벡터 검색)의 설정 |
| `priming` | 프라이밍(자동 기억 획득)의 토큰 예산 |
| `image_gen` | 이미지 생성의 스타일 설정 |

<!-- AUTO-GENERATED:START config_fields -->
### 설정 항목 참조(자동 생성)

#### Anima 설정 (Anima별 재정의)

| 필드 | 유형 | 기본값 | 설명 |
|-----------|-----|----------|------|
| `supervisor` | `str | None` | None |  |
| `company` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `model` | `str | None` | None |  |
| `heartbeat_enabled` | `bool | None` | None |  |
| `background_review_enabled` | `bool | None` | None |  |
| `token_budget_monthly` | `int | None` | None |  |
| `aliases` | `list[str]` | `[]` |  |

#### 기본값 (anima_defaults)

| 필드 | 타입 | 기본값 | 설명 |
|-----------|-----|----------|------|
| `model` | `str` | `"claude-sonnet-5-5"` |  |
| `fallback_model` | `str | None` | None |  |
| `fallback_models` | `list[str]` | `[]` |  |
| `background_model` | `str | None` | None |  |
| `background_credential` | `str | None` | None |  |
| `background_thinking_effort` | `str | None` | None |  |
| `voice_thinking_effort` | `str | None` | None |  |
| `max_tokens` | `int` | `8192` |  |
| `credential` | `str` | `"anthropic"` |  |
| `context_threshold` | `float` | `0.5` |  |
| `context_absolute_ceiling` | `float` | `0.75` |  |
| `task_compaction_tokens` | `int` | `0` |  |
| `task_compaction_max` | `int` | `6` |  |
| `max_session_age_hours` | `float` | `24.0` |  |
| `conversation_history_threshold` | `float` | `0.3` |  |
| `execution_mode` | `str | None` | None |  |
| `supervisor` | `str | None` | None |  |
| `speciality` | `str | None` | None |  |
| `extra_mcp_servers` | `dict[str, dict]` | `{}` |  |
| `thinking` | `bool | None` | None |  |
| `thinking_effort` | `str | None` | None |  |
| `mode_s_auth` | `str | None` | None |  |
| `default_workspace` | `str` | `""` |  |
| `consolidation_enabled` | `bool` | `True` |  |
| `heartbeat_enabled` | `bool` | `True` |  |
| `token_budget_monthly` | `int | None` | None |  |

#### AnimaWorksConfig 최상위

| 섹션 | 설명 |
|-----------|------|
| `version` | 설정 파일 버전 |
| `setup_complete` | 설정 완료 플래그 |
| `locale` | 로캘 설정 |
| `system` | 시스템 설정(모드, 로그 수준) |
| `credentials` | API 인증 정보 |
| `model_modes` | 모델명→실행 모드 매핑 |
| `model_context_windows` |  |
| `model_max_tokens` |  |
| `anima_defaults` | Anima 설정 기본값 |
| `animas` | Anima별 설정 오버라이드 |
| `consolidation` | 기억 통합 설정 |
| `helper_models` |  |
| `background_review` |  |
| `rag` | RAG(검색 증강 생성) 설정 |
| `gpu` |  |
| `memory` |  |
| `skills` |  |
| `chatwork_tool` |  |
| `prompt` |  |
| `priming` | 프라이밍(자동 기억 회상) 설정 |
| `image_gen` | 이미지 생성 설정 |
| `human_notification` |  |
| `interaction` |  |
| `server` |  |
| `llm_rate_guard` |  |
| `mcp` |  |
| `external_messaging` |  |
| `external_tasks` |  |
| `github_webhook` |  |
| `event_export` |  |
| `background_task` |  |
| `activity_log` |  |
| `logging` |  |
| `heartbeat` |  |
| `voice` |  |
| `phone` |  |
| `housekeeping` |  |
| `inbox` |  |
| `local_llm` |  |
| `workspaces` |  |
| `github_identities` |  |
| `activity_level` |  |
| `activity_schedule` |  |
| `icon_url_template` |  |
| `ui` |  |
| `cli` |  |
| `enclave` |  |
| `enclaves` |  |

<!-- AUTO-GENERATED:END -->

## 새 Anima 추가 방법

Anima 추가 방법은 3가지가 있다. 모두 `animaworks anima create` 또는 `animaworks init`로 실행한다.
`animaworks anima create`는 `--role`와 `--supervisor` 옵션이 있으며, 권장된다.

### 방법 1: 템플릿에서 생성

사전 정의된 템플릿(`templates/ja/anima_templates/` 또는 `templates/en/anima_templates/` 아래)을 사용하는 방법입니다.
템플릿에는 IDENTITY.md, SOUL.md, AGENTS.md, permissions.json, 스킬 등이 포함되어 있습니다.

```bash
# テンプレートから作成（テンプレート名はディレクトリ名）
animaworks anima create --template <テンプレート名>

# 名前を変えて作成
animaworks anima create --template <テンプレート名> --name <anima名>
```

템플릿을 사용하는 방법이 가장 권장됩니다. 캐릭터 설정이 이미 갖춰져 있어 bootstrap(최초 시작 시 자기 정의)을 건너뛸 수 있습니다.

### 방법 2: Markdown 파일에서 생성

캐릭터 시트(Markdown)를 준비하고, 이를 바탕으로 Anima를 생성합니다.

```bash
animaworks anima create --from-md /path/to/character.md [--name ken] [--role engineer] [--supervisor aoi]
```

- `--name`: Anima 이름(생략하면 시트에서 추출)
- `--role`: 역할 템플릿(engineer, researcher, manager, writer, ops, general). 기본값: general
- `--supervisor`: 상급자 Anima 이름(캐릭터 시트의 ‘상급자’를 덮어씀)

Markdown 파일은 `character_sheet.md`로 Anima 디렉터리에 복사됩니다.
시트의 ‘기본 정보’는 IDENTITY.md에, ‘성격’은 SOUL.md에 반영됩니다(프로필 제목은 IDENTITY.md에 반영). ‘역할·행동 방침’은 AGENTS.md에 반영됩니다.
역할 템플릿에서 permissions.json과 specialty_prompt.md가 적용됩니다.

Markdown 파일에는 다음 항목을 포함하는 것이 SHOULD입니다.
- `# Character: 名前` 형식의 제목 또는 기본 정보 표의 ‘영문명’ 행(이름 자동 추출에 사용)
- `## 基本情報` — 영문명, 상급자, 모델 등의 표. IDENTITY.md에 반영
- `## 人格` — SOUL.md에 반영(프로필 관련 제목은 IDENTITY.md로 분류)
- `## 役割・行動方針` — AGENTS.md에 반영

### 방법 3: 빈 생성

최소한의 스켈레톤 파일로 Anima를 생성한다.

```bash
animaworks anima create --name aoi
```

`--name`는 필수. 빈 생성에서는 `{name}` 플레이스홀더가 실명으로 치환된 스켈레톤 파일이 생성된다.
최초 시작의 bootstrap에서 에이전트가 사용자와의 대화를 통해 캐릭터를 자기 정의한다.

### 생성 후 디렉터리 구조

어떤 방법을 사용하든 다음 디렉터리와 파일이 생성됩니다.

```
~/.animaworks/animas/{name}/
├── IDENTITY.md          # プロフィール・外見（不変ベースライン）
├── SOUL.md              # 人格・価値観（不変ベースライン）
├── AGENTS.md            # 役割・行動指針（可変）
├── BOOTSTRAP.md         # 初回起動指示（完了後に削除）
├── permissions.json       # ツール・コマンド権限
├── HEARTBEAT.md         # ハートビート設定
├── cron.md              # 定時タスク設定
├── episodes/            # エピソード記憶（日別ログ）
├── knowledge/           # 意味記憶（学んだ知識）
├── procedures/          # 手続き記憶（手順書）
├── skills/              # 個人スキル
├── state/               # ワーキングメモリ
│   └── current_state.md  # 現在のタスク
└── shortterm/           # 短期記憶（セッション継続用）
    └── archive/
```

### Anima 이름 규칙

Anima 이름은 다음 규칙을 따라야 한다(MUST):
- 소문자 영숫자, 하이픈(`-`), 언더스코어(`_`)만 사용 가능
- 첫 글자는 영문자(`a-z`)여야 함
- 언더스코어 시작은 불가(템플릿 예약)
- 예: `aoi`, `taro-dev`, `worker01`

## 실행 모드(S / C / D / G / A / B)

AnimaWorks는 **6**개의 실행 모드를 가진다. 모델명에서 자동 판정되지만, `status.json`의 `execution_mode`로 덮어쓸 수도 있다.

### Mode S (SDK): Claude Agent SDK

Claude 모델 전용. Claude Code 서브프로세스를 사용하여 가장 풍부한 도구 실행이 가능.

- **대상 모델**: `claude-*`(예: `claude-sonnet-5-5`, `claude-opus-5-5`)
- **특징**: 파일 조작, Bash 실행, 기억의 자율 검색을 모두 Claude Agent SDK를 통해 수행
- **credential**: `anthropic` 사용(MUST)

### Mode C (Codex): Codex CLI

OpenAI Codex 계열을 Codex CLI 래퍼를 통해 실행합니다.

- **대상 모델**: `codex/*` (예: `codex/o4-mini`, `codex/gpt-4.1`)
- **특징**: MCP 도구와 AnimaWorks 외부 도구의 통합 경로는 Mode S/D/G와 유사
- **credential**: Codex / OpenAI 측 요구 사항에 따름

### Mode D (Cursor Agent): Cursor Agent CLI

Cursor의 `cursor-agent` CLI를 하위 프로세스로 실행합니다. MCP를 통해 AnimaWorks 도구를 사용합니다.

- **대상 모델**: `cursor/*`
- **특징**: 호스트에 CLI와 인증이 필요합니다. 실패 시 Mode A (LiteLLM)로 전환 가능
- **credential**: Cursor / `agent login`의 인증 상태에 의존

### Mode G (Gemini CLI): Gemini CLI

Google의 `gemini` CLI를 하위 프로세스로 실행합니다. MCP 통합.

- **대상 모델**: `gemini/*`
- **특징**: CLI 또는 `GEMINI_API_KEY`가 필요합니다. 폴백 시 Mode A에서 `gemini/` → `google/` 등으로 리매핑될 수 있음
- **credential**: CLI 로그인 또는 API 키

### Mode A (Autonomous): LiteLLM + tool_use 루프

tool_use를 지원하는 클라우드·로컬 모델용입니다. LiteLLM으로 프로바이더를 통일합니다.

- **대상 모델**: `openai/gpt-4.1`, `google/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash`, `ollama/qwen3:30b` 등
- **특징**: LiteLLM을 통해 tool_use 루프를 실행합니다. 도구 실행은 프레임워크가 디스패치
- **credential**: 각 프로바이더에 대응하는 credential을 지정

Mode B는 폐지되었습니다. tool_use 미지원 모델은 권장하지 않습니다. 설정에 남아 있는 `execution_mode: "B"`는 Mode A로 처리됩니다.

### 모드 자동 판정 메커니즘

`~/.animaworks/models.json`에서 명시적으로 매핑을 추가할 수 있습니다.
미지정 시 코드 내 기본 패턴(fnmatch 형식)으로 매칭됩니다.

```json
{
  "model_modes": {
    "ollama/my-custom-model": "A",
    "ollama/experimental-*": "A"
  }
}
```

판정 우선순위:
1. Anima의 `execution_mode` 필드 (per-anima override)
2. `~/.animaworks/models.json` (완전 일치 → 와일드카드)
3. `config.json`의 `model_modes` (비권장 폴백)
4. 코드의 기본 패턴 (완전 일치 → 와일드카드)
5. 어디에도 매칭되지 않으면 Mode A

## 크레덴셜 설정

API 키는 `credentials` 섹션에서 이름을 붙여 관리합니다.

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

각 Anima는 `credential` 필드에서 어떤 credential을 사용할지 지정합니다.

- `api_key` — API 키 문자열. 빈 문자열인 경우 환경 변수에서 폴백을 시도
- `base_url` — 커스텀 엔드포인트. Ollama나 프록시 사용 시 설정합니다. `null`에서 기본값

**보안**: config.json는 파일 권한 `0600`으로 저장됩니다 (MUST). API 키를 포함하므로 다른 사용자의 읽기를 방지합니다.

## 권한 설정 (permissions.json)

각 Anima의 `permissions.json`에서 사용 가능한 도구·접근 가능한 경로·실행 가능한 명령을 정의합니다.

```markdown
# Permissions: aoi

## 使えるツール
Read, Write, Edit, Bash, Grep, Glob

## 読める場所
- 自分のディレクトリ配下すべて
- /shared/ 配下

## 書ける場所
- 自分のディレクトリ配下すべて

## 実行できるコマンド
全般的なコマンド

## 実行できないコマンド
rm -rf, システム設定の変更

## 外部ツール
- image_gen: yes
- web_search: yes
- slack: no
```

권한 관련 규칙:
- 각 Anima는 자신의 `permissions.json`를 시작 시 읽습니다 (MUST)
- ToolHandler가 권한 체크를 수행하고, 허용되지 않은 작업을 차단
- 외부 도구 (Slack, Gmail, GitHub 등)는 `外部ツール` 섹션에서 개별적으로 허용/거부
- `読める場所` / `書ける場所`는 자연어로 기술하며, ToolHandler가 해석

### 블록 명령

`permissions.json`에 `## 実行できないコマンド` 섹션을 기재하면, 지정된 명령의 실행이 차단됩니다.
시스템 전체의 하드코딩된 블록 목록 (`rm -rf /` 등의 위험한 명령)에 더해, Anima 개별 블록 목록이 적용됩니다.

```markdown
## 実行できないコマンド
rm -rf, docker rm, git push --force
```

파이프라인 중의 명령도 개별적으로 체크됩니다 (예: `cat file | rm -rf`는 `rm -rf`가 차단됨).

## Anima 설정의 해결 (2계층 병합)

Anima의 모델 설정은 **`status.json`가 Single Source of Truth (SSoT)** 입니다.

### 설정 해결의 2계층 구조

| 우선순위 | 소스 | 설명 |
|--------|--------|------|
| 1 (최우선) | `status.json` | 각 Anima 디렉터리에 배치. 모델·실행 파라미터의 전체 설정을 보유 |
| 2 (폴백) | `anima_defaults` | `config.json`의 전체 기본값. `status.json`에 미설정된 필드에 적용 |

`config.json`의 `animas` 섹션은 **조직 레이아웃** (`supervisor`, `speciality`)만 보유합니다.
모델 이름·credential 등의 모델 설정은 `status.json`에 기록됩니다.

### status.json의 구조

파일 경로: `~/.animaworks/animas/{name}/status.json`

```json
{
  "enabled": true,
  "role": "engineer",
  "model": "claude-opus-5-5",
  "credential": "anthropic",
  "max_tokens": 16384,
  "context_threshold": 0.80,
  "execution_mode": null
}
```

### 모델 변경

모델을 변경하려면 CLI 명령을 사용합니다:

```bash
animaworks anima set-model <anima名> <モデル名> [--credential <credential名>]

# 全 Anima のモデルを一括変更
animaworks anima set-model --all <モデル名>
```

슈퍼바이저가 부하의 모델을 변경하는 경우 `set_subordinate_model` 도구를 사용합니다.

### 설정 리로드

root/CLI가 허용된 `status.json` 설정을 변경한 후, 프로세스를 재시작하지 않고 모델 설정을 반영하려면 `reload` 명령을 사용한다. `set-model` 명령은 시작 중인 Anima에 reload를 요청한다:

```bash
# 単一 Anima のリロード
animaworks anima reload <anima名>

# 全 Anima のリロード
animaworks anima reload --all
```

리로드는 IPC를 통해 즉시 반영된다(다운타임 없음). 실행 중인 세션은 이전 설정으로 완료되고, 다음 세션부터 새 설정이 적용된다.

**일반적인 설정 변경 워크플로우**:

1. `animaworks anima set-model <name> <model>`로 모델을 변경한다(root API가 `status.json`를 업데이트하고, 시작 중인 Anima에 reload를 요청한다)
2. 자동 알림되지 않는 root/CLI 업데이트 후에는 `animaworks anima reload <name>`을 사용한다

Anima 프로세스에서 `status.json`을 직접 편집하지 않는다. root 소유 CLI/API 작업을 사용한다.

### 기본값 목록 (anima_defaults)

| 필드 | 기본값 | 설명 |
|-----------|-------------|------|
| `model` | `claude-sonnet-5-5` | 사용할 LLM 모델 |
| `max_tokens` | `8192` | 1회 응답의 최대 토큰 수 |
| `credential` | `"anthropic"` | 사용할 credential 이름 |
| `context_threshold` | `0.50` | 컨텍스트 사용률이 이 임계값을 초과하면 단기 기억을 외부화 |

### 계층 구조

`supervisor` 필드만으로 조직 계층을 정의합니다 (`config.json`의 `animas` 섹션에 기재).

- `supervisor: null` — 톱레벨 Anima (지휘 계통의 최상위)
- `supervisor: "aoi"` — aoi의 부하로 동작

계층은 메시징을 통한 지침·보고로 기능합니다. 상급자는 부하에게 작업을 위임할 수 있고, 부하는 상급자에게 결과를 보고합니다.

## Anima 관리 명령어 목록

일상적인 Anima의 운영·관리에 사용하는 CLI 명령어.
서버가 시작 중(`animaworks start` 완료)인 상태에서 실행한다.

| 명령어 | 설명 | 다운타임 |
|---------|------|-----------|
| `animaworks anima list` | 전체 Anima의 목록과 상태를 표시 | 없음 |
| `animaworks anima status [name]` | 지정 Anima(생략 시 전체)의 프로세스 상태를 표시 | 없음 |
| `animaworks anima reload <name>` | status.json을 다시 로드하여 모델 설정을 즉시 반영(프로세스 재시작 없음) | 없음 |
| `animaworks anima reload --all` | 전체 Anima의 설정을 일괄 리로드 | 없음 |
| `animaworks anima restart <name>` | Anima 프로세스를 완전히 재시작(코드 변경 반영 시 사용) | 15-30초 |
| `animaworks anima set-model <name> <model>` | root 소유 status를 업데이트하고, 시작 중 Anima의 모델 설정을 reload | 없음 |
| `animaworks anima set-model --all <model>` | 전체 Anima의 모델을 일괄 변경 | 없음 |
| `animaworks anima enable <name>` | 휴지 중인 Anima를 활성화하여 프로세스를 시작 | — |
| `animaworks anima disable <name>` | Anima를 휴지(프로세스 종료, status.json의 enabled=false) | — |
| `animaworks anima create` | 새 Anima를 생성(`--from-md`, `--template`, `--blank`) | — |
| `animaworks anima delete <name>` | Anima를 삭제(기본적으로 아카이브 저장) | — |

### 서버 관리 명령

| 명령 | 설명 |
|---------|------|
| `animaworks start` | 서버 시작 |
| `animaworks stop` | 서버 종료 |
| `animaworks restart` | 서버 완전 재시작 (전체 프로세스 재생성) |
| `animaworks status` | 시스템 전체의 상태를 표시 |
| `animaworks reset` | 서버 종료 후, 런타임 디렉터리를 삭제하고 재초기화 (파괴적) |
| `animaworks reset --restart` | 위의 후, 서버를 재시작 |

### reload / restart / system reload의 역할 구분

| 명령 | 동작 | 다운타임 | 사용 사례 |
|---------|------|-----------|-------------|
| `anima reload` | IPC로 ModelConfig 교체 | 없음 | status.json의 모델/파라미터 변경 |
| `anima restart` | 프로세스 kill → 재생성 | 15-30초 | 코드 변경 반영, 메모리 누수 대책 |
| 서버 restart | 전체 Anima 재시작 + 신규 검출 | 15-30초 | Anima 추가/삭제 반영 |
