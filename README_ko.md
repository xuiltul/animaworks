# AnimaWorks — Organization-as-Code

**혼자서는 아무것도 할 수 없습니다. 그래서 조직을 만들었습니다.**

AI 에이전트를 "도구"가 아닌 "자율적으로 일하는 사람"으로 다루는 프레임워크입니다. 각 Anima는 고유한 이름과 성격, 기억, 스케줄을 지니고 있습니다. 메시지로 대화하고 스스로 판단하며 팀으로 움직입니다. 리더에게 한마디만 건네면 나머지는 알아서 돌아갑니다.

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — 실시간 조직도와 활동 피드" width="720">
  <br><em>Workspace 대시보드: 각 Anima의 역할, 상태, 최근 활동을 실시간으로 확인.</em>
</p>

<p align="center">
  <img src="docs/images/workspace-demo.gif" alt="AnimaWorks 3D Workspace — 에이전트가 자율적으로 협업" width="720">
  <br><em>3D 오피스: Anima들이 책상에 앉아 일하고, 돌아다니며 메시지를 주고받음 — 모두 자율적으로.</em>
</p>

**[English README](README.md)** | **[日本語版 README](README_ja.md)** | **[简体中文 README](README_zh.md)**

---

## 다른 프레임워크와의 비교

|  | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents |
|--|-----------|--------|-----------|----------|---------------|
| **설계 철학** | 자율 에이전트 조직 | 역할 기반 팀 | 그래프 워크플로 | 개인 어시스턴트 | 경량 SDK |
| **기억** | 뇌과학 기반: 통합, 3단계 망각, 6채널 자동 회상 (신뢰 태그 포함) | 인지 메모리 (수동 forget) | 체크포인트 + 크로스스레드 저장 | SuperMemory 지식 그래프 | 세션 범위 |
| **자율성** | 하트비트(관찰→계획→회고) + Cron + TaskExec — 24/7 운영 | 수동 트리거 | 수동 트리거 | Cron + 하트비트 | 수동 트리거 |
| **조직 구조** | 상사→부하 계층, 위임, 감사, 대시보드 | Crew 내 플랫 역할 | — | 단일 에이전트 | Handoff만 |
| **프로세스** | 에이전트별 독립 Unix 프로세스, 소켓 IPC, 자동 재시작 | 공유 프로세스 | 공유 프로세스 | 단일 프로세스 | 공유 프로세스 |
| **멀티 모델** | 4 엔진: Claude SDK / Codex / LiteLLM / Assisted | LiteLLM | LangChain 모델 | OpenAI 호환 | OpenAI 중심 |

> AnimaWorks는 태스크 러너가 아닙니다 — 생각하고, 기억하고, 잊고, 성장하는 조직입니다. 팀으로서 업무를 지원하고, 회사로서 운영할 수 있습니다.

---

## :rocket: 지금 바로 체험 — Docker 데모

60초면 충분합니다. API 키와 Docker만 있으면 됩니다.

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
cp .env.example .env          # ANTHROPIC_API_KEY를 붙여넣기
docker compose up              # http://localhost:18500 을 열기
```

3인 팀(매니저 + 엔지니어 + 코디네이터)이 바로 실행되고, 3일치 활동 이력이 미리 로드된 상태로 시작됩니다. [데모 상세 보기 →](demo/README.md)

> 언어/스타일 변경: `PRESET=ja-anime docker compose up` — [전체 프리셋 목록](demo/README.md#presets)

---

## 퀵 스타트

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start     # 서버 시작 — 첫 실행 시 설정 마법사가 열림
```

**http://localhost:18500/** 을 열면 설정 마법사가 단계별로 안내해줍니다:

1. **언어** — UI 표시 언어 선택
2. **사용자 정보** — 소유자 계정 생성
3. **API 키** — LLM API 키 입력 (실시간 연결 검증)
4. **첫 Anima** — 첫 번째 에이전트에 이름 부여

`.env` 파일을 직접 편집할 필요는 없습니다. 설정 마법사가 `config.json`에 자동으로 저장해줍니다.

설정 스크립트가 [uv](https://docs.astral.sh/uv/) 설치부터 리포지토리 클론, Python 3.12+ 및 모든 의존성 다운로드까지 알아서 처리합니다. **macOS, Linux, WSL**에서 Python 사전 설치 없이 바로 동작합니다.

> **다른 LLM을 사용하고 싶다면?** AnimaWorks는 Claude, GPT, Gemini, 로컬 모델 등 다양한 LLM을 지원해요. 설정 마법사에서 API 키를 입력하거나, 나중에 대시보드의 **Settings**에서 추가할 수도 있습니다. 자세한 내용은 [API 키 레퍼런스](#api-키-레퍼런스)를 참고하세요.

<details>
<summary><strong>대안: 스크립트를 확인한 후 실행</strong></summary>

`curl | bash`를 바로 실행하는 게 꺼려진다면, 먼저 스크립트 내용을 직접 확인하고 실행할 수 있습니다:

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh -o setup.sh
cat setup.sh            # 스크립트 내용 확인
bash setup.sh           # 확인 후 실행
```

</details>

<details>
<summary><strong>대안: uv로 단계별 수동 설치</strong></summary>

```bash
# uv 설치 (이미 설치되어 있으면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 클론 및 설치
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
uv sync                 # Python 3.12+와 모든 의존성을 자동 다운로드

# 시작
uv run animaworks start
```

</details>

<details>
<summary><strong>대안: pip으로 수동 설치</strong></summary>

> **macOS 사용자:** macOS Sonoma 이전 버전의 시스템 Python (`/usr/bin/python3`)은 3.9 버전이라 AnimaWorks의 요구사항(3.12+)을 충족하지 않습니다. [Homebrew](https://brew.sh/)로 `brew install python@3.13`을 설치하거나, Python을 자동으로 관리해주는 위의 uv 방식을 이용하세요.

Python 3.12 이상이 시스템에 설치되어 있어야 합니다.

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
python3 --version       # 3.12+ 인지 확인
pip install --upgrade pip && pip install -e .
animaworks start
```

</details>

---

## 주요 기능

### 대시보드

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks 대시보드 — 19개 에이전트 조직도" width="720">
  <br><em>대시보드: 4개 계층, 19개 Anima가 가동 중. 실시간 상태 표시.</em>
</p>

- **채팅** — 원하는 Anima와 실시간으로 대화. 스트리밍 응답, 이미지 첨부, 멀티스레드 대화, 전체 히스토리
- **음성 채팅** — 브라우저에서 바로 음성으로 대화 (푸시-투-토크 또는 핸즈프리). VOICEVOX / SBV2 / ElevenLabs 지원
- **Board** — Slack 스타일의 공유 채널. Anima들이 자율적으로 토론하고 협력
- **활동** — 조직 전체에서 일어나는 일을 실시간 피드로 확인
- **기억** — 각 Anima가 무엇을 기억하는지 확인 — 에피소드, 지식, 절차
- **3D Workspace** — Anima들이 3D 오피스에서 일하는 모습을 볼 수 있어요
- **다국어 지원** — UI 17개 언어 지원; 일본어 + 영어 템플릿, 자동 폴백

### 팀을 만들고, 맡기기

리더에게 필요한 인재를 말하기만 하면, 적절한 역할·성격·보고 체계를 판단해 새 멤버를 만들어줍니다. 설정 파일도 필요 없고, CLI 명령도 필요 없습니다. 대화만으로 조직이 성장합니다.

팀이 갖춰지면, 사람이 없어도 알아서 돌아갑니다:

- **하트비트** — 각 Anima가 주기적으로 상황을 확인하고, 다음에 할 일을 스스로 판단합니다
- **Cron 작업** — 일일 보고, 주간 요약, 모니터링 — Anima별로 스케줄 설정
- **태스크 위임** — 매니저가 부하에게 업무를 할당하고 진행 상황을 추적하며 보고를 받습니다
- **병렬 태스크 실행** — 여러 태스크를 동시에 제출하면 의존성을 해결하고 독립 태스크는 병렬로 실행
- **야간 통합** — 낮 동안 쌓인 에피소드 기억이 수면 중에 지식으로 정제됩니다
- **팀 협력** — 공유 채널과 DM으로 모든 구성원이 자동으로 동기화됩니다

### 기억 시스템

기존 AI 에이전트는 컨텍스트 윈도우에 들어오는 것만 기억합니다. AnimaWorks의 에이전트는 영속적인 기억을 가지고 있어서, 필요할 때 직접 검색해 떠올립니다. 책장에서 책을 꺼내는 것과 같습니다.

- **자동 회상(Priming)** — 메시지가 도착하면 6개 채널의 병렬 검색이 자동으로 실행됩니다: 발신자 프로필, 최근 활동, 관련 지식, 스킬, 미완료 태스크, 과거 에피소드. 따로 지시하지 않아도 알아서 기억해냅니다
- **통합(Consolidation)** — 매일 밤, 낮 동안 쌓인 에피소드가 지식으로 정제됩니다 — 뇌과학에서 말하는 수면 중 기억 공고화와 같은 메커니즘입니다. 해결된 이슈는 자동으로 절차서가 됩니다
- **망각(Forgetting)** — 사용되지 않는 기억은 3단계에 걸쳐 서서히 희미해집니다: 마킹, 병합, 아카이빙. 중요한 절차서와 스킬은 보호됩니다. 인간의 뇌와 마찬가지로, 잊는 것도 중요합니다

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks 채팅 — 여러 Anima와의 멀티스레드 대화" width="720">
  <br><em>채팅: 매니저가 코드 수정을 리뷰하고, 엔지니어가 진행 상황을 보고하는 모습.</em>
</p>

### 멀티 모델 지원

어떤 LLM이든 동작합니다. Anima마다 다른 모델을 사용할 수 있습니다.

| 모드 | 엔진 | 대상 | 도구 |
|------|------|------|------|
| S (SDK) | Claude Agent SDK | Claude 모델 (권장) | 풀: Read/Write/Edit/Bash/Grep/Glob |
| C (Codex) | Codex SDK | OpenAI Codex CLI 모델 | 풀: Mode S와 동일 |
| A (Autonomous) | LiteLLM + tool_use | GPT, Gemini, Mistral, vLLM 등 | search_memory, read/write_file, send_message 등 |
| B (Basic) | LiteLLM 1-shot | Ollama, 소형 로컬 모델 | 프레임워크가 모델 대신 기억 I/O를 대행 |

모드는 모델명에서 자동 감지됩니다. 하트비트, Cron, Inbox는 메인 모델보다 가벼운 모델로 실행할 수 있습니다 (비용 최적화). Extended thinking도 지원 모델에서 사용할 수 있습니다.

### 아바타 자동 생성

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks 에셋 관리 — 사실적인 초상화와 표정 변형" width="720">
  <br><em>전신, 상반신, 표정 변형 — 모두 성격 설정에서 자동 생성. 상사의 화풍을 자동 계승하는 Vibe Transfer 포함.</em>
</p>

NovelAI(애니메이션 스타일), fal.ai/Flux(스타일라이즈드/포토리얼), Meshy(3D 모델)를 지원합니다. 이미지 서비스를 설정하지 않아도 동작합니다 — 아바타가 없을 뿐이에요. 아바타가 생기면 정이 듭니다.

---

## 왜 AnimaWorks인가?

이 프로젝트는 세 가지 커리어가 만나는 지점에서 태어났습니다.

**경영자로서** — 혼자서는 아무것도 이룰 수 없다는 걸 뼈저리게 알고 있습니다. 뛰어난 엔지니어도 있어야 하고, 소통이 잘 되는 사람도 필요합니다. 묵묵히 일을 밀어붙이는 사람도, 때로 기발한 아이디어 하나로 판을 뒤집는 사람도 있어야 합니다. 천재 혼자서는 조직이 굴러가지 않습니다. 저마다의 강점을 한데 모을 때, 개인으로는 닿지 못할 곳에 닿을 수 있습니다.

**정신과 의사로서** — LLM의 내부 구조를 들여다봤을 때, 인간의 뇌와 놀라울 만큼 닮아 있다는 걸 발견했습니다. 회상, 학습, 망각, 공고화 — 뇌가 기억을 다루는 방식을 LLM의 기억 시스템으로 그대로 옮길 수 있습니다. 그렇다면 LLM을 유사 인간으로 대우하고, 사람처럼 조직을 꾸릴 수 있을 것입니다.

**엔지니어로서** — 30년 동안 코드를 써왔습니다. 로직을 짜는 즐거움, 자동화가 돌아갈 때의 쾌감을 잘 알고 있습니다. 모든 이상을 코드에 담으면, 내가 바라는 조직을 직접 만들 수 있습니다.

우수한 "단독 AI 비서" 프레임워크는 이미 많습니다. 하지만 코드로 인간을 재현하고, 그것을 조직으로 기능하게 만든 프로젝트는 아직 없었습니다. AnimaWorks는 실제 사업에 통합하며 매일 키워가고 있는, 진짜 조직 그 자체입니다.

> *불완전한 개인들의 협업이 단일 전지전능한 존재보다 견고한 조직을 만든다.*

세 가지 원칙이 이를 뒷받침합니다:

- **캡슐화** — 내부 사고와 기억은 밖에서 보이지 않아요. 다른 존재와는 오직 텍스트 대화로만 이어집니다. 실제 조직과 다르지 않죠.
- **서재형 기억** — 컨텍스트 윈도우에 모든 것을 밀어넣지 않습니다. 필요한 순간 책장에서 책을 꺼내듯 스스로 기억을 검색해 꺼내옵니다.
- **자율성** — 지시를 기다리지 않습니다. 자신만의 리듬으로 움직이고, 자신의 가치관으로 판단합니다.

---

<details>
<summary><strong>API 키 레퍼런스</strong></summary>

#### LLM 프로바이더

| 키 | 서비스 | 모드 | 발급처 |
|----|--------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**Azure OpenAI**, **Vertex AI (Gemini)**, **AWS Bedrock**, **vLLM**은 `config.json`의 `credentials` 섹션에서 설정합니다. 자세한 내용은 [기술 사양](docs/spec.md)을 참조하세요.

**Ollama** 등 로컬 모델은 API 키가 필요 없습니다. `OLLAMA_SERVERS` (기본값: `http://localhost:11434`)로 접속 대상을 지정합니다.

#### 이미지 생성 (선택 사항)

| 키 | 서비스 | 출력물 | 발급처 |
|----|--------|--------|--------|
| `NOVELAI_API_TOKEN` | NovelAI | 애니메이션 스타일 캐릭터 이미지 | [novelai.net](https://novelai.net/) |
| `FAL_KEY` | fal.ai (Flux) | 스타일라이즈드 / 포토리얼 | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) |
| `MESHY_API_KEY` | Meshy | 3D 캐릭터 모델 | [meshy.ai](https://www.meshy.ai/) |

#### 음성 채팅 (선택 사항)

| 요구사항 | 서비스 | 비고 |
|----------|--------|------|
| `pip install faster-whisper` | STT (Whisper) | 첫 사용 시 모델 자동 다운로드. GPU 권장 |
| VOICEVOX Engine 실행 | TTS (VOICEVOX) | 기본값: `http://localhost:50021` |
| AivisSpeech/SBV2 실행 | TTS (Style-BERT-VITS2) | 기본값: `http://localhost:5000` |
| `ELEVENLABS_API_KEY` | TTS (ElevenLabs) | 클라우드 API |

#### 외부 연동 (선택 사항)

| 키 | 서비스 | 발급처 |
|----|--------|--------|
| `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` | Slack | [설정 가이드](docs/slack-socket-mode-setup.md) |
| `CHATWORK_API_TOKEN` | Chatwork | [chatwork.com](https://www.chatwork.com/) |

</details>

<details>
<summary><strong>계층과 역할</strong></summary>

`supervisor` 필드 하나로 상하 관계를 정의합니다. 미설정 시 최상위 레벨로 동작합니다.

역할 템플릿을 사용하면 직책에 맞는 전문 프롬프트, 권한, 모델이 자동으로 적용됩니다:

| 역할 | 기본 모델 | 용도 |
|------|-----------|------|
| `engineer` | Claude Opus 4.6 | 복잡한 추론, 코드 생성 |
| `manager` | Claude Opus 4.6 | 조정, 의사결정 |
| `writer` | Claude Sonnet 4.6 | 콘텐츠 작성 |
| `researcher` | Claude Sonnet 4.6 | 정보 수집 |
| `ops` | vLLM (GLM-4.7-flash) | 로그 모니터링, 정형 업무 |
| `general` | Claude Sonnet 4.6 | 범용 |

매니저에게는 **수퍼바이저 도구**가 자동으로 부여됩니다: 태스크 위임, 진행 추적, 부하 재시작/비활성화, 조직 대시보드, 부하 상태 조회 — 실제 관리자가 하는 일과 동일합니다.

각 Anima는 ProcessSupervisor가 독립 프로세스로 기동하고, Unix Domain Socket을 통해 통신합니다.

</details>

<details>
<summary><strong>보안</strong></summary>

자율적으로 동작하는 에이전트에게 도구를 부여하는 이상, 보안은 진지하게 다뤄야 합니다. 실제로 프로덕션에서 사용하고 있어서 타협의 여지가 없습니다. AnimaWorks는 10개 계층의 심층 방어를 구현하고 있습니다:

| 계층 | 내용 |
|------|------|
| **신뢰 경계 라벨링** | 외부 데이터(웹 검색, Slack, 이메일)는 모두 `untrusted` 태그 부착 — 신뢰할 수 없는 소스의 지시를 따르지 않도록 모델에 명시 |
| **5계층 명령어 보안** | 셸 인젝션 탐지 → 하드코딩 차단 목록 → 에이전트별 금지 명령어 → 에이전트별 허용 목록 → 경로 탐색 검사 |
| **파일 샌드박싱** | 각 에이전트는 자체 디렉토리에 격리. `permissions.json`나 `identity.md`는 에이전트 자신이 수정 불가 |
| **프로세스 격리** | 에이전트당 독립 OS 프로세스. Unix Domain Socket 통신 (TCP 미사용) |
| **3계층 속도 제한** | 세션 내 중복 제거 → 역할별 발신 한도 → 최근 발신 이력의 프롬프트 주입을 통한 자기 인식 |
| **연쇄 방지** | 깊이 제한 + 연쇄 탐지. 5분 쿨다운 및 지연 처리 |
| **인증 · 세션 관리** | Argon2id 해싱, 48바이트 랜덤 토큰, 최대 10세션 |
| **Webhook 검증** | Slack (리플레이 방지 포함 HMAC-SHA256) 및 Chatwork 서명 검증 |
| **SSRF 완화** | 미디어 프록시가 사설 IP 차단, HTTPS 강제, Content-Type 검증, DNS 해석 검사 |
| **아웃바운드 라우팅** | 알 수 없는 수신자는 fail-closed. 명시적 설정 없이 임의의 외부 발신 불가 |

상세 내용: **[보안 아키텍처](docs/security.md)**

</details>

<details>
<summary><strong>CLI 레퍼런스 (고급 사용자용)</strong></summary>

CLI는 파워 유저와 자동화를 위한 기능입니다. 일상적인 사용은 Web UI에서 하시면 됩니다.

### 서버

| 명령어 | 설명 |
|--------|------|
| `animaworks start [--host HOST] [--port PORT] [-f]` | 서버 시작 (`-f`로 포그라운드 실행) |
| `animaworks stop [--force]` | 서버 중지 |
| `animaworks restart [--host HOST] [--port PORT]` | 서버 재시작 |

### 초기화

| 명령어 | 설명 |
|--------|------|
| `animaworks init` | 런타임 디렉토리 초기화 (비대화형) |
| `animaworks init --force` | 템플릿 차분 병합 (데이터 보존) |
| `animaworks reset [--restart]` | 런타임 디렉토리 초기화 |

### Anima 관리

| 명령어 | 설명 |
|--------|------|
| `animaworks anima create [--from-md PATH] [--template NAME] [--role ROLE] [--supervisor NAME] [--name NAME]` | 신규 생성 |
| `animaworks anima list [--local]` | 전체 Anima 목록 |
| `animaworks anima info ANIMA [--json]` | 상세 설정 |
| `animaworks anima status [ANIMA]` | 프로세스 상태 표시 |
| `animaworks anima restart ANIMA` | 프로세스 재시작 |
| `animaworks anima disable ANIMA` / `enable ANIMA` | 비활성화 / 활성화 |
| `animaworks anima set-model ANIMA MODEL` | 모델 변경 |
| `animaworks anima set-background-model ANIMA MODEL` | 백그라운드 모델 설정 |
| `animaworks anima reload ANIMA [--all]` | status.json에서 핫리로드 |

### 커뮤니케이션

| 명령어 | 설명 |
|--------|------|
| `animaworks chat ANIMA "메시지" [--from NAME]` | 메시지 전송 |
| `animaworks send FROM TO "메시지"` | Anima 간 메시지 |
| `animaworks heartbeat ANIMA` | 하트비트 수동 트리거 |

### 설정 · 유지보수

| 명령어 | 설명 |
|--------|------|
| `animaworks config list [--section SECTION]` | 설정 목록 |
| `animaworks config get KEY` / `set KEY VALUE` | 값 조회 / 설정 |
| `animaworks status` | 시스템 상태 |
| `animaworks logs [ANIMA] [--lines N] [--all]` | 로그 표시 |
| `animaworks index [--reindex] [--anima NAME]` | RAG 인덱스 관리 |
| `animaworks models list` / `models info MODEL` | 모델 목록 / 상세 |

</details>

<details>
<summary><strong>기술 스택</strong></summary>

| 구성 요소 | 기술 |
|-----------|------|
| 에이전트 실행 | Claude Agent SDK / Codex SDK / Anthropic SDK / LiteLLM |
| LLM 프로바이더 | Anthropic, OpenAI, Google, Azure, Vertex AI, AWS Bedrock, Ollama, vLLM |
| 웹 프레임워크 | FastAPI + Uvicorn |
| 태스크 스케줄링 | APScheduler |
| 설정 관리 | Pydantic 2.0+ / JSON / Markdown |
| 기억 기반 / RAG | ChromaDB + sentence-transformers + NetworkX |
| 음성 채팅 | faster-whisper (STT) + VOICEVOX / SBV2 / ElevenLabs (TTS) |
| 사용자 알림 | Slack, Chatwork, LINE, Telegram, ntfy |
| 외부 메시징 | Slack Socket Mode, Chatwork Webhook |
| 이미지 생성 | NovelAI, fal.ai (Flux), Meshy (3D) |

</details>

<details>
<summary><strong>프로젝트 구조</strong></summary>

```
animaworks/
├── main.py              # CLI 엔트리포인트
├── core/                # Digital Anima 코어 엔진
│   ├── anima.py, agent.py, lifecycle.py  # 코어 엔티티 · 오케스트레이터
│   ├── memory/          # 기억 서브시스템 (priming, consolidation, forgetting, RAG)
│   ├── execution/       # 실행 엔진 (S/C/A/B)
│   ├── tooling/         # 도구 디스패치 · 권한 검사
│   ├── prompt/          # 시스템 프롬프트 빌더 (6그룹 구조)
│   ├── supervisor/      # 프로세스 감시
│   ├── voice/           # 음성 채팅 (STT + TTS)
│   ├── config/          # 설정 관리 (Pydantic 모델)
│   ├── notification/    # 사용자 알림 채널
│   └── tools/           # 외부 도구 구현
├── cli/                 # CLI 패키지
├── server/              # FastAPI 서버 + Web UI
└── templates/           # 초기화 템플릿 (ja / en)
```

</details>

---

## 문서

**[전체 문서 인덱스](docs/README.md)** — 읽기 가이드, 아키텍처 심층 분석, 설계 사양.

| 문서 | 설명 |
|------|------|
| [비전](docs/vision.md) | 핵심 철학: 불완전한 개인의 협업이 단일 전지전능 모델을 이긴다 |
| [기능 목록](docs/features.md) | AnimaWorks로 할 수 있는 모든 것 |
| [기억 시스템](docs/memory.md) | 일화 기억, 의미 기억, 절차 기억; 프라이밍; 능동적 망각 |
| [보안](docs/security.md) | 심층 방어 모델, 출처 추적, 적대적 위협 분석 |
| [뇌과학 매핑](docs/brain-mapping.md) | 각 모듈과 인간 뇌 영역의 대응 관계 |
| [기술 사양](docs/spec.md) | 실행 모드, 프롬프트 구성, 설정 해석 |

## 라이선스

Apache License 2.0. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
