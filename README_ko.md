# AnimaWorks — Organization-as-Code

### 메신저(LINE·카톡)처럼 말만 걸면, 뒤에서 24시간 소프트웨어를 출하하는 AI 조직.

AnimaWorks는 장기 기억·인격·직책·스케줄을 가진 영속적인 AI 에이전트(**Digital Anima**)들을 하나로 묶어, **'살아있는 조직'**으로서 24시간 자율 가동시키는 오픈소스 프레임워크입니다.

인간이 마주하는 화면은 친숙한 메신저 앱(LINE)과 같은 쾌적한 대화방입니다.
하지만 그 이면에서는 독립된 OS 프로세스에서 움직이는 에이전트들이 목표를 분해하고, 병렬 `git worktree` 샌드박스에서 코드를 작성하며, 서로 테스트와 멀티모델 코드 리뷰를 돌리고, Pull Request를 생성해 CI 실패를 스스로 복구하며, 배포 후 로그 관측까지 인간의 손을 빌리지 않고 완수합니다.

```text
인간의 한마디 / GitHub 이벤트 / 정기 스케줄 (Heartbeat · Cron)
   ↓
리더 Anima가 작업 분해 및 위임 (SQLite TaskStore · 배타 키 · 위임 계약)
   ↓
엔지니어 Anima가 격리된 worktree에서 구현 → 테스트 → 멀티모델 PR 리뷰 → CI 자가 복구 → 배포 관측
   ↓
판단이 필요한 예외나 승인 게이트만 LINE풍 채팅·음성 통화·알림으로 인간에게 에스컬레이션
```

AnimaWorks는 고정된 빌드 파이프라인을 하드코딩하지 않습니다. 프레임워크가 제공하는 것은 **뇌과학 기반 장기 기억·프로세스 격리·원자적 태스크 보드·권한 및 개인정보 격리(Enclave)·다채널 통신망**이라는 '조직의 인프라'입니다. 그렇기 때문에 동일한 조직이 코드 출하뿐만 아니라 경쟁사 조사, 고객 응대, 회의록 작성, Slack이나 전화 연락까지 자율적으로 수행합니다. 워크플로우 엔진이 아니라 **조직 그 자체**이기 때문입니다.

**[English README](README.md)** | **[日本語 README](README_ja.md)** | **[简体中文 README](README_zh.md)**

---

## 프로덕션 환경 가동 실적

장난감 데모가 아닙니다. 지난 6개월 동안 8명의 Anima로 구성된 AnimaWorks 조직이 실제 상용 SaaS 프로덕트의 일상적인 개발·운영·유지보수를 쉬지 않고 돌려왔습니다.

| 지표 (2026년 3월~8월 · 1차 실행 로그 집계) | 실적치 |
|---|---|
| 에이전트가 스스로 생성한 Pull Request | **302건** (그중 **267건** 머지) |
| 에이전트가 운영한 Pull Request (리뷰·CI 복구·충돌 해결) | **752건** (그중 **721건** 머지) |
| 조직이 자발적으로 기안·실행한 태스크 비율 | **99.7%** (전체 31,215건 중 인간 기점은 단 92건) |
| GitHub 이벤트에서 자동 생성된 태스크 수 (8월 단일 월) | **2,508건** |

<p align="center">
  <img src="docs/images/chat-memory.png" alt="AnimaWorks LINE풍 채팅 화면과 상주 버스트업" width="760">
  <br><em>새로워진 LINE풍 채팅 UI: 왼쪽 Anima 레일에서 멤버 가동 상태를 한눈에 확인하고, 초록/흰색 말풍선과 인라인 답장 인용·승인 알림 카드·첨부 파일 카드로 소통하며, 오른쪽 아래 상주 버스트업이 실시간으로 표정을 바꿉니다.</em>
</p>

<p align="center">
  <img src="docs/images/workspace-dashboard.gif" alt="AnimaWorks Workspace — 실시간 조직 트리와 액티비티 피드" width="760">
  <br><em>3D Workspace: 조직 트리, 각 Anima의 상태, 최근 액션을 실시간으로 시각화.</em>
</p>

<p align="center">
  <img src="docs/images/pixel-workspace.gif" alt="AnimaWorks 픽셀 오피스 — 가동 중인 조직의 라이브 뷰" width="760">
  <br><em>픽셀 오피스(Pixel Workspace)는 단순한 연출이 아닙니다. 말풍선 하나하나가 지금 이 순간 뒤에서 돌아가고 있는 실제 태스크입니다.</em>
</p>

---

## :rocket: 60초 만에 체험하기 (데모 환경)

**Claude Code CLI가 설치되어 있거나 Codex에 로그인되어 있다면 API 키조차 필요 없습니다.**

원라이너로 클론 및 셋업하고, 3일간의 활동 이력과 3인 팀(PM `kaito`·리드 엔지니어 `sora`·어시스턴트 `hina`)이 포함된 데모 조직을 즉시 가동할 수 있습니다. 실제 `~/.animaworks`와는 완전히 분리된 전용 디렉토리(`~/.animaworks-demo`)에서 작동하므로 환경을 어지럽히지 않습니다.

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks demo --preset ja-business
```

**http://localhost:18501**을 열기만 하면 이미 일을 진행하고 있는 팀과 LINE풍 화면에서 대화를 시작할 수 있습니다 ([데모 상세 안내 →](demo/README.md)).

> **선택 가능한 4가지 프리셋**: `en-business` (영어·리얼풍, 기본값) / `en-anime` (영어·애니풍) / `ja-business` (일본어·리얼풍) / `ja-anime` (일본어·애니풍). 프리셋을 전환할 때는 `--reset`을 붙여 다시 실행하세요 (예: `uv run animaworks demo --reset --preset ja-anime`).

---

## 퀵스타트 (나만의 조직 만들기)

### macOS / Linux / WSL

```bash
curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash
cd animaworks
uv run animaworks start
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.ps1 | iex
cd animaworks
uv run animaworks start
```

> **Windows 참고**: AnimaWorks는 Unix 전용 기능(`fcntl`, `pty`, `os.killpg` 등)을 사용하므로 네이티브 Windows 직접 실행은 지원하지 않습니다. PowerShell에서 `setup.ps1`을 실행하면 WSL 도입부터 Linux 환경 내 AnimaWorks 셋업까지 자동으로 완료됩니다.

브라우저에서 **http://localhost:18500**을 열고 셋업 위저드를 따라 진행하세요:
1. **언어 및 테마 선택** (`ko` / `ja` / `en` / `zh`)
2. **LLM 연결 설정** (Claude Code 구독 로그인, Codex 로그인, 각종 API 키, 로컬 Ollama 지원)
3. **첫 번째 Anima 생성** (`dev-lead`, `full-stack`, `assistant`, `marketer`, `translator` 등 템플릿 선택 또는 커스텀 생성)

브라우저 없이 CLI만으로 초기화부터 대화·운영까지 완결할 수도 있습니다:

```bash
uv run animaworks init
uv run animaworks start
uv run animaworks chat alice "오늘의 스프린트 목표를 확인하고 팀원들에게 태스크를 배분해 줘"
```

<details>
<summary><strong>Docker Compose / pip 수동 설치 절차</strong></summary>

#### Docker (호스트 환경을 더럽히지 않는 권장 구성)

`docker compose up`으로 Node.js, GitHub CLI (`gh`), Playwright, Bubblewrap, Claude Code / Codex / Gemini CLI가 모두 갖춰진 격리 컨테이너가 기동됩니다.

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
# .env에 인증 정보 설정 (ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN / GH_TOKEN 등)
docker compose up -d --build
```

브라우저 위저드 없이 헤드리스로 초기화하는 예시:

```bash
docker exec -it <container> animaworks init --skip-anima
docker exec -it <container> animaworks anima create --name alice --template dev-lead
docker exec -it <container> animaworks config set setup_complete true
docker exec -it <container> animaworks send <your-name> alice "안녕하세요"
```

#### pip로 수동 설치하는 경우 (Python 3.12 / 3.13 권장)

```bash
git clone https://github.com/xuiltul/animaworks.git && cd animaworks
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[all-tools]"
animaworks start
```

</details>

---

## 앞면의 얼굴: 인간에게는 'LINE 감각으로 대화할 수 있는 팀'

많은 자율 에이전트 프레임워크가 인간에게 대량의 로그 감시나 복잡한 노드 편집 화면을 강요합니다. AnimaWorks의 채팅 화면은 정반대입니다. **평소 연락은 메신저처럼 가볍게, 깊이 파고들고 싶을 때만 원탭으로 기억과 뒷작업 내역을 열어보는** 설계입니다.

### 1. LINE풍 토크 화면·왼쪽 Anima 레일·오른쪽 아래 상주 버스트업
- **직관적인 말풍선 토크**: 인간의 발언은 오른쪽 초록색 말풍선, Anima의 응답은 왼쪽 흰색 말풍선으로 표시되며 스트리밍 중에도 사고 프리뷰와 도구 실행 현황이 매끄럽게 애니메이션됩니다.
- **왼쪽 세로 레일 (Anima Rail)**: 화면 왼쪽 끝에 멤버 얼굴 아이콘이 나란히 배치되어 누가 작업 중(파란 링)이고 누가 당신의 답장을 기다리는지(호박색 답장 대기 배지) 한눈에 알 수 있습니다. 자주 대화하는 멤버만 남기고 `×`로 접거나 `+`로 다시 열 수 있으며, 최대 4페인 화면 분할이나 최대 5명의 **미팅 모드(사회자가 있는 그룹 토론)**로도 원클릭 전환됩니다.
- **오른쪽 아래 버스트업 상주 (Presence Panel)**: 토크 화면 오른쪽 아래에는 대화 중인 Anima의 버스트업이 상주합니다. 응답 감정(`smile` / `thinking` / `troubled` / `surprised` 등)에 맞춰 실시간으로 표정을 바꾸며, 작업 중에는 '무엇을 하고 있는지' 작은 말풍선으로 중얼거립니다.

### 2. 인간 중심 피드 정리와 '원탭 승인·인용 답장'
- **노이즈 없는 대화 피드**: 24시간 움직이는 조직에서는 뒤에서 수백 번의 Heartbeat 순찰이나 Anima 간의 DM, 정기 Cron이 돌아갑니다. 새 채팅 UI는 이를 **인간과의 대화를 방해하지 않는 1줄 접기 번들**로 자동 집약합니다 (헤더의 `뒷작업` 토글로 언제든 전체 펼치기 가능).
- **알림 카드와 인용 답장**: Anima가 권한 밖의 판단을 구할 때(`call_human`)는 채팅 내에 **승인 / 반려 / 코멘트** 버튼이 달린 알림 카드가 도착합니다. 과거 알림이나 메시지에 인용 답장(`Reply Chip`)을 달아 회신하면 Anima는 어떤 안건에 대한 지시인지 정확히 인식하고 작업을 재개합니다.

### 3. 양방향 파일 첨부·인라인 뷰어·Slack 미러링
- **모든 파일을 드래그 앤 드롭**: 이미지뿐만 아니라 PDF·Markdown·CSV·코드·Office 문서·오디오·비디오를 그대로 채팅에 첨부 가능.
- **Anima가 보내는 성과물 파일 첨부 (`attach_file`)**: Anima 측도 작성한 리포트나 집계 CSV, 생성 이미지를 답장 말풍선에 직접 첨부해 전달합니다. 첨부 카드를 클릭하면 브라우저 내장 **풀스크린 뷰어**(Markdown 안전 렌더링, 코드 하이라이팅, 오디오·비디오 플레이어)로 즉시 내용을 확인할 수 있습니다.
- **Slack 채팅 미러링**: 외출 중 Slack에서 Anima와 주고받은 대화도 웹 채팅 타임라인에 자동으로 통합 표시됩니다.

### 4. 얼굴을 탭해 여는 '기억 팝오버'와 풀스크린 음성 통화
- **D3 엔티티 상관 버블 차트**: 채팅 헤더의 `기억` 버튼이나 얼굴 아이콘을 클릭하면 해당 Anima의 머릿속이 말풍선 팝오버로 열립니다. 대화에서 추출된 인물·서비스·토픽의 연결고리를 시각화하는 **상관 버블 차트(`연결`)**를 비롯해 현재 작업 상태(`지금 상태`), 축적한 지식·절차서(`기억 파일`), 일별 에피소드(`일기`), 도구 실행 이력(`액티비티`), Heartbeat/Cron 설정(`정기 작업`)을 그 자리에서 탐색할 수 있습니다.
- **풀스크린 음성 통화 (`Call Screen`)**: 전화 아이콘을 누르면 저지연 프론트 레인 모델 + VOICEVOX / Style-BERT-VITS2 / ElevenLabs / Irodori TTS + `faster-whisper` STT 기반 음성 통화 화면으로 전환됩니다. 립싱크·눈 깜빡임 버스트업과 실시간 자막·트랜스크립트가 함께 제공되어 핸즈프리 벽치기가 가능합니다. 나아가 긴급 장애 시에는 Twilio 연동(`core/phone/`)을 통해 **Anima가 인간의 스마트폰으로 실제 전화를 걸어 깨우는** 것조차 가능합니다.

<p align="center">
  <img src="docs/images/memory-popover.png" alt="AnimaWorks 기억 팝오버 — D3 엔티티 상관 버블 차트" width="760">
  <br>
  <em>채팅 헤더의 '기억' 버튼이나 얼굴 아이콘에서 원탭으로 열리는 기억 팝오버. 사실·지식·절차·일기와 인물·서비스·화제의 상관 버블 차트를 그 자리에서 확인할 수 있습니다.</em>
</p>

<p align="center">
  <img src="docs/images/dashboard.png" alt="AnimaWorks 대시보드 — 실시간 조직도와 태스크 현황" width="760">
  <br>
  <em>Dashboard(홈): 조직도, 대응 필요 아이템, 모델 쿼터 소비율, 프로세스 헬스, 최근 액티비티를 한 화면에서 총괄.</em>
</p>

---

## 뒷면의 얼굴: 24시간 소프트웨어와 사업을 돌리는 자율 조직 엔진

친근한 채팅 화면의 이면에서는 24시간 365일 멈추지 않고 계속 움직이기 위한 견고한 분산 멀티프로세스 아키텍처가 돌아가고 있습니다.

```text
FastAPI Server (경량 컨트롤 플레인 / WebSocket & SSE / Enclave 클라이언트)
  ├── MemoryService Process (전용 ChromaDB + Embedding HTTP 오너 프로세스, SQLite WAL)
  ├── Anima Root Process: kaito (Manager)
  │     ├── Front Lane (저지연 즉시 응답 / 음성 통화 / 라우팅)
  │     ├── Main Chat / Inbox / Heartbeat / Cron Lanes
  │     └── Task Runner Subprocesses (IPC v2 — 무거운 태스크별 격리 서브프로세스 실행)
  ├── Anima Root Process: sora (Lead Engineer)
  └── Anima Root Process: hina (Assistant)
```

### 1. '어제의 이어서' 대화할 수 있다 — 다층 기억과 자율 유지보수
세션이 끝날 때마다 기억을 잃는 일회성 채팅 래퍼와 달리, 각 Anima는 파일시스템 우선·뇌과학 기반의 다층 기억 아키텍처를 갖습니다:
- **6채널 자동 프라이밍 (`core/memory/priming/`)**: 매 발화 직전, 동적 토큰 예산 내에서 (A) 발신자 프로필, (B) 최근 액티비티, (C) 관련 지식, (E) 보류 태스크, (F) 에피소드 기억, (G) 그래프/커뮤니티/최근 원자적 사실을 자동으로 회상해 프롬프트에 주입합니다.
- **하이브리드 검색 파이프라인 (`core/memory/retrieval/`)**: 벡터 검색(ChromaDB) + BM25 어휘 검색 + 원자적 사실(`FactRecord`) + 엔티티 레지스트리 + RRF 융합 + 신뢰도 게이팅을 결합합니다. 백엔드는 안정적인 `legacy`(기본값) 외에 **Neo4j 그래프 메모리(`neo4j`)**를 글로벌 또는 Anima별로 선택할 수 있습니다.
- **수면 중 기억 고정화와 능동적 망각 (`core/memory/maintenance/`)**: 매일 밤 하루의 활동 로그를 구조화된 지식으로 증류·통합(Consolidation)하고, 중요도와 시간 감쇠에 따라 오래된 저가치 기억을 정리(Active Forgetting)하되 핵심 정체성·절차·미완료 태스크는 엄격히 보호합니다.

### 2. 지시를 기다리지 않고 움직인다 — Heartbeat·Cron·SQLite TaskStore·GitHub 게이트웨이
- **Heartbeat & Cron**: 적응형 간격과 야간 조용한 시간(Quiet Hours)을 지원하는 자율 순찰(`HEARTBEAT.md`) 및 정각 스케줄 실행(`cron.md`).
- **SQLite TaskStore (`shared/taskboard.sqlite3`)**: 조직 전체의 태스크를 원자적으로 관리하는 보드입니다. 의존 관계, 동일 리소스의 충돌을 막는 **배타 키(`mutex_key`)**, 부하가 완료하면 자동으로 상사의 검수 태스크를 깨우는 **위임 계약(`delegated_tasks`)**, 중단된 워커의 자동 재시도·회수를 지원합니다.
- **GitHub 이벤트 게이트웨이**: CI 실패, PR 리뷰 코멘트, 머지 충돌을 Webhook으로 수신해 중복 제거 후 담당 엔지니어 Anima의 태스크로 자동 투입합니다.

### 3. 프로세스 격리로 멈추지 않는다 — Phase 3 프로세스 트리와 IPC v2
- **Anima별 Root 프로세스 + Task Runner 서브프로세스**: 각 Anima는 `ProcessSupervisor`가 감시하는 독립 OS 프로세스로 기동되며, 무거운 백그라운드 태스크나 위임 태스크는 IPC v2(`STDIN/STDOUT` JSON-RPC + 바이너리 프레이밍)로 연결된 전용 `TaskRunner` 서브프로세스를 띄워 격리 실행합니다. 특정 도구가 멈추거나 크래시해도 해당 서브프로세스만 회수될 뿐 서버나 다른 Anima는 절대 멈추지 않습니다.
- **전용 MemoryService 프로세스**: 벡터 스토어(ChromaDB)와 임베딩 모델을 단일 오너 프로세스로 집약(비오너 프로세스는 모두 `HttpVectorStore`를 통해 로컬 HTTP 경유 접근)하여 멀티프로세스 동시 쓰기로 인한 인덱스 파손과 메모리 중복 소비를 원천 차단합니다.

### 4. 기밀 정보와 PII를 에이전트로부터 격리한다 — Enclave & 다층 가드
- **Enclave (`core/enclave/`)**: 외부 서비스의 원본 인증 정보(API 키·쿠키)나 고객 개인정보(PII)를 에이전트 프로세스에 직접 넘기지 않고, 별도 OS 유저 권한으로 움직이는 **Enclave 데몬** 뒤에 격리합니다. Anima는 Unix 도메인 소켓 너머로 허용된 작업(`gmail_list`, `gmail_send`, `git_push`, `sql_query` 등)만 요청할 수 있어 임의의 Bash 명령을 실행하더라도 원본 비밀키를 훔칠 수 없으며, 응답 시에는 **Egress PII 마스킹**(이메일·전화번호·카드번호·커스텀 정규식 자동 가림)이 적용됩니다.
- **다층 권한 게이트 (`ToolHandler`)**: 글로벌 `permissions.global.json`, Anima별 `permissions.json`, 위험 명령어 패턴 차단, 프롬프트 인젝션 신뢰 경계(`trusted` vs `external`)로 폭주를 방지합니다.

### 5. 멀티 엔진 실행 & 하이브리드 라우팅
직무와 예산에 맞춰 Anima마다 최적의 실행 엔진과 모델을 조합할 수 있으며, Heartbeat/Cron 순찰에는 저비용 `background_model`을 별도 지정할 수 있습니다:

| 모드 | 실행 엔진 | 주요 대상 모델 | 특징 및 도구 연동 |
|---|---|---|---|
| **S** | Claude Agent SDK | `claude-*` (권장) | Claude Code 내장 도구(Read/Write/Edit/Bash/Grep/Glob) + **stdio MCP (`mcp__aw__*`)**로 내부 도구 직결. Claude Code 구독 로그인 지원 |
| **C** | Codex SDK / CLI | `codex/*`, `openai-codex/*` | Codex 샌드박스 실행 + AnimaWorks MCP 연동. `codex login` 지원 |
| **D** | Cursor Agent CLI | `cursor/*` | Cursor CLI 세션·MCP 통합 에이전트 루프 |
| **G** | Gemini CLI | `gemini/*` | Google Gemini CLI 스트림 JSON·도구 루프 연동 |
| **X** | Grok Build CLI | `grok/*` | ACP stdio 경유 Grok Build 에이전트 루프 |
| **A** | LiteLLM Autonomous | GPT, Gemini, DeepSeek, Bedrock, Vertex, Azure, Ollama, vLLM 등 | 모든 API 모델·로컬 LLM에서 CC 호환 도구 + 기억·태스크·스킬 생성 실행 |

<p align="center">
  <img src="docs/images/asset-management.png" alt="AnimaWorks 에셋 관리 — 캐릭터 에셋과 표정 차분" width="760">
  <br><em>에셋 관리: 성격 설정으로부터 전신 일러스트·7종 표정 버스트업·치비 캐릭터·도트 시트·3D 모델까지 일괄 생성. 상사의 화풍을 이어받는 Vibe Transfer도 지원.</em>
</p>

---

## 다른 프레임워크와의 차이

| 관점 | AnimaWorks | CrewAI | LangGraph | OpenClaw | OpenAI Agents SDK |
|---|---|---|---|---|---|
| **기본 단위와 설계 사상** | **영속적인 자율 에이전트의 '조직'** | 역할 기반 태스크 팀 | 상태 그래프 워크플로우 | 개인용 AI 비서 | 경량 멀티에이전트 SDK |
| **인간과의 UI 경험** | **LINE풍 토크 화면 + 상주 버스트업 + D3 기억 그래프 + 3D/도트 오피스 + 음성 통화** | CLI / 간이 로그 | LangSmith Studio (개발자용) | 채팅 UI | 없음 (코드만) |
| **장기 기억** | **뇌과학 기반**: 6ch 자동 회상·벡터+BM25+Facts+Neo4j·일일 통합·보호 포함 능동적 망각 | Cognitive Memory | Checkpointer + Store | SuperMemory | 세션 내 한정 |
| **자율 가동 트리거** | **Heartbeat + Cron + TaskStore + GitHub 이벤트 게이트웨이 (24/7 가동)** | 인간 트리거 중심 | 인간/API 트리거 | Cron + Heartbeat | 인간/API 트리거 |
| **프로세스·내장애성** | **Anima별 root 프로세스 + 분리 task runner (IPC v2)·자동 재시작·메모리 인덱스 자가 복구** | 단일 공유 프로세스 | 단일 공유 프로세스 | 단일 프로세스 | 단일 공유 프로세스 |
| **기밀·PII 보호** | **Enclave (별도 OS 유저 + Unix 소켓 + Egress PII 마스킹 + 다층 신뢰 경계)** | 없음 | 없음 | 없음 | Guardrails만 |
| **멀티모델 실행** | **7종 엔진 (Claude SDK / Codex / Cursor / Gemini CLI / Grok / LiteLLM 등) + 프로바이더 횡단 레이트 가드** | LiteLLM | LangChain 모델 | OpenAI 호환 | OpenAI 중심 |

---

## 왜 AnimaWorks인가

**혼자서는 아무것도 할 수 없다. 그래서, 조직을 만들었습니다.**

이 프로젝트는 3가지 커리어의 교차점에서 탄생했습니다.

**경영자로서** — 저는 '혼자서는 아무것도 할 수 없다'는 것을 알고 있습니다. 우수한 엔지니어도 필요하고, 커뮤니케이션이 능숙한 스태프도 있습니다. 묵묵히 일하는 워커도 있고, 때때로 날카로운 아이디어를 내주는 사람도 있습니다. 천재만으로는 조직이 돌아가지 않습니다. 다양한 힘을 합쳤을 때, 혼자서는 이루지 못했던 것을 이룰 수 있습니다.

**정신과 의사로서** — LLM의 내부 구조를 관찰했을 때, 인간의 뇌와 놀랍도록 유사한 구조가 있다는 것을 깨달았습니다. 회상, 학습, 망각, 고정화 — 뇌가 기억을 처리하는 메커니즘을 LLM의 기억 시스템으로 그대로 구현한다면, 인간의 뇌를 재현할 수 있을지도 모릅니다. 그렇다면 LLM을 '유사 인간'으로 취급할 수 있다면, 인간과 똑같이 조직을 만들 수 있을 것입니다.

**엔지니어로서** — 30년간 코드를 작성해 왔습니다. 로직을 짜는 즐거움, 자동화의 쾌감을 알고 있습니다. 이상을 모두 코드에 담아내면, 제 이상의 조직을 만들 수 있습니다.

우수한 '단독 AI 비서' 프레임워크는 이미 많이 있습니다. 하지만 코드로 인간에 가까운 단위를 만들고, 그것을 조직으로 기능시키는 프로젝트는 아직 적다고 느꼈습니다. AnimaWorks는 제 자신이 사업에 도입하고, 매일 사용하면서 키우고 있는 AI 조직입니다.

> *불완전한 개인의 협력이, 단일의 전지전능자보다 견고한 조직을 만든다.*

3가지 원칙이 이를 뒷받침합니다:

- **캡슐화** — 내부의 사고·기억은 외부에서 보이지 않습니다. 타인과는 텍스트 대화로만 연결됩니다. 현실의 조직과 같습니다.
- **RAG 기억(서고형)** — 윈도우에 모든 것을 담지 않습니다. Priming이 RAG로 관련 청크를 집어 올리고, 에이전트는 `search_memory` 등으로 스스로 떠올립니다.
- **자율성** — 지침을 기다리지 않습니다. 자신의 시계로 움직이고, 자신의 가치관으로 판단합니다.

---

<details>
<summary><strong>API 키·인증·외부 서비스 연동 레퍼런스</strong></summary>

#### LLM 프로바이더 및 CLI 로그인

| 인증 방법 / 키 | 대응 서비스 | 실행 모드 | 비고 |
|---|---|---|---|
| `claude` CLI 로그인 | Claude Code 구독 | S | API 키 불필요. 위저드 또는 Settings에서 선택 |
| `codex login` | OpenAI Codex CLI | C | API 키 불필요. ChatGPT Plus/Pro 등에서 이용 가능 |
| `ANTHROPIC_API_KEY` | Anthropic API | S / A | [console.anthropic.com](https://console.anthropic.com/) |
| `OPENAI_API_KEY` | OpenAI API | A / C | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | Google AI (Gemini) | A / G | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `grok login` | Grok Build CLI | X | ACP stdio 경유로 `grok/*` 모델 실행 |
| `cursor-agent` 로그인 | Cursor Agent CLI | D | `cursor/*` 모델 실행 |
| `OLLAMA_SERVERS` | Ollama / vLLM 등 로컬 LLM | A | API 키 불필요 (기본값: `http://localhost:11434`) |

- **Azure OpenAI / AWS Bedrock / Google Vertex AI**도 `config.json`의 `credentials`로 완벽히 지원합니다.
- 인증 정보는 `config.json` → 암호화 Vault(`animaworks vault`) → 공유 크레덴셜 → 환경 변수 우선순위로 해결됩니다.

#### 이미지·3D 생성 (옵션)

| 키 / 설정 | 서비스 | 용도 |
|---|---|---|
| `NOVELAI_TOKEN` | NovelAI | 애니풍 아바타·표정 차분·Vibe Transfer |
| `FAL_KEY` | fal.ai (Flux) | 리얼풍 / 스타일라이즈드 아바타 생성 |
| Codex 이미지 생성 / 로컬 Diffusers | OpenAI / Local | 전신·버스트업·도트 스프라이트(`pixel_sheet.png`) 생성 |
| `MESHY_API_KEY` | Meshy | 3D 캐릭터 모델 생성(`.glb`) |

#### 음성 통화·긴급 전화 발신 (옵션)

| 컴포넌트 | 대응 엔진 / 서비스 |
|---|---|
| **STT (음성 인식)** | `faster-whisper` (LocalAgreement-2 스트리밍 확정, `pip install animaworks[transcribe]`) |
| **TTS (음성 합성)** | VOICEVOX (`:50021`) / Style-BERT-VITS2·AivisSpeech (`:5000`) / Irodori (`:7861`) / ElevenLabs API |
| **긴급 전화 알림** | Twilio Voice (`core/phone/` — 중대 인시던트 발생 시 인간의 전화번호로 발신 및 음성 안내) |

#### 외부 메시징·SaaS 연동 (옵션)

| 서비스 | 기능 |
|---|---|
| **GitHub** | Webhook 게이트웨이 (CI 실패·PR 리뷰·충돌 자동 태스크화) + Anima별 identity로 `gh` 조작 |
| **Slack** | Socket Mode 양방향 대화, 공유 Board 동기화, **웹 채팅으로의 Slack 대화 미러링**, 인간 알림 |
| **Discord / Chatwork** | Gateway / Webhook 수신, 양방향 메시지, 인간 알림 |
| **Zoom RTMS** | 회의 실시간 음성·자막 수집 및 회의록·액션 아이템화 |
| **Google Workspace / Notion** | Gmail, Google Calendar, Google Sheets, Google Tasks, Notion 페이지 읽기/쓰기 |
| **알림 채널** | Slack / Discord / Chatwork / LINE / Telegram / ntfy / Twilio 전화 |

</details>

<details>
<summary><strong>조직 계층·롤 템플릿·복수 회사(Company) 관리</strong></summary>

`status.json`의 `supervisor` 필드 하나로 보고 라인(상사·부하)이 정의됩니다. 채팅에서 리더 Anima에게 "이런 스킬을 가진 멤버를 채용해 줘"라고 부탁하기만 하면, 리더 스스로 적절한 역할·성격·권한을 설계하여 새로운 Anima를 생성할 수도 있습니다.

| 롤 템플릿 | 주요 용도 | 자동 부여되는 능력 |
|---|---|---|
| `manager` | 의사결정, 태스크 분해·위임, 진척 감사 | 슈퍼바이저 도구 (`delegate_task`, `org_dashboard`, 부하 상태 확인·재시작·채용) |
| `engineer` | 코드 구현, 테스트 작성, CI 복구, PR 생성 | 격리 worktree·스크래치 환경 개발 규약, Git/GitHub 연동 |
| `researcher` | 기술 조사, 경쟁사 분석, 문서 정밀 조사 | 웹 검색·지식 구조화·리포트 작성 |
| `writer` | 콘텐츠 제작, 문서·릴리스 노트 집필 | 문서 품질 관리·파일 첨부 출력 |
| `ops` | 로그 감시, 정형 순찰, 인프라 헬스 체크 | 경량 모델(로컬 LLM 등)을 통한 저비용 상시 감시 |
| `general` | 어시스턴트, 스케줄 조정, 창구 업무 | 팀 횡단 연락 허브·회의록 정리 |

나아가 `animaworks company` 명령어군을 통해 단일 런타임 내에서 **복수의 회사(Company)나 사업부**를 정의하고, 채널 스코프 분리나 회사 분할(`split`)·내보내기(`export`)까지 수행할 수 있습니다.

</details>

<details>
<summary><strong>다층 보안과 신뢰 경계</strong></summary>

자율 에이전트에게 셸과 외부 도구를 넘겨주는 만큼, AnimaWorks는 '성선설'에 의존하지 않는 다층 방어를 구현하고 있습니다 (상세: [보안](docs/ko/security.md) / [Enclave](docs/ko/enclave.md)).

1. **Enclave (PII·프로덕션 데이터 격리 런타임)**: 개인정보나 프로덕션 DB/S3를 별도 OS 유저의 격리 공간에 가두고, Unix 도메인 소켓(`SO_PEERCRED`)과 Egress PII 마스킹 파이프라인을 거친 안전한 사실만을 본체 조직에 반환합니다.
2. **신뢰 경계 라벨링 (Provenance & Trust)**: 웹 검색·Slack·이메일 등 외부 입력에는 출처(`untrusted`) 태그가 부여되며 세션 중 최소 신뢰도가 전파됩니다. 외부 데이터 유래 기억도 RAG 메타데이터까지 출처가 추적되어 프롬프트 인젝션에 의한 권한 상승을 방지합니다.
3. **5단계 명령어·경로 검사**: 셸 인젝션 검출 → 글로벌 금지 리스트(`permissions.global.json`이 없으면 서버 기동 거부) → Anima별 금지 명령어 → Anima별 허용 리스트 → CWD 상대 경로를 포함한 경로 트래버설 차단·심볼릭 링크 월경 방지.
4. **파일 샌드박스와 게이트 도구**: 각 Anima는 허용된 `file_roots` 외에는 쓸 수 없으며, 자신의 `IDENTITY.md` / `SOUL.md`나 `permissions.json`도 자기 개조로부터 보호됩니다. 부작용이 큰 외부 송신 도구에는 사전 액션 규칙 대조(Action Gate)를 적용할 수 있습니다.
5. **인증·SSRF 방지·Webhook 검증**: Argon2id 패스워드 해시, 48바이트 세션 토큰, Slack/Chatwork/Zoom/GitHub의 HMAC 서명 검증(리플레이 방지 포함), 프라이빗 IP와 DNS 리바인딩을 차단하는 미디어 프록시.

</details>

<details>
<summary><strong>CLI 명령어 레퍼런스</strong></summary>

일상적인 대화나 감시는 모두 웹 UI에서 완결되지만, 자동화 및 운영 관리를 위한 강력한 CLI(`animaworks` 및 Anima 내부용 `animaworks-tool`)를 갖추고 있습니다 (전체 상세는 [CLI 레퍼런스](docs/ko/reference/cli.md)).

| 카테고리 | 주요 명령어 | 설명 |
|---|---|---|
| **서버 / 데모** | `animaworks start [-f] [--port 18500]`<br>`animaworks stop` / `restart`<br>`animaworks demo [--preset ja-business] [--reset]` | 서버 기동·정지·재시작<br>3일치 이력이 포함된 데모 팀 기동 (기본 포트 18501) |
| **Anima 관리** | `animaworks anima create / list / info / status`<br>`animaworks anima set-model / set-background-model / set-role`<br>`animaworks anima restart / reload / enable / disable`<br>`animaworks anima audit / permissions / repair-bootstrap` | Anima 생성·상태 확인·모델 변경·핫 리로드·권한/감사 진단 |
| **태스크 관리** | `animaworks task board / list / show / add`<br>`animaworks task claim / release / done / cancel / note`<br>`animaworks task autonomy [--days N]` | SQLite TaskStore 조작·리스 관리·**조직 자율 수행률 리포트** |
| **대화·연락** | `animaworks chat ANIMA "메시지"`<br>`animaworks send FROM TO "메시지"`<br>`animaworks board read / post / dm-history`<br>`animaworks heartbeat ANIMA` | CLI에서의 채팅·Anima 간 DM·공유 Board 조작·Heartbeat 수동 발화 |
| **기억·RAG·스킬** | `animaworks memory status / migrate / backup / cleanup`<br>`animaworks index [--anima NAME] [--full]`<br>`animaworks repair-rag --anima NAME --full`<br>`animaworks skills list / install / inspect / remove` | 기억 백엔드(`legacy` / `neo4j`) 관리·RAG 인덱스 재구축·Skill Hub 조작 |
| **Enclave·Vault·회사** | `animaworks enclave doctor / status / audit`<br>`animaworks vault init / store / get / list`<br>`animaworks company create / list / assign / split / export`<br>`animaworks cost [--today]` / `models list` | Enclave 격리 환경 점검·암호화 크레덴셜 관리·복수 회사 관리·LLM 비용 집계 |

</details>

<details>
<summary><strong>기술 스택 및 프로젝트 구성</strong></summary>

```text
animaworks/
├── main.py                  # CLI 엔트리포인트
├── core/                    # Digital Anima 코어 프레임워크
│   ├── anima/               # DigitalAnima 퍼사드·라이프사이클·Heartbeat·Inbox·첨부파일 관리
│   ├── agent/               # AgentCore·대화 사이클·Executor 팩토리·사전 컨텍스트
│   ├── runtime/             # Anima root process·task runner supervisor·IPC / IPC v2·스트리밍
│   ├── execution/           # 7종 실행 엔진 (claude, codex, cursor, gemini, grok, litellm) + watchdog·세션
│   ├── memory/              # 장기 기억 (priming, retrieval, rag, facts/connections, consolidation, activity)
│   ├── tasks/               # SQLite TaskStore (board, attempts, leases, wakeups, delegation)·외부 태스크 수집
│   ├── enclave/             # 기밀·PII 격리 런타임 (gateway, egress masker, raw_store, SSM tunnel, guards)
│   ├── tooling/             # ToolHandler·권한 정책·액션 게이트·MCP/외부 도구 디스패치
│   ├── integrations/        # 자동 검출되는 외부 연동 도구 (GitHub, Slack, Discord, Notion, Google, 이미지 생성 등)
│   ├── mcp/                 # Mode S / 외부 클라이언트용 stdio MCP 서버 (mcp__aw__*)
│   ├── skills/              # Skill Hub·라우터·큐레이터·절차 기억으로부터의 스킬 승격
│   ├── prompt/              # 시스템 프롬프트·조직 컨텍스트·도구 가이드 조립
│   ├── messaging/           # Anima 간 DM·공유 채널·외부 송신 라우팅
│   ├── notification/        # 인간 알림 (call_human)·인라인 승인·답장 라우팅
│   ├── voice/               # 음성 대화 (faster-whisper STT + 멀티 엔진 TTS + 저지연 프론트 레인)
│   ├── phone/               # Twilio 긴급 음성 통화 알림
│   ├── org/                 # 복수 회사(Company)·조직 계층·워크스페이스 관리
│   ├── config/              # Pydantic 설정 스키마·모델 해결·글로벌 권한 캐시
│   ├── auth/                # Argon2id 인증·세션 관리
│   └── i18n/                # 다국어 메시지 카탈로그 (t())
├── server/                  # FastAPI 서버 + WebSocket/SSE + 게이트웨이 + 웹 UI
│   ├── app.py               # ASGI 앱·lifespan·Readiness 게이트·베이스 패스 제어
│   ├── supervisor/          # ProcessSupervisor·헬스 체크·지수 백오프 재시작·자동 업데이트
│   ├── gateways/            # GitHub Webhook, Slack Socket Mode, Discord Gateway, Zoom RTMS
│   ├── routes/              # REST / WebSocket / SSE 라우트 (chat, sessions, memory, assets, attachment_preview 등)
│   └── static/              # 웹 UI (LINE풍 Chat·왼쪽 Rail·Presence·D3 기억 팝오버·통화 화면·3D/도트 Workspace)
├── cli/                     # animaworks / animaworks-tool CLI 구현·데모 러너·TUI
├── templates/               # 로케일별 (ja / en / ko) 초기 프롬프트·롤별 작업 규약·systemd 템플릿
└── demo/                    # 4종 프리셋 (ja/en × business/anime) 및 동봉된 3일치 액티비티 이력
```

</details>

---

## 문서 (Documentation)

**[문서 종합 인덱스 (`docs/ko/README.md`)](docs/ko/README.md)** ([English](docs/en/README.md) / [日本語](docs/ja/README.md))에서 설계 사상·아키텍처 상세·운영 가이드·자동 생성 레퍼런스를 확인할 수 있습니다:

| 문서 | 내용 |
|---|---|
| **[설계 이념 (Vision)](docs/ko/vision.md)** | '불완전한 개인의 협력'이라는 근본 사상과 조직 설계 원칙 |
| **[기능 개관 (Overview)](docs/ko/overview.md)** | AnimaWorks의 전체 기능 맵과 각 상세 장으로의 가이드 |
| **[뇌과학과의 대응 (Brain Mapping)](docs/ko/brain-mapping.md)** | 해마·편도체·전두전야 등 인간의 뇌 메커니즘과 각 모듈의 대응 관계 |
| **[아키텍처 전체](docs/ko/architecture/index.md)** | 프로세스 구성(Phase 3), 실행 모드, 라이프사이클, 프롬프트, 태스크 관리 |
| **[기억 시스템](docs/ko/memory/index.md)** | 자동 회상(Priming), 하이브리드 검색, 일일·주간 기억 통합과 능동적 망각 |
| **[보안](docs/ko/security.md)** | 신뢰 경계, 명령어 보호, 파일 샌드박스, 인증 및 감사 |
| **[Enclave (격리 런타임)](docs/ko/enclave.md)** | 개인정보·프로덕션 데이터를 분리하는 Enclave 구축·SSM 터널·Egress 마스킹 운영 |
| **[각종 레퍼런스](docs/ko/reference/cli.md)** | [CLI](docs/ko/reference/cli.md) / [도구 CLI](docs/ko/reference/tool-cli.md) / [HTTP API](docs/ko/reference/api.md) / [설정 항목](docs/ko/reference/config.md) / [모듈 목록](docs/ko/reference/modules.md) |

---

## 라이선스 (License)

Apache License 2.0 — 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

