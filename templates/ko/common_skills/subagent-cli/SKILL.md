---
name: subagent-cli
description: >-
  외부 AI 에이전트 CLI를 Bash로 비대화 실행하는 스킬. codex exec·cursor-agent로 코딩 작업을 위임한다.
  Use when: 복잡한 구현 위임, 코드 리뷰, 다중 파일 변경, Bash에서 서브에이전트 시작이 필요할 때.
---

# subagent-cli

외부 AI 에이전트 CLI를 Bash를 통해 서브프로세스로 실행하여 복잡한 코딩 태스크를 위임합니다.
자신의 아이덴티티, 판단, 기억을 유지하면서 실행 능력을 확장하기 위한 "파워 도구"로 사용합니다.

## 프레임워크 실행 모드와의 관계

이 스킬은 **Bash 도구가 사용 가능한 경우**에만 적용됩니다.

| 모드 | 구현 | Bash | 스킬 적용 |
|------|------|------|----------|
| **Mode S** | `agent_sdk.py` (Claude Agent SDK) | 기본 사용 가능 | 적용됨. Read/Write/Edit/Bash/Grep/Glob/WebFetch/WebSearch 사용 가능 |
| **Mode A/B** | LiteLLM + tool_use / 1-shot | permissions.json에서 허가 시만 | Bash 허가가 있으면 적용 |
| **Mode C** | `codex_sdk.py` (Codex SDK) | Codex CLI 도구셋에 의존 | **codex exec 불필요** — 프레임워크가 Codex를 직접 실행. cursor-agent / claude -p는 Bash에서 호출 가능 (Bash 사용 가능한 경우) |

**중요**: Mode C (codex/* 모델)의 Anima는 프레임워크가 Codex SDK를 통해 Codex를 직접 실행합니다. 이 경우 직접 `codex exec`를 Bash에서 호출할 필요가 없습니다. cursor-agent나 claude -p를 사용하고 싶을 때만 이 스킬의 해당 섹션을 참조하세요.

## 도구 선택 우선순위

**비용 효율 순으로 선택하세요.**

| 우선순위 | 도구 | 비용 | 적합 분야 |
|---------|------|------|----------|
| 1 | `codex exec` | 최저 (Codex) | 코드 생성, 편집, 리뷰 |
| 2 | `cursor-agent -p` | 낮음 (Cursor) | 코드 생성, 편집, 다중 파일 |
| 3 | `claude -p` | 높음 (Claude API) | 최후 수단. 위 둘로 해결되지 않을 때만 |

**원칙: codex exec를 먼저 시도합니다. 실패 시 또는 부적합한 태스크에만 cursor-agent → claude 순으로 폴백합니다.**

## 사용해야 할 때

- 다중 파일에 걸친 코드 변경
- 테스트 작성 또는 수정
- 코드 리뷰
- 리팩터링
- 버그 조사 및 구현
- 새 기능 구현

## 사용하지 말아야 할 때

- 단일 파일의 작은 편집 (직접 수행)
- 기억 읽기/쓰기 (자신의 도구를 사용)
- 외부 API 호출 (전용 도구를 사용)
- 검색 또는 조사만 (web_search나 Read로 충분)

---

## 1. codex exec (권장)

**적용 조건**: Mode S 또는 Mode A/B (Bash 허가). Mode C에서는 프레임워크가 Codex를 실행하므로 불필요.

### 기본 구문

```bash
codex exec --full-auto -C /path/to/workspace "프롬프트"
```

작업 디렉토리 `-C`에는 프로젝트 경로를 지정합니다. 메인 프로젝트의 경우 `$ANIMAWORKS_PROJECT_DIR`을 사용할 수 있습니다 (Mode S Bash 실행 환경에서 설정됨).

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--full-auto` | 자동 승인 + 샌드박스 (workspace-write) |
| `-C /path` | 작업 디렉토리 (필수) |
| `-m model` | 모델 (예: `o4-mini`, `o3`) |
| `--sandbox workspace-write` | 워크스페이스 쓰기 권한 (full-auto에 포함) |
| `--json` | JSONL 출력 |
| `-o file` | 최종 메시지를 파일에 기록 |
| `--ephemeral` | 세션 파일 저장 안 함 |

### 실행 예시

#### 코드 생성

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "Implement Markdown parser in src/utils/parser.py. Do not break existing tests."
```

#### 코드 리뷰

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  review
```

#### 테스트 작성

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  "Create unit tests for src/utils/parser.py in tests/test_parser.py."
```

#### 결과를 파일에 저장

```bash
codex exec --full-auto --ephemeral -C /home/main/dev/myproject \
  -o /tmp/codex_result.txt \
  "Analyze this project's architecture and suggest improvements."
```

---

## 2. cursor-agent -p (대안)

**적용 조건**: Mode S 또는 Mode A/B (Bash 허가). Mode C에서도 Bash가 사용 가능하면 적용 가능.

### 기본 구문

```bash
cursor-agent -p --trust --force --workspace /path/to/workspace "프롬프트"
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-p` / `--print` | 비대화 모드 (필수) |
| `--trust` | 워크스페이스 자동 신뢰 |
| `--force` | 커맨드 자동 승인 |
| `--workspace /path` | 작업 디렉토리 (필수) |
| `--model model` | 모델 (예: `sonnet-4`, `gpt-5`) |
| `--output-format text\|json` | 출력 형식 |
| `--mode plan\|ask` | 읽기 전용 모드 (조사용) |

### 실행 예시

#### 코드 생성

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  "Add POST /users endpoint to src/api/routes.py. Include validation."
```

#### 읽기 전용 조사

```bash
cursor-agent -p --trust --mode ask \
  --workspace /home/main/dev/myproject \
  "Are there security issues in this auth flow?"
```

#### 결과를 파일에 저장

```bash
cursor-agent -p --trust --force \
  --workspace /home/main/dev/myproject \
  --output-format text \
  "Find modules with low test coverage and improve them" > /tmp/cursor_result.txt
```

---

## 3. claude -p (폴백)

**적용 조건**: Mode S 또는 Mode A/B (Bash 허가). Mode C에서도 Bash가 사용 가능하면 적용 가능.

codex/cursor-agent로 대응할 수 없을 때만 사용합니다. API 비용이 높습니다.

### 기본 구문

```bash
claude -p --dangerously-skip-permissions --output-format text "프롬프트"
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `-p` / `--print` | 비대화 모드 (필수) |
| `--dangerously-skip-permissions` | 권한 검사 생략 |
| `--model model` | 모델 (예: `sonnet`, `haiku`) |
| `--allowedTools "tools"` | 허용 도구 제한 (예: `"Read Edit Bash(git:*)"`) |
| `--output-format text\|json` | 출력 형식 |
| `--max-budget-usd N` | 비용 상한 (달러) |
| `--no-session-persistence` | 세션 저장 안 함 |

### 실행 예시

```bash
claude -p --dangerously-skip-permissions --no-session-persistence \
  --model haiku --max-budget-usd 0.5 \
  --output-format text \
  "Improve error handling in src/core/parser.py"
```

---

## 프롬프트 작성법

서브에이전트에는 AnimaWorks 컨텍스트가 없습니다. 명확하고 자기 완결적인 프롬프트를 작성하세요.

### 좋은 프롬프트

```
Implement a Python module with these requirements:

File: src/utils/validator.py

Requirements:
- Pydantic v2 BaseModel-based validator
- email, username, password fields
- Password: 8+ chars, alphanumeric
- Raise custom exception on validation error

Constraints:
- from __future__ import annotations at top
- Google-style docstring
- Do not break existing tests
```

### 나쁜 프롬프트

```
Fix the validation somehow
```

→ 컨텍스트가 없고 "somehow"가 모호합니다.

---

## 출력 처리

### 표준 출력 캡처

```bash
RESULT=$(codex exec --full-auto --ephemeral -C /path "프롬프트" 2>/dev/null)
echo "$RESULT"
```

### 파일 경유 (codex 권장)

```bash
codex exec --full-auto --ephemeral -C /path \
  -o /tmp/result.txt "프롬프트"
# 결과 읽기
cat /tmp/result.txt
```

### 종료 코드로 성패 판정

```bash
codex exec --full-auto --ephemeral -C /path "프롬프트"
if [ $? -eq 0 ]; then
  echo "성공"
else
  echo "실패 — cursor-agent로 폴백"
  cursor-agent -p --trust --force --workspace /path "같은 프롬프트"
fi
```

---

## 백그라운드 실행 (중요)

서브에이전트 실행은 **5~20분 이상** 소요될 수 있습니다.
포그라운드로 대기하면 세션이 블로킹되므로 **반드시 백그라운드에서 실행하세요**.

### 기본 패턴: nohup + 결과 파일

```bash
nohup codex exec --full-auto --ephemeral -C /path/to/workspace \
  -o /tmp/codex_result.txt \
  "프롬프트" > /tmp/codex_stdout.log 2>&1 &
echo "PID: $!"
```

cursor-agent의 경우:

```bash
nohup cursor-agent -p --trust --force \
  --workspace /path/to/workspace \
  "프롬프트" > /tmp/cursor_result.txt 2>&1 &
echo "PID: $!"
```

### 완료 확인

```bash
# 프로세스가 아직 실행 중인지 확인
ps -p <PID> > /dev/null 2>&1 && echo "실행 중" || echo "완료"

# 결과 읽기 (완료 후)
cat /tmp/codex_result.txt
# 또는
cat /tmp/cursor_result.txt
```

### 타임아웃

폭주를 방지하기 위해 `timeout`을 병용합니다:

```bash
nohup timeout 30m codex exec --full-auto --ephemeral -C /path \
  -o /tmp/codex_result.txt \
  "프롬프트" > /tmp/codex_stdout.log 2>&1 &
```

- 권장 타임아웃: **30분** (`30m`)
- 작은 태스크: **10분** (`10m`)
- 큰 리팩터링: **60분** (`60m`)

### 실행 중 다른 작업 계속

백그라운드 실행 후 완료를 기다리지 않고 다른 태스크를 진행할 수 있습니다.
정기적으로 프로세스 생존을 확인하고 완료되면 결과를 읽어서 episodes/에 기록하세요.

---

## 안전 가이드라인

1. **작업 디렉토리를 반드시 지정** — 미지정 시 현재 디렉토리에서 실행됨
2. **기밀 정보를 프롬프트에 포함하지 않음** — API 키, 비밀번호 등
3. **codex는 `--full-auto`로 샌드박스 내에서 실행** — 워크스페이스 외 쓰기가 제한됨
4. **실행 후 git diff로 변경 확인** — 의도하지 않은 변경이 없는지 점검
5. **`--ephemeral` 사용** — 불필요한 세션 파일 축적을 방지

---

## 폴백 전략

```
1. codex exec로 시도
   ↓ 실패 또는 품질 부족
2. cursor-agent -p로 재시도
   ↓ 실패 또는 품질 부족
3. claude -p (--max-budget-usd로 비용 제한)로 최종 시도
   ↓ 그래도 실패
4. 직접 실행을 시도하거나 상사에게 보고
```

## 주의사항

- 서브에이전트는 AnimaWorks의 기억이나 도구에 접근할 수 없습니다. 어디까지나 "코딩하는 손"입니다
- 실행 결과를 자신의 episodes/에 기록하고 학습한 패턴을 knowledge/에 축적하세요
- 실행에 5~20분 이상 소요됩니다. 반드시 백그라운드로 실행하고 timeout을 설정하세요
- git 관리되는 리포지토리에서 작업하세요 (변경 추적 및 롤백이 용이)
- Mode S에서는 Bash 실행 시 `ANIMAWORKS_ANIMA_DIR`과 `ANIMAWORKS_PROJECT_DIR`이 환경변수로 설정됩니다
