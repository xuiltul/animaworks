# 모델 선택 및 설정 가이드

AnimaWorks의 모델 설정에 관한 종합 가이드입니다.
실행 모드, 지원 모델, 설정 방법, 컨텍스트 윈도우 동작을 설명합니다.

---

## 실행 모드

AnimaWorks는 모델 이름에서 실행 모드를 자동 판별합니다. 4종류의 실행 모드가 있습니다:

| 모드 | 이름 | 개요 | 대상 모델 예시 |
|------|------|------|---------------|
| **S** | SDK | Claude Agent SDK 경유. 가장 고기능 | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **C** | Codex | Codex CLI 경유 | `codex/o4-mini`, `codex/gpt-4.1` |
| **A** | Autonomous | LiteLLM + tool_use 루프 | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `ollama/qwen3:14b` |
| **B** | Basic | 1샷 실행. 프레임워크가 기억 I/O를 대행 | `ollama/gemma3:4b`, `ollama/deepseek-r1*` |

### 모드 판별 우선순위

1. Per-anima `status.json`의 `execution_mode` 명시 지정
2. `~/.animaworks/models.json` (사용자 편집 가능)
3. `config.json` `model_modes` (비권장)
4. 코드 기본 패턴 매칭
5. 불명 → Mode B (안전 측)

---

## 지원 모델 목록

`animaworks models list`로 최신 목록을 확인할 수 있습니다. 주요 모델:

### Claude / Anthropic (Mode S)

| 모델 | 설명 |
|------|------|
| `claude-opus-4-6` | 최고 성능, 권장 |
| `claude-sonnet-4-6` | 균형형, 권장 |
| `claude-haiku-4-5-20251001` | 경량, 고속 |

### OpenAI (Mode A)

| 모델 | 설명 |
|------|------|
| `openai/gpt-4.1` | 최신, 코딩에 강함 |
| `openai/gpt-4.1-mini` | 고속, 저비용 |
| `openai/o3-2025-04-16` | 추론 특화 |

### Google Gemini (Mode A)

| 모델 | 설명 |
|------|------|
| `google/gemini-2.5-pro` | 최고 성능 |
| `google/gemini-2.5-flash` | 고속 균형형 |

### Azure OpenAI (Mode A)

| 모델 | 설명 |
|------|------|
| `azure/gpt-4.1-mini` | Azure OpenAI |
| `azure/gpt-4.1` | Azure OpenAI |

### Vertex AI (Mode A)

| 모델 | 설명 |
|------|------|
| `vertex_ai/gemini-2.5-flash` | Vertex AI Flash |
| `vertex_ai/gemini-2.5-pro` | Vertex AI Pro |

### 로컬 모델 / vLLM / Ollama

| 모델 | 모드 | 설명 |
|------|------|------|
| `openai/qwen3.5-35b-a3b` | A | **권장** — Sonnet 동등 성능 (벤치마크 검증 완료) |
| `ollama/qwen3:14b` | A | 중형, tool_use 지원 |
| `ollama/glm-4.7` | A | tool_use 지원 |
| `ollama/gemma3:4b` | B | 경량 |

### AWS Bedrock

| 모델 | 모드 | 설명 |
|------|------|------|
| `openai/zai.glm-4.7` | A | Bedrock Mantle 경유. 단발 작업 전용 |
| `bedrock/qwen.qwen3-next-80b-a3b` | A | 도구 호출 능력 부족 (비권장) |

---

## 권장 OSS 모델 (벤치마크 검증 완료)

### Qwen3.5-35B — 로컬 GPU 권장 모델

`openai/qwen3.5-35b-a3b` (vLLM 경유)는 AnimaWorks Mode A 에이전트로서 벤치마크 검증된 **권장 로컬 모델**입니다.
Claude Sonnet 4.6과 동등한 종합 점수를 기록했으며, **background_model로서 최적**입니다.

#### 벤치마크 데이터 (2026-03-11 실시)

측정 조건: Mode A (LiteLLM tool_use 루프) 통일, 15개 작업 x 3회/모델

| 모델 | T1 기본 조작 | T2 멀티스텝 | T3 판단/오류 | 종합 | 평균 시간 | 비용 |
|------|:----------:|:---------:|:----------:|:----:|:-------:|:----:|
| **Qwen3.5-35B (local)** | **100%** | **100%** | 60% | **88%** | 9.6s | **$0** |
| Claude Sonnet 4.6 | 100% | 100% | 60% | 88% | 8.5s | ~$0.015/task |
| GLM-4.7 (Bedrock) | 87% | 33% | 53% | 55% | 5.9s | ~$0.003/task |
| Qwen3-Next 80B (Bedrock) | 40% | 27% | 40% | 35% | 5.2s | ~$0.005/task |

#### 주목할 점

- **T1 (기본 조작: 파일 I/O, 도구 호출)**: Sonnet과 동일하게 100%
- **T2 (멀티스텝: CSV 집계, JSON 파싱→쓰기 등)**: Sonnet과 동일하게 100%
- **계산 정확도 (T3-3)**: Qwen3.5가 3/3, Sonnet이 1/3으로 Qwen3.5가 앞섬
- **프롬프트 인젝션 내성 (T3-4)**: 전 모델 0/3 (프레임워크 레벨 대책 필요)
- 파라미터 수가 성능에 직결되지 않음 (80B Qwen3-Next보다 35B Qwen3.5가 크게 앞섬)

#### 권장 설정

```bash
# vLLM credential 설정
# config.json > credentials에 추가:
# "vllm-local": { "api_key": "dummy", "base_url": "http://<vllm-host>:8000/v1" }

# models.json에 추가
# "openai/qwen3.5*": { "mode": "A", "context_window": 64000 }

# background_model로 설정 (Chat=Sonnet, HB/Inbox/Cron=Qwen3.5)
animaworks anima set-background-model {name} openai/qwen3.5-35b-a3b --credential vllm-local
```

#### vLLM 실행 예시

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

#### 용도별 모델 선택 가이드

| 용도 | 권장 모델 | 이유 |
|------|----------|------|
| background_model (HB/Inbox/Cron) | **Qwen3.5-35B** | 비용 $0으로 Sonnet 동등 안정성 |
| foreground (인간 Chat) | Sonnet 4.6 | 오류 처리 안정성과 언어 품질 |
| TaskExec (위임 작업 실행) | Qwen3.5-35B | 비용 $0으로 안정적인 도구 체이닝 |
| 경량 단순 응답 (분류/요약) | GLM-4.7 | 가장 빠르지만 멀티스텝 불가 |

---

## models.json

`~/.animaworks/models.json`에서 모델별 실행 모드와 컨텍스트 윈도우를 정의합니다.
fnmatch 와일드카드 패턴을 사용할 수 있습니다.

### 스키마

```json
{
  "pattern": {
    "mode": "S" | "A" | "B" | "C",
    "context_window": token_count
  }
}
```

### 예시

```json
{
  "claude-opus-4-6":    { "mode": "S", "context_window": 1000000 },
  "claude-sonnet-4-6":  { "mode": "S", "context_window": 1000000 },
  "claude-*":           { "mode": "S", "context_window": 200000 },
  "openai/gpt-4.1*":   { "mode": "A", "context_window": 1000000 },
  "openai/*":           { "mode": "A", "context_window": 128000 },
  "ollama/gemma3*":     { "mode": "B", "context_window": 8192 }
}
```

구체적인 패턴이 우선합니다. `claude-opus-4-6`은 `claude-*`보다 먼저 매칭됩니다.

### 확인 명령어

```bash
animaworks models show            # models.json 내용 표시
animaworks models info {model}    # 해석 결과 확인
```

---

## 모델 변경 절차

### 특정 Anima의 모델을 변경

```bash
# 1. 모델 설정 (status.json 업데이트)
animaworks anima set-model {name} {model_name}

# 2. credential이 필요한 경우
animaworks anima set-model {name} {model_name} --credential {cred_name}

# 3. 서버 실행 중이면 재시작
animaworks anima restart {name}
```

### 전체 Anima를 일괄 변경

```bash
animaworks anima set-model --all {model_name}
```

### 현재 설정 확인

```bash
animaworks anima info {name}    # 모델, 실행 모드, credential 등 표시
animaworks anima list --local   # 전체 Anima의 모델 목록
```

---

## 컨텍스트 윈도우

### 해석 순서

1. `models.json`의 `context_window`
2. `config.json`의 `model_context_windows` (와일드카드 패턴)
3. 코드의 하드코딩 기본값 (`MODEL_CONTEXT_WINDOWS`)
4. 최종 폴백: 128,000 토큰

### 임계값 자동 스케일링

컨텍스트 윈도우 크기에 따라 컴팩션 임계값이 자동 조정됩니다:

- **200K 이상**: 설정값 그대로 (기본 0.50)
- **200K 미만**: 0.98을 향해 선형 스케일

소형 모델에서는 시스템 프롬프트만으로 컨텍스트의 대부분을 차지하므로, 임계값을 높여 오동작을 방지합니다.

---

## 프로바이더별 credential 설정

### Anthropic (기본값)

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-..."
    }
  }
}
```

### Azure OpenAI

```json
{
  "credentials": {
    "azure": {
      "api_key": "",
      "base_url": "https://YOUR_RESOURCE.openai.azure.com",
      "keys": { "api_version": "2024-12-01-preview" }
    }
  }
}
```

### Vertex AI

```json
{
  "credentials": {
    "vertex": {
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "us-central1",
        "vertex_credentials": "/path/to/service-account.json"
      }
    }
  }
}
```

### vLLM (로컬 GPU 추론)

```json
{
  "credentials": {
    "vllm-local": {
      "api_key": "dummy",
      "base_url": "http://192.168.1.100:8000/v1"
    }
  }
}
```

credential 설정 후 Anima에 연결합니다:

```bash
animaworks anima set-model {name} {model_name} --credential {cred_name}
```

---

## 백그라운드 모델 (비용 최적화)

Heartbeat / Inbox / Cron은 메인 모델과 별도의 경량 모델로 실행할 수 있습니다.
`background_model`을 설정하면 이러한 백그라운드 처리 비용을 대폭 절감할 수 있습니다.

### foreground / background 구분

| 구분 | 사용 모델 | 대상 트리거 |
|------|----------|------------|
| **foreground** | 메인 모델 (`model`) | `chat` (인간과의 대화), `task:*` (TaskExec 실작업) |
| **background** | `background_model` (미설정 시 메인 모델) | `heartbeat`, `inbox:*` (Anima 간 DM), `cron:*` |

Heartbeat / Inbox / Cron은 "판단 및 트리아지"가 주 목적이며, 실행은 TaskExec(메인 모델)이 담당합니다.

### 해석 순서

1. Per-anima `status.json`의 `background_model`
2. `config.json`의 `heartbeat.default_model` (글로벌 기본값)
3. 메인 모델 (`model`)로 폴백

### 설정 방법

```bash
# 특정 Anima에 background_model 설정
animaworks anima set-background-model {name} claude-sonnet-4-6

# credential이 다른 프로바이더인 경우
animaworks anima set-background-model {name} azure/gpt-4.1-mini --credential azure

# 전체 Anima에 일괄 설정
animaworks anima set-background-model --all claude-sonnet-4-6

# background_model 삭제 (메인 모델로 폴백)
animaworks anima set-background-model {name} --clear

# 서버 실행 중이면 재시작
animaworks anima restart {name}
```

### status.json에서의 확인

```json
{
  "model": "claude-opus-4-6",
  "background_model": "claude-sonnet-4-6",
  "background_credential": null
}
```

`background_model`이 미설정이거나 메인 모델과 동일한 경우, 전환은 생략됩니다.

---

## 역할 템플릿과 기본 모델

`animaworks anima set-role`로 역할을 변경하면 기본 모델도 변경됩니다:

| 역할 | 기본 모델 | background_model | max_turns | max_chains |
|------|----------|-----------------|-----------|------------|
| engineer | claude-opus-4-6 | claude-sonnet-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | claude-sonnet-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | — | 80 | 5 |
| researcher | claude-sonnet-4-6 | — | 30 | 2 |
| ops | openai/glm-4.7-flash | — | 30 | 2 |
| general | claude-sonnet-4-6 | — | 20 | 2 |

Opus 계열 역할(engineer, manager)은 `background_model`로 Sonnet이 자동 설정됩니다.
Sonnet 이하의 역할은 이미 비용 효율적이므로 `background_model`이 미설정입니다.

---

## 자주 묻는 질문

### 모델을 변경했는데 반영되지 않음

`set-model`은 `status.json`만 업데이트합니다. 서버 실행 중에는 `anima restart {name}` 또는 `anima reload {name}`이 필요합니다.

### models.json을 편집했는데 반영되지 않음

models.json은 파일의 mtime으로 자동 리로드됩니다. `anima reload`로도 반영 가능합니다.

### 컨텍스트 윈도우를 늘리고 싶음

`models.json`의 `context_window`를 변경하거나, `config.json`의 `model_context_windows`로 오버라이드하세요.

### 어떤 모델을 선택해야 할지 모르겠음

- **고품질, 자율 실행이 필요** → `claude-opus-4-6` (Mode S)
- **균형, 비용 중시** → `claude-sonnet-4-6` (Mode S)
- **저비용, 대량 처리** → `openai/gpt-4.1-mini` (Mode A)
- **로컬 GPU, 비용 $0** → `openai/qwen3.5-35b-a3b` (Mode A, vLLM) **권장**
- **로컬, 경량** → `ollama/qwen3:14b` (Mode A)

### Heartbeat / Cron 비용을 줄이고 싶음

`background_model`을 설정하세요. 자세한 내용은 위의 "백그라운드 모델 (비용 최적화)" 섹션을 참조하세요.
Opus를 메인으로 사용하는 경우, `background_model`에 Sonnet을 설정하는 것만으로 Heartbeat + Inbox 비용을 약 73% 절감할 수 있습니다.

vLLM으로 `openai/qwen3.5-35b-a3b`를 `background_model`로 설정하면 **백그라운드 처리 비용을 완전히 $0으로 만들 수 있습니다**. Sonnet 동등의 88% 종합 점수가 확인되었습니다.
