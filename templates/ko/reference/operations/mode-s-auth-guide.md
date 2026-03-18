# Mode S (Agent SDK) 인증 모드 설정 가이드

Mode S (Claude Agent SDK)에서 사용하는 인증 방식을 Anima별로 전환하는 방법입니다.
인증 모드는 **`mode_s_auth`**라는 명시적인 설정으로 지정합니다 (credential의 자동 판별이 아님).

구현: `core/execution/agent_sdk.py`의 `_build_env()`가 Claude Code 자식 프로세스의 환경 변수를 구성합니다.

## 인증 모드 목록

| 모드 | mode_s_auth 값 | 연결 대상 | 용도 |
|------|----------------|----------|------|
| **API 직접** | `"api"` | Anthropic API | 최고속 스트리밍. API 크레딧 소비 |
| **Bedrock** | `"bedrock"` | AWS Bedrock | AWS 통합 / VPC 내 이용 |
| **Vertex AI** | `"vertex"` | Google Vertex AI | GCP 통합 |
| **Max plan** | `"max"` 또는 미설정 | Anthropic Max plan | 구독 인증. API 크레딧 불필요 |

`mode_s_auth`가 미설정(`null` 또는 생략)인 경우 Max plan이 사용됩니다.

## 설정 우선순위

`mode_s_auth`는 다음 순서로 해석됩니다:

1. **status.json** (Anima 개별) — 최우선
2. **config.json anima_defaults** — 글로벌 기본값

credential의 내용에서 자동 판별하지 않습니다. 반드시 `mode_s_auth`를 명시적으로 지정하세요.

## 설정 방법

### 1. API 직접 모드

Anthropic API에 직접 연결합니다. 스트리밍이 가장 매끄럽습니다.

**config.json의 credential 설정:**

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-xxxxx"
    }
  }
}
```

- `api_key`: Anthropic API 키. 비어 있으면 환경 변수 `ANTHROPIC_API_KEY`로 폴백
- `base_url`: 커스텀 엔드포인트 (선택). 지정 시 `ANTHROPIC_BASE_URL`로 자식 프로세스에 전달 (프록시 또는 온프레미스 이용 시)

**status.json (Anima 개별):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "anthropic",
  "mode_s_auth": "api"
}
```

`mode_s_auth`가 `"api"`이지만 credential에 `api_key`가 없고 환경 변수에도 없는 경우, Max plan으로 폴백합니다.

### 2. Bedrock 모드

AWS Bedrock을 경유하여 연결합니다. credential의 `keys`가 `extra_keys`로 ModelConfig에 전달되어 환경 변수에 매핑됩니다.

**config.json의 credential 설정:**

```json
{
  "credentials": {
    "bedrock": {
      "api_key": "",
      "keys": {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "aws_region_name": "us-east-1",
        "aws_session_token": "",
        "aws_profile": ""
      }
    }
  }
}
```

| keys 키 | 환경 변수 | 설명 |
|---------|----------|------|
| aws_access_key_id | AWS_ACCESS_KEY_ID | 필수 |
| aws_secret_access_key | AWS_SECRET_ACCESS_KEY | 필수 |
| aws_region_name | AWS_REGION | 리전 |
| aws_session_token | AWS_SESSION_TOKEN | 임시 인증 (선택) |
| aws_profile | AWS_PROFILE | 프로필 이름 (선택) |

`keys`에 값이 없는 항목은 위 해당 환경 변수로 폴백합니다. 프로덕션에서는 `AWS_PROFILE`만 설정하고 키를 config에 넣지 않는 운용도 가능합니다.

**status.json (Anima 개별):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

Bedrock을 Mode S에서 사용하려면 `execution_mode: "S"`와 `mode_s_auth: "bedrock"` 양쪽 모두 지정해야 합니다.

### 3. Vertex AI 모드

Google Vertex AI를 경유하여 연결합니다. credential의 `keys`가 `extra_keys`로 ModelConfig에 전달되어 환경 변수에 매핑됩니다.

**config.json의 credential 설정:**

```json
{
  "credentials": {
    "vertex": {
      "api_key": "",
      "keys": {
        "vertex_project": "my-gcp-project",
        "vertex_location": "us-central1",
        "vertex_credentials": "/path/to/service-account.json"
      }
    }
  }
}
```

| keys 키 | 환경 변수 | 설명 |
|---------|----------|------|
| vertex_project | CLOUD_ML_PROJECT_ID | GCP 프로젝트 ID |
| vertex_location | CLOUD_ML_REGION | 리전 (예: us-central1) |
| vertex_credentials | GOOGLE_APPLICATION_CREDENTIALS | 서비스 계정 JSON 경로 |

`keys`에 값이 없는 항목은 위 해당 환경 변수로 폴백합니다. ADC(Application Default Credentials) 이용 시 `vertex_credentials`를 생략할 수 있습니다.

**status.json (Anima 개별):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "vertex",
  "execution_mode": "S",
  "mode_s_auth": "vertex"
}
```

### 4. Max plan 모드 (기본값)

Claude Code의 구독 인증(Max plan 등)을 사용합니다.

**config.json의 credential 설정:**

```json
{
  "credentials": {
    "max": {
      "api_key": ""
    }
  }
}
```

**status.json (Anima 개별):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "max"
}
```

`mode_s_auth`를 생략하거나 `"max"`로 설정하면 Max plan이 됩니다.

## Anima별 인증 모드 혼용

같은 조직 내에서 인증 모드를 혼용하려면, 각 Anima의 status.json에서 `mode_s_auth`와 `credential`을 지정합니다:

```json
{
  "credentials": {
    "anthropic": { "api_key": "sk-ant-api03-xxxxx" },
    "max": { "api_key": "" },
    "bedrock": {
      "api_key": "",
      "keys": {
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "aws_region_name": "us-east-1"
      }
    }
  }
}
```

| 예시 (역할) | credential | mode_s_auth | 인증 모드 | 이유 |
|------------|-----------|-------------|----------|------|
| Max plan 이용 Anima | `"max"` | 생략 | Max plan | API 비용 불필요 |
| API 직접 이용 Anima | `"anthropic"` | `"api"` | API 직접 | 고속 스트리밍이 필요 |
| Bedrock 이용 Anima | `"bedrock"` | `"bedrock"` | Bedrock | AWS VPC 내부에서만 접근 |

**현재 구성을 확인하는 방법:** 각 Anima의 `status.json`에서 `credential`과 `mode_s_auth`를 확인합니다:

```bash
# 특정 Anima의 mode_s_auth 확인
cat ~/.animaworks/animas/{name}/status.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model={d.get(\"model\")}, credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')"

# 전체 Anima 목록
for d in ~/.animaworks/animas/*/; do name=$(basename "$d"); python3 -c "import json; d=json.load(open('$d/status.json')); print(f'$name: credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')" 2>/dev/null; done
```

## 글로벌 기본값 (anima_defaults)

전체 Anima에서 Bedrock을 기본값으로 사용하려면 config.json의 `anima_defaults`에 설정합니다:

```json
{
  "anima_defaults": {
    "mode_s_auth": "bedrock"
  },
  "credentials": {
    "bedrock": { "api_key": "", "keys": { "aws_access_key_id": "...", ... } }
  }
}
```

개별 Anima의 status.json에서 `mode_s_auth`를 오버라이드할 수 있습니다.

## 주의 사항

- 인증 모드는 `_build_env()`를 통해 Claude Code 자식 프로세스의 환경 변수로 전달됩니다
- `mode_s_auth`는 credential 내용에서 자동 판별되지 않습니다. 명시 지정이 필수입니다
- `mode_s_auth=api`이지만 credential에 `api_key`가 없고 환경 변수에도 없는 경우, Max plan으로 폴백합니다
- Bedrock / Vertex에서는 credential의 `keys`가 `extra_keys`로 전달되어 환경 변수에 매핑됩니다. `keys`에 미설정인 항목은 동일 이름의 환경 변수로 폴백
- API 모드에서 커스텀 엔드포인트를 사용할 경우, credential의 `base_url`을 지정하면 `ANTHROPIC_BASE_URL`로 자식 프로세스에 전달됩니다
- 설정 변경 후 서버 재시작이 필요합니다
- Mode A/B에서는 기존과 같이 LiteLLM이 credential을 사용합니다 (이 설정은 Mode S 전용)
