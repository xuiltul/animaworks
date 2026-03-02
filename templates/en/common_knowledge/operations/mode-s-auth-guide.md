# Mode S (Agent SDK) Authentication Mode Configuration Guide

How to switch the authentication method used by Mode S (Claude Agent SDK) per Anima.
Authentication mode is specified by the explicit **`mode_s_auth`** setting (not auto-detected from credential).

## Authentication Modes

| Mode | mode_s_auth value | Connection | Use case |
|------|-------------------|------------|----------|
| **API Direct** | `"api"` | Anthropic API | Fastest streaming. Consumes API credits |
| **Bedrock** | `"bedrock"` | AWS Bedrock | AWS integration / use within VPC |
| **Vertex AI** | `"vertex"` | Google Vertex AI | GCP integration |
| **Max plan** | `"max"` or unset | Anthropic Max plan | Subscription auth. No API credits needed |

When `mode_s_auth` is unset (`null` or omitted), Max plan is used.

## Resolution Priority

`mode_s_auth` is resolved in this order:

1. **status.json** (per-Anima) — highest priority
2. **config.json anima_defaults** — global default

It is not auto-detected from credential content. You must set `mode_s_auth` explicitly.

## Configuration

### 1. API Direct Mode

Connect directly to Anthropic API. Provides the smoothest streaming experience.

**config.json credential:**

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-xxxxx"
    }
  }
}
```

**status.json (per-Anima):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "anthropic",
  "mode_s_auth": "api"
}
```

If `mode_s_auth` is `"api"` but the credential has no `api_key`, it falls back to Max plan.

### 2. Bedrock Mode

Connect via AWS Bedrock.

**config.json credential:**

```json
{
  "credentials": {
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

**status.json (per-Anima):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

For Bedrock in Mode S, both `execution_mode: "S"` and `mode_s_auth: "bedrock"` are required.

### 3. Vertex AI Mode

Connect via Google Vertex AI.

**config.json credential:**

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

**status.json (per-Anima):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "vertex",
  "execution_mode": "S",
  "mode_s_auth": "vertex"
}
```

### 4. Max Plan Mode (Default)

Uses Claude Code subscription authentication (Max plan etc.).

**config.json credential:**

```json
{
  "credentials": {
    "max": {
      "api_key": ""
    }
  }
}
```

**status.json (per-Anima):**

```json
{
  "model": "claude-sonnet-4-6",
  "credential": "max"
}
```

Omit `mode_s_auth` or set it to `"max"` for Max plan.

## Mixing Auth Modes Per Anima

To use different auth modes within the same organization, set `mode_s_auth` and `credential` per Anima in status.json:

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

| Example (role) | credential | mode_s_auth | Auth mode | Reason |
|----------------|-----------|-------------|-----------|--------|
| Max plan Anima | `"max"` | omitted | Max plan | No API cost |
| API Direct Anima | `"anthropic"` | `"api"` | API Direct | Requires fast streaming |
| Bedrock Anima | `"bedrock"` | `"bedrock"` | Bedrock | Access only from within AWS VPC |

**How to verify current configuration:** Check `credential` and `mode_s_auth` in each Anima's `status.json`:

```bash
# Check mode_s_auth for a specific Anima
cat ~/.animaworks/animas/{name}/status.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model={d.get(\"model\")}, credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')"

# List all Animas
for d in ~/.animaworks/animas/*/; do name=$(basename "$d"); python3 -c "import json; d=json.load(open('$d/status.json')); print(f'$name: credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')" 2>/dev/null; done
```

## Global Default (anima_defaults)

To use Bedrock as the default for all Animas, set it in config.json `anima_defaults`:

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

Individual Animas can override `mode_s_auth` in their status.json.

## Notes

- Auth mode is passed as environment variables to the Claude Code child process via `_build_env()`
- `mode_s_auth` is not auto-detected from credential content. Explicit setting is required
- When `mode_s_auth=api` but credential has no `api_key`, it falls back to Max plan
- For Bedrock / Vertex, set provider-specific keys in credential `keys` and specify the mode with `mode_s_auth`
- Server restart is required after configuration changes
- Mode A/B use credentials via LiteLLM as before (this setting is Mode S-specific)
