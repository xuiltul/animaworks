# Mode S (Agent SDK) Authentication Guide

How to configure the authentication method per Anima for Mode S (Claude Agent SDK).
The authentication mode is auto-detected from the credential configuration.

## Authentication Modes

| Mode | Condition | Connection | Use case |
|------|-----------|-----------|----------|
| **API Direct** | credential has `api_key` | Anthropic API | Smoothest streaming. Consumes API credits |
| **Bedrock** | credential `keys` has `aws_access_key_id` | AWS Bedrock | AWS integration / VPC-only access |
| **Vertex AI** | credential `keys` has `vertex_project` | Google Vertex AI | GCP integration |
| **Max plan** | None of the above (default) | Anthropic Max plan | Subscription auth. No API credits needed |

Detection is evaluated top-to-bottom. If `api_key` is present, API Direct mode is always selected.

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
  "credential": "anthropic"
}
```

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
  "credential": "bedrock"
}
```

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
  "credential": "vertex"
}
```

### 4. Max Plan Mode (Default)

Use a credential with no `api_key` and no provider keys.
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

## Mixing Auth Modes Across Animas

Different Animas in the same organization can use different auth modes:

```json
{
  "credentials": {
    "anthropic": { "api_key": "sk-ant-api03-xxxxx" },
    "max": { "api_key": "" },
    "bedrock": { "api_key": "", "keys": { "aws_access_key_id": "AKIA...", "aws_secret_access_key": "...", "aws_region_name": "us-east-1" } }
  }
}
```

| Anima | credential | Auth mode | Reason |
|-------|-----------|-----------|--------|
| sakura | `"max"` | Max plan | Manager role. No API cost |
| kotoha | `"anthropic"` | API Direct | Needs fast streaming |
| rin | `"bedrock"` | Bedrock | VPC-only access from AWS |

## Notes

- Auth mode is passed as environment variables to the Claude Code child process via `_build_env()`
- When both `api_key` and provider keys exist in a credential, `api_key` takes priority (API Direct mode)
- To use Bedrock, create a separate credential with an empty `api_key`
- Server restart is required after configuration changes
- Mode A/B executors use credentials via LiteLLM as before (this setting is Mode S-specific)
