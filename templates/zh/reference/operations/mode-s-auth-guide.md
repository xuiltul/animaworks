# Mode S（Agent SDK）认证模式配置指南

按 Anima 切换 Mode S（Claude Agent SDK）中使用的认证方式。
认证模式通过 **`mode_s_auth`** 这一显式配置指定（并非自动判定 credential）。

实现：`core/execution/engines/claude/executor.py` 的 `_build_env()` 构建 Claude Code 子进程的环境变量。

## 认证模式列表

| 模式 | mode_s_auth 值 | 连接目标 | 用途 |
|--------|----------------|--------|------|
| **API 直接** | `"api"` | Anthropic API | 最快的流式传输。消耗 API 额度 |
| **Bedrock** | `"bedrock"` | AWS Bedrock | AWS 集成・VPC 内使用 |
| **Vertex AI** | `"vertex"` | Google Vertex AI | GCP 集成 |
| **Max plan** | `"max"` 或未设置 | Anthropic Max plan | 订阅认证。无需 API 额度 |

`mode_s_auth` 未设置（`null` 或省略）时为 Max plan。

## 配置的优先级

`mode_s_auth` 按以下顺序解析：

1. **status.json**（Anima 个别）— 最高优先级
2. **config.json anima_defaults** — 全局默认

不会根据 credential 的内容自动判定。请显式指定 `mode_s_auth`。

## 配置方法

### 1. API 直接模式

直接连接 Anthropic API。流式传输最为顺畅。

**config.json 的 credential 配置：**

```json
{
  "credentials": {
    "anthropic": {
      "api_key": "sk-ant-api03-xxxxx"
    }
  }
}
```

- `api_key`：Anthropic API 密钥。为空时回退到环境变量 `ANTHROPIC_API_KEY`
- `base_url`：自定义端点（可选）。指定时作为 `ANTHROPIC_BASE_URL` 传递给子进程（用于代理或本地部署场景）

**status.json（Anima 个别）：**

```json
{
  "model": "claude-sonnet-5-5",
  "credential": "anthropic",
  "mode_s_auth": "api"
}
```

`mode_s_auth` 为 `"api"` 且 credential 中没有 `api_key`、环境变量中也没有时，回退到 Max plan。

### 2. Bedrock 模式

通过 AWS Bedrock 连接。credential 的 `keys` 作为 `extra_keys` 传递给 ModelConfig，并映射到环境变量。

**config.json 的 credential 配置：**

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

| keys 键 | 环境变量 | 说明 |
|-----------|----------|------|
| aws_access_key_id | AWS_ACCESS_KEY_ID | 必填 |
| aws_secret_access_key | AWS_SECRET_ACCESS_KEY | 必填 |
| aws_region_name | AWS_REGION | 区域 |
| aws_session_token | AWS_SESSION_TOKEN | 临时认证（可选） |
| aws_profile | AWS_PROFILE | 配置文件名称（可选） |

`keys` 中没有值的项目，会回退到上表对应的环境变量。生产环境中也可以只设置 `AWS_PROFILE`，不将密钥写入 config。

**status.json（Anima 个别）：**

```json
{
  "model": "claude-sonnet-5-5",
  "credential": "bedrock",
  "execution_mode": "S",
  "mode_s_auth": "bedrock"
}
```

在 Mode S 中使用 Bedrock 时，需同时指定 `execution_mode: "S"` 和 `mode_s_auth: "bedrock"`。

### 3. Vertex AI 模式

通过 Google Vertex AI 连接。credential 的 `keys` 作为 `extra_keys` 传递给 ModelConfig，并映射到环境变量。

**config.json 的 credential 配置：**

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

| keys 键 | 环境变量 | 说明 |
|-----------|----------|------|
| vertex_project | CLOUD_ML_PROJECT_ID | GCP 项目 ID |
| vertex_location | CLOUD_ML_REGION | 区域（例：us-central1） |
| vertex_credentials | GOOGLE_APPLICATION_CREDENTIALS | 服务账号 JSON 路径 |

`keys` 中没有值的项目，会回退到上表对应的环境变量。使用 ADC（Application Default Credentials）时可省略 `vertex_credentials`。

**status.json（Anima 个别）：**

```json
{
  "model": "claude-sonnet-5-5",
  "credential": "vertex",
  "execution_mode": "S",
  "mode_s_auth": "vertex"
}
```

### 4. Max plan 模式（默认）

使用 Claude Code 的订阅认证（Max plan 等）。

**config.json 的 credential 配置：**

```json
{
  "credentials": {
    "max": {
      "api_key": ""
    }
  }
}
```

**status.json（Anima 个别）：**

```json
{
  "model": "claude-sonnet-5-5",
  "credential": "max"
}
```

省略 `mode_s_auth` 或将其设为 `"max"` 即为 Max plan。

## 按 Anima 分别使用的示例

在同一组织内混用认证模式时，在各 Anima 的 status.json 中指定 `mode_s_auth` 和 `credential`：

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

| 示例（角色） | credential | mode_s_auth | 认证模式 | 理由 |
|-----------|-----------|-------------|-----------|------|
| 使用 Max plan 的 Anima | `"max"` | 省略 | Max plan | 无需 API 成本 |
| 直接使用 API 的 Anima | `"anthropic"` | `"api"` | API 直接 | 需要高速流式传输 |
| 使用 Bedrock 的 Anima | `"bedrock"` | `"bedrock"` | Bedrock | 仅从 AWS VPC 内访问 |

**确认当前配置的方法：** 在各 Anima 的 `status.json` 中确认 `credential` 和 `mode_s_auth`：

```bash
# 特定 Anima の mode_s_auth 確認
cat ~/.animaworks/animas/{name}/status.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'model={d.get(\"model\")}, credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')"

# 全 Anima の一覧
for d in ~/.animaworks/animas/*/; do name=$(basename "$d"); python3 -c "import json; d=json.load(open('$d/status.json')); print(f'$name: credential={d.get(\"credential\")}, mode_s_auth={d.get(\"mode_s_auth\")}')" 2>/dev/null; done
```

## 全局默认（anima_defaults）

希望将所有 Anima 的默认设为 Bedrock 时，在 config.json 的 `anima_defaults` 中配置：

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

可通过个别 Anima 的 status.json 覆盖 `mode_s_auth`。

## 注意事项

- 认证模式通过 `_build_env()` 作为 Claude Code 子进程的环境变量传递
- `mode_s_auth` 不会根据 credential 的内容自动判定。必须显式指定
- `mode_s_auth=api` 中 credential 没有 `api_key` 且环境变量中也没有时，回退到 Max plan
- 在 Bedrock / Vertex 中，credential 的 `keys` 作为 `extra_keys` 传递，并映射到环境变量。`keys` 未设置的项目会回退到同名环境变量
- 在 API 模式中使用自定义端点时，指定 credential 的 `base_url` 后会作为 `ANTHROPIC_BASE_URL` 传递给子进程
- 配置变更后需要重启服务器
- 在 Mode A/B 中，LiteLLM 照旧使用 credential（此配置仅适用于 Mode S）
