# Model Selection and Configuration Guide

Comprehensive guide for model configuration in AnimaWorks.
Covers execution modes, supported models, configuration methods, and context window behavior.

---

## Execution Modes

AnimaWorks automatically determines the execution mode from the model name. There are 6 execution modes:

| Mode | Name | Overview | Example Models |
|------|------|----------|---------------|
| **S** | SDK | Via Claude Agent SDK. Most capable | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **C** | Codex | Via Codex CLI | `codex/o4-mini`, `codex/gpt-4.1` |
| **D** | Cursor Agent | Via Cursor Agent CLI (`cursor-agent`). MCP-integrated | `cursor/*` |
| **G** | Gemini CLI | Via Gemini CLI. MCP-integrated | `gemini/*` |
| **A** | Autonomous | LiteLLM + tool_use loop | `openai/gpt-4.1`, `google/gemini-2.5-pro`, `ollama/qwen3:14b` |
| **B** | Basic | Single-shot execution. Framework handles memory I/O | `ollama/gemma3:4b`, `ollama/deepseek-r1*` |

### Mode Resolution Priority

1. Per-anima `status.json` explicit `execution_mode`
2. `~/.animaworks/models.json` (user-editable)
3. `config.json` `model_modes` (deprecated)
4. Code default pattern matching
5. Unknown → Mode B (safe side)

---

## Supported Models

Run `animaworks models list` for the latest catalog. Key models:

### Claude / Anthropic (Mode S)

| Model | Description |
|-------|-------------|
| `claude-opus-4-6` | Highest performance, recommended |
| `claude-sonnet-4-6` | Balanced, recommended |
| `claude-haiku-4-5-20251001` | Lightweight, fast |

### OpenAI (Mode A)

| Model | Description |
|-------|-------------|
| `openai/gpt-4.1` | Latest, strong at coding |
| `openai/gpt-4.1-mini` | Fast, low cost |
| `openai/o3-2025-04-16` | Reasoning-focused |

### Google Gemini (Mode A)

| Model | Description |
|-------|-------------|
| `google/gemini-2.5-pro` | Highest performance |
| `google/gemini-2.5-flash` | Fast, balanced |

### Azure OpenAI (Mode A)

| Model | Description |
|-------|-------------|
| `azure/gpt-4.1-mini` | Azure OpenAI |
| `azure/gpt-4.1` | Azure OpenAI |

### Vertex AI (Mode A)

| Model | Description |
|-------|-------------|
| `vertex_ai/gemini-2.5-flash` | Vertex AI Flash |
| `vertex_ai/gemini-2.5-pro` | Vertex AI Pro |

### Local Models / vLLM / Ollama

| Model | Mode | Description |
|-------|------|-------------|
| `openai/qwen3.5-35b-a3b` | A | **Recommended** — Sonnet-equivalent performance (benchmark verified) |
| `ollama/qwen3:14b` | A | Medium, tool_use capable |
| `ollama/glm-4.7` | A | tool_use capable |
| `ollama/gemma3:4b` | B | Lightweight |

### AWS Bedrock

| Model | Mode | Description |
|-------|------|-------------|
| `openai/zai.glm-4.7` | A | Via Bedrock Mantle. Single-step tasks only |
| `bedrock/qwen.qwen3-next-80b-a3b` | A | Insufficient tool calling ability (not recommended) |

---

## Recommended OSS Model (Benchmark Verified)

### Qwen3.5-35B — Recommended Local GPU Model

`openai/qwen3.5-35b-a3b` (via vLLM) is a **benchmark-verified recommended local model** for AnimaWorks Mode A agents.
It achieved the same overall score as Claude Sonnet 4.6 and is **optimal as a background_model**.

#### Benchmark Data (2026-03-11)

Conditions: Mode A (LiteLLM tool_use loop) unified, 15 tasks × 3 runs per model

| Model | T1 Basic | T2 Multi-step | T3 Judgment | Overall | Avg Time | Cost |
|-------|:--------:|:-------------:|:-----------:|:-------:|:--------:|:----:|
| **Qwen3.5-35B (local)** | **100%** | **100%** | 60% | **88%** | 9.6s | **$0** |
| Claude Sonnet 4.6 | 100% | 100% | 60% | 88% | 8.5s | ~$0.015/task |
| GLM-4.7 (Bedrock) | 87% | 33% | 53% | 55% | 5.9s | ~$0.003/task |
| Qwen3-Next 80B (Bedrock) | 40% | 27% | 40% | 35% | 5.2s | ~$0.005/task |

#### Key Findings

- **T1 (Basic: file I/O, tool calls)**: Tied with Sonnet at 100%
- **T2 (Multi-step: CSV aggregation, JSON parse→write, etc.)**: Tied with Sonnet at 100%
- **Calculation accuracy (T3-3)**: Qwen3.5 outperformed Sonnet (3/3 vs 1/3)
- **Prompt injection resistance (T3-4)**: All models scored 0/3 (framework-level mitigation required)
- Parameter count does not determine performance (35B Qwen3.5 significantly outperformed 80B Qwen3-Next)

#### Recommended Configuration

```bash
# vLLM credential setup
# Add to config.json > credentials:
# "vllm-local": { "api_key": "dummy", "base_url": "http://<vllm-host>:8000/v1" }

# Add to models.json
# "openai/qwen3.5*": { "mode": "A", "context_window": 64000 }

# Set as background_model (Chat=Sonnet, HB/Inbox/Cron=Qwen3.5)
animaworks anima set-background-model {name} openai/qwen3.5-35b-a3b --credential vllm-local
```

#### vLLM Launch Example

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

#### Model Selection by Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| background_model (HB/Inbox/Cron) | **Qwen3.5-35B** | $0 cost with Sonnet-equivalent stability |
| foreground (human Chat) | Sonnet 4.6 | Error handling stability and language quality |
| TaskExec (delegated tasks) | Qwen3.5-35B | $0 cost with stable tool chaining |
| Lightweight responses (classification/summary) | GLM-4.7 | Fastest, but multi-step incapable |

---

## models.json

`~/.animaworks/models.json` defines execution mode and context window per model.
fnmatch wildcard patterns are supported.

### Schema

```json
{
  "pattern": {
    "mode": "S" | "C" | "D" | "G" | "A" | "B",
    "context_window": token_count
  }
}
```

### Example

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

More specific patterns take priority. `claude-opus-4-6` matches before `claude-*`.

### Verification commands

```bash
animaworks models show            # Show models.json contents
animaworks models info {model}    # Check resolved result
```

---

## Changing Models

### Change a specific Anima's model

```bash
# 1. Update model (writes to status.json)
animaworks anima set-model {name} {model_name}

# 2. If credential is needed
animaworks anima set-model {name} {model_name} --credential {cred_name}

# 3. Restart if server is running
animaworks anima restart {name}
```

### Change all Anima at once

```bash
animaworks anima set-model --all {model_name}
```

### Check current config

```bash
animaworks anima info {name}    # Show model, execution mode, credential, etc.
animaworks anima list --local   # Model column for all Anima
```

---

## Context Window

### Resolution order

1. `models.json` `context_window`
2. `config.json` `model_context_windows` (wildcard patterns)
3. Hardcoded defaults (`MODEL_CONTEXT_WINDOWS`)
4. Final fallback: 128,000 tokens

### Threshold auto-scaling

Compaction threshold auto-adjusts based on context window size:

- **200K+**: Uses configured value as-is (default 0.50)
- **Below 200K**: Linear scale toward 0.98

Small models have system prompts consuming most of the context, so higher thresholds prevent false triggers.

---

## Provider Credential Setup

### Anthropic (default)

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

### vLLM (Local GPU Inference)

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

After setting credentials, bind to Anima:

```bash
animaworks anima set-model {name} {model_name} --credential {cred_name}
```

---

## Background Model (Cost Optimization)

Heartbeat / Inbox / Cron can run on a lighter model separate from the main model.
Setting `background_model` can drastically reduce the cost of these background processes.

### Foreground / Background Scope

| Scope | Model Used | Triggers |
|-------|-----------|----------|
| **foreground** | Main model (`model`) | `chat` (human interaction), `task:*` (TaskExec work) |
| **background** | `background_model` (falls back to main model if unset) | `heartbeat`, `inbox:*` (Anima-to-Anima DM), `cron:*` |

Heartbeat / Inbox / Cron primarily do triage and judgment; actual execution is handled by TaskExec (main model).

### Resolution Order

1. Per-anima `status.json` `background_model`
2. `config.json` `heartbeat.default_model` (global default)
3. Falls back to main model (`model`)

### Configuration

```bash
# Set background_model for a specific Anima
animaworks anima set-background-model {name} claude-sonnet-4-6

# When using a different provider credential
animaworks anima set-background-model {name} azure/gpt-4.1-mini --credential azure

# Set for all Anima at once
animaworks anima set-background-model --all claude-sonnet-4-6

# Remove background_model (falls back to main model)
animaworks anima set-background-model {name} --clear

# Restart if server is running
animaworks anima restart {name}
```

### Checking in status.json

```json
{
  "model": "claude-opus-4-6",
  "background_model": "claude-sonnet-4-6",
  "background_credential": null
}
```

When `background_model` is unset or identical to the main model, the swap is skipped.

---

## Role Templates and Default Models

Changing role with `animaworks anima set-role` also updates the default model:

| Role | Default Model | background_model | max_turns | max_chains |
|------|--------------|-----------------|-----------|------------|
| engineer | claude-opus-4-6 | claude-sonnet-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | claude-sonnet-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | — | 80 | 5 |
| researcher | claude-sonnet-4-6 | — | 30 | 2 |
| ops | openai/glm-4.7-flash | — | 30 | 2 |
| general | claude-sonnet-4-6 | — | 20 | 2 |

Opus-tier roles (engineer, manager) get Sonnet as `background_model` automatically.
Sonnet-tier and below are already cost-efficient, so `background_model` is left unset.

---

## FAQ

### Model change not taking effect

`set-model` only updates `status.json`. When the server is running, you need `anima restart {name}` or `anima reload {name}`.

### models.json edit not reflected

models.json auto-reloads based on file mtime. `anima reload` can also trigger refresh.

### How to increase context window

Edit `context_window` in `models.json`, or override via `config.json` `model_context_windows`.

### Which model to choose

- **High quality, autonomous execution** → `claude-opus-4-6` (Mode S)
- **Balanced, cost-conscious** → `claude-sonnet-4-6` (Mode S)
- **Low cost, high volume** → `openai/gpt-4.1-mini` (Mode A)
- **Local GPU, $0 cost** → `openai/qwen3.5-35b-a3b` (Mode A, vLLM) **Recommended**
- **Local, lightweight** → `ollama/qwen3:14b` (Mode A)

### How to reduce Heartbeat / Cron costs

Set `background_model`. See the "Background Model (Cost Optimization)" section above.
For Opus-based Anima, setting Sonnet as `background_model` reduces Heartbeat + Inbox costs by ~73%.

With `openai/qwen3.5-35b-a3b` via vLLM as `background_model`, **background processing costs drop to $0**. Verified at 88% overall score, equivalent to Sonnet.
