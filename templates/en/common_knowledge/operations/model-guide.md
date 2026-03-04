# Model Selection and Configuration Guide

Comprehensive guide for model configuration in AnimaWorks.
Covers execution modes, supported models, configuration methods, and context window behavior.

---

## Execution Modes

AnimaWorks automatically determines the execution mode from the model name. There are 4 execution modes:

| Mode | Name | Overview | Example Models |
|------|------|----------|---------------|
| **S** | SDK | Via Claude Agent SDK. Most capable | `claude-opus-4-6`, `claude-sonnet-4-6` |
| **C** | Codex | Via Codex CLI | `codex/o4-mini`, `codex/gpt-4.1` |
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

### Local Models / Ollama

| Model | Mode | Description |
|-------|------|-------------|
| `ollama/qwen3:14b` | A | Medium, tool_use capable |
| `ollama/glm-4.7` | A | tool_use capable |
| `ollama/gemma3:4b` | B | Lightweight |

---

## models.json

`~/.animaworks/models.json` defines execution mode and context window per model.
fnmatch wildcard patterns are supported.

### Schema

```json
{
  "pattern": {
    "mode": "S" | "A" | "B" | "C",
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
    "vllm-gpu41": {
      "api_key": "dummy",
      "base_url": "http://192.168.12.41:8000/v1"
    }
  }
}
```

After setting credentials, bind to Anima:

```bash
animaworks anima set-model {name} {model_name} --credential {cred_name}
```

---

## Role Templates and Default Models

Changing role with `animaworks anima set-role` also updates the default model:

| Role | Default Model | max_turns | max_chains |
|------|--------------|-----------|------------|
| engineer | claude-opus-4-6 | 200 | 10 |
| manager | claude-opus-4-6 | 50 | 3 |
| writer | claude-sonnet-4-6 | 80 | 5 |
| researcher | claude-sonnet-4-6 | 30 | 2 |
| ops | openai/glm-4.7-flash | 30 | 2 |
| general | claude-sonnet-4-6 | 20 | 2 |

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
- **Local, private** → `ollama/qwen3:14b` (Mode A)
