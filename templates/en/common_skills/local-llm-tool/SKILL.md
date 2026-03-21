---
name: local-llm-tool
description: >-
  Local LLM execution tool. Text generation and chat via Ollama/vLLM.
  "local LLM" "ollama" "text generation" "local model"
tags: [llm, local, ollama, external]
---

# Local LLM Tool

External tool for text generation and chat via local LLM (Ollama/vLLM).

## Invocation via Bash

Use **Bash** with `animaworks-tool local_llm <subcommand> [args]`. See Actions below for syntax.

## Actions

### generate — Text generation
```json
{"tool_name": "local_llm", "action": "generate", "args": {"prompt": "prompt text", "system": "system prompt (optional)", "temperature": 0.7, "max_tokens": 2048}}
```

### chat — Multi-turn chat
```json
{"tool_name": "local_llm", "action": "chat", "args": {"messages": [{"role": "user", "content": "question"}], "system": "system prompt (optional)"}}
```

### models — List available models
```json
{"tool_name": "local_llm", "action": "models", "args": {}}
```

### status — Server status
```json
{"tool_name": "local_llm", "action": "status", "args": {}}
```

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool local_llm generate "prompt" [-S "system prompt"]
animaworks-tool local_llm list
animaworks-tool local_llm status
```

## Notes

- Ollama or vLLM server must be running
- Use -s/--server to specify server URL
- Use -m/--model to specify model
