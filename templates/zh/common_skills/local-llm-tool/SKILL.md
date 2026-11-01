---
name: local-llm-tool
description: >-
  本地LLM运行工具。通过Ollama或vLLM向GPU上的模型请求文本生成和聊天。
  Use when: 需要进行本地推理、调用Ollama端点、使用本地模型进行摘要和生成时。
tags: [llm, local, ollama, external]
---


# Local LLM 工具

通过Ollama/vLLM）进行文本生成和聊天的本地LLM外部工具。

## 调用方法

**Bash**: 使用`animaworks-tool local_llm <サブコマンド> [引数]`执行

## 操作列表

### generate — 文本生成
```bash
animaworks-tool local_llm generate "プロンプト" [-S "システムプロンプト"]
```

### chat — 聊天（多轮）
```bash
animaworks-tool local_llm chat [--messages JSON] [-S "システムプロンプト"]
```

### models — 模型列表
```bash
animaworks-tool local_llm list
```

### status — 服务器状态确认
```bash
animaworks-tool local_llm status
```

## CLI用法

```bash
animaworks-tool local_llm generate "プロンプト" [-S "システムプロンプト"]
animaworks-tool local_llm list
animaworks-tool local_llm status
```

## 注意事项

- Ollama服务器或vLLM服务器已启动
- 可通过-s/--server指定服务器URL
- 可通过-m/--model指定模型
