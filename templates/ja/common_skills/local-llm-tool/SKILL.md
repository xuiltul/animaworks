---
name: local-llm-tool
description: >-
  ローカルLLM実行ツール。OllamaやvLLMでGPU上のモデルにテキスト生成・チャットを依頼する。
  Use when: オンプレ推論、Ollamaエンドポイント呼び出し、ローカルモデルでの要約・生成が必要なとき。
tags: [llm, local, ollama, external]
---

# Local LLM ツール

ローカルLLM（Ollama/vLLM）経由でテキスト生成・チャットを行う外部ツール。

## 呼び出し方法

**Bash**: `animaworks-tool local_llm <サブコマンド> [引数]` で実行

## アクション一覧

### generate — テキスト生成
```bash
animaworks-tool local_llm generate "プロンプト" [-S "システムプロンプト"]
```

### chat — チャット（複数ターン）
```bash
animaworks-tool local_llm chat [--messages JSON] [-S "システムプロンプト"]
```

### models — モデル一覧
```bash
animaworks-tool local_llm list
```

### status — サーバー状態確認
```bash
animaworks-tool local_llm status
```

## CLI使用法

```bash
animaworks-tool local_llm generate "プロンプト" [-S "システムプロンプト"]
animaworks-tool local_llm list
animaworks-tool local_llm status
```

## 注意事項

- Ollamaサーバーまたは vLLM サーバーが起動していること
- -s/--server でサーバーURL指定可能
- -m/--model でモデル指定可能
