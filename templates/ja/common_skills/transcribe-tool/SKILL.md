---
name: transcribe-tool
description: >-
  音声文字起こしツール。Whisperで音声ファイルをテキスト化し、必要に応じLLM後処理する。
  Use when: 会議録音の転写、ポッドキャストの書き起こし、音声ファイルからテキスト抽出が必要なとき。
tags: [audio, transcription, whisper, external]
---

# Transcribe ツール

Whisper (faster-whisper) を使った音声文字起こしツール。

## 呼び出し方法

**Bash**: `animaworks-tool transcribe transcribe <音声ファイル> [オプション]` で実行

### audio — 音声文字起こし
```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| audio_path | string | (必須) | 音声ファイルのパス |
| language | string | null | 言語コード (ja, en 等)。null で自動検出 |
| model | string | "large-v3-turbo" | Whisperモデル名 |
| raw | boolean | false | true の場合、LLM後処理をスキップ |

## CLI使用法

```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## 注意事項

- faster-whisper のインストールが必要
- GPU使用時はCUDA対応の ctranslate2 が必要
- 初回実行時にモデルが自動ダウンロードされる
