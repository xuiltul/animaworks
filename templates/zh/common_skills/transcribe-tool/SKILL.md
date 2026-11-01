---
name: transcribe-tool
description: >-
  语音转文字工具。使用Whisper将音频文件转为文本，必要时进行LLM后处理。
  Use when: 需要转录会议录音、播客听写、从音频文件中提取文本时。
tags: [audio, transcription, whisper, external]
---


# Transcribe 工具

使用 Whisper (faster-whisper) 的语音转文字工具。

## 调用方法

**Bash**: 通过 `animaworks-tool transcribe transcribe <音声ファイル> [オプション]` 执行

### audio — 语音转文字
```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|-----|-----------|------|
| audio_path | string | (必填) | 音频文件路径 |
| language | string | null | 语言代码 (ja, en 等)。null 时自动检测 |
| model | string | "large-v3-turbo" | Whisper 模型名称 |
| raw | boolean | false | 为 true 时跳过 LLM 后处理 |

## CLI 用法

```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## 注意事项

- 需要安装 faster-whisper
- 使用 GPU 时需要支持 CUDA 的 ctranslate2
- 首次运行时模型会自动下载
