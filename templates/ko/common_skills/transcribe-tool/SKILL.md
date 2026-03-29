---
name: transcribe-tool
description: >-
  음성 문자 변환 도구. Whisper로 오디오를 텍스트로 바꾸고 필요 시 LLM 후처리한다.
  Use when: 회의 녹음 전사, 팟캐스트 텍스트화, 녹음 파일에서 본문 추출이 필요할 때.
tags: [audio, transcription, whisper, external]
---

# Transcribe 도구

Whisper (faster-whisper)를 사용한 음성 문자 변환 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool transcribe transcribe <오디오 파일> [옵션]`으로 실행합니다.

### audio — 음성 문자 변환
```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| audio_path | string | (필수) | 오디오 파일 경로 |
| language | string | null | 언어 코드 (ja, en 등). null이면 자동 감지 |
| model | string | "large-v3-turbo" | Whisper 모델명 |
| raw | boolean | false | true인 경우 LLM 후처리를 건너뜀 |

## CLI 사용법

```bash
animaworks-tool transcribe transcribe audio_file.wav [-l ja] [-m large-v3-turbo]
```

## 주의사항

- faster-whisper가 설치되어 있어야 합니다
- GPU 사용 시 CUDA 호환 ctranslate2가 필요합니다
- 최초 실행 시 모델이 자동 다운로드됩니다
