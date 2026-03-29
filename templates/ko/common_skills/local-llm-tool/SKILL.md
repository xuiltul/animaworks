---
name: local-llm-tool
description: >-
  로컬 LLM 실행 도구. Ollama·vLLM으로 GPU 모델에 텍스트 생성·채팅을 요청한다.
  Use when: 온프레미스 추론, Ollama 엔드포인트 호출, 로컬 모델 요약·생성이 필요할 때.
tags: [llm, local, ollama, external]
---

# Local LLM 도구

로컬 LLM(Ollama/vLLM)을 통해 텍스트 생성 및 채팅을 수행하는 외부 도구입니다.

## 호출 방법

**Bash**: `animaworks-tool local_llm <서브커맨드> [인수]`로 실행합니다.

## 액션 목록

### generate — 텍스트 생성
```bash
animaworks-tool local_llm generate "프롬프트" [-S "시스템 프롬프트"]
```

### chat — 채팅 (다중 턴)
```bash
animaworks-tool local_llm chat [--messages JSON] [-S "시스템 프롬프트"]
```

### models — 모델 목록
```bash
animaworks-tool local_llm list
```

### status — 서버 상태 확인
```bash
animaworks-tool local_llm status
```

## CLI 사용법

```bash
animaworks-tool local_llm generate "프롬프트" [-S "시스템 프롬프트"]
animaworks-tool local_llm list
animaworks-tool local_llm status
```

## 주의사항

- Ollama 서버 또는 vLLM 서버가 실행 중이어야 합니다
- -s/--server로 서버 URL 지정 가능
- -m/--model로 모델 지정 가능
