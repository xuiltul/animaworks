# 음성 채팅 (Voice Chat) 가이드

Anima와의 음성 대화 기능에 대한 레퍼런스입니다.
브라우저 마이크 입력 → STT (음성 인식) → 채팅 파이프라인 → TTS (음성 합성) → 브라우저 재생.

## 아키텍처 개요

```
브라우저 (AudioWorklet 16kHz mono PCM)
  → WebSocket /ws/voice/{name}
    → VoiceSTT (faster-whisper)
      → ProcessSupervisor IPC → Anima 채팅 (기존 파이프라인)
    → StreamingSentenceSplitter (문장 분할)
      → TTS Provider (VOICEVOX / SBV2 / ElevenLabs)
    ← audio binary + JSON 제어 메시지
  ← VoicePlayback (Web Audio API)
```

음성 채팅은 기존의 텍스트 채팅 파이프라인을 경유합니다. Anima 입장에서는 일반 채팅 메시지와 동일하게 처리됩니다 (STT로 변환된 텍스트가 전달됨).

---

## 의존성 및 설치

### STT (음성 인식)

`faster-whisper`가 필요합니다:

```bash
pip install faster-whisper
# 또는 pip install animaworks[transcribe]
```

최초 STT 실행 시 Whisper 모델 (기본값: `large-v3-turbo`)이 자동 다운로드됩니다.
GPU 사용 시 CUDA 지원 `ctranslate2`가 필요합니다.

### TTS (음성 합성)

TTS는 외부 서비스로 별도 기동이 필요합니다:

| 프로바이더 | 특징 | 기동 방법 | 기본 URL |
|-----------|------|----------|----------|
| **VOICEVOX** | 무료, 일본어 특화, 다수의 캐릭터 목소리 | Docker: `docker run -p 50021:50021 voicevox/voicevox_engine` | `http://localhost:50021` |
| **Style-BERT-VITS2** | 고품질, 커스텀 음성 모델 지원 | SBV2 또는 AivisSpeech Engine 기동 | `http://localhost:5000` |
| **ElevenLabs** | 클라우드 API, 다국어, 고품질 | 환경 변수 `ELEVENLABS_API_KEY` 설정 | 클라우드 (로컬 기동 불필요) |

---

## 설정

### 글로벌 설정 (config.json의 `voice` 섹션)

전체 Anima 공통의 기본 설정입니다:

```json
{
  "voice": {
    "stt_model": "large-v3-turbo",
    "stt_device": "auto",
    "stt_compute_type": "default",
    "stt_language": null,
    "stt_refine_enabled": false,
    "default_tts_provider": "voicevox",
    "audio_format": "wav",
    "voicevox": { "base_url": "http://localhost:50021" },
    "elevenlabs": { "api_key_env": "ELEVENLABS_API_KEY", "model_id": "eleven_flash_v2_5" },
    "style_bert_vits2": { "base_url": "http://localhost:5000" }
  }
}
```

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `stt_model` | `large-v3-turbo` | Whisper 모델 이름. 선택지: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `stt_device` | `auto` | `auto` (GPU 우선) / `cpu` / `cuda` |
| `stt_compute_type` | `default` | CTranslate2 양자화 유형: `default`, `int8`, `float16` |
| `stt_language` | `null` | 언어 코드 (`ja`, `en` 등). `null`이면 자동 감지 |
| `stt_refine_enabled` | `false` | STT 결과의 LLM 후처리 (활성화 시 레이턴시 1-3초 추가) |
| `default_tts_provider` | `voicevox` | 기본 TTS 프로바이더: `voicevox` / `style_bert_vits2` / `elevenlabs` |
| `audio_format` | `wav` | TTS 출력 오디오 형식 |

### Per-Anima 음성 설정 (status.json의 `voice` 섹션)

각 Anima의 `status.json`에 `voice` 키로 개별 설정이 가능합니다:

```json
{
  "voice": {
    "tts_provider": "voicevox",
    "voice_id": "3",
    "speed": 1.0,
    "pitch": 0.0,
    "extra": {}
  }
}
```

| 필드 | 설명 |
|------|------|
| `tts_provider` | 이 Anima에서 사용할 TTS 프로바이더. 미설정 시 글로벌 기본값 사용 |
| `voice_id` | 프로바이더별 목소리 ID (아래 참조) |
| `speed` | 말하기 속도 (1.0 = 표준) |
| `pitch` | 피치 (0.0 = 표준) |
| `extra` | 프로바이더별 추가 파라미터 (선택). ElevenLabs에서는 `model_id`로 모델 오버라이드 가능 |

#### voice_id 지정 방법

| 프로바이더 | voice_id 형식 | 확인 방법 |
|-----------|---------------|----------|
| VOICEVOX | 화자 ID (숫자 문자열), 예: `"3"` = 즌다몬 | `curl http://localhost:50021/speakers`로 목록 확인 |
| Style-BERT-VITS2 | `model_id:speaker_id` 또는 `model_id:speaker_id:style`. 예: `0:0` | `curl http://localhost:5000/models/info`로 목록 확인 |
| ElevenLabs | voice_id 문자열 | ElevenLabs 대시보드 또는 API로 확인 |

미설정이거나 `voice_id`가 비어 있으면 프로바이더의 기본 목소리가 사용됩니다.

---

## WebSocket 프로토콜

엔드포인트: `ws://HOST/ws/voice/{anima_name}`

### 인증

연결 시 다음 중 하나로 인증됩니다:

- **local_trust 모드**: 인증 불필요
- **trust_localhost 활성화 + localhost 연결**: 루프백 주소에서의 연결은 인증 불필요 (CSRF 검증 있음)
- **그 외**: `session_token` 쿠키로 검증. 유효하지 않으면 4001 Unauthorized로 종료

WebSocket은 HTTP Upgrade 시 쿠키를 전송하므로, 로그인된 브라우저에서는 자동으로 인증됩니다.

### 연결 제한

- **1 Anima = 1 활성 세션**: 동일 Anima에 대한 새 연결로 기존 세션이 교체됩니다 (기존 측은 4000 "Replaced by new session"으로 종료)
- **잘못된 name**: `name`에 `/`나 `..`이 포함되거나 비어 있거나 `.`으로 시작하면 4000 "Invalid anima name"으로 종료

### 클라이언트 → 서버

| 유형 | 형식 | 설명 |
|------|------|------|
| 오디오 데이터 | binary | 16kHz mono 16-bit PCM 바이너리 |
| `{"type": "speech_end"}` | JSON | 발화 종료 알림 → STT 실행 트리거 |
| `{"type": "interrupt"}` | JSON | TTS 재생 중단 (barge-in) |
| `{"type": "config", "vad_mode": "ptt"}` 또는 `{"type": "config", "vad_mode": "vad"}` | JSON | 선택 사항. 모드 전환 알림 (서버에서는 현재 처리하지 않음) |

### 서버 → 클라이언트

| 유형 | 형식 | 설명 |
|------|------|------|
| `{"type": "status", "state": "loading"}` | JSON | 세션 초기화 중 (STT 로딩) |
| `{"type": "status", "state": "ready"}` | JSON | 세션 준비 완료 |
| `{"type": "transcript", "text": "..."}` | JSON | STT 결과 텍스트 |
| `{"type": "response_start"}` | JSON | Anima 응답 스트림 시작 |
| `{"type": "response_text", "text": "...", "done": false}` | JSON | Anima 응답 텍스트 (청크) |
| `{"type": "response_done", "emotion": "..."}` | JSON | Anima 응답 완료 (emotion 메타데이터 포함) |
| `{"type": "thinking_status", "thinking": true/false}` | JSON | 확장 사고 (thinking) 시작/종료 |
| `{"type": "thinking_delta", "text": "..."}` | JSON | 확장 사고 텍스트 청크 |
| TTS 오디오 데이터 | binary | TTS 오디오 바이너리 |
| `{"type": "tts_start"}` | JSON | TTS 오디오 전송 시작 |
| `{"type": "tts_error", "message": "..."}` | JSON | TTS 합성 실패 시 (tts_done 직전에 전송) |
| `{"type": "tts_done"}` | JSON | TTS 오디오 전송 완료 |
| `{"type": "error", "message": "..."}` | JSON | 오류 알림 |

---

## 프론트엔드 UI

대시보드와 워크스페이스의 양쪽 채팅 화면에 마이크 버튼이 표시됩니다.

연결 후 서버에서 `status: loading` → `status: ready`가 전송되며, 준비 완료 후 음성 입력을 시작할 수 있습니다.

### 음성 입력 모드

| 모드 | 조작 | 설명 |
|------|------|------|
| **PTT (Push-to-Talk)** | 마이크 버튼 길게 누름 → 놓기 | 확실한 제어. 누르는 동안만 녹음 |
| **VAD (Voice Activity Detection)** | 자동 | 말하기 시작을 자동 감지하여 녹음 시작, 침묵 시 자동 전송 |

UI의 토글로 전환 가능합니다.

### 기능

- **볼륨 컨트롤**: TTS 재생 볼륨 조절 슬라이더
- **TTS 인디케이터**: Anima가 말하는 동안의 시각적 피드백 (녹음 인디케이터와 별도)
- **끼어들기 (Barge-in)**: TTS 재생 중에 말하기 시작하면 Anima 음성을 자동 중단

---

## TTS 프로바이더 상세

### VOICEVOX

- 무료 오픈소스 일본어 음성 합성 엔진
- 50개 이상의 캐릭터 목소리
- 로컬 실행 (인터넷 불필요)
- Docker: `docker run -p 50021:50021 voicevox/voicevox_engine`
- GPU: `docker run --gpus all -p 50021:50021 voicevox/voicevox_engine`

### Style-BERT-VITS2 / AivisSpeech

- 고품질 일본어 음성 합성
- 커스텀 음성 모델의 학습 및 사용 가능
- AivisSpeech Engine은 SBV2 호환의 간편 설치 버전
- 로컬 실행

### ElevenLabs

- 클라우드 기반 다국어 음성 합성 API
- 고품질, 자연스러운 음성
- API 키 필요 (`ELEVENLABS_API_KEY` 환경 변수)
- 종량 과금

---

## 트러블슈팅

### STT가 동작하지 않음

- `faster-whisper`가 설치되어 있는지 확인: `pip show faster-whisper`
- GPU 사용 시 `ctranslate2`의 CUDA 버전이 맞는지 확인
- `stt_device: "cpu"`로 전환하여 CPU 모드로 시도

### TTS가 오디오를 반환하지 않음

- TTS 프로바이더가 기동되어 있는지 확인
  - VOICEVOX: `curl http://localhost:50021/speakers`에서 응답이 있는지
  - SBV2: `curl http://localhost:5000/models/info`에서 응답이 있는지
  - ElevenLabs: `ELEVENLABS_API_KEY` 환경 변수가 설정되어 있는지
- 서버 로그에 `TTS unavailable` 오류가 없는지 확인
- 클라이언트에서 `tts_error` 메시지가 도착하지 않았는지 확인 (TTS 합성 실패 시)
- TTS를 사용할 수 없으면 텍스트 응답만 반환됩니다 (오디오 없음)

### 오디오가 끊기거나 레이턴시가 큼

- 네트워크 대역폭 확인 (음성 스트리밍은 리얼타임)
- `stt_model`을 보다 경량의 모델(`base`, `small`)로 변경
- `stt_refine_enabled: false`를 확인 (LLM 후처리는 레이턴시 증가)
- VOICEVOX/SBV2를 GPU 모드로 기동

### voice_id가 잘못되어 오디오가 나오지 않음

- 지정한 `voice_id`가 프로바이더에 존재하는지 확인
- 잘못된 경우 프로바이더의 기본 목소리로 폴백 + 경고 로그
- VOICEVOX: `curl http://localhost:50021/speakers | jq`로 유효한 ID 확인
- SBV2: `curl http://localhost:5000/models/info`로 모델 목록을 확인하고 `model_id:speaker_id` 형식으로 지정

---

## 기술적 보충

### VoiceSession 내부 동작

`core/voice/session.py`가 하나의 음성 대화 세션을 관리합니다:

1. **오디오 버퍼 관리**: 최대 60초분 (16kHz x 16bit x 60초 = 약 1.9MB). 오버플로우 시 클리어
2. **최소 발화 필터**: 0.35초 미만의 오디오, 무음 (RMS 임계값 이하)은 폐기
3. **STT 실행**: `speech_end` 수신 시 버퍼를 일괄 변환
4. **음성 모드 부가**: 메시지 끝에 `[voice-mode: 음성 대화입니다. 구어체로 200자 이내로 간결하게...]`를 부가하여 Anima에 전달
5. **채팅 연동**: ProcessSupervisor IPC 경유로 기존 채팅 파이프라인에 텍스트 전송
6. **TTS 전 정리**: Markdown 기법(제목, 굵게, 목록, 코드 블록 등)을 제거 후 TTS에 전달
7. **스트리밍 응답**: Anima 응답 텍스트를 문장 단위로 분할 (구두점: 。！？!?; 줄바꿈)
8. **문장 단위 TTS**: 분할된 문장마다 TTS를 실행하여 First-byte latency를 최소화
9. **TTS 헬스 체크**: 최초 호출 시 프로바이더 가용성 확인. 성공 시에만 캐시, 실패 시 다음번에 재확인. 연속 3회 TTS 합성 실패 시 캐시를 무효화
10. **동시 처리 가드**: 여러 `speech_end`가 병렬로 처리되는 것을 방지

IPC 스트림 타임아웃은 기본 60초입니다. `config.json`의 `server.ipc_stream_timeout`으로 오버라이드 가능합니다.

### 문장 분할 (StreamingSentenceSplitter)

스트리밍 텍스트의 문장 분할 (`core/voice/sentence_splitter.py`):
- 문장 부호(。！？!?) 직후에 즉시 분할
- 줄바꿈에서도 분할
- 불완전한 문장은 버퍼링하여 유지

### Barge-in (끼어들기)

TTS 재생 중에 사용자가 말하기 시작한 경우:
1. 클라이언트가 `{"type": "interrupt"}`를 전송
2. 서버가 진행 중인 TTS 처리를 중단
3. 클라이언트가 재생 큐를 클리어
4. 새로운 사용자 발화의 처리를 시작
