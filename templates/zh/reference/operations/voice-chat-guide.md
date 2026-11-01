# 语音聊天（Voice Chat）指南

与 Anima 进行语音对话功能的参考。
浏览器麦克风输入 → STT（语音识别） → 聊天管道 → TTS（语音合成） → 浏览器播放。

## 架构概览

```
ブラウザ (AudioWorklet 16kHz mono PCM)
  → WebSocket /ws/voice/{name}
    → VoiceSTT (faster-whisper)
      → ProcessSupervisor IPC → Animaチャット（既存パイプライン）
    → StreamingSentenceSplitter (文分割)
      → TTS Provider (VOICEVOX / SBV2 / ElevenLabs)
    ← audio binary + JSON制御メッセージ
  ← VoicePlayback (Web Audio API)
```

语音聊天经由现有的文本聊天管道处理。从 Anima 的角度来看，它与普通的聊天消息一样被处理（经过 STT 转换后的文本会送达）。

---

## 依赖关系与安装

### STT（语音识别）

需要 `faster-whisper`:

```bash
pip install faster-whisper
# または pip install animaworks[transcribe]
```

首次执行 STT 时会自动下载 Whisper 模型（默认: `large-v3-turbo`）。
使用 GPU 时需要 CUDA 兼容的 `ctranslate2`。

### TTS（语音合成）

TTS 需要作为外部服务单独启动:

| 提供商 | 特点 | 启动方法 | 默认 URL |
|-----------|------|---------|-------------|
| **VOICEVOX** | 免费・专精日语・多种角色音色 | Docker: `docker run -p 50021:50021 voicevox/voicevox_engine` | `http://localhost:50021` |
| **Style-BERT-VITS2** | 高质量・支持自定义语音模型 | 启动 SBV2 或 AivisSpeech Engine | `http://localhost:5000` |
| **ElevenLabs** | 云端 API・多语言・高质量 | 设置环境变量 `ELEVENLABS_API_KEY` | 云端（无需本地启动） |

---

## 配置

### 全局配置（config.json 的 `voice` 部分）

所有 Anima 共用的默认配置:

```json
{
  "voice": {
    "stt_model": "large-v3-turbo",
    "stt_device": "auto",
    "stt_compute_type": "default",
    "stt_language": null,
    "stt_refine_enabled": false,
    "default_tts_provider": "voicevox",
    "voicevox": { "base_url": "http://localhost:50021" },
    "elevenlabs": { "api_key_env": "ELEVENLABS_API_KEY", "model_id": "eleven_flash_v2_5" },
    "style_bert_vits2": { "base_url": "http://localhost:5000" }
  }
}
```

| 字段 | 默认值 | 说明 |
|-----------|-----------|------|
| `stt_model` | `large-v3-turbo` | Whisper 模型名称。选项: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `stt_device` | `auto` | `auto`（优先 GPU）/ `cpu` / `cuda` |
| `stt_compute_type` | `default` | CTranslate2 量化类型: `default`, `int8`, `float16` |
| `stt_language` | `null` | 语言代码（`ja`, `en` 等）。`null` 为自动检测 |
| `stt_refine_enabled` | `false` | STT 结果的 LLM 后处理（启用后增加 1-3 秒延迟） |
| `default_tts_provider` | `voicevox` | 默认 TTS 提供商: `voicevox` / `style_bert_vits2` / `elevenlabs` |

### Per-Anima 语音配置（status.json 的 `voice` 部分）

在各 Anima 的 `status.json` 中通过 `voice` 键进行单独配置:

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

| 字段 | 说明 |
|-----------|------|
| `tts_provider` | 此 Anima 使用的 TTS 提供商。未设置时使用全局默认值 |
| `voice_id` | 提供商特有的声音 ID（后述） |
| `speed` | 语速（1.0 = 标准） |
| `pitch` | 音调（0.0 = 标准） |
| `extra` | 提供商特有的附加参数（可选）。在 ElevenLabs 中可通过 `model_id` 覆盖模型 |

#### voice_id 的指定方法

| 提供商 | voice_id 的格式 | 确认方法 |
|-----------|----------------|---------|
| VOICEVOX | 说话者 ID（数值字符串）例: `"3"` = ずんだもん | 通过 `curl http://localhost:50021/speakers` 获取列表 |
| Style-BERT-VITS2 | `model_id:speaker_id` 或 `model_id:speaker_id:style`。例: `0:0` | 通过 `curl http://localhost:5000/models/info` 获取列表 |
| ElevenLabs | voice_id 字符串 | 通过 ElevenLabs 仪表板或 API 确认 |

若无配置或 `voice_id` 为空，则使用提供商的默认声音。

---

## WebSocket 协议

端点: `ws://HOST/ws/voice/{anima_name}`

### 认证

连接时，通过以下任一方式认证:

- **local_trust 模式**: 无需认证
- **启用 trust_localhost 且为 localhost 连接**: 来自回环地址的连接无需认证（有 CSRF 验证）
- **上述以外**: 通过 `session_token` cookie 验证。无效时以 4001 Unauthorized 关闭

WebSocket 在 HTTP Upgrade 时会发送 cookie，因此已登录的浏览器会自动完成认证。

### 连接限制

- **1 个 Anima = 1 个活动会话**: 对同一 Anima 的新连接会替换现有会话（现有侧以 4000 "Replaced by new session" 关闭）
- **非法 name**: 若 `name` 中包含 `/` 或 `..`、为空、或以 `.` 开头，则以 4000 "Invalid anima name" 关闭

### 客户端 → 服务器

| 类型 | 格式 | 说明 |
|--------|------|------|
| 音频数据 | binary | 16kHz mono 16-bit PCM 二进制 |
| `{"type": "speech_end"}` | JSON | 发言结束通知 → 触发 STT 执行 |
| `{"type": "interrupt"}` | JSON | 中断 TTS 播放（barge-in） |
| `{"type": "config", "vad_mode": "ptt"}` 或 `{"type": "config", "vad_mode": "vad"}` | JSON | 可选。模式切换通知（服务器当前不处理） |

### 服务器 → 客户端

| 类型 | 格式 | 说明 |
|--------|------|------|
| `{"type": "status", "state": "loading"}` | JSON | 会话初始化中（STT 加载） |
| `{"type": "status", "state": "ready"}` | JSON | 会话准备完成 |
| `{"type": "transcript", "text": "..."}` | JSON | STT 结果文本 |
| `{"type": "response_start"}` | JSON | Anima 响应流开始 |
| `{"type": "response_text", "text": "...", "done": false}` | JSON | Anima 响应文本（分块） |
| `{"type": "response_done", "emotion": "..."}` | JSON | Anima 响应完成（附带 emotion 元数据） |
| `{"type": "thinking_status", "thinking": true/false}` | JSON | 扩展思考（thinking）开始/结束 |
| `{"type": "thinking_delta", "text": "..."}` | JSON | 扩展思考的文本分块 |
| TTS 音频数据 | binary | TTS 音频二进制 |
| `{"type": "tts_start"}` | JSON | TTS 音频发送开始 |
| `{"type": "tts_error", "message": "..."}` | JSON | TTS 合成失败时（在 tts_done 之前发送） |
| `{"type": "tts_done"}` | JSON | TTS 音频发送完成 |
| `{"type": "error", "message": "..."}` | JSON | 错误通知 |

---

## 前端 UI

仪表板和工作区的两个聊天画面都会显示麦克风按钮。

连接后，服务器会发送 `status: loading` → `status: ready`，准备完成后即可开始语音输入。

### 语音输入模式

| 模式 | 操作 | 说明 |
|--------|------|------|
| **PTT（Push-to-Talk）** | 长按麦克风按钮 → 松开 | 可靠的控制。仅在按住期间录音 |
| **VAD（Voice Activity Detection）** | 自动 | 自动检测开始说话并开始录音，沉默时自动发送 |

可通过 UI 的开关切换。

### 功能

- **音量控制**: 调整 TTS 播放音量的滑块
- **TTS 指示器**: Anima 说话期间的视觉反馈（与录音指示器分开）
- **打断（Barge-in）**: TTS 播放中开始说话时会自动中断 Anima 语音

---

## TTS 提供商详情

### VOICEVOX

- 免费・开源的日语语音合成引擎
- 50 种以上的角色声音
- 本地运行（无需互联网）
- Docker: `docker run -p 50021:50021 voicevox/voicevox_engine`
- GPU 版: `docker run --gpus all -p 50021:50021 voicevox/voicevox_engine`

### Style-BERT-VITS2 / AivisSpeech

- 高质量的日语语音合成
- 可训练・使用自定义语音模型
- AivisSpeech Engine 是 SBV2 兼容的简易安装版
- 本地运行

### ElevenLabs

- 基于云的多语言语音合成 API
- 高质量・自然的语音
- 需要 API 密钥（`ELEVENLABS_API_KEY` 环境变量）
- 按量计费

---

## 故障排查

### STT 无法工作

- 确认是否已安装 `faster-whisper`: `pip show faster-whisper`
- 使用 GPU 时确认 `ctranslate2` 的 CUDA 版本是否匹配
- 切换到 `stt_device: "cpu"` 尝试 CPU 模式

### TTS 不返回语音

- 确认 TTS 提供商是否已启动
  - VOICEVOX: 确认 `curl http://localhost:50021/speakers` 是否有响应
  - SBV2: 确认 `curl http://localhost:5000/models/info` 是否有响应
  - ElevenLabs: 确认是否已设置 `ELEVENLABS_API_KEY` 环境变量
- 确认服务器日志中是否有 `TTS unavailable` 错误
- 确认客户端是否未收到 `tts_error` 消息（TTS 合成失败时）
- TTS 不可用时，仅返回文本响应（无语音）

### 语音断续 / 延迟较大

- 确认网络带宽（语音流式传输是实时的）
- 将 `stt_model` 更改为更轻量的模型（`base`, `small`）
- 确认 `stt_refine_enabled: false`（LLM 后处理会增加延迟）
- 以 GPU 模式启动 VOICEVOX/SBV2

### voice_id 非法导致无语音输出

- 确认指定的 `voice_id` 是否存在于提供商中
- 若非法则回退到提供商的默认声音 + 警告日志
- VOICEVOX: 通过 `curl http://localhost:50021/speakers | jq` 确认有效 ID
- SBV2: 通过 `curl http://localhost:5000/models/info` 确认模型列表，并以 `model_id:speaker_id` 格式指定

---

## 技术补充

### VoiceSession 的内部行为

`core/voice/session.py` 管理一个语音对话会话:

1. **音频缓冲区管理**: 最大 60 秒（16kHz × 16bit × 60 秒 ≈ 1.9MB）。溢出时清空
2. **最小发言过滤**: 丢弃不足 0.35 秒的音频、静音（低于 RMS 阈值）
3. **STT 执行**: 收到 `speech_end` 时对缓冲区进行批量转写
4. **附加语音模式**: 在消息末尾附加 `[voice-mode: 音声会話です。話し言葉で200文字以内で簡潔に...]` 后传递给 Anima
5. **聊天联动**: 通过 ProcessSupervisor IPC 将文本发送到现有聊天管道
6. **TTS 前净化**: 去除 Markdown 记法（标题・粗体・列表・代码块等）后再传给 TTS
7. **流式响应**: 将 Anima 响应文本按句子分割（标点: 。！？、换行）
8. **按句 TTS**: 对分割后的每个句子执行 TTS，以最小化首字节延迟
9. **TTS 健康检查**: 首次调用时确认提供商的可用性。仅在成功时缓存，失败时下次重新检查。连续 3 次 TTS 合成失败则使缓存失效
10. **并发处理保护**: 防止同时处理多个 `speech_end`

IPC 流的超时默认为 60 秒。可通过 `config.json` 的 `server.ipc_stream_timeout` 覆盖。

### 句子分割（StreamingSentenceSplitter）

日语文本的流式句子分割（`core/voice/sentence_splitter.py`）:
- 在标点（。！？!?）之后立即分割
- 换行也会分割
- 缓冲并保留不完整的句子

### Barge-in（打断）

当用户在 TTS 播放中开始说话时:
1. 客户端发送 `{"type": "interrupt"}`
2. 服务器中断进行中的 TTS 处理
3. 客户端清空播放队列
4. 开始处理新的用户发言
