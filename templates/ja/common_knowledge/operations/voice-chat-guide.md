# 音声チャット（Voice Chat）ガイド

Animaとの音声会話機能のリファレンス。
ブラウザのマイク入力 → STT（音声認識） → チャットパイプライン → TTS（音声合成） → ブラウザ再生。

## アーキテクチャ概要

```
ブラウザ (AudioWorklet 16kHz PCM)
  → WebSocket /ws/voice/{name}
    → VoiceSTT (faster-whisper)
      → ProcessSupervisor IPC → Animaチャット（既存パイプライン）
    → StreamingSentenceSplitter (文分割)
      → TTS Provider (VOICEVOX / SBV2 / ElevenLabs)
    ← audio binary + JSON制御メッセージ
  ← VoicePlayback (Web Audio API)
```

音声チャットは既存のテキストチャットパイプラインを経由する。Animaから見ると通常のチャットメッセージと同じように処理される（STTで変換済みテキストが届く）。

---

## 依存関係とインストール

### STT（音声認識）

`faster-whisper` が必要:

```bash
pip install faster-whisper
```

初回STT実行時にWhisperモデル（デフォルト: `large-v3-turbo`）が自動ダウンロードされる。
GPU使用時は CUDA 対応の `ctranslate2` が必要。

### TTS（音声合成）

TTSは外部サービスとして別途起動が必要:

| プロバイダ | 特徴 | 起動方法 | デフォルトURL |
|-----------|------|---------|-------------|
| **VOICEVOX** | 無料・日本語特化・多数のキャラ声 | Docker: `docker run -p 50021:50021 voicevox/voicevox_engine` | `http://localhost:50021` |
| **Style-BERT-VITS2** | 高品質・カスタム音声モデル対応 | SBV2 または AivisSpeech Engine を起動 | `http://localhost:5000` |
| **ElevenLabs** | クラウドAPI・多言語・高品質 | 環境変数 `ELEVENLABS_API_KEY` を設定 | クラウド（ローカル起動不要） |

---

## 設定

### グローバル設定（config.json の `voice` セクション）

全Anima共通のデフォルト設定:

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

| フィールド | デフォルト | 説明 |
|-----------|-----------|------|
| `stt_model` | `large-v3-turbo` | Whisperモデル名。選択肢: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `stt_device` | `auto` | `auto`（GPU優先）/ `cpu` / `cuda` |
| `stt_compute_type` | `default` | CTranslate2量子化タイプ: `default`, `int8`, `float16` |
| `stt_language` | `null` | 言語コード（`ja`, `en`等）。`null` で自動検出 |
| `stt_refine_enabled` | `false` | STT結果のLLM後処理（有効化でレイテンシ1-3秒追加） |
| `default_tts_provider` | `voicevox` | デフォルトTTSプロバイダ: `voicevox` / `style_bert_vits2` / `elevenlabs` |
| `audio_format` | `wav` | TTS出力音声形式 |

### Per-Anima音声設定（status.json の `voice` セクション）

各Animaの `status.json` に `voice` キーで個別設定:

```json
{
  "voice": {
    "tts_provider": "voicevox",
    "voice_id": "3",
    "speed": 1.0,
    "pitch": 0.0
  }
}
```

| フィールド | 説明 |
|-----------|------|
| `tts_provider` | このAnimaで使用するTTSプロバイダ。未設定時はグローバルデフォルト |
| `voice_id` | プロバイダ固有の声ID（後述） |
| `speed` | 話速（1.0 = 標準） |
| `pitch` | ピッチ（0.0 = 標準） |

#### voice_id の指定方法

| プロバイダ | voice_id の形式 | 確認方法 |
|-----------|----------------|---------|
| VOICEVOX | 話者ID（数値文字列）例: `"3"` = ずんだもん | `curl http://localhost:50021/speakers` で一覧取得 |
| Style-BERT-VITS2 | モデル名 | SBV2サーバーのAPIで確認 |
| ElevenLabs | voice_id文字列 | ElevenLabsダッシュボードまたはAPIで確認 |

設定がない場合や `voice_id` が空の場合、プロバイダのデフォルト声が使用される。

---

## WebSocketプロトコル

エンドポイント: `ws://HOST/ws/voice/{anima_name}`

### 認証

接続後、最初に認証メッセージを送信:
```json
{"type": "auth", "token": "SESSION_TOKEN"}
```
localhost信頼モード（Hybrid Localhost Trust）が有効な場合は不要。

### クライアント → サーバー

| タイプ | 形式 | 説明 |
|--------|------|------|
| 音声データ | binary | 16kHz mono 16-bit PCM バイナリ |
| `{"type": "speech_end"}` | JSON | 発話終了通知 → STT実行トリガー |
| `{"type": "interrupt"}` | JSON | TTS再生中断（barge-in） |
| `{"type": "config", ...}` | JSON | 設定変更（mode切替等） |

### サーバー → クライアント

| タイプ | 形式 | 説明 |
|--------|------|------|
| `{"type": "transcript", "text": "..."}` | JSON | STT結果テキスト |
| `{"type": "response_text", "text": "..."}` | JSON | Anima応答テキスト（チャンク） |
| TTS音声データ | binary | TTS音声バイナリ |
| `{"type": "tts_start"}` | JSON | TTS音声送信開始 |
| `{"type": "tts_done"}` | JSON | TTS音声送信完了 |
| `{"type": "error", "message": "..."}` | JSON | エラー通知 |

---

## フロントエンドUI

ダッシュボードとワークスペースの両チャット画面にマイクボタンが表示される。

### 音声入力モード

| モード | 操作 | 説明 |
|--------|------|------|
| **PTT（Push-to-Talk）** | マイクボタン長押し → 離す | 確実な制御。押している間だけ録音 |
| **VAD（Voice Activity Detection）** | 自動 | 話し始めを自動検出して録音開始、沈黙で自動送信 |

UIのトグルで切替可能。

### 機能

- **音量コントロール**: TTS再生音量の調整スライダー
- **TTSインジケータ**: Animaが話している間の視覚的フィードバック（録音インジケータとは別）
- **割り込み（Barge-in）**: TTS再生中に話し始めるとAnima音声を自動中断

---

## TTSプロバイダの詳細

### VOICEVOX

- 無料・オープンソースの日本語音声合成エンジン
- 50以上のキャラクター声
- ローカル実行（インターネット不要）
- Docker: `docker run -p 50021:50021 voicevox/voicevox_engine`
- GPU版: `docker run --gpus all -p 50021:50021 voicevox/voicevox_engine`

### Style-BERT-VITS2 / AivisSpeech

- 高品質な日本語音声合成
- カスタム音声モデルの学習・利用が可能
- AivisSpeech Engine は SBV2 互換の簡易インストール版
- ローカル実行

### ElevenLabs

- クラウドベースの多言語音声合成API
- 高品質・自然な音声
- API キーが必要（`ELEVENLABS_API_KEY` 環境変数）
- 従量課金

---

## トラブルシューティング

### STTが動作しない

- `faster-whisper` がインストールされているか確認: `pip show faster-whisper`
- GPU使用時は `ctranslate2` の CUDA バージョンが合っているか確認
- `stt_device: "cpu"` に切り替えてCPUモードで試す

### TTSが音声を返さない

- TTSプロバイダが起動しているか確認
  - VOICEVOX: `curl http://localhost:50021/speakers` でレスポンスがあるか
  - SBV2: `curl http://localhost:5000/voice/speakers` でレスポンスがあるか
  - ElevenLabs: `ELEVENLABS_API_KEY` 環境変数が設定されているか
- サーバーログに `TTS unavailable` エラーがないか確認
- TTSが不可用の場合、テキスト応答のみが返される（音声なし）

### 音声が途切れる / レイテンシが大きい

- ネットワーク帯域を確認（音声ストリーミングはリアルタイム）
- `stt_model` をより軽量なモデル（`base`, `small`）に変更
- `stt_refine_enabled: false` を確認（LLM後処理はレイテンシ増加）
- VOICEVOX/SBV2 をGPUモードで起動

### voice_id が不正で音声が出ない

- 指定した `voice_id` がプロバイダに存在するか確認
- 不正な場合はプロバイダのデフォルト声にフォールバック + 警告ログ
- VOICEVOX: `curl http://localhost:50021/speakers | jq` で有効なIDを確認

---

## 技術的な補足

### VoiceSession の内部動作

`core/voice/session.py` が1つの音声会話セッションを管理:

1. **音声バッファ管理**: 最大60秒分（16kHz × 16bit × 60秒 ≈ 1.9MB）。オーバーフロー時はクリア
2. **STT実行**: `speech_end` 受信時にバッファを一括転写
3. **チャット連携**: ProcessSupervisor IPC経由で既存チャットパイプラインにテキスト送信
4. **ストリーミング応答**: Anima応答テキストを文単位で分割（句読点: 。！？、…）
5. **文単位TTS**: 分割された文ごとにTTSを実行してFirst-byte latencyを最小化
6. **TTSヘルスチェック**: 初回呼び出し時にプロバイダの可用性を確認（結果をキャッシュ）
7. **並行処理ガード**: 同時に複数の `speech_end` が処理されるのを防止

### 文分割（StreamingSentenceSplitter）

日本語テキストのストリーミング文分割:
- 句読点（。！？）で即座に分割
- 読点（、）や省略記号（…）でも分割（短い応答のレイテンシ改善）
- 改行でも分割
- バッファリングして不完全な文を保持

### Barge-in（割り込み）

TTS再生中にユーザーが話し始めた場合:
1. クライアントが `{"type": "interrupt"}` を送信
2. サーバーが進行中のTTS処理を中断
3. クライアントが再生キューをクリア
4. 新しいユーザー発話の処理を開始
