/**
 * VoiceManager — Orchestrates voice chat (WebSocket + AudioWorklet + VAD + Playback).
 */
import { VoicePlayback } from './voice-playback.js';
import { VoiceVAD } from './voice-vad.js';

export class VoiceManager {
  constructor() {
    this._ws = null;
    this._animaName = null;
    this._mode = 'ptt'; // 'ptt' or 'vad'
    this._recording = false;
    this._connected = false;
    this._startingRecording = false;
    this._pendingStop = false;
    this._audioContext = null;
    this._workletNode = null;
    this._mediaStream = null;
    this._playback = new VoicePlayback();
    this._playback.onPlaybackEnd = () => {
      if (this._ttsPlaying) {
        this._ttsPlaying = false;
        this._emit('playbackEnd');
      }
    };
    this._vad = null;
    this._ttsPlaying = false;
    this._listeners = {};
    this._reconnectTimer = null;
    this._reconnectAttempts = 0;
    this._maxReconnectAttempts = 5;
    this._connGen = 0; // connection generation to ignore stale WS events
  }

  on(event, fn) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(fn);
  }

  off(event, fn) {
    if (!this._listeners[event]) return;
    this._listeners[event] = this._listeners[event].filter((f) => f !== fn);
  }

  _emit(event, data) {
    (this._listeners[event] || []).forEach((f) => f(data));
  }

  connect(animaName) {
    this.disconnect();
    this._animaName = animaName;
    const gen = ++this._connGen;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/voice/${encodeURIComponent(animaName)}`;

    return new Promise((resolve, reject) => {
      let settled = false;
      this._ws = new WebSocket(url);
      this._ws.binaryType = 'arraybuffer';

      this._ws.onopen = () => {
        if (gen !== this._connGen) return;
        settled = true;
        this._connected = true;
        this._reconnectAttempts = 0;
        this._emit('connected');
        resolve();
      };
      this._ws.onclose = (e) => {
        if (gen !== this._connGen) return;
        this._connected = false;
        if (!settled) {
          settled = true;
          reject(new Error('WebSocket closed before open'));
          return;
        }
        this._emit('disconnected', { code: e.code });
        this._tryReconnect();
      };
      this._ws.onerror = () => {
        if (gen !== this._connGen) return;
        this._emit('error', { message: 'WebSocket error' });
      };
      this._ws.onmessage = (e) => {
        if (gen !== this._connGen) return;
        this._handleMessage(e);
      };
    });
  }

  disconnect() {
    this._stopRecordingInternal();
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._connected = false;
    this._animaName = null;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._playback.destroy();
    this._playback = new VoicePlayback();
    this._playback.onPlaybackEnd = () => {
      if (this._ttsPlaying) {
        this._ttsPlaying = false;
        this._emit('playbackEnd');
      }
    };
    if (this._vad) {
      this._vad.destroy();
      this._vad = null;
    }
  }

  async startRecording() {
    if (this._recording || this._startingRecording || !this._connected) return;

    if (this._ttsPlaying) {
      this.interrupt();
    }

    this._startingRecording = true;
    this._pendingStop = false;

    try {
      this._mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 48000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      if (this._pendingStop) {
        this._mediaStream.getTracks().forEach((t) => t.stop());
        this._mediaStream = null;
        this._startingRecording = false;
        this._pendingStop = false;
        return;
      }

      this._audioContext = new AudioContext({ sampleRate: 48000 });

      await this._audioContext.audioWorklet.addModule('/modules/voice-worklet.js');

      if (this._pendingStop) {
        this._audioContext.close();
        this._audioContext = null;
        this._mediaStream.getTracks().forEach((t) => t.stop());
        this._mediaStream = null;
        this._startingRecording = false;
        this._pendingStop = false;
        return;
      }

      const source = this._audioContext.createMediaStreamSource(this._mediaStream);
      this._workletNode = new AudioWorkletNode(this._audioContext, 'voice-pcm-processor', {
        processorOptions: { sampleRate: this._audioContext.sampleRate },
      });

      this._workletNode.port.onmessage = (e) => {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
          this._ws.send(e.data);
        }
      };

      const silentGain = this._audioContext.createGain();
      silentGain.gain.value = 0;
      source.connect(this._workletNode);
      this._workletNode.connect(silentGain);
      silentGain.connect(this._audioContext.destination);

      this._recording = true;
      this._startingRecording = false;
      this._emit('recordingStart');
    } catch (err) {
      this._startingRecording = false;
      this._pendingStop = false;
      this._emit('error', { message: `Microphone error: ${err.message}` });
    }
  }

  stopRecording() {
    if (this._startingRecording) {
      this._pendingStop = true;
      this._emit('recordingStop');
      return;
    }
    if (!this._recording) return;
    this._stopRecordingInternal();
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'speech_end' }));
    }
    this._emit('recordingStop');
  }

  _stopRecordingInternal() {
    this._recording = false;
    if (this._workletNode) {
      this._workletNode.disconnect();
      this._workletNode = null;
    }
    if (this._audioContext) {
      this._audioContext.close();
      this._audioContext = null;
    }
    if (this._mediaStream) {
      this._mediaStream.getTracks().forEach((t) => t.stop());
      this._mediaStream = null;
    }
  }

  interrupt() {
    this._playback.stop();
    this._ttsPlaying = false;
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    this._emit('interrupted');
  }

  setMode(mode) {
    if (mode === this._mode) return;
    this._mode = mode;
    if (mode === 'vad') {
      this._startVAD();
    } else {
      if (this._vad) this._vad.stop();
    }
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'config', vad_mode: mode }));
    }
    this._emit('modeChange', { mode });
  }

  async _startVAD() {
    if (this._vad) {
      await this._vad.start();
      return;
    }
    this._vad = new VoiceVAD({
      onSpeechStart: () => this.startRecording(),
      onSpeechEnd: () => this.stopRecording(),
    });
    await this._vad.start();
  }

  _handleMessage(event) {
    if (event.data instanceof ArrayBuffer) {
      this._playback.enqueue(event.data);
      return;
    }
    try {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'transcript':
          this._emit('transcript', { text: msg.text });
          break;
        case 'response_start':
          this._emit('responseStart');
          break;
        case 'response_text':
          this._emit('responseText', { text: msg.text, done: msg.done });
          break;
        case 'response_done':
          this._emit('responseDone', { emotion: msg.emotion });
          break;
        case 'tts_start':
          this._ttsPlaying = true;
          this._emit('ttsStart');
          break;
        case 'tts_done':
          if (this._playback.queueLength === 0 && !this._playback.isPlaying) {
            this._ttsPlaying = false;
          }
          this._emit('ttsDone');
          break;
        case 'thinking_status':
          this._emit('thinkingStatus', msg.thinking);
          break;
        case 'thinking_delta':
          this._emit('thinkingDelta', { text: msg.text });
          break;
        case 'error':
          this._emit('error', { message: msg.message });
          break;
        case 'status':
          this._emit('status', { state: msg.state });
          break;
      }
    } catch {
      // Ignore parse errors
    }
  }

  _tryReconnect() {
    if (this._reconnectAttempts >= this._maxReconnectAttempts || !this._animaName) return;
    const delay = Math.min(1000 * Math.pow(2, this._reconnectAttempts), 30000);
    this._reconnectAttempts++;
    this._reconnectTimer = setTimeout(() => this.connect(this._animaName), delay);
  }

  setVolume(v) {
    this._playback.setVolume(v);
  }

  get isConnected() {
    return this._connected;
  }
  get isRecording() {
    return this._recording;
  }
  get isTTSPlaying() {
    return this._ttsPlaying;
  }
  get mode() {
    return this._mode;
  }
}

export const voiceManager = new VoiceManager();
