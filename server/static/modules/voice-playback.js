/**
 * Audio playback queue for voice chat TTS output.
 * Decodes received audio chunks and plays them sequentially.
 */
export class VoicePlayback {
  constructor() {
    this._ctx = null;
    this._queue = [];
    this._playing = false;
    this._currentSource = null;
    this._gainNode = null;
    this._volume = 1.0;
    this._onPlaybackStart = null;
    this._onPlaybackEnd = null;
  }

  _ensureContext() {
    if (this._ctx && this._ctx.state === 'closed') {
      this._ctx = null;
      this._gainNode = null;
    }
    if (!this._ctx) {
      this._ctx = new AudioContext();
      this._gainNode = this._ctx.createGain();
      this._gainNode.gain.value = this._volume;
      this._gainNode.connect(this._ctx.destination);
    }
    if (this._ctx.state === 'suspended') {
      this._ctx.resume();
    }
  }

  async enqueue(audioData) {
    // audioData is ArrayBuffer (wav or mp3)
    this._ensureContext();
    if (this._ctx.state === 'suspended') {
      await this._ctx.resume();
    }
    try {
      const buffer = await this._ctx.decodeAudioData(audioData.slice(0));
      this._queue.push(buffer);
      if (!this._playing) this._playNext();
    } catch (err) {
      console.warn('[VoicePlayback] Failed to decode audio:', err);
    }
  }

  _playNext() {
    if (this._queue.length === 0) {
      this._playing = false;
      if (this._onPlaybackEnd) this._onPlaybackEnd();
      return;
    }
    this._playing = true;
    if (this._onPlaybackStart && this._queue.length === 1) {
      this._onPlaybackStart();
    }
    const buffer = this._queue.shift();
    const source = this._ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this._gainNode);
    this._currentSource = source;
    source.onended = () => {
      this._currentSource = null;
      this._playNext();
    };
    source.start(0);
  }

  stop() {
    this._queue = [];
    if (this._currentSource) {
      try {
        this._currentSource.stop();
      } catch (_) {
        // Ignore if already stopped
      }
      this._currentSource = null;
    }
    this._playing = false;
  }

  setVolume(v) {
    this._volume = Math.max(0, Math.min(1, v));
    if (this._gainNode) this._gainNode.gain.value = this._volume;
  }

  get isPlaying() {
    return this._playing;
  }

  get queueLength() {
    return this._queue.length;
  }

  set onPlaybackStart(fn) {
    this._onPlaybackStart = fn;
  }
  set onPlaybackEnd(fn) {
    this._onPlaybackEnd = fn;
  }

  destroy() {
    this.stop();
    if (this._ctx) {
      this._ctx.close();
      this._ctx = null;
    }
  }
}
