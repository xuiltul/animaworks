/**
 * Audio playback queue for voice chat TTS output.
 * Decodes received audio chunks and plays them sequentially.
 *
 * Once RTC loopback AEC is up, output goes
 * gain → MediaStreamDestination → RTCPeerConnection pair → <audio srcObject>,
 * so Chrome-family AEC can use TTS as the echo reference.
 */
export class VoicePlayback {
  constructor() {
    this._ctx = null;
    this._queue = [];
    this._playing = false;
    this._paused = false;
    this._currentSource = null;
    this._gainNode = null;
    this._volume = 1.0;
    this._onPlaybackStart = null;
    this._onPlaybackEnd = null;
    this._onCaption = null;
    this._aecActive = false;
    this._aecFailed = false;
    this._aecStarting = false;
    this._msDest = null;
    this._pc1 = null;
    this._pc2 = null;
    this._aecAudio = null;
    this._analyser = null;
  }

  _ensureContext() {
    if (this._ctx && this._ctx.state === 'closed') {
      this._teardownAec();
      this._ctx = null;
      this._gainNode = null;
    }
    if (!this._ctx) {
      this._ctx = new AudioContext();
      this._gainNode = this._ctx.createGain();
      this._gainNode.gain.value = this._volume;
      // RMS tap for lip-sync: gain -> analyser (+ destination downstream).
      this._analyser = this._ctx.createAnalyser();
      this._analyser.fftSize = 512;
      this._analyser.smoothingTimeConstant = 0.5;
      this._gainNode.connect(this._analyser);
      this._gainNode.connect(this._ctx.destination);
      this._startAecLoopback();
    }
    if (this._ctx.state === 'suspended' && !this._paused) {
      this._ctx.resume();
    }
  }

  get aecActive() {
    return this._aecActive;
  }

  async _startAecLoopback() {
    if (this._aecFailed || this._aecActive || this._aecStarting) return;
    this._aecStarting = true;
    const ctx = this._ctx;
    const res = { pc1: null, pc2: null, msDest: null, audio: null };
    try {
      res.msDest = ctx.createMediaStreamDestination();
      res.pc1 = new RTCPeerConnection();
      res.pc2 = new RTCPeerConnection();
      res.pc1.onicecandidate = (e) => {
        if (e.candidate) res.pc2.addIceCandidate(e.candidate).catch(() => {});
      };
      res.pc2.onicecandidate = (e) => {
        if (e.candidate) res.pc1.addIceCandidate(e.candidate).catch(() => {});
      };

      const gotTrack = new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error('ontrack timeout')), 5000);
        res.pc2.ontrack = (e) => {
          clearTimeout(t);
          resolve(e.streams[0] || new MediaStream([e.track]));
        };
      });

      for (const track of res.msDest.stream.getAudioTracks()) {
        res.pc1.addTrack(track, res.msDest.stream);
      }

      const offer = await res.pc1.createOffer();
      await res.pc1.setLocalDescription(offer);
      await _waitIce(res.pc1);
      if (this._ctx !== ctx) throw new Error('aborted');
      await res.pc2.setRemoteDescription(res.pc1.localDescription);
      const answer = await res.pc2.createAnswer();
      await res.pc2.setLocalDescription(answer);
      await _waitIce(res.pc2);
      if (this._ctx !== ctx) throw new Error('aborted');
      await res.pc1.setRemoteDescription(res.pc2.localDescription);

      const remote = await gotTrack;
      if (this._ctx !== ctx || !this._gainNode) throw new Error('aborted');

      res.audio = document.createElement('audio');
      res.audio.style.display = 'none';
      res.audio.autoplay = true;
      res.audio.srcObject = remote;
      document.body.appendChild(res.audio);
      await res.audio.play();

      if (this._ctx !== ctx || !this._gainNode) throw new Error('aborted');

      // Switch off the direct destination so we never double-play.
      // disconnect() severs ALL connections — restore the analyser tap too.
      this._gainNode.disconnect();
      if (this._analyser) this._gainNode.connect(this._analyser);
      this._gainNode.connect(res.msDest);
      this._msDest = res.msDest;
      this._pc1 = res.pc1;
      this._pc2 = res.pc2;
      this._aecAudio = res.audio;
      this._aecActive = true;
    } catch (err) {
      _disposeAecResources(res);
      if (this._ctx !== ctx) return;
      this._aecFailed = true;
      console.warn('[VoicePlayback] AEC loopback failed, falling back to direct output:', err);
      if (this._gainNode && !this._aecActive) {
        try {
          this._gainNode.disconnect();
          if (this._analyser) this._gainNode.connect(this._analyser);
          this._gainNode.connect(this._ctx.destination);
        } catch (_) {
          // Already wired to destination, or context gone.
        }
      }
    } finally {
      this._aecStarting = false;
    }
  }

  _teardownAec() {
    this._aecActive = false;
    _disposeAecResources({
      pc1: this._pc1,
      pc2: this._pc2,
      audio: this._aecAudio,
      msDest: this._msDest,
    });
    this._pc1 = null;
    this._pc2 = null;
    this._aecAudio = null;
    this._msDest = null;
  }

  async enqueue(audioData, caption = null) {
    // audioData is ArrayBuffer (wav or mp3); caption is shown when this
    // buffer actually starts playing (playback-synced subtitle).
    this._ensureContext();
    if (this._ctx.state === 'suspended' && !this._paused) {
      await this._ctx.resume();
    }
    try {
      const buffer = await this._ctx.decodeAudioData(audioData.slice(0));
      this._queue.push({ buffer, caption });
      // While paused (barge-in probe) nothing may start — resume() drains.
      if (!this._playing && !this._paused) this._playNext();
    } catch (err) {
      console.warn('[VoicePlayback] Failed to decode audio:', err);
    }
  }

  _playNext() {
    if (this._queue.length === 0) {
      this._playing = false;
      if (this._onCaption) this._onCaption(null);
      if (this._onPlaybackEnd) this._onPlaybackEnd();
      return;
    }
    this._playing = true;
    if (this._onPlaybackStart && this._queue.length === 1) {
      this._onPlaybackStart();
    }
    const { buffer, caption } = this._queue.shift();
    if (caption != null && this._onCaption) this._onCaption(caption);
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
    this._paused = false;
    if (this._ctx && this._ctx.state === 'suspended') {
      this._ctx.resume().catch(() => {});
    }
    this._queue = [];
    if (this._onCaption) this._onCaption(null);
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

  pause() {
    this._paused = true;
    if (this._ctx && this._ctx.state === 'running') {
      return this._ctx.suspend().catch(() => {});
    }
    return Promise.resolve();
  }

  resume() {
    this._paused = false;
    const kick = () => {
      if (!this._playing && this._queue.length > 0) this._playNext();
    };
    if (this._ctx && this._ctx.state === 'suspended') {
      return this._ctx.resume().catch(() => {}).then(kick);
    }
    kick();
    return Promise.resolve();
  }

  setVolume(v) {
    this._volume = Math.max(0, Math.min(1, v));
    if (this._gainNode) this._gainNode.gain.value = this._volume;
  }

  get isPlaying() {
    return this._playing;
  }

  get isPaused() {
    return this._paused;
  }

  /**
   * 正規化RMS(0..1) of the currently playing TTS output.
   * Returns 0 when the audio context is not available.
   */
  get rms() {
    if (this._paused || !this._ctx || !this._analyser || this._ctx.state !== "running") return 0;
    const data = new Float32Array(this._analyser.fftSize);
    this._analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    const raw = Math.sqrt(sum / data.length);
    // Scale so loud speech (raw RMS ~0.15-0.3) reaches the mouth thresholds.
    return Math.min(1, raw * 3);
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
  set onCaption(fn) {
    this._onCaption = fn;
  }

  destroy() {
    this.stop();
    this._teardownAec();
    if (this._ctx) {
      this._ctx.close();
      this._ctx = null;
    }
    this._gainNode = null;
    this._analyser = null;
  }
}

function _waitIce(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      pc.removeEventListener('icegatheringstatechange', onChange);
      resolve();
    };
    const t = setTimeout(done, 2000);
    const onChange = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(t);
        done();
      }
    };
    pc.addEventListener('icegatheringstatechange', onChange);
  });
}

function _disposeAecResources(res) {
  if (!res) return;
  if (res.pc1) {
    try { res.pc1.close(); } catch (_) { /* already closed */ }
  }
  if (res.pc2) {
    try { res.pc2.close(); } catch (_) { /* already closed */ }
  }
  if (res.audio) {
    try {
      res.audio.pause();
      res.audio.srcObject = null;
      res.audio.remove();
    } catch (_) { /* already removed */ }
  }
  if (res.msDest) {
    try { res.msDest.disconnect(); } catch (_) { /* already disconnected */ }
    try { res.msDest.stream.getTracks().forEach((t) => t.stop()); } catch (_) { /* no tracks */ }
  }
}
