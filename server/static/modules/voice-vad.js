/**
 * Voice Activity Detection integration using @ricky0123/vad-web.
 * Detects speech start/end and triggers recording callbacks.
 *
 * Loads onnxruntime-web + vad-web bundle via <script> tags on first use
 * (dynamic import() doesn't work with vad-web's CommonJS dist).
 */

const _ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const _VAD_CDN = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/';

let _scriptsLoaded = false;
let _scriptsLoading = null;

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = resolve;
    el.onerror = () => reject(new Error(`Failed to load: ${src}`));
    document.head.appendChild(el);
  });
}

async function _ensureScripts() {
  if (_scriptsLoaded && window.vad) return;
  if (_scriptsLoading) return _scriptsLoading;
  _scriptsLoading = (async () => {
    await _loadScript(`${_ORT_CDN}ort.wasm.min.js`);
    await _loadScript(`${_VAD_CDN}bundle.min.js`);
    if (!window.vad) throw new Error('window.vad not found after loading scripts');
    _scriptsLoaded = true;
  })();
  return _scriptsLoading;
}

export class VoiceVAD {
  constructor(options = {}) {
    this._onSpeechStart = options.onSpeechStart || (() => {});
    this._onSpeechRealStart = options.onSpeechRealStart || (() => {});
    this._onSpeechEnd = options.onSpeechEnd || (() => {});
    this._onFrameProcessed = options.onFrameProcessed || (() => {});
    // vad-web fires onVADMisfire *instead of* onSpeechEnd when the utterance
    // was shorter than minSpeechMs — without it a noise blip leaves the
    // caller's recording running forever.
    this._onMisfire = options.onMisfire || (() => {});
    this._positiveSpeechThreshold = options.positiveSpeechThreshold;
    this._negativeSpeechThreshold = options.negativeSpeechThreshold;
    this._minSpeechMs = options.minSpeechMs;
    this._redemptionMs = options.redemptionMs;
    this._preSpeechPadMs = options.preSpeechPadMs;
    this._model = options.model;
    this._getStream = options.getStream;
    this._pauseStream = options.pauseStream;
    this._resumeStream = options.resumeStream;
    this._myvad = null;
    this._startPromise = null;
    this._active = false;
  }

  async start() {
    if (this._startPromise) return this._startPromise;
    this._startPromise = this._start().finally(() => {
      this._startPromise = null;
    });
    return this._startPromise;
  }

  async _start() {
    if (this._myvad) {
      this._active = true;
      await this._myvad.start();
      return true;
    }

    try {
      await _ensureScripts();
    } catch (err) {
      console.warn('[VoiceVAD] Failed to load vad-web:', err.message);
      return false;
    }

    try {
      const vadOpts = {
        onnxWASMBasePath: _ORT_CDN,
        baseAssetPath: _VAD_CDN,
        onSpeechStart: () => {
          if (this._active) this._onSpeechStart();
        },
        onSpeechRealStart: () => {
          if (this._active) this._onSpeechRealStart();
        },
        onSpeechEnd: (audio) => {
          if (this._active) this._onSpeechEnd(audio);
        },
        onVADMisfire: () => {
          if (this._active) this._onMisfire();
        },
        onFrameProcessed: (probabilities) => {
          if (this._active) this._onFrameProcessed(probabilities);
        },
      };
      if (this._positiveSpeechThreshold != null) {
        vadOpts.positiveSpeechThreshold = this._positiveSpeechThreshold;
      }
      if (this._negativeSpeechThreshold != null) {
        vadOpts.negativeSpeechThreshold = this._negativeSpeechThreshold;
      }
      if (this._minSpeechMs != null) vadOpts.minSpeechMs = this._minSpeechMs;
      if (this._redemptionMs != null) vadOpts.redemptionMs = this._redemptionMs;
      if (this._preSpeechPadMs != null) vadOpts.preSpeechPadMs = this._preSpeechPadMs;
      if (this._model != null) vadOpts.model = this._model;
      if (this._getStream) vadOpts.getStream = this._getStream;
      if (this._pauseStream) vadOpts.pauseStream = this._pauseStream;
      if (this._resumeStream) vadOpts.resumeStream = this._resumeStream;
      this._myvad = await window.vad.MicVAD.new(vadOpts);
      this._active = true;
      await this._myvad.start();
      return true;
    } catch (err) {
      this._active = false;
      console.warn('[VoiceVAD] Failed to initialize:', err);
      return false;
    }
  }

  stop() {
    this._active = false;
    if (this._myvad) {
      this._myvad.pause();
    }
  }

  destroy() {
    this._active = false;
    if (this._myvad) {
      this._myvad.destroy();
      this._myvad = null;
    }
  }
}
