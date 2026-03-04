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
    this._onSpeechEnd = options.onSpeechEnd || (() => {});
    this._myvad = null;
    this._active = false;
  }

  async start() {
    if (this._myvad) {
      this._active = true;
      this._myvad.start();
      return true;
    }

    try {
      await _ensureScripts();
    } catch (err) {
      console.warn('[VoiceVAD] Failed to load vad-web:', err.message);
      return false;
    }

    try {
      this._myvad = await window.vad.MicVAD.new({
        onnxWASMBasePath: _ORT_CDN,
        baseAssetPath: _VAD_CDN,
        onSpeechStart: () => {
          if (this._active) this._onSpeechStart();
        },
        onSpeechEnd: (audio) => {
          if (this._active) this._onSpeechEnd(audio);
        },
      });
      this._myvad.start();
      this._active = true;
      return true;
    } catch (err) {
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
