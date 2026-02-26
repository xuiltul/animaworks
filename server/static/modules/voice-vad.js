/**
 * Voice Activity Detection integration using @ricky0123/vad-web.
 * Detects speech start/end and triggers recording callbacks.
 */
export class VoiceVAD {
  constructor(options = {}) {
    this._onSpeechStart = options.onSpeechStart || (() => {});
    this._onSpeechEnd = options.onSpeechEnd || (() => {});
    this._myvad = null;
    this._active = false;
  }

  async start() {
    // Dynamic import of vad-web from CDN
    if (!window.vad) {
      try {
        const module = await import(
          'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/index.js'
        );
        window.vad = module;
      } catch {
        console.warn('[VoiceVAD] Failed to load vad-web, VAD disabled');
        return false;
      }
    }

    try {
      this._myvad = await window.vad.MicVAD.new({
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
