/**
 * AudioWorklet processor for voice chat.
 * Resamples input audio to 16kHz mono and outputs 16-bit PCM chunks.
 */
class VoicePCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this._targetRate = 16000;
    this._inputRate = options.processorOptions?.sampleRate || 48000;
    this._ratio = this._inputRate / this._targetRate;
    this._buffer = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // mono channel
    // Simple decimation resampling
    for (let i = 0; i < samples.length; i++) {
      this._buffer.push(samples[i]);
    }

    // Resample and send chunks
    const targetSamples = Math.floor(this._buffer.length / this._ratio);
    if (targetSamples > 0) {
      const pcm16 = new Int16Array(targetSamples);
      for (let i = 0; i < targetSamples; i++) {
        const srcIdx = Math.round(i * this._ratio);
        const sample = Math.max(-1, Math.min(1, this._buffer[srcIdx] || 0));
        pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      // Remove consumed samples
      this._buffer = this._buffer.slice(Math.round(targetSamples * this._ratio));
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor('voice-pcm-processor', VoicePCMProcessor);
