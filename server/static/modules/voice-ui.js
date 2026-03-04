/**
 * Voice UI components — mic button, recording indicator, mode toggle, volume slider.
 * Integrates into existing chat input areas and renders voice chat bubbles.
 */
import { voiceManager } from './voice.js';
import { t } from '/shared/i18n.js';

let _uiElements = null;
let _voiceStreamingMsg = null;
let _chatCallbacks = null;
let _voiceListeners = [];

const MIC_ICON_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>`;

function _bindVoice(event, handler) {
  voiceManager.on(event, handler);
  _voiceListeners.push([event, handler]);
}

function _unbindAllVoiceListeners() {
  for (const [event, handler] of _voiceListeners) {
    voiceManager.off(event, handler);
  }
  _voiceListeners = [];
}

/**
 * @param {HTMLElement} chatInputForm
 * @param {string} animaName
 * @param {object} [callbacks] Chat bubble callbacks:
 *   addUserBubble(text), addStreamingBubble() => msgRef,
 *   updateStreamingBubble(msgRef), finalizeStreamingBubble(msgRef)
 * @param {object} [opts] Options: { autoConnect: boolean }
 */
export function initVoiceUI(chatInputForm, animaName, callbacks, opts) {
  if (_uiElements) destroyVoiceUI();
  _chatCallbacks = callbacks || null;

  // Mic button — stays in .chat-input-actions grid alongside other buttons
  const micContainer = document.createElement('div');
  micContainer.className = 'voice-controls';

  const micBtn = document.createElement('button');
  micBtn.type = 'button';
  micBtn.className = 'voice-mic-btn';
  micBtn.title = t('voice.mic_input');
  micBtn.innerHTML = MIC_ICON_SVG;

  micContainer.append(micBtn);

  // Voice toolbar — separate row below input, shown when voice is active
  const toolbar = document.createElement('div');
  toolbar.className = 'voice-toolbar';
  toolbar.style.display = 'none';

  const tbStatus = document.createElement('div');
  tbStatus.className = 'voice-toolbar-status';

  const recIndicator = document.createElement('span');
  recIndicator.className = 'voice-rec-indicator';
  recIndicator.style.display = 'none';

  const ttsIndicator = document.createElement('span');
  ttsIndicator.className = 'voice-tts-indicator';
  ttsIndicator.style.display = 'none';

  const thinkingIndicator = document.createElement('span');
  thinkingIndicator.className = 'voice-thinking-indicator';
  thinkingIndicator.textContent = '考え中...';
  thinkingIndicator.style.display = 'none';

  tbStatus.append(recIndicator, ttsIndicator, thinkingIndicator);

  const tbControls = document.createElement('div');
  tbControls.className = 'voice-toolbar-controls';

  const modeToggle = document.createElement('button');
  modeToggle.type = 'button';
  modeToggle.className = 'voice-mode-toggle';
  modeToggle.textContent = 'PTT';
  modeToggle.title = t('voice.mode_toggle');

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.className = 'voice-volume-slider';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.value = '80';

  tbControls.append(modeToggle, volumeSlider);
  toolbar.append(tbStatus, tbControls);

  // Insert mic button into the pre-allocated slot
  const voiceSlot = chatInputForm.querySelector('.voice-controls-slot');
  if (voiceSlot) {
    voiceSlot.appendChild(micContainer);
  } else {
    chatInputForm.appendChild(micContainer);
  }

  // Insert toolbar after .chat-input-wrap (inside the input area)
  const inputWrap = chatInputForm.querySelector('.chat-input-wrap');
  if (inputWrap) {
    inputWrap.after(toolbar);
  } else {
    chatInputForm.appendChild(toolbar);
  }

  _uiElements = {
    container: micContainer,
    toolbar,
    micBtn,
    recIndicator,
    ttsIndicator,
    modeToggle,
    volumeSlider,
  };

  let voiceActive = false;
  let _connecting = false;

  // PTT state
  let _pttHolding = false;
  let _pttTapTimer = null;
  const _PTT_HOLD_THRESHOLD = 200;

  // Queued action to run after connect completes
  let _postConnectAction = null;

  async function _ensureConnected() {
    if (voiceActive) return true;
    if (_connecting) return false;
    _connecting = true;
    try {
      await voiceManager.connect(animaName);
      _connecting = false;
      voiceActive = true;
      micBtn.classList.add('active');
      toolbar.style.display = '';
      return true;
    } catch {
      _connecting = false;
      return false;
    }
  }

  function _handlePressStart(e) {
    e.preventDefault();

    if (!voiceActive) {
      _postConnectAction = 'hold';
      _ensureConnected().then((ok) => {
        if (!ok) { _postConnectAction = null; return; }
        const action = _postConnectAction;
        _postConnectAction = null;
        if (voiceManager.mode !== 'ptt') return;
        if (action === 'hold') {
          _pttHolding = true;
          voiceManager.startRecording();
        } else if (action === 'tap') {
          voiceManager.startRecording();
        }
      });
      return;
    }

    if (voiceManager.mode !== 'ptt') return;

    _pttHolding = false;
    _pttTapTimer = setTimeout(() => {
      _pttHolding = true;
      if (!voiceManager.isRecording) {
        voiceManager.startRecording();
      }
    }, _PTT_HOLD_THRESHOLD);
  }

  function _handlePressEnd() {
    if (_pttTapTimer) {
      clearTimeout(_pttTapTimer);
      _pttTapTimer = null;
    }

    if (_postConnectAction === 'hold') {
      _postConnectAction = 'tap';
      return;
    }

    if (!voiceActive) return;
    if (voiceManager.mode !== 'ptt') return;

    if (_pttHolding) {
      _pttHolding = false;
      voiceManager.stopRecording();
    } else {
      if (voiceManager.isRecording) {
        voiceManager.stopRecording();
      } else {
        voiceManager.startRecording();
      }
    }
  }

  function _handlePressCancel() {
    if (_pttTapTimer) {
      clearTimeout(_pttTapTimer);
      _pttTapTimer = null;
    }
    if (_postConnectAction) {
      _postConnectAction = null;
      return;
    }
    _pttHolding = false;
    if (voiceActive && voiceManager.mode === 'ptt') {
      voiceManager.stopRecording();
    }
  }

  function _toggleRecordingVAD() {
    if (!voiceActive || voiceManager.mode !== 'vad') return false;
    // In AUTO(VAD), mic button should not manually start/stop recording.
    // Recording is controlled only by VAD speech start/end callbacks.
    return true;
  }

  micBtn.addEventListener('mousedown', (e) => {
    if (e.button === 0) _handlePressStart(e);
  });
  micBtn.addEventListener('mouseup', _handlePressEnd);
  micBtn.addEventListener('mouseleave', _handlePressCancel);
  micBtn.addEventListener('click', (e) => {
    if (_toggleRecordingVAD()) e.stopPropagation();
  });

  micBtn.addEventListener('touchstart', (e) => {
    _handlePressStart(e);
  }, { passive: false });
  micBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!_toggleRecordingVAD()) _handlePressEnd();
  }, { passive: false });
  micBtn.addEventListener('touchcancel', _handlePressCancel);

  modeToggle.addEventListener('click', () => {
    const newMode = voiceManager.mode === 'ptt' ? 'vad' : 'ptt';
    voiceManager.setMode(newMode);
    modeToggle.textContent = newMode === 'ptt' ? 'PTT' : 'AUTO';
  });

  volumeSlider.addEventListener('input', () => {
    voiceManager.setVolume(parseInt(volumeSlider.value, 10) / 100);
  });

  _bindVoice('recordingStart', () => {
    recIndicator.style.display = '';
    micBtn.classList.add('recording');
  });
  _bindVoice('recordingStop', () => {
    recIndicator.style.display = 'none';
    micBtn.classList.remove('recording');
  });
  _bindVoice('ttsStart', () => {
    ttsIndicator.style.display = '';
  });
  _bindVoice('ttsDone', () => {
    if (!voiceManager.isTTSPlaying) ttsIndicator.style.display = 'none';
  });
  _bindVoice('playbackEnd', () => {
    ttsIndicator.style.display = 'none';
  });
  _bindVoice('transcript', ({ text }) => {
    if (!text || !animaName || !_chatCallbacks) return;
    _chatCallbacks.addUserBubble(text);
  });
  _bindVoice('responseStart', () => {
    if (!animaName || !_chatCallbacks) return;
    _voiceStreamingMsg = _chatCallbacks.addStreamingBubble();
  });
  _bindVoice('responseText', ({ text }) => {
    if (!_voiceStreamingMsg || !_chatCallbacks) return;
    _voiceStreamingMsg.text += text;
    _chatCallbacks.updateStreamingBubble(_voiceStreamingMsg);
  });
  _bindVoice('responseDone', ({ emotion }) => {
    if (!_voiceStreamingMsg || !_chatCallbacks) return;
    _voiceStreamingMsg.streaming = false;
    if (!_voiceStreamingMsg.text) _voiceStreamingMsg.text = '(空の応答)';
    _chatCallbacks.finalizeStreamingBubble(_voiceStreamingMsg);
    _chatCallbacks.applyEmotion?.(emotion);
    _voiceStreamingMsg = null;
  });
  _bindVoice('thinkingStatus', (thinking) => {
    thinkingIndicator.style.display = thinking ? '' : 'none';
    if (_voiceStreamingMsg && _chatCallbacks) {
      if (thinking) {
        _voiceStreamingMsg.thinkingText = _voiceStreamingMsg.thinkingText || '';
        _voiceStreamingMsg.thinking = true;
      } else {
        _voiceStreamingMsg.thinking = false;
      }
      _chatCallbacks.updateStreamingBubble(_voiceStreamingMsg);
    }
  });
  _bindVoice('thinkingDelta', ({ text }) => {
    if (_voiceStreamingMsg && _chatCallbacks) {
      _voiceStreamingMsg.thinkingText = (_voiceStreamingMsg.thinkingText || '') + text;
      _chatCallbacks.updateStreamingBubble(_voiceStreamingMsg);
    }
  });
  _bindVoice('disconnected', () => {
    voiceActive = false;
    _voiceStreamingMsg = null;
    micBtn.classList.remove('active', 'recording');
    recIndicator.style.display = 'none';
    ttsIndicator.style.display = 'none';
    thinkingIndicator.style.display = 'none';
    toolbar.style.display = 'none';
  });
  _bindVoice('error', ({ message }) => {
    console.warn('[VoiceUI] Error:', message);
  });

  // Auto-connect if voice was active before anima switch
  if (opts?.autoConnect) {
    _ensureConnected().catch(() => {});
  }

  return _uiElements;
}

export function destroyVoiceUI() {
  _unbindAllVoiceListeners();
  if (_uiElements) {
    voiceManager.disconnect();
    _uiElements.container.remove();
    _uiElements.toolbar?.remove();
    _uiElements = null;
  }
  _voiceStreamingMsg = null;
  _chatCallbacks = null;
}

/**
 * Called when switching Anima tabs.
 * Returns true if voice was previously active (caller should auto-reconnect).
 */
export function updateVoiceUIAnima(animaName) {
  const wasActive = voiceManager.isConnected;
  if (wasActive) {
    voiceManager.disconnect();
  }
  return wasActive;
}
