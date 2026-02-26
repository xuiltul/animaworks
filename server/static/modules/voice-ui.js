/**
 * Voice UI components — mic button, recording indicator, mode toggle, volume slider.
 * Integrates into existing chat input areas and renders voice chat bubbles.
 */
import { voiceManager } from './voice.js';

let _uiElements = null;
let _voiceStreamingMsg = null;
let _chatCallbacks = null;

const MIC_ICON_SVG = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>`;

/**
 * @param {HTMLElement} chatInputForm
 * @param {string} animaName
 * @param {object} [callbacks] Chat bubble callbacks:
 *   addUserBubble(text), addStreamingBubble() => msgRef,
 *   updateStreamingBubble(msgRef), finalizeStreamingBubble(msgRef)
 */
export function initVoiceUI(chatInputForm, animaName, callbacks) {
  if (_uiElements) destroyVoiceUI();
  _chatCallbacks = callbacks || null;

  const container = document.createElement('div');
  container.className = 'voice-controls';

  const micBtn = document.createElement('button');
  micBtn.type = 'button';
  micBtn.className = 'voice-mic-btn';
  micBtn.title = '音声入力';
  micBtn.innerHTML = MIC_ICON_SVG;

  const recIndicator = document.createElement('span');
  recIndicator.className = 'voice-rec-indicator';
  recIndicator.style.display = 'none';

  const ttsIndicator = document.createElement('span');
  ttsIndicator.className = 'voice-tts-indicator';
  ttsIndicator.style.display = 'none';

  const modeToggle = document.createElement('button');
  modeToggle.type = 'button';
  modeToggle.className = 'voice-mode-toggle';
  modeToggle.textContent = 'PTT';
  modeToggle.title = '入力モード切替（PTT/自動）';
  modeToggle.style.display = 'none';

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.className = 'voice-volume-slider';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.value = '80';
  volumeSlider.style.display = 'none';

  const thinkingIndicator = document.createElement('span');
  thinkingIndicator.className = 'voice-thinking-indicator';
  thinkingIndicator.textContent = '考え中...';
  thinkingIndicator.style.display = 'none';

  container.append(micBtn, recIndicator, ttsIndicator, thinkingIndicator, modeToggle, volumeSlider);

  const sendBtn = chatInputForm.querySelector(
    '[id$="SendBtn"], .chat-send-btn, button[type="submit"]'
  );
  if (sendBtn && sendBtn.parentNode) {
    sendBtn.parentNode.insertBefore(container, sendBtn);
  } else {
    chatInputForm.appendChild(container);
  }

  _uiElements = {
    container,
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
      modeToggle.style.display = '';
      volumeSlider.style.display = '';
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
    if (voiceManager.isRecording) {
      voiceManager.stopRecording();
    } else {
      voiceManager.startRecording();
    }
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

  voiceManager.on('recordingStart', () => {
    recIndicator.style.display = '';
    micBtn.classList.add('recording');
  });
  voiceManager.on('recordingStop', () => {
    recIndicator.style.display = 'none';
    micBtn.classList.remove('recording');
  });
  voiceManager.on('ttsStart', () => {
    ttsIndicator.style.display = '';
  });
  voiceManager.on('ttsDone', () => {
    if (!voiceManager.isTTSPlaying) ttsIndicator.style.display = 'none';
  });
  voiceManager.on('playbackEnd', () => {
    ttsIndicator.style.display = 'none';
  });
  voiceManager.on('transcript', ({ text }) => {
    if (!text || !animaName || !_chatCallbacks) return;
    _chatCallbacks.addUserBubble(text);
  });
  voiceManager.on('responseStart', () => {
    if (!animaName || !_chatCallbacks) return;
    _voiceStreamingMsg = _chatCallbacks.addStreamingBubble();
  });
  voiceManager.on('responseText', ({ text }) => {
    if (!_voiceStreamingMsg || !_chatCallbacks) return;
    _voiceStreamingMsg.text += text;
    _chatCallbacks.updateStreamingBubble(_voiceStreamingMsg);
  });
  voiceManager.on('responseDone', () => {
    if (!_voiceStreamingMsg || !_chatCallbacks) return;
    _voiceStreamingMsg.streaming = false;
    if (!_voiceStreamingMsg.text) _voiceStreamingMsg.text = '(空の応答)';
    _chatCallbacks.finalizeStreamingBubble(_voiceStreamingMsg);
    _voiceStreamingMsg = null;
  });
  voiceManager.on('thinkingStatus', (thinking) => {
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
  voiceManager.on('thinkingDelta', ({ text }) => {
    if (_voiceStreamingMsg && _chatCallbacks) {
      _voiceStreamingMsg.thinkingText = (_voiceStreamingMsg.thinkingText || '') + text;
      _chatCallbacks.updateStreamingBubble(_voiceStreamingMsg);
    }
  });
  voiceManager.on('disconnected', () => {
    voiceActive = false;
    _voiceStreamingMsg = null;
    micBtn.classList.remove('active', 'recording');
    recIndicator.style.display = 'none';
    ttsIndicator.style.display = 'none';
    thinkingIndicator.style.display = 'none';
    modeToggle.style.display = 'none';
    volumeSlider.style.display = 'none';
  });
  voiceManager.on('error', ({ message }) => {
    console.warn('[VoiceUI] Error:', message);
  });

  return _uiElements;
}

export function destroyVoiceUI() {
  if (_uiElements) {
    voiceManager.disconnect();
    _uiElements.container.remove();
    _uiElements = null;
  }
  _voiceStreamingMsg = null;
  _chatCallbacks = null;
}

export function updateVoiceUIAnima(animaName) {
  if (voiceManager.isConnected) {
    voiceManager.disconnect();
  }
}
