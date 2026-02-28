// ── Workspace Chat Controller ──────────────────────
// Orchestrates conversation in the workspace overlay using sub-modules.
// Workspace-specific: Live2D hooks, 3D office integration, conversation overlay DOM.

import { getState, setState } from "./state.js";
import { initBustup, setCharacter, setExpression, setTalking, onClick as onBustupClick } from "./live2d.js";
import { createImageInput, initLightbox } from "../../shared/image-input.js";
import { initTextArtifactHandlers } from "../../shared/text-artifact.js";

import { initHistory, renderConvMessages, loadAndRenderConvMessages, disconnectScrollObserver, getHistoryState, refreshSentinel } from "./chat-history.js";
import { initStreaming, submitConversation, addToQueue, resumeConversationStream, wsUpdateSendButton, wsHidePendingIndicator, updateStreamingBubble } from "./chat-streaming.js";
import { initThreads, renderWsThreadTabs } from "./chat-thread.js";
import {
  initMobile, setupMobileListeners, destroyMobile,
  isMobileView, closeMobileSidebar, closeMobileCharacter, closeMobileMemory,
  updateConvInputPlaceholder, wsSaveDraft, wsLoadDraft,
  triggerGreeting, buildVoiceChatCallbacks, initVoiceUI, updateVoiceUIAnima,
} from "./chat-mobile.js";

export { submitConversation, addToQueue };

// ── Module State ──────────────────────
let _dom = {};
let bustupInitialized = false;
let convStreamController = null;
let convImageInputManager = null;
let convPendingQueue = [];

// ── State Accessors ──────────────────────

function _getDom() { return _dom; }
function _getStreamController() { return convStreamController; }
function _setStreamController(v) { convStreamController = v; }
function _getImageManager() { return convImageInputManager; }
function _getPendingQueue() { return convPendingQueue; }

// ── Conversation Lifecycle ──────────────────────

export async function openConversation(animaName) {
  if (!_dom.convOverlay) return;

  wsSaveDraft();
  const wasVoiceActive = updateVoiceUIAnima(animaName);
  setState({ conversationOpen: true, conversationAnima: animaName, activeThreadId: "default" });

  const { threads } = getState();
  if (!threads[animaName]) {
    setState({ threads: { ...threads, [animaName]: [{ id: "default", label: "メイン", unread: false }] } });
  }

  _dom.convOverlay.classList.remove("hidden");
  if (_dom.convAnimaName) _dom.convAnimaName.textContent = animaName;
  updateConvInputPlaceholder();
  if (_dom.convInput) {
    _dom.convInput.value = wsLoadDraft(animaName);
    _dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 100 : 120;
    _dom.convInput.style.height = Math.min(_dom.convInput.scrollHeight, maxH) + "px";
  }

  closeMobileSidebar();
  closeMobileCharacter();

  if (!bustupInitialized && _dom.convCanvas) {
    initBustup(_dom.convCanvas);
    bustupInitialized = true;
    onBustupClick(() => {
      setExpression("surprised");
      setTimeout(() => setExpression("smile"), 1200);
      setTimeout(() => setExpression("neutral"), 2500);
    });
  }

  await setCharacter(animaName);
  setExpression("neutral");

  await loadAndRenderConvMessages(animaName, { resumeStream: resumeConversationStream });
  renderWsThreadTabs();
  triggerGreeting(animaName, { renderConvMessages });

  _dom.convInput?.focus();

  const convInputArea = document.querySelector(".ws-conv-input-area");
  if (convInputArea && animaName) {
    initVoiceUI(convInputArea, animaName, buildVoiceChatCallbacks(animaName, { renderConvMessages, updateStreamingBubble }), { autoConnect: wasVoiceActive });
  }
}

export function closeConversation() {
  if (!_dom.convOverlay) return;

  wsSaveDraft();
  _dom.convOverlay.classList.add("hidden");

  closeMobileSidebar();
  closeMobileCharacter();
  closeMobileMemory();
  destroyMobile();

  disconnectScrollObserver();

  const animaName = getState().conversationAnima;
  const historyState = getHistoryState();
  if (animaName) delete historyState[animaName];

  setState({ conversationOpen: false, conversationAnima: null });
  setTalking(false);

  convPendingQueue = [];
  wsHidePendingIndicator();

  if (convStreamController) { convStreamController.abort(); convStreamController = null; }
}

export function isConvStreaming() {
  return !!convStreamController;
}

// ── Initialization ──────────────────────

export function initChatController(dom) {
  _dom = dom;

  initHistory({ getDom: _getDom });
  initStreaming({
    getDom: _getDom,
    getStreamController: _getStreamController,
    setStreamController: _setStreamController,
    getImageManager: _getImageManager,
    getPendingQueue: _getPendingQueue,
  });
  initThreads({
    getDom: _getDom,
    getHistoryState,
    renderConvMessages,
    refreshSentinel,
  });
  initMobile({ getDom: _getDom });

  dom.convBack?.addEventListener("click", closeConversation);
  dom.convOverlay?.addEventListener("click", e => { if (e.target === dom.convOverlay) closeConversation(); });
  dom.convSend?.addEventListener("click", () => submitConversation());
  dom.convQueueBtn?.addEventListener("click", () => addToQueue());
  dom.convInput?.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.altKey) { e.preventDefault(); addToQueue(); }
    else if (e.key === "Enter") {
      if (isMobileView()) { if (!e.shiftKey) { e.preventDefault(); submitConversation(); } }
      else { if (e.ctrlKey || e.metaKey) { e.preventDefault(); submitConversation(); } }
    }
  });

  const convInputWrap = document.querySelector(".ws-conv-input-area .chat-input-wrap");
  const focusHandler = e => {
    if (e.target instanceof Element && e.target.closest("button, input, select, textarea, a")) return;
    dom.convInput?.focus();
  };
  convInputWrap?.addEventListener("pointerdown", focusHandler);
  convInputWrap?.addEventListener("click", focusHandler);

  dom.convPendingCancel?.addEventListener("click", () => {
    convPendingQueue = [];
    wsHidePendingIndicator();
    wsUpdateSendButton(!!convStreamController);
  });

  dom.convInput?.addEventListener("input", () => {
    dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 140 : 220;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
    wsSaveDraft();
    wsUpdateSendButton(!!convStreamController);
  });

  if (dom.convAttachBtn && dom.convFileInput) {
    dom.convAttachBtn.addEventListener("click", () => dom.convFileInput.click());
    dom.convFileInput.addEventListener("change", () => {
      if (dom.convFileInput.files.length > 0) {
        convImageInputManager?.addFiles(dom.convFileInput.files);
        dom.convFileInput.value = "";
      }
    });
  }

  const convMain = document.querySelector(".ws-conv-main");
  if (convMain && dom.convInput && dom.convPreviewBar) {
    convImageInputManager = createImageInput({
      container: convMain,
      inputArea: dom.convInput,
      previewContainer: dom.convPreviewBar,
    });
  }

  initLightbox();
  initTextArtifactHandlers();

  setupMobileListeners(dom);
}
