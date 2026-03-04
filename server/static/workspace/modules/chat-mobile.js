// ── Workspace Chat Mobile / Voice / Greeting / Draft ──────────────────────
// Mobile responsive helpers, voice chat callback builder, greeting trigger, draft save/load.

import { getState, setState } from "./state.js";
import { greetAnima } from "./api.js";
import { getCurrentUser } from "./login.js";
import { setExpression } from "./live2d.js";
import { initVoiceUI, destroyVoiceUI, updateVoiceUIAnima } from "../../modules/voice-ui.js";
import { SwipeHandler } from "../../modules/touch.js";
import { getDraftKey, saveDraft, loadDraft, clearDraft } from "../../shared/chat/draft.js";
import { createLogger } from "../../shared/logger.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";

const logger = createLogger("ws-chat-mobile");

// ── Module State ──────────────────────
let _getDom = () => ({});
let _swiperInstance = null;
let _mobileMediaQuery = null;

let _greetingInFlight = false;
const _GREET_COOLDOWN_MS = 3600 * 1000;
const _lastGreetTime = {};

// ── Init / Destroy ──────────────────────

export function initMobile({ getDom }) {
  _getDom = getDom;
}

export function setupMobileListeners(dom) {
  dom.mobileSidebarToggle?.addEventListener("click", () => {
    if (dom.convSidebar?.classList.contains("mobile-open")) closeMobileSidebar();
    else { closeMobileCharacter(); openMobileSidebar(); }
  });
  dom.mobileCharacterToggle?.addEventListener("click", () => {
    if (dom.convCharacter?.classList.contains("mobile-open")) closeMobileCharacter();
    else { closeMobileSidebar(); toggleMobileCharacter(); }
  });
  dom.sidebarBackdrop?.addEventListener("click", closeMobileSidebar);
  dom.mobileMemoryClose?.addEventListener("click", closeMobileMemory);

  updateConvInputPlaceholder();
  _mobileMediaQuery = window.matchMedia("(max-width: 768px)");
  _mobileMediaQuery.addEventListener("change", updateConvInputPlaceholder);

  if ("ontouchstart" in window && dom.convOverlay) {
    _swiperInstance = new SwipeHandler(dom.convOverlay);
    _swiperInstance.onSwipeRight(info => { if (isMobileView() && info.startX < 30) openMobileSidebar(); });
    _swiperInstance.onSwipeLeft(() => { if (isMobileView()) closeMobileSidebar(); });
  }
}

export function destroyMobile() {
  destroyVoiceUI();
  cleanupMobileResources();
}

// ── Mobile View Helpers ──────────────────────

export function isMobileView() {
  return window.matchMedia("(max-width: 768px)").matches;
}

export function openMobileSidebar() {
  const dom = _getDom();
  dom.convSidebar?.classList.add("mobile-open");
  dom.sidebarBackdrop?.classList.add("visible");
}

export function closeMobileSidebar() {
  const dom = _getDom();
  dom.convSidebar?.classList.remove("mobile-open");
  dom.sidebarBackdrop?.classList.remove("visible");
}

export function toggleMobileCharacter() {
  const dom = _getDom();
  dom.convCharacter?.classList.toggle("mobile-open");
}

export function closeMobileCharacter() {
  const dom = _getDom();
  dom.convCharacter?.classList.remove("mobile-open");
}

export function openMobileMemory() {
  const dom = _getDom();
  dom.memoryPanel?.classList.add("mobile-open");
}

export function closeMobileMemory() {
  const dom = _getDom();
  dom.memoryPanel?.classList.remove("mobile-open");
}

export function updateConvInputPlaceholder() {
  const dom = _getDom();
  if (!dom.convInput) return;
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  dom.convInput.placeholder = isMobileView()
    ? `${animaName} にメッセージ... (Enter)`
    : `メッセージを入力... (Ctrl+Enter)`;
}

export function cleanupMobileResources() {
  if (_swiperInstance) { _swiperInstance.destroy(); _swiperInstance = null; }
  if (_mobileMediaQuery) { _mobileMediaQuery.removeEventListener("change", updateConvInputPlaceholder); _mobileMediaQuery = null; }
}

// ── Draft ──────────────────────

export function wsDraftKey(animaName, threadId) {
  return getDraftKey("workspace-conv", getCurrentUser() || "guest", animaName, threadId);
}

export function wsSaveDraft() {
  const dom = _getDom();
  const { conversationAnima, activeThreadId } = getState();
  if (!conversationAnima || !dom.convInput) return;
  saveDraft(wsDraftKey(conversationAnima, activeThreadId || "default"), dom.convInput.value || "");
}

export function wsLoadDraft(animaName, threadId) {
  if (!animaName) return "";
  return loadDraft(wsDraftKey(animaName, threadId || "default"));
}

export function wsClearDraft(animaName, threadId) {
  if (!animaName) return;
  clearDraft(wsDraftKey(animaName, threadId || "default"));
}

// ── Greeting ──────────────────────

export async function triggerGreeting(animaName, { renderConvMessages }) {
  if (_greetingInFlight) return;
  const lastTs = _lastGreetTime[animaName];
  if (lastTs && Date.now() - lastTs < _GREET_COOLDOWN_MS) return;

  _greetingInFlight = true;
  try {
    const data = await greetAnima(animaName);
    if (!data.response || data.cached) return;

    _lastGreetTime[animaName] = Date.now();
    const now = new Date().toISOString();

    const { conversationAnima, activeThreadId } = getState();
    const threadId = activeThreadId || "default";
    const mgr = ChatSessionManager.getInstance();
    mgr.addMessage(conversationAnima, threadId, { role: "system", text: "デスクを訪問しました", timestamp: now });
    mgr.addMessage(conversationAnima, threadId, { role: "assistant", text: data.response, timestamp: now });
    renderConvMessages();

    if (data.emotion) {
      setExpression(data.emotion);
      setTimeout(() => setExpression("neutral"), 3000);
    }
  } catch (err) {
    logger.error("Failed to greet", { anima: animaName, error: err.message });
  } finally {
    _greetingInFlight = false;
  }
}

// ── Voice Chat Callbacks ──────────────────────

export function buildVoiceChatCallbacks(animaName, { renderConvMessages, updateStreamingBubble }) {
  const mgr = ChatSessionManager.getInstance();

  return {
    addUserBubble(text) {
      const { conversationAnima, activeThreadId } = getState();
      const threadId = activeThreadId || "default";
      const ts = new Date().toISOString();
      mgr.addMessage(conversationAnima, threadId, { role: "user", text, timestamp: ts });
      renderConvMessages();
    },
    addStreamingBubble() {
      const { conversationAnima, activeThreadId } = getState();
      const threadId = activeThreadId || "default";
      const ts = new Date().toISOString();
      const msg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: ts, thinkingText: "", thinking: false };
      mgr.addMessage(conversationAnima, threadId, msg);
      const session = mgr.getSession(conversationAnima, threadId);
      session._streamingMsg = msg;
      renderConvMessages();
      return msg;
    },
    updateStreamingBubble(msg) { updateStreamingBubble(msg); },
    finalizeStreamingBubble() {
      const { conversationAnima, activeThreadId } = getState();
      const threadId = activeThreadId || "default";
      const session = mgr.getSession(conversationAnima, threadId);
      session._streamingMsg = null;
      // Do NOT call keepOnlyStreaming - it would clear completed messages before API has them
      renderConvMessages();
    },
    applyEmotion(emotion) {
      if (!emotion) return;
      setExpression(emotion);
      setTimeout(() => setExpression("neutral"), 3000);
    },
  };
}

// ── Voice UI Proxies ──────────────────────

export { initVoiceUI, updateVoiceUIAnima };
