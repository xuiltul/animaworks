// ── Workspace Chat Streaming ──────────────────────
// Message sending, streaming connection/resume, send button UI, queue management.
// Now delegates stream/queue state to ChatSessionManager; keeps Live2D hooks.

import { getState, setState } from "./state.js";
import { getCurrentUser } from "./login.js";
import { escapeHtml } from "./utils.js";
import { setExpression, setTalking } from "./live2d.js";
import { createLogger } from "../../shared/logger.js";
import { renderConvMessages, renderOpts } from "./chat-history.js";
import { renderStreamingBubbleInner } from "../../shared/chat/render-utils.js";
import { renderWsThreadTabs } from "./chat-thread.js";
import { wsSaveDraft, wsClearDraft, isMobileView } from "./chat-mobile.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";

const logger = createLogger("ws-chat-streaming");
let _getDom = () => ({});
let _getImageManager = () => null;
let _convRafPending = false;
let _convLatestStreamingMsg = null;

export function initStreaming({ getDom, getImageManager }) {
  _getDom = getDom;
  _getImageManager = getImageManager;
}

function _mgr() { return ChatSessionManager.getInstance(); }

function _animaThread() {
  const st = getState();
  return { anima: st.conversationAnima, thread: st.activeThreadId || "default" };
}

function _drainQueue(explicitAnima, explicitThread) {
  const { anima: curAnima, thread: curThread } = _animaThread();
  const anima = explicitAnima || curAnima;
  const thread = explicitThread || curThread;
  if (!anima) return;
  const mgr = _mgr();
  const q = mgr.getPendingQueue(anima, thread);
  if (q.length === 0) return;
  const next = mgr.dequeue(anima, thread);
  wsShowPendingIndicator();
  if (mgr.getPendingQueue(anima, thread).length === 0) wsHidePendingIndicator();
  setTimeout(() => _sendConversation(next.text, { images: next.images, displayImages: next.displayImages }), 150);
}

function _baseCallbacks(streamingMsg) {
  return {
    onCompressionStart: () => { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg); },
    onCompressionEnd: () => { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg); },
    onToolStart: (n) => { streamingMsg.activeTool = n; setExpression("thinking"); updateStreamingBubble(streamingMsg); },
    onToolEnd: () => { streamingMsg.activeTool = null; setExpression("neutral"); updateStreamingBubble(streamingMsg); },
    onThinkingStart: () => { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; updateStreamingBubble(streamingMsg); },
    onThinkingDelta: (t) => { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; scheduleStreamingUpdate(streamingMsg); },
    onThinkingEnd: () => { streamingMsg.thinking = false; updateStreamingBubble(streamingMsg); },
  };
}

function _enqueueInput() {
  const dom = _getDom();
  const text = dom.convInput?.value?.trim();
  const im = _getImageManager();
  const hasImages = im && im.getImageCount() > 0;
  if (!text && !hasImages) return null;

  const { anima, thread } = _animaThread();
  if (!anima) return null;

  const entry = { text: text || "", images: im?.getPendingImages() || [], displayImages: im?.getDisplayImages() || [] };
  _mgr().enqueue(anima, thread, entry);
  if (dom.convInput) { dom.convInput.value = ""; dom.convInput.style.height = "auto"; }
  wsSaveDraft(); im?.clearImages();
  return entry;
}

export function scheduleStreamingUpdate(msg) {
  _convLatestStreamingMsg = msg;
  if (_convRafPending) return;
  _convRafPending = true;
  requestAnimationFrame(() => { _convRafPending = false; if (_convLatestStreamingMsg) updateStreamingBubble(_convLatestStreamingMsg); });
}

export function updateStreamingBubble(msg) {
  const dom = _getDom();
  if (!dom.convMessages) return;
  const bubbles = dom.convMessages.querySelectorAll(".chat-bubble.assistant.streaming");
  const bubble = bubbles[bubbles.length - 1];
  if (!bubble) return;
  bubble.innerHTML = renderStreamingBubbleInner(msg, renderOpts());
  dom.convMessages.scrollTop = dom.convMessages.scrollHeight;
}

export function submitConversation() {
  const { anima, thread } = _animaThread();
  if (!anima) return;
  const mgr = _mgr();
  const isStreaming = mgr.isStreamingFor(anima, thread);

  if (!isStreaming) {
    _enqueueInput();
    const q = mgr.getPendingQueue(anima, thread);
    if (q.length === 0) return;
    const next = mgr.dequeue(anima, thread);
    wsShowPendingIndicator();
    if (mgr.getPendingQueue(anima, thread).length === 0) wsHidePendingIndicator();
    _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
    return;
  }
  if (_enqueueInput()) { wsShowPendingIndicator(); wsUpdateSendButton(true); return; }
  wsStopStreaming();
}

export function addToQueue() {
  if (!_enqueueInput()) return;
  const { anima, thread } = _animaThread();
  wsShowPendingIndicator(); wsUpdateSendButton(anima ? _mgr().isStreamingFor(anima, thread) : false);
}

async function _sendConversation(text, overrideImages = null) {
  const dom = _getDom();
  const im = _getImageManager();
  const images = overrideImages?.images || im?.getPendingImages() || [];
  const displayImages = overrideImages?.displayImages || im?.getDisplayImages() || [];
  if (!text && images.length === 0) return;
  const { anima, thread } = _animaThread();
  if (!anima) return;

  dom.convInput.value = ""; dom.convInput.disabled = true; dom.convSend.disabled = true;
  if (!overrideImages) im?.clearImages();
  wsUpdateSendButton(true);
  renderWsThreadTabs();

  const mgr = _mgr();
  let talkingStarted = false;

  // Use let + onStreamCreated to avoid TDZ: const destructuring from
  // await would not be initialized when SSE callbacks fire during streaming.
  let streamingMsg = null;

  const { success, error } = await mgr.sendChat(anima, thread, text, {
    images,
    displayImages,
    callbacks: {
      onStreamCreated: (msg) => { streamingMsg = msg; renderConvMessages(); },
      onTextDelta: (d) => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.afterHeartbeatRelay = false;
        if (!talkingStarted) { setTalking(true); setExpression("neutral"); talkingStarted = true; }
        streamingMsg.text += d; scheduleStreamingUpdate(streamingMsg);
      },
      onCompressionStart: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg); } },
      onCompressionEnd: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg); } },
      onToolStart: (n) => { if (streamingMsg?.streaming) { streamingMsg.activeTool = n; setExpression("thinking"); updateStreamingBubble(streamingMsg); } },
      onToolEnd: () => { if (streamingMsg?.streaming) { streamingMsg.activeTool = null; setExpression("neutral"); updateStreamingBubble(streamingMsg); } },
      onThinkingStart: () => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; updateStreamingBubble(streamingMsg); } },
      onThinkingDelta: (t) => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; scheduleStreamingUpdate(streamingMsg); } },
      onThinkingEnd: () => { if (streamingMsg?.streaming) { streamingMsg.thinking = false; updateStreamingBubble(streamingMsg); } },
      onHeartbeatRelayStart: () => { if (streamingMsg?.streaming) { streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = ""; scheduleStreamingUpdate(streamingMsg); } },
      onHeartbeatRelay: ({ text: t }) => { if (streamingMsg?.streaming) { streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + t; scheduleStreamingUpdate(streamingMsg); } },
      onHeartbeatRelayDone: () => { if (streamingMsg?.streaming) { streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true; scheduleStreamingUpdate(streamingMsg); } },
      onDone: ({ summary, emotion, images: di }) => {
        if (streamingMsg) {
          if (summary) { streamingMsg.text = summary; updateStreamingBubble(streamingMsg); }
          streamingMsg.images = di || [];
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
        }
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => {
        setExpression("troubled");
        if (streamingMsg) { streamingMsg.text += `\n[エラー: ${m}]`; updateStreamingBubble(streamingMsg); }
      },
      onAbort: () => {
        if (streamingMsg) {
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
          if (!streamingMsg.text) streamingMsg.text = "(中断されました)";
        }
      },
    },
    onFinally: () => {
      try {
        setTalking(false);
        if (streamingMsg?.streaming) {
          streamingMsg.streaming = false;
          if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        }
        renderConvMessages();
        renderWsThreadTabs();
        if (dom.convInput) dom.convInput.disabled = false;
        wsUpdateSendButton(false); wsSaveDraft(); dom.convInput?.focus();

        const st = getState();
        const threadList = st.threads[anima] || [];
        const entry = threadList.find(t => t.id === thread);
        if (entry && entry.label === "新しいスレッド" && (text || "").trim()) {
          const lbl = (text || "").trim().slice(0, 20) + ((text || "").trim().length > 20 ? "..." : "");
          setState({ threads: { ...st.threads, [anima]: threadList.map(t => t.id === thread ? { ...t, label: lbl } : t) } });
          renderWsThreadTabs();
        }
      } finally {
        _drainQueue(anima, thread);
      }
    },
  });

  renderConvMessages();

  if (!success && error && error.name !== "AbortError") {
    logger.error("Conversation stream error", { anima, error: error.message });
    setExpression("troubled");
  }
}

export async function resumeConversationStream(animaName) {
  const mgr = _mgr();
  const threadId = getState().activeThreadId || "default";
  if (mgr.isStreamingFor(animaName, threadId)) return;
  const dom = _getDom();

  wsUpdateSendButton(true);

  let streamingMsg = null;

  const { success } = await mgr.resumeStream(animaName, threadId, {
    callbacks: {
      onStreamCreated: (msg) => { streamingMsg = msg; renderConvMessages(); },
      onTextDelta: (d) => { if (streamingMsg?.streaming) { streamingMsg.text += d; scheduleStreamingUpdate(streamingMsg); } },
      onCompressionStart: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg); } },
      onCompressionEnd: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg); } },
      onToolStart: (n) => { if (streamingMsg?.streaming) { streamingMsg.activeTool = n; setExpression("thinking"); updateStreamingBubble(streamingMsg); } },
      onToolEnd: () => { if (streamingMsg?.streaming) { streamingMsg.activeTool = null; setExpression("neutral"); updateStreamingBubble(streamingMsg); } },
      onThinkingStart: () => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; updateStreamingBubble(streamingMsg); } },
      onThinkingDelta: (t) => { if (streamingMsg?.streaming) { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; scheduleStreamingUpdate(streamingMsg); } },
      onThinkingEnd: () => { if (streamingMsg?.streaming) { streamingMsg.thinking = false; updateStreamingBubble(streamingMsg); } },
      onDone: ({ summary, emotion, images: di }) => {
        if (streamingMsg) {
          if (summary) streamingMsg.text = summary;
          streamingMsg.images = di || [];
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
        }
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => {
        if (streamingMsg) { streamingMsg.text += `\n[エラー: ${m}]`; streamingMsg.streaming = false; }
        setExpression("troubled");
      },
    },
    onFinally: () => {
      try {
        setTalking(false);
        if (streamingMsg?.streaming) {
          streamingMsg.streaming = false;
          if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        }
        renderConvMessages();
        renderWsThreadTabs();
        wsUpdateSendButton(false); dom.convInput?.focus();
      } finally {
        _drainQueue(animaName, threadId);
      }
    },
  });

  if (streamingMsg) renderConvMessages();
  if (!success && !streamingMsg) wsUpdateSendButton(false);
}

// ── Send Button / Pending Queue UI ──────────────────────

const _ICONS = {
  send: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg>`,
  stop: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg>`,
  interrupt: `<span class="chat-send-icon-group" aria-hidden="true"><svg class="chat-send-icon chat-send-icon-square" viewBox="0 0 24 24" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg><svg class="chat-send-icon" viewBox="0 0 24 24" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg></span>`,
};

export function wsUpdateSendButton(isStreaming) {
  const dom = _getDom();
  const { anima, thread } = _animaThread();
  const mgr = _mgr();
  const q = anima ? mgr.getPendingQueue(anima, thread) : [];
  const hasInput = (dom.convInput?.value?.trim() || "").length > 0;
  if (dom.convQueueBtn) dom.convQueueBtn.disabled = !hasInput;
  if (!dom.convSend) return;
  dom.convSend.classList.remove("stop", "interrupt");
  if (!isStreaming) { dom.convSend.innerHTML = _ICONS.send; dom.convSend.disabled = !hasInput && q.length === 0; }
  else if (hasInput) { dom.convSend.innerHTML = _ICONS.send; dom.convSend.disabled = false; }
  else if (q.length > 0) { dom.convSend.innerHTML = _ICONS.interrupt; dom.convSend.classList.add("interrupt"); dom.convSend.disabled = false; }
  else { dom.convSend.innerHTML = _ICONS.stop; dom.convSend.classList.add("stop"); dom.convSend.disabled = false; }
}

export function wsShowPendingIndicator() {
  const dom = _getDom();
  const { anima, thread } = _animaThread();
  if (!anima) return;
  const mgr = _mgr();
  const q = mgr.getPendingQueue(anima, thread);
  if (!dom.convPending || !dom.convPendingList) return;
  if (q.length === 0) { dom.convPending.style.display = "none"; return; }
  if (dom.convPendingLabel) dom.convPendingLabel.textContent = `キュー (${q.length})`;
  dom.convPendingList.innerHTML = q.map((p, i) => {
    const txt = escapeHtml(p.text.length > 50 ? p.text.slice(0, 50) + "…" : p.text);
    const img = p.images?.length ? ` <span style="opacity:0.6">(+${p.images.length}画像)</span>` : "";
    return `<div class="pending-queue-item" data-idx="${i}"><span class="pending-queue-item-num">${i + 1}.</span><span class="pending-queue-item-text">${txt || "(画像のみ)"}${img}</span><button class="pending-queue-item-del" data-idx="${i}" type="button">✕</button></div>`;
  }).join("");
  dom.convPending.style.display = "";
  dom.convPendingList.onclick = (e) => {
    const delBtn = e.target.closest(".pending-queue-item-del");
    if (delBtn) {
      e.stopPropagation();
      mgr.removeFromQueue(anima, thread, parseInt(delBtn.dataset.idx, 10));
      wsShowPendingIndicator();
      wsUpdateSendButton(mgr.isStreamingFor(anima, thread));
      return;
    }
    const item = e.target.closest(".pending-queue-item");
    if (!item) return;
    const removed = mgr.removeFromQueue(anima, thread, parseInt(item.dataset.idx, 10));
    if (removed && dom.convInput) {
      dom.convInput.value = removed.text; dom.convInput.style.height = "auto";
      dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, isMobileView() ? 100 : 120) + "px";
      dom.convInput.focus();
    }
    wsShowPendingIndicator(); wsUpdateSendButton(mgr.isStreamingFor(anima, thread));
  };
}

export function wsHidePendingIndicator() {
  const dom = _getDom();
  if (dom.convPending) dom.convPending.style.display = "none";
}

export function wsStopStreaming() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  _mgr().stopStreaming(animaName);
}
