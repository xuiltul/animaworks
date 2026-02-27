// ── Workspace Chat Streaming ──────────────────────
// Message sending, streaming connection/resume, send button UI, queue management.

import { getState, setState } from "./state.js";
import { getCurrentUser } from "./login.js";
import { escapeHtml } from "./utils.js";
import { setExpression, setTalking } from "./live2d.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../../shared/chat-stream.js";
import { createLogger } from "../../shared/logger.js";
import { renderConvMessages, renderOpts } from "./chat-history.js";
import { renderStreamingBubbleInner } from "../../shared/chat/render-utils.js";
import { renderWsThreadTabs } from "./chat-thread.js";
import { wsSaveDraft, wsClearDraft, isMobileView } from "./chat-mobile.js";

const logger = createLogger("ws-chat-streaming");
let _getDom = () => ({});
let _getStreamController = () => null;
let _setStreamController = (_v) => {};
let _getImageManager = () => null;
let _getPendingQueue = () => [];
let _convRafPending = false;
let _convLatestStreamingMsg = null;

export function initStreaming({ getDom, getStreamController, setStreamController, getImageManager, getPendingQueue }) {
  _getDom = getDom; _getStreamController = getStreamController; _setStreamController = setStreamController;
  _getImageManager = getImageManager; _getPendingQueue = getPendingQueue;
}

function _commitThread(animaName, threadId) {
  const st = getState();
  const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
  const nbt = { ...st.chatMessagesByThread };
  if (!nbt[animaName]) nbt[animaName] = {};
  nbt[animaName] = { ...nbt[animaName], [threadId]: [...arr] };
  setState({ chatMessagesByThread: nbt });
}

function _drainQueue() {
  const q = _getPendingQueue();
  if (q.length === 0) return;
  const next = q.shift();
  wsShowPendingIndicator();
  if (q.length === 0) wsHidePendingIndicator();
  setTimeout(() => _sendConversation(next.text, { images: next.images, displayImages: next.displayImages }), 150);
}

function _baseCallbacks(streamingMsg) {
  return {
    onToolStart: (n) => { streamingMsg.activeTool = n; setExpression("thinking"); updateStreamingBubble(streamingMsg); },
    onToolEnd: () => { streamingMsg.activeTool = null; setExpression("neutral"); updateStreamingBubble(streamingMsg); },
    onThinkingStart: () => { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; updateStreamingBubble(streamingMsg); },
    onThinkingDelta: (t) => { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; scheduleStreamingUpdate(streamingMsg); },
    onThinkingEnd: () => { streamingMsg.thinking = false; updateStreamingBubble(streamingMsg); },
  };
}

function _finalize(msg, animaName, threadId) {
  msg.streaming = false;
  if (!msg.text) msg.text = "(空の応答)";
  _commitThread(animaName, threadId); renderConvMessages();
}

function _enqueueInput() {
  const dom = _getDom();
  const text = dom.convInput?.value?.trim();
  const im = _getImageManager();
  const hasImages = im && im.getImageCount() > 0;
  if (!text && !hasImages) return null;
  const entry = { text: text || "", images: im?.getPendingImages() || [], displayImages: im?.getDisplayImages() || [] };
  _getPendingQueue().push(entry);
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
  const isStreaming = !!_getStreamController();
  const q = _getPendingQueue();
  if (!isStreaming) {
    _enqueueInput();
    if (q.length === 0) return;
    const next = q.shift();
    wsShowPendingIndicator();
    if (q.length === 0) wsHidePendingIndicator();
    _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
    return;
  }
  if (_enqueueInput()) { wsShowPendingIndicator(); wsUpdateSendButton(true); return; }
  wsStopStreaming();
}

export function addToQueue() {
  if (!_enqueueInput()) return;
  wsShowPendingIndicator(); wsUpdateSendButton(!!_getStreamController());
}

async function _sendConversation(text, overrideImages = null) {
  const dom = _getDom();
  const im = _getImageManager();
  const images = overrideImages?.images || im?.getPendingImages() || [];
  const displayImages = overrideImages?.displayImages || im?.getDisplayImages() || [];
  if (!text && images.length === 0) return;
  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  dom.convInput.value = ""; dom.convInput.disabled = true; dom.convSend.disabled = true;
  const { chatMessagesByThread } = getState();
  const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
  const sendTs = new Date().toISOString();
  const userMsg = { role: "user", text: text || "", images: displayImages, timestamp: sendTs };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: sendTs, thinkingText: "", thinking: false };
  const nbt = { ...chatMessagesByThread };
  if (!nbt[animaName]) nbt[animaName] = {};
  nbt[animaName][threadId] = [...current, userMsg, streamingMsg];
  setState({ chatMessagesByThread: nbt }); renderConvMessages();
  if (!overrideImages) im?.clearImages();
  const ctrl = new AbortController();
  _setStreamController(ctrl); wsUpdateSendButton(true);

  try {
    const bodyObj = { message: text || "", from_person: getCurrentUser() || "guest", thread_id: threadId };
    if (images.length > 0) bodyObj.images = images;
    let talkingStarted = false;
    await streamChat(animaName, JSON.stringify(bodyObj), ctrl.signal, {
      ..._baseCallbacks(streamingMsg),
      onTextDelta: (d) => {
        streamingMsg.afterHeartbeatRelay = false;
        if (!talkingStarted) { setTalking(true); setExpression("neutral"); talkingStarted = true; }
        streamingMsg.text += d; scheduleStreamingUpdate(streamingMsg);
      },
      onHeartbeatRelayStart: () => { streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = ""; scheduleStreamingUpdate(streamingMsg); },
      onHeartbeatRelay: ({ text: t }) => { streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + t; scheduleStreamingUpdate(streamingMsg); },
      onHeartbeatRelayDone: () => { streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true; scheduleStreamingUpdate(streamingMsg); },
      onDone: ({ summary, emotion, images: di }) => {
        if (summary) { streamingMsg.text = summary; updateStreamingBubble(streamingMsg); }
        streamingMsg.images = di || [];
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => { setExpression("troubled"); streamingMsg.text += `\n[エラー: ${m}]`; updateStreamingBubble(streamingMsg); },
    });
    setTalking(false);
    _finalize(streamingMsg, animaName, threadId);
    const st = getState();
    const threadList = st.threads[animaName] || [];
    const entry = threadList.find(t => t.id === threadId);
    if (entry && entry.label === "新しいスレッド" && (text || "").trim()) {
      const lbl = (text || "").trim().slice(0, 20) + ((text || "").trim().length > 20 ? "..." : "");
      setState({ threads: { ...st.threads, [animaName]: threadList.map(t => t.id === threadId ? { ...t, label: lbl } : t) } });
      renderWsThreadTabs();
    }
    if (dom.convInput && dom.convInput.value.trim() === text.trim()) {
      dom.convInput.value = ""; dom.convInput.style.height = "auto"; wsClearDraft(animaName);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      streamingMsg.streaming = false; streamingMsg.activeTool = null;
      if (!streamingMsg.text) streamingMsg.text = "(中断されました)";
    } else {
      logger.error("Conversation stream error", { anima: animaName, error: err.message });
      streamingMsg.text = `[エラー] ${err.message}`; streamingMsg.streaming = false; streamingMsg.activeTool = null;
      setExpression("troubled");
    }
    setTalking(false); _commitThread(animaName, threadId); renderConvMessages();
  } finally {
    if (dom.convInput) dom.convInput.disabled = false;
    _setStreamController(null); wsUpdateSendButton(false); wsSaveDraft(); dom.convInput?.focus();
    _drainQueue();
  }
}

export async function resumeConversationStream(animaName) {
  if (_getStreamController()) return;
  const dom = _getDom();
  try {
    const active = await fetchActiveStream(animaName);
    if (!active || active.status !== "streaming") return;
    const progress = await fetchStreamProgress(animaName, active.response_id);
    if (!progress) return;
    const { activeThreadId, chatMessagesByThread } = getState();
    const threadId = activeThreadId || "default";
    const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
    const last = current.length > 0 ? current[current.length - 1] : null;
    let streamingMsg = (last && last.role === "assistant" && last.streaming) ? last : null;
    if (!streamingMsg) {
      streamingMsg = { role: "assistant", text: progress.full_text || "", streaming: true, activeTool: progress.active_tool || null };
    } else {
      Object.assign(streamingMsg, { text: progress.full_text || streamingMsg.text || "", activeTool: progress.active_tool || streamingMsg.activeTool || null, streaming: true });
    }
    const nbt = { ...chatMessagesByThread };
    if (!nbt[animaName]) nbt[animaName] = {};
    nbt[animaName][threadId] = streamingMsg === last ? [...current] : [...current, streamingMsg];
    setState({ chatMessagesByThread: nbt }); renderConvMessages();
    const ctrl = new AbortController();
    _setStreamController(ctrl); wsUpdateSendButton(true);
    const resumeBody = JSON.stringify({ message: "", from_person: getCurrentUser() || "guest", resume: active.response_id, last_event_id: progress.last_event_id || "" });
    await streamChat(animaName, resumeBody, ctrl.signal, {
      ..._baseCallbacks(streamingMsg),
      onTextDelta: (d) => { streamingMsg.text += d; scheduleStreamingUpdate(streamingMsg); },
      onDone: ({ summary, emotion, images: di }) => {
        if (summary) streamingMsg.text = summary;
        streamingMsg.images = di || [];
        _finalize(streamingMsg, animaName, threadId);
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => {
        streamingMsg.text += `\n[エラー: ${m}]`;
        _finalize(streamingMsg, animaName, threadId); setExpression("troubled");
      },
    });
    setTalking(false);
    if (streamingMsg.streaming) _finalize(streamingMsg, animaName, threadId);
  } catch (err) {
    if (err.name !== "AbortError") logger.error("Resume stream error", { anima: animaName, error: err.message });
  } finally {
    _setStreamController(null); wsUpdateSendButton(false); dom.convInput?.focus();
    _drainQueue();
  }
}

// ── Send Button / Pending Queue UI ──────────────────────

const _ICONS = {
  send: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg>`,
  stop: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg>`,
  interrupt: `<span class="chat-send-icon-group" aria-hidden="true"><svg class="chat-send-icon chat-send-icon-square" viewBox="0 0 24 24" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg><svg class="chat-send-icon" viewBox="0 0 24 24" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg></span>`,
};

export function wsUpdateSendButton(isStreaming) {
  const dom = _getDom(), q = _getPendingQueue();
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
  const dom = _getDom(), q = _getPendingQueue();
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
    if (delBtn) { e.stopPropagation(); q.splice(parseInt(delBtn.dataset.idx, 10), 1); wsShowPendingIndicator(); wsUpdateSendButton(!!_getStreamController()); return; }
    const item = e.target.closest(".pending-queue-item");
    if (!item) return;
    const removed = q.splice(parseInt(item.dataset.idx, 10), 1)[0];
    if (removed && dom.convInput) {
      dom.convInput.value = removed.text; dom.convInput.style.height = "auto";
      dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, isMobileView() ? 100 : 120) + "px";
      dom.convInput.focus();
    }
    wsShowPendingIndicator(); wsUpdateSendButton(!!_getStreamController());
  };
}

export function wsHidePendingIndicator() {
  const dom = _getDom();
  if (dom.convPending) dom.convPending.style.display = "none";
}

export function wsStopStreaming() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  const ctrl = _getStreamController();
  if (ctrl) ctrl.abort();
  fetch(`/api/animas/${encodeURIComponent(animaName)}/interrupt`, { method: "POST" }).catch(() => {});
}
