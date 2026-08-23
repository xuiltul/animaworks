// ── Workspace Chat Streaming ──────────────────────
// Message sending, streaming connection/resume, send button UI, queue management.
// Now delegates stream/queue state to ChatSessionManager; keeps Live2D hooks.

import { getState, setState } from "./state.js";
import { t } from "../../shared/i18n.js";
import { getCurrentUser } from "./login.js";
import { escapeHtml } from "./utils.js";
import { setExpression, setTalking } from "./live2d.js";
import { createLogger } from "../../shared/logger.js";
import { renderConvMessages, renderOpts } from "./chat-history.js";
import { renderStreamingBubbleInner, updateStreamingZone, TextAnimator, stripThinkTags } from "../../shared/chat/render-utils.js";
import { renderWsThreadTabs } from "./chat-thread.js";
import { wsSaveDraft, wsClearDraft, isMobileView } from "./chat-mobile.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";
import { fetchAvailableModels } from "./api.js";
import { modelAlias } from "./anima.js";

const logger = createLogger("ws-chat-streaming");
let _getDom = () => ({});
let _getImageManager = () => null;
let _convRafPending = false;
let _convLatestStreamingMsg = null;

export function initStreaming({ getDom, getImageManager }) {
  _getDom = getDom;
  _getImageManager = getImageManager;
}

/**
 * Populate the per-message model picker from GET /api/system/available-models.
 * The empty option (“anima既定”) sends no model so the anima default is used.
 */
export async function initChatModelPicker() {
  const dom = _getDom();
  const select = dom.convModel;
  if (!select) return;
  let models = [];
  try {
    const data = await fetchAvailableModels();
    models = data?.models || [];
  } catch (err) {
    logger.error("Failed to load chat model picker options", { error: err?.message });
    return;
  }
  const defaultOpt = `<option value="">${t("chat.model_default")}</option>`;
  const options = models
    .filter(m => m && m.id)
    .map(m => {
      const label = modelAlias(m.id) || m.label || m.id;
      return `<option value="${escapeHtml(m.id)}">${escapeHtml(label)}</option>`;
    });
  select.innerHTML = defaultOpt + options.join("");
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
    onCompressionStart: () => { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg, "text"); },
    onCompressionEnd: () => { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg, "text"); },
    onToolStart: (n, detail) => {
      streamingMsg.activeTool = n;
      if (!streamingMsg.toolHistory) streamingMsg.toolHistory = [];
      streamingMsg.toolHistory.push({ tool_name: n, tool_id: detail?.tool_id || "", started_at: Date.now() });
      setExpression("thinking");
      updateStreamingBubble(streamingMsg, "tools");
    },
    onToolDetail: (_toolName, detailText, info) => {
      if (streamingMsg.toolHistory && info?.tool_id) {
        for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) {
          const entry = streamingMsg.toolHistory[i];
          if (entry.tool_id === info.tool_id && !entry.completed) {
            entry.detail = detailText;
            break;
          }
        }
      }
      updateStreamingBubble(streamingMsg, "tools");
    },
    onToolEnd: (detail) => {
      streamingMsg.activeTool = null;
      if (streamingMsg.toolHistory && detail?.tool_id) {
        for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) {
          const entry = streamingMsg.toolHistory[i];
          if (entry.tool_id === detail.tool_id && !entry.completed) {
            entry.completed = true;
            entry.duration_ms = Date.now() - entry.started_at;
            entry.result_summary = detail.result_summary || "";
            entry.input_summary = detail.input_summary || "";
            entry.is_error = !!detail.is_error;
            break;
          }
        }
      }
      setExpression("neutral");
      updateStreamingBubble(streamingMsg, "tools");
    },
    onThinkingStart: () => { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; updateStreamingBubble(streamingMsg, "thinking"); },
    onThinkingDelta: (t) => { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; scheduleStreamingUpdate(streamingMsg, "thinking"); },
    onThinkingEnd: () => { streamingMsg.thinking = false; updateStreamingBubble(streamingMsg, "thinking"); },
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

let _convLatestZone = "all";

export function scheduleStreamingUpdate(msg, zone = "text") {
  _convLatestStreamingMsg = msg;
  if (_convLatestZone !== "all") _convLatestZone = zone;
  if (_convRafPending) return;
  _convRafPending = true;
  requestAnimationFrame(() => {
    _convRafPending = false;
    const z = _convLatestZone;
    _convLatestZone = "all";
    if (_convLatestStreamingMsg) updateStreamingBubble(_convLatestStreamingMsg, z);
  });
}

export function updateStreamingBubble(msg, zone = "all") {
  const dom = _getDom();
  if (!dom.convMessages) return;
  let bubble = null;
  if (msg.streamId) {
    bubble = dom.convMessages.querySelector(`.chat-bubble.assistant.streaming[data-stream-id="${CSS.escape(String(msg.streamId))}"]`);
  }
  if (!bubble) return;
  updateStreamingZone(bubble, msg, renderOpts(), zone);
  if (zone !== "thinking") {
    requestAnimationFrame(() => { dom.convMessages.scrollTop = dom.convMessages.scrollHeight; });
  }
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

function _modelSelection() {
  const dom = _getDom();
  return dom.convModel?.value || "";
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

  const _wsToolDetailTimers = new Map();
  const _throttledWsToolDetail = (toolId) => {
    if (_wsToolDetailTimers.has(toolId)) return;
    updateStreamingBubble(streamingMsg, "tools");
    _wsToolDetailTimers.set(toolId, setTimeout(() => { _wsToolDetailTimers.delete(toolId); updateStreamingBubble(streamingMsg, "tools"); }, 200));
  };
  let _textAnimator = null;
  let _thinkingAnimator = null;

  const { success, error } = await mgr.sendChat(anima, thread, text, {
    model: _modelSelection() || undefined,
    images,
    displayImages,
    callbacks: {
      onStreamCreated: (msg) => {
        streamingMsg = msg;
        _textAnimator = new TextAnimator({
          onUpdate: (displayText) => {
            if (!streamingMsg) return;
            streamingMsg._displayText = displayText;
            scheduleStreamingUpdate(streamingMsg, "text");
          },
        });
        _textAnimator.start();
        renderConvMessages();
      },
      onTextDelta: (d) => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.afterHeartbeatRelay = false;
        if (!talkingStarted) { setTalking(true); setExpression("neutral"); talkingStarted = true; }
        streamingMsg.text += d;
        if (_textAnimator) _textAnimator.push(d);
      },
      onCompressionStart: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg, "text"); } },
      onCompressionEnd: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg, "text"); } },
      onToolStart: (n, detail) => { if (streamingMsg?.streaming) { streamingMsg.activeTool = n; if (!streamingMsg.toolHistory) streamingMsg.toolHistory = []; streamingMsg.toolHistory.push({ tool_name: n, tool_id: detail?.tool_id || "", started_at: Date.now() }); setExpression("thinking"); updateStreamingBubble(streamingMsg, "tools"); } },
      onToolDetail: (_toolName, detailText, info) => { if (streamingMsg?.streaming && streamingMsg.toolHistory && info?.tool_id) { for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) { const entry = streamingMsg.toolHistory[i]; if (entry.tool_id === info.tool_id && !entry.completed) { entry.detail = detailText; break; } } } _throttledWsToolDetail(info?.tool_id || "_"); },
      onToolEnd: (detail) => { if (streamingMsg?.streaming) { streamingMsg.activeTool = null; if (streamingMsg.toolHistory && detail?.tool_id) { for (let i = streamingMsg.toolHistory.length - 1; i >= 0; i--) { const entry = streamingMsg.toolHistory[i]; if (entry.tool_id === detail.tool_id && !entry.completed) { entry.completed = true; entry.duration_ms = Date.now() - entry.started_at; break; } } } setExpression("neutral"); updateStreamingBubble(streamingMsg, "tools"); } },
      onThinkingStart: () => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.thinkingText = ""; streamingMsg.thinking = true;
        _thinkingAnimator = new TextAnimator({
          onUpdate: (displayText) => {
            if (!streamingMsg) return;
            streamingMsg._displayThinkingText = displayText;
            scheduleStreamingUpdate(streamingMsg, "thinking");
          },
        });
        _thinkingAnimator.start();
        updateStreamingBubble(streamingMsg, "thinking");
      },
      onThinkingDelta: (t) => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t;
        if (_thinkingAnimator) _thinkingAnimator.push(t);
      },
      onThinkingEnd: () => {
        if (!streamingMsg?.streaming) return;
        if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
        delete streamingMsg._displayThinkingText;
        streamingMsg.thinking = false;
        updateStreamingBubble(streamingMsg, "thinking");
      },
      onHeartbeatRelayStart: () => { if (streamingMsg?.streaming) { streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = ""; scheduleStreamingUpdate(streamingMsg, "text"); } },
      onHeartbeatRelay: ({ text: t }) => { if (streamingMsg?.streaming) { streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + t; scheduleStreamingUpdate(streamingMsg, "text"); } },
      onHeartbeatRelayDone: () => { if (streamingMsg?.streaming) { streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true; scheduleStreamingUpdate(streamingMsg, "text"); } },
      onDone: ({ summary, emotion, images: di, thinkingSummary }) => {
        if (_textAnimator) _textAnimator.flush();
        if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
        if (streamingMsg) {
          let text = summary || streamingMsg.text || "";
          const { thinking: strippedThinking, response: cleanResponse } = stripThinkTags(text);
          streamingMsg.text = cleanResponse || streamingMsg.text || "";
          if (strippedThinking && !streamingMsg.thinkingText) {
            streamingMsg.thinkingText = strippedThinking;
          }
          delete streamingMsg._displayText;
          delete streamingMsg._displayThinkingText;
          streamingMsg.images = di || [];
          if (!streamingMsg.thinkingText && thinkingSummary) {
            streamingMsg.thinkingText = thinkingSummary;
          }
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
          updateStreamingBubble(streamingMsg);
        }
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => {
        if (_textAnimator) _textAnimator.flush();
        if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
        setExpression("troubled");
        if (streamingMsg) { streamingMsg.text += `\n${t("chat.error_prefix")} ${m}`; delete streamingMsg._displayText; delete streamingMsg._displayThinkingText; updateStreamingBubble(streamingMsg); }
      },
      onAbort: () => {
        if (_textAnimator) _textAnimator.flush();
        if (_thinkingAnimator) { _thinkingAnimator.flush(); _thinkingAnimator = null; }
        if (streamingMsg) {
          delete streamingMsg._displayText;
          delete streamingMsg._displayThinkingText;
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
          if (!streamingMsg.text) streamingMsg.text = t("chat.interrupted");
        }
      },
    },
    onFinally: () => {
      if (_textAnimator) { _textAnimator.stop(); _textAnimator = null; }
      if (_thinkingAnimator) { _thinkingAnimator.stop(); _thinkingAnimator = null; }
      for (const t of _wsToolDetailTimers.values()) clearTimeout(t);
      _wsToolDetailTimers.clear();
      try {
        setTalking(false);
        if (streamingMsg?.streaming) {
          streamingMsg.streaming = false;
          if (!streamingMsg.text) streamingMsg.text = t("chat.empty_response");
        }
        renderConvMessages();
        renderWsThreadTabs();
        if (dom.convInput) dom.convInput.disabled = false;
        wsUpdateSendButton(false); wsSaveDraft(); dom.convInput?.focus();

        const st = getState();
        const threadList = st.threads[anima] || [];
        const entry = threadList.find(t => t.id === thread);
        if (entry && entry.label === t("thread.new") && (text || "").trim()) {
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
  let _resumeAnimator = null;
  let _resumeThinkingAnimator = null;

  const { success } = await mgr.resumeStream(animaName, threadId, {
    callbacks: {
      onStreamCreated: (msg) => {
        streamingMsg = msg;
        const resumeBase = msg.text || "";
        _resumeAnimator = new TextAnimator({
          onUpdate: (displayText) => {
            if (!streamingMsg) return;
            streamingMsg._displayText = resumeBase + displayText;
            scheduleStreamingUpdate(streamingMsg, "text");
          },
        });
        _resumeAnimator.start();
        renderConvMessages();
      },
      onTextDelta: (d) => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.text += d;
        if (_resumeAnimator) _resumeAnimator.push(d);
      },
      onCompressionStart: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = true; updateStreamingBubble(streamingMsg, "text"); } },
      onCompressionEnd: () => { if (streamingMsg?.streaming) { streamingMsg.compressing = false; updateStreamingBubble(streamingMsg, "text"); } },
      onToolStart: (n) => { if (streamingMsg?.streaming) { streamingMsg.activeTool = n; setExpression("thinking"); updateStreamingBubble(streamingMsg, "tools"); } },
      onToolEnd: () => { if (streamingMsg?.streaming) { streamingMsg.activeTool = null; setExpression("neutral"); updateStreamingBubble(streamingMsg, "tools"); } },
      onThinkingStart: () => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.thinkingText = ""; streamingMsg.thinking = true;
        _resumeThinkingAnimator = new TextAnimator({
          onUpdate: (displayText) => {
            if (!streamingMsg) return;
            streamingMsg._displayThinkingText = displayText;
            scheduleStreamingUpdate(streamingMsg, "thinking");
          },
        });
        _resumeThinkingAnimator.start();
        updateStreamingBubble(streamingMsg, "thinking");
      },
      onThinkingDelta: (t) => {
        if (!streamingMsg?.streaming) return;
        streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t;
        if (_resumeThinkingAnimator) _resumeThinkingAnimator.push(t);
      },
      onThinkingEnd: () => {
        if (!streamingMsg?.streaming) return;
        if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
        delete streamingMsg._displayThinkingText;
        streamingMsg.thinking = false;
        updateStreamingBubble(streamingMsg, "thinking");
      },
      onDone: ({ summary, emotion, images: di, thinkingSummary }) => {
        if (_resumeAnimator) _resumeAnimator.flush();
        if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
        if (streamingMsg) {
          let text = summary || streamingMsg.text || "";
          const { thinking: strippedThinking, response: cleanResponse } = stripThinkTags(text);
          streamingMsg.text = cleanResponse || streamingMsg.text || "";
          if (strippedThinking && !streamingMsg.thinkingText) {
            streamingMsg.thinkingText = strippedThinking;
          }
          delete streamingMsg._displayText;
          delete streamingMsg._displayThinkingText;
          streamingMsg.images = di || [];
          if (!streamingMsg.thinkingText && thinkingSummary) {
            streamingMsg.thinkingText = thinkingSummary;
          }
          streamingMsg.streaming = false; streamingMsg.activeTool = null;
        }
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: m }) => {
        if (_resumeAnimator) _resumeAnimator.flush();
        if (_resumeThinkingAnimator) { _resumeThinkingAnimator.flush(); _resumeThinkingAnimator = null; }
        if (streamingMsg) { streamingMsg.text += `\n${t("chat.error_prefix")} ${m}`; delete streamingMsg._displayText; delete streamingMsg._displayThinkingText; streamingMsg.streaming = false; }
        setExpression("troubled");
      },
    },
    onFinally: () => {
      if (_resumeAnimator) { _resumeAnimator.stop(); _resumeAnimator = null; }
      if (_resumeThinkingAnimator) { _resumeThinkingAnimator.stop(); _resumeThinkingAnimator = null; }
      try {
        setTalking(false);
        if (streamingMsg?.streaming) {
          streamingMsg.streaming = false;
          if (!streamingMsg.text) streamingMsg.text = t("chat.empty_response");
        }
        delete streamingMsg?._displayText;
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
  if (dom.convPendingLabel) dom.convPendingLabel.textContent = t("chat.queue_count", { count: q.length });
  dom.convPendingList.innerHTML = q.map((p, i) => {
    const txt = escapeHtml(p.text.length > 50 ? p.text.slice(0, 50) + "…" : p.text);
    const img = p.images?.length ? ` <span style="opacity:0.6">${t("chat.image_count", { count: p.images.length })}</span>` : "";
    return `<div class="pending-queue-item" data-idx="${i}"><span class="pending-queue-item-num">${i + 1}.</span><span class="pending-queue-item-text">${txt || t("chat.image_only")}${img}</span><button class="pending-queue-item-del" data-idx="${i}" type="button">✕</button></div>`;
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
