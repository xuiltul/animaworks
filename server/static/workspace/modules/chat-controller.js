// ── Workspace Chat Controller ──────────────────────
// Orchestrates conversation in the workspace overlay using shared/chat/ modules.
// Workspace-specific: Live2D hooks, 3D office integration, conversation overlay DOM.

import { getState, setState } from "./state.js";
import { fetchConversationHistory, greetAnima } from "./api.js";
import { escapeHtml, renderSimpleMarkdown, smartTimestamp } from "./utils.js";
import { getCurrentUser } from "./login.js";
import { initBustup, setCharacter, setExpression, setTalking, onClick as onBustupClick, setLive2dAppearance } from "./live2d.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../../shared/chat-stream.js";
import { createImageInput, initLightbox, renderChatImages } from "../../shared/image-input.js";
import { initVoiceUI, destroyVoiceUI, updateVoiceUIAnima } from "../../modules/voice-ui.js";
import { SwipeHandler } from "../../modules/touch.js";
import { createLogger } from "../../shared/logger.js";

import {
  renderHistoryMessage, renderSessionDivider, bindToolCallHandlers,
  renderLiveBubble, renderStreamingBubbleInner,
} from "../../shared/chat/render-utils.js";
import { createScrollObserver } from "../../shared/chat/scroll-observer.js";
import { getDraftKey, saveDraft, loadDraft, clearDraft } from "../../shared/chat/draft.js";
import {
  renderThreadTabsHtml, createThread as sharedCreateThread,
  closeThread as sharedCloseThread, threadTimeValue,
} from "../../shared/chat/thread-logic.js";
import { createHistoryState, applyHistoryData } from "../../shared/chat/history-loader.js";

const logger = createLogger("ws-chat");

// ── Module State ──────────────────────
let _dom = {};
let bustupInitialized = false;
let convStreamController = null;
let convImageInputManager = null;
let convPendingQueue = [];
let _scrollObserver = null;
let _swiperInstance = null;
let _mobileMediaQuery = null;
const _historyState = {};
const HISTORY_PAGE_SIZE = 50;
const TOOL_RESULT_TRUNCATE = 500;

let _convRafPending = false;
let _convLatestStreamingMsg = null;

let _greetingInFlight = false;
const _GREET_COOLDOWN_MS = 3600 * 1000;
const _lastGreetTime = {};

// ── Render Options Builder ──────────────────────

function _renderOpts() {
  return {
    escapeHtml,
    renderMarkdown: renderSimpleMarkdown,
    smartTimestamp,
    renderChatImages,
    animaName: getState().conversationAnima,
    truncateLen: TOOL_RESULT_TRUNCATE,
    labels: {
      thinking: "考え中...",
      toolRunning: (tool) => `${escapeHtml(tool)} を実行中...`,
      heartbeatRelay: "ハートビート処理中...",
      heartbeatRelayDone: "応答を準備中...",
    },
  };
}

// ── Draft ──────────────────────

function _wsDraftKey(animaName) {
  return getDraftKey("workspace-conv", getCurrentUser() || "guest", animaName);
}

function _wsSaveDraft() {
  const animaName = getState().conversationAnima;
  if (!animaName || !_dom.convInput) return;
  saveDraft(_wsDraftKey(animaName), _dom.convInput.value || "");
}

function _wsLoadDraft(animaName) {
  if (!animaName) return "";
  return loadDraft(_wsDraftKey(animaName));
}

function _wsClearDraft(animaName) {
  if (!animaName) return;
  clearDraft(_wsDraftKey(animaName));
}

// ── Conversation Lifecycle ──────────────────────

export async function openConversation(animaName) {
  if (!_dom.convOverlay) return;

  _wsSaveDraft();
  const wasVoiceActive = updateVoiceUIAnima(animaName);
  setState({ conversationOpen: true, conversationAnima: animaName, activeThreadId: "default" });

  const { threads } = getState();
  if (!threads[animaName]) {
    setState({ threads: { ...threads, [animaName]: [{ id: "default", label: "メイン", unread: false }] } });
  }

  _dom.convOverlay.classList.remove("hidden");
  if (_dom.convAnimaName) _dom.convAnimaName.textContent = animaName;
  _updateConvInputPlaceholder();
  if (_dom.convInput) {
    _dom.convInput.value = _wsLoadDraft(animaName);
    _dom.convInput.style.height = "auto";
    const maxH = _isMobileView() ? 100 : 120;
    _dom.convInput.style.height = Math.min(_dom.convInput.scrollHeight, maxH) + "px";
  }

  _closeMobileSidebar();
  _closeMobileCharacter();

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

  await _loadAndRenderConvMessages(animaName);
  _renderWsThreadTabs();
  _triggerGreeting(animaName);

  _dom.convInput?.focus();

  const convInputArea = document.querySelector(".ws-conv-input-area");
  if (convInputArea && animaName) {
    initVoiceUI(convInputArea, animaName, _buildVoiceChatCallbacks(animaName), { autoConnect: wasVoiceActive });
  }
}

export function closeConversation() {
  if (!_dom.convOverlay) return;

  _wsSaveDraft();
  _dom.convOverlay.classList.add("hidden");

  _closeMobileSidebar();
  _closeMobileCharacter();
  _closeMobileMemory();
  _cleanupMobileResources();

  if (_scrollObserver) { _scrollObserver.disconnect(); _scrollObserver = null; }

  const animaName = getState().conversationAnima;
  if (animaName) delete _historyState[animaName];

  setState({ conversationOpen: false, conversationAnima: null });
  setTalking(false);

  convPendingQueue = [];
  _wsHidePendingIndicator();

  if (convStreamController) { convStreamController.abort(); convStreamController = null; }
  destroyVoiceUI();
}

// ── Thread Tabs ──────────────────────

function _renderWsThreadTabs() {
  const container = _dom.threadTabs;
  const animaName = getState().conversationAnima;
  if (!container || !animaName) return;

  const list = getState().threads[animaName] || [{ id: "default", label: "メイン", unread: false }];
  const activeThreadId = getState().activeThreadId || "default";

  container.innerHTML = renderThreadTabsHtml(list, activeThreadId, {
    escapeHtml,
    newBtnId: "wsNewThreadBtn",
    moreSelectId: "wsThreadMoreSelect",
  });

  container.querySelectorAll(".thread-tab").forEach(btn => {
    btn.addEventListener("click", e => {
      const tid = e.target.dataset.thread;
      if (tid) _selectWsThread(tid);
    });
  });
  container.querySelectorAll(".thread-tab-close").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const tid = e.target.dataset.thread;
      if (tid) _closeWsThread(tid);
    });
  });
  const newBtn = document.getElementById("wsNewThreadBtn");
  if (newBtn) newBtn.addEventListener("click", () => _createWsNewThread());
}

async function _selectWsThread(threadId) {
  const current = getState().activeThreadId;
  if (threadId === current) return;

  setState({ activeThreadId: threadId });
  _renderWsThreadTabs();

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const hs = _historyState[animaName]?.[threadId];
  const needLoad = !hs || hs.sessions.length === 0;

  if (needLoad) {
    if (!_historyState[animaName]) _historyState[animaName] = {};
    _historyState[animaName][threadId] = { ...createHistoryState(), loading: true };
    _renderConvMessages();

    try {
      const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, null, threadId);
      _historyState[animaName][threadId] = createHistoryState();
      applyHistoryData(_historyState[animaName][threadId], data);
    } catch {
      _historyState[animaName][threadId] = createHistoryState();
    }
  }

  _renderConvMessages();
  _refreshSentinel();
}

function _createWsNewThread() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const { threads, chatMessagesByThread } = getState();
  const list = threads[animaName] || [{ id: "default", label: "メイン", unread: false }];
  const { updatedList, newThreadId } = sharedCreateThread(list, animaName);

  const nextThreads = { ...threads, [animaName]: updatedList };
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][newThreadId] = [];

  setState({ threads: nextThreads, chatMessagesByThread: nextByThread, activeThreadId: newThreadId });

  if (!_historyState[animaName]) _historyState[animaName] = {};
  _historyState[animaName][newThreadId] = createHistoryState();

  _renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}

function _closeWsThread(threadId) {
  if (threadId === "default") return;
  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const { threads, chatMessagesByThread, activeThreadId } = getState();
  const list = threads[animaName];
  if (!list || !list.some(t => t.id === threadId)) return;

  const nextList = sharedCloseThread(list, threadId);
  const nextThreads = { ...threads, [animaName]: nextList };
  const nextByThread = { ...chatMessagesByThread };
  if (nextByThread[animaName]) {
    const { [threadId]: _, ...rest } = nextByThread[animaName];
    nextByThread[animaName] = rest;
  }
  delete _historyState[animaName]?.[threadId];

  const switchToDefault = activeThreadId === threadId;
  setState({
    threads: nextThreads,
    chatMessagesByThread: nextByThread,
    ...(switchToDefault ? { activeThreadId: "default" } : {}),
  });

  _renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}

// ── Chat Rendering ──────────────────────

function _renderConvMessages() {
  if (!_dom.convMessages) return;

  const { activeThreadId, conversationAnima } = getState();
  const animaName = conversationAnima;
  const threadMessages = getState().chatMessagesByThread?.[animaName]?.[activeThreadId || "default"] || [];
  const hs = animaName ? _historyState[animaName]?.[activeThreadId || "default"] : null;

  if ((!hs || hs.sessions.length === 0) && threadMessages.length === 0) {
    if (hs && hs.loading) {
      _dom.convMessages.innerHTML = '<div class="chat-empty"><span class="tool-spinner"></span> 読み込み中...</div>';
    } else {
      _dom.convMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
    }
    return;
  }

  const opts = _renderOpts();
  let html = "";

  if (hs && hs.hasMore) {
    if (hs.loading) {
      html += '<div class="history-loading-more"><span class="tool-spinner"></span> 過去の会話を読み込み中...</div>';
    }
    html += '<div class="chat-load-sentinel"></div>';
  }

  if (hs && hs.sessions.length > 0) {
    for (let si = 0; si < hs.sessions.length; si++) {
      const session = hs.sessions[si];
      html += renderSessionDivider(session, si === 0, opts);
      if (session.messages) {
        for (const msg of session.messages) html += renderHistoryMessage(msg, opts);
      }
    }
  }

  if (threadMessages.length > 0) {
    if (hs && hs.sessions.length > 0) {
      html += '<div class="session-divider"><span class="session-divider-label">現在のセッション</span></div>';
    }
    html += threadMessages.map(m => renderLiveBubble(m, opts)).join("");
  }

  _dom.convMessages.innerHTML = html;
  bindToolCallHandlers(_dom.convMessages);
  _refreshSentinel();
  _dom.convMessages.scrollTop = _dom.convMessages.scrollHeight;
}

async function _loadAndRenderConvMessages(animaName) {
  if (!animaName) return;

  const threadId = getState().activeThreadId || "default";

  if (!_historyState[animaName]) _historyState[animaName] = {};
  _historyState[animaName][threadId] = { ...createHistoryState(), loading: true };

  const { chatMessagesByThread } = getState();
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][threadId] = [];
  setState({ chatMessagesByThread: nextByThread });
  _renderConvMessages();

  try {
    const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, null, threadId);
    _historyState[animaName][threadId] = createHistoryState();
    applyHistoryData(_historyState[animaName][threadId], data);
  } catch (err) {
    logger.error("Failed to load conversation", { anima: animaName, error: err.message });
    _historyState[animaName][threadId] = createHistoryState();
  }

  _renderConvMessages();
  _setupScrollObserver();
  _resumeConversationStream(animaName);
}

// ── Infinite Scroll ──────────────────────

function _setupScrollObserver() {
  if (_scrollObserver) _scrollObserver.disconnect();
  if (!_dom.convMessages) return;
  _scrollObserver = createScrollObserver({ container: _dom.convMessages, onLoadMore: _loadMoreHistory });
  _scrollObserver.observe();
}

function _refreshSentinel() {
  if (_scrollObserver) _scrollObserver.refresh();
}

async function _loadMoreHistory() {
  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  const hs = _historyState[animaName]?.[threadId];
  if (!hs || !hs.hasMore || hs.loading) return;

  hs.loading = true;
  const existingIndicator = _dom.convMessages.querySelector(".history-loading-more");
  if (!existingIndicator) {
    const indicator = document.createElement("div");
    indicator.className = "history-loading-more";
    indicator.innerHTML = '<span class="tool-spinner"></span> 過去の会話を読み込み中...';
    _dom.convMessages.insertBefore(indicator, _dom.convMessages.firstChild);
  }

  try {
    const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, hs.nextBefore, threadId);
    applyHistoryData(hs, data, { prepend: true });
  } catch (err) {
    logger.error("Failed to load more history", { anima: animaName, error: err.message });
    hs.hasMore = false;
  }
  hs.loading = false;

  const prevScrollHeight = _dom.convMessages.scrollHeight;
  _renderConvMessages();
  const newScrollHeight = _dom.convMessages.scrollHeight;
  _dom.convMessages.scrollTop += (newScrollHeight - prevScrollHeight);
}

// ── Streaming ──────────────────────

function _scheduleStreamingUpdate(msg) {
  _convLatestStreamingMsg = msg;
  if (_convRafPending) return;
  _convRafPending = true;
  requestAnimationFrame(() => {
    _convRafPending = false;
    if (_convLatestStreamingMsg) _updateStreamingBubble(_convLatestStreamingMsg);
  });
}

function _updateStreamingBubble(msg) {
  if (!_dom.convMessages) return;
  const bubbles = _dom.convMessages.querySelectorAll(".chat-bubble.assistant.streaming");
  const bubble = bubbles[bubbles.length - 1];
  if (!bubble) return;

  bubble.innerHTML = renderStreamingBubbleInner(msg, _renderOpts());
  _dom.convMessages.scrollTop = _dom.convMessages.scrollHeight;
}

export function submitConversation() {
  const text = _dom.convInput?.value?.trim();
  const hasImages = convImageInputManager && convImageInputManager.getImageCount() > 0;
  const isStreaming = !!convStreamController;

  if (!isStreaming) {
    if (text || hasImages) {
      convPendingQueue.push({
        text: text || "",
        images: convImageInputManager?.getPendingImages() || [],
        displayImages: convImageInputManager?.getDisplayImages() || [],
      });
      _dom.convInput.value = "";
      _dom.convInput.style.height = "auto";
      _wsSaveDraft();
      convImageInputManager?.clearImages();
    }
    if (convPendingQueue.length === 0) return;
    const next = convPendingQueue.shift();
    _wsShowPendingIndicator();
    if (convPendingQueue.length === 0) _wsHidePendingIndicator();
    _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
    return;
  }

  if (text || hasImages) {
    convPendingQueue.push({
      text: text || "",
      images: convImageInputManager?.getPendingImages() || [],
      displayImages: convImageInputManager?.getDisplayImages() || [],
    });
    _dom.convInput.value = "";
    _dom.convInput.style.height = "auto";
    _wsSaveDraft();
    convImageInputManager?.clearImages();
    _wsShowPendingIndicator();
    _wsUpdateSendButton(true);
    return;
  }

  if (convPendingQueue.length > 0) { _wsStopStreaming(); return; }
  _wsStopStreaming();
}

export function addToQueue() {
  const text = _dom.convInput?.value?.trim();
  const hasImages = convImageInputManager && convImageInputManager.getImageCount() > 0;
  if (!text && !hasImages) return;
  convPendingQueue.push({
    text: text || "",
    images: convImageInputManager?.getPendingImages() || [],
    displayImages: convImageInputManager?.getDisplayImages() || [],
  });
  if (_dom.convInput) { _dom.convInput.value = ""; _dom.convInput.style.height = "auto"; }
  _wsSaveDraft();
  convImageInputManager?.clearImages();
  _wsShowPendingIndicator();
  _wsUpdateSendButton(!!convStreamController);
}

async function _sendConversation(text, overrideImages = null) {
  const images = overrideImages?.images || convImageInputManager?.getPendingImages() || [];
  const displayImages = overrideImages?.displayImages || convImageInputManager?.getDisplayImages() || [];
  if (!text && images.length === 0) return;

  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  _dom.convInput.value = "";
  _dom.convInput.disabled = true;
  _dom.convSend.disabled = true;

  const { chatMessagesByThread } = getState();
  const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
  const sendTs = new Date().toISOString();
  const userMsg = { role: "user", text: text || "", images: displayImages, timestamp: sendTs };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: sendTs, thinkingText: "", thinking: false };
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][threadId] = [...current, userMsg, streamingMsg];
  setState({ chatMessagesByThread: nextByThread });
  _renderConvMessages();

  if (!overrideImages) convImageInputManager?.clearImages();

  convStreamController = new AbortController();
  _wsUpdateSendButton(true);

  try {
    let sendSucceeded = false;
    const userName = getCurrentUser() || "guest";
    const bodyObj = { message: text || "", from_person: userName, thread_id: threadId };
    if (images.length > 0) bodyObj.images = images;
    const body = JSON.stringify(bodyObj);

    let talkingStarted = false;

    await streamChat(animaName, body, convStreamController.signal, {
      onTextDelta: (deltaText) => {
        streamingMsg.afterHeartbeatRelay = false;
        if (!talkingStarted) { setTalking(true); setExpression("neutral"); talkingStarted = true; }
        streamingMsg.text += deltaText;
        _scheduleStreamingUpdate(streamingMsg);
      },
      onToolStart: (toolName) => { streamingMsg.activeTool = toolName; setExpression("thinking"); _updateStreamingBubble(streamingMsg); },
      onToolEnd: () => { streamingMsg.activeTool = null; setExpression("neutral"); _updateStreamingBubble(streamingMsg); },
      onHeartbeatRelayStart: () => { streamingMsg.heartbeatRelay = true; streamingMsg.heartbeatText = ""; _scheduleStreamingUpdate(streamingMsg); },
      onHeartbeatRelay: ({ text: t }) => { streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + t; _scheduleStreamingUpdate(streamingMsg); },
      onHeartbeatRelayDone: () => { streamingMsg.heartbeatRelay = false; streamingMsg.heartbeatText = ""; streamingMsg.afterHeartbeatRelay = true; _scheduleStreamingUpdate(streamingMsg); },
      onThinkingStart: () => { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; _updateStreamingBubble(streamingMsg); },
      onThinkingDelta: (t) => { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; _scheduleStreamingUpdate(streamingMsg); },
      onThinkingEnd: () => { streamingMsg.thinking = false; _updateStreamingBubble(streamingMsg); },
      onDone: ({ summary, emotion, images: doneImages }) => {
        if (summary) { streamingMsg.text = summary; _updateStreamingBubble(streamingMsg); }
        streamingMsg.images = doneImages || [];
        setExpression(emotion);
        setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: errorMsg }) => {
        setExpression("troubled");
        streamingMsg.text += `\n[エラー: ${errorMsg}]`;
        _updateStreamingBubble(streamingMsg);
      },
    });

    setTalking(false);

    streamingMsg.streaming = false;
    if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
    const st = getState();
    const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
    const nbt = { ...st.chatMessagesByThread };
    if (!nbt[animaName]) nbt[animaName] = {};
    nbt[animaName] = { ...nbt[animaName], [threadId]: [...arr] };
    setState({ chatMessagesByThread: nbt });

    const threadList = st.threads[animaName] || [];
    const entry = threadList.find(t => t.id === threadId);
    if (entry && entry.label === "新しいスレッド" && (text || "").trim()) {
      const firstLine = (text || "").trim().slice(0, 20) + ((text || "").trim().length > 20 ? "..." : "");
      const nextThreads = { ...st.threads, [animaName]: threadList.map(t => t.id === threadId ? { ...t, label: firstLine } : t) };
      setState({ threads: nextThreads });
      _renderWsThreadTabs();
    }
    _renderConvMessages();
    sendSucceeded = true;

    if (sendSucceeded && _dom.convInput && _dom.convInput.value.trim() === text.trim()) {
      _dom.convInput.value = "";
      _dom.convInput.style.height = "auto";
      _wsClearDraft(animaName);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      streamingMsg.streaming = false; streamingMsg.activeTool = null;
      if (!streamingMsg.text) streamingMsg.text = "(中断されました)";
    } else {
      logger.error("Conversation stream error", { anima: animaName, error: err.message });
      streamingMsg.text = `[エラー] ${err.message}`;
      streamingMsg.streaming = false; streamingMsg.activeTool = null;
      setExpression("troubled");
    }
    setTalking(false);
    const st = getState();
    const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
    const nbt = { ...st.chatMessagesByThread };
    if (!nbt[animaName]) nbt[animaName] = {};
    nbt[animaName] = { ...nbt[animaName], [threadId]: [...arr] };
    setState({ chatMessagesByThread: nbt });
    _renderConvMessages();
  } finally {
    convStreamController = null;
    _wsUpdateSendButton(false);
    _wsSaveDraft();
    _dom.convInput?.focus();

    if (convPendingQueue.length > 0) {
      const next = convPendingQueue.shift();
      _wsShowPendingIndicator();
      if (convPendingQueue.length === 0) _wsHidePendingIndicator();
      setTimeout(() => _sendConversation(next.text, { images: next.images, displayImages: next.displayImages }), 150);
    }
  }
}

async function _resumeConversationStream(animaName) {
  if (convStreamController) return;

  try {
    const active = await fetchActiveStream(animaName);
    if (!active || active.status !== "streaming") return;
    const progress = await fetchStreamProgress(animaName, active.response_id);
    if (!progress) return;

    const { activeThreadId, chatMessagesByThread } = getState();
    const threadId = activeThreadId || "default";
    const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
    let streamingMsg = null;
    if (current.length > 0) {
      const last = current[current.length - 1];
      if (last && last.role === "assistant" && last.streaming) streamingMsg = last;
    }
    if (!streamingMsg) {
      streamingMsg = { role: "assistant", text: progress.full_text || "", streaming: true, activeTool: progress.active_tool || null };
    } else {
      streamingMsg.text = progress.full_text || streamingMsg.text || "";
      streamingMsg.activeTool = progress.active_tool || streamingMsg.activeTool || null;
      streamingMsg.streaming = true;
    }
    const nextByThread = { ...chatMessagesByThread };
    if (!nextByThread[animaName]) nextByThread[animaName] = {};
    nextByThread[animaName][threadId] = streamingMsg === current[current.length - 1] ? [...current] : [...current, streamingMsg];
    setState({ chatMessagesByThread: nextByThread });
    _renderConvMessages();

    convStreamController = new AbortController();
    _wsUpdateSendButton(true);

    const resumeBody = JSON.stringify({ message: "", from_person: getCurrentUser() || "guest", resume: active.response_id, last_event_id: progress.last_event_id || "" });

    await streamChat(animaName, resumeBody, convStreamController.signal, {
      onTextDelta: (d) => { streamingMsg.text += d; _scheduleStreamingUpdate(streamingMsg); },
      onToolStart: (n) => { streamingMsg.activeTool = n; setExpression("thinking"); _updateStreamingBubble(streamingMsg); },
      onToolEnd: () => { streamingMsg.activeTool = null; setExpression("neutral"); _updateStreamingBubble(streamingMsg); },
      onThinkingStart: () => { streamingMsg.thinkingText = ""; streamingMsg.thinking = true; _updateStreamingBubble(streamingMsg); },
      onThinkingDelta: (t) => { streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + t; _scheduleStreamingUpdate(streamingMsg); },
      onThinkingEnd: () => { streamingMsg.thinking = false; _updateStreamingBubble(streamingMsg); },
      onDone: ({ summary, emotion, images: di }) => {
        if (summary) streamingMsg.text = summary;
        streamingMsg.images = di || [];
        if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        streamingMsg.streaming = false; streamingMsg.activeTool = null;
        setExpression(emotion); setTimeout(() => setExpression("neutral"), 3000);
        const st = getState();
        const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
        const nbt = { ...st.chatMessagesByThread }; if (!nbt[animaName]) nbt[animaName] = {};
        nbt[animaName][threadId] = [...arr]; setState({ chatMessagesByThread: nbt });
        _renderConvMessages();
      },
      onError: ({ message: m }) => {
        streamingMsg.text += `\n[エラー: ${m}]`; streamingMsg.streaming = false;
        setExpression("troubled");
        const st = getState();
        const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
        const nbt = { ...st.chatMessagesByThread }; if (!nbt[animaName]) nbt[animaName] = {};
        nbt[animaName][threadId] = [...arr]; setState({ chatMessagesByThread: nbt });
        _renderConvMessages();
      },
    });

    setTalking(false);
    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
      const st = getState();
      const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
      const nbt = { ...st.chatMessagesByThread }; if (!nbt[animaName]) nbt[animaName] = {};
      nbt[animaName][threadId] = [...arr]; setState({ chatMessagesByThread: nbt });
      _renderConvMessages();
    }
  } catch (err) {
    if (err.name !== "AbortError") logger.error("Resume stream error", { anima: animaName, error: err.message });
  } finally {
    convStreamController = null;
    _wsUpdateSendButton(false);
    _dom.convInput?.focus();

    if (convPendingQueue.length > 0) {
      const next = convPendingQueue.shift();
      _wsShowPendingIndicator();
      if (convPendingQueue.length === 0) _wsHidePendingIndicator();
      setTimeout(() => _sendConversation(next.text, { images: next.images, displayImages: next.displayImages }), 150);
    }
  }
}

// ── Greeting ──────────────────────

async function _triggerGreeting(animaName) {
  if (_greetingInFlight) return;
  const lastTs = _lastGreetTime[animaName];
  if (lastTs && Date.now() - lastTs < _GREET_COOLDOWN_MS) return;

  _greetingInFlight = true;
  try {
    const data = await greetAnima(animaName);
    if (!data.response || data.cached) return;

    _lastGreetTime[animaName] = Date.now();
    const now = new Date().toISOString();

    const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
    const threadId = activeThreadId || "default";
    const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
    const newMessages = [
      ...current,
      { role: "system", text: "デスクを訪問しました", timestamp: now },
      { role: "assistant", text: data.response, timestamp: now },
    ];
    const nextByThread = { ...chatMessagesByThread };
    if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
    nextByThread[conversationAnima][threadId] = newMessages;
    setState({ chatMessagesByThread: nextByThread });
    _renderConvMessages();

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

// ── Queue / Send Button UI ──────────────────────

const _WS_SEND_BTN_ICONS = {
  send: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg>`,
  stop: `<svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg>`,
  interrupt: `<span class="chat-send-icon-group" aria-hidden="true"><svg class="chat-send-icon chat-send-icon-square" viewBox="0 0 24 24" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2.5" /></svg><svg class="chat-send-icon" viewBox="0 0 24 24" focusable="false"><path d="M12 19V5M5 12l7-7 7 7" /></svg></span>`,
};

function _wsUpdateSendButton(isStreaming) {
  const hasInput = (_dom.convInput?.value?.trim() || "").length > 0;
  if (_dom.convQueueBtn) _dom.convQueueBtn.disabled = !hasInput;
  if (!_dom.convSend) return;
  _dom.convSend.classList.remove("stop", "interrupt");
  if (!isStreaming) {
    _dom.convSend.innerHTML = _WS_SEND_BTN_ICONS.send;
    _dom.convSend.disabled = !hasInput && convPendingQueue.length === 0;
  } else if (hasInput) {
    _dom.convSend.innerHTML = _WS_SEND_BTN_ICONS.send;
    _dom.convSend.disabled = false;
  } else if (convPendingQueue.length > 0) {
    _dom.convSend.innerHTML = _WS_SEND_BTN_ICONS.interrupt;
    _dom.convSend.classList.add("interrupt");
    _dom.convSend.disabled = false;
  } else {
    _dom.convSend.innerHTML = _WS_SEND_BTN_ICONS.stop;
    _dom.convSend.classList.add("stop");
    _dom.convSend.disabled = false;
  }
}

function _wsShowPendingIndicator() {
  if (!_dom.convPending || !_dom.convPendingList) return;
  if (convPendingQueue.length === 0) { _dom.convPending.style.display = "none"; return; }
  if (_dom.convPendingLabel) _dom.convPendingLabel.textContent = `キュー (${convPendingQueue.length})`;
  _dom.convPendingList.innerHTML = convPendingQueue.map((p, i) => {
    const txt = escapeHtml(p.text.length > 50 ? p.text.slice(0, 50) + "…" : p.text);
    const img = p.images?.length ? ` <span style="opacity:0.6">(+${p.images.length}画像)</span>` : "";
    return `<div class="pending-queue-item" data-idx="${i}"><span class="pending-queue-item-num">${i + 1}.</span><span class="pending-queue-item-text">${txt || "(画像のみ)"}${img}</span><button class="pending-queue-item-del" data-idx="${i}" type="button">✕</button></div>`;
  }).join("");
  _dom.convPending.style.display = "";

  _dom.convPendingList.onclick = (e) => {
    const delBtn = e.target.closest(".pending-queue-item-del");
    if (delBtn) {
      e.stopPropagation();
      convPendingQueue.splice(parseInt(delBtn.dataset.idx, 10), 1);
      _wsShowPendingIndicator();
      _wsUpdateSendButton(!!convStreamController);
      return;
    }
    const item = e.target.closest(".pending-queue-item");
    if (item) {
      const removed = convPendingQueue.splice(parseInt(item.dataset.idx, 10), 1)[0];
      if (removed && _dom.convInput) {
        _dom.convInput.value = removed.text;
        _dom.convInput.style.height = "auto";
        const maxH = _isMobileView() ? 100 : 120;
        _dom.convInput.style.height = Math.min(_dom.convInput.scrollHeight, maxH) + "px";
        _dom.convInput.focus();
      }
      _wsShowPendingIndicator();
      _wsUpdateSendButton(!!convStreamController);
    }
  };
}

function _wsHidePendingIndicator() {
  if (_dom.convPending) _dom.convPending.style.display = "none";
}

function _wsStopStreaming() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  if (convStreamController) convStreamController.abort();
  fetch(`/api/animas/${encodeURIComponent(animaName)}/interrupt`, { method: "POST" }).catch(() => {});
}

// ── Voice Chat Callbacks ──────────────────────

function _buildVoiceChatCallbacks(animaName) {
  return {
    addUserBubble(text) {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const ts = new Date().toISOString();
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...current, { role: "user", text, timestamp: ts }];
      setState({ chatMessagesByThread: nextByThread });
      _renderConvMessages();
    },
    addStreamingBubble() {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const ts = new Date().toISOString();
      const msg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: ts, thinkingText: "", thinking: false };
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...current, msg];
      setState({ chatMessagesByThread: nextByThread });
      _renderConvMessages();
      return msg;
    },
    updateStreamingBubble(msg) { _updateStreamingBubble(msg); },
    finalizeStreamingBubble() {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const arr = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...arr];
      setState({ chatMessagesByThread: nextByThread });
      _renderConvMessages();
    },
    applyEmotion(emotion) {
      if (!emotion) return;
      setExpression(emotion);
      setTimeout(() => setExpression("neutral"), 3000);
    },
  };
}

// ── Mobile Helpers ──────────────────────

function _isMobileView() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function _openMobileSidebar() {
  _dom.convSidebar?.classList.add("mobile-open");
  _dom.sidebarBackdrop?.classList.add("visible");
}

function _closeMobileSidebar() {
  _dom.convSidebar?.classList.remove("mobile-open");
  _dom.sidebarBackdrop?.classList.remove("visible");
}

function _toggleMobileCharacter() {
  _dom.convCharacter?.classList.toggle("mobile-open");
}

function _closeMobileCharacter() {
  _dom.convCharacter?.classList.remove("mobile-open");
}

function _openMobileMemory() {
  _dom.memoryPanel?.classList.add("mobile-open");
}

function _closeMobileMemory() {
  _dom.memoryPanel?.classList.remove("mobile-open");
}

function _updateConvInputPlaceholder() {
  if (!_dom.convInput) return;
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  _dom.convInput.placeholder = _isMobileView()
    ? `${animaName} にメッセージ... (Enter)`
    : `メッセージを入力... (Ctrl+Enter)`;
}

function _cleanupMobileResources() {
  if (_swiperInstance) { _swiperInstance.destroy(); _swiperInstance = null; }
  if (_mobileMediaQuery) { _mobileMediaQuery.removeEventListener("change", _updateConvInputPlaceholder); _mobileMediaQuery = null; }
}

// ── Initialization ──────────────────────

export function initChatController(dom) {
  _dom = dom;

  dom.convBack?.addEventListener("click", closeConversation);
  dom.convOverlay?.addEventListener("click", e => { if (e.target === dom.convOverlay) closeConversation(); });
  dom.convSend?.addEventListener("click", () => submitConversation());
  dom.convQueueBtn?.addEventListener("click", () => addToQueue());
  dom.convInput?.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.altKey) { e.preventDefault(); addToQueue(); }
    else if (e.key === "Enter") {
      if (_isMobileView()) { if (!e.shiftKey) { e.preventDefault(); submitConversation(); } }
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
    _wsHidePendingIndicator();
    _wsUpdateSendButton(!!convStreamController);
  });

  dom.convInput?.addEventListener("input", () => {
    dom.convInput.style.height = "auto";
    const maxH = _isMobileView() ? 140 : 220;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
    _wsSaveDraft();
    _wsUpdateSendButton(!!convStreamController);
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

  // Mobile controls
  dom.mobileSidebarToggle?.addEventListener("click", () => {
    if (dom.convSidebar?.classList.contains("mobile-open")) _closeMobileSidebar();
    else { _closeMobileCharacter(); _openMobileSidebar(); }
  });
  dom.mobileCharacterToggle?.addEventListener("click", () => {
    if (dom.convCharacter?.classList.contains("mobile-open")) _closeMobileCharacter();
    else { _closeMobileSidebar(); _toggleMobileCharacter(); }
  });
  dom.sidebarBackdrop?.addEventListener("click", _closeMobileSidebar);
  dom.mobileMemoryClose?.addEventListener("click", _closeMobileMemory);

  _updateConvInputPlaceholder();
  _mobileMediaQuery = window.matchMedia("(max-width: 768px)");
  _mobileMediaQuery.addEventListener("change", _updateConvInputPlaceholder);

  // Touch gestures
  if ("ontouchstart" in window && dom.convOverlay) {
    _swiperInstance = new SwipeHandler(dom.convOverlay);
    _swiperInstance.onSwipeRight(info => { if (_isMobileView() && info.startX < 30) _openMobileSidebar(); });
    _swiperInstance.onSwipeLeft(() => { if (_isMobileView()) _closeMobileSidebar(); });
  }
}

export function isConvStreaming() {
  return !!convStreamController;
}
