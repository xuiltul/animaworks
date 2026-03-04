// ── Workspace Chat History ──────────────────────
// Conversation history rendering, infinite scroll, and load logic.
// Now delegates data management to ChatSessionManager.

import { getState, setState } from "./state.js";
import { escapeHtml, renderSimpleMarkdown, smartTimestamp } from "./utils.js";
import { renderChatImages } from "../../shared/image-input.js";
import {
  renderHistoryMessage, renderSessionDivider, bindToolCallHandlers,
  renderLiveBubble, renderStreamingBubbleInner,
} from "../../shared/chat/render-utils.js";
import { createScrollObserver } from "../../shared/chat/scroll-observer.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";
import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-chat-history");

// ── Module State ──────────────────────
let _getDom = () => ({});
let _scrollObserver = null;
export const HISTORY_PAGE_SIZE = 50;
const TOOL_RESULT_TRUNCATE = 500;

// ── Init ──────────────────────

export function initHistory({ getDom }) {
  _getDom = getDom;
}

// ── Render Options Builder ──────────────────────

export function renderOpts() {
  return {
    escapeHtml,
    renderMarkdown: renderSimpleMarkdown,
    smartTimestamp,
    renderChatImages,
    animaName: getState().conversationAnima,
    avatarMap: getState().chatAvatarMap || {},
    truncateLen: TOOL_RESULT_TRUNCATE,
    labels: {
      thinking: "考え中...",
      toolRunning: (tool) => `${escapeHtml(tool)} を実行中...`,
      heartbeatRelay: "ハートビート処理中...",
      heartbeatRelayDone: "応答を準備中...",
    },
  };
}

// ── Chat Rendering ──────────────────────

export function renderConvMessages() {
  const dom = _getDom();
  if (!dom.convMessages) return;

  const { activeThreadId, conversationAnima } = getState();
  const animaName = conversationAnima;
  const mgr = ChatSessionManager.getInstance();
  const threadMessages = animaName ? mgr.getMessages(animaName, activeThreadId || "default") : [];
  const hs = animaName ? mgr.getHistoryState(animaName, activeThreadId || "default") : null;

  if ((!hs || hs.sessions.length === 0) && threadMessages.length === 0) {
    if (hs && hs.loading) {
      dom.convMessages.innerHTML = '<div class="chat-empty"><span class="tool-spinner"></span> 読み込み中...</div>';
    } else {
      dom.convMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
    }
    return;
  }

  const opts = renderOpts();
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
    const hasStreaming = threadMessages.some(m => m.streaming);
    const lastSession = hs?.sessions?.[hs.sessions.length - 1];
    const lastSessionLastTs = lastSession?.messages?.slice(-1)[0]?.ts ?? "";
    const lastLiveTs = threadMessages[threadMessages.length - 1]?.timestamp ?? "";
    const liveIsNewer = hasStreaming || !lastSessionLastTs || lastLiveTs > lastSessionLastTs;
    if (liveIsNewer) {
      if (hs && hs.sessions.length > 0) {
        html += '<div class="session-divider"><span class="session-divider-label">現在のセッション</span></div>';
      }
      html += threadMessages.map(m => renderLiveBubble(m, opts)).join("");
    }
  }

  dom.convMessages.innerHTML = html;
  bindToolCallHandlers(dom.convMessages);
  refreshSentinel();
  dom.convMessages.scrollTop = dom.convMessages.scrollHeight;
}

export async function loadAndRenderConvMessages(animaName, { resumeStream }) {
  if (!animaName) return;

  const threadId = getState().activeThreadId || "default";
  const mgr = ChatSessionManager.getInstance();

  mgr.setMessages(animaName, threadId, []);
  renderConvMessages();

  await mgr.loadHistory(animaName, threadId, HISTORY_PAGE_SIZE);

  renderConvMessages();
  setupScrollObserver();
  resumeStream(animaName);
}

// ── Infinite Scroll ──────────────────────

export function setupScrollObserver() {
  const dom = _getDom();
  if (_scrollObserver) _scrollObserver.disconnect();
  if (!dom.convMessages) return;
  _scrollObserver = createScrollObserver({ container: dom.convMessages, onLoadMore: loadMoreHistory });
  _scrollObserver.observe();
}

export function refreshSentinel() {
  if (_scrollObserver) _scrollObserver.refresh();
}

export async function loadMoreHistory() {
  const dom = _getDom();
  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  const mgr = ChatSessionManager.getInstance();
  const hs = mgr.getHistoryState(animaName, threadId);
  if (!hs || !hs.hasMore || hs.loading) return;

  const existingIndicator = dom.convMessages.querySelector(".history-loading-more");
  if (!existingIndicator) {
    const indicator = document.createElement("div");
    indicator.className = "history-loading-more";
    indicator.innerHTML = '<span class="tool-spinner"></span> 過去の会話を読み込み中...';
    dom.convMessages.insertBefore(indicator, dom.convMessages.firstChild);
  }

  const prevScrollHeight = dom.convMessages.scrollHeight;
  await mgr.loadMoreHistory(animaName, threadId, HISTORY_PAGE_SIZE);
  renderConvMessages();
  const newScrollHeight = dom.convMessages.scrollHeight;
  dom.convMessages.scrollTop += (newScrollHeight - prevScrollHeight);
}

// ── History State Access ──────────────────────

export function getHistoryState() {
  return {};
}

export function disconnectScrollObserver() {
  if (_scrollObserver) { _scrollObserver.disconnect(); _scrollObserver = null; }
}
