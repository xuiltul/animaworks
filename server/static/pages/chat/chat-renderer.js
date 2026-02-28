// ── Chat Rendering / Infinite Scroll / Polling ──
import {
  $, setThreadUnread, refreshAnimaUnread, threadTimeValue,
  mergeThreadsFromSessions, scheduleSaveChatUiState, CONSTANTS,
} from "./ctx.js";
import {
  renderHistoryMessage as _sharedRenderHistoryMessage,
  renderSessionDivider as _sharedRenderSessionDivider,
  bindToolCallHandlers as _sharedBindToolCallHandlers,
  renderLiveBubble,
  renderStreamingBubbleInner,
} from "../../shared/chat/render-utils.js";
import { createScrollObserver } from "../../shared/chat/scroll-observer.js";
import { mergePolledHistory } from "../../shared/chat/history-loader.js";
import { initTextArtifactHandlers } from "../../shared/text-artifact.js";

export function createChatRenderer(ctx) {
  const { state, deps } = ctx;
  const { api, t, escapeHtml, renderMarkdown, smartTimestamp, timeStr, renderChatImages } = deps;

  const _renderOpts = () => ({
    escapeHtml, renderMarkdown, smartTimestamp, renderChatImages,
    animaName: state.selectedAnima,
    truncateLen: CONSTANTS.TOOL_RESULT_TRUNCATE,
    labels: {
      thinking: t("chat.thinking"),
      toolRunning: (tool) => t("chat.tool_running", { tool }),
      currentSession: t("chat.current_session"),
      heartbeatRelay: t("chat.heartbeat_relay"),
      heartbeatRelayDone: t("chat.heartbeat_relay_done"),
    },
  });

  function renderHistoryMessage(msg) {
    return _sharedRenderHistoryMessage(msg, _renderOpts());
  }

  function renderSessionDivider(session, isFirst) {
    return _sharedRenderSessionDivider(session, isFirst, _renderOpts());
  }

  function bindToolCallHandlers(container) {
    _sharedBindToolCallHandlers(container);
  }

  // ── Main Chat Rendering ──

  function renderChat(scrollToBottom = true) {
    const messagesEl = $("chatPageMessages");
    if (!messagesEl) return;

    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    const history = state.chatHistories[name]?.[tid] || [];
    const hs = state.historyState[name]?.[tid] || { sessions: [], hasMore: false, nextBefore: null, loading: false };

    if (hs.sessions.length === 0 && history.length === 0) {
      messagesEl.innerHTML = hs.loading
        ? `<div class="chat-empty"><span class="tool-spinner"></span> ${t("common.loading")}</div>`
        : `<div class="chat-empty">${t("chat.messages_empty")}</div>`;
      return;
    }

    let topHtml = "";
    if (hs.hasMore) {
      if (hs.loading) topHtml += `<div class="history-loading-more"><span class="tool-spinner"></span> ${t("chat.past_loading")}</div>`;
      topHtml += '<div class="chat-load-sentinel"></div>';
    }

    const prevScrollHeight = messagesEl.scrollHeight;

    let sessionsHtml = "";
    for (let si = 0; si < hs.sessions.length; si++) {
      const session = hs.sessions[si];
      sessionsHtml += renderSessionDivider(session, si === 0);
      if (session.messages) {
        for (const msg of session.messages) sessionsHtml += renderHistoryMessage(msg);
      }
    }

    let liveHtml = "";
    if (history.length > 0) {
      if (hs.sessions.length > 0) {
        liveHtml += `<div class="session-divider"><span class="session-divider-label">${t("chat.current_session")}</span></div>`;
      }
      const opts = _renderOpts();
      liveHtml += history.map(m => renderLiveBubble(m, opts)).join("");
    }

    messagesEl.innerHTML = topHtml + sessionsHtml + liveHtml;
    bindToolCallHandlers(messagesEl);
    initTextArtifactHandlers();

    if (scrollToBottom) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else {
      messagesEl.scrollTop += (messagesEl.scrollHeight - prevScrollHeight);
    }
    observeChatSentinel();
  }

  function renderStreamingBubble(msg) {
    const messagesEl = $("chatPageMessages");
    if (!messagesEl) return;
    let bubble = null;
    if (msg.streamId) {
      bubble = messagesEl.querySelector(`.chat-bubble.assistant.streaming[data-stream-id="${CSS.escape(String(msg.streamId))}"]`);
    }
    if (!bubble) {
      const bubbles = messagesEl.querySelectorAll(".chat-bubble.assistant.streaming");
      bubble = bubbles[bubbles.length - 1];
    }
    if (!bubble) return;

    bubble.innerHTML = renderStreamingBubbleInner(msg, _renderOpts());
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function markResponseComplete(animaName, threadId) {
    if (!animaName || !threadId) return;
    const isActive = state.selectedAnima === animaName && state.selectedThreadId === threadId;
    setThreadUnread(ctx, animaName, threadId, !isActive);
    refreshAnimaUnread(ctx, animaName);
    if (animaName === state.selectedAnima) ctx.controllers.thread.renderThreadTabs();
    ctx.controllers.anima.renderAnimaTabs();
    scheduleSaveChatUiState(ctx);
  }

  // ── Infinite Scroll (shared scroll-observer) ──

  let _scrollObs = null;

  function setupChatObserver() {
    if (_scrollObs) _scrollObs.disconnect();
    if (state.chatObserver) { state.chatObserver.disconnect(); state.chatObserver = null; }
    const messagesEl = $("chatPageMessages");
    if (!messagesEl) return;
    _scrollObs = createScrollObserver({ container: messagesEl, onLoadMore: loadOlderMessages });
    _scrollObs.observe();
  }

  function observeChatSentinel() {
    if (_scrollObs) _scrollObs.refresh();
  }

  async function loadOlderMessages() {
    const name = state.selectedAnima;
    const tid = state.selectedThreadId;
    if (!name) return;
    const hs = state.historyState[name]?.[tid];
    if (!hs || !hs.hasMore || hs.loading) return;
    const activeS = state.activeStreams[name];
    if (activeS && activeS.thread === tid) return;

    hs.loading = true;
    renderChat(false);

    try {
      const data = await fetchConversationHistory(name, CONSTANTS.HISTORY_PAGE_SIZE, hs.nextBefore, tid);
      if (data && data.sessions && data.sessions.length > 0) {
        hs.sessions = [...data.sessions, ...hs.sessions];
        hs.hasMore = data.has_more || false;
        hs.nextBefore = data.next_before || null;
      } else {
        hs.hasMore = false;
      }
    } catch (err) {
      deps.logger.error("Failed to load older messages", { error: err.message });
    }
    hs.loading = false;
    renderChat(false);
  }

  // ── Conversation History API ──

  async function fetchConversationHistory(animaName, limit = CONSTANTS.HISTORY_PAGE_SIZE, before = null, threadId = "default") {
    let url = `/api/animas/${encodeURIComponent(animaName)}/conversation/history?limit=${limit}`;
    if (before) url += `&before=${encodeURIComponent(before)}`;
    url += `&thread_id=${encodeURIComponent(threadId)}`;
    if (threadId !== "default") url += `&strict_thread=1`;
    return await api(url);
  }

  // ── Polling ──

  async function pollSelectedChat() {
    const name = state.selectedAnima;
    const tid = state.selectedThreadId || "default";
    if (!name || state.chatPollingInFlight) return;
    const activeS = state.activeStreams[name];
    if (activeS && activeS.thread === tid) return;
    if (Object.keys(state.activeStreams).length > 0 && state.activeStreams[name]) return;

    state.chatPollingInFlight = true;
    try {
      const [conv, sessionsData] = await Promise.all([
        fetchConversationHistory(name, CONSTANTS.HISTORY_PAGE_SIZE, null, tid).catch(() => null),
        api(`/api/animas/${encodeURIComponent(name)}/sessions`).catch(() => null),
      ]);

      if (sessionsData) {
        const prevThreadLastTs = new Map(
          (state.threads[name] || []).map(th => [th.id, threadTimeValue(th.lastTs || "")]),
        );
        mergeThreadsFromSessions(ctx, name, sessionsData);
        for (const th of state.threads[name] || []) {
          if (!th?.id || th.id === tid) continue;
          const prev = prevThreadLastTs.get(th.id) || 0;
          const curr = threadTimeValue(th.lastTs || "");
          if (curr > prev) setThreadUnread(ctx, name, th.id, true);
        }
        refreshAnimaUnread(ctx, name);
        ctx.controllers.anima.renderAnimaTabs();
        ctx.controllers.thread.renderThreadTabs();
      }

      if (!conv || !Array.isArray(conv.sessions)) return;

      if (!state.historyState[name]) state.historyState[name] = {};
      const prev = state.historyState[name][tid];

      const { changed, merged: mergedHs } = mergePolledHistory(prev, conv);
      if (!changed && prev) return;

      state.historyState[name][tid] = mergedHs;

      if (state.chatHistories[name]?.[tid]) {
        state.chatHistories[name][tid] = state.chatHistories[name][tid].filter(m => m.streaming);
      }

      const messagesEl = $("chatPageMessages");
      const shouldStick = messagesEl
        ? (messagesEl.scrollHeight - (messagesEl.scrollTop + messagesEl.clientHeight)) <= 80
        : true;
      renderChat(shouldStick);
    } finally {
      state.chatPollingInFlight = false;
    }
  }

  return {
    renderChat, renderStreamingBubble, markResponseComplete,
    setupChatObserver, observeChatSentinel, fetchConversationHistory,
    pollSelectedChat, renderHistoryMessage, renderSessionDivider,
    bindToolCallHandlers,
  };
}
