// ── Chat Rendering / Infinite Scroll / Polling ──
import {
  setThreadUnread, refreshAnimaUnread, threadTimeValue,
  mergeThreadsFromSessions, scheduleSaveChatUiState, CONSTANTS,
} from "./ctx.js";
import {
  renderHistoryMessage as _sharedRenderHistoryMessage,
  renderSessionDivider as _sharedRenderSessionDivider,
  bindToolCallHandlers as _sharedBindToolCallHandlers,
  renderLiveBubble,
  renderStreamingBubbleInner,
  updateStreamingZone,
} from "../../shared/chat/render-utils.js";
import { createScrollObserver } from "../../shared/chat/scroll-observer.js";
import { mergePolledHistory } from "../../shared/chat/history-loader.js";
import { initTextArtifactHandlers } from "../../shared/text-artifact.js";

export function createChatRenderer(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { api, t, escapeHtml, renderMarkdown, smartTimestamp, timeStr, renderChatImages } = deps;

  // ── Smart Scroll (sticky-bottom with floating button) ──
  const NEAR_BOTTOM_PX = 80;
  let _userDetached = false;

  function isNearBottom(el) {
    if (!el) return true;
    return (el.scrollHeight - (el.scrollTop + el.clientHeight)) <= NEAR_BOTTOM_PX;
  }

  function scrollToBottom(el) {
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    _userDetached = false;
    _updateScrollBtn();
  }

  function _updateScrollBtn() {
    const btn = $("chatScrollToBottom");
    if (!btn) return;
    btn.classList.toggle("visible", _userDetached);
  }

  function initScrollTracking() {
    const messagesEl = $("chatPageMessages");
    if (!messagesEl) return;
    messagesEl.addEventListener("scroll", () => {
      _userDetached = !isNearBottom(messagesEl);
      _updateScrollBtn();
    }, { passive: true });
    const btn = $("chatScrollToBottom");
    if (btn) {
      btn.addEventListener("click", () => scrollToBottom(messagesEl));
    }
  }

  const _renderOpts = () => ({
    escapeHtml, renderMarkdown, smartTimestamp, renderChatImages,
    animaName: state.selectedAnima,
    avatarMap: state.animaTabAvatarUrls || {},
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
    const mgr = state.manager;
    const history = name ? mgr.getMessages(name, tid) : [];
    const hs = name ? mgr.getHistoryState(name, tid) : { sessions: [], hasMore: false, nextBefore: null, loading: false };

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
      const hasStreaming = history.some(m => m.streaming);
      const lastSession = hs.sessions[hs.sessions.length - 1];
      const lastSessionLastTs = lastSession?.messages?.slice(-1)[0]?.ts ?? "";
      const lastLiveTs = history[history.length - 1]?.timestamp ?? "";
      const liveIsNewer = hasStreaming || !lastSessionLastTs || lastLiveTs > lastSessionLastTs;
      if (liveIsNewer) {
        if (hs.sessions.length > 0) {
          liveHtml += `<div class="session-divider"><span class="session-divider-label">${t("chat.current_session")}</span></div>`;
        }
        const opts = _renderOpts();
        liveHtml += history.map(m => renderLiveBubble(m, opts)).join("");
      }
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

  function renderStreamingBubble(msg, zone = "all") {
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

    updateStreamingZone(bubble, msg, _renderOpts(), zone);
    if (!_userDetached) messagesEl.scrollTop = messagesEl.scrollHeight;
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
    const mgr = state.manager;
    const hs = mgr.getHistoryState(name, tid);
    if (!hs || !hs.hasMore || hs.loading) return;
    if (mgr.isStreamingFor(name, tid)) return;

    renderChat(false);

    await mgr.loadMoreHistory(name, tid, CONSTANTS.HISTORY_PAGE_SIZE);
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
    const mgr = state.manager;
    if (mgr.isStreamingFor(name, tid)) return;

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

      const { changed } = mgr.mergePolledHistory(name, tid, conv);
      if (!changed) return;

      // Do NOT call keepOnlyStreaming here. It would clear session.messages (user + completed
      // assistant) before the API has caught up (activity_log write can be delayed). That causes
      // the last exchange to disappear from the UI. Live messages are deduplicated in renderChat
      // when they're already in the last session.

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
    bindToolCallHandlers, initScrollTracking, scrollToBottom,
    isUserDetached: () => _userDetached,
    reattach: () => { _userDetached = false; _updateScrollBtn(); },
  };
}
