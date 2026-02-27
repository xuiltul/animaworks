// ── Thread CRUD / Tab Controller ──────────────
import {
  $, isTabOpen, refreshAnimaUnread, clearUnreadForActiveThread,
  setThreadUnread, threadTimeValue, scheduleSaveChatUiState,
  CONSTANTS,
} from "./ctx.js";
import {
  renderThreadTabsHtml,
  createThread as sharedCreateThread,
  closeThread as sharedCloseThread,
} from "../../shared/chat/thread-logic.js";

export function createThreadController(ctx) {
  const { state, deps } = ctx;
  const { escapeHtml } = deps;

  function renderThreadTabs() {
    const container = $("chatThreadTabs");
    if (!container || !state.selectedAnima) return;

    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
    container.innerHTML = renderThreadTabsHtml(list, state.selectedThreadId, {
      escapeHtml,
      maxVisible: CONSTANTS.THREAD_VISIBLE_NON_DEFAULT,
    });

    container.querySelectorAll(".thread-tab").forEach(btn => {
      btn.addEventListener("click", e => {
        const tid = e.target.dataset.thread;
        if (tid) selectThread(tid);
      });
    });
    container.querySelectorAll(".thread-tab-close").forEach(btn => {
      btn.addEventListener("click", e => {
        e.stopPropagation();
        const tid = e.target.dataset.thread;
        if (tid) closeThread(tid);
      });
    });
    const newBtn = $("chatNewThreadBtn");
    if (newBtn) newBtn.addEventListener("click", () => createNewThread());

    const moreSelect = $("chatThreadMoreSelect");
    if (moreSelect) {
      moreSelect.addEventListener("change", e => {
        const tid = e.target.value;
        if (tid) selectThread(tid);
        e.target.value = "";
      });
    }
  }

  async function selectThread(threadId) {
    if (threadId === state.selectedThreadId) return;
    state.selectedThreadId = threadId;
    state.activeThreadByAnima[state.selectedAnima] = threadId;
    clearUnreadForActiveThread(ctx, state.selectedAnima, threadId);
    refreshAnimaUnread(ctx, state.selectedAnima);
    ctx.controllers.anima.renderAnimaTabs();
    renderThreadTabs();

    const name = state.selectedAnima;
    if (!name) return;

    const hs = state.historyState[name]?.[threadId];
    const needLoad = !hs || hs.sessions.length === 0;
    ctx.controllers.renderer.renderChat();

    if (needLoad) {
      try {
        const conv = await ctx.controllers.renderer.fetchConversationHistory(name, CONSTANTS.HISTORY_PAGE_SIZE, null, threadId);
        if (!state.historyState[name]) state.historyState[name] = {};
        if (conv && conv.sessions && conv.sessions.length > 0) {
          state.historyState[name][threadId] = {
            sessions: conv.sessions, hasMore: conv.has_more || false,
            nextBefore: conv.next_before || null, loading: false,
          };
        } else {
          state.historyState[name][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
        }
      } catch {
        if (!state.historyState[name]) state.historyState[name] = {};
        state.historyState[name][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
      }
    }
    ctx.controllers.renderer.renderChat();
    scheduleSaveChatUiState(ctx);
  }

  function createNewThread() {
    if (!state.selectedAnima) return;
    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
    const { updatedList, newThreadId } = sharedCreateThread(list, state.selectedAnima);
    state.threads[state.selectedAnima] = updatedList;

    if (!state.chatHistories[state.selectedAnima]) state.chatHistories[state.selectedAnima] = {};
    state.chatHistories[state.selectedAnima][newThreadId] = [];

    if (!state.historyState[state.selectedAnima]) state.historyState[state.selectedAnima] = {};
    state.historyState[state.selectedAnima][newThreadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };

    renderThreadTabs();
    selectThread(newThreadId);
    scheduleSaveChatUiState(ctx);
  }

  function closeThread(threadId) {
    if (threadId === "default" || !state.selectedAnima) return;
    const list = state.threads[state.selectedAnima];
    if (!list) return;
    if (!list.some(th => th.id === threadId)) return;

    state.threads[state.selectedAnima] = sharedCloseThread(list, threadId);
    delete state.chatHistories[state.selectedAnima]?.[threadId];
    delete state.historyState[state.selectedAnima]?.[threadId];

    if (state.selectedThreadId === threadId) {
      state.selectedThreadId = "default";
      state.activeThreadByAnima[state.selectedAnima] = "default";
    }
    refreshAnimaUnread(ctx, state.selectedAnima);
    ctx.controllers.anima.renderAnimaTabs();
    renderThreadTabs();
    ctx.controllers.renderer.renderChat();
    scheduleSaveChatUiState(ctx);
  }

  return { renderThreadTabs, selectThread, createNewThread, closeThread };
}
