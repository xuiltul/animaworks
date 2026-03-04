// ── Chat Context Factory ──────────────────────
import { t } from "/shared/i18n.js";
import { api } from "../../modules/api.js";
import { escapeHtml, renderMarkdown, renderSafeMarkdown, timeStr, smartTimestamp } from "../../modules/state.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../../shared/chat-stream.js";
import { createLogger } from "../../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../../shared/image-input.js";
import { initVoiceUI, updateVoiceUIAnima } from "../../modules/voice-ui.js";
import { getIcon, getDisplaySummary } from "../../shared/activity-types.js";
import {
  getDraftKey as _getDraftKey, saveDraft as _saveDraft, loadDraft as _loadDraft, clearDraft as _clearDraft,
} from "../../shared/chat/draft.js";
import {
  threadTimeValue, defaultThreadLabel as _defaultThreadLabel,
  mergeThreadsFromSessions as _mergeThreads,
} from "../../shared/chat/thread-logic.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";

const logger = createLogger("chat-page");

export const CONSTANTS = Object.freeze({
  HISTORY_PAGE_SIZE: 50,
  TOOL_RESULT_TRUNCATE: 500,
  THREAD_VISIBLE_NON_DEFAULT: 5,
  CHAT_POLL_INTERVAL_MS: 5000,
});

export function createChatContext() {
  const mgr = ChatSessionManager.getInstance();
  mgr.configure({
    streamChat, fetchActiveStream, fetchStreamProgress,
    getUser: () => localStorage.getItem("animaworks_user") || "human",
    fetchHistory: async (animaName, limit, before, threadId) => {
      let url = `/api/animas/${encodeURIComponent(animaName)}/conversation/history?limit=${limit}`;
      if (before) url += `&before=${encodeURIComponent(before)}`;
      url += `&thread_id=${encodeURIComponent(threadId)}`;
      if (threadId !== "default") url += `&strict_thread=1`;
      return await api(url);
    },
  });

  const state = {
    container: null,
    animas: [],
    selectedAnima: null,
    animaDetail: null,
    animaTabs: [],
    activeRightTab: "state",
    activeMemoryTab: "episodes",
    intervals: [],
    boundListeners: [],
    chatObserver: null,
    selectedThreadId: "default",
    threads: {},
    activeThreadByAnima: {},
    animaLastAccess: {},
    imageInputManager: null,
    bustupUrl: null,
    chatUiStateSaveTimer: null,
    animaTabAvatarUrls: {},
    animaTabAvatarLoading: {},
    chatPollingInFlight: false,
    rightPaneVisible: true,
    paneIdx: 0,
    /** @type {ChatSessionManager} */
    manager: mgr,
  };

  const deps = {
    t, api, escapeHtml, renderMarkdown, renderSafeMarkdown, timeStr, smartTimestamp,
    streamChat, fetchActiveStream, fetchStreamProgress,
    logger,
    createImageInput, initLightbox, renderChatImages,
    initVoiceUI, updateVoiceUIAnima,
    getIcon, getDisplaySummary,
  };

  function $(id) {
    return state.container?.querySelector(`[data-chat-id="${id}"]`) ?? null;
  }

  function $root(id) {
    return state.rootContainer?.querySelector(`[data-chat-id="${id}"]`) ?? null;
  }

  return { state, deps, controllers: {}, $, $root };
}

export function chatInputMaxHeight() {
  return window.matchMedia("(max-width: 768px)").matches ? 140 : 260;
}

// ── Draft Persistence (delegates to shared/chat/draft.js) ──
export function getDraftKey(animaName, threadId) {
  const user = localStorage.getItem("animaworks_user") || "guest";
  return _getDraftKey("dashboard-chat", user, animaName, threadId);
}

export function saveDraft(animaName, text, threadId) {
  if (!animaName) return;
  _saveDraft(getDraftKey(animaName, threadId), text || "");
}

export function loadDraft(animaName, threadId) {
  if (!animaName) return "";
  return _loadDraft(getDraftKey(animaName, threadId));
}

export function clearDraft(animaName, threadId) {
  if (!animaName) return;
  _clearDraft(getDraftKey(animaName, threadId));
}

// ── Tab / Thread Helpers ──
export function getTabEntry(ctx, animaName) {
  return ctx.state.animaTabs.find(tab => tab.name === animaName) || null;
}

export function isTabOpen(ctx, animaName) {
  return Boolean(getTabEntry(ctx, animaName));
}

export function setThreadUnread(ctx, animaName, threadId, unread) {
  const list = ctx.state.threads[animaName];
  if (!Array.isArray(list)) return;
  const item = list.find(th => th.id === threadId);
  if (item) item.unread = Boolean(unread);
}

export { threadTimeValue };

export function defaultThreadLabel(threadId, lastTs = "") {
  return _defaultThreadLabel(threadId, lastTs, timeStr);
}

export function refreshAnimaUnread(ctx, animaName) {
  const tab = getTabEntry(ctx, animaName);
  if (!tab) return;
  const list = ctx.state.threads[animaName] || [];
  tab.unreadStar = list.some(th => th.unread);
}

export function clearUnreadForActiveThread(ctx, animaName, threadId) {
  setThreadUnread(ctx, animaName, threadId, false);
  refreshAnimaUnread(ctx, animaName);
}

export function isBusinessTheme() {
  return document.body.classList.contains("mode-realistic");
}

export function mergeThreadsFromSessions(ctx, animaName, sessionsData) {
  if (!animaName || !sessionsData) return;
  const existing = ctx.state.threads[animaName] || [{ id: "default", label: "メイン", unread: false }];
  ctx.state.threads[animaName] = _mergeThreads(existing, sessionsData, { timeStr });
}

// ── Chat UI State Persistence ──
export function serializeChatUiState(ctx) {
  const { animaTabs, threads, activeThreadByAnima, selectedAnima, animaLastAccess } = ctx.state;
  const threadState = {};
  for (const tab of animaTabs) {
    const name = tab.name;
    const list = threads[name] || [{ id: "default", label: "メイン", unread: false }];
    threadState[name] = {
      active_thread_id: activeThreadByAnima[name] || "default",
      threads: list.map(th => {
        const o = { id: th.id, label: th.label, unread: Boolean(th.unread) };
        if (th.archived) o.archived = true;
        return o;
      }),
    };
  }
  return {
    version: 1,
    active_anima: selectedAnima,
    anima_tabs: animaTabs.map(tab => ({ name: tab.name, unread_star: Boolean(tab.unreadStar) })),
    anima_last_access: { ...animaLastAccess },
    thread_state: threadState,
  };
}

export async function saveChatUiStateNow(ctx) {
  try {
    await ctx.deps.api("/api/chat/ui-state", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: serializeChatUiState(ctx) }),
    });
  } catch (err) {
    ctx.deps.logger.debug("Failed to persist chat ui state", err);
  }
}

export function scheduleSaveChatUiState(ctx) {
  if (ctx.state.chatUiStateSaveTimer) clearTimeout(ctx.state.chatUiStateSaveTimer);
  ctx.state.chatUiStateSaveTimer = setTimeout(() => {
    ctx.state.chatUiStateSaveTimer = null;
    saveChatUiStateNow(ctx);
  }, 300);
}

export async function fetchChatUiState(ctx) {
  try {
    const data = await ctx.deps.api("/api/chat/ui-state");
    return data?.state || null;
  } catch {
    return null;
  }
}
