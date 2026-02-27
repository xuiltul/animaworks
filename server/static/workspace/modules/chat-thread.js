// ── Workspace Chat Threads ──────────────────────
// Thread tab rendering, selection, creation, and closing.

import { getState, setState } from "./state.js";
import { fetchConversationHistory } from "./api.js";
import { escapeHtml } from "./utils.js";
import {
  renderThreadTabsHtml, createThread as sharedCreateThread,
  closeThread as sharedCloseThread,
} from "../../shared/chat/thread-logic.js";
import { createHistoryState, applyHistoryData } from "../../shared/chat/history-loader.js";
import { HISTORY_PAGE_SIZE } from "./chat-history.js";

// ── Module State ──────────────────────
let _getDom = () => ({});
let _getHistoryState = () => ({});
let _renderConvMessages = () => {};
let _refreshSentinel = () => {};

// ── Init ──────────────────────

export function initThreads({ getDom, getHistoryState, renderConvMessages, refreshSentinel }) {
  _getDom = getDom;
  _getHistoryState = getHistoryState;
  _renderConvMessages = renderConvMessages;
  _refreshSentinel = refreshSentinel;
}

// ── Thread Tabs ──────────────────────

export function renderWsThreadTabs() {
  const dom = _getDom();
  const container = dom.threadTabs;
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
      if (tid) selectWsThread(tid);
    });
  });
  container.querySelectorAll(".thread-tab-close").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const tid = e.target.dataset.thread;
      if (tid) closeWsThread(tid);
    });
  });
  const newBtn = document.getElementById("wsNewThreadBtn");
  if (newBtn) newBtn.addEventListener("click", () => createWsNewThread());
}

export async function selectWsThread(threadId) {
  const current = getState().activeThreadId;
  if (threadId === current) return;

  setState({ activeThreadId: threadId });
  renderWsThreadTabs();

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const historyState = _getHistoryState();
  const hs = historyState[animaName]?.[threadId];
  const needLoad = !hs || hs.sessions.length === 0;

  if (needLoad) {
    if (!historyState[animaName]) historyState[animaName] = {};
    historyState[animaName][threadId] = { ...createHistoryState(), loading: true };
    _renderConvMessages();

    try {
      const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, null, threadId);
      historyState[animaName][threadId] = createHistoryState();
      applyHistoryData(historyState[animaName][threadId], data);
    } catch {
      historyState[animaName][threadId] = createHistoryState();
    }
  }

  _renderConvMessages();
  _refreshSentinel();
}

export function createWsNewThread() {
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

  const historyState = _getHistoryState();
  if (!historyState[animaName]) historyState[animaName] = {};
  historyState[animaName][newThreadId] = createHistoryState();

  renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}

export function closeWsThread(threadId) {
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

  const historyState = _getHistoryState();
  delete historyState[animaName]?.[threadId];

  const switchToDefault = activeThreadId === threadId;
  setState({
    threads: nextThreads,
    chatMessagesByThread: nextByThread,
    ...(switchToDefault ? { activeThreadId: "default" } : {}),
  });

  renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}
