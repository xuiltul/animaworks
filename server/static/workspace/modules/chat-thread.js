// ── Workspace Chat Threads ──────────────────────
// Thread tab rendering, selection, creation, and closing.
// Now delegates history state to ChatSessionManager.

import { getState, setState } from "./state.js";
import { escapeHtml } from "./utils.js";
import { t } from "../../shared/i18n.js";
import {
  renderThreadTabsHtml, createThread as sharedCreateThread,
  closeThread as sharedCloseThread,
} from "../../shared/chat/thread-logic.js";
import { ChatSessionManager } from "../../shared/chat/session-manager.js";
import { HISTORY_PAGE_SIZE } from "./chat-history.js";
import { wsSaveDraft, wsLoadDraft, isMobileView } from "./chat-mobile.js";

// ── Module State ──────────────────────
let _getDom = () => ({});
let _renderConvMessages = () => {};
let _refreshSentinel = () => {};

// ── Init ──────────────────────

export function initThreads({ getDom, renderConvMessages, refreshSentinel }) {
  _getDom = getDom;
  _renderConvMessages = renderConvMessages;
  _refreshSentinel = refreshSentinel;
}

// ── Thread Tabs ──────────────────────

export function renderWsThreadTabs() {
  const dom = _getDom();
  const container = dom.threadTabs;
  const animaName = getState().conversationAnima;
  if (!container || !animaName) return;

  const list = getState().threads[animaName] || [{ id: "default", label: t("thread.default_label"), unread: false }];
  const activeThreadId = getState().activeThreadId || "default";
  const mgr = ChatSessionManager.getInstance();
  const streamCtx = mgr.getStreamingContext(animaName);

  container.innerHTML = renderThreadTabsHtml(list, activeThreadId, {
    escapeHtml,
    newBtnId: "wsNewThreadBtn",
    moreSelectId: "wsThreadMoreSelect",
    streamingThreadId: streamCtx?.thread || null,
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

  wsSaveDraft();

  setState({ activeThreadId: threadId });
  renderWsThreadTabs();

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const dom = _getDom();
  if (dom.convInput) {
    dom.convInput.value = wsLoadDraft(animaName, threadId);
    dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 100 : 120;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
  }

  const mgr = ChatSessionManager.getInstance();
  const hs = mgr.getHistoryState(animaName, threadId);
  const needLoad = !hs || hs.sessions.length === 0;

  if (needLoad) {
    await mgr.loadHistory(animaName, threadId, HISTORY_PAGE_SIZE);
  }

  _renderConvMessages();
  _refreshSentinel();
}

export function createWsNewThread() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const { threads } = getState();
  const list = threads[animaName] || [{ id: "default", label: t("thread.default_label"), unread: false }];
  const { updatedList, newThreadId } = sharedCreateThread(list, animaName);

  const nextThreads = { ...threads, [animaName]: updatedList };
  setState({ threads: nextThreads, activeThreadId: newThreadId });

  const mgr = ChatSessionManager.getInstance();
  mgr.setMessages(animaName, newThreadId, []);

  renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}

export function closeWsThread(threadId) {
  if (threadId === "default") return;
  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const { threads, activeThreadId } = getState();
  const list = threads[animaName];
  if (!list || !list.some(t => t.id === threadId)) return;

  const nextList = sharedCloseThread(list, threadId);
  const nextThreads = { ...threads, [animaName]: nextList };

  ChatSessionManager.getInstance().destroySession(animaName, threadId);

  const switchToDefault = activeThreadId === threadId;
  setState({
    threads: nextThreads,
    ...(switchToDefault ? { activeThreadId: "default" } : {}),
  });

  renderWsThreadTabs();
  _renderConvMessages();
  _refreshSentinel();
}
