// ── Thread CRUD / Tab Controller ──────────────
import {
  isTabOpen, refreshAnimaUnread, clearUnreadForActiveThread,
  setThreadUnread, threadTimeValue, scheduleSaveChatUiState,
  saveDraft, loadDraft, chatInputMaxHeight,
  CONSTANTS,
} from "./ctx.js";
import {
  renderThreadTabsHtml,
  createThread as sharedCreateThread,
  closeThread as sharedCloseThread,
  restoreThread as sharedRestoreThread,
} from "../../shared/chat/thread-logic.js";

export function createThreadController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { escapeHtml } = deps;

  function renderThreadTabs() {
    const container = $("chatThreadTabs");
    if (!container || !state.selectedAnima) return;

    _updateThreadDropdownLabel();

    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
    const streamCtx = state.manager.getStreamingContext(state.selectedAnima);
    container.innerHTML = renderThreadTabsHtml(list, state.selectedThreadId, {
      escapeHtml,
      maxVisible: CONSTANTS.THREAD_VISIBLE_NON_DEFAULT,
      streamingThreadId: streamCtx?.thread || null,
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

    const archiveBtn = $("chatArchiveBtn");
    if (archiveBtn) {
      archiveBtn.addEventListener("click", e => {
        e.stopPropagation();
        const wrap = archiveBtn.closest(".thread-archive-wrap");
        if (wrap) wrap.classList.toggle("open");
      });
    }

    const archiveMenu = $("chatArchiveMenu");
    if (archiveMenu) {
      archiveMenu.querySelectorAll(".thread-archive-item").forEach(btn => {
        btn.addEventListener("click", e => {
          e.stopPropagation();
          const tid = e.currentTarget.dataset.thread;
          if (tid) restoreThread(tid);
          const wrap = archiveMenu.closest(".thread-archive-wrap");
          if (wrap) wrap.classList.remove("open");
        });
      });
    }

    // Close archive dropdown on outside click
    function closeArchiveDropdown(e) {
      const wrap = container.querySelector(".thread-archive-wrap.open");
      if (wrap && !wrap.contains(e.target)) wrap.classList.remove("open");
    }
    document.removeEventListener("click", closeArchiveDropdown);
    document.addEventListener("click", closeArchiveDropdown);
  }

  async function selectThread(threadId) {
    if (threadId === state.selectedThreadId) return;
    const prevThread = state.selectedThreadId;
    const name = state.selectedAnima;
    const input = $("chatPageInput");
    if (name && input) {
      saveDraft(name, input.value || "", prevThread);
    }

    state.selectedThreadId = threadId;
    state.activeThreadByAnima[state.selectedAnima] = threadId;
    clearUnreadForActiveThread(ctx, state.selectedAnima, threadId);
    refreshAnimaUnread(ctx, state.selectedAnima);
    ctx.controllers.anima.renderAnimaTabs();
    renderThreadTabs();

    if (!name) return;

    if (input) {
      input.value = loadDraft(name, threadId);
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, chatInputMaxHeight()) + "px";
    }
    ctx.controllers.streaming.showPendingIndicator();
    ctx.controllers.streaming.updateSendButton();

    const mgr = state.manager;
    const hs = mgr.getHistoryState(name, threadId);
    const needLoad = !hs || hs.sessions.length === 0;
    ctx.controllers.renderer.renderChat();

    if (needLoad) {
      await mgr.loadHistory(name, threadId, CONSTANTS.HISTORY_PAGE_SIZE);
    }
    ctx.controllers.renderer.renderChat();
    scheduleSaveChatUiState(ctx);
    state.paneHost?.savePaneStates?.();
  }

  function createNewThread() {
    if (!state.selectedAnima) return;
    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
    const { updatedList, newThreadId } = sharedCreateThread(list, state.selectedAnima);
    state.threads[state.selectedAnima] = updatedList;

    state.manager.setMessages(state.selectedAnima, newThreadId, []);

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
    // Don't destroy session — archived threads can be restored

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

  function restoreThread(threadId) {
    if (!state.selectedAnima) return;
    const list = state.threads[state.selectedAnima];
    if (!list) return;

    state.threads[state.selectedAnima] = sharedRestoreThread(list, threadId);
    renderThreadTabs();
    selectThread(threadId);
    scheduleSaveChatUiState(ctx);
  }

  function renderThreadDropdownMenu() {
    const menu = $("chatThreadDropdownMenu");
    const label = $("chatThreadDropdownLabel");
    if (!menu || !state.selectedAnima) return;

    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
    const active = list.find(th => th.id === state.selectedThreadId);
    if (label) label.textContent = active?.label || "メイン";

    const visible = list.filter(th => !th.archived);
    const archived = list.filter(th => th.archived);
    const streamCtx = state.manager.getStreamingContext(state.selectedAnima);

    let html = visible.map(th => {
      const isCurrent = th.id === state.selectedThreadId;
      const streaming = streamCtx?.thread === th.id;
      let cls = "chat-thread-dd-item";
      if (isCurrent) cls += " active";
      if (streaming) cls += " is-streaming";
      if (th.unread) cls += " has-unread";
      const closeBtn = th.id !== "default"
        ? ` <button class="chat-thread-dd-close" data-thread="${escapeHtml(th.id)}" aria-label="閉じる">&times;</button>`
        : "";
      return `<div class="${cls}" data-thread="${escapeHtml(th.id)}">`
        + `<span class="chat-thread-dd-label">${escapeHtml(th.label || th.id)}</span>`
        + closeBtn + `</div>`;
    }).join("");

    if (archived.length > 0) {
      html += `<div class="chat-thread-dd-sep"></div>`;
      archived.forEach(th => {
        html += `<div class="chat-thread-dd-item archived" data-thread="${escapeHtml(th.id)}">`
          + `<span class="chat-thread-dd-label">${escapeHtml(th.label || th.id)}</span>`
          + `<span class="chat-thread-dd-restore">↩</span></div>`;
      });
    }

    html += `<div class="chat-thread-dd-sep"></div>`;
    html += `<div class="chat-thread-dd-new" data-chat-id="chatThreadDdNew">＋ 新しいスレッド</div>`;
    menu.innerHTML = html;

    menu.querySelectorAll(".chat-thread-dd-item:not(.archived)").forEach(el => {
      el.addEventListener("click", e => {
        if (e.target.classList.contains("chat-thread-dd-close")) return;
        const tid = el.dataset.thread;
        if (tid) selectThread(tid);
        $("chatThreadDropdown")?.classList.remove("open");
      });
    });
    menu.querySelectorAll(".chat-thread-dd-close").forEach(btn => {
      btn.addEventListener("click", e => {
        e.stopPropagation();
        const tid = btn.dataset.thread;
        if (tid) closeThread(tid);
        renderThreadDropdownMenu();
      });
    });
    menu.querySelectorAll(".chat-thread-dd-item.archived").forEach(el => {
      el.addEventListener("click", () => {
        const tid = el.dataset.thread;
        if (tid) restoreThread(tid);
        $("chatThreadDropdown")?.classList.remove("open");
      });
    });
    const newBtn = $("chatThreadDdNew");
    if (newBtn) {
      newBtn.addEventListener("click", () => {
        createNewThread();
        $("chatThreadDropdown")?.classList.remove("open");
      });
    }
  }

  function _updateThreadDropdownLabel() {
    const label = $("chatThreadDropdownLabel");
    if (!label || !state.selectedAnima) return;
    const list = state.threads[state.selectedAnima] || [{ id: "default", label: "メイン" }];
    const active = list.find(th => th.id === state.selectedThreadId);
    label.textContent = active?.label || "メイン";
  }

  return { renderThreadTabs, selectThread, createNewThread, closeThread, restoreThread, renderThreadDropdownMenu };
}
