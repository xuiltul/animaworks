// ── Pane Host — manages multiple independent Chat pane instances ──
import { t } from "/shared/i18n.js";
import { createChatContext, CONSTANTS } from "./ctx.js";
import { createAnimaController } from "./anima-controller.js";
import { createThreadController } from "./thread-controller.js";
import { createChatRenderer } from "./chat-renderer.js";
import { createAvatarController } from "./avatar-controller.js";
import { createActivityController } from "./activity-controller.js";
import { createStreamingController } from "./streaming-controller.js";
import { createHistoryController } from "./history-controller.js";
import { createMemoryController } from "./memory-controller.js";
import { createSidebarController } from "./sidebar-controller.js";
import { createEventsController } from "./events-controller.js";
import { createImageVoiceController } from "./image-voice-controller.js";
import { initSplitter } from "./splitter.js";
import { onEvent } from "../../modules/websocket.js";

const LAYOUT_KEY = "aw-chat-pane-layout";

function paneHtml() {
  return `
    <div class="chat-page-main">
      <div class="chat-anima-tabs-header" data-chat-id="chatAnimaTabsHeader">
        <button class="chat-unified-hamburger" data-chat-id="chatUnifiedHamburger" aria-label="${t("pane.menu")}">&#x2630;</button>
        <div class="anima-tabs" data-chat-id="chatAnimaTabs"></div>
        <div class="chat-header-actions">
          <div class="chat-add-conversation" data-chat-id="chatAddConversationArea">
            <button type="button" data-chat-id="chatAddConversationBtn" class="chat-add-conversation-btn">${t("chat.anima_select")}</button>
            <div data-chat-id="chatAddConversationMenu" class="chat-add-conversation-menu" role="listbox" aria-label="${t("chat.anima_select")}"></div>
          </div>
        </div>
        <div class="chat-thread-dropdown" data-chat-id="chatThreadDropdown">
          <button type="button" class="chat-thread-dropdown-btn" data-chat-id="chatThreadDropdownBtn" aria-label="${t("pane.thread_label")}">
            <span class="chat-thread-dropdown-label" data-chat-id="chatThreadDropdownLabel">${t("thread.default_label")}</span>
            <svg class="chat-thread-dropdown-chevron" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
          <div class="chat-thread-dropdown-menu" data-chat-id="chatThreadDropdownMenu"></div>
        </div>
        <button class="chat-split-pane-btn" data-chat-id="chatSplitPaneBtn" aria-label="${t("pane.split")}" title="${t("pane.split")}">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/></svg>
        </button>
        <button class="chat-close-pane-btn" data-chat-id="chatClosePaneBtn" aria-label="${t("pane.close")}" title="${t("pane.close")}" style="display:none">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
        <button class="chat-unified-info-btn" data-chat-id="chatUnifiedInfoBtn" aria-label="${t("pane.info_panel")}" title="${t("pane.info_panel")}">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
        </button>
        <button class="chat-unified-user-btn" data-chat-id="chatUnifiedUserBtn" aria-label="${t("pane.user")}">
          <span class="chat-unified-user-initial" data-chat-id="chatUnifiedUserInitial">?</span>
        </button>
        <div class="chat-unified-user-menu" data-chat-id="chatUnifiedUserMenu">
          <div class="chat-unified-user-name" data-chat-id="chatUnifiedUserName"></div>
          <div class="chat-unified-user-status" data-chat-id="chatUnifiedUserStatus"></div>
          <hr class="chat-unified-user-sep">
          <button class="chat-unified-user-logout" data-chat-id="chatUnifiedUserLogout">${t("ws.logout")}</button>
        </div>
      </div>

      <div class="thread-tabs" data-chat-id="chatThreadTabs">
        <button class="thread-tab active" data-thread="default">${t("thread.default_label")}</button>
        <button class="thread-tab-new" data-chat-id="chatNewThreadBtn" title="${t("thread.new")}">＋</button>
      </div>

      <div class="chat-messages-area">
        <div data-chat-id="chatPageMessages" class="chat-messages" style="flex:1; overflow-y:auto; padding:1rem;">
          <div class="chat-empty">${t("chat.anima_select_first")}</div>
        </div>
        <button type="button" data-chat-id="chatScrollToBottom" class="scroll-to-bottom-btn" aria-label="Scroll to bottom">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
      </div>

      <form data-chat-id="chatPageForm" class="chat-input-form">
        <div class="image-preview-bar" data-chat-id="chatPagePreviewBar" style="display:none"></div>
        <div class="pending-queue-bar" data-chat-id="chatPagePending" style="display:none">
          <div class="pending-queue-header">
            <span class="pending-queue-label" data-chat-id="chatPagePendingLabel">${t("chat.queue_label")}</span>
            <button class="pending-queue-clear" data-chat-id="chatPagePendingCancel" type="button" title="${t("chat.queue_clear_all")}">✕ all</button>
          </div>
          <div data-chat-id="chatPagePendingList"></div>
        </div>
        <div class="chat-input-wrap">
          <textarea
            data-chat-id="chatPageInput"
            class="chat-input"
            placeholder="${t("chat.placeholder")}"
            autocomplete="off"
            rows="1"
            disabled
          ></textarea>
          <div class="chat-input-actions">
            <button type="button" class="chat-attach-btn" data-chat-id="chatPageAttachBtn" title="${t("chat.attach_image")}">+</button>
            <button type="button" class="chat-queue-btn" data-chat-id="chatPageQueueBtn" disabled title="${t("chat.queue_add")}">
              <svg class="chat-queue-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M12 5v14M5 12l7 7 7-7" />
              </svg>
            </button>
            <div class="voice-controls-slot"></div>
            <button type="submit" class="chat-send-btn" data-chat-id="chatPageSendBtn" disabled>
              <svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </div>
        </div>
        <input type="file" data-chat-id="chatPageFileInput" accept="image/jpeg,image/png,image/gif,image/webp" multiple style="display:none" />
      </form>
    </div>
  `;
}

const MAX_PANES = 4;

export function createPaneHost(rootContainer) {
  const panes = [];
  let focusedIdx = 0;
  let nextId = 0;
  const _sharedListeners = [];
  let _sharedBound = false;
  let _savedPaneStates = null;

  const hostEl = rootContainer.querySelector('[data-chat-id="chatPaneHost"]');

  function _isMobile() {
    return window.matchMedia("(max-width: 768px)").matches;
  }

  function addPane() {
    if (panes.length >= MAX_PANES) return null;
    if (_isMobile() && panes.length >= 1) return null;

    const id = nextId++;
    const paneEl = document.createElement("div");
    paneEl.className = "chat-pane" + (panes.length === 0 ? " focused" : "");
    paneEl.dataset.paneId = String(id);
    paneEl.innerHTML = paneHtml();

    if (panes.length > 0) {
      const splitterEl = document.createElement("div");
      splitterEl.className = "chat-pane-splitter";
      hostEl.appendChild(splitterEl);
      initSplitter(splitterEl, hostEl, { onResize: _saveSplitterWidths });
    }
    hostEl.appendChild(paneEl);

    const ctx = createChatContext();
    ctx.state.container = paneEl;
    ctx.state.rootContainer = rootContainer;
    ctx.state.paneId = id;
    ctx.state.paneIdx = panes.length;
    ctx.state.paneHost = { splitPane, removePane, savePaneStates, getPaneState };

    ctx.controllers.anima = createAnimaController(ctx);
    ctx.controllers.thread = createThreadController(ctx);
    ctx.controllers.renderer = createChatRenderer(ctx);
    ctx.controllers.avatar = createAvatarController(ctx);
    ctx.controllers.activity = createActivityController(ctx);
    ctx.controllers.streaming = createStreamingController(ctx);
    ctx.controllers.history = createHistoryController(ctx);
    ctx.controllers.memory = createMemoryController(ctx);
    ctx.controllers.sidebar = createSidebarController(ctx);
    ctx.controllers.events = createEventsController(ctx);
    ctx.controllers.imageVoice = createImageVoiceController(ctx);

    const pane = { id, el: paneEl, ctx, intervals: [] };
    panes.push(pane);

    paneEl.addEventListener("pointerdown", () => _handlePaneFocus(id), true);
    paneEl.addEventListener("focusin", () => _handlePaneFocus(id));

    ctx.controllers.sidebar.initRightPaneVisibility();
    ctx.controllers.events.bindPaneEvents();
    ctx.controllers.anima.loadAnimas();

    const chatInterval = setInterval(
      () => ctx.controllers.renderer.pollSelectedChat(),
      CONSTANTS.CHAT_POLL_INTERVAL_MS,
    );
    pane.intervals.push(chatInterval);

    const unsubBootstrap = onEvent("anima.bootstrap", (data) => {
      const { name, status: bsStatus } = data;
      const anima = ctx.state.animas.find(a => a.name === name);
      if (!anima) return;

      if (bsStatus === "started") {
        anima.status = "bootstrapping";
        anima.bootstrapping = true;
        anima._bootstrapStartedAt = Date.now();
        anima._bootstrapFailed = null;
      } else if (bsStatus === "completed") {
        anima.status = "idle";
        anima.bootstrapping = false;
        anima._bootstrapFailed = null;
        if (name === ctx.state.selectedAnima) {
          ctx.controllers.renderer.showBootstrapComplete(
            ctx.$("chatPageMessages"), anima
          );
          setTimeout(() => ctx.controllers.renderer.renderChat(), 1000);
        }
      } else if (bsStatus === "failed") {
        anima.status = "error";
        anima.bootstrapping = false;
        anima._bootstrapFailed = "failed";
      } else if (bsStatus === "max_retries_exceeded") {
        anima.status = "error";
        anima.bootstrapping = false;
        anima._bootstrapFailed = "max_retries";
      }

      if (name === ctx.state.selectedAnima && bsStatus !== "completed") {
        ctx.controllers.renderer.renderChat();
      }
      ctx.controllers.anima.renderAnimaTabs();
    });
    pane.intervals.push(unsubBootstrap);

    if (panes.length === 1) {
      focusedIdx = 0;
      _startFocusedIntervals();
    }

    _saveLayout();
    _updatePaneControls();
    return pane;
  }

  function removePane(id) {
    const idx = panes.findIndex(p => p.id === id);
    if (idx < 0 || panes.length <= 1) return;

    const pane = panes[idx];
    _destroyPane(pane);
    panes.splice(idx, 1);

    const splitterBefore = pane.el.previousElementSibling;
    if (splitterBefore?.classList.contains("chat-pane-splitter")) {
      splitterBefore.remove();
    } else {
      const splitterAfter = pane.el.nextElementSibling;
      if (splitterAfter?.classList.contains("chat-pane-splitter")) splitterAfter.remove();
    }
    pane.el.remove();

    if (focusedIdx >= panes.length) focusedIdx = panes.length - 1;
    _applyFocusVisual();
    _syncSidebar();
    _saveLayout();
    _updatePaneControls();
  }

  function _handlePaneFocus(id) {
    const newIdx = panes.findIndex(p => p.id === id);
    if (newIdx < 0 || newIdx === focusedIdx) return;

    _clearFocusedIntervals();
    focusedIdx = newIdx;
    _applyFocusVisual();
    _syncSidebar();
    _startFocusedIntervals();
    _syncVoiceToFocused();
  }

  function _applyFocusVisual() {
    for (let i = 0; i < panes.length; i++) {
      panes[i].el.classList.toggle("focused", i === focusedIdx);
    }
  }

  let _focusedActivityInterval = null;

  function _startFocusedIntervals() {
    const ctx = getFocused()?.ctx;
    if (!ctx) return;
    _focusedActivityInterval = setInterval(
      () => ctx.controllers.activity.loadActivity(),
      30000,
    );
  }

  function _clearFocusedIntervals() {
    if (_focusedActivityInterval) {
      clearInterval(_focusedActivityInterval);
      _focusedActivityInterval = null;
    }
  }

  function _syncSidebar() {
    const ctx = getFocused()?.ctx;
    if (!ctx) return;
    ctx.controllers.activity.renderAnimaState();
    ctx.controllers.activity.loadActivity();
    ctx.controllers.memory.loadMemoryTab();
    if (ctx.state.activeRightTab === "history" && ctx.state.selectedAnima) {
      ctx.controllers.history.loadSessionList();
    }
  }

  function _syncVoiceToFocused() {
    const ctx = getFocused()?.ctx;
    if (!ctx || !ctx.state.selectedAnima) return;
    ctx.controllers.imageVoice.updateVoiceAnima(ctx.state.selectedAnima);
  }

  function getFocused() {
    return panes[focusedIdx] || null;
  }

  function bindSharedEvents() {
    if (_sharedBound) return;
    _sharedBound = true;

    const bustupEscapeHandler = (e) => {
      if (e.key === "Escape") getFocused()?.ctx.controllers.avatar.dismissBustupOverlay();
    };
    document.addEventListener("keydown", bustupEscapeHandler);
    _sharedListeners.push({ el: document, event: "keydown", handler: bustupEscapeHandler });

    const $r = (id) => rootContainer.querySelector(`[data-chat-id="${id}"]`);

    function _addShared(id, event, handler) {
      const el = $r(id);
      if (!el) return;
      el.addEventListener(event, handler);
      _sharedListeners.push({ el, event, handler });
    }

    for (const tabId of ["chatMobileTabChat", "chatMobileTabInfo"]) {
      _addShared(tabId, "click", e => {
        getFocused()?.ctx.controllers.sidebar.switchMobileTab(e.target.dataset.panel);
      });
    }

    for (const tabId of ["chatTabState", "chatTabActivity", "chatTabHistory"]) {
      _addShared(tabId, "click", e => {
        getFocused()?.ctx.controllers.sidebar.switchRightTab(e.target.dataset.tab);
      });
    }

    _addShared("chatRightPaneToggleBtn", "click", () => {
      getFocused()?.ctx.controllers.sidebar.toggleRightPane();
    });

    rootContainer.querySelectorAll(".memory-tab").forEach(btn => {
      const handler = () => {
        const ctx = getFocused()?.ctx;
        if (!ctx) return;
        ctx.state.activeMemoryTab = btn.dataset.tab;
        rootContainer.querySelectorAll(".memory-tab").forEach(b =>
          b.classList.toggle("active", b.dataset.tab === ctx.state.activeMemoryTab),
        );
        const contentArea = ctx.$root("chatMemoryContentArea");
        const fileList = ctx.$root("chatMemoryFileList");
        if (contentArea) contentArea.style.display = "none";
        if (fileList) fileList.style.display = "";
        ctx.controllers.memory.loadMemoryTab();
      };
      btn.addEventListener("click", handler);
      _sharedListeners.push({ el: btn, event: "click", handler });
    });

    _addShared("chatMemoryBackBtn", "click", () => {
      const ctx = getFocused()?.ctx;
      if (!ctx) return;
      const ca = ctx.$root("chatMemoryContentArea");
      const fl = ctx.$root("chatMemoryFileList");
      if (ca) ca.style.display = "none";
      if (fl) fl.style.display = "";
    });

    _addShared("chatHistoryBackBtn", "click", () => {
      const ctx = getFocused()?.ctx;
      if (!ctx) return;
      const detail = ctx.$root("chatHistoryDetail");
      const list = ctx.$root("chatHistorySessionList");
      if (detail) detail.style.display = "none";
      if (list) list.style.display = "";
    });
  }

  function _destroyPane(pane) {
    const { ctx } = pane;
    for (const entry of pane.intervals) {
      if (typeof entry === "function") entry();
      else clearInterval(entry);
    }
    pane.intervals = [];
    ctx.controllers.renderer.clearBootstrapInterval();

    if (ctx.state.chatUiStateSaveTimer) {
      clearTimeout(ctx.state.chatUiStateSaveTimer);
      ctx.state.chatUiStateSaveTimer = null;
    }
    for (const { el, event, handler } of ctx.state.boundListeners) {
      el.removeEventListener(event, handler);
    }
    ctx.state.boundListeners = [];
    if (ctx.state.chatObserver) {
      ctx.state.chatObserver.disconnect();
      ctx.state.chatObserver = null;
    }
    if (ctx.state.manager) {
      for (const anima of ctx.state.animas) {
        ctx.state.manager.destroyAllForAnima(anima.name);
      }
    }
    ctx.controllers.avatar.removeBustupOverlay();
  }

  function destroy() {
    _clearFocusedIntervals();
    for (const { el, event, handler } of _sharedListeners) {
      el.removeEventListener(event, handler);
    }
    _sharedListeners.length = 0;
    _sharedBound = false;

    for (const pane of panes) _destroyPane(pane);
    panes.length = 0;
    hostEl.innerHTML = "";
  }

  function _saveLayout() {
    try {
      const layout = {
        paneCount: panes.length,
        focusedIdx,
        widths: panes.map(p => p.el.style.flex || ""),
        paneStates: panes.map(p => ({
          anima: p.ctx.state.selectedAnima || null,
          threadId: p.ctx.state.selectedThreadId || "default",
        })),
      };
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
    } catch { /* quota exceeded */ }
  }

  function savePaneStates() {
    _saveLayout();
  }

  function getPaneState(idx) {
    if (!_savedPaneStates || idx < 0 || idx >= _savedPaneStates.length) return null;
    return _savedPaneStates[idx];
  }

  function _saveSplitterWidths(widths) {
    try {
      const raw = localStorage.getItem(LAYOUT_KEY);
      const existing = raw ? JSON.parse(raw) : {};
      existing.widths = widths;
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(existing));
    } catch { /* quota exceeded */ }
  }

  function restoreLayout() {
    try {
      const raw = localStorage.getItem(LAYOUT_KEY);
      if (!raw) return;
      const layout = JSON.parse(raw);
      if (!layout || typeof layout.paneCount !== "number") return;

      if (Array.isArray(layout.paneStates)) {
        _savedPaneStates = layout.paneStates;
      }

      const count = _isMobile() ? 1 : Math.min(layout.paneCount, MAX_PANES);
      while (panes.length < count) addPane();

      if (layout.widths && Array.isArray(layout.widths)) {
        for (let i = 0; i < panes.length && i < layout.widths.length; i++) {
          if (layout.widths[i]) panes[i].el.style.flex = layout.widths[i];
        }
      }
      if (typeof layout.focusedIdx === "number" && layout.focusedIdx < panes.length) {
        focusedIdx = layout.focusedIdx;
        _applyFocusVisual();
      }
    } catch { /* corrupt data */ }
  }

  function _updatePaneControls() {
    const multi = panes.length > 1;
    const full = panes.length >= MAX_PANES;
    for (const p of panes) {
      const closeBtn = p.el.querySelector('[data-chat-id="chatClosePaneBtn"]');
      const splitBtn = p.el.querySelector('[data-chat-id="chatSplitPaneBtn"]');
      if (closeBtn) closeBtn.style.display = multi ? "" : "none";
      if (splitBtn) splitBtn.style.display = full ? "none" : "";
    }
  }

  function splitPane() {
    if (_isMobile()) return null;
    return addPane();
  }

  return {
    addPane,
    removePane,
    splitPane,
    getFocused,
    bindSharedEvents,
    restoreLayout,
    destroy,
    get panes() { return panes; },
  };
}
