// ── Pane Host — manages multiple independent Chat pane instances ──
import { t } from "/shared/i18n.js";
import { api } from "../../modules/api.js";
import { escapeHtml } from "../../modules/state.js";
import { createChatContext, CONSTANTS, modelKey, syncModelSelect, updateModelSelectTitle, scheduleSaveChatUiState } from "./ctx.js";
import { fetchModelCatalog, populateModelSelect } from "../../shared/chat/model-picker.js";
import { createAnimaController } from "./anima-controller.js";
import { createThreadController } from "./thread-controller.js";
import { createChatRenderer } from "./chat-renderer.js";
import { createAvatarController } from "./avatar-controller.js";
import { createActivityController } from "./activity-controller.js";
import { createMemoryPopover } from "./memory-popover.js";
import { createStreamingController } from "./streaming-controller.js";
import { createEventsController } from "./events-controller.js";
import { createImageVoiceController } from "./image-voice-controller.js";
import { createWorkIndicatorController } from "./work-indicator-controller.js";
import { initSplitter } from "./splitter.js";
import { onEvent } from "../../modules/websocket.js";
import { invalidateAvatarCache } from "../../modules/avatar-resolver.js";
import { presence, initPresence } from "../../modules/anima-presence.js";
import { wireReplyToasts } from "../../modules/presence-ui.js";
import { createPaneHead } from "./pane-head.js";
import { createAnimaRail } from "./anima-rail.js";
import { createPresencePanel } from "./presence-panel.js";
import { createSlashPalette } from "./slash-palette.js";
import { startSessionsRefreshFallback } from "./session-refresh.js";

const LAYOUT_KEY = "aw-chat-pane-layout";
const PHONE_QUERY = window.matchMedia("(max-width: 560px)");

function paneHtml() {
  return `
    <div class="chat-page-main">
      <div class="chat-anima-tabs-header" data-chat-id="chatAnimaTabsHeader">
        <button class="chat-unified-hamburger" data-chat-id="chatUnifiedHamburger" aria-label="${t("pane.menu")}">&#x2630;</button>
        <div class="anima-tabs" data-chat-id="chatAnimaTabs"></div>
        <div class="chat-pane-head-id" data-chat-id="chatPaneHeadId">
          <img class="chat-pane-head-face" data-chat-id="chatPaneHeadFace" alt="" hidden>
          <div class="chat-pane-head-text">
            <div class="chat-pane-head-name" data-chat-id="chatPaneHeadName"></div>
            <div class="chat-pane-head-status" data-chat-id="chatPaneHeadStatus"></div>
          </div>
          <button type="button" class="chat-rail-toggle" data-chat-id="chatRailToggle" aria-expanded="false" aria-label="${t("chat.rail.toggle")}" title="${t("chat.rail.toggle")}">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
        </div>
        <div class="thread-tabs" data-chat-id="chatThreadTabs">
          <button class="thread-tab active" data-thread="default">${t("thread.default_label")}</button>
          <button class="thread-tab-new" data-chat-id="chatNewThreadBtn" title="${t("thread.new")}" aria-label="${escapeHtml(t("thread.new"))}">＋</button>
        </div>
        <div class="chat-pane-head-tools" data-slot="head-tools"></div>
        <div class="chat-header-actions">
          <div class="chat-add-conversation" data-chat-id="chatAddConversationArea">
            <button type="button" data-chat-id="chatAddConversationBtn" class="chat-add-conversation-btn">${t("chat.anima_select")}</button>
            <div data-chat-id="chatAddConversationMenu" class="chat-add-conversation-menu" role="listbox" aria-label="${t("chat.anima_select")}"></div>
          </div>
          <div class="chat-pane-head-tools" data-slot="head-tools">
            <label class="chat-bg-toggle" title="${t("chat.line.show_background")}">
              <input type="checkbox" data-chat-id="chatShowBackground" aria-label="${t("chat.line.show_background")}">
              <span>${t("chat.line.show_background")}</span>
            </label>
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
        <button class="chat-unified-info-btn" data-chat-id="chatUnifiedInfoBtn" aria-label="${t("chat.memory.open")}" title="${t("chat.memory.open")}">
          <span aria-hidden="true">◎</span><span>${t("chat.memory.open")}</span>
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

      <div class="chat-messages-area">
        <div data-chat-id="chatPageMessages" class="chat-messages chat-line-talk" style="flex:1; overflow-y:auto; padding:1rem;">
          <div class="chat-empty">${t("chat.anima_select_first")}</div>
        </div>
      </div>

      <form data-chat-id="chatPageForm" class="chat-input-form">
        <button type="button" data-chat-id="chatScrollToBottom" class="scroll-to-bottom-btn" aria-label="${t("chat.line.scroll_bottom")}" title="${t("chat.line.scroll_bottom")}">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </button>
        <div class="chat-line-reply-chip" data-chat-id="chatReplyChip" hidden>
          <span data-chat-id="chatReplyText"></span>
          <button type="button" data-chat-id="chatReplyCancel" aria-label="${t("chat.line.cancel_reply")}">×</button>
        </div>
        <div class="image-preview-bar" data-chat-id="chatPagePreviewBar" style="display:none"></div>
        <div class="chat-work-indicator" data-chat-id="chatWorkIndicator" hidden></div>
        <div class="pending-queue-bar" data-chat-id="chatPagePending" style="display:none">
          <div class="pending-queue-header">
            <span class="pending-queue-label" data-chat-id="chatPagePendingLabel">${t("chat.queue_label")}</span>
            <button class="pending-queue-clear" data-chat-id="chatPagePendingCancel" type="button" title="${t("chat.queue_clear_all")}" aria-label="${escapeHtml(t("chat.queue_clear_all"))}">✕ all</button>
          </div>
          <div data-chat-id="chatPagePendingList"></div>
        </div>
        <div class="chat-prompt-chips" data-chat-id="chatPromptChips" style="display:none"></div>
        <div class="chat-input-wrap">
          <textarea
            data-chat-id="chatPageInput"
            class="chat-input"
            placeholder="${t("chat.placeholder")}"
            aria-label="${escapeHtml(t("chat.message_input"))}"
            autocomplete="off"
            rows="1"
            disabled
          ></textarea>
          <div class="chat-input-actions">
            <button type="button" class="chat-attach-btn" data-chat-id="chatPageAttachBtn" title="${t("chat.attach.add_file")}" aria-label="${escapeHtml(t("chat.attach.add_file"))}">+</button>
            <select class="chat-model-select" data-chat-id="chatPageModel" data-i18n-title="chat.model_selector" title="${t("chat.model_selector")}" aria-label="${escapeHtml(t("chat.model_selector"))}">
              <option value=""></option>
            </select>
            <div class="context-ring-wrap" data-chat-id="chatContextRing" title="${t("chat.context_ring_title")}">
              <span class="context-ring-label" data-chat-id="chatContextLabel">${t("chat.presence.memory_label")} 0%</span>
              <svg class="context-ring" viewBox="0 0 36 36" aria-hidden="true">
                <circle class="context-ring-bg" cx="18" cy="18" r="15.5" fill="none" stroke-width="3"/>
                <circle class="context-ring-fg" cx="18" cy="18" r="15.5" fill="none" stroke-width="3"
                  stroke-dasharray="97.4" stroke-dashoffset="97.4"
                  transform="rotate(-90 18 18)"/>
              </svg>
            </div>
            <div class="voice-controls-slot"></div>
            <button type="submit" class="chat-send-btn" data-chat-id="chatPageSendBtn" disabled aria-label="${escapeHtml(t("chat.send"))}">
              <svg class="chat-send-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </div>
        </div>
        <input type="file" data-chat-id="chatPageFileInput" multiple style="display:none" />
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

  // ── Presence: shared store, left rail, header / panel refresh ──
  const _syncedStreaming = new Set();
  let _refreshQueued = false;
  const layoutEl = hostEl.parentElement;
  const railEl = document.createElement("nav");
  railEl.className = "chat-rail";
  railEl.setAttribute("aria-label", t("chat.presence.rail_label"));
  layoutEl.insertBefore(railEl, hostEl);

  /** Phones keep the rail folded away until the header's name is tapped. */
  function setRailOpen(open) {
    layoutEl.classList.toggle("rail-open", open);
    hostEl.querySelectorAll('[data-chat-id="chatRailToggle"]').forEach((btn) => btn.setAttribute("aria-expanded", String(open)));
  }

  const rail = createAnimaRail({
    el: railEl,
    getPanes: () => panes,
    getFocused,
    t,
    escapeHtml,
    onToolsChanged: () => refreshPresence(),
    onPick: () => setRailOpen(false),
  });
  const _presenceOffs = [];

  /** Pull stream / unread facts the chat controllers already hold into the store. */
  function _syncPresenceFromPanes() {
    const streaming = new Set();
    const shown = new Set(panes.map((p) => p.ctx.state.selectedAnima).filter(Boolean));
    for (const p of panes) {
      for (const a of p.ctx.state.animas || []) {
        if (p.ctx.state.manager?.isStreamingForAnima(a.name)) streaming.add(a.name);
      }
    }
    const fstate = getFocused()?.ctx.state;
    for (const a of fstate?.animas || []) {
      const threads = fstate.threads?.[a.name] || [];
      const tab = fstate.animaTabs.find((x) => x.name === a.name);
      const n = threads.filter((th) => th.unread).length || (tab?.unreadStar ? 1 : 0);
      presence.setUnread(a.name, shown.has(a.name) ? 0 : n);
    }
    for (const name of streaming) {
      if (!_syncedStreaming.has(name)) {
        presence.setWorking(name, true, "stream");
        presence.setAwaiting(name, false);
        _syncedStreaming.add(name);
      }
    }
    for (const name of [..._syncedStreaming]) {
      if (!streaming.has(name)) {
        presence.setWorking(name, false, "stream");
        _syncedStreaming.delete(name);
      }
    }
    presence.setVisibleAnimas(panes.map((p) => p.ctx.state.selectedAnima));
  }

  function _repaintPresence() {
    _refreshQueued = false;
    rail.render();
    for (const p of panes) {
      p.ctx.controllers.head?.refresh();
      p.ctx.controllers.presencePanel?.refresh();
    }
  }

  /** Re-sync from the chat controllers and repaint (coalesced per frame). */
  function refreshPresence() {
    _syncPresenceFromPanes();
    if (_refreshQueued) return;
    _refreshQueued = true;
    requestAnimationFrame(_repaintPresence);
  }

  function renderRail() { refreshPresence(); }

  initPresence({ api, onEvent });
  wireReplyToasts();
  _presenceOffs.push(presence.subscribe(() => { if (!_refreshQueued) { _refreshQueued = true; requestAnimationFrame(_repaintPresence); } }));
  const _onOpenAnima = (ev) => {
    const name = ev.detail?.anima;
    const focused = getFocused();
    if (!name || !focused) return;
    presence.takePendingOpen();
    focused.ctx.controllers.anima.openOrSelectAnima(name);
  };
  document.addEventListener("aw:open-anima", _onOpenAnima);
  _presenceOffs.push(() => document.removeEventListener("aw:open-anima", _onOpenAnima));

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
    ctx.state.paneHost = { splitPane, removePane, savePaneStates, getPaneState, renderRail, getOpenThreadIds };

    ctx.controllers.anima = createAnimaController(ctx);
    ctx.controllers.thread = createThreadController(ctx);
    ctx.controllers.renderer = createChatRenderer(ctx);
    ctx.controllers.avatar = createAvatarController(ctx);
    ctx.controllers.activity = createActivityController(ctx);
    ctx.controllers.memoryPopover = createMemoryPopover(ctx);
    ctx.controllers.streaming = createStreamingController(ctx);
    ctx.controllers.events = createEventsController(ctx);
    ctx.controllers.imageVoice = createImageVoiceController(ctx);
    ctx.controllers.workIndicator = createWorkIndicatorController(ctx);
    ctx.controllers.head = createPaneHead(ctx);
    ctx.controllers.presencePanel = createPresencePanel(ctx);
    ctx.controllers.slash = createSlashPalette(ctx);

    const pane = { id, el: paneEl, ctx, intervals: [] };
    panes.push(pane);

    paneEl.addEventListener("pointerdown", () => _handlePaneFocus(id), true);
    paneEl.addEventListener("focusin", () => _handlePaneFocus(id));

    _initModelSelect(ctx, paneEl);

    ctx.controllers.events.bindPaneEvents();
    ctx.controllers.anima.loadAnimas();

    // A member hired after this page loaded is otherwise unknown here, so its
    // DMs with the leader would not fold into the activity bundle.
    const onAnimasChanged = () => {
      api("/api/animas")
        .then((list) => { if (Array.isArray(list)) ctx.state.animas = list; })
        .catch(() => {});
    };
    window.addEventListener("aw:animas-changed", onAnimasChanged);
    ctx.state.boundListeners.push({ el: window, event: "aw:animas-changed", handler: onAnimasChanged });

    // Right after server start the Anima processes may not be registered yet;
    // keep re-fetching until the list is populated instead of staying empty.
    const emptyListRetry = setInterval(() => {
      if (ctx.state.animas.length === 0) ctx.controllers.anima.loadAnimas();
    }, 5000);
    pane.intervals.push(emptyListRetry);

    const chatInterval = setInterval(
      () => ctx.controllers.renderer.pollSelectedChat(),
      CONSTANTS.CHAT_POLL_INTERVAL_MS,
    );
    pane.intervals.push(chatInterval);

    pane.intervals.push(startSessionsRefreshFallback(
      () => ctx.controllers.anima.refreshSessions(),
    ));

    let sessionsWsTimer = null;
    const scheduleSessionsRefresh = (data = {}, eventType = "") => {
      if (eventType === "anima.status" && ["busy", "running_task", "bootstrapping"].includes(data.status)) return;
      const selected = ctx.state.selectedAnima;
      const related = [data.name, data.anima, data.from_person, data.to_person];
      if (!selected || !related.includes(selected)) return;
      clearTimeout(sessionsWsTimer);
      sessionsWsTimer = setTimeout(() => {
        sessionsWsTimer = null;
        void ctx.controllers.anima.refreshSessions(selected);
      }, 250);
    };
    for (const eventType of ["anima.status", "anima.heartbeat", "anima.interaction", "anima.proactive_message", "chat.response"]) {
      pane.intervals.push(onEvent(eventType, (data) => scheduleSessionsRefresh(data, eventType)));
    }
    pane.intervals.push(() => clearTimeout(sessionsWsTimer));

    const unsubBootstrap = onEvent("anima.bootstrap", (data) => {
      const { name, status: bsStatus } = data;
      const anima = ctx.state.animas.find(a => a.name === name);
      if (!anima) return;

      if (bsStatus === "started") {
        // Interactive setup is a conversation: keep its messages and input
        // visible while the Anima asks questions or writes its profile.
        if (anima.bootstrap_state?.mode === "interactive") return;
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
        anima.bootstrap_state = data.bootstrap_state || anima.bootstrap_state;
      } else if (bsStatus === "needs_repair") {
        anima.status = "error";
        anima.bootstrapping = false;
        anima.needs_repair = true;
        anima.bootstrap_state = data.bootstrap_state || anima.bootstrap_state;
      } else if (bsStatus === "max_retries_exceeded") {
        anima.status = "error";
        anima.bootstrapping = false;
        anima._bootstrapFailed = "max_retries";
        anima.needs_repair = true;
        anima.bootstrap_state = data.bootstrap_state || anima.bootstrap_state;
      }

      if (name === ctx.state.selectedAnima && bsStatus !== "completed") {
        ctx.controllers.renderer.renderChat();
      }
      ctx.controllers.anima.renderAnimaTabs();
    });
    pane.intervals.push(unsubBootstrap);

    const refreshAvatar = async (name) => {
      if (!name || !ctx.state.animas.some(a => a.name === name)) return;
      await invalidateAvatarCache(name);
      delete ctx.state.animaTabAvatarUrls[name];
      ctx.controllers.anima.renderAnimaTabs();
      if (name === ctx.state.selectedAnima) {
        await ctx.controllers.avatar.updateAvatar();
        ctx.controllers.renderer.renderChat(false);
      }
    };
    pane.intervals.push(onEvent("anima.assets_updated", ({ name }) => {
      refreshAvatar(name).catch(() => {});
    }));
    // CLI/background generation may produce the first portrait before its
    // completion event. Retry an absent avatar without reloading the page.
    pane.intervals.push(setInterval(() => {
      const name = ctx.state.selectedAnima;
      if (name && !ctx.state.animaTabAvatarUrls[name] && !ctx.state.animaTabAvatarLoading[name]) {
        refreshAvatar(name).catch(() => {});
      }
    }, 30000));

    if (panes.length === 1) focusedIdx = 0;

    _saveLayout();
    _updatePaneControls();
    return pane;
  }

  function _initModelSelect(ctx, paneEl) {
    const select = paneEl.querySelector('[data-chat-id="chatPageModel"]');
    if (!select) return;

    // Initial text of the anima-default option comes from JS (i18n).
    const defaultOpt = select.options[0];
    if (defaultOpt) defaultOpt.textContent = t("chat.model_default");

    // Populate the picker once; the catalog is shared/memoized across panes.
    // Initial selection is the anima default; anima/thread switches will
    // sync the stored per-thread value via syncModelSelect().
    fetchModelCatalog()
      .then(catalog => {
        if (!select.isConnected) return;
        populateModelSelect(select, catalog, "", {
          defaultLabel: t("chat.model_default"),
          otherLabel: t("chat.model_group_other"),
          errorTitle: t("chat.model_load_error"),
        });
        syncModelSelect(ctx);
      })
      .catch(() => {});

    select.addEventListener("change", () => {
      const { selectedAnima, selectedThreadId } = ctx.state;
      if (!selectedAnima) return;
      ctx.state.modelByThread[modelKey(selectedAnima, selectedThreadId)] = select.value;
      updateModelSelectTitle(select);
      scheduleSaveChatUiState(ctx);
    });
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
    _saveLayout();
    _updatePaneControls();
  }

  function _handlePaneFocus(id) {
    const newIdx = panes.findIndex(p => p.id === id);
    if (newIdx < 0 || newIdx === focusedIdx) return;

    focusedIdx = newIdx;
    _applyFocusVisual();
    _syncVoiceToFocused();
  }

  function _applyFocusVisual() {
    for (let i = 0; i < panes.length; i++) {
      panes[i].el.classList.toggle("focused", i === focusedIdx);
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

    // Phones: the anima's name in the header folds the rail out; any other tap folds it away.
    const railToggleHandler = (e) => {
      if (!PHONE_QUERY.matches) return;
      const open = layoutEl.classList.contains("rail-open");
      if (e.target.closest?.(".chat-pane-head-status.wait")) return;
      if (e.target.closest?.('[data-chat-id="chatRailToggle"], .chat-pane-head-text')) setRailOpen(!open);
      else if (open) setRailOpen(false);
    };
    hostEl.addEventListener("click", railToggleHandler);
    _sharedListeners.push({ el: hostEl, event: "click", handler: railToggleHandler });
  }

  function _destroyPane(pane) {
    const { ctx } = pane;
    for (const entry of pane.intervals) {
      if (typeof entry === "function") entry();
      else clearInterval(entry);
    }
    pane.intervals = [];
    ctx.controllers.renderer.clearBootstrapInterval();
    ctx.controllers.memoryPopover?.destroy();
    ctx.controllers.workIndicator?.destroy();
    ctx.controllers.presencePanel?.destroy();
    ctx.controllers.slash?.destroy();

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
    for (const { el, event, handler } of _sharedListeners) {
      el.removeEventListener(event, handler);
    }
    _sharedListeners.length = 0;
    _sharedBound = false;

    for (const pane of panes) _destroyPane(pane);
    panes.length = 0;
    for (const off of _presenceOffs) off();
    _presenceOffs.length = 0;
    presence.setVisibleAnimas([]);
    railEl.remove();
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

  function getOpenThreadIds(animaName) {
    return panes
      .filter((pane) => pane.ctx.state.selectedAnima === animaName)
      .map((pane) => pane.ctx.state.selectedThreadId || "default");
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
    hostEl.classList.toggle("has-split", multi);
    refreshPresence();
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
