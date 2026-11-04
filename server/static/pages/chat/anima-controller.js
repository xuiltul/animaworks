// ── Anima Selection / Tab / Avatar Controller ──
import {
  isTabOpen, refreshAnimaUnread,
  clearUnreadForActiveThread, loadDraft, saveDraft, chatInputMaxHeight,
  fetchChatUiState, scheduleSaveChatUiState, mergeThreadsFromSessions,
  setThreadUnread, threadTimeValue, syncModelSelect, CONSTANTS,
} from "./ctx.js";
import { hasHistoryContent } from "../../shared/chat/history-loader.js";
import { animaStatusLabel } from "../../shared/anima-status.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";
import { companyColor } from "../../shared/avatar-utils.js";
import { openCallScreen } from "./call-screen.js";
import { presence, displayStatus } from "../../modules/anima-presence.js";

const ANIMA_TAB_LONG_PRESS_MS = 500;

export function createAnimaController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { api, escapeHtml, t, logger } = deps;
  let _selectGen = 0;
  let _tooltipEl = null;
  let _tooltipHideTimer = null;
  let _callButton = null;
  const _sessionsInFlight = new Map();

  function refreshSessions(name = state.selectedAnima) {
    if (!name) return Promise.resolve(null);
    if (_sessionsInFlight.has(name)) return _sessionsInFlight.get(name);

    const promise = api(`/api/animas/${encodeURIComponent(name)}/sessions`)
      .then((sessionsData) => {
        const previousLastTs = new Map(
          (state.threads[name] || []).map((thread) => [thread.id, threadTimeValue(thread.lastTs || "")]),
        );
        mergeThreadsFromSessions(ctx, name, sessionsData);
        for (const thread of state.threads[name] || []) {
          if (!thread?.id || thread.id === state.selectedThreadId && name === state.selectedAnima) continue;
          const previous = previousLastTs.get(thread.id) || 0;
          if (threadTimeValue(thread.lastTs || "") > previous) setThreadUnread(ctx, name, thread.id, true);
        }
        refreshAnimaUnread(ctx, name);

        if (name === state.selectedAnima) {
          if (!state.threads[name].some((thread) => thread.id === state.selectedThreadId)) {
            state.selectedThreadId = "default";
            state.activeThreadByAnima[name] = "default";
          }
          ctx.controllers.anima.renderAnimaTabs();
          ctx.controllers.thread.renderThreadTabs();
        }
        return sessionsData;
      })
      .catch(() => null)
      .finally(() => {
        if (_sessionsInFlight.get(name) === promise) _sessionsInFlight.delete(name);
      });
    _sessionsInFlight.set(name, promise);
    return promise;
  }

  function _ensureCallButton() {
    const header = state.container?.querySelector(".chat-anima-tabs-header");
    if (!header) return;

    let slot = header.querySelector('[data-slot="head-tools"]');
    if (!slot) {
      const actions = header.querySelector(".chat-header-actions");
      if (!actions) return;
      slot = document.createElement("div");
      slot.className = "chat-pane-head-tools";
      slot.dataset.slot = "head-tools";
      actions.appendChild(slot);
    }
    if (!_callButton || !_callButton.isConnected) {
      _callButton = document.createElement("button");
      _callButton.type = "button";
      _callButton.className = "chat-call-btn";
      _callButton.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6.6 10.8a15.1 15.1 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.25 11.4 11.4 0 0 0 3.6.57 1 1 0 0 1 1 1V20a1 1 0 0 1-1 1A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1c0 1.25.2 2.45.57 3.6a1 1 0 0 1-.25 1z"/></svg>';
      _callButton.addEventListener("click", () => {
        const anima = state.selectedAnima;
        if (!anima) return;
        const page = state.container?.closest(".chat-page-layout") || state.rootContainer?.querySelector(".chat-page-layout");
        openCallScreen(anima, {
          container: page,
          threadId: state.selectedThreadId || "default",
          onClose: () => {
            if (state.selectedAnima) ctx.controllers.imageVoice?.updateVoiceAnima(state.selectedAnima);
          },
          onRecord: (record) => {
            if (state.selectedAnima === anima && (state.selectedThreadId || "default") === record.thread_id) {
              ctx.controllers.renderer?.pollSelectedChat?.();
            }
          },
        });
      });
      slot.appendChild(_callButton);
    } else if (_callButton.parentElement !== slot) {
      slot.appendChild(_callButton);
    }

    const name = state.selectedAnima || "";
    _callButton.hidden = !name;
    _callButton.title = t("chat.call.call_title", { name });
    _callButton.setAttribute("aria-label", t("chat.call.call_title", { name }));
  }

  function _suppressLongPressClick(el) {
    let timer = null;
    let longPressFired = false;
    const clearTimer = (reset = false) => {
      if (timer) clearTimeout(timer);
      timer = null;
      if (reset) longPressFired = false;
    };

    el.addEventListener("pointerdown", (event) => {
      if (event.button != null && event.button !== 0) return;
      clearTimer(true);
      timer = setTimeout(() => {
        timer = null;
        longPressFired = true;
      }, ANIMA_TAB_LONG_PRESS_MS);
    });
    el.addEventListener("pointerup", () => clearTimer());
    el.addEventListener("pointerleave", () => clearTimer());
    el.addEventListener("pointercancel", () => clearTimer());
    el.addEventListener("contextmenu", (event) => {
      if (timer || longPressFired) event.preventDefault();
    });
    el.addEventListener("click", (event) => {
      if (!longPressFired) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      longPressFired = false;
    }, true);
  }

  function _getTooltip() {
    if (_tooltipEl) return _tooltipEl;
    _tooltipEl = document.createElement("div");
    _tooltipEl.className = "anima-tab-tooltip";
    document.body.appendChild(_tooltipEl);
    return _tooltipEl;
  }

  function _showTabTooltip(btn) {
    const name = btn.dataset?.anima;
    if (!name) return;
    const nameEl = btn.querySelector(".anima-tab-name");
    if (nameEl && getComputedStyle(nameEl).display !== "none") return;

    clearTimeout(_tooltipHideTimer);
    const tip = _getTooltip();
    tip.textContent = name;
    tip.classList.add("visible");

    const rect = btn.getBoundingClientRect();
    const collapsed = document.body.classList.contains("sidebar-collapsed");
    if (collapsed) {
      tip.style.left = `${rect.right + 8}px`;
      tip.style.top = `${rect.top + rect.height / 2}px`;
      tip.style.transform = "translateY(-50%)";
    } else {
      tip.style.left = `${rect.left + rect.width / 2}px`;
      tip.style.top = `${rect.bottom + 6}px`;
      tip.style.transform = "translateX(-50%)";
    }
  }

  function _hideTabTooltip() {
    clearTimeout(_tooltipHideTimer);
    if (_tooltipEl) _tooltipEl.classList.remove("visible");
  }

  function ensureAnimaTabAvatar(name) {
    if (!name) return Promise.resolve();
    if (Object.prototype.hasOwnProperty.call(state.animaTabAvatarUrls, name)) return Promise.resolve();
    if (state.animaTabAvatarLoading[name]) return state.animaTabAvatarLoading[name];

    state.animaTabAvatarLoading[name] = (async () => {
      const found = await resolveCachedAvatar(name, bustupCandidates(), "M");
      state.animaTabAvatarUrls[name] = found;
      delete state.animaTabAvatarLoading[name];
      renderAnimaTabs();
      renderAddConversationMenu();
    })();
    return state.animaTabAvatarLoading[name];
  }

  function companyRingStyle(name) {
    const anima = (state.animas || []).find(a => a.name === name);
    const color = companyColor(anima?.company);
    return color ? ` style="box-shadow:0 0 0 2px ${color}"` : "";
  }

  function buildAnimaTabAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    const url = state.animaTabAvatarUrls[name];
    const ring = companyRingStyle(name);
    if (url) {
      return `<img class="anima-tab-avatar anima-tab-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}"${ring}>`;
    }
    return `<span class="anima-tab-avatar anima-tab-avatar-initial"${ring}>${initial}</span>`;
  }

  function buildAddConversationAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    const url = state.animaTabAvatarUrls[name];
    const ring = companyRingStyle(name);
    if (url) {
      return `<img class="add-conversation-avatar add-conversation-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}"${ring}>`;
    }
    return `<span class="add-conversation-avatar add-conversation-avatar-initial"${ring}>${initial}</span>`;
  }

  /** Running animas are split into working / waiting for a reply / standby. */
  function _menuStatusLabel(p) {
    const raw = animaStatusLabel(p.status);
    if (p.status !== "running" && p.status !== "idle") return raw;
    const kind = displayStatus(presence.get(p.name));
    if (kind === "waiting") return t("chat.presence.status_waiting");
    if (kind === "working" || kind === "calling") return t("chat.presence.status_working");
    return t("chat.presence.status_idle");
  }

  function renderAddConversationMenu() {
    const menu = $("chatAddConversationMenu");
    if (!menu) return;

    const sorted = [...state.animas].sort((a, b) => {
      const at = Number(state.animaLastAccess[a.name] || 0);
      const bt = Number(state.animaLastAccess[b.name] || 0);
      if (bt !== at) return bt - at;
      return String(a.name || "").localeCompare(String(b.name || ""), "ja");
    });

    let html = "";
    for (const p of sorted) {
      const statusLabel = p.status ? ` (${escapeHtml(_menuStatusLabel(p))})` : "";
      const openLabel = isTabOpen(ctx, p.name) ? t("animas.tab_viewing") : "";
      const disabled = p.status === "bootstrapping" || p.bootstrapping;
      const sleepBadge = p.status === "not_found" || p.status === "stopped" ? "\uD83D\uDCA4 " : "";
      const avatar = buildAddConversationAvatar(p.name);
      if (disabled) {
        html += `<div class="chat-add-conversation-item disabled">${avatar}<span class="chat-add-conversation-name">\u23F3 ${escapeHtml(p.name)}${statusLabel}</span></div>`;
      } else {
        html += `<button type="button" class="chat-add-conversation-item" data-anima="${escapeHtml(p.name)}">${avatar}<span class="chat-add-conversation-name">${sleepBadge}${escapeHtml(p.name)}${statusLabel}${openLabel}</span></button>`;
      }
    }
    menu.innerHTML = html || `<div class="chat-add-conversation-empty">${t("chat.anima_select_first")}</div>`;

    for (const p of sorted) ensureAnimaTabAvatar(p.name);

    menu.querySelectorAll(".chat-add-conversation-item[data-anima]").forEach(el => {
      el.addEventListener("click", e => {
        const name = e.currentTarget?.dataset?.anima;
        if (!name) return;
        openOrSelectAnima(name);
        const area = $("chatAddConversationArea");
        if (area) area.classList.remove("open");
      });
    });
  }

  function renderAnimaTabs() {
    _ensureCallButton();
    const container = $("chatAnimaTabs");
    if (!container) return;
    if (state.animaTabs.length === 0) { container.innerHTML = ""; state.paneHost?.renderRail?.(); return; }

    const html = state.animaTabs.map(tab => {
      const activeClass = tab.name === state.selectedAnima ? " active" : "";
      const streamingClass = state.manager.isStreamingForAnima(tab.name) ? " is-streaming" : "";
      const completedClass = tab.unreadStar ? " has-unread-complete" : "";
      const avatar = buildAnimaTabAvatar(tab.name);
      const closeBtn = state.animaTabs.length > 1
        ? ` <button type="button" class="anima-tab-close" data-anima="${escapeHtml(tab.name)}" title="${escapeHtml(t("animas.tab_close"))}" aria-label="${escapeHtml(t("common.aria_close"))}">&times;</button>`
        : "";
      return `<span class="anima-tab-wrap"><button type="button" class="anima-tab${activeClass}${streamingClass}${completedClass}" data-anima="${escapeHtml(tab.name)}" title="${escapeHtml(tab.name)}" aria-label="${escapeHtml(tab.name)}">${avatar}<span class="anima-tab-name">${escapeHtml(tab.name)}</span></button>${closeBtn}</span>`;
    }).join("");
    container.innerHTML = html;

    for (const tab of state.animaTabs) ensureAnimaTabAvatar(tab.name);

    container.querySelectorAll(".anima-tab").forEach(btn => {
      btn.addEventListener("click", e => {
        const anima = e.currentTarget?.dataset?.anima;
        if (!anima) return;
        if (anima === state.selectedAnima) {
          ctx.controllers.memoryPopover.open(e.currentTarget);
          return;
        }
        openOrSelectAnima(anima);
      });
      _suppressLongPressClick(btn);
      btn.addEventListener("mouseenter", () => _showTabTooltip(btn));
      btn.addEventListener("mouseleave", () => _hideTabTooltip());
      btn.addEventListener("touchstart", () => {
        _showTabTooltip(btn);
        _tooltipHideTimer = setTimeout(_hideTabTooltip, 1500);
      }, { passive: true });
    });
    container.querySelectorAll(".anima-tab-close").forEach(btn => {
      btn.addEventListener("click", e => {
        e.stopPropagation();
        const anima = e.currentTarget?.dataset?.anima;
        if (anima) closeAnimaTab(anima);
      });
    });
    state.paneHost?.renderRail?.();
  }

  async function selectAnima(name) {
    const gen = ++_selectGen;
    const prevAnima = state.selectedAnima;
    const prevThread = state.selectedThreadId || "default";
    const currentInput = $("chatPageInput");
    if (prevAnima && prevAnima !== name) ctx.controllers.events?.clearReplyTarget();
    if (prevAnima && currentInput) {
      saveDraft(prevAnima, currentInput.value || "", prevThread);
      state.activeThreadByAnima[prevAnima] = prevThread;
    }

    state.selectedAnima = name;
    state.animaSummary = null;
    state.animaDetail = null;
    state.animaDetailLoadedFor = null;
    state.animaLastAccess[name] = Date.now();
    state.bustupUrl = null;
    state.selectedThreadId = state.activeThreadByAnima[name] || "default";
    clearUnreadForActiveThread(ctx, name, state.selectedThreadId);
    ctx.controllers.imageVoice.updateVoiceAnima(name);
    ctx.controllers.workIndicator?.onAnimaChange();

    if (!state.threads[name]) {
      state.threads[name] = [{ id: "default", label: t("thread.default_label"), unread: false }];
    }
    if (!state.threads[name].some(th => th.id === state.selectedThreadId)) {
      state.selectedThreadId = "default";
    }
    state.activeThreadByAnima[name] = state.selectedThreadId;
    if (!isTabOpen(ctx, name)) {
      state.animaTabs.push({ name, unreadStar: false });
    }
    refreshAnimaUnread(ctx, name);

    renderAddConversationMenu();
    ensureAnimaTabAvatar(name).catch(() => {});
    renderAnimaTabs();

    const input = $("chatPageInput");
    const sendBtn = $("chatPageSendBtn");
    if (input) { input.disabled = false; input.placeholder = t("chat.message_to", { name }); }
    if (sendBtn) sendBtn.disabled = false;
    if (input) {
      input.value = loadDraft(name, state.selectedThreadId);
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, chatInputMaxHeight()) + "px";
    }
    ctx.controllers.streaming.showPendingIndicator();
    ctx.controllers.streaming.updateSendButton();
    ctx.controllers.streaming.restoreContextRing(name);
    syncModelSelect(ctx);

    const tid = state.selectedThreadId;

    const mgr = state.manager;
    // Start the load before rendering so the wait shows a spinner, not "no messages".
    const convPromise = hasHistoryContent(mgr.getHistoryState(name, tid))
      ? Promise.resolve(null)
      : mgr.loadHistory(name, tid, CONSTANTS.HISTORY_PAGE_SIZE, state.fetchChatFeedHistory);
    ctx.controllers.renderer.renderChat();

    const summaryPromise = api(`/api/animas/${encodeURIComponent(name)}/summary`).catch(() => null);
    void refreshSessions(name);

    const [loadedHs, summary] = await Promise.all([convPromise, summaryPromise]);

    if (gen !== _selectGen) return;

    if (loadedHs) resolveHistoryAvatars(loadedHs.sessions);

    if (!state.threads[name].some(th => th.id === state.selectedThreadId)) {
      state.selectedThreadId = "default";
      state.activeThreadByAnima[name] = "default";
    }

    ctx.controllers.thread.renderThreadTabs();
    ctx.controllers.renderer.renderChat();

    state.animaSummary = summary;
    if (summary) {
      const anima = state.animas.find((entry) => entry.name === name);
      if (anima) {
        anima.role = summary.role;
        anima.model = summary.model;
        anima.enabled = summary.enabled;
        anima.busy = summary.busy;
        anima.status = summary.status?.status || anima.status;
        anima.bootstrapping = summary.status?.bootstrapping ?? anima.bootstrapping;
        anima.needs_bootstrap = summary.status?.needs_bootstrap ?? anima.needs_bootstrap;
        anima.needs_user_input = summary.status?.needs_user_input ?? anima.needs_user_input;
        anima.needs_repair = summary.status?.needs_repair ?? anima.needs_repair;
      }
      const busy = Boolean(summary.busy?.is_busy)
        || ["busy", "running_task"].includes(summary.status?.status);
      presence.setWorking(name, busy, "status", summary.status?.active_label || "");
      ctx.controllers.anima.renderAnimaTabs();
    }
    ctx.controllers.streaming.resumeActiveStream(name);
    scheduleSaveChatUiState(ctx);
    state.paneHost?.savePaneStates?.();
  }

  function openOrSelectAnima(name) {
    if (!name) return;
    if (!isTabOpen(ctx, name)) {
      state.animaTabs.push({ name, unreadStar: false });
      if (!state.threads[name]) {
        state.threads[name] = [{ id: "default", label: t("thread.default_label"), unread: false }];
      }
      state.activeThreadByAnima[name] = state.activeThreadByAnima[name] || "default";
    }
    renderAnimaTabs();
    selectAnima(name);
  }

  function closeAnimaTab(name) {
    if (!name || state.animaTabs.length <= 1) return;
    const idx = state.animaTabs.findIndex(tab => tab.name === name);
    if (idx < 0) return;
    const wasSelected = state.selectedAnima === name;
    state.animaTabs.splice(idx, 1);

    if (wasSelected) {
      const next = state.animaTabs[Math.max(0, idx - 1)];
      if (next) openOrSelectAnima(next.name);
    } else {
      renderAddConversationMenu();
      renderAnimaTabs();
    }
    scheduleSaveChatUiState(ctx);
  }

  function restoreChatUiState(uiState) {
    if (!uiState || typeof uiState !== "object") return;
    const known = new Set((state.animas || []).map(a => a.name));

    state.animaTabs = [];
    state.threads = {};
    state.activeThreadByAnima = {};
    state.animaLastAccess = {};
    state.modelByThread = {};

    const tabs = Array.isArray(uiState.anima_tabs) ? uiState.anima_tabs : [];
    const threadState = uiState.thread_state && typeof uiState.thread_state === "object" ? uiState.thread_state : {};
    const accessState = uiState.anima_last_access && typeof uiState.anima_last_access === "object" ? uiState.anima_last_access : {};

    for (const tab of tabs) {
      const name = tab?.name;
      if (!name || !known.has(name)) continue;
      state.animaTabs.push({ name, unreadStar: Boolean(tab.unread_star) });
    }

    for (const tab of state.animaTabs) {
      const name = tab.name;
      const persisted = threadState[name] || {};
      const list = Array.isArray(persisted.threads) ? persisted.threads : [];
      const normalized = list
        .filter(th => th && typeof th.id === "string")
        .map(th => {
          const o = {
            id: th.id,
            label: typeof th.label === "string" && th.label ? th.label : t("thread.new"),
            unread: Boolean(th.unread),
          };
          if (th.archived) o.archived = true;
          if (typeof th.model === "string" && th.model) {
            state.modelByThread[`${name}|${th.id}`] = th.model;
          }
          return o;
        });
      if (!normalized.some(th => th.id === "default")) {
        normalized.unshift({ id: "default", label: t("thread.default_label"), unread: false });
      }
      state.threads[name] = normalized;
      state.activeThreadByAnima[name] = persisted.active_thread_id || "default";
      refreshAnimaUnread(ctx, name);
    }

    for (const [name, ts] of Object.entries(accessState)) {
      if (!known.has(name)) continue;
      const value = Number(ts);
      if (Number.isFinite(value) && value > 0) state.animaLastAccess[name] = value;
    }

    const active = uiState.active_anima;
    if (typeof active === "string" && isTabOpen(ctx, active)) {
      state.selectedAnima = active;
      state.selectedThreadId = state.activeThreadByAnima[active] || "default";
    }
  }

  async function loadAnimas() {
    try {
      const [animas, uiState, users] = await Promise.all([
        api("/api/animas"),
        fetchChatUiState(ctx),
        api("/api/users").catch(() => []),
      ]);
      state.animas = animas || [];
      state.users = Array.isArray(users) ? users : [];
      restoreChatUiState(uiState);
      renderAddConversationMenu();
      renderAnimaTabs();

      const paneState = state.paneHost?.getPaneState?.(state.paneIdx);
      const known = new Set(state.animas.map(a => a.name));
      if (paneState?.anima && known.has(paneState.anima)) {
        state.selectedAnima = null;
        if (!isTabOpen(ctx, paneState.anima)) {
          state.animaTabs.push({ name: paneState.anima, unreadStar: false });
        }
        state.activeThreadByAnima[paneState.anima] = paneState.threadId || "default";
        renderAnimaTabs();
        openOrSelectAnima(paneState.anima);
      } else if (state.animas.length > 0 && !state.selectedAnima) {
        const firstTab = state.animaTabs[0]?.name;
        // With nothing remembered, start with the organization's leader.
        const leader = state.animas.find(a => !a.supervisor && a.enabled !== false);
        openOrSelectAnima(firstTab || leader?.name || state.animas[0].name);
      } else if (state.selectedAnima) {
        selectAnima(state.selectedAnima);
      }

      // A toast "reply" asked for a conversation before this page existed.
      const pending = presence.takePendingOpen();
      if (pending && known.has(pending)) openOrSelectAnima(pending);
    } catch (err) {
      logger.error("Failed to load animas", err);
    }
  }

  function resolveHistoryAvatars(sessions) {
    if (!sessions || sessions.length === 0) return;
    const known = new Set(Object.keys(state.animaTabAvatarUrls));
    const knownAnimas = new Set((state.animas || []).map(a => a.name).filter(Boolean));
    const names = new Set();
    for (const s of sessions) {
      if (!s.messages) continue;
      for (const m of s.messages) {
        if (m.from_person && knownAnimas.has(m.from_person) && !known.has(m.from_person)) {
          names.add(m.from_person);
        }
      }
    }
    for (const n of names) ensureAnimaTabAvatar(n);
  }

  return {
    loadAnimas, restoreChatUiState, selectAnima, openOrSelectAnima,
    closeAnimaTab, renderAnimaTabs, renderAddConversationMenu,
    ensureAnimaTabAvatar, resolveHistoryAvatars, refreshSessions,
  };
}
