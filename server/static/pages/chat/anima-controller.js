// ── Anima Selection / Tab / Avatar Controller ──
import {
  isTabOpen, refreshAnimaUnread,
  clearUnreadForActiveThread, loadDraft, saveDraft, chatInputMaxHeight,
  fetchChatUiState, scheduleSaveChatUiState, mergeThreadsFromSessions,
} from "./ctx.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";

export function createAnimaController(ctx) {
  const $ = ctx.$;
  const { state, deps } = ctx;
  const { api, escapeHtml, t, logger } = deps;
  let _selectGen = 0;
  let _tooltipEl = null;
  let _tooltipHideTimer = null;

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
      const found = await resolveCachedAvatar(name, bustupCandidates(), "S");
      state.animaTabAvatarUrls[name] = found;
      delete state.animaTabAvatarLoading[name];
      renderAnimaTabs();
      renderAddConversationMenu();
    })();
    return state.animaTabAvatarLoading[name];
  }

  function buildAnimaTabAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    const url = state.animaTabAvatarUrls[name];
    if (url) {
      return `<img class="anima-tab-avatar anima-tab-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
    }
    return `<span class="anima-tab-avatar anima-tab-avatar-initial">${initial}</span>`;
  }

  function buildAddConversationAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    const url = state.animaTabAvatarUrls[name];
    if (url) {
      return `<img class="add-conversation-avatar add-conversation-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
    }
    return `<span class="add-conversation-avatar add-conversation-avatar-initial">${initial}</span>`;
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
      const statusLabel = p.status ? ` (${p.status})` : "";
      const openLabel = isTabOpen(ctx, p.name) ? " · 表示中" : "";
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
    const container = $("chatAnimaTabs");
    if (!container) return;
    if (state.animaTabs.length === 0) { container.innerHTML = ""; return; }

    const html = state.animaTabs.map(tab => {
      const activeClass = tab.name === state.selectedAnima ? " active" : "";
      const streamingClass = state.manager.isStreamingForAnima(tab.name) ? " is-streaming" : "";
      const completedClass = tab.unreadStar ? " has-unread-complete" : "";
      const avatar = buildAnimaTabAvatar(tab.name);
      const closeBtn = state.animaTabs.length > 1
        ? ` <button type="button" class="anima-tab-close" data-anima="${escapeHtml(tab.name)}" title="タブを閉じる" aria-label="閉じる">&times;</button>`
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
          ctx.controllers.avatar.showBustupOverlay();
          return;
        }
        openOrSelectAnima(anima);
      });
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
  }

  async function selectAnima(name) {
    const gen = ++_selectGen;
    const prevAnima = state.selectedAnima;
    const prevThread = state.selectedThreadId || "default";
    const currentInput = $("chatPageInput");
    if (prevAnima && currentInput) {
      saveDraft(prevAnima, currentInput.value || "", prevThread);
      state.activeThreadByAnima[prevAnima] = prevThread;
    }

    state.selectedAnima = name;
    state.animaLastAccess[name] = Date.now();
    state.bustupUrl = null;
    state.selectedThreadId = state.activeThreadByAnima[name] || "default";
    clearUnreadForActiveThread(ctx, name, state.selectedThreadId);
    ctx.controllers.imageVoice.updateVoiceAnima(name);

    if (!state.threads[name]) {
      state.threads[name] = [{ id: "default", label: "メイン", unread: false }];
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

    const tid = state.selectedThreadId;

    ctx.controllers.renderer.renderChat();

    const mgr = state.manager;
    const existingHs = mgr.getHistoryState(name, tid);
    const needConv = !existingHs || existingHs.sessions.length === 0;
    const convPromise = needConv
      ? ctx.controllers.renderer.fetchConversationHistory(name, 50, null, tid).catch(() => null)
      : Promise.resolve(null);
    const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);
    const sessionsPromise = api(`/api/animas/${encodeURIComponent(name)}/sessions`).catch(() => null);

    const [conv, detail, sessionsData] = await Promise.all([convPromise, detailPromise, sessionsPromise]);

    if (gen !== _selectGen) return;

    if (conv && conv.sessions && conv.sessions.length > 0) {
      const { createHistoryState, applyHistoryData } = await import("../../shared/chat/session-manager.js");
      const hs = createHistoryState();
      applyHistoryData(hs, conv);
      mgr.setHistoryState(name, tid, hs);
      resolveHistoryAvatars(conv.sessions);
    } else if (needConv) {
      const { createHistoryState } = await import("../../shared/chat/session-manager.js");
      mgr.setHistoryState(name, tid, createHistoryState());
    }

    if (sessionsData) mergeThreadsFromSessions(ctx, name, sessionsData);
    if (!state.threads[name].some(th => th.id === state.selectedThreadId)) {
      state.selectedThreadId = "default";
      state.activeThreadByAnima[name] = "default";
    }

    ctx.controllers.thread.renderThreadTabs();
    ctx.controllers.renderer.renderChat();

    if (detail) {
      state.animaDetail = detail;
      ctx.controllers.activity.renderAnimaState();
    } else {
      state.animaDetail = null;
      const stateEl = $("chatAnimaState");
      if (stateEl) stateEl.textContent = t("animas.detail_load_failed");
    }

    const secondary = [ctx.controllers.memory.loadMemoryTab(), ctx.controllers.activity.loadActivity()];
    if (state.activeRightTab === "history") secondary.push(ctx.controllers.history.loadSessionList());
    await Promise.all(secondary);

    ctx.controllers.streaming.resumeActiveStream(name);
    scheduleSaveChatUiState(ctx);
    state.paneHost?.savePaneStates?.();
  }

  function openOrSelectAnima(name) {
    if (!name) return;
    if (!isTabOpen(ctx, name)) {
      state.animaTabs.push({ name, unreadStar: false });
      if (!state.threads[name]) {
        state.threads[name] = [{ id: "default", label: "メイン", unread: false }];
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
            label: typeof th.label === "string" && th.label ? th.label : "新しいスレッド",
            unread: Boolean(th.unread),
          };
          if (th.archived) o.archived = true;
          return o;
        });
      if (!normalized.some(th => th.id === "default")) {
        normalized.unshift({ id: "default", label: "メイン", unread: false });
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
      const [animas, uiState] = await Promise.all([api("/api/animas"), fetchChatUiState(ctx)]);
      state.animas = animas || [];
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
        openOrSelectAnima(firstTab || state.animas[0].name);
      } else if (state.selectedAnima) {
        selectAnima(state.selectedAnima);
      }
    } catch (err) {
      logger.error("Failed to load animas", err);
    }
  }

  function resolveHistoryAvatars(sessions) {
    if (!sessions || sessions.length === 0) return;
    const known = new Set(Object.keys(state.animaTabAvatarUrls));
    const names = new Set();
    for (const s of sessions) {
      if (!s.messages) continue;
      for (const m of s.messages) {
        if (m.from_person && m.from_person !== "human" && !known.has(m.from_person)) {
          names.add(m.from_person);
        }
      }
    }
    for (const n of names) ensureAnimaTabAvatar(n);
  }

  return {
    loadAnimas, restoreChatUiState, selectAnima, openOrSelectAnima,
    closeAnimaTab, renderAnimaTabs, renderAddConversationMenu,
    ensureAnimaTabAvatar, resolveHistoryAvatars,
  };
}
