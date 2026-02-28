// ── Anima Selection / Tab / Avatar Controller ──
import {
  $, isTabOpen, isBusinessTheme, refreshAnimaUnread,
  clearUnreadForActiveThread, loadDraft, saveDraft, chatInputMaxHeight,
  fetchChatUiState, scheduleSaveChatUiState, mergeThreadsFromSessions,
} from "./ctx.js";

export function createAnimaController(ctx) {
  const { state, deps } = ctx;
  const { api, escapeHtml, t, logger } = deps;
  let _selectGen = 0;

  function ensureAnimaTabAvatar(name) {
    if (!name || isBusinessTheme()) return Promise.resolve();
    if (Object.prototype.hasOwnProperty.call(state.animaTabAvatarUrls, name)) return Promise.resolve();
    if (state.animaTabAvatarLoading[name]) return state.animaTabAvatarLoading[name];

    state.animaTabAvatarLoading[name] = (async () => {
      let found = null;
      const candidates = ["avatar_bustup.png"];
      for (const filename of candidates) {
        const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
        try {
          const resp = await fetch(url, { method: "HEAD" });
          if (resp.ok) { found = url; break; }
        } catch { /* next */ }
      }
      state.animaTabAvatarUrls[name] = found;
      delete state.animaTabAvatarLoading[name];
      renderAnimaTabs();
      renderAddConversationMenu();
    })();
    return state.animaTabAvatarLoading[name];
  }

  function buildAnimaTabAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    if (isBusinessTheme()) {
      return `<span class="anima-tab-avatar anima-tab-avatar-initial">${initial}</span>`;
    }
    const url = state.animaTabAvatarUrls[name];
    if (url) {
      return `<img class="anima-tab-avatar anima-tab-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
    }
    return `<span class="anima-tab-avatar anima-tab-avatar-initial">${initial}</span>`;
  }

  function buildAddConversationAvatar(name) {
    const initial = escapeHtml((name || "").charAt(0).toUpperCase() || "?");
    if (isBusinessTheme()) {
      return `<span class="add-conversation-avatar add-conversation-avatar-initial">${initial}</span>`;
    }
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
      const streamingClass = state.activeStreams[tab.name] ? " is-streaming" : "";
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
    const currentInput = $("chatPageInput");
    if (prevAnima && currentInput) {
      saveDraft(prevAnima, currentInput.value || "");
      state.activeThreadByAnima[prevAnima] = state.selectedThreadId || "default";
    }

    state.selectedAnima = name;
    state.animaLastAccess[name] = Date.now();
    state.bustupUrl = null;
    state.pendingQueue = [];
    ctx.controllers.streaming.hidePendingIndicator();
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
      input.value = loadDraft(name);
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, chatInputMaxHeight()) + "px";
    }
    ctx.controllers.streaming.updateSendButton();

    const tid = state.selectedThreadId;

    ctx.controllers.renderer.renderChat();

    const needConv = !state.historyState[name]?.[tid] || state.historyState[name][tid].sessions.length === 0;
    const convPromise = needConv
      ? ctx.controllers.renderer.fetchConversationHistory(name, 50, null, tid).catch(() => null)
      : Promise.resolve(null);
    const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);
    const sessionsPromise = api(`/api/animas/${encodeURIComponent(name)}/sessions`).catch(() => null);

    const [conv, detail, sessionsData] = await Promise.all([convPromise, detailPromise, sessionsPromise]);

    if (gen !== _selectGen) return;

    if (!state.historyState[name]) state.historyState[name] = {};
    if (conv && conv.sessions && conv.sessions.length > 0) {
      state.historyState[name][tid] = {
        sessions: conv.sessions, hasMore: conv.has_more || false,
        nextBefore: conv.next_before || null, loading: false,
      };
    } else if (needConv) {
      state.historyState[name][tid] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
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
        .map(th => ({
          id: th.id,
          label: typeof th.label === "string" && th.label ? th.label : "新しいスレッド",
          unread: Boolean(th.unread),
        }));
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
      if (state.animas.length > 0 && !state.selectedAnima && Object.keys(state.activeStreams).length === 0) {
        const firstTab = state.animaTabs[0]?.name;
        openOrSelectAnima(firstTab || state.animas[0].name);
      } else if (state.selectedAnima) {
        selectAnima(state.selectedAnima);
      }
    } catch (err) {
      logger.error("Failed to load animas", err);
    }
  }

  return {
    loadAnimas, restoreChatUiState, selectAnima, openOrSelectAnima,
    closeAnimaTab, renderAnimaTabs, renderAddConversationMenu,
    ensureAnimaTabAvatar,
  };
}
