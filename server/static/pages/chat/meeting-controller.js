// ── Meeting Mode Controller ──────────────────────
import { t } from "../../shared/i18n.js";

export function createMeetingController(ctx) {
  const { state, deps } = ctx;
  const $ = ctx.$;
  const { api, escapeHtml } = deps;

  // ── State (mutate ctx.state directly) ──────────
  function init() {
    state.meetingMode = false;
    state.meetingRoom = null;
    state.meetingParticipants = [];
    state.meetingChair = null;
    _bindEvents();
  }

  function _bindEvents() {
    const toggle = $("meetingModeToggle");
    if (toggle) {
      toggle.addEventListener("click", () => toggleMeetingMode());
    }
  }

  function toggleMeetingMode() {
    state.meetingMode = !state.meetingMode;
    if (!state.meetingMode) {
      state.meetingRoom = null;
      state.meetingParticipants = [];
    }
    _updateToggleUI();
    _updateMeetingPanel();
    _updateAnimaTabsVisibility();
    ctx.controllers.renderer?.renderChat();
  }

  function _updateToggleUI() {
    const toggle = $("meetingModeToggle");
    if (toggle) {
      toggle.classList.toggle("active", state.meetingMode);
      toggle.title = state.meetingMode ? t("meeting.toggle_active") : t("meeting.toggle");
    }
  }

  function _updateMeetingPanel() {
    const panel = $("meetingParticipantPanel");
    if (!panel) return;

    if (!state.meetingMode) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "flex";

    if (state.meetingRoom) {
      _renderActiveRoomPanel(panel);
    } else {
      _renderSetupPanel(panel);
    }
  }

  function _updateAnimaTabsVisibility() {
    const tabsContainer = $("chatAnimaTabs");
    const addBtn = $("chatAddConversationArea");
    if (tabsContainer) {
      tabsContainer.style.display = state.meetingMode ? "none" : "";
    }
    if (addBtn) {
      addBtn.style.display = state.meetingMode ? "none" : "";
    }
  }

  function _avatarHtml(name) {
    const url = state.animaTabAvatarUrls?.[name];
    const initial = escapeHtml((name || "?").charAt(0).toUpperCase());
    if (url) {
      return `<img class="chip-avatar chip-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
    }
    return `<span class="chip-avatar chip-avatar-initial">${initial}</span>`;
  }

  function _renderSetupPanel(panel) {
    const animas = state.animas.filter(
      (a) => a.status === "running" || a.status === "idle"
    );
    const selected = new Set(state.meetingParticipants);
    const chair = state.meetingChair || null;

    let animaListHtml = "";
    for (const a of animas) {
      const isSelected = selected.has(a.name);
      const isChair = chair === a.name;
      animaListHtml += `
        <label class="meeting-setup-anima ${isSelected ? "selected" : ""}" data-anima="${escapeHtml(a.name)}">
          <input type="checkbox" ${isSelected ? "checked" : ""} data-anima="${escapeHtml(a.name)}">
          <input type="radio" name="meeting-chair" value="${escapeHtml(a.name)}" ${isChair ? "checked" : ""} ${!isSelected ? "disabled" : ""}>
          ${_avatarHtml(a.name)}
          <span>${escapeHtml(a.name)}</span>
          ${isChair ? " 👑" : ""}
        </label>`;
    }

    for (const a of animas) {
      ctx.controllers.anima?.ensureAnimaTabAvatar?.(a.name)?.then(() => {
        const avatarEl = panel.querySelector(`.meeting-setup-anima[data-anima="${a.name}"] .chip-avatar`);
        if (avatarEl) {
          const url = state.animaTabAvatarUrls?.[a.name];
          if (url) avatarEl.outerHTML = `<img class="chip-avatar chip-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(a.name)}">`;
        }
      });
    }

    panel.innerHTML = `
      <div class="meeting-setup">
        <div class="meeting-setup-label">${t("meeting.select_participants")}</div>
        <div class="meeting-setup-anima-list">${animaListHtml || t("meeting.no_animas")}</div>
        <div class="meeting-setup-actions">
          <button type="button" class="meeting-start-btn" data-chat-id="meetingStartBtn" disabled>
            ${t("meeting.start")}
          </button>
        </div>
      </div>`;

    panel.querySelectorAll(".meeting-setup-anima").forEach((el) => {
      el.addEventListener("click", (e) => {
        const anima = el.dataset.anima;
        if (!anima) return;
        const checkbox = el.querySelector('input[type="checkbox"]');
        if (e.target === checkbox) return;
        const wasSelected = selected.has(anima);
        if (wasSelected) {
          selected.delete(anima);
          if (chair === anima) state.meetingChair = null;
        } else {
          selected.add(anima);
          if (!chair && selected.size > 0) state.meetingChair = anima;
        }
        state.meetingParticipants = [...selected];
        _updateMeetingPanel();
      });
    });

    panel.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
      cb.addEventListener("change", (e) => {
        e.stopPropagation();
        const anima = cb.dataset.anima;
        if (!anima) return;
        if (cb.checked) {
          selected.add(anima);
          if (!chair) state.meetingChair = anima;
        } else {
          selected.delete(anima);
          if (chair === anima) state.meetingChair = null;
        }
        state.meetingParticipants = [...selected];
        panel.querySelectorAll('input[type="radio"]').forEach((r) => {
          r.disabled = !selected.has(r.value);
          if (r.value === chair && !selected.has(r.value)) state.meetingChair = null;
        });
        _updateMeetingPanel();
      });
    });

    panel.querySelectorAll('input[type="radio"]').forEach((r) => {
      r.addEventListener("change", (e) => {
        state.meetingChair = e.target.value;
        _updateMeetingPanel();
      });
    });

    const startBtn = panel.querySelector('[data-chat-id="meetingStartBtn"]');
    if (startBtn) {
      startBtn.disabled = selected.size < 2 || !chair;
      startBtn.addEventListener("click", () => createRoom());
    }
  }

  function _renderActiveRoomPanel(panel) {
    const room = state.meetingRoom;
    if (!room) return;

    const participants = room.participants || [];
    const chair = room.chair || "";

    let chipsHtml = participants
      .map((p) => {
        const name = typeof p === "string" ? p : p.name || p;
        const isChair = name === chair;
        return `
          <span class="meeting-participant-chip ${isChair ? "is-chair" : ""}" data-name="${escapeHtml(name)}">
            ${_avatarHtml(name)}
            <span>${escapeHtml(name)}</span>
            ${isChair ? " 👑" : ""}
            ${!isChair ? `<button type="button" class="chip-remove" data-name="${escapeHtml(name)}" title="${t("meeting.remove")}">✕</button>` : ""}
          </span>`;
      })
      .join("");

    panel.innerHTML = `
      ${chipsHtml}
      <button type="button" class="meeting-add-btn" data-chat-id="meetingAddBtn">${t("meeting.add")} +</button>
      <button type="button" class="meeting-end-btn" data-chat-id="meetingEndBtn">${t("meeting.end")}</button>`;

    panel.querySelectorAll(".chip-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const name = btn.dataset.name;
        if (name) removeParticipant(name);
      });
    });

    const addBtn = panel.querySelector('[data-chat-id="meetingAddBtn"]');
    if (addBtn) {
      addBtn.addEventListener("click", () => _showAddParticipantMenu(panel));
    }

    const endBtn = panel.querySelector('[data-chat-id="meetingEndBtn"]');
    if (endBtn) {
      endBtn.addEventListener("click", () => endMeeting());
    }
  }

  function _showAddParticipantMenu(panel) {
    const room = state.meetingRoom;
    if (!room) return;

    const current = new Set((room.participants || []).map((p) => (typeof p === "string" ? p : p.name || p)));
    const available = state.animas.filter(
      (a) =>
        (a.status === "running" || a.status === "idle") && !current.has(a.name)
    );

    if (available.length === 0) {
      return;
    }

    const menu = document.createElement("div");
    menu.className = "meeting-add-menu";
    menu.innerHTML = available
      .map(
        (a) =>
          `<button type="button" class="meeting-add-item" data-name="${escapeHtml(a.name)}">${_avatarHtml(a.name)} ${escapeHtml(a.name)}</button>`
      )
      .join("");

    menu.querySelectorAll(".meeting-add-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        addParticipant(btn.dataset.name);
        menu.remove();
      });
    });

    const addBtn = panel.querySelector('[data-chat-id="meetingAddBtn"]');
    if (addBtn) {
      const existing = panel.querySelector(".meeting-add-menu");
      if (existing) existing.remove();
      addBtn.after(menu);
    }
  }

  async function createRoom() {
    const participants = [...state.meetingParticipants];
    const chair = state.meetingChair;
    if (participants.length < 2 || !chair) return;

    try {
      const res = await api("/api/rooms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          participants,
          chair,
          title: state.meetingRoomTitle || t("meeting.default_title"),
        }),
      });
      state.meetingRoom = res;
      _updateMeetingPanel();
      const input = $("chatPageInput");
      if (input) input.placeholder = t("meeting.placeholder");
      ctx.controllers.streaming?.updateSendButton?.();
      ctx.controllers.renderer?.renderChat();
    } catch (err) {
      deps.logger?.error?.("Failed to create meeting room", err);
    }
  }

  async function _refetchRoom(roomId) {
    try {
      const room = await api(`/api/rooms/${encodeURIComponent(roomId)}`);
      state.meetingRoom = room;
    } catch (err) {
      deps.logger?.error?.("Failed to refetch room", err);
    }
  }

  async function addParticipant(name) {
    const room = state.meetingRoom;
    if (!room?.room_id) return;

    try {
      await api(`/api/rooms/${encodeURIComponent(room.room_id)}/participants`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      await _refetchRoom(room.room_id);
      _updateMeetingPanel();
    } catch (err) {
      deps.logger?.error?.("Failed to add participant", err);
    }
  }

  async function removeParticipant(name) {
    const room = state.meetingRoom;
    if (!room?.room_id) return;

    try {
      await api(
        `/api/rooms/${encodeURIComponent(room.room_id)}/participants/${encodeURIComponent(name)}`,
        { method: "DELETE" }
      );
      await _refetchRoom(room.room_id);
      _updateMeetingPanel();
    } catch (err) {
      deps.logger?.error?.("Failed to remove participant", err);
    }
  }

  async function endMeeting() {
    const room = state.meetingRoom;
    if (!room?.room_id) return;

    try {
      await api(`/api/rooms/${encodeURIComponent(room.room_id)}/close`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    } catch (err) {
      deps.logger?.error?.("Failed to close meeting", err);
    }
    state.meetingRoom = null;
    state.meetingMode = false;
    _updateToggleUI();
    _updateMeetingPanel();
    _updateAnimaTabsVisibility();
    ctx.controllers.renderer?.renderChat();
    ctx.controllers.streaming?.updateSendButton?.();
  }

  return {
    init,
    toggleMeetingMode,
    createRoom,
    addParticipant,
    removeParticipant,
    endMeeting,
    isActive: () => Boolean(state.meetingMode && state.meetingRoom != null),
    getRoom: () => state.meetingRoom,
    updatePanel: _updateMeetingPanel,
  };
}
