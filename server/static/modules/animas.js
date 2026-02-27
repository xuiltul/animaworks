/* ── Anima Dropdown, Selection, Avatar ────── */

import { state, dom, escapeHtml } from "./state.js";
import { t } from "/shared/i18n.js";
import { api } from "./api.js";
import { renderChat, resumeActiveStream, loadConversationHistory, setupScrollObserver, updateVoiceAnima } from "./chat.js";
import { loadMemoryTab } from "./memory.js";
import { hideHistoryDetail, loadSessionList } from "./history.js";
import { animaHashColor } from "../shared/avatar-utils.js";

export async function loadAnimas() {
  try {
    state.animas = await api("/api/animas");
    renderAnimaDropdown();
    if (state.animas.length > 0 && !state.selectedAnima) {
      selectAnima(state.animas[0].name);
    }
  } catch (err) {
    console.error("Failed to load animas:", err);
  }
}

// ── Anima State Class Helper ──────────────────
function statusIndicator(emoji, lucideIcon) {
  const isBusiness = document.body.classList.contains('theme-business');
  if (isBusiness) {
    return `<i data-lucide="${lucideIcon}" style="width:14px;height:14px;display:inline-block;vertical-align:middle;"></i>`;
  }
  return emoji;
}

export function animaStateClass(anima) {
  if (anima.status === "bootstrapping" || anima.bootstrapping) {
    return "anima-item--loading";
  }
  if (anima.status === "not_found" || anima.status === "stopped") {
    return "anima-item--sleeping";
  }
  return "";
}

export function renderAnimaDropdown() {
  const dropdown = dom.animaDropdown || document.getElementById("animaDropdown");
  if (!dropdown) return; // Anima dropdown not in DOM (page not active)

  let html = `<option value="" disabled>${t("chat.anima_select")}</option>`;
  for (const p of state.animas) {
    const selected = p.name === state.selectedAnima ? " selected" : "";
    if (p.status === "bootstrapping" || p.bootstrapping) {
      html += `<option value="${escapeHtml(p.name)}"${selected} disabled>${statusIndicator('\u23F3', 'loader')} ${escapeHtml(p.name)} (${t("animas.bootstrapping")})</option>`;
    } else if (p.status === "not_found" || p.status === "stopped") {
      html += `<option value="${escapeHtml(p.name)}"${selected}>${statusIndicator('\uD83D\uDCA4', 'moon')} ${escapeHtml(p.name)} (${t("animas.stopped")})</option>`;
    } else {
      const statusLabel = p.status ? ` (${p.status})` : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${statusLabel}</option>`;
    }
  }
  dropdown.innerHTML = html;
  if (window.lucide) lucide.createIcons();
}

export async function selectAnima(name) {
  // Check if anima is sleeping — offer to start
  const anima = state.animas.find((p) => p.name === name);
  if (anima && (anima.status === "not_found" || anima.status === "stopped")) {
    const ok = confirm(t("animas.start_confirm", { name }));
    if (!ok) return;
    try {
      await api(`/api/animas/${encodeURIComponent(name)}/start`, { method: "POST" });
    } catch (err) {
      console.error("Failed to start anima:", err);
    }
    // Status update will come via WebSocket
    return;
  }

  state.selectedAnima = name;
  updateVoiceAnima(name);

  // Update dropdown
  const dropdown = dom.animaDropdown || document.getElementById("animaDropdown");
  if (dropdown) dropdown.value = name;

  // Enable chat
  const chatInput = dom.chatInput || document.getElementById("chatInput");
  const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (chatInput) {
    chatInput.disabled = false;
    chatInput.placeholder = t("chat.message_to", { name });
  }
  if (chatSendBtn) chatSendBtn.disabled = false;

  // Load conversation history (activity_log API) + anima detail in parallel
  const convPromise = loadConversationHistory(name);
  const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);

  const [, detail] = await Promise.all([convPromise, detailPromise]);

  // Setup infinite scroll observer for chat history
  setupScrollObserver();

  // Check if anima is currently processing — resume stream
  const selectedAnimaObj = state.animas.find((p) => p.name === name);
  if (selectedAnimaObj && (selectedAnimaObj.status === "thinking" || selectedAnimaObj.status === "processing")) {
    resumeActiveStream(name);
  }

  // Apply anima detail
  if (detail) {
    state.animaDetail = detail;
    renderAnimaState();
  } else {
    state.animaDetail = null;
    const stateContent = dom.animaStateContent || document.getElementById("animaStateContent");
    if (stateContent) stateContent.textContent = t("animas.detail_load_failed");
    const memoryList = dom.memoryFileList || document.getElementById("memoryFileList");
    if (memoryList) memoryList.innerHTML = `<div class="loading-placeholder">${t("animas.detail_load_failed")}</div>`;
  }

  // Load memory, session list, and avatar in parallel
  const secondaryPromises = [loadMemoryTab(state.activeMemoryTab), updateAnimaAvatar()];
  if (state.activeRightTab === "history") {
    hideHistoryDetail();
    secondaryPromises.push(loadSessionList());
  }
  await Promise.all(secondaryPromises);
}

// ── Anima Avatar ───────────────────────────

// animaHashColor is now imported from shared/avatar-utils.js
export { animaHashColor };

export async function updateAnimaAvatar() {
  const container = dom.animaAvatar || document.getElementById("animaAvatar");
  if (!container) return;

  const name = state.selectedAnima;
  if (!name) {
    container.innerHTML = "";
    return;
  }

  const initial = escapeHtml(name.charAt(0).toUpperCase());
  const color = animaHashColor(name);

  // Try bust-up first, then chibi
  let imgHtml = "";
  const candidates = ["avatar_bustup.png", "avatar_chibi.png"];
  for (const filename of candidates) {
    const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        imgHtml = `<img class="anima-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
        break;
      }
    } catch { /* try next */ }
  }

  // Always render both: img and initial div. CSS variables control visibility per theme.
  const initialDiv = `<div class="anima-avatar-initial" style="background: ${color}; width:36px; height:36px;">${initial}</div>`;
  container.innerHTML = imgHtml + initialDiv;

  if (window.lucide) lucide.createIcons();
}

// ── Anima State ───────────────────────────

export function renderAnimaState() {
  const stateContent = dom.animaStateContent || document.getElementById("animaStateContent");
  if (!stateContent) return; // State panel not in DOM

  const d = state.animaDetail;
  if (!d || !d.state) {
    stateContent.textContent = t("animas.no_state");
    return;
  }
  const stateText = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
  stateContent.textContent = stateText;
}

export async function refreshSelectedAnima() {
  if (!state.selectedAnima) return;
  try {
    state.animaDetail = await api(`/api/animas/${encodeURIComponent(state.selectedAnima)}`);
    renderAnimaState();
  } catch {
    // Silently ignore refresh errors
  }
}
