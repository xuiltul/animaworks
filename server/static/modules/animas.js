/* ── Anima Dropdown, Selection, Avatar ────── */

import { state, dom, escapeHtml } from "./state.js";
import { api } from "./api.js";
import { renderChat, resumeActiveStream } from "./chat.js";
import { loadMemoryTab } from "./memory.js";
import { hideHistoryDetail, loadSessionList } from "./history.js";

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

  let html = '<option value="" disabled>Animaを選択...</option>';
  for (const p of state.animas) {
    const selected = p.name === state.selectedAnima ? " selected" : "";
    if (p.status === "bootstrapping" || p.bootstrapping) {
      html += `<option value="${escapeHtml(p.name)}"${selected} disabled>\u23F3 ${escapeHtml(p.name)} (制作中...)</option>`;
    } else if (p.status === "not_found" || p.status === "stopped") {
      html += `<option value="${escapeHtml(p.name)}"${selected}>\uD83D\uDCA4 ${escapeHtml(p.name)} (停止中)</option>`;
    } else {
      const statusLabel = p.status ? ` (${p.status})` : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${statusLabel}</option>`;
    }
  }
  dropdown.innerHTML = html;
}

export async function selectAnima(name) {
  // Check if anima is sleeping — offer to start
  const anima = state.animas.find((p) => p.name === name);
  if (anima && (anima.status === "not_found" || anima.status === "stopped")) {
    const ok = confirm(`${name} は現在停止中です。起動しますか？`);
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

  // Update dropdown
  const dropdown = dom.animaDropdown || document.getElementById("animaDropdown");
  if (dropdown) dropdown.value = name;

  // Enable chat
  const chatInput = dom.chatInput || document.getElementById("chatInput");
  const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (chatInput) {
    chatInput.disabled = false;
    chatInput.placeholder = `${name} にメッセージ...`;
  }
  if (chatSendBtn) chatSendBtn.disabled = false;

  // Load conversation history + anima detail in parallel
  const needConv = !state.chatHistories[name] || state.chatHistories[name].length === 0;
  const convPromise = needConv
    ? api(`/api/animas/${encodeURIComponent(name)}/conversation/full?limit=20`).catch(() => null)
    : Promise.resolve(null);
  const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);

  const [conv, detail] = await Promise.all([convPromise, detailPromise]);

  // Apply conversation history
  if (conv && conv.turns && conv.turns.length > 0) {
    state.chatHistories[name] = conv.turns.map((t) => ({
      role: t.role === "human" ? "user" : "assistant",
      text: t.content,
    }));
  }

  // Render chat history
  renderChat();

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
    if (stateContent) stateContent.textContent = "詳細の読み込み失敗";
    const memoryList = dom.memoryFileList || document.getElementById("memoryFileList");
    if (memoryList) memoryList.innerHTML = '<div class="loading-placeholder">詳細の読み込み失敗</div>';
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

export async function updateAnimaAvatar() {
  const container = dom.animaAvatar || document.getElementById("animaAvatar");
  if (!container) return;

  const name = state.selectedAnima;
  if (!name) {
    container.innerHTML = "";
    return;
  }

  // Try bust-up first, then chibi
  const candidates = ["avatar_bustup.png", "avatar_chibi.png"];
  for (const filename of candidates) {
    const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="anima-avatar-img">`;
        return;
      }
    } catch { /* try next */ }
  }

  // Fallback: initial letter
  container.innerHTML = `<div class="anima-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
}

// ── Anima State ───────────────────────────

export function renderAnimaState() {
  const stateContent = dom.animaStateContent || document.getElementById("animaStateContent");
  if (!stateContent) return; // State panel not in DOM

  const d = state.animaDetail;
  if (!d || !d.state) {
    stateContent.textContent = "状態情報なし";
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
