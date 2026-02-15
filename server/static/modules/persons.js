/* ── Person Dropdown, Selection, Avatar ────── */

import { state, dom, escapeHtml } from "./state.js";
import { api } from "./api.js";
import { renderChat } from "./chat.js";
import { loadMemoryTab } from "./memory.js";
import { hideHistoryDetail, loadSessionList } from "./history.js";

export async function loadPersons() {
  try {
    state.persons = await api("/api/persons");
    renderPersonDropdown();
    if (state.persons.length > 0 && !state.selectedPerson) {
      selectPerson(state.persons[0].name);
    }
  } catch (err) {
    console.error("Failed to load persons:", err);
  }
}

// ── Person State Class Helper ──────────────────
export function personStateClass(person) {
  if (person.status === "bootstrapping" || person.bootstrapping) {
    return "person-item--loading";
  }
  if (person.status === "not_found" || person.status === "stopped") {
    return "person-item--sleeping";
  }
  return "";
}

export function renderPersonDropdown() {
  const dropdown = dom.personDropdown || document.getElementById("personDropdown");
  if (!dropdown) return; // Person dropdown not in DOM (page not active)

  let html = '<option value="" disabled>パーソンを選択...</option>';
  for (const p of state.persons) {
    const selected = p.name === state.selectedPerson ? " selected" : "";
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

export async function selectPerson(name) {
  // Check if person is sleeping — offer to start
  const person = state.persons.find((p) => p.name === name);
  if (person && (person.status === "not_found" || person.status === "stopped")) {
    const ok = confirm(`${name} は現在停止中です。起動しますか？`);
    if (!ok) return;
    try {
      await api(`/api/persons/${encodeURIComponent(name)}/start`, { method: "POST" });
    } catch (err) {
      console.error("Failed to start person:", err);
    }
    // Status update will come via WebSocket
    return;
  }

  state.selectedPerson = name;

  // Update dropdown
  const dropdown = dom.personDropdown || document.getElementById("personDropdown");
  if (dropdown) dropdown.value = name;

  // Enable chat
  const chatInput = dom.chatInput || document.getElementById("chatInput");
  const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (chatInput) {
    chatInput.disabled = false;
    chatInput.placeholder = `${name} にメッセージ...`;
  }
  if (chatSendBtn) chatSendBtn.disabled = false;

  // Load conversation history + person detail in parallel
  const needConv = !state.chatHistories[name] || state.chatHistories[name].length === 0;
  const convPromise = needConv
    ? api(`/api/persons/${encodeURIComponent(name)}/conversation/full?limit=20`).catch(() => null)
    : Promise.resolve(null);
  const detailPromise = api(`/api/persons/${encodeURIComponent(name)}`).catch(() => null);

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

  // Apply person detail
  if (detail) {
    state.personDetail = detail;
    renderPersonState();
  } else {
    state.personDetail = null;
    const stateContent = dom.personStateContent || document.getElementById("personStateContent");
    if (stateContent) stateContent.textContent = "詳細の読み込み失敗";
    const memoryList = dom.memoryFileList || document.getElementById("memoryFileList");
    if (memoryList) memoryList.innerHTML = '<div class="loading-placeholder">詳細の読み込み失敗</div>';
  }

  // Load memory, session list, and avatar in parallel
  const secondaryPromises = [loadMemoryTab(state.activeMemoryTab), updatePersonAvatar()];
  if (state.activeRightTab === "history") {
    hideHistoryDetail();
    secondaryPromises.push(loadSessionList());
  }
  await Promise.all(secondaryPromises);
}

// ── Person Avatar ───────────────────────────

export async function updatePersonAvatar() {
  const container = dom.personAvatar || document.getElementById("personAvatar");
  if (!container) return;

  const name = state.selectedPerson;
  if (!name) {
    container.innerHTML = "";
    return;
  }

  // Try bust-up first, then chibi
  const candidates = ["avatar_bustup.png", "avatar_chibi.png"];
  for (const filename of candidates) {
    const url = `/api/persons/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="person-avatar-img">`;
        return;
      }
    } catch { /* try next */ }
  }

  // Fallback: initial letter
  container.innerHTML = `<div class="person-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
}

// ── Person State ───────────────────────────

export function renderPersonState() {
  const stateContent = dom.personStateContent || document.getElementById("personStateContent");
  if (!stateContent) return; // State panel not in DOM

  const d = state.personDetail;
  if (!d || !d.state) {
    stateContent.textContent = "状態情報なし";
    return;
  }
  const stateText = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
  stateContent.textContent = stateText;
}

export async function refreshSelectedPerson() {
  if (!state.selectedPerson) return;
  try {
    state.personDetail = await api(`/api/persons/${encodeURIComponent(state.selectedPerson)}`);
    renderPersonState();
  } catch {
    // Silently ignore refresh errors
  }
}
