/* ── Anima Dropdown, Selection, Avatar ────── */

import { state, dom, escapeHtml } from "./state.js";
import { t } from "/shared/i18n.js";
import { api } from "./api.js";
import { loadMemoryTab } from "./memory.js";
import { animaHashColor } from "../shared/avatar-utils.js";
import { bustupCandidates, resolveCachedAvatar } from "./avatar-resolver.js";

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

// ── Shared Anima / Process helpers (UI consolidation) ──

/**
 * Thin wrapper around GET /api/animas (no caching).
 * @returns {Promise<Array>}
 */
export async function fetchAnimasList() {
  return api("/api/animas");
}

/**
 * Load system process map, falling back to empty object.
 * @returns {Promise<Record<string, object>>}
 */
export async function fetchProcessMap() {
  try {
    const data = await api("/api/system/status");
    return data.processes || {};
  } catch {
    return {};
  }
}

/**
 * Merge /api/animas list with /api/system/status process details.
 * Process fields override anima fields when present.
 * @returns {Promise<Array<object>>}
 */
export async function fetchAnimasWithProcessStatus() {
  const [animas, processes] = await Promise.all([
    fetchAnimasList(),
    fetchProcessMap(),
  ]);
  const processEntries = Object.entries(processes);

  if (animas.length === 0 && processEntries.length > 0) {
    return processEntries.map(([name, proc]) => ({ name, ...proc }));
  }

  return animas.map((a) => {
    const proc = processes[a.name];
    if (proc) return { ...a, ...proc, name: a.name };
    return a;
  });
}

/**
 * Health-dot HTML (same visuals as the former processes page).
 * @param {string} status
 * @param {number} [missedPings=0]
 * @returns {string}
 */
export function healthIndicatorHtml(status, missedPings = 0) {
  if (status === "error" || status === "down") {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;" title="${t("processes.health_error")}"></span>`;
  }
  if (missedPings > 0) {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f59e0b;" title="${t("processes.health_warning")}"></span>`;
  }
  if (status === "running" || status === "idle") {
    return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#22c55e;" title="${t("processes.health_ok")}"></span>`;
  }
  return `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#9ca3af;" title="${t("common.unknown")}"></span>`;
}

/**
 * Status badge HTML (process-monitoring style).
 * @param {string} status
 * @returns {string}
 */
export function statusBadgeHtml(status) {
  const s = status || "unknown";
  let cls = "warning";
  if (s === "running" || s === "idle") cls = "success";
  else if (s === "error" || s === "down") cls = "error";
  else if (s === "stopped" || s === "not_found" || s === "offline") cls = "";
  return `<span class="status-badge ${cls}">${escapeHtml(s)}</span>`;
}

/**
 * Format uptime seconds to a short human string.
 * @param {number} seconds
 * @returns {string}
 */
export function formatUptime(seconds) {
  if (!seconds || seconds < 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return t("animas.uptime_hm", { h, m });
  return t("animas.uptime_m", { m });
}

/**
 * Build process action buttons HTML for a single anima.
 * @param {string} name
 * @param {string} status
 * @returns {string}
 */
export function processActionButtonsHtml(name, status) {
  const eName = escapeHtml(name);
  const btnStyle = 'style="font-size:0.8rem; padding:0.25rem 0.5rem;"';

  if (status === "running" || status === "idle") {
    return `
      <button class="btn-primary process-trigger-btn" data-name="${eName}" ${btnStyle}>Heartbeat</button>
      <button class="btn-warning process-interrupt-btn" data-name="${eName}" ${btnStyle}>${t("processes.interrupt")}</button>
      <button class="btn-warning process-restart-btn" data-name="${eName}" ${btnStyle}>${t("processes.restart")}</button>
      <button class="btn-danger process-stop-btn" data-name="${eName}" ${btnStyle}>${t("processes.stop")}</button>
    `;
  }

  if (status === "stopped" || status === "not_found" || status === "offline") {
    return `
      <button class="btn-success process-start-btn" data-name="${eName}" ${btnStyle}>${t("processes.start")}</button>
    `;
  }

  if (status === "starting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">${t("processes.starting")}</span>`;
  }

  if (status === "restarting") {
    return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">${t("processes.restarting")}</span>`;
  }

  return `<span style="font-size:0.8rem; color:var(--aw-color-text-muted);">--</span>`;
}

/**
 * Bind process action button handlers inside a container.
 * @param {HTMLElement} container
 * @param {{ onReload?: () => void }} [opts]
 */
export function bindProcessActionButtons(container, opts = {}) {
  if (!container) return;
  const onReload = opts.onReload;

  container.querySelectorAll(".process-trigger-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      _handleProcessAction(btn, "trigger", {
        label: "Heartbeat",
        busyLabel: t("processes.running"),
        doneLabel: t("processes.done"),
        onReload,
      });
    });
  });

  container.querySelectorAll(".process-stop-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const name = btn.dataset.name;
      if (!confirm(t("processes.confirm_stop", { name }))) return;
      _handleProcessAction(btn, "stop", {
        label: t("processes.stop"),
        busyLabel: t("processes.stopping"),
        doneLabel: t("processes.stop_done"),
        reload: true,
        onReload,
      });
    });
  });

  container.querySelectorAll(".process-start-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      _handleProcessAction(btn, "start", {
        label: t("processes.start"),
        busyLabel: t("processes.starting"),
        doneLabel: t("processes.start_done"),
        reload: true,
        onReload,
      });
    });
  });

  container.querySelectorAll(".process-restart-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const name = btn.dataset.name;
      if (!confirm(t("processes.confirm_restart", { name }))) return;
      _handleProcessAction(btn, "restart", {
        label: t("processes.restart"),
        busyLabel: t("processes.restarting"),
        doneLabel: t("processes.restart_done"),
        reload: true,
        onReload,
      });
    });
  });

  container.querySelectorAll(".process-interrupt-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      _handleProcessAction(btn, "interrupt", {
        label: t("processes.interrupt"),
        busyLabel: t("processes.interrupting"),
        doneLabel: t("processes.interrupt_done"),
        reload: true,
        onReload,
      });
    });
  });
}

async function _handleProcessAction(btn, action, opts) {
  const name = btn.dataset.name;
  btn.disabled = true;
  btn.textContent = opts.busyLabel;

  try {
    await api(`/api/animas/${encodeURIComponent(name)}/${action}`, { method: "POST" });
    btn.textContent = opts.doneLabel;

    if (opts.reload && typeof opts.onReload === "function") {
      setTimeout(opts.onReload, 1000);
    }
    setTimeout(() => {
      btn.textContent = opts.label;
      btn.disabled = false;
    }, 2000);
  } catch {
    btn.textContent = t("animas.failed");
    setTimeout(() => {
      btn.textContent = opts.label;
      btn.disabled = false;
    }, 2000);
  }
}

// ── Anima State Class Helper ──────────────────
function statusIndicator(emoji, lucideIcon) {
  const isBusiness = document.body.classList.contains('mode-realistic');
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

  // Update dropdown
  const dropdown = dom.animaDropdown || document.getElementById("animaDropdown");
  if (dropdown) dropdown.value = name;

  // Load anima detail
  const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);
  const detail = await detailPromise;

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

  // Load memory and avatar in parallel
  await Promise.all([loadMemoryTab(state.activeMemoryTab), updateAnimaAvatar()]);
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

  let imgHtml = "";
  const url = await resolveCachedAvatar(name, bustupCandidates(), "S");
  if (url) {
    imgHtml = `<img class="anima-avatar-img" src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`;
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
