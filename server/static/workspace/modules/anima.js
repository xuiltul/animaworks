// ── Anima Selector & Status Panel ──────────────────────
// Dropdown for anima selection + detail card rendering.

import { getState, setState } from "./state.js";
import { fetchAnimas, fetchAnimaDetail } from "./api.js";
import { escapeHtml, renderSimpleMarkdown } from "./utils.js";

// ── Private State ──────────────────────

let _selectorContainer = null;
let _statusContainer = null;
let _onAnimaSelect = null;

// ── Helpers ──────────────────────

/**
 * Map a status string to its CSS class.
 */
function statusClassName(status) {
  if (!status) return "status-offline";
  const s = status.toLowerCase();
  if (s === "idle" || s === "running") return "status-idle";
  if (s === "thinking" || s === "processing" || s === "busy" || s === "bootstrapping") return "status-thinking";
  if (s === "error") return "status-error";
  return "status-offline";
}

/**
 * Map a status string to a display label.
 */
function statusLabel(status) {
  if (!status) return "offline";
  return status.toLowerCase();
}

/**
 * Map a full model name to a short display alias.
 */
function modelAlias(model) {
  if (!model) return "";
  const m = model.toLowerCase();
  // A1: Claude models
  if (m.includes("opus")) return "Opus";
  if (m.includes("sonnet")) return "Sonnet";
  if (m.includes("haiku")) return "Haiku";
  // A2: OpenAI
  if (m.includes("gpt-4o")) return "GPT-4o";
  if (m.includes("gpt-4")) return "GPT-4";
  if (m.includes("o3")) return "o3";
  if (m.includes("o1")) return "o1";
  // A2: Google
  if (m.includes("gemini")) return "Gemini";
  // B: Ollama / OSS
  if (m.includes("ollama/")) return model.split("/").pop();
  // Fallback: last segment
  const parts = model.split("/");
  return parts[parts.length - 1];
}

// ── Dropdown Rendering ──────────────────────

function renderDropdown() {
  if (!_selectorContainer) return;

  const { animas, selectedAnima } = getState();

  let html = '<select class="anima-dropdown" id="wsAnimaDropdown">';
  html += '<option value="" disabled>Select an anima...</option>';

  for (const p of animas) {
    const selected = p.name === selectedAnima ? " selected" : "";
    if (p.status === "bootstrapping" || p.bootstrapping) {
      html += `<option value="${escapeHtml(p.name)}"${selected} disabled>\u23F3 ${escapeHtml(p.name)} (制作中...)</option>`;
    } else if (p.status === "not_found" || p.status === "stopped") {
      html += `<option value="${escapeHtml(p.name)}"${selected}>\uD83D\uDCA4 ${escapeHtml(p.name)} (停止中)</option>`;
    } else {
      const st = p.status ? ` (${p.status})` : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${st}</option>`;
    }
  }

  html += "</select>";
  _selectorContainer.innerHTML = html;

  // Bind change event
  const dropdown = _selectorContainer.querySelector("#wsAnimaDropdown");
  if (dropdown) {
    dropdown.addEventListener("change", (e) => {
      const name = e.target.value;
      if (name) selectAnima(name);
    });
  }
}

// ── Status Panel Rendering ──────────────────────

function renderStatusPanel() {
  if (!_statusContainer) return;

  const { animaDetail, selectedAnima } = getState();

  if (!selectedAnima) {
    _statusContainer.innerHTML = `
      <div class="anima-status-panel">
        <div class="loading-placeholder">Select an anima to view details</div>
      </div>
    `;
    return;
  }

  if (!animaDetail) {
    _statusContainer.innerHTML = `
      <div class="anima-status-panel">
        <div class="loading-placeholder">Loading...</div>
      </div>
    `;
    return;
  }

  const d = animaDetail;
  const rawStatus = d.status;
  const statusStr = (rawStatus && typeof rawStatus === "object") ? (rawStatus.status || "offline") : (rawStatus || "offline");
  const dotClass = statusClassName(statusStr);

  // Resolve model alias from animas list
  const { animas } = getState();
  const animaEntry = animas.find((a) => a.name === selectedAnima);
  const alias = modelAlias(animaEntry?.model);

  // Build sections
  let sectionsHtml = "";

  // Identity section (Markdown)
  if (d.identity) {
    sectionsHtml += `
      <div class="status-section">
        <div class="status-section-title">Identity</div>
        <div class="status-section-body md-content">${renderSimpleMarkdown(d.identity)}</div>
      </div>
    `;
  }

  // Injection section (Markdown)
  if (d.injection) {
    sectionsHtml += `
      <div class="status-section">
        <div class="status-section-title">Injection</div>
        <div class="status-section-body md-content">${renderSimpleMarkdown(d.injection)}</div>
      </div>
    `;
  }

  // State section (Markdown for strings, JSON for objects)
  if (d.state) {
    const isString = typeof d.state === "string";
    const body = isString
      ? `<div class="md-content">${renderSimpleMarkdown(d.state)}</div>`
      : `<pre>${escapeHtml(JSON.stringify(d.state, null, 2))}</pre>`;
    sectionsHtml += `
      <div class="status-section">
        <div class="status-section-title">State</div>
        <div class="status-section-body">${body}</div>
      </div>
    `;
  }

  // Pending section (Markdown for strings, JSON for objects)
  if (d.pending) {
    const isString = typeof d.pending === "string";
    const body = isString
      ? `<div class="md-content">${renderSimpleMarkdown(d.pending)}</div>`
      : `<pre>${escapeHtml(JSON.stringify(d.pending, null, 2))}</pre>`;
    sectionsHtml += `
      <div class="status-section">
        <div class="status-section-title">Pending</div>
        <div class="status-section-body">${body}</div>
      </div>
    `;
  }

  // Fallback if no sections
  if (!sectionsHtml) {
    sectionsHtml = '<div class="loading-placeholder">No detail available</div>';
  }

  _statusContainer.innerHTML = `
    <div class="anima-status-panel">
      <div class="status-header">
        <span class="status-dot ${dotClass}"></span>
        <span class="status-anima-name">${escapeHtml(selectedAnima)}</span>
        <span class="status-label">${escapeHtml(statusLabel(statusStr))}</span>
        ${alias ? `<span class="status-model">${escapeHtml(alias)}</span>` : ""}
      </div>
      ${sectionsHtml}
    </div>
  `;
}

function truncate(str, maxLen) {
  if (!str) return "";
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + "...";
}

// ── Public API ──────────────────────

/**
 * Render the anima selector dropdown into the given container.
 */
export function renderAnimaSelector(container) {
  _selectorContainer = container || _selectorContainer;
  renderDropdown();
}

/**
 * Render the anima status detail panel into the given container.
 */
export function renderStatus(container) {
  _statusContainer = container || _statusContainer;
  renderStatusPanel();
}

/**
 * Initialize the anima module.
 * @param {HTMLElement} selectorContainer - DOM element for the dropdown
 * @param {HTMLElement} statusContainer - DOM element for the status panel
 * @param {function} onAnimaSelect - Callback invoked with anima name on selection
 */
export function initAnima(selectorContainer, statusContainer, onAnimaSelect) {
  _selectorContainer = selectorContainer;
  _statusContainer = statusContainer;
  _onAnimaSelect = onAnimaSelect;

  // Render initial empty state
  renderDropdown();
  renderStatusPanel();
}

/**
 * Load animas list from API and update the dropdown.
 */
export async function loadAnimas() {
  try {
    const animas = await fetchAnimas();
    setState({ animas });
    renderDropdown();
  } catch (err) {
    console.error("Failed to load animas:", err);
    if (_selectorContainer) {
      _selectorContainer.innerHTML =
        '<div class="loading-placeholder" style="color:#ef4444;">Failed to load animas</div>';
    }
  }
}

/**
 * Select an anima by name. Fetches detail and updates state + UI.
 */
export async function selectAnima(name) {
  setState({ selectedAnima: name, animaDetail: null });
  renderDropdown();
  renderStatusPanel();

  try {
    const detail = await fetchAnimaDetail(name);
    setState({ animaDetail: detail });
    renderStatusPanel();
  } catch (err) {
    console.error(`Failed to load anima detail for "${name}":`, err);
    if (_statusContainer) {
      _statusContainer.innerHTML = `
        <div class="anima-status-panel">
          <div class="loading-placeholder" style="color:#ef4444;">
            Failed to load details for ${escapeHtml(name)}
          </div>
        </div>
      `;
    }
  }

  // Notify callback
  if (_onAnimaSelect) {
    _onAnimaSelect(name);
  }
}
