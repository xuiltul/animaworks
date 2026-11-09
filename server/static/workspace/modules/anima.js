// ── Anima Selector ──────────────────────
// Header dropdown for selecting the active anima.

import { getState, setState } from "./state.js";
import { t } from "/shared/i18n.js";
import { fetchAnimas } from "./api.js";
import { escapeHtml } from "./utils.js";
import { animaStatusLabel } from "/shared/anima-status.js";

// ── Private State ──────────────────────

let _selectorContainer = null;
let _onAnimaSelect = null;

// ── Dropdown Rendering ──────────────────────

function renderDropdown() {
  if (!_selectorContainer) return;

  const { animas, selectedAnima } = getState();

  let html = `<select class="anima-dropdown" id="wsAnimaDropdown" aria-label="${t("ws.select_anima")}">`;
  html += `<option value="" disabled>${t("ws.select_anima")}</option>`;

  for (const p of animas) {
    const selected = p.name === selectedAnima ? " selected" : "";
    if (p.status === "bootstrapping" || p.bootstrapping) {
      html += `<option value="${escapeHtml(p.name)}"${selected} disabled>\u23F3 ${escapeHtml(p.name)} (${t("animas.bootstrapping")})</option>`;
    } else if (p.status === "not_found" || p.status === "stopped") {
      html += `<option value="${escapeHtml(p.name)}"${selected}>\uD83D\uDCA4 ${escapeHtml(p.name)} (${t("animas.stopped")})</option>`;
    } else {
      const st = p.status ? ` (${escapeHtml(animaStatusLabel(p.status))})` : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${st}</option>`;
    }
  }

  html += "</select>";
  _selectorContainer.innerHTML = html;

  const dropdown = _selectorContainer.querySelector("#wsAnimaDropdown");
  if (dropdown) {
    dropdown.addEventListener("change", (e) => {
      const name = e.target.value;
      if (name) selectAnima(name);
    });
  }
}

// ── Public API ──────────────────────

/**
 * Initialize the anima selector.
 * @param {HTMLElement} selectorContainer - DOM element for the dropdown
 * @param {function} onAnimaSelect - Callback invoked with anima name on selection
 */
export function initAnima(selectorContainer, onAnimaSelect) {
  _selectorContainer = selectorContainer;
  _onAnimaSelect = onAnimaSelect;
  renderDropdown();
}

/** Re-render the dropdown after external status updates. */
export function renderAnimaSelector() {
  renderDropdown();
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
        `<div class="loading-placeholder" style="color:#ef4444;">${t("animas.detail_load_failed")}</div>`;
    }
  }
}

/**
 * Select an anima by name and notify the shell.
 */
export function selectAnima(name) {
  setState({ selectedAnima: name });
  renderDropdown();
  if (_onAnimaSelect) {
    _onAnimaSelect(name);
  }
}
