// ── Board mobile top bar ─────────────────────
// Shared one-line header used by every board view on phones: the app menu,
// the channel-list toggle, the view title and the view's own actions.
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";

export function boardTopBar({ title, actions = "", titleClass = "board-view-title", railToggleId = "" }) {
  const railId = railToggleId ? ` id="${escapeAttr(railToggleId)}"` : "";
  return `<div class="board-view-top">
    <button type="button" class="board-nav-toggle" data-board-nav aria-label="${escapeAttr(t("common.aria_menu"))}" title="${escapeAttr(t("common.aria_menu"))}">☰</button>
    <button type="button" class="board-mobile-toggle"${railId} data-board-rail-toggle aria-expanded="false" aria-label="${escapeAttr(t("board.rail_toggle"))}" title="${escapeAttr(t("board.rail_toggle"))}">
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M6 9l6 6 6-6"></path></svg>
    </button>
    <div class="${titleClass}">${title}</div>
    <div class="board-view-actions">${actions}</div>
  </div>`;
}

export function boardViewLink(target, label) {
  return `<button type="button" class="board-view-link" data-board-view-link="${escapeAttr(target)}" aria-label="${escapeAttr(label)}" title="${escapeAttr(label)}">${escapeHtml(label)}</button>`;
}
