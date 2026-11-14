// ── Board channel header ─────────────────────
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { numberText } from "./time.js";
import { boardTopBar } from "./topbar.js";

const FILTER_KINDS = ["all", "anima", "human", "external", "operator"];

function _kindLabel(kind) {
  return t(`board.filter_kind_${kind}`);
}

function _filterChips(store) {
  const { kindFilter, filterCounts } = store.state;
  return FILTER_KINDS
    .filter((kind) => kind === "all" || (Number(filterCounts[kind]) || 0) > 0)
    .map((kind) => {
      const count = kind === "all"
        ? Number(filterCounts.all) || 0
        : Number(filterCounts[kind]) || 0;
      const active = kindFilter === kind;
      return `<button type="button" class="board-filter-chip${active ? " board-is-active" : ""}" data-board-kind="${escapeAttr(kind)}" aria-pressed="${active ? "true" : "false"}">
        <span class="board-filter-chip-label">${escapeHtml(_kindLabel(kind))}</span>
        <span class="board-filter-chip-count">${escapeHtml(numberText(count))}</span>
      </button>`;
    })
    .join("");
}

function _filterMarkup(store, channel) {
  const state = store.state;
  const serverFiltered = state.kindFilter !== "all" || Boolean(state.query);
  const filtered = serverFiltered || Boolean(state.personFilter);
  const total = Number(state.boardTotal) || Number(channel.message_count) || 0;
  const shown = state.personFilter
    ? store.visibleMessages().length
    : serverFiltered
      ? Number(state.filterTotal) || 0
      : total;
  const countText = filtered
    ? `<span class="board-filter-total" data-board-filter-count>${escapeHtml(t("board.filter_count", { shown: numberText(shown), total: numberText(total) }))}</span>`
    : "";

  return `<div class="board-header-filters" role="group" aria-label="${escapeAttr(t("board.filter_search_label"))}"${state.filtersOpen ? "" : " hidden"}>
    <div class="board-filter-chips">${_filterChips(store)}</div>
    <label class="board-filter-search" for="boardSearchInput">
      <span class="board-filter-search-icon" aria-hidden="true">⌕</span>
      <input id="boardSearchInput" type="search" data-board-search-input
        placeholder="${escapeAttr(t("board.filter_search_placeholder"))}"
        aria-label="${escapeAttr(t("board.filter_search_label"))}"
        value="${escapeAttr(state.query)}" autocomplete="off">
    </label>
    ${countText}
  </div>`;
}

function _searchButton(store) {
  const state = store.state;
  const open = Boolean(state.filtersOpen);
  const active = !open && store.isFiltered();
  return `<button type="button" class="board-search-toggle${active ? " board-has-filter" : ""}" data-board-search-toggle aria-expanded="${open ? "true" : "false"}" aria-label="${escapeAttr(t("board.search_toggle"))}" title="${escapeAttr(t("board.search_toggle"))}">
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="11" cy="11" r="7"></circle><path d="M16.5 16.5L21 21"></path></svg>
  </button>`;
}

function _infoButton() {
  return `<button type="button" class="board-info-button" data-board-info aria-expanded="false" aria-controls="boardDrawer" aria-label="${escapeAttr(t("board.info_button"))}" title="${escapeAttr(t("board.info_button"))}">
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="9"></circle><path d="M12 10.5v5"></path><circle class="board-info-dot" cx="12" cy="7.5" r="1"></circle></svg>
  </button>`;
}

/**
 * Render the channel header. A pinned board may pass ``titleHtml`` (already
 * escaped markup) to replace the "# name" title with its own control.
 */
export function renderChannelHeader(element, channel, store, { titleHtml = "" } = {}) {
  if (!channel) {
    element.innerHTML = `<div class="board-header-title">${escapeHtml(t("board.loading"))}</div>`;
    element.classList.remove("board-is-filters-open");
    return;
  }

  const state = store.state;
  element.classList.toggle("board-is-filters-open", Boolean(state.filtersOpen));

  const previousInput = element.querySelector("[data-board-search-input]");
  const hadFocus = Boolean(state.filtersOpen)
    && previousInput
    && document.activeElement === previousInput;
  const caret = hadFocus ? previousInput.selectionStart : null;

  const title = titleHtml || `<span class="board-header-hash" aria-hidden="true">#</span>
    <span class="board-header-name" title="${escapeAttr(channel.name)}">${escapeHtml(channel.name)}</span>`;

  element.innerHTML = `${boardTopBar({
    title,
    titleClass: "board-header-title",
    actions: `${_searchButton(store)}${_infoButton()}`,
  })}
  ${_filterMarkup(store, channel)}`;

  if (hadFocus) {
    const input = element.querySelector("[data-board-search-input]");
    if (input) {
      input.focus();
      const position = caret === null ? input.value.length : Math.min(caret, input.value.length);
      input.setSelectionRange(position, position);
    }
  }
}
