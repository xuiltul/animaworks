// ── Board channel header ─────────────────────
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";
import { numberText } from "./time.js";

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
  const total = Number(state.boardTotal) || Number(channel.message_count) || 0;
  const shown = state.personFilter
    ? store.visibleMessages().length
    : serverFiltered
      ? Number(state.filterTotal) || 0
      : total;
  const countText = serverFiltered || state.personFilter
    ? t("board.filter_count", { shown: numberText(shown), total: numberText(total) })
    : t("board.filter_count_all", { total: numberText(total) });

  return `<div class="board-header-filters" role="group" aria-label="${escapeAttr(t("board.filter_search_label"))}">
    ${_filterChips(store)}
    <label class="board-filter-search" for="boardSearchInput">
      <span class="board-filter-search-icon" aria-hidden="true">⌕</span>
      <input id="boardSearchInput" type="search" data-board-search-input
        placeholder="${escapeAttr(t("board.filter_search_placeholder"))}"
        aria-label="${escapeAttr(t("board.filter_search_label"))}"
        value="${escapeAttr(state.query)}" autocomplete="off">
    </label>
    <span class="board-filter-total" data-board-filter-count>${escapeHtml(countText)}</span>
  </div>`;
}

function _accessBadge(access, memberCount) {
  if (access === "closed") return `<span class="board-badge board-badge-access board-badge-access-closed">${escapeHtml(t("board.access_closed"))}</span>`;
  if (access === "members") {
    return `<span class="board-badge board-badge-access board-badge-access-members">${escapeHtml(t("board.access_members", { count: memberCount }))}</span>`;
  }
  return `<span class="board-badge board-badge-access board-badge-access-open">${escapeHtml(t("board.access_open"))}</span>`;
}

export function renderChannelHeader(element, channel, store) {
  if (!channel) {
    element.innerHTML = `<div class="board-header-title">${escapeHtml(t("board.loading"))}</div>`;
    return;
  }

  const previousInput = element.querySelector("[data-board-search-input]");
  const hadFocus = previousInput && document.activeElement === previousInput;
  const caret = hadFocus ? previousInput.selectionStart : null;

  const members = Array.isArray(channel.members) ? channel.members : [];
  const visibleMembers = members.slice(0, 6);
  const memberAvatars = visibleMembers.map((name) => avatarMarkup({
    name,
    kind: "anima",
    url: store.avatarUrl(name),
    size: "sm",
  })).join("");
  const moreMembers = members.length > visibleMembers.length
    ? `<span class="board-members-more" title="${escapeHtml(members.slice(6).join(", "))}">${escapeHtml(t("board.members_more", { count: members.length - visibleMembers.length }))}</span>`
    : "";

  const slack = channel.slack || {};
  const slackBadges = [
    slack.inbound ? `<span class="board-badge board-badge-slack">${escapeHtml(t("board.slack_inbound"))}</span>` : "",
    slack.outbound ? `<span class="board-badge board-badge-slack">${escapeHtml(t("board.slack_outbound"))}</span>` : "",
  ].filter(Boolean).join("");

  element.innerHTML = `
    <div class="board-header-top">
      <div class="board-header-title">
        <span class="board-header-hash" aria-hidden="true">#</span>
        <span class="board-header-name">${escapeHtml(channel.name)}</span>
      </div>
      <button type="button" class="board-info-button" data-board-info aria-expanded="false" aria-controls="boardDrawer">${escapeHtml(t("board.info_button"))}</button>
    </div>
    <div class="board-header-meta">
      ${_accessBadge(channel.access, members.length)}
      ${channel.company ? `<span class="board-badge">${escapeHtml(channel.company)}</span>` : ""}
      ${slackBadges}
      ${members.length ? `<span class="board-members">${memberAvatars}${moreMembers}</span>` : ""}
    </div>
    ${channel.description ? `<div class="board-header-description">${escapeHtml(channel.description)}</div>` : ""}
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
