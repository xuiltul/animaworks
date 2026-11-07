// ── Board DM rail / header ───────────────────
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { avatarMarkup } from "./avatar.js";
import { relativeTime } from "./time.js";

function _participantAvatars(participants, store) {
  return (participants || []).map((name) => avatarMarkup({
    name,
    kind: "anima",
    url: store.avatarUrl(name),
    size: "xs",
  })).join("");
}

function _filterOptions(store) {
  const names = new Set();
  for (const pair of store.state.dmPairs) {
    for (const name of pair.participants || []) names.add(name);
  }
  const sorted = [...names].sort((a, b) => a.localeCompare(b, "ja"));
  return [`<option value="">${escapeHtml(t("board.dm_filter_all"))}</option>`]
    .concat(sorted.map((name) => `<option value="${escapeAttr(name)}"${name === store.state.dmFilter ? " selected" : ""}>${escapeHtml(name)}</option>`))
    .join("");
}

export function renderDmList(store) {
  const all = store.state.dmPairs;
  const filtered = (store.state.dmFilter
    ? all.filter((pair) => (pair.participants || []).includes(store.state.dmFilter))
    : [...all]
  ).sort((a, b) => String(b.last_message_ts || "").localeCompare(String(a.last_message_ts || "")));

  const header = `<div class="board-rail-label board-dm-label">${escapeHtml(t("board.dm_pairs_label"))}<span class="board-rail-count">${all.length}</span></div>
    <select class="board-dm-filter" data-board-dm-filter aria-label="${escapeAttr(t("board.dm_filter_title"))}" title="${escapeAttr(t("board.dm_filter_title"))}">${_filterOptions(store)}</select>`;

  if (filtered.length === 0) {
    const message = all.length === 0 ? t("board.dm_none") : t("board.dm_none_filtered");
    return `${header}<div class="board-rail-empty">${escapeHtml(message)}</div>`;
  }

  const rows = filtered.map((pair) => {
    const participants = pair.participants || [];
    const active = store.state.selectedType === "dm" && store.state.selectedName === pair.pair;
    const count = Number(pair.message_count) || 0;
    const labels = participants.map((name) => `<span>${escapeHtml(name)}</span>`).join('<span class="board-dm-arrow">↔</span>');
    return `<button type="button" class="board-rail-item board-dm-item${active ? " board-is-active" : ""}" data-board-type="dm" data-board-name="${escapeAttr(pair.pair || "")}"${active ? ' aria-current="page"' : ""}>
      <span class="board-dm-participants">${_participantAvatars(participants, store)}${labels}</span>
      <span class="board-dm-meta">${count > 0 ? `<span class="board-rail-count">${escapeHtml(t("board.dm_count", { count }))}</span>` : ""}<span class="board-rail-sub">${escapeHtml(relativeTime(pair.last_message_ts))}</span></span>
    </button>`;
  }).join("");

  return `${header}${rows}`;
}

export function renderDmHeader(element, pair, store) {
  const participants = pair?.participants || String(store.state.selectedName || "").split("-");
  const title = participants.map((name) => `${avatarMarkup({ name, kind: "anima", url: store.avatarUrl(name), size: "sm" })}<span>${escapeHtml(name)}</span>`).join('<span class="board-dm-arrow">↔</span>');
  element.innerHTML = `
    <div class="board-header-title">${title}</div>
    <div class="board-header-meta">
      <span class="board-badge">${escapeHtml(t("board.dm_readonly"))}</span>
      ${pair ? `<span class="board-header-muted">${escapeHtml(t("board.dm_count", { count: pair.message_count || 0 }))}</span>` : ""}
    </div>`;
}
