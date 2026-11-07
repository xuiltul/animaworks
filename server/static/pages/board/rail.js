// ── Board rail (channels / DMs) ──────────────
import { t } from "/shared/i18n.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { renderDmList } from "./dm.js";
import { relativeTime } from "./time.js";

function _sparkline(values) {
  const series = Array.isArray(values) && values.length ? values.slice(-14) : new Array(14).fill(0);
  const width = 40;
  const height = 14;
  const pad = 2;
  const max = Math.max(1, ...series.map((value) => Number(value) || 0));
  const points = series.map((value, index) => {
    const x = series.length === 1 ? width / 2 : pad + (index / (series.length - 1)) * (width - pad * 2);
    const y = height - pad - ((Number(value) || 0) / max) * (height - pad * 2);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  });
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const first = points[0];
  const last = points[points.length - 1];
  return `<svg class="board-spark" viewBox="0 0 ${width} ${height}" aria-hidden="true" focusable="false"><polyline class="board-spark-line" points="${line}"></polyline><circle class="board-spark-dot" cx="${first.x}" cy="${first.y}" r="1.4"></circle><circle class="board-spark-dot" cx="${last.x}" cy="${last.y}" r="1.4"></circle></svg>`;
}

function _channelRow(channel, selectedName) {
  const active = selectedName === channel.name;
  const count = Number(channel.message_count) || 0;
  const last = channel.last_post_ts
    ? `${escapeHtml(channel.last_from || "?")} · ${escapeHtml(relativeTime(channel.last_post_ts))}`
    : escapeHtml(t("board.rail_no_posts"));
  return `<button type="button" class="board-rail-item${active ? " board-is-active" : ""}" data-board-type="channel" data-board-name="${escapeAttr(channel.name)}"${active ? ' aria-current="page"' : ""}>
    <span class="board-rail-icon" aria-hidden="true">#</span>
    <span class="board-rail-text">
      <span class="board-rail-name">${escapeHtml(channel.name)}</span>
      <span class="board-rail-sub">${last}</span>
    </span>
    <span class="board-rail-side">
      ${count > 0 ? `<span class="board-rail-count">${count}</span>` : ""}
      ${_sparkline(channel.activity_14d)}
    </span>
  </button>`;
}

function _groupLabel(key) {
  return key || t("board.rail_groups_other");
}

function _channelGroups(store) {
  const groups = new Map();
  for (const channel of store.state.channels) {
    if (channel.access === "closed") continue;
    const key = String(channel.company || "").trim();
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(channel);
  }

  const result = [...groups.entries()].map(([key, rows]) => {
    rows.sort((a, b) => String(b.last_post_ts || "").localeCompare(String(a.last_post_ts || "")));
    return { key, rows, latest: rows[0]?.last_post_ts || "" };
  });

  result.sort((a, b) => {
    if (a.key === "" && b.key !== "") return 1;
    if (a.key !== "" && b.key === "") return -1;
    const byLatest = String(b.latest).localeCompare(String(a.latest));
    return byLatest || a.key.localeCompare(b.key);
  });
  return result;
}

export function renderRail(element, store) {
  const mode = store.state.mode;
  const modeButtons = `
    <div class="board-mode-switch" role="group" aria-label="${escapeAttr(t("board.mode_switch"))}">
      <button type="button" class="board-mode-btn${mode === "board" ? " board-is-active" : ""}" data-board-mode="board" aria-pressed="${mode === "board"}">${escapeHtml(t("board.mode_board"))}</button>
      <button type="button" class="board-mode-btn${mode === "dm" ? " board-is-active" : ""}" data-board-mode="dm" aria-pressed="${mode === "dm"}">${escapeHtml(t("board.mode_dm"))}</button>
    </div>`;

  if (mode === "dm") {
    element.innerHTML = `<div class="board-rail-top">${modeButtons}</div>${renderDmList(store)}`;
    return;
  }

  const groups = _channelGroups(store);
  let body = "";
  if (store.state.channels.length === 0) {
    body = `<div class="board-rail-empty">${escapeHtml(t("board.channel_none"))}</div>`;
  } else {
    for (const group of groups) {
      body += `<div class="board-rail-group">
        <div class="board-rail-label">${escapeHtml(_groupLabel(group.key))}</div>
        ${group.rows.map((channel) => _channelRow(channel, store.state.selectedType === "channel" ? store.state.selectedName : null)).join("")}
      </div>`;
    }
  }

  const archived = store.state.channels
    .filter((channel) => channel.access === "closed")
    .sort((a, b) => String(b.last_post_ts || "").localeCompare(String(a.last_post_ts || "")));
  const archive = archived.length
    ? `<details class="board-rail-group board-rail-archive" data-board-archive${store.state.archiveOpen ? " open" : ""}>
        <summary class="board-rail-label board-rail-summary">${escapeHtml(t("board.rail_archive"))}<span class="board-rail-archive-count">${archived.length}</span></summary>
        ${archived.map((channel) => _channelRow(channel, store.state.selectedType === "channel" ? store.state.selectedName : null)).join("")}
      </details>`
    : "";

  element.innerHTML = `<div class="board-rail-top">${modeButtons}</div><div class="board-rail-list">${body}${archive}</div>`;

  const details = element.querySelector("[data-board-archive]");
  if (details) {
    details.addEventListener("toggle", () => store.setArchiveOpen(details.open));
  }
}
