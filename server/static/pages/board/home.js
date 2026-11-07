// ── Board home ──────────────────────────────
// A single overview of the boards that are still open, newest post first.
import { api } from "../../modules/api.js";
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";
import { avatarMarkup } from "./avatar.js";
import { numberText, relativeTime } from "./time.js";

const SPARK_WIDTH = 240;
const SPARK_HEIGHT = 42;

function _activity(channel) {
  const values = Array.isArray(channel.activity_14d) ? channel.activity_14d.slice(-14) : [];
  return values.length ? values.map((value) => Number(value) || 0) : new Array(14).fill(0);
}

function _sparkline(values) {
  const series = values.length ? values : new Array(14).fill(0);
  const width = SPARK_WIDTH;
  const height = SPARK_HEIGHT;
  const pad = 4;
  const max = Math.max(1, ...series);
  const floor = height - pad;
  const points = series.map((value, index) => {
    const x = series.length === 1
      ? width / 2
      : pad + (index / (series.length - 1)) * (width - pad * 2);
    const y = floor - (value / max) * (height - pad * 2);
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
  });
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `${points[0].x},${floor} ${line} ${points[points.length - 1].x},${floor}`;
  const first = points[0];
  const last = points[points.length - 1];
  return `<svg class="board-home-spark" viewBox="0 0 ${width} ${height}" aria-hidden="true" focusable="false">
    <polygon class="board-home-spark-area" points="${area}"></polygon>
    <polyline class="board-home-spark-line" points="${line}"></polyline>
    <circle class="board-home-spark-dot" cx="${first.x}" cy="${first.y}" r="2.2"></circle>
    <circle class="board-home-spark-dot" cx="${last.x}" cy="${last.y}" r="2.2"></circle>
  </svg>`;
}

function _slackDirection(slack) {
  const inbound = Boolean(slack?.inbound);
  const outbound = Boolean(slack?.outbound);
  if (inbound && outbound) return t("board.slack_both");
  if (inbound) return t("board.slack_inbound");
  if (outbound) return t("board.slack_outbound");
  return t("board.slack_none");
}

function _unread(channel) {
  const raw = channel.unread;
  if (raw && typeof raw === "object") return Number(raw.count || raw.unread || 0) || 0;
  return Number(raw) || 0;
}

function _headLines(text) {
  const lines = String(text || "")
    .replace(/\r/g, "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.slice(0, 2).join("\n");
}

function _lastMarkup(channel, cached, store) {
  const ts = String(channel.last_post_ts || "");
  if (!ts) {
    return `<div class="board-home-last board-home-last-empty">${escapeHtml(t("board.rail_no_posts"))}</div>`;
  }
  const from = String(cached?.from || channel.last_from || "?");
  const text = _headLines(cached?.text);
  const kind = store.hasAnima(from) ? "anima" : "human";
  const body = text
    ? escapeHtml(text)
    : escapeHtml(t("board.loading"));
  return `<div class="board-home-last">
    ${avatarMarkup({ name: from, kind, url: store.avatarUrl(from), size: "xs" })}
    <div class="board-home-last-main">
      <div class="board-home-last-head"><b>${escapeHtml(from)}</b><span>${escapeHtml(relativeTime(ts))}</span></div>
      <div class="board-home-last-text">${body}</div>
    </div>
  </div>`;
}

function _card(channel, cached, store) {
  const activity = _activity(channel);
  const total = activity.reduce((sum, value) => sum + value, 0);
  const unread = _unread(channel);
  const name = channel.name;
  return `<button type="button" class="board-home-card" data-board-home-card="${escapeAttr(name)}" aria-label="${escapeAttr(t("board.home_open_board", { name }))}">
    <span class="board-home-card-top">
      <span class="board-home-hash" aria-hidden="true">#</span>
      <span class="board-home-name">${escapeHtml(name)}</span>
      ${unread > 0 ? `<span class="board-home-unread" title="${escapeAttr(t("board.home_unread", { count: unread }))}">${numberText(unread)}</span>` : ""}
      ${channel.company ? `<span class="board-home-company">${escapeHtml(channel.company)}</span>` : ""}
    </span>
    ${_sparkline(activity)}
    ${_lastMarkup(channel, cached, store)}
    <span class="board-home-card-foot">
      <span>${escapeHtml(t("board.home_activity_14d", { count: numberText(total) }))}</span>
      <span>${escapeHtml(_slackDirection(channel.slack))}</span>
    </span>
  </button>`;
}

export function createHomeView({ element, store }) {
  const cache = new Map();
  const pending = new Set();
  let destroyed = false;

  function channels() {
    return store.state.channels
      .filter((channel) => channel && channel.access !== "closed")
      .sort((a, b) => String(b.last_post_ts || "").localeCompare(String(a.last_post_ts || "")));
  }

  function fetchLast(channel) {
    const name = channel.name;
    const ts = String(channel.last_post_ts || "");
    if (!name || !ts || pending.has(name)) return;
    const cached = cache.get(name);
    if (cached && cached.ts === ts) return;
    pending.add(name);
    api(`/api/channels/${encodeURIComponent(name)}?limit=1`)
      .then((data) => {
        const rows = Array.isArray(data?.messages) ? data.messages : [];
        const last = rows[rows.length - 1] || null;
        cache.set(name, {
          ts,
          from: last?.from || channel.last_from || "",
          text: last?.text || "",
        });
      })
      .catch(() => {
        cache.set(name, { ts, from: channel.last_from || "", text: "" });
      })
      .finally(() => {
        pending.delete(name);
        if (!destroyed) render();
      });
  }

  function render() {
    if (destroyed) return;
    const rows = channels();
    if (rows.length === 0) {
      element.innerHTML = `<div class="board-home-head">
        <h1 class="board-home-title">${escapeHtml(t("board.home_title"))}</h1>
        <p class="board-home-subtitle">${escapeHtml(t("board.home_subtitle"))}</p>
      </div><div class="board-home-empty">${escapeHtml(t("board.home_empty"))}</div>`;
      return;
    }

    element.innerHTML = `<div class="board-home-head">
      <h1 class="board-home-title">${escapeHtml(t("board.home_title"))}</h1>
      <p class="board-home-subtitle">${escapeHtml(t("board.home_subtitle"))}</p>
    </div>
    <div class="board-home-grid">
      ${rows.map((channel) => _card(channel, cache.get(channel.name), store)).join("")}
    </div>`;

    for (const channel of rows) fetchLast(channel);
  }

  function destroy() {
    destroyed = true;
    pending.clear();
    cache.clear();
    element.innerHTML = "";
  }

  return { render, destroy };
}
