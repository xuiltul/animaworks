// ── Flow charts (SVG) ───────────────────────
// Pure markup builders for the call-flow view: the stage ribbon, the person
// map and the per-day lanes. All values are derived from the same call list
// the list and the chain use, so the figures cannot disagree with each other.
import { escapeAttr, escapeHtml } from "../../modules/state.js";
import { t } from "/shared/i18n.js";
import { clockTime, localDayKey } from "./time.js";

const DAY_MS = 24 * 60 * 60 * 1000;

function _empty() {
  return `<div class="board-flow-empty">${escapeHtml(t("board.flows_empty"))}</div>`;
}

function _values(calls) {
  const delivered = calls.filter((call) => call.delivered).length;
  const read = calls.filter((call) => call.read).length;
  const replied = calls.filter((call) => call.reply).length;
  return {
    all: calls.length,
    delivered,
    undelivered: calls.length - delivered,
    read,
    unread: calls.filter((call) => call.delivered && !call.read).length,
    replied,
    noreply: calls.filter((call) => call.read && !call.reply).length,
  };
}

function _linkPath(a, b, value, offset, k) {
  const width = 14;
  const sx = a.x + width;
  const sy = a.y + offset * k;
  const tx = b.x;
  const ty = b.y;
  const height = Math.max(value * k, 0);
  const mx = (sx + tx) / 2;
  const h = Math.max(height, 2);
  return `M${sx} ${sy.toFixed(1)} C${mx} ${sy.toFixed(1)} ${mx} ${ty.toFixed(1)} ${tx} ${ty.toFixed(1)} L${tx} ${(ty + h).toFixed(1)} C${mx} ${(ty + h).toFixed(1)} ${mx} ${(sy + h).toFixed(1)} ${sx} ${(sy + h).toFixed(1)} Z`;
}

export function sankeyMarkup(calls, stage = "") {
  if (calls.length === 0) return _empty();
  const values = _values(calls);
  const n = Math.max(1, calls.length);
  const W = 640;
  const H = 260;
  const T = 14;
  const NW = 14;
  const GAP = 24;
  const X = [10, 180, 350, 520];
  const k = (H - T - 16 - GAP) / n;

  const nodes = {
    all: { x: X[0], y: T, h: values.all * k, cls: "good" },
    delivered: { x: X[1], y: T, h: values.delivered * k, cls: "good" },
    undelivered: { x: X[1], y: T + values.delivered * k + GAP, h: values.undelivered * k, cls: "bad" },
    read: { x: X[2], y: T, h: values.read * k, cls: "good" },
    unread: { x: X[2], y: T + values.read * k + GAP, h: values.unread * k, cls: "wait" },
    replied: { x: X[3], y: T, h: values.replied * k, cls: "reply" },
    noreply: { x: X[3], y: T + values.replied * k + GAP, h: values.noreply * k, cls: "mute" },
  };
  const labels = {
    all: t("board.flows_stage_all"),
    delivered: t("board.flows_stage_delivered"),
    undelivered: t("board.flows_stage_undelivered"),
    read: t("board.flows_stage_read"),
    unread: t("board.flows_stage_unread"),
    replied: t("board.flows_stage_replied"),
    noreply: t("board.flows_stage_noreply"),
  };
  const parentOf = {
    all: "all",
    delivered: "all",
    undelivered: "all",
    read: "delivered",
    unread: "delivered",
    replied: "read",
    noreply: "read",
  };
  const links = [
    ["all", "delivered", values.delivered, 0],
    ["all", "undelivered", values.undelivered, values.delivered],
    ["delivered", "read", values.read, 0],
    ["delivered", "unread", values.unread, values.read],
    ["read", "replied", values.replied, 0],
    ["read", "noreply", values.noreply, values.replied],
  ];

  let body = "";
  for (const [source, target, value, offset] of links) {
    if (!value) continue;
    const path = _linkPath(nodes[source], nodes[target], value, offset, k);
    const cls = `board-flow-link board-flow-link-${nodes[target].cls}${stage === target ? " is-selected" : ""}`;
    body += `<path class="${cls}" data-flow-stage="${target}" d="${path}"><title>${escapeHtml(`${labels[source]} → ${labels[target]}: ${value}`)}</title></path>`;
  }

  for (const [key, node] of Object.entries(nodes)) {
    const value = values[key];
    const height = Math.max(node.h, 2);
    const parent = parentOf[key];
    const parentValue = values[parent];
    const percent = key === "all" || !parentValue ? "" : `${Math.round((value / parentValue) * 100)}%`;
    const labelY = node.y + Math.min(height / 2, 18) + 4;
    const stageValue = key === "all" ? "" : key;
    body += `<g class="board-flow-node${stage === key ? " is-selected" : ""}" data-flow-stage="${stageValue}" tabindex="0" role="button" aria-label="${escapeAttr(`${labels[key]} ${value}`)}">
      <rect class="board-flow-node-rect board-flow-node-${node.cls}" x="${node.x}" y="${node.y.toFixed(1)}" width="${NW}" height="${height.toFixed(1)}" rx="3"></rect>
      <text class="board-flow-node-label" x="${node.x + NW + 6}" y="${(labelY - 6).toFixed(1)}">${escapeHtml(labels[key])}</text>
      <text class="board-flow-node-value" x="${node.x + NW + 6}" y="${(labelY + 9).toFixed(1)}">${value}${percent ? `<tspan class="board-flow-node-percent" dx="6">${percent}</tspan>` : ""}</text>
    </g>`;
  }

  return `<svg class="board-flow-sankey${stage ? " has-selection" : ""}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeAttr(t("board.flows_sankey_label"))}">${body}</svg>`;
}

function _personStats(calls) {
  const stats = new Map();
  const touch = (name) => {
    if (!stats.has(name)) stats.set(name, { n: 0, waiting: 0 });
    return stats.get(name);
  };
  const now = Date.now();
  for (const call of calls) {
    touch(call.from).n += 1;
    const to = touch(call.to);
    to.n += 1;
    if (call.delivered && !call.reply) {
      const start = new Date(call.t).getTime();
      if (!Number.isNaN(start) && now - start >= 0 && now - start <= 2 * DAY_MS) {
        to.waiting += 1;
      }
    }
  }
  return stats;
}

export function mapMarkup(calls, { person = "", isHuman = () => false } = {}) {
  if (calls.length === 0) return _empty();
  const stats = _personStats(calls);
  const people = [...stats.keys()].sort((a, b) => {
    const diff = stats.get(b).n - stats.get(a).n;
    return diff || a.localeCompare(b);
  });
  const W = 560;
  const H = 400;
  const cx = 280;
  const cy = 205;
  const R = 140;
  const pos = new Map();
  people.forEach((name, index) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / Math.max(1, people.length);
    pos.set(name, { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle), angle });
  });

  const edges = new Map();
  for (const call of calls) {
    const key = `${call.from}|${call.to}`;
    if (!edges.has(key)) edges.set(key, { n: 0, dm: 0 });
    const edge = edges.get(key);
    edge.n += 1;
    if (call.via === "dm") edge.dm += 1;
  }

  const maxEdge = Math.max(1, ...[...edges.values()].map((edge) => edge.n));
  const maxNode = Math.max(1, ...[...stats.values()].map((stat) => stat.n));
  const radius = (name) => 10 + (9 * (stats.get(name)?.n || 0)) / maxNode;

  let edgeMarkup = "";
  const sortedEdges = [...edges.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  for (const [key, edge] of sortedEdges) {
    const [from, to] = key.split("|");
    if (!pos.has(from) || !pos.has(to)) continue;
    if (person && from !== person && to !== person) continue;
    const a = pos.get(from);
    const b = pos.get(to);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length;
    const ny = dx / length;
    const offset = 16 + length * 0.08;
    const qx = (a.x + b.x) / 2 + nx * offset;
    const qy = (a.y + b.y) / 2 + ny * offset;
    const unit = (px, py, r) => {
      const l = Math.hypot(px, py) || 1;
      return [px / l * r, py / l * r];
    };
    const [sx, sy] = unit(qx - a.x, qy - a.y, radius(from) + 2);
    const [ex, ey] = unit(b.x - qx, b.y - qy, radius(to) + 5);
    const dm = edge.dm > edge.n / 2;
    const human = isHuman(from) || isHuman(to);
    const focused = Boolean(person) && (from === person || to === person);
    const width = 1.2 + (6 * edge.n) / maxEdge;
    const arrow = human ? "board-flow-arrow-human" : "board-flow-arrow";
    const classes = [
      "board-flow-map-edge",
      dm ? "is-dm" : "",
      human ? "is-human" : "",
      focused ? "is-focus" : "",
    ].filter(Boolean).join(" ");
    const breakdown = `${t("board.flows_legend_dm")} ${edge.dm} · ${t("board.flows_legend_board")} ${edge.n - edge.dm}`;
    edgeMarkup += `<path class="${classes}" d="M${(a.x + sx).toFixed(1)} ${(a.y + sy).toFixed(1)} Q${qx.toFixed(1)} ${qy.toFixed(1)} ${(b.x - ex).toFixed(1)} ${(b.y - ey).toFixed(1)}" stroke-width="${width.toFixed(1)}" marker-end="url(#${arrow})"><title>${escapeHtml(`${from} → ${to}: ${edge.n} (${breakdown})`)}</title></path>`;
  }

  let nodeMarkup = "";
  for (const name of people) {
    const { x, y, angle } = pos.get(name);
    const r = radius(name);
    const human = isHuman(name);
    const focused = person === name;
    const connected = person && person !== name && calls.some(
      (call) => (call.from === person && call.to === name) || (call.to === person && call.from === name),
    );
    const labelX = x + Math.cos(angle) * (r + 12);
    const labelY = y + Math.sin(angle) * (r + 12) + 4;
    const anchor = Math.abs(Math.cos(angle)) < 0.25 ? "middle" : (Math.cos(angle) > 0 ? "start" : "end");
    const shape = human
      ? `<rect class="board-flow-map-shape is-human" x="${(x - r).toFixed(1)}" y="${(y - r).toFixed(1)}" width="${(2 * r).toFixed(1)}" height="${(2 * r).toFixed(1)}" rx="${(r * 0.55).toFixed(1)}"></rect>
         <rect class="board-flow-map-ring is-human" x="${(x - r).toFixed(1)}" y="${(y - r).toFixed(1)}" width="${(2 * r).toFixed(1)}" height="${(2 * r).toFixed(1)}" rx="${(r * 0.55).toFixed(1)}"></rect>`
      : `<circle class="board-flow-map-shape" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}"></circle>
         <circle class="board-flow-map-ring" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}"></circle>`;
    const waiting = stats.get(name)?.waiting || 0;
    const badge = waiting
      ? `<circle class="board-flow-map-wait" cx="${(x + r * 0.75).toFixed(1)}" cy="${(y - r * 0.75).toFixed(1)}" r="8"></circle><text class="board-flow-map-wait-text" x="${(x + r * 0.75).toFixed(1)}" y="${(y - r * 0.75 + 3.3).toFixed(1)}" text-anchor="middle">${waiting}</text>`
      : "";
    const classes = ["board-flow-map-node", focused ? "is-focus" : "", connected ? "is-connected" : ""]
      .filter(Boolean).join(" ");
    nodeMarkup += `<g class="${classes}" data-flow-person="${escapeAttr(name)}" tabindex="0" role="button" aria-label="${escapeAttr(t("board.flows_select_person", { name }))}">
      ${shape}${badge}<text class="board-flow-map-label" x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" text-anchor="${anchor}">${escapeHtml(name)}</text>
    </g>`;
  }

  return `<svg class="board-flow-map${person ? " has-selection" : ""}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeAttr(t("board.flows_map_label"))}">
    <defs>
      <marker id="board-flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path class="board-flow-arrow-head" d="M0 0L10 5L0 10z"></path></marker>
      <marker id="board-flow-arrow-human" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path class="board-flow-arrow-head is-human" d="M0 0L10 5L0 10z"></path></marker>
    </defs>
    ${edgeMarkup}${nodeMarkup}</svg>`;
}

function _dayBounds(dayKey) {
  const [year, month, day] = String(dayKey).split("-").map(Number);
  const start = new Date(year, (month || 1) - 1, day || 1).getTime();
  const end = new Date(year, (month || 1) - 1, (day || 1) + 1).getTime();
  return { start, end };
}

export function lanesMarkup(calls, { dayKey = "", selectedId = "" } = {}) {
  const { start, end } = _dayBounds(dayKey);
  const dayCalls = calls.filter((call) => {
    const ts = new Date(call.t).getTime();
    return !Number.isNaN(ts) && ts >= start && ts < end;
  });
  if (dayCalls.length === 0) return _empty();

  const names = new Set();
  for (const call of dayCalls) {
    names.add(call.from);
    names.add(call.to);
  }
  const lanes = [...names].sort((a, b) => a.localeCompare(b));
  const today = localDayKey(new Date());
  const times = [];
  for (const call of dayCalls) {
    for (const value of [call.t, call.read, call.reply]) {
      if (!value) continue;
      const ts = new Date(value).getTime();
      if (!Number.isNaN(ts)) times.push(ts);
    }
  }
  if (dayKey === today) times.push(Date.now());
  let t0 = times.length ? Math.min(...times) - 30 * 60_000 : start + 8 * 3_600_000;
  let t1 = times.length ? Math.max(...times) + 10 * 60_000 : start + 18 * 3_600_000;
  t0 = Math.floor(t0 / 3_600_000) * 3_600_000;
  t1 = Math.ceil(t1 / 3_600_000) * 3_600_000;
  if (t1 - t0 < 3 * 3_600_000) t1 = t0 + 3 * 3_600_000;

  const W = 900;
  const L = 78;
  const top = 22;
  const LH = 30;
  const H = top + lanes.length * LH + 8;
  const X = (ts) => L + ((ts - t0) / (t1 - t0)) * (W - L - 16);
  const Y = (name) => top + lanes.indexOf(name) * LH + LH / 2;

  let body = "";
  lanes.forEach((name, index) => {
    body += `<rect class="board-flow-lane${index % 2 ? " is-alt" : ""}" x="0" y="${top + index * LH}" width="${W}" height="${LH}"></rect>
      <text class="board-flow-lane-label" x="10" y="${top + index * LH + LH / 2 + 4}">${escapeHtml(name)}</text>`;
  });

  const step = t1 - t0 > 8 * 3_600_000 ? 2 * 3_600_000 : 3_600_000;
  for (let ts = t0; ts <= t1; ts += step) {
    body += `<line class="board-flow-tick" x1="${X(ts).toFixed(1)}" x2="${X(ts).toFixed(1)}" y1="${top}" y2="${H - 8}"></line>
      <text class="board-flow-tick-label" x="${Math.min(X(ts), W - 18).toFixed(1)}" y="14" text-anchor="middle">${escapeHtml(clockTime(ts))}</text>`;
  }
  if (dayKey === today) {
    const now = Date.now();
    if (now >= t0 && now <= t1) {
      const nx = X(now);
      body += `<line class="board-flow-now" x1="${nx.toFixed(1)}" x2="${nx.toFixed(1)}" y1="${top}" y2="${H - 8}"></line>
        <text class="board-flow-now-label" x="${(nx - 4).toFixed(1)}" y="${top - 4}" text-anchor="end">${escapeHtml(t("board.flows_now"))}</text>`;
    }
  }

  for (const call of dayCalls) {
    if (!lanes.includes(call.from) || !lanes.includes(call.to)) continue;
    const sent = new Date(call.t).getTime();
    if (Number.isNaN(sent)) continue;
    const x1 = X(sent);
    const y1 = Y(call.from);
    const y2 = Y(call.to);
    const read = call.read ? new Date(call.read).getTime() : 0;
    const endTs = read && !Number.isNaN(read) ? read : Math.min(sent + 6 * 60_000, t1);
    const x2 = X(Math.max(sent, endTs));
    const dm = call.via === "dm";
    const classes = [
      "board-flow-call",
      dm ? "is-dm" : "",
      call.delivered ? "" : "is-undelivered",
      call.id === selectedId ? "is-selected" : "",
    ].filter(Boolean).join(" ");
    const path = `M${x1.toFixed(1)} ${y1} C${(x1 + 12).toFixed(1)} ${y1} ${(x2 - 12).toFixed(1)} ${y2} ${x2.toFixed(1)} ${y2}`;
    let group = `<g class="${classes}" data-flow-call="${escapeAttr(call.id)}" tabindex="0" role="button" aria-label="${escapeAttr(`${clockTime(call.t)} ${call.from} → ${call.to}`)}">
      <path class="board-flow-call-hit" d="${path}"></path>
      <path class="board-flow-call-line" d="${path}"><title>${escapeHtml(`${clockTime(call.t)} ${call.from} → ${call.to} (${dm ? "DM" : `#${call.place || ""}`})`)}</title></path>
      <circle class="board-flow-call-dot" cx="${x1.toFixed(1)}" cy="${y1}" r="3"></circle>`;
    if (!call.delivered) {
      group += `<path class="board-flow-call-x" d="M${(x2 - 4).toFixed(1)} ${y2 - 4}l8 8M${(x2 + 4).toFixed(1)} ${y2 - 4}l-8 8"></path>`;
    } else if (!call.read) {
      group += `<circle class="board-flow-call-unread" cx="${x2.toFixed(1)}" cy="${y2}" r="4"></circle>`;
    }
    if (call.reply) {
      const replyTs = new Date(call.reply).getTime();
      if (!Number.isNaN(replyTs)) {
        const xr = X(replyTs);
        const replyPath = `M${xr.toFixed(1)} ${y2} C${(xr + 7).toFixed(1)} ${y2} ${(xr + 7).toFixed(1)} ${y1} ${(xr + 14).toFixed(1)} ${y1}`;
        group += `<path class="board-flow-call-reply-hit" d="${replyPath}"></path>
          <path class="board-flow-call-reply${dm ? " is-dm" : ""}" d="${replyPath}"><title>${escapeHtml(t("board.flows_reply_title", { time: clockTime(call.reply), from: call.to, to: call.from }))}</title></path>`;
      }
    }
    body += `${group}</g>`;
  }

  return `<div class="board-flow-svgwrap"><svg class="board-flow-lanes${selectedId ? " has-selection" : ""}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeAttr(t("board.flows_lanes_label"))}">${body}</svg></div>`;
}
