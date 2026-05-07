/**
 * Organization dashboard — Canvas-based node graph layout.
 *
 * Each Anima is a draggable card on an absolute-positioned canvas.
 * Hierarchy connections are drawn as SVG cubic-bezier curves.
 * Initial placement uses a tree-layout algorithm; dragged positions
 * are persisted to localStorage under `aw-org-positions`.
 */
import { createLogger } from "../../shared/logger.js";
import { basePath } from "/shared/base-path.js";
import { escapeHtml } from "./utils.js";
import { getState } from "./state.js";
import { animaHashColor } from "../../shared/avatar-utils.js";
import {
  bustupCandidates,
  bustupExpressionCandidates,
  resolveCachedAvatar,
} from "../../modules/avatar-resolver.js";
import { ReplayEngine } from "./replay-engine.js";
import { ReplayUI } from "./replay-ui.js";

const logger = createLogger("org-dashboard");

// ── Constants ──────────────────────

const CARD_W = 280;
const CARD_H = 100;
const GAP_X = 60;
const GAP_Y = 120;
const GRID_SNAP = 20;
const STORAGE_KEY = "aw-org-positions";

function _snapToGrid(x, y) {
  return {
    x: Math.round(x / GRID_SNAP) * GRID_SNAP,
    y: Math.round(y / GRID_SNAP) * GRID_SNAP,
  };
}

// ── Module State ──────────────────────

let _container = null;
let _viewport = null;
let _svgLayer = null;
let _nodesLayer = null;
let _kpiBar = null;
let _onNodeClick = null;

const _positions = new Map();
const _nodeData = new Map();
const _cardEls = new Map();
const _cardStreams = new Map();
let _draggingCard = null;
let _didDrag = false;
let _resizeRafId = null;

let _panActive = false;
let _panStartX = 0;
let _panStartY = 0;
let _panScrollLeft = 0;
let _panScrollTop = 0;

// ── Message Line State ──────────────────────
let _connectionsGroup = null;
let _msgLinesGroup = null;
let _msgLineCounter = 0;

// ── Replay State ──────────────────────
let _replayEngine = null;
let _replayUI = null;
let _replayMode = false;

// ── External Channel Nodes ──────────────────────
const _EXTERNAL_CHANNELS = [
  { id: "ext_slack", name: "Slack", icon: "💬", color: "#4A154B" },
  { id: "ext_chatwork", name: "Chatwork", icon: "💭", color: "#D4371C" },
  { id: "ext_github", name: "GitHub", icon: "🐙", color: "#24292E" },
  { id: "ext_gmail", name: "Gmail", icon: "📧", color: "#EA4335" },
  { id: "ext_web", name: "Web", icon: "🌐", color: "#4285F4" },
];

// ── Card Stream Visibility Filter ──────────────────────
const _CARD_VISIBLE_TYPES = new Set([
  "heartbeat_start", "heartbeat_end",
  "cron_executed",
  "message_sent", "message_received",
  "response_sent",
  "channel_post",
  "task_created", "task_updated", "issue_resolved",
  "inbox_processing_start", "inbox_processing_end",
  "human_notify",
]);

// Keep in sync with core/memory/activity.py _VISIBLE_TOOL_NAMES
export const VISIBLE_TOOL_NAMES = new Set([
  "delegate_task", "update_task", "backlog_task",
  "submit_tasks", "call_human",
  "post_channel", "send_message",
]);

function _isCardVisible(ev) {
  const type = ev.type || ev.name || "";
  if (_CARD_VISIBLE_TYPES.has(type)) return true;
  if (type === "tool_use" && VISIBLE_TOOL_NAMES.has(ev.tool)) return true;
  return false;
}

// ── Avatar Variant State ──────────────────────
const MESSAGE_LINE_DURATION = 2000;
const MESSAGE_LINE_FADE = 500;

const _avatarExpressions = new Map();
const EXPRESSIONS = ["neutral", "smile", "laugh", "troubled", "surprised", "thinking", "embarrassed"];

const STATUS_TO_EXPRESSION = {
  idle: "neutral",
  thinking: "thinking",
  working: "thinking",
  chatting: "smile",
  talking: "smile",
  error: "troubled",
  sleeping: "neutral",
  bootstrapping: "thinking",
  heartbeat: "smile",
  reporting: "smile",
};

// ── Org Tree Builder ──────────────────────

function buildOrgTree(animas) {
  const nodeMap = new Map();
  for (const p of animas) {
    nodeMap.set(p.name, {
      name: p.name,
      role: p.role || null,
      speciality: p.speciality || null,
      supervisor: p.supervisor || null,
      status: p.status,
      department: p.department || "",
      title: p.title || "",
      model: p.model || "",
      children: [],
    });
  }
  const roots = [];
  for (const node of nodeMap.values()) {
    if (node.role === "commander" || !node.supervisor || !nodeMap.has(node.supervisor)) {
      roots.push(node);
    } else {
      const parent = nodeMap.get(node.supervisor);
      if (parent) parent.children.push(node);
    }
  }
  return { roots: roots.length ? roots : [...nodeMap.values()], nodeMap };
}

// ── Status Helpers ──────────────────────

function getStatusDotClass(status) {
  if (!status) return "dot-unknown";
  const s = typeof status === "object" ? (status.state || status.status || "") : String(status);
  const lower = s.toLowerCase();
  if (lower === "idle") return "dot-idle";
  if (lower === "running") return "dot-running";
  if (lower === "thinking" || lower === "working") return "dot-active";
  if (lower === "sleeping" || lower === "stopped" || lower === "not_found") return "dot-sleeping";
  if (lower.includes("error")) return "dot-error";
  if (lower.includes("bootstrap")) return "dot-bootstrap";
  return "dot-unknown";
}

function getStatusLabel(status) {
  if (!status) return "unknown";
  const s = typeof status === "object" ? (status.state || status.status || "unknown") : String(status);
  const lower = s.toLowerCase();
  if (lower === "running") return "Running";
  return lower;
}

function getStatusAttr(status) {
  if (!status) return "idle";
  const s = typeof status === "object" ? (status.state || status.status || "") : String(status);
  const lower = s.toLowerCase();
  if (lower.includes("error")) return "error";
  if (lower.includes("bootstrap")) return "bootstrapping";
  if (lower === "thinking" || lower === "working") return "working";
  if (lower === "running") return "running";
  if (lower.includes("chat") || lower.includes("talk")) return "chatting";
  return "idle";
}

function _syncCardSpinner(name) {
  const card = _cardEls.get(name);
  if (!card) return;
  const entries = _cardStreams.get(name) || [];
  const hasRunning = entries.some(e => e.status === "running");
  if (hasRunning) {
    card.dataset.status = "working";
  } else if (card.dataset.status === "working") {
    const node = _nodeData.get(name);
    card.dataset.status = node ? getStatusAttr(node.status) : "idle";
  }
}

// ── Tree Layout Algorithm ──────────────────────

function _computeTreeLayout(roots, viewportWidth) {
  const positions = new Map();

  function measure(node) {
    if (!node.children.length) return { w: CARD_W, h: CARD_H, node };
    const childMeasures = node.children.map(measure);
    const totalChildW = childMeasures.reduce((s, m) => s + m.w, 0)
      + GAP_X * (childMeasures.length - 1);
    return {
      w: Math.max(CARD_W, totalChildW),
      h: CARD_H + GAP_Y + Math.max(...childMeasures.map(m => m.h)),
      node,
      children: childMeasures,
    };
  }

  function layout(measured, x, y) {
    const cx = x + measured.w / 2 - CARD_W / 2;
    const snapped = _snapToGrid(cx, y);
    positions.set(measured.node.name, snapped);
    if (!measured.children) return;
    let childX = x;
    for (const child of measured.children) {
      layout(child, childX, y + CARD_H + GAP_Y);
      childX += child.w + GAP_X;
    }
  }

  const measured = roots.map(measure);
  const totalW = measured.reduce((s, m) => s + m.w, 0)
    + GAP_X * (measured.length - 1);
  let startX = Math.max(40, (viewportWidth - totalW) / 2);
  for (const m of measured) {
    layout(m, startX, 40);
    startX += m.w + GAP_X;
  }
  return positions;
}

// ── localStorage Persistence ──────────────────────

function _isValidPos(v) {
  return v && typeof v === "object" && typeof v.x === "number" && typeof v.y === "number"
    && Number.isFinite(v.x) && Number.isFinite(v.y);
}

function _loadPositions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const map = new Map();
    for (const [k, v] of Object.entries(parsed)) {
      if (_isValidPos(v)) map.set(k, { x: v.x, y: v.y });
    }
    return map.size ? map : null;
  } catch {
    return null;
  }
}

function _persistPositions() {
  try {
    const obj = {};
    for (const [k, v] of _positions) obj[k] = v;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
  } catch { /* quota exceeded — ignore */ }
}

// ── SVG Connections ──────────────────────

function _getCardDimensions() {
  const sample = _cardEls.values().next().value;
  if (sample) {
    const w = sample.offsetWidth;
    const h = sample.offsetHeight;
    if (w > 0 && h > 0) return { w, h };
  }
  return { w: CARD_W, h: CARD_H };
}

function _updateConnections() {
  if (!_svgLayer) return;

  if (!_connectionsGroup) {
    _connectionsGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    _connectionsGroup.setAttribute("class", "org-connections-group");
    _svgLayer.prepend(_connectionsGroup);
  }
  _connectionsGroup.innerHTML = "";

  if (!_msgLinesGroup) {
    _msgLinesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    _msgLinesGroup.setAttribute("class", "org-msg-lines-group");
    _svgLayer.appendChild(_msgLinesGroup);
  }

  const { w: cardW, h: cardH } = _getCardDimensions();

  for (const [name, node] of _nodeData) {
    if (!node.supervisor) continue;
    const parentPos = _positions.get(node.supervisor);
    const childPos = _positions.get(name);
    if (!parentPos || !childPos) continue;

    const x1 = parentPos.x + cardW / 2;
    const y1 = parentPos.y + cardH;
    const x2 = childPos.x + cardW / 2;
    const y2 = childPos.y;
    const midY = (y1 + y2) / 2;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}`);
    path.setAttribute("class", "org-connection-line");
    _connectionsGroup.appendChild(path);
  }
}

// ── Drag Implementation ──────────────────────

function _setupDrag(cardEl, name) {
  let dragging = false;
  let moved = false;
  let startX, startY, cardStartX, cardStartY;

  cardEl.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (e.target.closest(".org-card-expand")) return;
    if (e.target.closest(".org-card-stream")) return;
    if (e.target.closest(".org-card-detail")) return;
    dragging = true;
    moved = false;
    _draggingCard = name;
    cardEl.setPointerCapture(e.pointerId);
    startX = e.clientX;
    startY = e.clientY;

    const pos = _positions.get(name);
    if (pos) {
      cardStartX = pos.x;
      cardStartY = pos.y;
    } else {
      cardStartX = parseInt(cardEl.style.left, 10) || 0;
      cardStartY = parseInt(cardEl.style.top, 10) || 0;
    }
    cardEl.classList.add("org-card--dragging");
    e.preventDefault();
    e.stopPropagation();
  });

  cardEl.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
    const x = cardStartX + dx;
    const y = cardStartY + dy;
    cardEl.style.left = `${x}px`;
    cardEl.style.top = `${y}px`;
    _positions.set(name, { x, y });
    requestAnimationFrame(() => _updateConnections());
  });

  const _applySnap = () => {
    const cur = _positions.get(name);
    if (!cur) return;
    const snapped = _snapToGrid(cur.x, cur.y);
    if (snapped.x === cur.x && snapped.y === cur.y) return;
    _positions.set(name, snapped);
    cardEl.classList.add("org-card--snapping");
    cardEl.style.left = `${snapped.x}px`;
    cardEl.style.top = `${snapped.y}px`;
    requestAnimationFrame(() => _updateConnections());
    const onEnd = () => {
      cardEl.classList.remove("org-card--snapping");
      cardEl.removeEventListener("transitionend", onEnd);
    };
    cardEl.addEventListener("transitionend", onEnd, { once: true });
    setTimeout(onEnd, 100);
  };

  cardEl.addEventListener("pointerup", () => {
    if (!dragging) return;
    dragging = false;
    if (moved) _didDrag = true;
    _draggingCard = null;
    cardEl.classList.remove("org-card--dragging");
    _applySnap();
    _persistPositions();
  });

  cardEl.addEventListener("pointercancel", () => {
    if (!dragging) return;
    dragging = false;
    _draggingCard = null;
    cardEl.classList.remove("org-card--dragging");
    _applySnap();
    _persistPositions();
  });
}

// ── Canvas Pan ──────────────────────

function _setupPan(viewport) {
  viewport.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".org-card")) return;
    if (e.button !== 0) return;
    _panActive = true;
    _panStartX = e.clientX;
    _panStartY = e.clientY;
    _panScrollLeft = viewport.scrollLeft;
    _panScrollTop = viewport.scrollTop;
    viewport.style.cursor = "grabbing";
    e.preventDefault();
  });

  viewport.addEventListener("pointermove", (e) => {
    if (!_panActive) return;
    const dx = e.clientX - _panStartX;
    const dy = e.clientY - _panStartY;
    viewport.scrollLeft = _panScrollLeft - dx;
    viewport.scrollTop = _panScrollTop - dy;
  });

  const endPan = () => {
    if (!_panActive) return;
    _panActive = false;
    viewport.style.cursor = "";
  };
  viewport.addEventListener("pointerup", endPan);
  viewport.addEventListener("pointercancel", endPan);
}

// ── Card Rendering ──────────────────────

function _createCardEl(node) {
  const statusDot = getStatusDotClass(node.status);
  const statusLabel = getStatusLabel(node.status);
  const statusAttr = getStatusAttr(node.status);
  const initial = (node.name || "?")[0].toUpperCase();
  const color = animaHashColor(node.name);
  const roleLabel = node.role || "";
  const specLabel = node.speciality || "";
  const tagHtml = roleLabel || specLabel
    ? `<span class="org-card-tags">${roleLabel ? `<span class="org-card-role">${escapeHtml(roleLabel)}</span>` : ""}${specLabel ? `<span class="org-card-spec">${escapeHtml(specLabel)}</span>` : ""}</span>`
    : "";

  // Department / Title / Model meta line
  const dept = node.department || "";
  const ttl = node.title || "";
  const mdl = node.model || "";
  const metaParts = [];
  if (dept) metaParts.push(`<span class="org-card-dept">${escapeHtml(dept)}</span>`);
  if (ttl) metaParts.push(`<span class="org-card-title">${escapeHtml(ttl)}</span>`);
  if (mdl) metaParts.push(`<span class="org-card-model">${escapeHtml(mdl)}</span>`);
  const metaHtml = metaParts.length
    ? `<div class="org-card-meta">${metaParts.join('<span class="org-card-meta-sep">·</span>')}</div>`
    : "";

  const card = document.createElement("div");
  card.className = "org-card";
  card.dataset.name = node.name;
  card.dataset.status = statusAttr;
  card.id = `orgCard_${node.name}`;
  card.innerHTML = `
    <div class="org-card-header">
      <div class="org-card-avatar" style="background:${color}" data-anima="${escapeHtml(node.name)}">${initial}</div>
      <div class="org-card-info">
        <span class="org-card-name">${escapeHtml(node.name)}</span>
        ${tagHtml}
        ${metaHtml}
      </div>
      <span class="org-card-status">
        <span class="org-card-dot ${statusDot}"></span>
        <span class="org-card-status-label">${escapeHtml(statusLabel)}</span>
      </span>
    </div>
    <div class="org-card-stream" id="orgStream_${CSS.escape(node.name)}">
      <div class="org-stream-idle">\u{1F4A4} idle</div>
    </div>
  `;
  return card;
}

// ── KPI Bar ──────────────────────

function _renderKpiBar() {
  if (!_kpiBar) return;
  const animas = getState().animas || [];
  const active = animas.filter(a => {
    const s = (typeof a.status === "object" ? a.status.state : a.status) || "";
    return !["idle", "sleeping", "stopped", "not_found", "disabled"].includes(String(s).toLowerCase());
  }).length;
  const total = animas.length;
  const errors = animas.filter(a => {
    const st = typeof a.status === "object" ? (a.status?.state ?? a.status?.status) : a.status;
    return String(st || "").toLowerCase().includes("error");
  }).length;

  _kpiBar.innerHTML = `
    <div class="org-kpi-card">
      <span class="org-kpi-value">${active}</span>
      <span class="org-kpi-label">Active / ${total}</span>
    </div>
    <div class="org-kpi-card">
      <span class="org-kpi-value" id="orgKpiEventsH">-</span>
      <span class="org-kpi-label">events/h</span>
    </div>
    <div class="org-kpi-card">
      <span class="org-kpi-value" id="orgKpiTasks">-</span>
      <span class="org-kpi-label">Tasks</span>
    </div>
    <div class="org-kpi-card org-kpi-card--error" style="display:${errors > 0 ? "flex" : "none"}">
      <span class="org-kpi-value">${errors}</span>
      <span class="org-kpi-label">Errors</span>
    </div>
  `;
  _applyKpiValues();
}

// ── SVG Sizing ──────────────────────

function _resizeSvg() {
  if (!_svgLayer || !_nodesLayer) return;
  const rect = _nodesLayer.getBoundingClientRect();
  const { w: cw, h: ch } = _getCardDimensions();
  let maxX = 0;
  let maxY = 0;
  for (const pos of _positions.values()) {
    if (!pos) continue;
    maxX = Math.max(maxX, pos.x + cw + 40);
    maxY = Math.max(maxY, pos.y + ch + 40);
  }
  maxX = Math.max(maxX, rect.width);
  maxY = Math.max(maxY, rect.height);
  _svgLayer.setAttribute("width", maxX);
  _svgLayer.setAttribute("height", maxY);
  _svgLayer.style.width = `${maxX}px`;
  _svgLayer.style.height = `${maxY}px`;
  _nodesLayer.style.width = `${maxX}px`;
  _nodesLayer.style.height = `${maxY}px`;
}

// ── Message Line Drawing ──────────────────────

export function showMessageLine(fromName, toName, _summary, options = {}) {
  if (!_svgLayer || !_msgLinesGroup) return;
  const fromPos = _positions.get(fromName);
  const toPos = _positions.get(toName);
  if (!fromPos || !toPos) return;

  const lineType = options.lineType || "internal";
  const replaySpeed = options.replaySpeed || 0;

  // 50x+: particle-only mode (skip bezier trail)
  const simplified = replaySpeed >= 50;

  const { w: cardW, h: cardH } = _getCardDimensions();
  const x1 = fromPos.x + cardW / 2;
  const y1 = fromPos.y + cardH / 2;
  const x2 = toPos.x + cardW / 2;
  const y2 = toPos.y + cardH / 2;

  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return;

  const offsetScale = Math.min(len * 0.3, 80);
  const sign = (++_msgLineCounter % 2 === 0) ? 1 : -1;
  const jitter = (Math.random() - 0.5) * 20;
  const nx = (-dy / len) * (offsetScale + jitter) * sign;
  const ny = (dx / len) * (offsetScale + jitter) * sign;
  const cx1 = (x1 + x2) / 2 + nx;
  const cy1 = (y1 + y2) / 2 + ny;

  const pathD = `M${x1},${y1} Q${cx1},${cy1} ${x2},${y2}`;

  const dur = replaySpeed >= 100 ? 200 : replaySpeed >= 50 ? 500 : replaySpeed >= 10 ? 1000 : MESSAGE_LINE_DURATION;
  const fade = replaySpeed >= 50 ? 200 : MESSAGE_LINE_FADE;

  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.classList.add("org-msg-line-group");

  if (!simplified) {
    const trail = document.createElementNS("http://www.w3.org/2000/svg", "path");
    trail.setAttribute("d", pathD);
    trail.setAttribute("class", `org-msg-trail org-msg-trail--${lineType}`);
    group.appendChild(trail);
  }

  const packet = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  packet.setAttribute("r", lineType === "internal" ? "5" : "6");
  packet.setAttribute("class", `org-msg-packet org-msg-packet--${lineType}`);

  const anim = document.createElementNS("http://www.w3.org/2000/svg", "animateMotion");
  anim.setAttribute("dur", `${dur}ms`);
  anim.setAttribute("repeatCount", "1");
  anim.setAttribute("fill", "freeze");
  anim.setAttribute("path", pathD);
  packet.appendChild(anim);
  group.appendChild(packet);

  _msgLinesGroup.appendChild(group);

  setTimeout(() => {
    group.classList.add("org-msg-line--fading");
    setTimeout(() => group.remove(), fade);
  }, dur);
}

function _createExternalNodes(container) {
  const bar = document.createElement("div");
  bar.className = "org-external-bar";
  bar.id = "orgExternalBar";

  for (const ch of _EXTERNAL_CHANNELS) {
    const node = document.createElement("div");
    node.className = "org-external-node";
    node.id = `orgExt_${ch.id}`;
    node.innerHTML = `
      <span class="org-external-icon" style="background:${ch.color}">${ch.icon}</span>
      <span class="org-external-label">${ch.name}</span>
    `;
    bar.appendChild(node);
    _positions.set(ch.id, null);
  }

  container.prepend(bar);
  return bar;
}

function _updateExternalPositions() {
  const bar = document.getElementById("orgExternalBar");
  if (!bar) return;
  const viewport = document.getElementById("orgCanvasViewport");
  if (!viewport) return;
  const vpRect = viewport.getBoundingClientRect();

  for (const ch of _EXTERNAL_CHANNELS) {
    const node = document.getElementById(`orgExt_${ch.id}`);
    if (node) {
      const rect = node.getBoundingClientRect();
      _positions.set(ch.id, {
        x: rect.left - vpRect.left + rect.width / 2,
        y: rect.top - vpRect.top + rect.height / 2,
      });
    }
  }
}

const _TOOL_TO_EXTERNAL = {
  slack: "ext_slack",
  chatwork: "ext_chatwork",
  github: "ext_github",
  gmail: "ext_gmail",
  web_search: "ext_web",
  x_search: "ext_web",
};

export function showExternalLine(animaName, toolName, direction = "out") {
  const extId = _TOOL_TO_EXTERNAL[String(toolName || "").toLowerCase()];
  if (!extId) return;

  if (direction === "out") {
    showMessageLine(animaName, extId, "", { lineType: "external" });
  } else {
    showMessageLine(extId, animaName, "", { lineType: "external_in" });
  }
}

// ── Avatar Loading & Variants ──────────────────────

async function _loadOrgAvatars(animas) {
  const candidates = bustupCandidates();
  for (const p of animas) {
    try {
      const url = await resolveCachedAvatar(p.name, candidates, "S");
      if (!url) continue;
      const el = _container?.querySelector(`.org-card-avatar[data-anima="${CSS.escape(p.name)}"]`);
      if (!el) continue;
      const img = new Image();
      img.src = url;
      img.alt = p.name;
      img.style.cssText = "width:100%;height:100%;object-fit:cover;border-radius:6px;";
      img.onload = () => { el.textContent = ""; el.appendChild(img); };
    } catch { /* skip */ }
  }
}

async function _preloadAvatarExpressions(animas) {
  for (const p of animas) {
    const exprs = {};
    for (const expr of EXPRESSIONS) {
      try {
        const candidates = bustupExpressionCandidates(expr);
        const url = await resolveCachedAvatar(p.name, candidates, "S");
        if (url) exprs[expr] = url;
      } catch { /* skip */ }
    }
    if (Object.keys(exprs).length > 0) {
      _avatarExpressions.set(p.name, exprs);
    }
  }
}

let _avatarUpdateRafPending = new Set();

export function updateAvatarExpression(name, status) {
  if (_avatarUpdateRafPending.has(name)) return;
  _avatarUpdateRafPending.add(name);

  requestAnimationFrame(() => {
    _avatarUpdateRafPending.delete(name);
    _applyAvatarExpression(name, status);
  });
}

function _applyAvatarExpression(name, status) {
  const exprs = _avatarExpressions.get(name);
  const avatarEl = _container?.querySelector(`.org-card-avatar[data-anima="${CSS.escape(name)}"] img`);
  if (!avatarEl) return;

  const expression = STATUS_TO_EXPRESSION[status] || "neutral";
  const url = exprs?.[expression] || exprs?.neutral;

  if (url && avatarEl.src !== url) {
    avatarEl.classList.add("org-avatar--transitioning");
    avatarEl.src = url;
    avatarEl.onload = () => avatarEl.classList.remove("org-avatar--transitioning");
  }

  const card = avatarEl.closest(".org-card");
  if (!exprs?.[expression] && expression !== "neutral") {
    if (card) card.dataset.expression = expression;
  } else {
    if (card) delete card.dataset.expression;
  }
}

// ── KPI Live Updates ──────────────────────

let _kpiTimerId = null;
let _kpiEventsH = "-";
let _kpiTasks = "-";

async function _loadKpiStats() {
  try {
    const resp = await fetch(`${basePath}/api/activity/recent?hours=1&limit=200`);
    if (resp.ok) {
      const data = await resp.json();
      _kpiEventsH = String((data.events ?? []).length);
    }
  } catch { /* ignore */ }
  try {
    const resp = await fetch(`${basePath}/api/tasks/summary`);
    if (resp.ok) {
      const data = await resp.json();
      _kpiTasks = String(data.total_active || 0);
    }
  } catch { /* ignore */ }
  _applyKpiValues();
}

function _applyKpiValues() {
  const el1 = document.getElementById("orgKpiEventsH");
  if (el1) el1.textContent = _kpiEventsH;
  const el2 = document.getElementById("orgKpiTasks");
  if (el2) el2.textContent = _kpiTasks;
}

function _startKpiPolling() {
  _stopKpiPolling();
  _loadKpiStats();
  _kpiTimerId = setInterval(_loadKpiStats, 60_000);
}

function _stopKpiPolling() {
  if (_kpiTimerId) { clearInterval(_kpiTimerId); _kpiTimerId = null; }
}

async function _loadInitialStreams(animas) {
  const [flatResults, activeGroups] = await Promise.all([
    Promise.allSettled(
      animas.map(async (a) => {
        const resp = await fetch(`${basePath}/api/activity/recent?hours=1&limit=5&anima=${encodeURIComponent(a.name)}`);
        if (!resp.ok) return null;
        const data = await resp.json();
        const events = data.events ?? [];
        if (!events.length) return null;
        return { name: a.name, events };
      })
    ),
    _fetchActiveGroups(),
  ]);

  for (const r of flatResults) {
    if (r.status !== "fulfilled" || !r.value) continue;
    const { name, events } = r.value;
    const entries = events.filter(_isCardVisible).slice(0, MAX_STREAM_ENTRIES).reverse().map(ev => ({
      id: ev.id || String(Date.now() + Math.random()),
      type: _mapEventType(ev.type || ev.name),
      text: _summarizeEvent(ev),
      status: "done",
      ts: (ev.ts || ev.timestamp) ? new Date(ev.ts || ev.timestamp).getTime() : Date.now(),
    }));
    _cardStreams.set(name, entries);
  }

  for (const grp of activeGroups) {
    const name = grp.anima;
    if (!name || !_cardEls.has(name)) continue;
    const entries = _cardStreams.get(name) || [];
    const grpType = _mapGroupType(grp.type);
    entries.push({
      id: grp.id || `active-${name}-${Date.now()}`,
      type: grpType,
      text: _summarizeGroup(grp),
      status: "running",
      ts: Date.now(),
    });
    if (entries.length > MAX_STREAM_ENTRIES * 2) {
      _cardStreams.set(name, entries.slice(-MAX_STREAM_ENTRIES));
    } else {
      _cardStreams.set(name, entries);
    }
  }

  for (const [name, entries] of _cardStreams) {
    const streamEl = document.getElementById(`orgStream_${CSS.escape(name)}`);
    if (streamEl) _renderStream(streamEl, entries.slice(-MAX_STREAM_ENTRIES));
    _syncCardSpinner(name);
  }

  if (activeGroups.length > 0) _ensureStaleTimer();
}

async function _fetchActiveGroups() {
  try {
    const resp = await fetch(`${basePath}/api/activity/recent?grouped=true&hours=1&group_limit=50`);
    if (!resp.ok) return [];
    const data = await resp.json();
    return (data.groups ?? []).filter(g => g.is_open);
  } catch {
    return [];
  }
}

function _mapGroupType(type) {
  if (!type) return "tool";
  if (type === "heartbeat") return "heartbeat";
  if (type === "cron") return "cron";
  return "tool";
}

function _summarizeGroup(grp) {
  if (grp.summary) return grp.summary.slice(0, 80);
  const type = grp.type || "";
  if (type === "heartbeat") return "Heartbeat";
  if (type === "cron") return "Cron";
  if (type === "task_exec") return `Task: ${grp.summary || ""}`.trim().slice(0, 80);
  return type || "activity";
}

function _mapEventType(type) {
  if (!type) return "tool";
  const t = type.toLowerCase();
  if (t.includes("heartbeat")) return "heartbeat";
  if (t.includes("cron")) return "cron";
  if (t.includes("channel") || t.includes("board")) return "board";
  if (t.includes("tool")) return "tool";
  return "tool";
}

function _summarizeEvent(ev) {
  if (ev.summary) return ev.summary.slice(0, 80);
  const type = ev.type || ev.name || "";
  if (type.includes("tool_use")) return `⚙ ${ev.tool || ev.tool_name || type}`;
  if (type.includes("heartbeat")) return "heartbeat";
  if (type.includes("cron")) return ev.task || "cron";
  if (type.includes("message")) return ev.intent ? `${type} (${ev.intent})` : type;
  return type.slice(0, 60) || "activity";
}

// ── Main API ──────────────────────

export async function initOrgDashboard(container, animas, { onNodeClick } = {}) {
  _container = container;
  _onNodeClick = onNodeClick || null;
  _cardEls.clear();
  _nodeData.clear();
  _positions.clear();

  const { roots, nodeMap } = buildOrgTree(animas);
  for (const [k, v] of nodeMap) _nodeData.set(k, v);

  container.innerHTML = `
    <div class="org-canvas-root">
      <div class="org-kpi-bar" id="orgKpiBar"></div>
      <div class="org-canvas-viewport" id="orgCanvasViewport">
        <svg class="org-canvas-svg" id="orgCanvasSvg"></svg>
        <div class="org-canvas-nodes" id="orgCanvasNodes"></div>
      </div>
    </div>
  `;

  _kpiBar = document.getElementById("orgKpiBar");
  _viewport = document.getElementById("orgCanvasViewport");
  _svgLayer = document.getElementById("orgCanvasSvg");
  _nodesLayer = document.getElementById("orgCanvasNodes");

  _createExternalNodes(container);
  _renderKpiBar();

  const vpWidth = _viewport.clientWidth || 1200;
  const computedPositions = _computeTreeLayout(roots, vpWidth);

  const saved = _loadPositions();
  for (const [name] of _nodeData) {
    if (saved && saved.has(name)) {
      _positions.set(name, saved.get(name));
    } else if (computedPositions.has(name)) {
      _positions.set(name, computedPositions.get(name));
    }
  }

  for (const [name, node] of _nodeData) {
    const card = _createCardEl(node);
    const pos = _positions.get(name) || { x: 0, y: 0 };
    card.style.left = `${pos.x}px`;
    card.style.top = `${pos.y}px`;
    _nodesLayer.appendChild(card);
    _cardEls.set(name, card);
    _setupDrag(card, name);
  }

  _resizeSvg();
  _updateConnections();
  _setupPan(_viewport);

  requestAnimationFrame(() => _updateExternalPositions());

  container.addEventListener("click", (e) => {
    if (_didDrag) { _didDrag = false; return; }

    const streamTarget = e.target.closest(".org-card-stream");
    if (streamTarget) {
      const card = streamTarget.closest(".org-card");
      if (card && card.dataset.name) {
        _toggleCardExpand(card.dataset.name);
      }
      return;
    }

    const card = e.target.closest(".org-card");
    if (!card) return;
    const name = card.dataset.name;
    if (!name) return;
    container.querySelectorAll(".org-card.selected").forEach(el => el.classList.remove("selected"));
    card.classList.add("selected");
    document.querySelectorAll(".org-card--expanded").forEach(el => {
      el.classList.remove("org-card--expanded");
      el.querySelector(".org-card-detail")?.remove();
    });
    if (_onNodeClick) _onNodeClick(name);
  });

  window.addEventListener("resize", _onResize);

  _loadOrgAvatars(animas);
  _startKpiPolling();
  _loadInitialStreams(animas);

  // Lazy-preload expression variants after neutral avatars are loaded
  requestIdleCallback(() => _preloadAvatarExpressions(animas), { timeout: 5000 });

  logger.info("Org dashboard initialized (canvas mode)", { animaCount: animas.length });
}

function _onResize() {
  if (_resizeRafId) cancelAnimationFrame(_resizeRafId);
  _resizeRafId = requestAnimationFrame(() => {
    _resizeRafId = null;
    _resizeSvg();
    _updateConnections();
    _updateExternalPositions();
  });
}

export function disposeOrgDashboard() {
  window.removeEventListener("resize", _onResize);
  _stopKpiPolling();
  if (_staleTimerId) { clearInterval(_staleTimerId); _staleTimerId = null; }
  if (_replayMode) { _replayMode = false; }
  _replayEngine?.dispose(); _replayEngine = null;
  _replayUI?.dispose(); _replayUI = null;
  if (_container) {
    _container.innerHTML = "";
  }
  _cardEls.clear();
  _cardStreams.clear();
  _nodeData.clear();
  _positions.clear();
  _avatarExpressions.clear();
  _avatarUpdateRafPending.clear();
  _container = null;
  _viewport = null;
  _svgLayer = null;
  _nodesLayer = null;
  _kpiBar = null;
  _onNodeClick = null;
  _draggingCard = null;
  _didDrag = false;
  _panActive = false;
  _connectionsGroup = null;
  _msgLinesGroup = null;
  _msgLineCounter = 0;
  if (_resizeRafId) { cancelAnimationFrame(_resizeRafId); _resizeRafId = null; }
}

export function updateAnimaStatus(name, status) {
  const node = _nodeData.get(name);
  if (node) node.status = status;

  const card = _cardEls.get(name);
  if (!card) return;

  const dotClass = getStatusDotClass(status);
  const label = getStatusLabel(status);

  const dot = card.querySelector(".org-card-dot");
  if (dot) dot.className = `org-card-dot ${dotClass}`;

  const labelEl = card.querySelector(".org-card-status-label");
  if (labelEl) labelEl.textContent = label;

  _syncCardSpinner(name);
  _renderKpiBar();
}

export function getCardPosition(name) {
  return _positions.get(name) || null;
}

const MAX_STREAM_ENTRIES = 4;
const STALE_TIMEOUT_MS = 600_000;
let _staleTimerId = null;

export function updateCardActivity(name, data) {
  let entries = _cardStreams.get(name) || [];
  const { eventType, toolName, toolId, isError, detail, channel, summary } = data;

  if (eventType === "message_sent") {
    entries.push({
      id: Date.now().toString(),
      type: "msg_out",
      text: `↑ ${data.to_person || ""}: ${(data.summary || data.content || "").slice(0, 60)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "message_received") {
    entries.push({
      id: Date.now().toString(),
      type: "msg_in",
      text: `↓ ${data.from_person || ""}: ${(data.summary || data.content || "").slice(0, 60)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "response_sent") {
    entries.push({
      id: Date.now().toString(),
      type: "msg_out",
      text: `↑ ${(data.summary || data.content || "").slice(0, 80)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "channel_post") {
    entries.push({
      id: Date.now().toString(),
      type: "board",
      text: `↑ #${data.channel || "?"}: ${(data.summary || data.content || "").slice(0, 60)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "task_created" || eventType === "task_updated") {
    entries.push({
      id: Date.now().toString(),
      type: "task",
      text: `⚙ ${(data.summary || eventType).slice(0, 80)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "tool_start") {
    if (!VISIBLE_TOOL_NAMES.has(toolName)) return;
    for (const e of entries) {
      if (e.type === "tool" && e.status === "running") e.status = "done";
    }
    entries.push({
      id: toolId || Date.now().toString(),
      type: "tool",
      text: toolName || "tool",
      status: "running",
      ts: Date.now(),
    });
  } else if (eventType === "tool_end" || eventType === "tool_use") {
    if (!VISIBLE_TOOL_NAMES.has(toolName)) return;
    const existing = toolId
      ? entries.find(e => e.id === toolId)
      : [...entries].reverse().find(e => e.type === "tool" && e.status === "running");
    if (existing) {
      existing.status = isError ? "error" : "done";
    }
  } else if (eventType === "tool_detail") {
    if (!VISIBLE_TOOL_NAMES.has(toolName)) return;
    const existing = toolId
      ? entries.find(e => e.id === toolId && e.status === "running")
      : [...entries].reverse().find(e => e.type === "tool" && e.status === "running");
    if (existing) {
      existing.text = `${toolName}: ${detail || ""}`.slice(0, 80);
    }
  } else if (eventType === "board_post") {
    entries.push({
      id: Date.now().toString(),
      type: "board",
      text: `#${channel}: ${(summary || "").slice(0, 60)}`,
      status: "done",
      ts: Date.now(),
    });
  } else if (eventType === "cron") {
    entries.push({
      id: Date.now().toString(),
      type: "cron",
      text: summary || "cron",
      status: "running",
      ts: Date.now(),
    });
  } else if (eventType === "heartbeat") {
    entries.push({
      id: Date.now().toString(),
      type: "heartbeat",
      text: "heartbeat",
      status: "running",
      ts: Date.now(),
    });
  } else if (eventType === "cron_end" || eventType === "heartbeat_end") {
    const last = [...entries].reverse().find(
      e => (e.type === "cron" || e.type === "heartbeat") && e.status === "running",
    );
    if (last) last.status = "done";
  }

  if (entries.length > MAX_STREAM_ENTRIES * 2) {
    entries = entries.slice(-MAX_STREAM_ENTRIES);
  }
  _cardStreams.set(name, entries);

  const streamEl = document.getElementById(`orgStream_${CSS.escape(name)}`);
  if (streamEl) _renderStream(streamEl, entries.slice(-MAX_STREAM_ENTRIES));
  _syncCardSpinner(name);
  _ensureStaleTimer();
}

function _renderStream(container, entries) {
  if (!entries.length) {
    container.innerHTML = '<div class="org-stream-idle">\u{1F4A4} idle</div>';
    return;
  }
  container.innerHTML = entries.map(e => {
    const typeIcon = { tool: "\u{1F527}", board: "\u{1F4CB}", cron: "\u23F0", heartbeat: "\u{1F49A}", msg_out: "", msg_in: "", task: "" }[e.type] ?? "\u{1F4CC}";
    const elapsed = e.status === "running" ? ` ${Math.round((Date.now() - e.ts) / 1000)}s` : "";
    const cls = `org-stream--${e.status}`;
    if (e.status === "running") {
      return `<div class="org-stream-entry ${cls}">
        <span class="org-stream-icon">${typeIcon}</span>
        <span class="org-stream-text">${escapeHtml(e.text)}${elapsed}</span>
        <span class="org-stream-spinner"></span>
      </div>`;
    }
    const statusIcon = e.status === "error" ? "\u2717" : "\u2713";
    return `<div class="org-stream-entry ${cls}">
      <span class="org-stream-icon">${typeIcon}</span>
      <span class="org-stream-text">${escapeHtml(e.text)}</span>
      <span class="org-stream-status">${statusIcon}</span>
    </div>`;
  }).join("");
}

function _ensureStaleTimer() {
  if (_staleTimerId) return;
  _staleTimerId = setInterval(() => {
    let hasRunning = false;
    const now = Date.now();
    for (const [name, entries] of _cardStreams) {
      let changed = false;
      let hadStaleExpiry = false;
      for (const e of entries) {
        if (e.status === "running") {
          if (now - e.ts > STALE_TIMEOUT_MS) {
            e.status = "done";
            e.text += " (timeout)";
            changed = true;
            hadStaleExpiry = true;
          } else {
            hasRunning = true;
            changed = true;
          }
        }
      }
      if (changed) {
        const streamEl = document.getElementById(`orgStream_${CSS.escape(name)}`);
        if (streamEl) _renderStream(streamEl, entries.slice(-MAX_STREAM_ENTRIES));
      }
      if (hadStaleExpiry) _syncCardSpinner(name);
    }
    if (!hasRunning) {
      clearInterval(_staleTimerId);
      _staleTimerId = null;
    }
  }, 1000);
}

function _toggleCardExpand(name) {
  const card = document.querySelector(`.org-card[data-name="${CSS.escape(name)}"]`);
  if (!card) return;

  if (card.classList.contains("org-card--expanded")) {
    card.classList.remove("org-card--expanded");
    const detail = card.querySelector(".org-card-detail");
    if (detail) detail.remove();
    _updateConnections();
    return;
  }

  document.querySelectorAll(".org-card--expanded").forEach(el => {
    el.classList.remove("org-card--expanded");
    el.querySelector(".org-card-detail")?.remove();
  });

  card.classList.add("org-card--expanded");

  const entries = _cardStreams.get(name) || [];
  const detail = document.createElement("div");
  detail.className = "org-card-detail";
  detail.innerHTML = `
    <div class="org-card-detail-header">\u76F4\u8FD1\u306E\u30A2\u30AF\u30C6\u30A3\u30D3\u30C6\u30A3</div>
    <div class="org-card-detail-list">
      ${entries.length === 0
        ? '<div class="org-detail-entry org-detail-entry--empty">\u30A2\u30AF\u30C6\u30A3\u30D3\u30C6\u30A3\u306A\u3057</div>'
        : entries.slice(-20).reverse().map(e => {
          const icon = { tool: "\u{1F527}", board: "\u{1F4CB}", cron: "\u23F0", heartbeat: "\u{1F49A}", msg_out: "", msg_in: "", task: "" }[e.type] ?? "\u{1F4CC}";
          const statusIcon = e.status === "running" ? "\u23F3" : e.status === "error" ? "\u2717" : "\u2713";
          return `<div class="org-detail-entry"><span>${icon}</span><span>${escapeHtml(e.text)}</span><span>${statusIcon}</span></div>`;
        }).join("")
      }
    </div>
  `;
  card.appendChild(detail);
  _updateConnections();
}

// ── Replay Mode API ──────────────────────

export function isReplayMode() {
  return _replayMode;
}

/**
 * Enter replay mode: load events and show replay controls.
 * @param {number} [hours=12]
 */
export async function startReplay(hours = 24) {
  if (_replayMode) return;
  _replayMode = true;

  const root = _container?.querySelector(".org-canvas-root");
  if (!root) { _replayMode = false; return; }

  _replayEngine = new ReplayEngine({
    onEvent: _handleReplayEvent,
    onSeekRebuild: _handleSeekRebuild,
    onComplete: _handleReplayComplete,
    onTimeUpdate: (ms, progress) => _replayUI?.updateTime(ms, progress),
  });

  _replayUI = new ReplayUI({
    container: root,
    onPlay: () => { _replayEngine?.play(); _replayUI?.setPlaying(true); },
    onPause: () => { _replayEngine?.pause(); _replayUI?.setPlaying(false); },
    onSeek: (ms) => _replayEngine?.seek(ms),
    onSpeedChange: (s) => _replayEngine?.setSpeed(s),
    onExit: () => stopReplay(),
    onRangeChange: (h) => _reloadReplayRange(h),
    initialHours: hours,
  });

  await _loadReplayData(hours);
}

async function _loadReplayData(hours) {
  _replayUI.setLoading(true);
  _replayUI.show();
  _clearAllCardStreams();

  try {
    await _replayEngine.load(hours);
    const range = _replayEngine.getTimeRange();
    _replayUI.updateTimeRange(range.start, range.end);
    _replayUI.setLoading(false);
    _replayEngine.seek(range.start);
  } catch {
    _replayUI.setLoading(false);
    stopReplay();
  }
}

async function _reloadReplayRange(hours) {
  if (!_replayMode || !_replayEngine || !_replayUI) return;
  _replayEngine.pause();
  _replayUI.setPlaying(false);
  await _loadReplayData(hours);
}

/**
 * Exit replay mode: dispose engine/UI, restore live state.
 */
export function stopReplay() {
  if (!_replayMode) return;
  _replayMode = false;

  const buffered = _replayEngine?.flushLiveBuffer() ?? [];
  _replayEngine?.dispose();
  _replayEngine = null;

  _replayUI?.dispose();
  _replayUI = null;

  // Restore live card streams
  _clearAllCardStreams();
  const animas = getState().animas || [];
  _loadInitialStreams(animas);

  // Re-apply buffered WS events
  for (const evt of buffered) {
    if (evt.handler) evt.handler(evt.data);
  }

  logger.info("Replay stopped, live mode restored");
}

/**
 * Buffer a WS event during replay for later application.
 * @param {object} handler - function to call when applying
 * @param {object} data - event data
 */
export function bufferReplayEvent(handler, data) {
  if (_replayMode && _replayEngine) {
    _replayEngine.bufferLiveEvent({ handler, data });
  }
}

function _clearAllCardStreams() {
  _cardStreams.clear();
  for (const [name] of _cardEls) {
    const streamEl = document.getElementById(`orgStream_${CSS.escape(name)}`);
    if (streamEl) streamEl.innerHTML = '<div class="org-stream-idle">\u{1F4A4} idle</div>';
  }
  if (_msgLinesGroup) _msgLinesGroup.innerHTML = "";
}

function _handleReplayEvent(evt, speed) {
  const type = evt.type || evt.name || "";
  const anima = evt.anima || (evt.animas && evt.animas[0]) || "";

  if (anima && _cardEls.has(anima)) {
    const data = {
      eventType: type,
      summary: evt.summary || "",
      content: evt.content || "",
      from_person: evt.meta?.from_person || "",
      to_person: evt.meta?.to_person || "",
      channel: evt.meta?.channel || evt.channel || "",
      toolName: evt.tool || evt.tool_name || "",
      toolId: evt.tool_id || "",
      isError: evt.is_error || false,
    };

    if (speed < 50 || type.includes("message") || type.includes("heartbeat") || type.includes("response") || type.includes("cron")) {
      updateCardActivity(anima, data);
    }
  }

  if (type === "message_sent" && evt.meta?.to_person && _cardEls.has(anima)) {
    const intent = evt.meta?.intent || "";
    let msgLineType = "internal";
    if (intent === "delegation") msgLineType = "delegation";
    showMessageLine(anima, evt.meta.to_person, evt.summary || "", { lineType: msgLineType, replaySpeed: speed });
  } else if (type === "dm_sent" || type === "dm_received") {
    const from = evt.meta?.from_person || anima;
    const to = evt.meta?.to_person || "";
    if (from && to) showMessageLine(from, to, evt.summary || "", { replaySpeed: speed });
  }
}

function _handleSeekRebuild({ cardStreams, cardStatus, kpiCounts }) {
  _clearAllCardStreams();

  for (const [name, entries] of cardStreams) {
    _cardStreams.set(name, entries);
    const streamEl = document.getElementById(`orgStream_${CSS.escape(name)}`);
    if (streamEl) _renderStream(streamEl, entries.slice(-MAX_STREAM_ENTRIES));
  }

  for (const [name, status] of cardStatus) {
    const card = _cardEls.get(name);
    if (card) card.dataset.status = status === "working" ? "working" : "idle";
  }

  const evH = document.getElementById("orgKpiEventsH");
  if (evH) evH.textContent = String(kpiCounts.eventsInLastHour);
}

function _handleReplayComplete() {
  _replayUI?.setPlaying(false);
  logger.info("Replay completed");
}

/** @deprecated Right-column activity feed removed. Kept as no-op for backward compat. */
export function addActivityItem(_item) {
  // no-op: right-column activity feed has been replaced by canvas layout
}
