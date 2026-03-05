/**
 * Organization dashboard — Canvas-based node graph layout.
 *
 * Each Anima is a draggable card on an absolute-positioned canvas.
 * Hierarchy connections are drawn as SVG cubic-bezier curves.
 * Initial placement uses a tree-layout algorithm; dragged positions
 * are persisted to localStorage under `aw-org-positions`.
 */
import { createLogger } from "../../shared/logger.js";
import { escapeHtml } from "./utils.js";
import { getState } from "./state.js";
import { animaHashColor } from "../../shared/avatar-utils.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";

const logger = createLogger("org-dashboard");

// ── Constants ──────────────────────

const CARD_W = 280;
const CARD_H = 80;
const GAP_X = 60;
const GAP_Y = 50;
const STORAGE_KEY = "aw-org-positions";

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
let _draggingCard = null;

let _panActive = false;
let _panStartX = 0;
let _panStartY = 0;
let _panScrollLeft = 0;
let _panScrollTop = 0;

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
  if (lower === "thinking" || lower === "working") return "dot-active";
  if (lower === "sleeping" || lower === "stopped" || lower === "not_found") return "dot-sleeping";
  if (lower.includes("error")) return "dot-error";
  if (lower.includes("bootstrap")) return "dot-bootstrap";
  return "dot-unknown";
}

function getStatusLabel(status) {
  if (!status) return "unknown";
  const s = typeof status === "object" ? (status.state || status.status || "unknown") : String(status);
  return s.toLowerCase();
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
    positions.set(measured.node.name, { x: cx, y });
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

function _loadPositions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return new Map(Object.entries(JSON.parse(raw)));
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

function _updateConnections() {
  if (!_svgLayer) return;
  _svgLayer.innerHTML = "";

  for (const [name, node] of _nodeData) {
    if (!node.supervisor) continue;
    const parentPos = _positions.get(node.supervisor);
    const childPos = _positions.get(name);
    if (!parentPos || !childPos) continue;

    const x1 = parentPos.x + CARD_W / 2;
    const y1 = parentPos.y + CARD_H;
    const x2 = childPos.x + CARD_W / 2;
    const y2 = childPos.y;
    const midY = (y1 + y2) / 2;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}`);
    path.setAttribute("class", "org-connection-line");
    _svgLayer.appendChild(path);
  }
}

// ── Drag Implementation ──────────────────────

function _setupDrag(cardEl, name) {
  let dragging = false;
  let startX, startY, cardStartX, cardStartY;

  cardEl.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (e.target.closest(".org-card-expand")) return;
    dragging = true;
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
    const x = cardStartX + dx;
    const y = cardStartY + dy;
    cardEl.style.left = `${x}px`;
    cardEl.style.top = `${y}px`;
    _positions.set(name, { x, y });
    requestAnimationFrame(() => _updateConnections());
  });

  cardEl.addEventListener("pointerup", () => {
    if (!dragging) return;
    dragging = false;
    _draggingCard = null;
    cardEl.classList.remove("org-card--dragging");
    _persistPositions();
  });

  cardEl.addEventListener("pointercancel", () => {
    if (!dragging) return;
    dragging = false;
    _draggingCard = null;
    cardEl.classList.remove("org-card--dragging");
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
  const initial = (node.name || "?")[0].toUpperCase();
  const color = animaHashColor(node.name);
  const roleLabel = node.role || "";
  const specLabel = node.speciality || "";
  const tagHtml = roleLabel || specLabel
    ? `<span class="org-card-tags">${roleLabel ? `<span class="org-card-role">${escapeHtml(roleLabel)}</span>` : ""}${specLabel ? `<span class="org-card-spec">${escapeHtml(specLabel)}</span>` : ""}</span>`
    : "";

  const card = document.createElement("div");
  card.className = "org-card";
  card.dataset.name = node.name;
  card.id = `orgCard_${node.name}`;
  card.innerHTML = `
    <div class="org-card-avatar" style="background:${color}" data-anima="${escapeHtml(node.name)}">${initial}</div>
    <div class="org-card-info">
      <span class="org-card-name">${escapeHtml(node.name)}</span>
      ${tagHtml}
    </div>
    <span class="org-card-status">
      <span class="org-card-dot ${statusDot}"></span>
      <span class="org-card-status-label">${escapeHtml(statusLabel)}</span>
    </span>
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
  const errors = animas.filter(a => String(a.status).includes("error")).length;

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
}

// ── SVG Sizing ──────────────────────

function _resizeSvg() {
  if (!_svgLayer || !_nodesLayer) return;
  const rect = _nodesLayer.getBoundingClientRect();
  let maxX = 0;
  let maxY = 0;
  for (const pos of _positions.values()) {
    maxX = Math.max(maxX, pos.x + CARD_W + 40);
    maxY = Math.max(maxY, pos.y + CARD_H + 40);
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

// ── Avatar Loading ──────────────────────

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

// ── KPI Live Updates ──────────────────────

async function _loadKpiStats() {
  try {
    const resp = await fetch("/api/activity/recent?hours=1&limit=200");
    if (!resp.ok) return;
    const data = await resp.json();
    const items = Array.isArray(data) ? data : (data.events || []);
    const eventsH = items.length;
    const el = document.getElementById("orgKpiEventsH");
    if (el) el.textContent = String(eventsH);
  } catch { /* ignore */ }
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

  container.addEventListener("click", (e) => {
    const card = e.target.closest(".org-card");
    if (!card) return;
    const name = card.dataset.name;
    if (!name) return;
    container.querySelectorAll(".org-card.selected").forEach(el => el.classList.remove("selected"));
    card.classList.add("selected");
    if (_onNodeClick) _onNodeClick(name);
  });

  window.addEventListener("resize", _onResize);

  _loadOrgAvatars(animas);
  _loadKpiStats();

  logger.info("Org dashboard initialized (canvas mode)", { animaCount: animas.length });
}

function _onResize() {
  _resizeSvg();
  _updateConnections();
}

export function disposeOrgDashboard() {
  window.removeEventListener("resize", _onResize);
  if (_container) {
    _container.innerHTML = "";
  }
  _cardEls.clear();
  _nodeData.clear();
  _positions.clear();
  _container = null;
  _viewport = null;
  _svgLayer = null;
  _nodesLayer = null;
  _kpiBar = null;
  _onNodeClick = null;
  _draggingCard = null;
  _panActive = false;
}

export function updateAnimaStatus(name, status) {
  const card = _cardEls.get(name);
  if (!card) return;

  const dotClass = getStatusDotClass(status);
  const label = getStatusLabel(status);

  const dot = card.querySelector(".org-card-dot");
  if (dot) dot.className = `org-card-dot ${dotClass}`;

  const labelEl = card.querySelector(".org-card-status-label");
  if (labelEl) labelEl.textContent = label;

  _renderKpiBar();
}

export function getCardPosition(name) {
  return _positions.get(name) || null;
}

export function updateCardActivity(name, _data) {
  // Placeholder for Issue 2 — live activity within cards
  void name;
}

/** @deprecated Right-column activity feed removed. Kept as no-op for backward compat. */
export function addActivityItem(_item) {
  // no-op: right-column activity feed has been replaced by canvas layout
}
