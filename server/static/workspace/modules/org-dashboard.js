/**
 * Organization dashboard view for the Workspace.
 * 2-column layout:
 * - Main (flex): Interactive organization tree with inline status
 * - Right (300px): Real-time activity feed
 */
import { createLogger } from "../../shared/logger.js";
import { escapeHtml, smartTimestamp } from "./utils.js";
import { animaHashColor } from "../../shared/avatar-utils.js";
import { bustupCandidates, resolveCachedAvatar } from "../../modules/avatar-resolver.js";

const logger = createLogger("org-dashboard");

let _container = null;
let _treeNodes = new Map();
let _activityFeed = null;
let _onNodeClick = null;
const MAX_ACTIVITY_ITEMS = 50;

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
  return roots.length ? roots : [...nodeMap.values()];
}

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

// ── Interactive Tree Node ──────────────────────

function renderInteractiveTreeNode(node, depth = 0, isLast = true, prefixLines = []) {
  const statusDot = getStatusDotClass(node.status);
  const statusLabel = getStatusLabel(node.status);
  const initial = (node.name || "?")[0].toUpperCase();
  const color = animaHashColor(node.name);

  let connector = "";
  if (depth > 0) {
    const prefix = prefixLines.map(hasLine => hasLine ? '<span class="org-itree-vline"></span>' : '<span class="org-itree-spacer"></span>').join("");
    const branch = isLast ? '<span class="org-itree-elbow"></span>' : '<span class="org-itree-tee"></span>';
    connector = `<span class="org-itree-connector">${prefix}${branch}</span>`;
  }

  const roleLabel = node.role || "";
  const specLabel = node.speciality || "";
  const tagHtml = roleLabel || specLabel
    ? `<span class="org-itree-tags">${roleLabel ? `<span class="org-itree-role">${escapeHtml(roleLabel)}</span>` : ""}${specLabel ? `<span class="org-itree-spec">${escapeHtml(specLabel)}</span>` : ""}</span>`
    : "";

  let html = `<div class="org-itree-node" data-name="${escapeHtml(node.name)}" id="orgNode_${escapeHtml(node.name)}">
    ${connector}
    <div class="org-itree-card">
      <div class="org-itree-avatar" style="background:${color}" data-anima="${escapeHtml(node.name)}">${initial}</div>
      <div class="org-itree-info">
        <span class="org-itree-name">${escapeHtml(node.name)}</span>
        ${tagHtml}
      </div>
      <span class="org-itree-status">
        <span class="org-itree-dot ${statusDot}"></span>
        <span class="org-itree-status-label">${escapeHtml(statusLabel)}</span>
      </span>
    </div>
  </div>`;

  const childPrefixLines = [...prefixLines, !isLast];
  for (let i = 0; i < node.children.length; i++) {
    const childIsLast = i === node.children.length - 1;
    html += renderInteractiveTreeNode(node.children[i], depth + 1, childIsLast, childPrefixLines);
  }
  return html;
}

// ── Activity Feed ──────────────────────

function renderActivityItem(item) {
  const time = smartTimestamp(item.ts || item.timestamp || "");
  const icon = item.type === "error" ? "⚠️" : "📌";
  const from = item.from ? `<span class="org-activity-from">${escapeHtml(item.from)}</span>` : "";
  const summary = escapeHtml(item.summary || item.content || item.type || "");
  return `<div class="org-activity-item">
    <span class="org-activity-icon">${icon}</span>
    <div class="org-activity-body">
      ${from}
      <span class="org-activity-text">${summary}</span>
    </div>
    <span class="org-activity-time">${time}</span>
  </div>`;
}

function addActivityItem(item) {
  if (!_activityFeed) return;
  const div = document.createElement("div");
  div.innerHTML = renderActivityItem(item);
  const el = div.firstElementChild;
  _activityFeed.prepend(el);
  while (_activityFeed.children.length > MAX_ACTIVITY_ITEMS) {
    _activityFeed.removeChild(_activityFeed.lastElementChild);
  }
}

// ── Main API ──────────────────────

export async function initOrgDashboard(container, animas, { onNodeClick } = {}) {
  _container = container;
  _treeNodes.clear();
  _onNodeClick = onNodeClick || null;

  const roots = buildOrgTree(animas);

  let treeHtml = "";
  for (let i = 0; i < roots.length; i++) {
    treeHtml += renderInteractiveTreeNode(roots[i], 0, i === roots.length - 1, []);
  }

  container.innerHTML = `
    <div class="org-dashboard">
      <div class="org-col-main">
        <div class="org-section-title">組織</div>
        <div class="org-itree">${treeHtml}</div>
      </div>
      <div class="org-col-right">
        <div class="org-section-title">アクティビティ</div>
        <div class="org-activity-feed" id="orgActivityFeed"></div>
      </div>
    </div>
  `;

  _activityFeed = document.getElementById("orgActivityFeed");

  for (const p of animas) {
    _treeNodes.set(p.name, document.getElementById(`orgNode_${p.name}`));
  }

  // Load recent activity
  try {
    const resp = await fetch("/api/activity/recent?hours=12&limit=20");
    if (resp.ok) {
      const data = await resp.json();
      const items = Array.isArray(data) ? data : (data.events || []);
      for (const item of items.reverse()) {
        addActivityItem(item);
      }
    }
  } catch (err) {
    logger.warn("Failed to load activity", { error: err.message });
  }

  // Tree node click → select anima
  container.addEventListener("click", (e) => {
    const node = e.target.closest(".org-itree-node");
    if (!node) return;
    const name = node.dataset.name;
    if (!name) return;

    // Visual highlight
    container.querySelectorAll(".org-itree-node.selected").forEach(el => el.classList.remove("selected"));
    node.classList.add("selected");

    if (_onNodeClick) _onNodeClick(name);
  });

  _loadOrgAvatars(animas);

  logger.info("Org dashboard initialized", { animaCount: animas.length });
}

async function _loadOrgAvatars(animas) {
  const candidates = bustupCandidates();
  for (const p of animas) {
    try {
      const url = await resolveCachedAvatar(p.name, candidates, "S");
      if (!url) continue;
      const el = _container?.querySelector(`.org-itree-avatar[data-anima="${CSS.escape(p.name)}"]`);
      if (!el) continue;
      const img = new Image();
      img.src = url;
      img.alt = p.name;
      img.style.cssText = "width:100%;height:100%;object-fit:cover;border-radius:6px;";
      img.onload = () => { el.textContent = ""; el.appendChild(img); };
    } catch { /* skip */ }
  }
}

export function disposeOrgDashboard() {
  if (_container) {
    _container.innerHTML = "";
  }
  _treeNodes.clear();
  _activityFeed = null;
  _container = null;
  _onNodeClick = null;
}

export function updateAnimaStatus(name, status) {
  const nodeEl = _treeNodes.get(name);
  if (!nodeEl) return;

  const state = typeof status === "object" ? (status.state || status.status || "unknown") : String(status);
  const dotClass = getStatusDotClass(status);

  const dot = nodeEl.querySelector(".org-itree-dot");
  if (dot) dot.className = `org-itree-dot ${dotClass}`;

  const label = nodeEl.querySelector(".org-itree-status-label");
  if (label) label.textContent = state.toLowerCase();
}

export { addActivityItem };
