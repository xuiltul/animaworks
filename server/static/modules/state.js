/* ── State & DOM refs ──────────────────────── */

export const state = {
  uiTheme: "default",
  currentUser: null,
  currentUserRole: null,
  authMode: null,
  animas: [],            // AnimaStatus[]
  selectedAnima: null,   // string (name)
  animaDetail: null,     // full detail object
  chatHistories: {},      // { [name]: [{role, text}] }
  activeMemoryTab: "episodes",
  activeRightTab: "state",
  wsConnected: false,
  sessionList: null,      // cached session list for selected anima
};

const $id = (id) => document.getElementById(id);

export const dom = {
  systemStatus: $id("systemStatus"),
  systemStatusText: $id("systemStatusText"),
  loginScreen: $id("loginScreen"),
  userList: $id("userList"),
  guestLoginBtn: $id("guestLoginBtn"),
  userInfo: $id("userInfo"),
  currentUserLabel: $id("currentUserLabel"),
  logoutBtn: $id("logoutBtn"),
};

// ── Helpers ────────────────────────────────

export function timeStr(isoOrTs) {
  if (!isoOrTs) return "--:--";
  const d = new Date(isoOrTs);
  if (isNaN(d.getTime())) return "--:--";
  return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
}

export function nowTimeStr() {
  return new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function smartTimestamp(isoOrTs) {
  if (!isoOrTs) return "";
  const d = new Date(isoOrTs);
  if (isNaN(d.getTime())) return "";
  const now = new Date();
  const time = d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) return time;
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  if (d.getFullYear() === now.getFullYear()) return `${mm}/${dd} ${time}`;
  return `${d.getFullYear()}/${mm}/${dd} ${time}`;
}

export function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

export function statusClass(status) {
  if (!status) return "status-offline";
  const s = status.toLowerCase();
  if (s === "idle" || s === "running") return "status-idle";
  if (s === "thinking" || s === "processing" || s === "busy" || s === "bootstrapping") return "status-thinking";
  if (s === "error") return "status-error";
  return "status-offline";
}

const _markedRenderer = new marked.Renderer();
const _origLinkRenderer = _markedRenderer.link.bind(_markedRenderer);
_markedRenderer.link = function (token) {
  const html = _origLinkRenderer(token);
  return html.replace(/^<a /, '<a target="_blank" rel="noopener noreferrer" ');
};
let _mdAnimaCtx = null;

function _resolveAnimaSrc(src) {
  if (!_mdAnimaCtx || !src) return src;
  if (src.startsWith("attachments/")) {
    const file = src.slice("attachments/".length);
    return `/api/animas/${encodeURIComponent(_mdAnimaCtx)}/attachments/${encodeURIComponent(file)}`;
  }
  if (src.startsWith("assets/")) {
    const file = src.slice("assets/".length);
    return `/api/animas/${encodeURIComponent(_mdAnimaCtx)}/assets/${encodeURIComponent(file)}`;
  }
  return src;
}

_markedRenderer.image = function (token) {
  const src = _resolveAnimaSrc(token.href || "");
  const alt = escapeHtml(token.text || "Image");
  return `<img src="${src}" alt="${alt}" class="chat-attached-image" loading="lazy" onerror="this.onerror=null;this.classList.add('chat-attached-image-error');this.alt='Image unavailable';" />`;
};

_markedRenderer.code = function (token) {
  const lang = (token.lang || "").trim();
  const fileMatch = lang.match(/^file:(.+)$/i);
  if (!fileMatch) {
    const escaped = escapeHtml(token.text || "");
    const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : "";
    return `<pre><code${langClass}>${escaped}</code></pre>`;
  }
  const filename = fileMatch[1].trim();
  const content = token.text || "";
  if (content.length > 100 * 1024) {
    const id = window.__registerArtifactContent?.(content) || "";
    return `<div class="text-artifact-card" data-filename="${escapeHtml(filename)}" data-content-id="${escapeHtml(id)}">` +
      `<span class="text-artifact-icon">\uD83D\uDCC4</span>` +
      `<span class="text-artifact-name">${escapeHtml(filename)}</span>` +
      `</div>`;
  }
  return `<div class="text-artifact-card" data-filename="${escapeHtml(filename)}" data-content="${escapeHtml(content)}">` +
    `<span class="text-artifact-icon">\uD83D\uDCC4</span>` +
    `<span class="text-artifact-name">${escapeHtml(filename)}</span>` +
    `</div>`;
};

const _markedOptions = { breaks: true, renderer: _markedRenderer };

export function renderMarkdown(text, animaName) {
  _mdAnimaCtx = animaName || null;
  try {
    return marked.parse(text, _markedOptions);
  } catch {
    return escapeHtml(text);
  } finally {
    _mdAnimaCtx = null;
  }
}

export function renderSafeMarkdown(text) {
  if (!text) return "";
  try {
    return marked.parse(escapeHtml(text), _markedOptions);
  } catch {
    return escapeHtml(text);
  }
}
