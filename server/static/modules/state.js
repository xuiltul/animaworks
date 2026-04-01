/* ── State & DOM refs ──────────────────────── */

export const state = {
  uiTheme: "default",
  currentUser: null,
  currentUserRole: null,
  authMode: null,
  demoMode: false,
  animas: [],            // AnimaStatus[]
  selectedAnima: null,   // string (name)
  animaDetail: null,     // full detail object
  // chatHistories removed — now managed by ChatSessionManager
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

export function escapeAttr(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;")
            .replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function statusClass(status) {
  if (!status) return "status-offline";
  const s = status.toLowerCase();
  if (s === "idle" || s === "running") return "status-idle";
  if (s === "thinking" || s === "processing" || s === "busy" || s === "bootstrapping") return "status-thinking";
  if (s === "error") return "status-error";
  return "status-offline";
}

// ── KaTeX integration ────────────────────────
let _katexInitialized = false;
function _ensureKatex() {
  if (_katexInitialized) return;
  if (typeof markedKatex !== "undefined") {
    marked.use(markedKatex({ throwOnError: false, output: "htmlAndMathml" }));
  }
  _katexInitialized = true;
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
    return `<div class="text-artifact-card" data-filename="${escapeAttr(filename)}" data-content-id="${escapeAttr(id)}">` +
      `<span class="text-artifact-icon">\uD83D\uDCC4</span>` +
      `<span class="text-artifact-name">${escapeHtml(filename)}</span>` +
      `</div>`;
  }
  return `<div class="text-artifact-card" data-filename="${escapeAttr(filename)}" data-content="${escapeAttr(content)}">` +
    `<span class="text-artifact-icon">\uD83D\uDCC4</span>` +
    `<span class="text-artifact-name">${escapeHtml(filename)}</span>` +
    `</div>`;
};

const _markedOptions = { breaks: true, renderer: _markedRenderer };

// ── Foster-parenting prevention ──────────────────
// An unclosed <table> in the HTML string makes the HTML5 parser enter
// "in table" mode where non-table end tags (</div>) are ignored.
// This causes subsequent chat-msg-row elements to nest inside the
// previous bubble.  Round-tripping through a detached element forces
// the browser to auto-close every tag, producing well-formed HTML.
const _sanitizerEl = document.createElement("div");

function _ensureClosedTags(html) {
  if (!html || !html.includes("<table")) return html;
  _sanitizerEl.innerHTML = html;
  return _sanitizerEl.innerHTML;
}

export function renderMarkdown(text, animaName) {
  _ensureKatex();
  _mdAnimaCtx = animaName || null;
  try {
    return _ensureClosedTags(marked.parse(text, _markedOptions));
  } catch {
    return escapeHtml(text);
  } finally {
    _mdAnimaCtx = null;
  }
}

export function renderSafeMarkdown(text) {
  if (!text) return "";
  _ensureKatex();
  try {
    return _ensureClosedTags(marked.parse(escapeHtml(text), _markedOptions));
  } catch {
    return escapeHtml(text);
  }
}
