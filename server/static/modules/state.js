/* ── State & DOM refs ──────────────────────── */

import { basePath } from "/shared/base-path.js";
import {
  escapeAttr,
  escapeHtml,
  nowTimeStr,
  smartTimestamp,
  timeStr,
} from "/shared/html-utils.js";
export { escapeAttr, escapeHtml, nowTimeStr, smartTimestamp, timeStr };

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
  marked.use({ extensions: [_cjkStrongExtension] });
  _katexInitialized = true;
}

// CommonMark only closes "**" when it is followed by a space or punctuation,
// so Japanese like "**桜庭 咲良（さくら）**と申します" stayed as literal
// asterisks. This tokenizer runs before marked's own and ignores flanking.
const _cjkStrongExtension = {
  name: "cjkStrong",
  level: "inline",
  start(src) {
    const i = src.indexOf("**");
    return i < 0 ? undefined : i;
  },
  tokenizer(src) {
    const m = /^\*\*(?![\s*])([^\n]*?[^\s*])\*\*/.exec(src);
    if (!m) return undefined;
    return { type: "strong", raw: m[0], text: m[1], tokens: this.lexer.inlineTokens(m[1]) };
  },
};

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
    return `${basePath}/api/animas/${encodeURIComponent(_mdAnimaCtx)}/attachments/${encodeURIComponent(file)}`;
  }
  if (src.startsWith("assets/")) {
    const file = src.slice("assets/".length);
    return `${basePath}/api/animas/${encodeURIComponent(_mdAnimaCtx)}/assets/${encodeURIComponent(file)}`;
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
// An unclosed <table> or <select> in the HTML string makes the HTML5
// parser enter a special insertion mode where non-table/select end tags
// (</div>) are ignored.  An unclosed <div> shifts nesting so that
// subsequent closing tags close the wrong elements.  Either case causes
// chat-msg-row elements to nest inside the previous bubble or, worse,
// inside a collapsed activity bundle body (display:none), making all
// following messages invisible.
//
// Round-tripping through a detached element forces the browser to
// auto-close every tag, producing well-formed HTML.
const _sanitizerEl = document.createElement("div");

function _ensureClosedTags(html) {
  if (!html) return html;
  // Case-insensitive check for block elements that commonly appear as
  // raw HTML in Anima messages and can break the surrounding structure
  // when left unclosed.
  if (/<(?:table|select|div|details)\b/i.test(html)) {
    _sanitizerEl.innerHTML = html;
    return _sanitizerEl.innerHTML;
  }
  return html;
}

// A "---" line right under a paragraph is a Setext <h2> in Markdown. Anima
// replies join progress notes with single newlines and then a "---" divider,
// which turned every note into one big bold heading. Treat it as a rule.
export function breakSetextHeadings(text) {
  if (typeof text !== "string" || !text.includes("---")) return text;
  const lines = text.split("\n");
  const out = [];
  let inFence = false;
  for (const line of lines) {
    if (/^\s{0,3}(```|~~~)/.test(line)) inFence = !inFence;
    if (!inFence && /^\s{0,3}-{3,}\s*$/.test(line) && out.length && out[out.length - 1].trim() !== "") {
      out.push("");
    }
    out.push(line);
  }
  return out.join("\n");
}

export function renderMarkdown(text, animaName) {
  _ensureKatex();
  _mdAnimaCtx = animaName || null;
  try {
    return _ensureClosedTags(marked.parse(breakSetextHeadings(text), _markedOptions));
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
    return _ensureClosedTags(marked.parse(breakSetextHeadings(escapeHtml(text)), _markedOptions));
  } catch {
    return escapeHtml(text);
  }
}
