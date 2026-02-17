/**
 * message-popup.js — Message detail popup for timeline events.
 *
 * Shows full message content when clicking a message-type timeline event.
 * Positioned at the top-left of the office panel, above the timeline.
 */

import { renderSimpleMarkdown, escapeHtml } from "./utils.js";
import { fetchMessage } from "./api.js";

// ── Module state ────────────────────────────────
/** @type {HTMLElement|null} */
let _overlay = null;
/** @type {HTMLElement|null} */
let _popup = null;
/** @type {HTMLElement|null} */
let _contentEl = null;
/** @type {boolean} */
let _visible = false;

// ── DOM construction ────────────────────────────

function _buildDOM(parentEl) {
  _overlay = document.createElement("div");
  _overlay.className = "ws-msg-popup-overlay hidden";
  _overlay.addEventListener("click", (e) => {
    if (e.target === _overlay) hide();
  });

  _popup = document.createElement("div");
  _popup.className = "ws-msg-popup";

  const header = document.createElement("div");
  header.className = "ws-msg-popup-header";

  const title = document.createElement("span");
  title.className = "ws-msg-popup-title";
  title.textContent = "Message Detail";

  const closeBtn = document.createElement("button");
  closeBtn.className = "ws-msg-popup-close";
  closeBtn.textContent = "\u00D7";
  closeBtn.addEventListener("click", hide);

  header.appendChild(title);
  header.appendChild(closeBtn);

  _contentEl = document.createElement("div");
  _contentEl.className = "ws-msg-popup-content";

  _popup.appendChild(header);
  _popup.appendChild(_contentEl);
  _overlay.appendChild(_popup);
  parentEl.appendChild(_overlay);
}

// ── Render ───────────────────────────────────────

function _renderLoading() {
  _contentEl.innerHTML = '<div class="ws-msg-popup-loading">Loading...</div>';
}

function _renderError(msg) {
  _contentEl.innerHTML = `<div class="ws-msg-popup-error">${escapeHtml(msg)}</div>`;
}

function _renderMessage(message) {
  const ts = message.timestamp
    ? new Date(message.timestamp).toLocaleString("ja-JP")
    : "";

  _contentEl.innerHTML = `
    <div class="ws-msg-popup-meta">
      <div class="ws-msg-popup-from">
        <span class="ws-msg-popup-label">From:</span>
        <span class="ws-msg-popup-value">${escapeHtml(message.from_person || "")}</span>
      </div>
      <div class="ws-msg-popup-to">
        <span class="ws-msg-popup-label">To:</span>
        <span class="ws-msg-popup-value">${escapeHtml(message.to_person || "")}</span>
      </div>
      <div class="ws-msg-popup-time">
        <span class="ws-msg-popup-label">Time:</span>
        <span class="ws-msg-popup-value">${escapeHtml(ts)}</span>
      </div>
    </div>
    <div class="ws-msg-popup-body">
      ${renderSimpleMarkdown(message.content || "")}
    </div>
  `;
}

// ── Public API ───────────────────────────────────

export function initMessagePopup(parentEl) {
  _buildDOM(parentEl);
}

export async function showMessage(messageId) {
  if (!_overlay || !_contentEl) return;

  _overlay.classList.remove("hidden");
  _visible = true;
  _renderLoading();

  try {
    const message = await fetchMessage(messageId);
    if (message.error) {
      _renderError(message.error);
    } else {
      _renderMessage(message);
    }
  } catch (err) {
    _renderError("Failed to load message: " + err.message);
  }
}

export function hide() {
  if (_overlay) {
    _overlay.classList.add("hidden");
  }
  _visible = false;
}

export function isVisible() {
  return _visible;
}
