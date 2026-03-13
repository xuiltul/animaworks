// ── Shared Chat Render Utilities ──────────────────────
// Pure HTML-generating functions used by both Dashboard and Workspace chat UIs.
// All functions are DOM-independent (return HTML strings) except bindToolCallHandlers.
import { t } from "../i18n.js";

// ── TextAnimator ──────────────────────────────────────
// Buffers incoming text deltas and drips them at a constant rate
// via requestAnimationFrame to smooth out bursty API token delivery.
// Adapts display speed dynamically based on measured incoming chunk rate.

const _DEFAULT_CHAR_INTERVAL_MS = 14;
const _MIN_CHAR_INTERVAL_MS = 4;
const _MAX_CHAR_INTERVAL_MS = 100;
const _CATCHUP_THRESHOLD_FAST = 300;
const _CATCHUP_THRESHOLD_MED = 150;
const _RATE_WINDOW_SIZE = 6;
const _RATE_DRAIN_FACTOR = 3;

export class TextAnimator {
  /**
   * @param {object} opts
   * @param {number}  [opts.charIntervalMs=8] - Initial ms per character (auto-adjusted)
   * @param {function(string, string): void} opts.onUpdate - (displayText, fullBuffer) called on each animation step
   */
  constructor({ charIntervalMs = _DEFAULT_CHAR_INTERVAL_MS, onUpdate } = {}) {
    this._buffer = "";
    this._displayLen = 0;
    this._charInterval = charIntervalMs;
    this._baseInterval = charIntervalMs;
    this._onUpdate = onUpdate;
    this._rafId = null;
    this._lastStepTime = 0;
    this._remainder = 0;
    this._running = false;
    this._pushHistory = [];
  }

  start() {
    this._buffer = "";
    this._displayLen = 0;
    this._running = true;
    this._lastStepTime = performance.now();
    this._remainder = 0;
    this._pushHistory = [];
    this._charInterval = this._baseInterval;
    this._scheduleTick();
  }

  push(delta) {
    if (!delta) return;
    this._buffer += delta;
    const now = performance.now();
    this._pushHistory.push({ t: now, len: delta.length });
    if (this._pushHistory.length > _RATE_WINDOW_SIZE) {
      this._pushHistory.shift();
    }
    if (this._pushHistory.length >= 3) {
      this._adaptRate();
    }
  }

  flush() {
    this._displayLen = this._buffer.length;
    this._running = false;
    this._cancelTick();
    if (this._onUpdate) this._onUpdate(this._buffer, this._buffer);
  }

  stop() {
    this._running = false;
    this._cancelTick();
  }

  get displayText() { return this._buffer.slice(0, this._displayLen); }
  get isAnimating() { return this._displayLen < this._buffer.length; }

  _adaptRate() {
    const h = this._pushHistory;
    if (h.length < 3) return;
    const span = h[h.length - 1].t - h[0].t;
    if (span <= 0) return;
    let totalChars = 0;
    for (let i = 0; i < h.length; i++) totalChars += h[i].len;
    const incomingMsPerChar = span / totalChars;
    this._charInterval = Math.max(
      _MIN_CHAR_INTERVAL_MS,
      Math.min(_MAX_CHAR_INTERVAL_MS, incomingMsPerChar * _RATE_DRAIN_FACTOR),
    );
  }

  _scheduleTick() {
    if (this._rafId != null) return;
    this._rafId = requestAnimationFrame((now) => {
      this._rafId = null;
      this._step(now);
    });
  }

  _cancelTick() {
    if (this._rafId != null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  _step(now) {
    const pending = this._buffer.length - this._displayLen;
    if (pending > 0) {
      const elapsed = now - this._lastStepTime;
      this._lastStepTime = now;
      let interval = this._charInterval;
      if (pending > _CATCHUP_THRESHOLD_FAST) interval = this._charInterval / 4;
      else if (pending > _CATCHUP_THRESHOLD_MED) interval = this._charInterval / 2;

      this._remainder += elapsed;
      const charsToAdd = Math.floor(this._remainder / Math.max(interval, 1));
      if (charsToAdd > 0) {
        this._remainder -= charsToAdd * interval;
        const prev = this._displayLen;
        this._displayLen = Math.min(this._displayLen + charsToAdd, this._buffer.length);
        if (this._displayLen !== prev && this._onUpdate) {
          this._onUpdate(this.displayText, this._buffer);
        }
      }
    }
    if (this._running || this._displayLen < this._buffer.length) {
      this._scheduleTick();
    }
  }
}

const DEFAULT_TOOL_RESULT_TRUNCATE = 500;

// ── Bubble Action Helpers ──────────────────────

function _escapeAttr(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/"/g, "&quot;")
            .replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _bubbleActionsHtml(rawText) {
  if (!rawText) return "";
  return `<div class="bubble-actions">`
    + `<button class="bubble-action-btn" data-action="copy" title="Copy"><i data-lucide="copy"></i></button>`
    + `<button class="bubble-action-btn" data-action="download" title="Download"><i data-lucide="download"></i></button>`
    + `</div>`;
}

const _RE_VOICE_MODE_SUFFIX = /\n*\[voice-mode:[^\]]*\]/g;

function _stripVoiceSuffix(text) {
  return text ? text.replace(_RE_VOICE_MODE_SUFFIX, "") : text;
}

const _USER_SVG = `<svg class="chat-msg-avatar-user" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 12 0v1"/></svg>`;

function _renderAvatar(name, avatarMap) {
  if (!avatarMap) return "";
  const url = avatarMap[name];
  if (url) {
    return `<div class="chat-msg-avatar"><img class="chat-msg-avatar-img" src="${url}" alt=""></div>`;
  }
  if (name) {
    const ch = (name.charAt(0) || "?").toUpperCase();
    return `<div class="chat-msg-avatar"><span class="chat-msg-avatar-initial">${ch}</span></div>`;
  }
  return `<div class="chat-msg-avatar">${_USER_SVG}</div>`;
}

function _wrapRow(role, bubbleHtml, avatarHtml) {
  if (!avatarHtml) return bubbleHtml;
  if (role === "user") {
    return `<div class="chat-msg-row user">${bubbleHtml}${avatarHtml}</div>`;
  }
  return `<div class="chat-msg-row assistant">${avatarHtml}${bubbleHtml}</div>`;
}

/**
 * Render a history message (from conversation API) to HTML.
 * @param {object} msg - Message object with role, content, tool_calls, images, ts, from_person
 * @param {object} opts
 * @param {function} opts.escapeHtml
 * @param {function} opts.renderMarkdown - Markdown→HTML (varies between dashboard/workspace)
 * @param {function} opts.renderChatImages - Image rendering helper
 * @param {function} opts.smartTimestamp
 * @param {string}  [opts.animaName]
 * @param {number}  [opts.truncateLen] - Tool result truncation length
 */
export function renderHistoryMessage(msg, opts) {
  const { escapeHtml, renderMarkdown, smartTimestamp } = opts;
  const renderImages = opts.renderChatImages || (() => "");
  const truncLen = opts.truncateLen || DEFAULT_TOOL_RESULT_TRUNCATE;
  const avatarMap = opts.avatarMap || null;

  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "system") {
    return `<div class="chat-bubble assistant" style="opacity:0.7; font-style:italic;">${escapeHtml(msg.content || "")}${tsHtml}</div>`;
  }

  if (msg.role === "assistant") {
    const rawText = msg.content || "";
    const content = rawText ? renderMarkdown(rawText, opts.animaName) : "";
    const toolHtml = renderToolCalls(msg.tool_calls, { escapeHtml, truncateLen: truncLen });
    const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });
    let thinkingHtml = "";
    if (msg.thinking_text) {
      thinkingHtml = `<details class="thinking-block"><summary class="thinking-summary">\u{1F4AD} Thinking</summary><pre class="thinking-content">${escapeHtml(msg.thinking_text)}</pre></details>`;
    }
    const toLabel = msg.to_person
      ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">→ ${escapeHtml(msg.to_person)}</div>`
      : "";
    const actionsHtml = _bubbleActionsHtml(rawText);
    const dataAttr = rawText ? ` data-raw-text="${_escapeAttr(rawText)}"` : "";
    const bubble = `<div class="chat-bubble assistant"${dataAttr}>${actionsHtml}${toLabel}${thinkingHtml}${content}${imagesHtml}${toolHtml}${tsHtml}</div>`;
    return _wrapRow("assistant", bubble, _renderAvatar(opts.animaName, avatarMap));
  }

  const isAnima = msg.from_person && msg.from_person !== "human";
  const fromLabel = isAnima
    ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">${escapeHtml(msg.from_person)}</div>`
    : "";
  const userContent = _stripVoiceSuffix(msg.content || "");
  const contentHtml = isAnima
    ? renderMarkdown(userContent)
    : `<div class="chat-text">${escapeHtml(userContent)}</div>`;
  const bubble = `<div class="chat-bubble user">${fromLabel}${contentHtml}${tsHtml}</div>`;
  const avatarHtml = _renderAvatar(isAnima ? msg.from_person : null, avatarMap);
  return _wrapRow("user", bubble, avatarHtml);
}

/**
 * Render a session divider between conversation sessions.
 * @param {object} session - Session object with trigger, session_start
 * @param {boolean} isFirst - Whether this is the first session (suppresses divider)
 * @param {object} opts
 * @param {function} opts.escapeHtml
 * @param {function} opts.smartTimestamp
 */
export function renderSessionDivider(session, isFirst, opts) {
  if (isFirst) return "";
  const { escapeHtml, smartTimestamp } = opts;

  const trigger = session.trigger || "chat";
  let label = "";
  let extraClass = "";

  if (trigger === "heartbeat") {
    label = t("chat.heartbeat_activity");
    extraClass = " session-divider-heartbeat";
  } else if (trigger === "cron") {
    label = t("chat.cron_activity");
    extraClass = " session-divider-cron";
  } else {
    label = session.session_start ? smartTimestamp(session.session_start) : "";
  }

  return `<div class="session-divider${extraClass}"><span class="session-divider-label">${escapeHtml(label)}</span></div>`;
}

/**
 * Render tool call rows (collapsed by default).
 * @param {Array|null} toolCalls
 * @param {object} opts
 * @param {function} opts.escapeHtml
 * @param {number}  [opts.truncateLen]
 */
export function renderToolCalls(toolCalls, opts) {
  if (!toolCalls || toolCalls.length === 0) return "";
  const { escapeHtml } = opts;
  const truncLen = opts.truncateLen || DEFAULT_TOOL_RESULT_TRUNCATE;

  const innerHtml = toolCalls.map((tc, idx) => {
    const errorClass = tc.is_error ? " tool-call-error" : "";
    const toolName = escapeHtml(tc.tool_name || "unknown");
    const errorLabel = tc.is_error ? ` ${t("chat.error_prefix")}` : "";

    return `<div class="tool-call-row${errorClass}" data-tool-idx="${idx}">` +
      `<span class="tool-call-row-icon">\u25B6</span>` +
      `<span class="tool-call-row-name">${toolName}${errorLabel}</span>` +
      `</div>` +
      `<div class="tool-call-detail" data-tool-idx="${idx}" style="display:none;">` +
      renderToolCallDetail(tc, { escapeHtml, truncateLen: truncLen }) +
      `</div>`;
  }).join("");

  const hasErrors = toolCalls.some(tc => tc.is_error);
  const errorIndicator = hasErrors ? ' <span class="tool-group-error-badge">!</span>' : "";
  return `<div class="tool-call-group">` +
    `<div class="tool-call-group-header">` +
    `<span class="tool-call-group-icon">\u25B6</span>` +
    `<span class="tool-call-group-label">Tool Calls (${toolCalls.length})${errorIndicator}</span>` +
    `</div>` +
    `<div class="tool-call-group-body" style="display:none;">${innerHtml}</div>` +
    `</div>`;
}

/**
 * Render the detail content of a single tool call.
 */
export function renderToolCallDetail(tc, opts) {
  const { escapeHtml } = opts;
  const truncLen = opts.truncateLen || DEFAULT_TOOL_RESULT_TRUNCATE;
  let html = "";

  const input = tc.input || "";
  if (input) {
    const inputStr = typeof input === "string" ? input : JSON.stringify(input, null, 2);
    html += `<div class="tool-call-label">${t("ws.tool_input")}</div><div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">${t("ws.tool_result")}</div>`;
    if (resultStr.length > truncLen) {
      const truncated = resultStr.slice(0, truncLen);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">${t("chat.show_more")}</button>`;
    } else {
      html += `<div class="tool-call-content">${escapeHtml(resultStr)}</div>`;
    }
  }

  return html;
}

/**
 * Bind expand/collapse and "show more" event handlers for tool calls.
 * This is the only DOM-dependent function in this module.
 * @param {HTMLElement|null} container
 */
export function bindToolCallHandlers(container) {
  if (!container) return;

  container.querySelectorAll(".tool-call-group-header").forEach(header => {
    header.addEventListener("click", () => {
      const group = header.parentElement;
      const body = group.querySelector(".tool-call-group-body");
      if (!body) return;
      const isExpanded = group.classList.contains("expanded");
      group.classList.toggle("expanded", !isExpanded);
      body.style.display = isExpanded ? "none" : "";
    });
  });

  container.querySelectorAll(".tool-call-row").forEach(row => {
    row.addEventListener("click", () => {
      const idx = row.dataset.toolIdx;
      const detail = row.nextElementSibling;
      if (!detail || detail.dataset.toolIdx !== idx) return;
      const isExpanded = row.classList.contains("expanded");
      row.classList.toggle("expanded", !isExpanded);
      detail.style.display = isExpanded ? "none" : "";
    });
  });

  container.querySelectorAll(".tool-call-show-more").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const contentEl = btn.previousElementSibling;
      if (!contentEl) return;
      const fullResult = contentEl.dataset.fullResult;
      if (fullResult) {
        contentEl.textContent = fullResult;
        delete contentEl.dataset.fullResult;
        btn.remove();
      }
    });
  });
}

/**
 * Render a collapsible background session group (heartbeat/cron/task).
 * @param {Array} sessions - Array of session objects (1 for heartbeat/task, possibly multiple for grouped cron)
 * @param {string} type - "heartbeat" | "cron" | "task"
 * @param {object} opts - Same opts as renderHistoryMessage (escapeHtml, renderMarkdown, smartTimestamp)
 * @returns {string} HTML string
 */
export function renderCollapsibleSession(sessions, type, opts) {
  const { escapeHtml, renderMarkdown, smartTimestamp } = opts;

  const allMessages = sessions.flatMap((s) => s.messages || []);
  if (allMessages.length === 0) return "";

  const startTs = sessions[0]?.session_start;
  const endTs = sessions[sessions.length - 1]?.session_end;
  const timeLabel = startTs ? smartTimestamp(startTs) : "";
  const timeRange =
    startTs && endTs && startTs !== endTs
      ? `${smartTimestamp(startTs)} \u301C ${smartTimestamp(endTs)}`
      : timeLabel;

  let headerLabel = "";
  if (type === "heartbeat") {
    headerLabel = t("chat.heartbeat_activity");
  } else if (type === "cron") {
    headerLabel = t("chat.bg_tasks_count", { count: allMessages.length });
  } else if (type === "task") {
    headerLabel = t("chat.task_exec_activity");
  }

  let bodyHtml = "";
  if (type === "heartbeat" || type === "task") {
    for (const msg of allMessages) {
      const content = msg.content || "";
      if (content) {
        bodyHtml += `<div class="bg-session-message">${renderMarkdown(escapeHtml(content))}</div>`;
      }
    }
  } else {
    for (const msg of allMessages) {
      const ts = msg.ts ? smartTimestamp(msg.ts) : "";
      const tsHtml = ts ? `<span class="bg-session-item-ts">${escapeHtml(ts)}</span>` : "";
      bodyHtml += `<div class="bg-session-item">${escapeHtml(msg.content || "")}${tsHtml}</div>`;
    }
  }

  return (
    `<div class="bg-session-group bg-session-group--${type}">` +
    `<div class="bg-session-header bg-session-header--${type}">` +
    `<span class="bg-session-chevron">\u25B6</span>` +
    `<span class="bg-session-label">${escapeHtml(headerLabel)}</span>` +
    `<span class="bg-session-time">${escapeHtml(timeRange)}</span>` +
    `</div>` +
    `<div class="bg-session-body" style="display:none;">${bodyHtml}</div>` +
    `</div>`
  );
}

/**
 * Bind expand/collapse handlers for collapsible background session groups.
 * @param {HTMLElement|null} container
 */
export function bindCollapsibleSessionHandlers(container) {
  if (!container) return;
  container.querySelectorAll(".bg-session-header").forEach((header) => {
    header.addEventListener("click", () => {
      const group = header.parentElement;
      if (!group) return;
      const body = group.querySelector(".bg-session-body");
      if (!body) return;
      const isExpanded = group.classList.contains("expanded");
      group.classList.toggle("expanded", !isExpanded);
      body.style.display = isExpanded ? "none" : "";
    });
  });
}

/**
 * Render a live (current session) chat bubble to HTML.
 * @param {object} msg - Live message with role, text, streaming, activeTool, images, etc.
 * @param {object} opts
 * @param {function} opts.escapeHtml
 * @param {function} opts.renderMarkdown
 * @param {function} opts.renderChatImages
 * @param {function} opts.smartTimestamp
 * @param {string}  [opts.animaName]
 * @param {object}  [opts.labels] - UI labels { thinking, toolRunning, currentSession, heartbeatRelay, heartbeatRelayDone }
 */
export function renderLiveBubble(msg, opts) {
  const { escapeHtml, renderMarkdown, smartTimestamp } = opts;
  const renderImages = opts.renderChatImages || (() => "");
  const labels = opts.labels || {};
  const avatarMap = opts.avatarMap || null;

  const ts = msg.timestamp ? smartTimestamp(msg.timestamp) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "thinking") {
    const thinkLabel = labels.thinking || t("chat.thinking");
    const bubble = `<div class="chat-bubble thinking"><span class="thinking-animation">${thinkLabel}</span></div>`;
    return _wrapRow("assistant", bubble, _renderAvatar(opts.animaName, avatarMap));
  }

  if (msg.role === "system") {
    return `<div class="chat-visit-marker">${escapeHtml(msg.text)}${tsHtml}</div>`;
  }

  if (msg.role === "user") {
    const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });
    const userText = _stripVoiceSuffix(msg.text || "");
    const textHtml = userText ? `<div class="chat-text">${escapeHtml(userText)}</div>` : "";
    const bubble = `<div class="chat-bubble user">${imagesHtml}${textHtml}${tsHtml}</div>`;
    return _wrapRow("user", bubble, _renderAvatar(null, avatarMap));
  }

  // assistant
  const streamClass = msg.streaming ? " streaming" : "";
  const streamIdAttr = msg.streaming && msg.streamId
    ? ` data-stream-id="${escapeHtml(String(msg.streamId))}"`
    : "";
  let thinkingHtml = "";
  if (msg.thinking && msg.thinkingText) {
    thinkingHtml = `<div class="thinking-inline-preview">${escapeHtml(msg.thinkingText)}</div>`;
  } else if (!msg.thinking && msg.thinkingText && !msg.streaming) {
    thinkingHtml = `<details class="thinking-block"><summary class="thinking-summary">\u{1F4AD} Thinking</summary><pre class="thinking-content">${escapeHtml(msg.thinkingText)}</pre></details>`;
  }

  const streamingCursor = '<span class="streaming-cursor">▌</span>';
  let content = "";
  if (msg.text) {
    content = renderMarkdown(msg.text, opts.animaName);
    if (msg.streaming) content += streamingCursor;
  } else if (msg.streaming) {
    content = streamingCursor;
  }

  const compLabel = labels.compressing || t("chat.compressing");
  const compressionHtml = msg.compressing
    ? `<div class="compression-indicator"><span class="tool-spinner"></span>${compLabel}</div>`
    : "";
  let toolHtml = "";
  const history = msg.toolHistory;
  if (history && history.length > 0) {
    toolHtml = renderToolActivityTimeline(history, msg.activeTool, { escapeHtml, labels });
  } else if (msg.activeTool) {
    const toolLabel = labels.toolRunning || ((tool) => t("chat.tool_running", { tool }));
    toolHtml = `<div class="tool-indicator"><span class="tool-spinner"></span>${typeof toolLabel === "function" ? toolLabel(msg.activeTool) : toolLabel}</div>`;
  }
  const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });

  const rawText = msg.text || "";
  const actionsHtml = msg.streaming ? "" : _bubbleActionsHtml(rawText);
  const dataRawAttr = rawText && !msg.streaming ? ` data-raw-text="${_escapeAttr(rawText)}"` : "";
  const bubble = `<div class="chat-bubble assistant${streamClass}"${streamIdAttr}${dataRawAttr}>${actionsHtml}${content}${imagesHtml}${compressionHtml}${toolHtml}${thinkingHtml}${tsHtml}</div>`;
  return _wrapRow("assistant", bubble, _renderAvatar(opts.animaName, avatarMap));
}

/**
 * Generate the full zoned HTML for a streaming bubble (initial render).
 * @param {object} msg - Streaming message state
 * @param {object} opts - Same as renderLiveBubble opts
 */
export function renderStreamingBubbleInner(msg, opts) {
  return `<div class="streaming-zone-text">${_renderTextZoneContent(msg, opts)}</div>`
    + `<div class="streaming-zone-tools">${_renderToolZoneContent(msg, opts)}</div>`
    + `<div class="streaming-zone-subordinate">${_renderSubordinateZoneContent(msg, opts)}</div>`
    + `<div class="streaming-zone-thinking">${_renderThinkingZoneContent(msg, opts)}</div>`;
}

/**
 * Update only the specified zone(s) inside an existing streaming bubble DOM element.
 * @param {HTMLElement} bubble - The .chat-bubble.assistant.streaming element
 * @param {object} msg - Streaming message state
 * @param {object} opts - Same as renderLiveBubble opts
 * @param {string} zone - Which zone to update: "text" | "tools" | "subordinate" | "thinking" | "all"
 */
const _MD_RERENDER_MS = 80;
const _MD_RERENDER_CHARS = 40;

export function updateStreamingZone(bubble, msg, opts, zone = "all") {
  if (!bubble) return;
  if (zone === "all") {
    bubble.innerHTML = renderStreamingBubbleInner(msg, opts);
    _scrollThinkingToBottom(bubble);
    return;
  }
  const sel = `.streaming-zone-${zone}`;
  const el = bubble.querySelector(sel);
  if (!el) {
    bubble.innerHTML = renderStreamingBubbleInner(msg, opts);
    _scrollThinkingToBottom(bubble);
    return;
  }
  if (zone === "subordinate") {
    _patchSubordinateZone(el, msg, opts);
    return;
  }

  // Fast path: append-only textContent update (no innerHTML replacement)
  if (zone === "text" && msg.streaming && msg._mdCache) {
    const visibleText = msg._displayText || msg.text;
    if (visibleText) {
      const c = msg._mdCache;
      const now = performance.now();
      if ((now - c.t < _MD_RERENDER_MS) && (visibleText.length - c.len < _MD_RERENDER_CHARS)) {
        const tailEl = el.querySelector(".streaming-tail");
        if (tailEl) {
          tailEl.textContent = visibleText.slice(c.len);
          return;
        }
      }
    }
  }

  if (zone === "thinking") {
    _patchThinkingZone(el, msg, opts);
    return;
  }

  const renderers = {
    text: _renderTextZoneContent,
    tools: _renderToolZoneContent,
  };
  const fn = renderers[zone];
  if (fn) el.innerHTML = fn(msg, opts);
}

/**
 * Patch thinking zone without destroying/recreating the DOM element.
 * Avoids scrollTop reset that causes visible scroll-jump on every update.
 */
function _patchThinkingZone(zoneEl, msg, opts) {
  const { escapeHtml } = opts;
  const visibleThinking = msg._displayThinkingText || msg.thinkingText;

  if (msg.thinking && visibleThinking) {
    let preview = zoneEl.querySelector(".thinking-inline-preview");
    if (preview) {
      preview.textContent = visibleThinking;
    } else {
      zoneEl.innerHTML = `<div class="thinking-inline-preview">${escapeHtml(visibleThinking)}</div>`;
      preview = zoneEl.querySelector(".thinking-inline-preview");
    }
    if (preview) preview.scrollTop = preview.scrollHeight;
    return;
  }

  zoneEl.innerHTML = _renderThinkingZoneContent(msg, opts);
}

/** Scroll thinking preview to bottom (used after full innerHTML replacement). */
function _scrollThinkingToBottom(container) {
  const el = container.querySelector(".thinking-inline-preview");
  if (el) el.scrollTop = el.scrollHeight;
}

function _renderTextZoneContent(msg, opts) {
  const { escapeHtml, renderMarkdown } = opts;
  const labels = opts.labels || {};

  if (msg.heartbeatRelay) {
    const relayLabel = labels.heartbeatRelay || t("chat.heartbeat_relay");
    let html = `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${relayLabel}</div>`;
    if (msg.heartbeatText) html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    return html;
  }
  if (msg.afterHeartbeatRelay && !msg.text) {
    const doneLabel = labels.heartbeatRelayDone || t("chat.heartbeat_relay_done");
    return `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${doneLabel}</div>`;
  }
  const visibleText = msg._displayText || msg.text;
  if (visibleText) {
    let html = renderMarkdown(visibleText, opts.animaName);
    if (msg.streaming) {
      msg._mdCache = { html, len: visibleText.length, t: performance.now() };
      html += '<span class="streaming-tail"></span>';
    } else {
      delete msg._mdCache;
    }
    html += '<span class="streaming-cursor">▌</span>';
    if (msg.compressing) {
      const compLabel = labels.compressing || t("chat.compressing");
      html += `<div class="compression-indicator"><span class="tool-spinner"></span>${compLabel}</div>`;
    }
    return html;
  }
  let html = '<span class="streaming-cursor">▌</span>';
  if (msg.compressing) {
    const compLabel = labels.compressing || t("chat.compressing");
    html += `<div class="compression-indicator"><span class="tool-spinner"></span>${compLabel}</div>`;
  }
  return html;
}

function _renderToolZoneContent(msg, opts) {
  const { escapeHtml } = opts;
  const labels = opts.labels || {};
  const history = msg.toolHistory;
  if (history && history.length > 0) {
    return renderToolActivityTimeline(history, msg.activeTool, { escapeHtml, labels });
  }
  if (msg.activeTool) {
    const toolLabel = labels.toolRunning || ((tool) => t("chat.tool_running", { tool }));
    return `<div class="tool-indicator"><span class="tool-spinner"></span>${typeof toolLabel === "function" ? toolLabel(msg.activeTool) : toolLabel}</div>`;
  }
  return "";
}

function _renderSubordinateZoneContent(msg, opts) {
  const { escapeHtml } = opts;
  const subActivity = msg.subordinateActivity;
  if (!subActivity || Object.keys(subActivity).length === 0) return "";
  let html = "";
  for (const [subName, act] of Object.entries(subActivity)) {
    if (act.type === "inbox_processing_end") continue;
    const icon = act.type === "inbox_processing_start"
      ? "⏳" : (act.type === "tool_end" || act.type === "tool_use") ? "✓" : "🔧";
    const label = act.summary || act.tool || act.type;
    html += `<div class="subordinate-activity subordinate-activity--animate" data-sub-name="${escapeHtml(subName)}">
      <img class="subordinate-avatar" src="/api/animas/${encodeURIComponent(subName)}/avatar" alt="" onerror="this.style.display='none'">
      <span class="subordinate-name">${escapeHtml(subName)}</span>
      <span class="subordinate-tool">${icon} ${escapeHtml(label)}</span>
    </div>`;
  }
  return html;
}

/**
 * Patch the subordinate zone with minimal DOM ops — reuse existing elements
 * if the same anima name is already present, avoiding img re-fetch.
 */
function _patchSubordinateZone(container, msg, opts) {
  const { escapeHtml } = opts;
  const subActivity = msg.subordinateActivity;
  if (!subActivity || Object.keys(subActivity).length === 0) {
    if (container.innerHTML) container.innerHTML = "";
    return;
  }

  const existingByName = new Map();
  for (const el of container.querySelectorAll(".subordinate-activity[data-sub-name]")) {
    existingByName.set(el.dataset.subName, el);
  }

  const desiredNames = new Set();
  for (const [subName, act] of Object.entries(subActivity)) {
    if (act.type === "inbox_processing_end") continue;
    desiredNames.add(subName);

    const icon = act.type === "inbox_processing_start"
      ? "⏳" : (act.type === "tool_end" || act.type === "tool_use") ? "✓" : "🔧";
    const label = act.summary || act.tool || act.type;

    const existing = existingByName.get(subName);
    if (existing) {
      const toolSpan = existing.querySelector(".subordinate-tool");
      if (toolSpan) toolSpan.textContent = `${icon} ${label}`;
    } else {
      const div = document.createElement("div");
      div.className = "subordinate-activity subordinate-activity--animate";
      div.dataset.subName = subName;
      div.innerHTML = `<img class="subordinate-avatar" src="/api/animas/${encodeURIComponent(subName)}/avatar" alt="" onerror="this.style.display='none'">` +
        `<span class="subordinate-name">${escapeHtml(subName)}</span>` +
        `<span class="subordinate-tool">${icon} ${escapeHtml(label)}</span>`;
      container.appendChild(div);
    }
  }

  for (const [name, el] of existingByName) {
    if (!desiredNames.has(name)) el.remove();
  }
}

function _renderThinkingZoneContent(msg, opts) {
  const { escapeHtml } = opts;
  const visibleThinking = msg._displayThinkingText || msg.thinkingText;

  // ストリーミング中: インラインプレビュー
  if (msg.thinking && visibleThinking) {
    return `<div class="thinking-inline-preview">${escapeHtml(visibleThinking)}</div>`;
  }

  // 完了後: 折りたたみブロック（thinkingTextが残っている場合）
  if (!msg.thinking && msg.thinkingText && msg.streaming === false) {
    return `<details class="thinking-block"><summary class="thinking-summary">\u{1F4AD} Thinking</summary><pre class="thinking-content">${escapeHtml(msg.thinkingText)}</pre></details>`;
  }

  return "";
}

/**
 * Render a compact tool activity timeline.
 * Running tools are always visible; completed tools are collapsed behind a count.
 */
function renderToolActivityTimeline(history, activeTool, { escapeHtml, labels }) {
  const completedEntries = history.filter(e => e.completed);
  const runningEntries = history.filter(e => !e.completed);
  const completedCount = completedEntries.length;

  let runningHtml = "";
  for (const entry of runningEntries) {
    const detailSpan = entry.detail
      ? `<span class="tool-activity-detail">${escapeHtml(entry.detail.slice(0, 120))}</span>`
      : "";
    runningHtml += `<div class="tool-activity-item tool-activity-item--running"><span class="tool-spinner"></span><span class="tool-activity-name">${escapeHtml(entry.tool_name)}</span>${detailSpan}<span class="tool-activity-dur">${t("chat.tool_running_label")}</span></div>`;
  }

  const summaryLabel = activeTool
    ? t("chat.tools_progress", { tool: activeTool, completed: completedCount, total: history.length })
    : t("chat.tools_completed", { count: completedCount });

  let completedHtml = "";
  if (completedCount > 0) {
    let completedItems = "";
    for (const entry of completedEntries) {
      const icon = entry.is_error
        ? '<span class="tool-activity-icon tool-activity-error">\u2717</span>'
        : '<span class="tool-activity-icon tool-activity-ok">\u2713</span>';
      const dur = entry.duration_ms != null ? `<span class="tool-activity-dur">${_formatDuration(entry.duration_ms)}</span>` : "";
      const summary = entry.result_summary
        ? `<span class="tool-activity-summary">${escapeHtml(entry.result_summary.slice(0, 120))}</span>`
        : "";
      completedItems += `<div class="tool-activity-item${entry.is_error ? " tool-activity-item--error" : ""}">${icon}<span class="tool-activity-name">${escapeHtml(entry.tool_name)}</span>${dur}${summary}</div>`;
    }
    const completedLabel = t("chat.tools_completed_details", { count: completedCount });
    completedHtml = `<details class="tool-activity-completed"><summary class="tool-activity-completed-summary">${completedLabel}</summary><div class="tool-activity-list">${completedItems}</div></details>`;
  }

  return `<div class="tool-activity-timeline"><div class="tool-activity-header"><span class="tool-spinner"${activeTool ? "" : ' style="display:none"'}></span>${summaryLabel}</div>${runningHtml}${completedHtml}</div>`;
}

function _formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// ── Bubble Action Handlers ──────────────────────

/**
 * Bind copy/download action handlers for assistant bubble action bars.
 * Uses event delegation on the container (single listener).
 * @param {HTMLElement|null} container
 */
export function bindBubbleActionHandlers(container) {
  if (!container || container.dataset.bubbleActionsBound) return;
  container.dataset.bubbleActionsBound = "1";

  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".bubble-action-btn");
    if (btn) {
      const bubble = btn.closest(".chat-bubble.assistant");
      if (!bubble) return;
      const rawText = bubble.dataset.rawText || bubble.textContent || "";
      const action = btn.dataset.action;
      if (action === "copy") _copyToClipboard(rawText, btn);
      else if (action === "download") _downloadAsText(rawText, bubble);
      return;
    }

    // Mobile: tap on bubble toggles action bar visibility
    const bubble = e.target.closest(".chat-bubble.assistant");
    if (bubble && !e.target.closest(".tool-call-row") && !e.target.closest(".tool-call-group-header") && !e.target.closest("a") && !e.target.closest(".text-artifact-card")) {
      const actions = bubble.querySelector(".bubble-actions");
      if (actions) {
        container.querySelectorAll(".bubble-actions.visible").forEach(a => {
          if (a !== actions) a.classList.remove("visible");
        });
        actions.classList.toggle("visible");
      }
    }
  });
}

async function _copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0;";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  const icon = btn.querySelector("[data-lucide]");
  if (icon) {
    const orig = icon.getAttribute("data-lucide");
    icon.setAttribute("data-lucide", "check");
    if (window.lucide) lucide.createIcons({ nodes: [icon] });
    setTimeout(() => {
      icon.setAttribute("data-lucide", orig);
      if (window.lucide) lucide.createIcons({ nodes: [icon] });
    }, 1500);
  }
}

function _downloadAsText(text, bubble) {
  const ts = _extractBubbleTimestamp(bubble);
  const filename = `response_${ts}.txt`;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

function _extractBubbleTimestamp(bubble) {
  const tsEl = bubble.querySelector(".chat-ts");
  if (tsEl) {
    const tsText = tsEl.textContent.trim();
    if (tsText) {
      const d = new Date(tsText);
      if (!isNaN(d.getTime())) return _formatTimestamp(d);
    }
  }
  return _formatTimestamp(new Date());
}

function _formatTimestamp(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
