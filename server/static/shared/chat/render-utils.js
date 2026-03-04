// ── Shared Chat Render Utilities ──────────────────────
// Pure HTML-generating functions used by both Dashboard and Workspace chat UIs.
// All functions are DOM-independent (return HTML strings) except bindToolCallHandlers.

const DEFAULT_TOOL_RESULT_TRUNCATE = 500;

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
    const content = msg.content ? renderMarkdown(msg.content, opts.animaName) : "";
    const toolHtml = renderToolCalls(msg.tool_calls, { escapeHtml, truncateLen: truncLen });
    const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });
    const toLabel = msg.to_person
      ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">→ ${escapeHtml(msg.to_person)}</div>`
      : "";
    const bubble = `<div class="chat-bubble assistant">${toLabel}${content}${imagesHtml}${toolHtml}${tsHtml}</div>`;
    return _wrapRow("assistant", bubble, _renderAvatar(opts.animaName, avatarMap));
  }

  const fromLabel = msg.from_person && msg.from_person !== "human"
    ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">${escapeHtml(msg.from_person)}</div>`
    : "";
  const userContent = _stripVoiceSuffix(msg.content || "");
  const bubble = `<div class="chat-bubble user">${fromLabel}<div class="chat-text">${escapeHtml(userContent)}</div>${tsHtml}</div>`;
  const isAnima = msg.from_person && msg.from_person !== "human";
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
    label = "❤ ハートビート";
    extraClass = " session-divider-heartbeat";
  } else if (trigger === "cron") {
    label = "⏰ Cronタスク";
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
    const errorLabel = tc.is_error ? " [ERROR]" : "";

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
    html += `<div class="tool-call-label">入力</div><div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">結果</div>`;
    if (resultStr.length > truncLen) {
      const truncated = resultStr.slice(0, truncLen);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">もっと見る</button>`;
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
    const thinkLabel = labels.thinking || "考え中...";
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
  }

  const streamingCursor = '<span class="streaming-cursor">▌</span>';
  let content = "";
  if (msg.text) {
    content = renderMarkdown(msg.text, opts.animaName);
    if (msg.streaming) content += streamingCursor;
  } else if (msg.streaming) {
    content = streamingCursor;
  }

  const compLabel = labels.compressing || "会話履歴を圧縮中...";
  const compressionHtml = msg.compressing
    ? `<div class="compression-indicator"><span class="tool-spinner"></span>${compLabel}</div>`
    : "";
  let toolHtml = "";
  const history = msg.toolHistory;
  if (history && history.length > 0) {
    toolHtml = renderToolActivityTimeline(history, msg.activeTool, { escapeHtml, labels });
  } else if (msg.activeTool) {
    const toolLabel = labels.toolRunning || ((tool) => `${tool} を実行中...`);
    toolHtml = `<div class="tool-indicator"><span class="tool-spinner"></span>${typeof toolLabel === "function" ? toolLabel(msg.activeTool) : toolLabel}</div>`;
  }
  const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });

  const bubble = `<div class="chat-bubble assistant${streamClass}"${streamIdAttr}>${content}${imagesHtml}${compressionHtml}${toolHtml}${thinkingHtml}${tsHtml}</div>`;
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
export function updateStreamingZone(bubble, msg, opts, zone = "all") {
  if (!bubble) return;
  if (zone === "all") {
    bubble.innerHTML = renderStreamingBubbleInner(msg, opts);
    return;
  }
  const sel = `.streaming-zone-${zone}`;
  const el = bubble.querySelector(sel);
  if (!el) {
    bubble.innerHTML = renderStreamingBubbleInner(msg, opts);
    return;
  }
  if (zone === "subordinate") {
    _patchSubordinateZone(el, msg, opts);
    return;
  }
  const renderers = {
    text: _renderTextZoneContent,
    tools: _renderToolZoneContent,
    thinking: _renderThinkingZoneContent,
  };
  const fn = renderers[zone];
  if (fn) el.innerHTML = fn(msg, opts);
}

function _renderTextZoneContent(msg, opts) {
  const { escapeHtml, renderMarkdown } = opts;
  const labels = opts.labels || {};

  if (msg.heartbeatRelay) {
    const relayLabel = labels.heartbeatRelay || "ハートビート処理中...";
    let html = `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${relayLabel}</div>`;
    if (msg.heartbeatText) html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    return html;
  }
  if (msg.afterHeartbeatRelay && !msg.text) {
    const doneLabel = labels.heartbeatRelayDone || "応答を準備中...";
    return `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${doneLabel}</div>`;
  }
  if (msg.text) {
    let html = renderMarkdown(msg.text, opts.animaName);
    html += '<span class="streaming-cursor">▌</span>';
    if (msg.compressing) {
      const compLabel = labels.compressing || "会話履歴を圧縮中...";
      html += `<div class="compression-indicator"><span class="tool-spinner"></span>${compLabel}</div>`;
    }
    return html;
  }
  let html = '<span class="streaming-cursor">▌</span>';
  if (msg.compressing) {
    const compLabel = labels.compressing || "会話履歴を圧縮中...";
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
    const toolLabel = labels.toolRunning || ((tool) => `${tool} を実行中...`);
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
  if (msg.thinking && msg.thinkingText) {
    return `<div class="thinking-inline-preview">${escapeHtml(msg.thinkingText)}</div>`;
  }
  return "";
}

/**
 * Render a collapsible tool activity timeline.
 */
function renderToolActivityTimeline(history, activeTool, { escapeHtml, labels }) {
  const completedCount = history.filter(e => e.completed).length;
  const totalCount = history.length;

  let items = "";
  for (const entry of history) {
    if (entry.completed) {
      const icon = entry.is_error
        ? '<span class="tool-activity-icon tool-activity-error">✗</span>'
        : '<span class="tool-activity-icon tool-activity-ok">✓</span>';
      const dur = entry.duration_ms != null ? `<span class="tool-activity-dur">${_formatDuration(entry.duration_ms)}</span>` : "";
      const summary = entry.result_summary
        ? `<span class="tool-activity-summary">${escapeHtml(entry.result_summary.slice(0, 120))}</span>`
        : "";
      items += `<div class="tool-activity-item${entry.is_error ? " tool-activity-item--error" : ""}">${icon}<span class="tool-activity-name">${escapeHtml(entry.tool_name)}</span>${dur}${summary}</div>`;
    } else {
      const detailSpan = entry.detail
        ? `<span class="tool-activity-detail">${escapeHtml(entry.detail.slice(0, 120))}</span>`
        : "";
      items += `<div class="tool-activity-item tool-activity-item--running"><span class="tool-spinner"></span><span class="tool-activity-name">${escapeHtml(entry.tool_name)}</span>${detailSpan}<span class="tool-activity-dur">実行中</span></div>`;
    }
  }

  const summaryLabel = activeTool
    ? `${escapeHtml(activeTool)} を実行中... (${completedCount}/${totalCount})`
    : `${completedCount} ツール完了`;

  return `<div class="tool-activity-timeline">
    <details${activeTool ? " open" : ""}>
      <summary class="tool-activity-header"><span class="tool-spinner"${activeTool ? "" : ' style="display:none"'}></span>${summaryLabel}</summary>
      <div class="tool-activity-list">${items}</div>
    </details>
  </div>`;
}

function _formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
