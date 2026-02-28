// ── Shared Chat Render Utilities ──────────────────────
// Pure HTML-generating functions used by both Dashboard and Workspace chat UIs.
// All functions are DOM-independent (return HTML strings) except bindToolCallHandlers.

const DEFAULT_TOOL_RESULT_TRUNCATE = 500;

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

  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "system") {
    return `<div class="chat-bubble assistant" style="opacity:0.7; font-style:italic;">${escapeHtml(msg.content || "")}${tsHtml}</div>`;
  }

  if (msg.role === "assistant") {
    const content = msg.content ? renderMarkdown(msg.content, opts.animaName) : "";
    const toolHtml = renderToolCalls(msg.tool_calls, { escapeHtml, truncateLen: truncLen });
    const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });
    return `<div class="chat-bubble assistant">${content}${imagesHtml}${toolHtml}${tsHtml}</div>`;
  }

  const fromLabel = msg.from_person && msg.from_person !== "human"
    ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">${escapeHtml(msg.from_person)}</div>`
    : "";
  return `<div class="chat-bubble user">${fromLabel}<div class="chat-text">${escapeHtml(msg.content || "")}</div>${tsHtml}</div>`;
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

  return toolCalls.map((tc, idx) => {
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

  const ts = msg.timestamp ? smartTimestamp(msg.timestamp) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "thinking") {
    const thinkLabel = labels.thinking || "考え中...";
    return `<div class="chat-bubble thinking"><span class="thinking-animation">${thinkLabel}</span></div>`;
  }

  if (msg.role === "system") {
    return `<div class="chat-visit-marker">${escapeHtml(msg.text)}${tsHtml}</div>`;
  }

  if (msg.role === "user") {
    const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });
    const textHtml = msg.text ? `<div class="chat-text">${escapeHtml(msg.text)}</div>` : "";
    return `<div class="chat-bubble user">${imagesHtml}${textHtml}${tsHtml}</div>`;
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

  let content = "";
  if (msg.text) {
    content = renderMarkdown(msg.text, opts.animaName);
  } else if (msg.streaming) {
    content = '<span class="cursor-blink"></span>';
  }

  const toolLabel = labels.toolRunning || ((tool) => `${tool} を実行中...`);
  const toolHtml = msg.activeTool
    ? `<div class="tool-indicator"><span class="tool-spinner"></span>${typeof toolLabel === "function" ? toolLabel(msg.activeTool) : toolLabel}</div>`
    : "";
  const imagesHtml = renderImages(msg.images, { animaName: opts.animaName });

  return `<div class="chat-bubble assistant${streamClass}"${streamIdAttr}>${thinkingHtml}${content}${imagesHtml}${toolHtml}${tsHtml}</div>`;
}

/**
 * Generate HTML for the streaming bubble's inner content (for incremental updates).
 * @param {object} msg - Streaming message state
 * @param {object} opts - Same as renderLiveBubble opts
 */
export function renderStreamingBubbleInner(msg, opts) {
  const { escapeHtml, renderMarkdown } = opts;
  const labels = opts.labels || {};

  const thinkingHtml = (msg.thinking && msg.thinkingText)
    ? `<div class="thinking-inline-preview">${escapeHtml(msg.thinkingText)}</div>`
    : "";

  let mainHtml = "";
  if (msg.heartbeatRelay) {
    const relayLabel = labels.heartbeatRelay || "ハートビート処理中...";
    mainHtml = `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${relayLabel}</div>`;
    if (msg.heartbeatText) {
      mainHtml += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    }
  } else if (msg.afterHeartbeatRelay && !msg.text) {
    const doneLabel = labels.heartbeatRelayDone || "応答を準備中...";
    mainHtml = `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${doneLabel}</div>`;
  } else if (msg.text) {
    mainHtml = renderMarkdown(msg.text, opts.animaName);
  } else {
    mainHtml = '<span class="cursor-blink"></span>';
  }

  let html = `${thinkingHtml}${mainHtml}`;
  if (msg.activeTool) {
    const toolLabel = labels.toolRunning || ((tool) => `${tool} を実行中...`);
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${typeof toolLabel === "function" ? toolLabel(msg.activeTool) : toolLabel}</div>`;
  }
  return html;
}
