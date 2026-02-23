/* ── Chat ──────────────────────────────────── */

import { state, dom, escapeHtml, renderMarkdown, smartTimestamp } from "./state.js";
import { addActivity } from "./activity.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../shared/chat-stream.js";
import { createLogger } from "../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../shared/image-input.js";
import { api } from "./api.js";

const logger = createLogger("chat");

let imageInputManager = null;

// ── Conversation History State ────────────────
// Per-anima history state loaded from the activity_log API
const _historyState = {};  // { [name]: { sessions, hasMore, nextBefore, loading } }

const HISTORY_PAGE_SIZE = 50;
const TOOL_RESULT_TRUNCATE = 500;

// ── History API ───────────────────────────────

/**
 * Load conversation history from the activity_log-based API.
 * Returns { sessions, has_more, next_before } or null on error.
 */
async function _fetchConversationHistory(animaName, limit = HISTORY_PAGE_SIZE, before = null) {
  let url = `/api/animas/${encodeURIComponent(animaName)}/conversation/history?limit=${limit}`;
  if (before) {
    url += `&before=${encodeURIComponent(before)}`;
  }
  try {
    return await api(url);
  } catch (err) {
    logger.error("Failed to fetch conversation history", { anima: animaName, error: err.message });
    return null;
  }
}

/**
 * Load initial conversation history for the selected anima.
 * Called when a chat page loads / anima is selected.
 */
export async function loadConversationHistory(animaName) {
  if (!animaName) return;

  const hs = _historyState[animaName];
  if (hs && hs.sessions && hs.sessions.length > 0) {
    // Already loaded, just render
    renderChat();
    return;
  }

  _historyState[animaName] = { sessions: [], hasMore: false, nextBefore: null, loading: true };
  renderChat();  // Shows loading indicator

  const data = await _fetchConversationHistory(animaName);
  if (data && data.sessions) {
    _historyState[animaName] = {
      sessions: data.sessions,
      hasMore: data.has_more || false,
      nextBefore: data.next_before || null,
      loading: false,
    };
  } else {
    _historyState[animaName] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
  }

  renderChat();
}

/**
 * Load more (older) conversation history for infinite scroll.
 */
async function _loadMoreHistory() {
  const name = state.selectedAnima;
  if (!name) return;

  const hs = _historyState[name];
  if (!hs || !hs.hasMore || hs.loading) return;

  hs.loading = true;
  _renderLoadingIndicator(true);

  const data = await _fetchConversationHistory(name, HISTORY_PAGE_SIZE, hs.nextBefore);
  if (data && data.sessions && data.sessions.length > 0) {
    // Prepend older sessions
    hs.sessions = [...data.sessions, ...hs.sessions];
    hs.hasMore = data.has_more || false;
    hs.nextBefore = data.next_before || null;
  } else {
    hs.hasMore = false;
  }
  hs.loading = false;

  // Re-render preserving scroll position
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (chatMessages) {
    const prevScrollHeight = chatMessages.scrollHeight;
    _renderHistoryMessages();
    const newScrollHeight = chatMessages.scrollHeight;
    chatMessages.scrollTop += (newScrollHeight - prevScrollHeight);
  }
}

// ── Tool Call Rendering ─────────────────────

function _renderToolCalls(toolCalls) {
  if (!toolCalls || toolCalls.length === 0) return "";

  return toolCalls.map((tc, idx) => {
    const errorClass = tc.is_error ? " tool-call-error" : "";
    const toolName = escapeHtml(tc.tool_name || "unknown");
    const errorLabel = tc.is_error ? " [ERROR]" : "";

    return `<div class="tool-call-row${errorClass}" data-tool-idx="${idx}">` +
      `<span class="tool-call-row-icon">\u25B6</span>` +
      `<span class="tool-call-row-name">${toolName}${errorLabel}</span>` +
      `</div>` +
      `<div class="tool-call-detail" data-tool-idx="${idx}" style="display:none;">` +
      _renderToolCallDetail(tc) +
      `</div>`;
  }).join("");
}

function _renderToolCallDetail(tc) {
  let html = "";

  // Input
  const input = tc.input || "";
  if (input) {
    const inputStr = typeof input === "string" ? input : JSON.stringify(input, null, 2);
    html += `<div class="tool-call-label">\u5165\u529B</div>`;
    html += `<div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  // Result
  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">\u7D50\u679C</div>`;
    if (resultStr.length > TOOL_RESULT_TRUNCATE) {
      const truncated = resultStr.slice(0, TOOL_RESULT_TRUNCATE);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">\u3082\u3063\u3068\u898B\u308B</button>`;
    } else {
      html += `<div class="tool-call-content">${escapeHtml(resultStr)}</div>`;
    }
  }

  return html;
}

/**
 * Bind click handlers for tool call rows (expand/collapse) and "show more" buttons.
 * Called after rendering history messages into the DOM.
 */
function _bindToolCallHandlers(container) {
  if (!container) return;

  // Tool call row toggle
  container.querySelectorAll(".tool-call-row").forEach(row => {
    row.addEventListener("click", () => {
      const idx = row.dataset.toolIdx;
      const detail = row.nextElementSibling;
      if (!detail || detail.dataset.toolIdx !== idx) return;

      const isExpanded = row.classList.contains("expanded");
      if (isExpanded) {
        row.classList.remove("expanded");
        detail.style.display = "none";
      } else {
        row.classList.add("expanded");
        detail.style.display = "";
      }
    });
  });

  // "Show more" buttons
  container.querySelectorAll(".tool-call-show-more").forEach(btn => {
    btn.addEventListener("click", (e) => {
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

// ── Session Divider Rendering ─────────────────

function _renderSessionDivider(session, isFirst) {
  if (isFirst) return "";  // No divider before the first session

  const trigger = session.trigger || "chat";
  let label = "";
  let extraClass = "";

  if (trigger === "heartbeat") {
    label = "\u2764 \u30CF\u30FC\u30C8\u30D3\u30FC\u30C8";
    extraClass = " session-divider-heartbeat";
  } else if (trigger === "cron") {
    label = "\u23F0 Cron\u30BF\u30B9\u30AF";
    extraClass = " session-divider-cron";
  } else {
    const ts = session.session_start ? smartTimestamp(session.session_start) : "";
    label = ts;
  }

  return `<div class="session-divider${extraClass}">` +
    `<span class="session-divider-label">${escapeHtml(label)}</span>` +
    `</div>`;
}

// ── Render History Messages ──────────────────

function _renderHistoryMessages() {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;

  const name = state.selectedAnima;
  const hs = _historyState[name];
  const history = state.chatHistories[name] || [];

  // If no history data at all and no streaming messages
  if ((!hs || hs.sessions.length === 0) && history.length === 0) {
    if (hs && hs.loading) {
      chatMessages.innerHTML = '<div class="chat-empty"><span class="tool-spinner"></span> \u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
    } else {
      chatMessages.innerHTML = '<div class="chat-empty">\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u307E\u3060\u3042\u308A\u307E\u305B\u3093</div>';
    }
    return;
  }

  let html = "";

  // Loading indicator at top for infinite scroll
  if (hs && hs.hasMore) {
    if (hs.loading) {
      html += '<div class="history-loading-more"><span class="tool-spinner"></span> \u904E\u53BB\u306E\u4F1A\u8A71\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
    }
    html += '<div class="chat-load-sentinel"></div>';
  }

  // Render sessions from history API
  if (hs && hs.sessions.length > 0) {
    for (let si = 0; si < hs.sessions.length; si++) {
      const session = hs.sessions[si];
      html += _renderSessionDivider(session, si === 0);

      if (session.messages) {
        for (const msg of session.messages) {
          html += _renderHistoryMessage(msg);
        }
      }
    }
  }

  // Render live chat messages (from current streaming session)
  if (history.length > 0) {
    // Add a session divider if there are history sessions before
    if (hs && hs.sessions.length > 0) {
      html += '<div class="session-divider"><span class="session-divider-label">\u73FE\u5728\u306E\u30BB\u30C3\u30B7\u30E7\u30F3</span></div>';
    }
    html += history.map((m) => _renderLiveChatMessage(m)).join("");
  }

  chatMessages.innerHTML = html;

  // Bind tool call handlers for history messages
  _bindToolCallHandlers(chatMessages);
}

function _renderHistoryMessage(msg) {
  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "system") {
    return `<div class="chat-bubble assistant" style="opacity:0.7; font-style:italic;">${escapeHtml(msg.content || "")}${tsHtml}</div>`;
  }

  if (msg.role === "assistant") {
    const content = msg.content ? renderMarkdown(msg.content) : "";
    const toolHtml = _renderToolCalls(msg.tool_calls);
    return `<div class="chat-bubble assistant">${content}${toolHtml}${tsHtml}</div>`;
  }

  // human / user
  const fromLabel = msg.from_person && msg.from_person !== "human"
    ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">${escapeHtml(msg.from_person)}</div>`
    : "";
  return `<div class="chat-bubble user">${fromLabel}<div class="chat-text">${escapeHtml(msg.content || "")}</div>${tsHtml}</div>`;
}

function _renderLiveChatMessage(m) {
  if (m.role === "thinking") {
    return `<div class="chat-bubble thinking"><span class="thinking-animation">\u8003\u3048\u4E2D</span></div>`;
  }
  if (m.role === "assistant") {
    const streamClass = m.streaming ? " streaming" : "";
    const notifClass = m.notification ? " notification" : "";
    let content = "";
    if (m.text) {
      content = renderMarkdown(m.text);
    } else if (m.streaming) {
      content = '<span class="cursor-blink"></span>';
    }
    const bootstrapHtml = m.bootstrapping
      ? `<div class="bootstrap-indicator"><span class="tool-spinner"></span>\u521D\u671F\u5316\u4E2D...</div>`
      : "";
    const toolHtml = m.activeTool
      ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(m.activeTool)} \u3092\u5B9F\u884C\u4E2D...</div>`
      : "";
    return `<div class="chat-bubble assistant${streamClass}${notifClass}">${content}${bootstrapHtml}${toolHtml}</div>`;
  }
  const imagesHtml = renderChatImages(m.images);
  const textHtml = m.text ? `<div class="chat-text">${escapeHtml(m.text)}</div>` : "";
  return `<div class="chat-bubble user">${imagesHtml}${textHtml}</div>`;
}

function _renderLoadingIndicator(show) {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;
  const existing = chatMessages.querySelector(".history-loading-more");
  if (show && !existing) {
    const indicator = document.createElement("div");
    indicator.className = "history-loading-more";
    indicator.innerHTML = '<span class="tool-spinner"></span> \u904E\u53BB\u306E\u4F1A\u8A71\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...';
    chatMessages.insertBefore(indicator, chatMessages.firstChild);
  } else if (!show && existing) {
    existing.remove();
  }
}

// ── Infinite Scroll (Upward) ──────────────────

let _scrollObserver = null;

export function setupScrollObserver() {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;

  if (_scrollObserver) _scrollObserver.disconnect();

  _scrollObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          _loadMoreHistory();
        }
      }
    },
    { root: chatMessages, rootMargin: "200px 0px 0px 0px" },
  );

  _observeSentinel();
}

function _observeSentinel() {
  if (!_scrollObserver) return;
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;
  const sentinel = chatMessages.querySelector(".chat-load-sentinel");
  if (sentinel) _scrollObserver.observe(sentinel);
}

// ── Render ─────────────────────────────────

export function renderChat() {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return; // Chat not in DOM (page not active)

  const name = state.selectedAnima;

  // Render history messages + live chat messages
  _renderHistoryMessages();

  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Re-observe sentinel after render
  _observeSentinel();
}

// ── SSE Streaming ─────────────────────────

function renderStreamingBubble(msg) {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;
  const bubble = chatMessages.querySelector(".chat-bubble.assistant.streaming");
  if (!bubble) return;

  let html = "";

  if (msg.heartbeatRelay) {
    html += '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>\u30CF\u30FC\u30C8\u30D3\u30FC\u30C8\u51E6\u7406\u4E2D...</div>';
    if (msg.heartbeatText) {
      html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    }
  } else if (msg.afterHeartbeatRelay && !msg.text) {
    html = '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>\u5FDC\u7B54\u3092\u6E96\u5099\u4E2D...</div>';
  } else if (msg.text) {
    try {
      html = marked.parse(msg.text, { breaks: true });
    } catch {
      html = escapeHtml(msg.text);
    }
  } else {
    html = '<span class="cursor-blink"></span>';
  }

  if (msg.bootstrapping) {
    html += `<div class="bootstrap-indicator"><span class="tool-spinner"></span>\u521D\u671F\u5316\u4E2D...</div>`;
  }

  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} \u3092\u5B9F\u884C\u4E2D...</div>`;
  }

  bubble.innerHTML = html;
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

export async function sendChat(message) {
  const name = state.selectedAnima;
  const images = imageInputManager?.getPendingImages() || [];
  if (!name || (!message.trim() && images.length === 0)) return;

  // Guard: block sending to bootstrapping animas
  const currentAnima = state.animas.find((p) => p.name === name);
  if (currentAnima?.status === "bootstrapping" || currentAnima?.bootstrapping) {
    const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
    if (chatMessages) {
      const systemMsg = document.createElement("div");
      systemMsg.className = "chat-bubble assistant";
      systemMsg.textContent = "\u3053\u306E\u30AD\u30E3\u30E9\u30AF\u30BF\u30FC\u306F\u73FE\u5728\u5236\u4F5C\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002";
      chatMessages.appendChild(systemMsg);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    return;
  }

  if (!state.chatHistories[name]) state.chatHistories[name] = [];
  const history = state.chatHistories[name];

  // Capture display images (with dataUrl for rendering in chat)
  const displayImages = imageInputManager?.getDisplayImages() || [];

  // Add user message + empty streaming bubble
  history.push({ role: "user", text: message, images: displayImages });
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null };
  history.push(streamingMsg);
  renderChat();

  const chatInput = dom.chatInput || document.getElementById("chatInput");
  const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (chatInput) chatInput.disabled = true;
  if (chatSendBtn) chatSendBtn.disabled = true;
  addActivity("chat", name, `\u30E6\u30FC\u30B6\u30FC: ${message}`);

  // Clear images after capturing
  imageInputManager?.clearImages();

  try {
    const bodyObj = { message, from_person: state.currentUser || "human" };
    if (images.length > 0) {
      bodyObj.images = images;
    }
    const body = JSON.stringify(bodyObj);

    logger.info(`[SSE-UI] sendChat START anima=${name} msg_len=${message.length}`);
    await streamChat(name, body, null, {
      onTextDelta: (text) => {
        streamingMsg.afterHeartbeatRelay = false;
        streamingMsg.text += text;
        logger.debug(`onTextDelta: delta_len=${text.length} total_len=${streamingMsg.text.length}`);
        renderStreamingBubble(streamingMsg);
      },
      onToolStart: (toolName) => {
        logger.debug(`onToolStart: ${toolName}`);
        streamingMsg.activeTool = toolName;
        renderStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {
        logger.debug("onToolEnd");
      },
      onBootstrap: (data) => {
        logger.debug(`onBootstrap: status=${data.status}`);
        if (data.status === "started") {
          streamingMsg.bootstrapping = true;
          renderStreamingBubble(streamingMsg);
        } else if (data.status === "completed") {
          streamingMsg.bootstrapping = false;
          renderStreamingBubble(streamingMsg);
        } else if (data.status === "busy") {
          streamingMsg.text = data.message || "\u73FE\u5728\u521D\u671F\u5316\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002";
          streamingMsg.streaming = false;
          streamingMsg.bootstrapping = false;
          renderChat();
          addActivity("system", name, "\u30D6\u30FC\u30C8\u30B9\u30C8\u30E9\u30C3\u30D7\u4E2D\u306E\u305F\u3081\u5FDC\u7B54\u4FDD\u7559");
        }
      },
      onChainStart: () => {
        logger.debug("onChainStart");
      },
      onHeartbeatRelayStart: ({ message }) => {
        logger.debug(`onHeartbeatRelayStart: ${message}`);
        streamingMsg.heartbeatRelay = true;
        streamingMsg.heartbeatText = "";
        streamingMsg.text = "";
        renderStreamingBubble(streamingMsg);
        addActivity("system", name, `\u30CF\u30FC\u30C8\u30D3\u30FC\u30C8\u4E2D\u7D99: ${message}`);
      },
      onHeartbeatRelay: ({ text }) => {
        streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
        logger.debug(`onHeartbeatRelay: delta_len=${text.length} total_len=${(streamingMsg.heartbeatText || "").length}`);
        renderStreamingBubble(streamingMsg);
      },
      onHeartbeatRelayDone: () => {
        logger.debug(`onHeartbeatRelayDone: transitioning to afterHeartbeatRelay, text_len=${streamingMsg.text.length}`);
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = true;
        renderStreamingBubble(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        logger.debug(`onError: ${errorMsg}`);
        streamingMsg.text += `\n[\u30A8\u30E9\u30FC] ${errorMsg}`;
        streamingMsg.streaming = false;
        renderChat();
      },
      onDone: ({ summary }) => {
        const summaryLen = (summary || "").length;
        const textLen = streamingMsg.text.length;
        logger.debug(`onDone: summary_len=${summaryLen} text_len=${textLen} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
        const text = summary || streamingMsg.text;
        streamingMsg.text = text || "(\u7A7A\u306E\u5FDC\u7B54)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = false;
        logger.debug(`onDone: final text_len=${streamingMsg.text.length}`);
        renderChat();
        addActivity("chat", name, `\u5FDC\u7B54: ${streamingMsg.text.slice(0, 100)}`);
      },
    });

    // Ensure streaming is finalized if stream ended without done event
    if (streamingMsg.streaming) {
      logger.info(`[SSE-UI] sendChat FINALIZE_FALLBACK anima=${name} text_len=${streamingMsg.text.length} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
      streamingMsg.streaming = false;
      if (!streamingMsg.text) {
        streamingMsg.text = streamingMsg.afterHeartbeatRelay
          ? "(\u5FDC\u7B54\u306E\u53D7\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044)"
          : "(\u7A7A\u306E\u5FDC\u7B54)";
      }
      streamingMsg.afterHeartbeatRelay = false;
      renderChat();
    }
    logger.info(`[SSE-UI] sendChat COMPLETE anima=${name} final_text_len=${streamingMsg.text.length}`);
  } catch (err) {
    if (err.name !== "AbortError") {
      logger.error("Chat stream error", { anima: name, error: err.message, name: err.name });
    }
    logger.info(`[SSE-UI] sendChat ERROR anima=${name} error=${err.name}:${err.message}`);
    streamingMsg.text = `[\u30A8\u30E9\u30FC] ${err.message}`;
    streamingMsg.streaming = false;
    streamingMsg.activeTool = null;
    renderChat();
  } finally {
    const chatInput = dom.chatInput || document.getElementById("chatInput");
    const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
    if (chatInput) {
      chatInput.disabled = false;
      chatInput.focus();
    }
    if (chatSendBtn) chatSendBtn.disabled = false;
  }
}

// ── Image Input Initialization ──────────────

export function initImageInput() {
  const chatContainer = document.querySelector(".panel-main");
  const chatInputForm = document.querySelector(".chat-input-form");
  const chatInput = dom.chatInput || document.getElementById("chatInput");

  if (!chatContainer || !chatInputForm || !chatInput) return;

  // Create preview container above input
  const previewEl = document.createElement("div");
  previewEl.className = "image-preview-bar";
  previewEl.style.display = "none";
  chatInputForm.insertBefore(previewEl, chatInputForm.firstChild);

  // Create file input + button
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/jpeg,image/png,image/gif,image/webp";
  fileInput.multiple = true;
  fileInput.style.display = "none";

  const attachBtn = document.createElement("button");
  attachBtn.type = "button";
  attachBtn.className = "chat-attach-btn";
  attachBtn.textContent = "+";
  attachBtn.title = "\u753B\u50CF\u3092\u6DFB\u4ED8";
  attachBtn.addEventListener("click", () => fileInput.click());

  // Insert attach button before send button
  const sendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (sendBtn) {
    chatInputForm.insertBefore(attachBtn, sendBtn);
  } else {
    chatInputForm.appendChild(attachBtn);
  }
  chatInputForm.appendChild(fileInput);

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      imageInputManager.addFiles(fileInput.files);
      fileInput.value = "";
    }
  });

  imageInputManager = createImageInput({
    container: chatContainer,
    inputArea: chatInput,
    previewContainer: previewEl,
  });

  // Initialize lightbox for image clicks
  initLightbox();
}

/**
 * Resume an active SSE stream after page reload.
 * Called when anima status is "thinking" or "processing".
 */
export async function resumeActiveStream(animaName) {
  try {
    logger.info(`[SSE-UI] resumeActiveStream START anima=${animaName}`);
    const active = await fetchActiveStream(animaName);
    if (!active || active.status !== "streaming") {
      logger.info(`[SSE-UI] resumeActiveStream no active stream anima=${animaName} active=${JSON.stringify(active)}`);
      return;
    }

    logger.info(`[SSE-UI] resumeActiveStream found active stream anima=${animaName} responseId=${active.response_id} events=${active.event_count}`);
    const progress = await fetchStreamProgress(animaName, active.response_id);
    if (!progress) {
      logger.info(`[SSE-UI] resumeActiveStream no progress anima=${animaName}`);
      return;
    }
    logger.info(`[SSE-UI] resumeActiveStream progress anima=${animaName} status=${progress.status} text_len=${(progress.full_text||"").length} lastEventId=${progress.last_event_id}`);

    // Show accumulated text in streaming bubble
    if (!state.chatHistories[animaName]) state.chatHistories[animaName] = [];
    const history = state.chatHistories[animaName];
    const streamingMsg = {
      role: "assistant",
      text: progress.full_text || "",
      streaming: true,
      activeTool: progress.active_tool || null,
    };
    history.push(streamingMsg);
    renderChat();

    // Resume SSE stream
    const resumeBody = JSON.stringify({
      message: "",
      from_person: state.currentUser || "human",
      resume: active.response_id,
      last_event_id: progress.last_event_id || "",
    });

    const chatInput = dom.chatInput || document.getElementById("chatInput");
    const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
    if (chatInput) chatInput.disabled = true;
    if (chatSendBtn) chatSendBtn.disabled = true;

    await streamChat(animaName, resumeBody, null, {
      onTextDelta: (text) => {
        streamingMsg.text += text;
        renderStreamingBubble(streamingMsg);
      },
      onToolStart: (toolName) => {
        streamingMsg.activeTool = toolName;
        renderStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {},
      onReconnecting: () => {
        streamingMsg.activeTool = "\u518D\u63A5\u7D9A\u4E2D...";
        renderStreamingBubble(streamingMsg);
      },
      onReconnected: () => {
        streamingMsg.activeTool = null;
        renderStreamingBubble(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[\u30A8\u30E9\u30FC] ${errorMsg}`;
        streamingMsg.streaming = false;
        renderChat();
      },
      onDone: ({ summary }) => {
        const text = summary || streamingMsg.text;
        streamingMsg.text = text || "(\u7A7A\u306E\u5FDC\u7B54)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.afterHeartbeatRelay = false;
        renderChat();
      },
    });

    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(\u7A7A\u306E\u5FDC\u7B54)";
      streamingMsg.afterHeartbeatRelay = false;
      renderChat();
    }
  } catch (err) {
    logger.info(`[SSE-UI] resumeActiveStream ERROR anima=${animaName} err=${err.name}:${err.message}`);
    logger.error("Resume stream error", { anima: animaName, error: err.message });
  } finally {
    const chatInput = dom.chatInput || document.getElementById("chatInput");
    const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
    if (chatInput) { chatInput.disabled = false; chatInput.focus(); }
    if (chatSendBtn) chatSendBtn.disabled = false;
  }
}

/**
 * Get the history state for an anima (for use by other modules like history.js).
 */
export function getHistoryState(animaName) {
  return _historyState[animaName] || null;
}

/**
 * Clear history state for an anima (e.g., on anima switch).
 */
export function clearHistoryState(animaName) {
  delete _historyState[animaName];
}

/**
 * Exported helpers for history.js to use the same rendering functions.
 */
export { _renderHistoryMessage as renderHistoryMessage };
export { _renderToolCalls as renderToolCalls };
export { _renderSessionDivider as renderSessionDivider };
export { _bindToolCallHandlers as bindToolCallHandlers };
export { _fetchConversationHistory as fetchConversationHistory };
