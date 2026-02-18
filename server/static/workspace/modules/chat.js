// ── Chat Module ──────────────────────────────────
// Chat UI + SSE streaming display.

import { getState, setState } from "./state.js";
import { fetchConversationFull } from "./api.js";
import { escapeHtml, renderSimpleMarkdown } from "./utils.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../../shared/chat-stream.js";
import { createLogger } from "../../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../../shared/image-input.js";

const logger = createLogger("ws-chat");

// ── Constants ──────────────────────────────────

const EMPTY_MSG = "メッセージはまだありません";
const PLACEHOLDER_DEFAULT = "Animaを選択してください";
const PAGE_SIZE = 30;

// ── DOM References ──────────────────────────────

let messagesEl = null;
let inputEl = null;
let sendBtnEl = null;
let _scrollObserver = null;
let imageInputManager = null;

// ── Render Helpers ──────────────────────────────

function renderBubble(msg) {
  if (msg.role === "user") {
    const imagesHtml = renderChatImages(msg.images);
    const textHtml = msg.text ? `<div class="chat-text">${escapeHtml(msg.text)}</div>` : "";
    return `<div class="chat-bubble user">${imagesHtml}${textHtml}</div>`;
  }

  // Assistant bubble
  const streamClass = msg.streaming ? " streaming" : "";
  let content = "";

  if (msg.text) {
    content = renderSimpleMarkdown(msg.text);
  } else if (msg.streaming) {
    content = '<span class="cursor-blink"></span>';
  }

  const toolHtml = msg.activeTool
    ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`
    : "";

  return `<div class="chat-bubble assistant${streamClass}">${content}${toolHtml}</div>`;
}

function renderAllMessages(scrollToBottom = true) {
  if (!messagesEl) return;

  const { chatMessages, chatPagination } = getState();

  if (chatMessages.length === 0) {
    messagesEl.innerHTML = `<div class="chat-empty">${EMPTY_MSG}</div>`;
    return;
  }

  // Sentinel for infinite scroll (older messages)
  let topHtml = "";
  if (chatPagination.hasMore) {
    if (chatPagination.loading) {
      topHtml += '<div class="chat-load-indicator"><span class="tool-spinner"></span> 読み込み中...</div>';
    }
    topHtml += '<div class="chat-load-sentinel"></div>';
  }

  const prevScrollHeight = messagesEl.scrollHeight;
  messagesEl.innerHTML = topHtml + chatMessages.map(renderBubble).join("");

  if (scrollToBottom) {
    requestAnimationFrame(() => {
      const last = messagesEl.lastElementChild;
      if (last) last.scrollIntoView({ block: "end", behavior: "instant" });
    });
  } else {
    // Maintain scroll position after prepending older messages
    const newScrollHeight = messagesEl.scrollHeight;
    messagesEl.scrollTop += (newScrollHeight - prevScrollHeight);
  }

  _observeSentinel();
}

// ── Streaming update with rAF throttle ──────────────────────

let _chatRafPending = false;
let _chatLatestStreamingMsg = null;
let _isSseStreaming = false;

function scheduleStreamingUpdate(msg) {
  _chatLatestStreamingMsg = msg;
  if (_chatRafPending) return;
  _chatRafPending = true;
  requestAnimationFrame(() => {
    _chatRafPending = false;
    if (_chatLatestStreamingMsg) {
      updateStreamingBubble(_chatLatestStreamingMsg);
    }
  });
}

function updateStreamingBubble(msg) {
  if (!messagesEl) return;

  const bubble = messagesEl.querySelector(".chat-bubble.assistant.streaming");
  if (!bubble) return;

  let html = "";

  if (msg.heartbeatRelay) {
    html += '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>ハートビート処理中...</div>';
    if (msg.heartbeatText) {
      html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    }
  } else if (msg.afterHeartbeatRelay && !msg.text) {
    html = '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>応答を準備中...</div>';
  } else if (msg.text) {
    html = renderSimpleMarkdown(msg.text);
  } else {
    html = '<span class="cursor-blink"></span>';
  }

  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`;
  }

  bubble.innerHTML = html;
  requestAnimationFrame(() => {
    bubble.scrollIntoView({ block: "end", behavior: "instant" });
  });
}

// ── Infinite Scroll ─────────────────────────────

function _setupScrollObserver() {
  if (_scrollObserver) _scrollObserver.disconnect();
  if (!messagesEl) return;

  _scrollObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) _loadOlderMessages();
      }
    },
    { root: messagesEl, rootMargin: "200px 0px 0px 0px" },
  );
}

function _observeSentinel() {
  if (!_scrollObserver || !messagesEl) return;
  const sentinel = messagesEl.querySelector(".chat-load-sentinel");
  if (sentinel) _scrollObserver.observe(sentinel);
}

async function _loadOlderMessages() {
  const { selectedAnima, chatPagination, chatMessages } = getState();
  if (!selectedAnima || !chatPagination.hasMore || chatPagination.loading) return;
  if (_isSseStreaming) return;

  setState({ chatPagination: { ...chatPagination, loading: true } });
  renderAllMessages(false);

  try {
    const offset = chatMessages.length;
    const data = await fetchConversationFull(selectedAnima, PAGE_SIZE, offset);
    const { chatMessages: current, chatPagination: pag } = getState();

    if (data.turns && data.turns.length > 0) {
      const older = data.turns.map((t) => ({
        role: t.role === "human" ? "user" : "assistant",
        text: t.content || "",
      }));
      const merged = [...older, ...current];
      setState({
        chatMessages: merged,
        chatPagination: {
          totalRaw: data.raw_turns,
          hasMore: data.turns.length >= PAGE_SIZE && merged.length < data.raw_turns,
          loading: false,
        },
      });
    } else {
      setState({ chatPagination: { ...pag, hasMore: false, loading: false } });
    }
  } catch (err) {
    logger.error("Failed to load older messages", { error: err.message });
    const { chatPagination: pag } = getState();
    setState({ chatPagination: { ...pag, loading: false } });
  }

  renderAllMessages(false);
}

function updateInputState() {
  if (!inputEl || !sendBtnEl) return;

  const { selectedAnima } = getState();
  const disabled = !selectedAnima;

  inputEl.disabled = disabled;
  sendBtnEl.disabled = disabled;
  const mobile = window.matchMedia("(max-width: 768px)").matches;
  const shortcut = mobile ? "Enter" : "Ctrl+Enter";
  inputEl.placeholder = selectedAnima
    ? `${selectedAnima} にメッセージ... (${shortcut} で送信)`
    : PLACEHOLDER_DEFAULT;
}

// ── Public API ──────────────────────────────────

/**
 * Build full chat UI (messages area + input form) into the container.
 */
export function renderChat(container) {
  container.innerHTML = `
    <div class="chat-container">
      <div class="chat-messages"></div>
      <div class="chat-input-area">
        <div class="image-preview-bar" style="display:none"></div>
        <div class="chat-input-row">
          <textarea class="chat-input" rows="1" placeholder="${PLACEHOLDER_DEFAULT}" disabled></textarea>
          <button class="chat-attach-btn" type="button" title="画像を添付">+</button>
          <button class="chat-send-btn" disabled>送信</button>
        </div>
        <input type="file" class="chat-file-input" accept="image/jpeg,image/png,image/gif,image/webp" multiple style="display:none" />
      </div>
    </div>
  `;

  messagesEl = container.querySelector(".chat-messages");
  inputEl = container.querySelector(".chat-input");
  sendBtnEl = container.querySelector(".chat-send-btn");
}

/**
 * Bind event listeners after renderChat.
 */
export function initChat(container) {
  if (!inputEl || !sendBtnEl) {
    // Ensure DOM refs if renderChat wasn't called
    messagesEl = container.querySelector(".chat-messages");
    inputEl = container.querySelector(".chat-input");
    sendBtnEl = container.querySelector(".chat-send-btn");
  }

  if (!inputEl || !sendBtnEl) return;

  _setupScrollObserver();

  // ── Image Input Setup ────────────────────
  const chatContainer = container.querySelector(".chat-container");
  const previewEl = container.querySelector(".image-preview-bar");
  const attachBtn = container.querySelector(".chat-attach-btn");
  const fileInput = container.querySelector(".chat-file-input");

  if (attachBtn && fileInput) {
    attachBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        imageInputManager?.addFiles(fileInput.files);
        fileInput.value = "";
      }
    });
  }

  if (chatContainer && inputEl && previewEl) {
    imageInputManager = createImageInput({
      container: chatContainer,
      inputArea: inputEl,
      previewContainer: previewEl,
    });
  }

  // Initialize lightbox for image clicks
  initLightbox();

  // Auto-resize textarea (100px on mobile, 200px on desktop)
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    const mobile = window.matchMedia("(max-width: 768px)").matches;
    const maxH = mobile ? 100 : 200;
    inputEl.style.height = Math.min(inputEl.scrollHeight, maxH) + "px";
  });

  // Enter key handling: mobile vs desktop
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const mobile = window.matchMedia("(max-width: 768px)").matches;
      if (mobile) {
        // Mobile: Enter sends, Shift+Enter for newline
        if (!e.shiftKey) {
          e.preventDefault();
          submitFromInput();
        }
      } else {
        // Desktop: Ctrl/Cmd+Enter sends
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          submitFromInput();
        }
      }
    }
  });

  // Mobile keyboard: keep input visible above virtual keyboard
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", () => {
      if (document.activeElement === inputEl) {
        requestAnimationFrame(() => {
          inputEl.scrollIntoView({ block: "nearest" });
        });
      }
    });
  }

  // Send button click
  sendBtnEl.addEventListener("click", (e) => {
    e.preventDefault();
    submitFromInput();
  });

  // Initial render
  updateInputState();
  renderAllMessages();
}

/**
 * Send a message and begin SSE streaming.
 */
export async function sendMessage(text) {
  const { selectedAnima, currentUser } = getState();
  const images = imageInputManager?.getPendingImages() || [];
  if (!selectedAnima || (!text.trim() && images.length === 0)) return;

  const trimmed = text.trim();

  // Capture display images (with dataUrl for rendering)
  const displayImages = imageInputManager?.getDisplayImages() || [];

  // Add user message + empty streaming assistant bubble (immutable)
  const userMsg = { role: "user", text: trimmed, images: displayImages };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null };
  const updated = [...getState().chatMessages, userMsg, streamingMsg];
  setState({ chatMessages: updated });
  renderAllMessages();

  // Disable input during streaming
  setInputEnabled(false);
  _isSseStreaming = true;

  // Clear images after capturing
  imageInputManager?.clearImages();

  try {
    const bodyObj = { message: trimmed, from_person: currentUser || "human" };
    if (images.length > 0) {
      bodyObj.images = images;
    }
    const body = JSON.stringify(bodyObj);

    await streamChat(selectedAnima, body, null, {
      onTextDelta: (text) => {
        streamingMsg.afterHeartbeatRelay = false;
        streamingMsg.text += text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onToolStart: (toolName) => {
        streamingMsg.activeTool = toolName;
        updateStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {
        // Keep last tool indicator visible — cleared on done
      },
      onChainStart: () => {
        // Session continuation — stream continues
      },
      onHeartbeatRelayStart: ({ message }) => {
        streamingMsg.heartbeatRelay = true;
        streamingMsg.heartbeatText = "";
        streamingMsg.text = "";
        scheduleStreamingUpdate(streamingMsg);
      },
      onHeartbeatRelay: ({ text }) => {
        streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onHeartbeatRelayDone: () => {
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = true;
        scheduleStreamingUpdate(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[エラー] ${errorMsg}`;
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        setState({ chatMessages: [...getState().chatMessages] });
        renderAllMessages();
      },
      onDone: ({ summary }) => {
        if (summary) {
          streamingMsg.text = summary;
        }
        if (!streamingMsg.text) {
          streamingMsg.text = "(空の応答)";
        }
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = false;
        setState({ chatMessages: [...getState().chatMessages] });
        renderAllMessages();
      },
    });

    // Ensure finalized if stream ended without done event
    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) {
        streamingMsg.text = streamingMsg.afterHeartbeatRelay
          ? "(応答の受信に失敗しました。再送信してください)"
          : "(空の応答)";
      }
      streamingMsg.afterHeartbeatRelay = false;
      setState({ chatMessages: [...getState().chatMessages] });
      renderAllMessages();
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      logger.error("Chat stream error", { anima: selectedAnima, error: err.message, name: err.name });
    }
    streamingMsg.text = `[エラー] ${err.message}`;
    streamingMsg.streaming = false;
    streamingMsg.activeTool = null;
    setState({ chatMessages: [...getState().chatMessages] });
    renderAllMessages();
  } finally {
    _isSseStreaming = false;
    setInputEnabled(true);
    if (inputEl) inputEl.focus();
  }
}

/**
 * Add a message from external source (e.g. WebSocket push).
 */
export function addMessage(role, text) {
  const { chatMessages } = getState();

  // Skip if SSE streaming is active (SSE handles display)
  if (_isSseStreaming) return;

  // Avoid duplicating the last message
  const last = chatMessages[chatMessages.length - 1];
  if (last && last.role === role && last.text === text) return;

  setState({ chatMessages: [...chatMessages, { role, text }] });
  renderAllMessages();
}

/**
 * Load full conversation history from server.
 */
export async function loadConversation() {
  const { selectedAnima } = getState();
  if (!selectedAnima) return;

  setState({ chatPagination: { totalRaw: 0, hasMore: false, loading: false } });

  try {
    const data = await fetchConversationFull(selectedAnima, PAGE_SIZE);
    if (data.turns && data.turns.length > 0) {
      const messages = data.turns.map((t) => ({
        role: t.role === "human" ? "user" : "assistant",
        text: t.content || "",
      }));
      const totalRaw = data.raw_turns || 0;
      setState({
        chatMessages: messages,
        chatPagination: {
          totalRaw,
          hasMore: messages.length < totalRaw,
          loading: false,
        },
      });
    } else {
      setState({
        chatMessages: [],
        chatPagination: { totalRaw: 0, hasMore: false, loading: false },
      });
    }
  } catch (err) {
    logger.error("Failed to load conversation", { anima: selectedAnima, error: err.message });
    setState({
      chatMessages: [],
      chatPagination: { totalRaw: 0, hasMore: false, loading: false },
    });
  }

  renderAllMessages(true);
  updateInputState();

  // Check for active stream to resume
  resumeActiveStream(getState().selectedAnima);
}

/**
 * Resume an active SSE stream after page reload.
 */
async function resumeActiveStream(animaName) {
  if (_isSseStreaming) return;

  try {
    const active = await fetchActiveStream(animaName);
    if (!active || active.status !== "streaming") return;

    const progress = await fetchStreamProgress(animaName, active.response_id);
    if (!progress) return;

    // Show accumulated text in streaming bubble
    const { chatMessages } = getState();
    const streamingMsg = {
      role: "assistant",
      text: progress.full_text || "",
      streaming: true,
      activeTool: progress.active_tool || null,
    };
    setState({ chatMessages: [...chatMessages, streamingMsg] });
    renderAllMessages();

    setInputEnabled(false);
    _isSseStreaming = true;

    const resumeBody = JSON.stringify({
      message: "",
      from_person: getState().currentUser || "human",
      resume: active.response_id,
      last_event_id: progress.last_event_id || "",
    });

    await streamChat(animaName, resumeBody, null, {
      onTextDelta: (text) => {
        streamingMsg.text += text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onToolStart: (toolName) => {
        streamingMsg.activeTool = toolName;
        updateStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {},
      onReconnecting: () => {
        streamingMsg.activeTool = "再接続中...";
        updateStreamingBubble(streamingMsg);
      },
      onReconnected: () => {
        streamingMsg.activeTool = null;
        updateStreamingBubble(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[エラー] ${errorMsg}`;
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        setState({ chatMessages: [...getState().chatMessages] });
        renderAllMessages();
      },
      onDone: ({ summary }) => {
        if (summary) streamingMsg.text = summary;
        if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.afterHeartbeatRelay = false;
        setState({ chatMessages: [...getState().chatMessages] });
        renderAllMessages();
      },
    });

    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
      setState({ chatMessages: [...getState().chatMessages] });
      renderAllMessages();
    }
  } catch (err) {
    logger.error("Resume stream error", { anima: animaName, error: err.message });
  } finally {
    _isSseStreaming = false;
    setInputEnabled(true);
    if (inputEl) inputEl.focus();
  }
}

// ── Internal Helpers ────────────────────────────

function submitFromInput() {
  if (!inputEl) return;
  const text = inputEl.value.trim();
  const hasImages = imageInputManager && imageInputManager.getImageCount() > 0;
  if (!text && !hasImages) return;

  inputEl.value = "";
  inputEl.style.height = "auto";
  sendMessage(text);
}

function setInputEnabled(enabled) {
  if (inputEl) inputEl.disabled = !enabled;
  if (sendBtnEl) sendBtnEl.disabled = !enabled;
}
