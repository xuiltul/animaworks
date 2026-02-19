/* ── Chat ──────────────────────────────────── */

import { state, dom, escapeHtml, renderMarkdown } from "./state.js";
import { addActivity } from "./activity.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../shared/chat-stream.js";
import { createLogger } from "../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../shared/image-input.js";

const logger = createLogger("chat");

let imageInputManager = null;

// ── Render ─────────────────────────────────

export function renderChat() {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return; // Chat not in DOM (page not active)

  const name = state.selectedAnima;
  const history = state.chatHistories[name] || [];
  if (history.length === 0) {
    chatMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
    return;
  }
  chatMessages.innerHTML = history.map((m) => {
    if (m.role === "thinking") {
      return `<div class="chat-bubble thinking"><span class="thinking-animation">考え中</span></div>`;
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
        ? `<div class="bootstrap-indicator"><span class="tool-spinner"></span>初期化中...</div>`
        : "";
      const toolHtml = m.activeTool
        ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(m.activeTool)} を実行中...</div>`
        : "";
      return `<div class="chat-bubble assistant${streamClass}${notifClass}">${content}${bootstrapHtml}${toolHtml}</div>`;
    }
    const imagesHtml = renderChatImages(m.images);
    const textHtml = m.text ? `<div class="chat-text">${escapeHtml(m.text)}</div>` : "";
    return `<div class="chat-bubble user">${imagesHtml}${textHtml}</div>`;
  }).join("");
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── SSE Streaming ─────────────────────────

function renderStreamingBubble(msg) {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return;
  const bubble = chatMessages.querySelector(".chat-bubble.assistant.streaming");
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
    try {
      html = marked.parse(msg.text, { breaks: true });
    } catch {
      html = escapeHtml(msg.text);
    }
  } else {
    html = '<span class="cursor-blink"></span>';
  }

  if (msg.bootstrapping) {
    html += `<div class="bootstrap-indicator"><span class="tool-spinner"></span>初期化中...</div>`;
  }

  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`;
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
      systemMsg.textContent = "このキャラクターは現在制作中です。完了までお待ちください。";
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
  addActivity("chat", name, `ユーザー: ${message}`);

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
          streamingMsg.text = data.message || "現在初期化中です。しばらくお待ちください。";
          streamingMsg.streaming = false;
          streamingMsg.bootstrapping = false;
          renderChat();
          addActivity("system", name, "ブートストラップ中のため応答保留");
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
        addActivity("system", name, `ハートビート中継: ${message}`);
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
        streamingMsg.text += `\n[エラー] ${errorMsg}`;
        streamingMsg.streaming = false;
        renderChat();
      },
      onDone: ({ summary }) => {
        const summaryLen = (summary || "").length;
        const textLen = streamingMsg.text.length;
        logger.debug(`onDone: summary_len=${summaryLen} text_len=${textLen} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
        const text = summary || streamingMsg.text;
        streamingMsg.text = text || "(空の応答)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = false;
        logger.debug(`onDone: final text_len=${streamingMsg.text.length}`);
        renderChat();
        addActivity("chat", name, `応答: ${streamingMsg.text.slice(0, 100)}`);
      },
    });

    // Ensure streaming is finalized if stream ended without done event
    if (streamingMsg.streaming) {
      logger.info(`[SSE-UI] sendChat FINALIZE_FALLBACK anima=${name} text_len=${streamingMsg.text.length} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
      streamingMsg.streaming = false;
      if (!streamingMsg.text) {
        streamingMsg.text = streamingMsg.afterHeartbeatRelay
          ? "(応答の受信に失敗しました。再送信してください)"
          : "(空の応答)";
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
    streamingMsg.text = `[エラー] ${err.message}`;
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
  attachBtn.title = "画像を添付";
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
        streamingMsg.activeTool = "再接続中...";
        renderStreamingBubble(streamingMsg);
      },
      onReconnected: () => {
        streamingMsg.activeTool = null;
        renderStreamingBubble(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[エラー] ${errorMsg}`;
        streamingMsg.streaming = false;
        renderChat();
      },
      onDone: ({ summary }) => {
        const text = summary || streamingMsg.text;
        streamingMsg.text = text || "(空の応答)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.afterHeartbeatRelay = false;
        renderChat();
      },
    });

    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
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
