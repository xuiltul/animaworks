/* ── Chat ──────────────────────────────────── */

import { state, dom, escapeHtml, renderMarkdown } from "./state.js";
import { addActivity } from "./activity.js";
import { parseConvSSE as parseSSEEvents, getErrorMessage } from "../shared/sse-parser.js";

// ── Render ─────────────────────────────────

export function renderChat() {
  const chatMessages = dom.chatMessages || document.getElementById("chatMessages");
  if (!chatMessages) return; // Chat not in DOM (page not active)

  const name = state.selectedPerson;
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
      return `<div class="chat-bubble assistant${streamClass}">${content}${bootstrapHtml}${toolHtml}</div>`;
    }
    return `<div class="chat-bubble user">${escapeHtml(m.text)}</div>`;
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
  if (msg.text) {
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
  const name = state.selectedPerson;
  if (!name || !message.trim()) return;

  // Guard: block sending to bootstrapping persons
  const currentPerson = state.persons.find((p) => p.name === name);
  if (currentPerson?.status === "bootstrapping" || currentPerson?.bootstrapping) {
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

  // Add user message + empty streaming bubble
  history.push({ role: "user", text: message });
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null };
  history.push(streamingMsg);
  renderChat();

  const chatInput = dom.chatInput || document.getElementById("chatInput");
  const chatSendBtn = dom.chatSendBtn || document.getElementById("chatSendBtn");
  if (chatInput) chatInput.disabled = true;
  if (chatSendBtn) chatSendBtn.disabled = true;
  addActivity("chat", name, `ユーザー: ${message}`);

  try {
    const response = await fetch(
      `/api/persons/${encodeURIComponent(name)}/chat/stream`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, from_person: state.currentUser || "human" }),
      }
    );

    if (!response.ok) throw new Error(`API ${response.status}: ${response.statusText}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const { parsed, remaining } = parseSSEEvents(buffer);
      buffer = remaining;

      for (const evt of parsed) {
        switch (evt.event) {
          case "text_delta":
            streamingMsg.text += evt.data.text;
            renderStreamingBubble(streamingMsg);
            break;

          case "tool_start":
            streamingMsg.activeTool = evt.data.tool_name;
            renderStreamingBubble(streamingMsg);
            break;

          case "tool_end":
            // Keep last tool indicator visible — cleared on done
            break;

          case "bootstrap":
            if (evt.data.status === "started") {
              streamingMsg.bootstrapping = true;
              renderStreamingBubble(streamingMsg);
            } else if (evt.data.status === "completed") {
              streamingMsg.bootstrapping = false;
              renderStreamingBubble(streamingMsg);
            } else if (evt.data.status === "busy") {
              streamingMsg.text = evt.data.message || "現在初期化中です。しばらくお待ちください。";
              streamingMsg.streaming = false;
              streamingMsg.bootstrapping = false;
              renderChat();
              addActivity("system", name, "ブートストラップ中のため応答保留");
            }
            break;

          case "chain_start":
            break;

          case "error":
            streamingMsg.text += `\n[エラー] ${getErrorMessage(evt.data)}`;
            streamingMsg.streaming = false;
            renderChat();
            break;

          case "done": {
            const summary = (evt.data && evt.data.summary) || streamingMsg.text;
            streamingMsg.text = summary || "(空の応答)";
            streamingMsg.streaming = false;
            streamingMsg.activeTool = null;
            renderChat();
            addActivity("chat", name, `応答: ${streamingMsg.text.slice(0, 100)}`);
            break;
          }
        }
      }
    }

    // Ensure streaming is finalized
    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
      renderChat();
    }
  } catch (err) {
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
