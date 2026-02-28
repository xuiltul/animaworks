// ── Board Tab ──────────────────────
// Extracted from app.js: Board channel/DM read + post

import { escapeHtml, smartTimestamp } from "./utils.js";
import { getCurrentUser } from "./login.js";
import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-board");

let _boardInitialized = false;
let _boardChannels = [];
let _boardDMs = [];
let _boardSelectedChannel = null;
let _boardSelectedType = null;  // "channel" | "dm"
let _paneBoard = null;

export function initBoard(paneBoardEl) {
  _paneBoard = paneBoardEl;
}

export function getSelectedBoard() {
  return { type: _boardSelectedType, channel: _boardSelectedChannel };
}

export async function initBoardTab() {
  if (!_paneBoard) return;

  if (!_boardInitialized) {
    _boardInitialized = true;
    _paneBoard.innerHTML = `
      <div class="ws-board-tab">
        <div class="ws-board-dropdown">
          <select class="ws-board-select" id="wsBoardSelect">
            <option value="">読み込み中...</option>
          </select>
        </div>
        <div class="ws-board-messages" id="wsBoardMessages">
          <div class="loading-placeholder">チャンネルを選択してください</div>
        </div>
        <div class="ws-board-input" id="wsBoardInputArea">
          <textarea class="ws-board-input-field" id="wsBoardInput" placeholder="メッセージを入力..." rows="1"></textarea>
          <button class="ws-board-send-btn" id="wsBoardSend">送信</button>
        </div>
      </div>`;

    const select = document.getElementById("wsBoardSelect");
    const sendBtn = document.getElementById("wsBoardSend");
    const input = document.getElementById("wsBoardInput");

    select?.addEventListener("change", () => {
      const val = select.value;
      if (!val) return;
      const [type, name] = val.split(":", 2);
      _boardSelectedType = type;
      _boardSelectedChannel = name;
      loadBoardMessages();
    });

    sendBtn?.addEventListener("click", sendBoardMessage);
    input?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendBoardMessage();
      }
    });
  }

  await loadBoardChannelList();
}

async function loadBoardChannelList() {
  const select = document.getElementById("wsBoardSelect");
  if (!select) return;

  try {
    const [chRes, dmRes] = await Promise.all([
      fetch("/api/channels"),
      fetch("/api/dm"),
    ]);

    _boardChannels = chRes.ok ? await chRes.json() : [];
    _boardDMs = dmRes.ok ? await dmRes.json() : [];

    let html = '<option value="">-- チャンネルを選択 --</option>';

    if (_boardChannels.length > 0) {
      html += '<optgroup label="Channels">';
      for (const ch of _boardChannels) {
        const count = ch.message_count || 0;
        html += `<option value="channel:${escapeHtml(ch.name)}">#${escapeHtml(ch.name)} (${count})</option>`;
      }
      html += "</optgroup>";
    }

    if (_boardDMs.length > 0) {
      html += '<optgroup label="DM">';
      for (const dm of _boardDMs) {
        const pair = dm.pair || dm.participants?.join(" & ") || "?";
        const count = dm.message_count || 0;
        html += `<option value="dm:${escapeHtml(pair)}">${escapeHtml(pair)} (${count})</option>`;
      }
      html += "</optgroup>";
    }

    if (_boardChannels.length === 0 && _boardDMs.length === 0) {
      html = '<option value="">チャンネルがありません</option>';
    }

    select.innerHTML = html;

    if (_boardSelectedChannel && _boardSelectedType) {
      select.value = `${_boardSelectedType}:${_boardSelectedChannel}`;
    }
  } catch (err) {
    logger.error("Failed to load board channels", { error: err.message });
    select.innerHTML = '<option value="">読み込み失敗</option>';
  }
}

async function loadBoardMessages() {
  const messagesEl = document.getElementById("wsBoardMessages");
  if (!messagesEl || !_boardSelectedChannel) return;

  messagesEl.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    let url;
    if (_boardSelectedType === "channel") {
      url = `/api/channels/${encodeURIComponent(_boardSelectedChannel)}?limit=50&offset=0`;
    } else {
      url = `/api/dm/${encodeURIComponent(_boardSelectedChannel)}?limit=50`;
    }

    const res = await fetch(url);
    if (!res.ok) {
      messagesEl.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
      return;
    }

    const data = await res.json();
    const messages = data.messages || [];

    if (messages.length === 0) {
      messagesEl.innerHTML = '<div class="loading-placeholder">メッセージはまだありません</div>';
      return;
    }

    messagesEl.innerHTML = messages.map(renderBoardMessage).join("");
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    logger.error("Failed to load board messages", { error: err.message });
    messagesEl.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}

function renderBoardMessage(msg) {
  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const from = escapeHtml(msg.from || "?");
  const text = escapeHtml(msg.text || "");
  const humanBadge = msg.source === "human" ? ' <span class="ws-board-human-badge">[human]</span>' : "";
  return `<div class="ws-board-msg">
    <span class="ws-board-msg-time">${escapeHtml(ts)}</span>
    <span class="ws-board-msg-from">[${from}]${humanBadge}</span>
    <span class="ws-board-msg-text">${text}</span>
  </div>`;
}

export function appendBoardMessage(msg) {
  const messagesEl = document.getElementById("wsBoardMessages");
  if (!messagesEl) return;

  const placeholder = messagesEl.querySelector(".loading-placeholder");
  if (placeholder) placeholder.remove();

  const div = document.createElement("div");
  div.innerHTML = renderBoardMessage(msg);
  const el = div.firstElementChild;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendBoardMessage() {
  const input = document.getElementById("wsBoardInput");
  const text = input?.value?.trim();
  if (!text || !_boardSelectedChannel || _boardSelectedType !== "channel") return;

  const userName = getCurrentUser() || "guest";
  input.value = "";

  try {
    const res = await fetch(`/api/channels/${encodeURIComponent(_boardSelectedChannel)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, from_name: userName }),
    });
    if (!res.ok) {
      logger.error("Failed to send board message", { status: res.status });
    }
  } catch (err) {
    logger.error("Failed to send board message", { error: err.message });
  }
}
