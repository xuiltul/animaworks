// ── Board Page (Channels & DMs) ──────────────
import { api } from "../modules/api.js";
import { escapeHtml, renderMarkdown, smartTimestamp } from "../modules/state.js";
import { createLogger } from "../shared/logger.js";

const logger = createLogger("board-page");

// ── Local State ────────────────────────────

let _container = null;
let _channels = [];
let _dmPairs = [];
let _selectedType = null;   // "channel" | "dm"
let _selectedName = null;    // channel name or dm pair
let _messages = [];
let _total = 0;
let _refreshInterval = null;
let _boundListeners = [];
let _wsHandler = null;
let _sidebarCollapsed = false;

// ── DOM refs (local) ───────────────────────

function _$(id) { return document.getElementById(id); }

// ── Render ─────────────────────────────────

export function render(container) {
  _container = container;
  _channels = [];
  _dmPairs = [];
  _selectedType = null;
  _selectedName = null;
  _messages = [];
  _total = 0;
  _boundListeners = [];
  _sidebarCollapsed = false;

  container.innerHTML = `
    <div class="board-layout">
      <!-- Mobile toggle for sidebar -->
      <button class="board-mobile-toggle" id="boardMobileToggle">チャンネル一覧を表示</button>

      <!-- Left Sidebar -->
      <div class="board-sidebar" id="boardSidebar">
        <div class="board-sidebar-section">Channels</div>
        <div id="boardChannelList">
          <div class="loading-placeholder">読み込み中...</div>
        </div>
        <div class="board-sidebar-divider"></div>
        <div class="board-sidebar-section">DMs</div>
        <div id="boardDmList">
          <div class="loading-placeholder">読み込み中...</div>
        </div>
      </div>

      <!-- Right Panel -->
      <div class="board-main" id="boardMain">
        <div class="board-placeholder" id="boardPlaceholder">
          <div class="board-placeholder-icon">&#x1F4CB;</div>
          <div class="board-placeholder-text">チャンネルまたはDMを選択してください</div>
        </div>

        <!-- Channel/DM view (hidden initially) -->
        <div id="boardView" style="display:none;">
          <div class="board-header" id="boardHeader">
            <div class="board-header-title" id="boardHeaderTitle"></div>
            <div class="board-header-meta" id="boardHeaderMeta"></div>
          </div>
          <div class="board-messages" id="boardMessages">
            <div class="board-messages-empty">メッセージはまだありません</div>
          </div>
          <form class="board-input-bar" id="boardInputBar" style="display:none;">
            <textarea
              id="boardInput"
              class="board-input"
              placeholder="メッセージを入力..."
              autocomplete="off"
              rows="1"
            ></textarea>
            <button type="submit" class="board-send-btn" id="boardSendBtn">送信</button>
          </form>
        </div>
      </div>
    </div>
  `;

  _bindEvents();
  _loadSidebarData();

  // Polling fallback: refresh messages every 30s
  _refreshInterval = setInterval(() => {
    if (_selectedType && _selectedName) {
      _loadMessages(_selectedType, _selectedName, true);
    }
  }, 30000);

  // Register WebSocket handler
  _wsHandler = _handleBoardPost.bind(null);
  window.__boardWsHandler = _wsHandler;
}

export function destroy() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
  for (const { el, event, handler } of _boundListeners) {
    el.removeEventListener(event, handler);
  }
  _boundListeners = [];
  window.__boardWsHandler = null;
  _wsHandler = null;
  _container = null;
  _channels = [];
  _dmPairs = [];
  _selectedType = null;
  _selectedName = null;
  _messages = [];
}

// ── Event Binding ──────────────────────────

function _bindEvents() {
  // Form submit
  _addListener("boardInputBar", "submit", (e) => {
    e.preventDefault();
    _submitMessage();
  });

  // Textarea: Enter (without shift) sends
  _addListener("boardInput", "keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      _submitMessage();
    }
  });

  // Auto-resize textarea
  _addListener("boardInput", "input", () => {
    const el = _$("boardInput");
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }
  });

  // Mobile sidebar toggle
  _addListener("boardMobileToggle", "click", () => {
    const sidebar = _$("boardSidebar");
    if (!sidebar) return;
    _sidebarCollapsed = !_sidebarCollapsed;
    sidebar.classList.toggle("mobile-hidden", _sidebarCollapsed);
    const btn = _$("boardMobileToggle");
    if (btn) {
      btn.textContent = _sidebarCollapsed ? "チャンネル一覧を表示" : "チャンネル一覧を隠す";
    }
  });
}

function _addListener(id, event, handler) {
  const el = _$(id);
  if (el) {
    el.addEventListener(event, handler);
    _boundListeners.push({ el, event, handler });
  }
}

// ── Sidebar Data Loading ───────────────────

async function _loadSidebarData() {
  const [channels, dms] = await Promise.all([
    api("/api/channels").catch(() => []),
    api("/api/dm").catch(() => []),
  ]);

  _channels = channels || [];
  _dmPairs = dms || [];

  _renderChannelList();
  _renderDmList();
}

function _renderChannelList() {
  const el = _$("boardChannelList");
  if (!el) return;

  if (_channels.length === 0) {
    el.innerHTML = '<div class="loading-placeholder" style="padding:8px 16px; font-size:0.82rem;">チャンネルなし</div>';
    return;
  }

  el.innerHTML = _channels.map(ch => {
    const activeClass = _selectedType === "channel" && _selectedName === ch.name ? " active" : "";
    const count = ch.message_count ?? 0;
    return `
      <div class="board-sidebar-item${activeClass}" data-type="channel" data-name="${escapeHtml(ch.name)}" tabindex="0" role="button">
        <span class="board-sidebar-item-prefix">#</span>
        <span>${escapeHtml(ch.name)}</span>
        ${count > 0 ? `<span class="board-sidebar-item-count">${count}</span>` : ""}
      </div>`;
  }).join("");

  _bindSidebarClicks(el);
}

function _renderDmList() {
  const el = _$("boardDmList");
  if (!el) return;

  if (_dmPairs.length === 0) {
    el.innerHTML = '<div class="loading-placeholder" style="padding:8px 16px; font-size:0.82rem;">DMなし</div>';
    return;
  }

  el.innerHTML = _dmPairs.map(dm => {
    const pair = dm.pair || "";
    const participants = (dm.participants || []).join(", ");
    const activeClass = _selectedType === "dm" && _selectedName === pair ? " active" : "";
    const count = dm.message_count ?? 0;
    return `
      <div class="board-sidebar-item${activeClass}" data-type="dm" data-name="${escapeHtml(pair)}" tabindex="0" role="button">
        <span>${escapeHtml(participants || pair)}</span>
        ${count > 0 ? `<span class="board-sidebar-item-count">${count}</span>` : ""}
      </div>`;
  }).join("");

  _bindSidebarClicks(el);
}

function _bindSidebarClicks(container) {
  container.querySelectorAll(".board-sidebar-item").forEach(item => {
    const handler = () => {
      const type = item.dataset.type;
      const name = item.dataset.name;
      _selectItem(type, name);
    };
    item.addEventListener("click", handler);
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        handler();
      }
    });
    _boundListeners.push({ el: item, event: "click", handler });
  });
}

// ── Selection ──────────────────────────────

function _selectItem(type, name) {
  _selectedType = type;
  _selectedName = name;

  // Update sidebar highlighting
  _renderChannelList();
  _renderDmList();

  // Collapse sidebar on mobile after selection
  const sidebar = _$("boardSidebar");
  if (sidebar && window.innerWidth <= 768) {
    _sidebarCollapsed = true;
    sidebar.classList.add("mobile-hidden");
    const btn = _$("boardMobileToggle");
    if (btn) btn.textContent = "チャンネル一覧を表示";
  }

  // Show the view panel
  const placeholder = _$("boardPlaceholder");
  const view = _$("boardView");
  if (placeholder) placeholder.style.display = "none";
  if (view) {
    view.style.display = "flex";
    view.style.flexDirection = "column";
    view.style.minHeight = "0";
    view.style.flex = "1";
  }

  // Update header
  const titleEl = _$("boardHeaderTitle");
  const metaEl = _$("boardHeaderMeta");
  if (type === "channel") {
    if (titleEl) titleEl.textContent = `# ${name}`;
    if (metaEl) metaEl.textContent = "";
  } else {
    const dmInfo = _dmPairs.find(d => d.pair === name);
    const participants = dmInfo ? (dmInfo.participants || []).join(" & ") : name;
    if (titleEl) titleEl.textContent = participants;
    if (metaEl) metaEl.textContent = "ダイレクトメッセージ";
  }

  // Show/hide input bar (hidden for DMs)
  const inputBar = _$("boardInputBar");
  if (inputBar) inputBar.style.display = type === "channel" ? "flex" : "none";

  // Update input placeholder
  const input = _$("boardInput");
  if (input && type === "channel") {
    input.placeholder = `#${name} にメッセージを投稿...`;
  }

  // Load messages
  _loadMessages(type, name, false);
}

// ── Message Loading ────────────────────────

async function _loadMessages(type, name, isPolling) {
  // Skip if selection changed during load
  if (type !== _selectedType || name !== _selectedName) return;

  const messagesEl = _$("boardMessages");
  if (!messagesEl) return;

  if (!isPolling) {
    messagesEl.innerHTML = '<div class="board-messages-empty">読み込み中...</div>';
  }

  try {
    let data;
    if (type === "channel") {
      data = await api(`/api/channels/${encodeURIComponent(name)}?limit=50&offset=0`);
    } else {
      data = await api(`/api/dm/${encodeURIComponent(name)}?limit=50`);
    }

    // Verify selection hasn't changed
    if (type !== _selectedType || name !== _selectedName) return;

    _messages = data.messages || [];
    _total = data.total || _messages.length;
    _renderMessages();
  } catch (err) {
    if (type !== _selectedType || name !== _selectedName) return;
    if (!isPolling) {
      messagesEl.innerHTML = `<div class="board-messages-empty">読み込み失敗: ${escapeHtml(err.message)}</div>`;
    }
    logger.error("Failed to load messages", { type, name, error: err.message });
  }
}

// ── Message Rendering ──────────────────────

function _renderMessages() {
  const messagesEl = _$("boardMessages");
  if (!messagesEl) return;

  if (_messages.length === 0) {
    messagesEl.innerHTML = '<div class="board-messages-empty">メッセージはまだありません</div>';
    return;
  }

  const wasAtBottom = _isScrolledToBottom(messagesEl);

  messagesEl.innerHTML = _messages.map(msg => {
    const from = msg.from || "unknown";
    const initial = from.charAt(0).toUpperCase();
    const isHuman = msg.source === "human";
    const avatarClass = isHuman ? "board-msg-avatar human" : "board-msg-avatar";
    const ts = msg.ts ? smartTimestamp(msg.ts) : "";
    const badge = isHuman ? '<span class="board-msg-badge">human</span>' : "";
    const text = msg.text || "";

    return `
      <div class="board-msg">
        <div class="${avatarClass}">${escapeHtml(initial)}</div>
        <div class="board-msg-body">
          <div class="board-msg-header">
            <span class="board-msg-from">${escapeHtml(from)}</span>
            ${badge}
            <span class="board-msg-ts">${escapeHtml(ts)}</span>
          </div>
          <div class="board-msg-text board-markdown">${renderMarkdown(text)}</div>
        </div>
      </div>`;
  }).join("");

  // Auto-scroll to bottom if user was already at bottom
  if (wasAtBottom) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function _isScrolledToBottom(el) {
  if (!el) return true;
  // Consider "at bottom" if within 50px of the bottom
  return el.scrollHeight - el.scrollTop - el.clientHeight < 50;
}

function _scrollToBottom() {
  const messagesEl = _$("boardMessages");
  if (messagesEl) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

// ── Message Posting ────────────────────────

async function _submitMessage() {
  const input = _$("boardInput");
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  if (_selectedType !== "channel" || !_selectedName) return;

  input.value = "";
  input.style.height = "auto";

  const sendBtn = _$("boardSendBtn");
  if (input) input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  try {
    const currentUser = localStorage.getItem("animaworks_user") || "human";
    await api(`/api/channels/${encodeURIComponent(_selectedName)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, from_name: currentUser }),
    });

    // Optimistic: add the message locally
    _messages.push({
      ts: new Date().toISOString(),
      from: currentUser,
      text,
      source: "human",
    });
    _renderMessages();
    _scrollToBottom();
  } catch (err) {
    logger.error("Failed to post message", { channel: _selectedName, error: err.message });
    // Show inline error
    const messagesEl = _$("boardMessages");
    if (messagesEl) {
      const errorDiv = document.createElement("div");
      errorDiv.style.cssText = "color:#ef4444; font-size:0.82rem; text-align:center; padding:4px;";
      errorDiv.textContent = `送信失敗: ${err.message}`;
      messagesEl.appendChild(errorDiv);
      _scrollToBottom();
    }
  } finally {
    if (input) { input.disabled = false; input.focus(); }
    if (sendBtn) sendBtn.disabled = false;
  }
}

// ── WebSocket: board.post ──────────────────

function _handleBoardPost(data) {
  if (!data) return;

  // Only update if viewing the matching channel
  if (_selectedType === "channel" && _selectedName === data.channel) {
    _messages.push({
      ts: data.ts || new Date().toISOString(),
      from: data.from || "unknown",
      text: data.text || "",
      source: data.source || "",
    });
    _renderMessages();
    _scrollToBottom();
  }

  // Update sidebar counts (channel might have new messages)
  _updateSidebarCount(data.channel);
}

function _updateSidebarCount(channelName) {
  const ch = _channels.find(c => c.name === channelName);
  if (ch) {
    ch.message_count = (ch.message_count || 0) + 1;
    _renderChannelList();
  }
}
