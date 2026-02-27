// ── Chat Page (Self-Contained) ──────────────
import { t } from "/shared/i18n.js";
import { api } from "../modules/api.js";
import { escapeHtml, renderMarkdown, renderSafeMarkdown, timeStr, smartTimestamp } from "../modules/state.js";
import { streamChat } from "../shared/chat-stream.js";
import { createLogger } from "../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../shared/image-input.js";
import { initVoiceUI, updateVoiceUIAnima } from "../modules/voice-ui.js";
import { getIcon, getDisplaySummary } from "../shared/activity-types.js";

const logger = createLogger("chat-page");

// ── Local State ────────────────────────────

let _container = null;
let _animas = [];
let _selectedAnima = null;
let _chatHistories = {};
let _animaDetail = null;
let _activeRightTab = "state";
let _activeMemoryTab = "episodes";
let _intervals = [];
let _boundListeners = [];
let _historyState = {};  // Per-anima: { sessions, hasMore, nextBefore, loading }
let _chatObserver = null;
let _isChatStreaming = false;
let _selectedThreadId = "default";
let _threads = {};  // { [animaName]: [{ id, label, unread }] }
const _HISTORY_PAGE_SIZE = 50;
const _TOOL_RESULT_TRUNCATE = 500;
let _imageInputManager = null;
let _bustupUrl = null;
let _pendingQueue = [];       // Array<{ text, images, displayImages }>
let _chatAbortController = null;

// ── Chat Draft Persistence ───────────────────

function _getDraftKey(animaName) {
  const user = localStorage.getItem("animaworks_user") || "guest";
  const anima = animaName || "_";
  return `aw:draft:dashboard-chat:${user}:${anima}`;
}

function _saveDraft(animaName, text) {
  if (!animaName) return;
  localStorage.setItem(_getDraftKey(animaName), text || "");
}

function _loadDraft(animaName) {
  if (!animaName) return "";
  return localStorage.getItem(_getDraftKey(animaName)) || "";
}

function _clearDraft(animaName) {
  if (!animaName) return;
  localStorage.removeItem(_getDraftKey(animaName));
}

// ── DOM refs (local) ───────────────────────

function _$(id) { return document.getElementById(id); }

// ── Render ─────────────────────────────────

export function render(container) {
  _container = container;
  _animas = [];
  _selectedAnima = null;
  _chatHistories = {};
  _historyState = {};
  _selectedThreadId = "default";
  _threads = {};
  _animaDetail = null;
  _activeRightTab = "state";
  _activeMemoryTab = "episodes";
  _intervals = [];
  _boundListeners = [];

  container.innerHTML = `
    <!-- Mobile Tab Bar (hidden on desktop, visible on mobile) -->
    <nav class="chat-mobile-tabs" id="chatMobileTabs">
      <button class="chat-mobile-tab active" data-panel="chat" id="chatMobileTabChat">${t("nav.chat")}</button>
      <button class="chat-mobile-tab" data-panel="info" id="chatMobileTabInfo">${t("chat.character_summary")}</button>
    </nav>

    <div class="chat-page-layout">
      <!-- Left: Chat Panel -->
      <div class="chat-page-main">
        <!-- Anima Selector -->
        <div style="display:flex; align-items:center; gap:0.75rem; padding:0.75rem; border-bottom:1px solid var(--border-color, #eee);">
          <div id="chatPageAvatar" class="anima-avatar-container"></div>
          <select id="chatPageAnimaSelect" class="anima-dropdown" style="flex:1;">
            <option value="" disabled selected>${t("chat.anima_select")}</option>
          </select>
        </div>

        <div class="thread-tabs" id="chatThreadTabs">
          <button class="thread-tab active" data-thread="default">メイン</button>
          <button class="thread-tab-new" id="chatNewThreadBtn" title="新しいスレッド">＋</button>
        </div>

        <!-- Chat Messages -->
        <div id="chatPageMessages" class="chat-messages" style="flex:1; overflow-y:auto; padding:1rem;">
          <div class="chat-empty">${t("chat.anima_select_first")}</div>
        </div>

        <!-- Chat Input -->
        <form id="chatPageForm" class="chat-input-form" style="padding:0.75rem; border-top:1px solid var(--border-color, #eee);">
          <div class="image-preview-bar" id="chatPagePreviewBar" style="display:none"></div>
          <div class="pending-queue-bar" id="chatPagePending" style="display:none">
            <div class="pending-queue-header">
              <span class="pending-queue-label" id="chatPagePendingLabel">${t("chat.queue_label")}</span>
              <button class="pending-queue-clear" id="chatPagePendingCancel" type="button" title="${t("chat.queue_clear_all")}">✕ all</button>
            </div>
            <div id="chatPagePendingList"></div>
          </div>
          <div class="chat-input-wrap">
            <textarea
              id="chatPageInput"
              class="chat-input"
              placeholder="${t("chat.placeholder")}"
              autocomplete="off"
              rows="1"
              disabled
            ></textarea>
            <div class="chat-input-actions">
              <button type="button" class="chat-attach-btn" id="chatPageAttachBtn" title="${t("chat.attach_image")}">+</button>
              <button type="button" class="chat-queue-btn" id="chatPageQueueBtn" disabled title="${t("chat.queue_add")}">↓</button>
              <button type="submit" class="chat-send-btn" id="chatPageSendBtn" disabled>↑</button>
            </div>
          </div>
          <input type="file" id="chatPageFileInput" accept="image/jpeg,image/png,image/gif,image/webp" multiple style="display:none" />
        </form>
      </div>

      <!-- Right: Sidebar (hidden on mobile by default) -->
      <div class="chat-page-sidebar mobile-hidden">
        <!-- Right tabs -->
        <nav class="right-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
          <button class="right-tab active" data-tab="state" id="chatTabState">${t("chat.state_current")}</button>
          <button class="right-tab" data-tab="activity" id="chatTabActivity">${t("nav.activity")}</button>
          <button class="right-tab" data-tab="history" id="chatTabHistory">${t("chat.history_conversation")}</button>
        </nav>

        <div id="chatRightTabContent" style="padding:0.75rem;">
          <!-- State pane (default) -->
          <div id="chatPaneState">
            <pre class="state-content" id="chatAnimaState" style="white-space:pre-wrap; word-break:break-word; margin:0;">${t("chat.anima_select_first")}</pre>
          </div>
          <!-- Activity pane -->
          <div id="chatPaneActivity" style="display:none;">
            <div id="chatActivityFeed" class="activity-feed">
              <div class="activity-empty">${t("activity.feed_empty")}</div>
            </div>
          </div>
          <!-- History pane -->
          <div id="chatPaneHistory" style="display:none;">
            <div id="chatHistorySessionList">
              <div class="loading-placeholder">${t("chat.anima_select_first")}</div>
            </div>
            <div id="chatHistoryDetail" style="display:none;">
              <button class="memory-back-btn" id="chatHistoryBackBtn">&larr; ${t("chat.back_list")}</button>
              <h3 id="chatHistoryDetailTitle" style="margin:0.5rem 0;"></h3>
              <div id="chatHistoryConversation" style="max-height:400px; overflow-y:auto;"></div>
            </div>
          </div>

          <!-- Memory Browser (inside scrollable area) -->
          <div class="chat-memory-section">
            <nav class="memory-tabs" style="display:flex; border-bottom:1px solid var(--border-color, #eee);">
              <button class="memory-tab active" data-tab="episodes">${t("chat.memory_episodes")}</button>
              <button class="memory-tab" data-tab="knowledge">${t("chat.memory_knowledge")}</button>
              <button class="memory-tab" data-tab="procedures">${t("chat.memory_procedures")}</button>
            </nav>
            <div id="chatMemoryFileList" class="memory-file-list" style="padding:0.5rem;">
              <div class="loading-placeholder">${t("chat.anima_select")}</div>
            </div>
            <div id="chatMemoryContentArea" style="display:none; padding:0.5rem;">
              <button class="memory-back-btn" id="chatMemoryBackBtn">&larr; ${t("chat.memory_back")}</button>
              <h3 id="chatMemoryContentTitle" style="margin:0.5rem 0;"></h3>
              <pre id="chatMemoryContentBody" class="memory-content-body" style="white-space:pre-wrap; word-break:break-word;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  _bindEvents();
  _loadAnimas();

  // Auto-refresh activity
  const actInterval = setInterval(_loadActivity, 30000);
  _intervals.push(actInterval);
}

export function destroy() {
  for (const id of _intervals) clearInterval(id);
  _intervals = [];
  for (const { el, event, handler } of _boundListeners) {
    el.removeEventListener(event, handler);
  }
  _boundListeners = [];
  if (_chatObserver) { _chatObserver.disconnect(); _chatObserver = null; }
  if (_chatAbortController) { _chatAbortController.abort(); _chatAbortController = null; }
  _pendingQueue = [];
  _removeBustupOverlay();
  _bustupUrl = null;
  _container = null;
  _animas = [];
  _selectedAnima = null;
  _chatHistories = {};
  _historyState = {};
  _selectedThreadId = "default";
  _threads = {};
  _animaDetail = null;
  _imageInputManager = null;
}

// ── Event Binding ──────────────────────────

function _bindEvents() {
  // Mobile tab switching
  for (const tabId of ["chatMobileTabChat", "chatMobileTabInfo"]) {
    _addListener(tabId, "click", (e) => {
      _switchMobileTab(e.target.dataset.panel);
    });
  }

  // Bustup overlay on avatar click
  _addListener("chatPageAvatar", "click", () => {
    if (_bustupUrl) _showBustupOverlay();
  });

  // Escape to dismiss bustup overlay
  document.addEventListener("keydown", _onBustupEscape);
  _boundListeners.push({ el: document, event: "keydown", handler: _onBustupEscape });

  // Anima selector
  _addListener("chatPageAnimaSelect", "change", (e) => {
    const name = e.target.value;
    if (name) _selectAnima(name);
  });

  // New thread button
  _addListener("chatNewThreadBtn", "click", () => _createNewThread());

  // Chat form submit
  _addListener("chatPageForm", "submit", (e) => {
    e.preventDefault();
    _submitChat();
  });

  // Textarea: Ctrl+Enter = send, Alt+Enter = queue
  _addListener("chatPageInput", "keydown", (e) => {
    if (e.key === "Enter" && e.altKey) {
      e.preventDefault();
      _addToQueue();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      _submitChat();
    }
  });

  // Queue button
  _addListener("chatPageQueueBtn", "click", () => _addToQueue());

  // Pending queue cancel all
  _addListener("chatPagePendingCancel", "click", () => {
    _pendingQueue = [];
    _hidePendingIndicator();
    _updateSendButton();
  });

  // Auto-resize textarea + dynamic button update
  _addListener("chatPageInput", "input", () => {
    const el = _$("chatPageInput");
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 200) + "px";
      _saveDraft(_selectedAnima, el.value || "");
    }
    _updateSendButton();
  });

  // Right tab switching
  for (const tabId of ["chatTabState", "chatTabActivity", "chatTabHistory"]) {
    _addListener(tabId, "click", (e) => {
      const tab = e.target.dataset.tab;
      _switchRightTab(tab);
    });
  }

  // Memory tabs
  _container.querySelectorAll(".memory-tab").forEach(btn => {
    const handler = () => {
      _activeMemoryTab = btn.dataset.tab;
      _container.querySelectorAll(".memory-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === _activeMemoryTab));
      const contentArea = _$("chatMemoryContentArea");
      const fileList = _$("chatMemoryFileList");
      if (contentArea) contentArea.style.display = "none";
      if (fileList) fileList.style.display = "";
      _loadMemoryTab();
    };
    btn.addEventListener("click", handler);
    _boundListeners.push({ el: btn, event: "click", handler });
  });

  // Image attach button + file input
  _addListener("chatPageAttachBtn", "click", () => {
    const fileInput = _$("chatPageFileInput");
    if (fileInput) fileInput.click();
  });

  _addListener("chatPageFileInput", "change", () => {
    const fileInput = _$("chatPageFileInput");
    if (fileInput && fileInput.files.length > 0) {
      _imageInputManager?.addFiles(fileInput.files);
      fileInput.value = "";
    }
  });

  // Initialize image input manager
  _initImageInput();

  // Memory back button
  _addListener("chatMemoryBackBtn", "click", () => {
    const contentArea = _$("chatMemoryContentArea");
    const fileList = _$("chatMemoryFileList");
    if (contentArea) contentArea.style.display = "none";
    if (fileList) fileList.style.display = "";
  });

  // History back button
  _addListener("chatHistoryBackBtn", "click", () => {
    const detail = _$("chatHistoryDetail");
    const list = _$("chatHistorySessionList");
    if (detail) detail.style.display = "none";
    if (list) list.style.display = "";
  });

  // Infinite scroll observer
  _setupChatObserver();
}

function _addListener(id, event, handler) {
  const el = _$(id);
  if (el) {
    el.addEventListener(event, handler);
    _boundListeners.push({ el, event, handler });
  }
}

// ── Anima Selection ───────────────────────

async function _loadAnimas() {
  try {
    _animas = await api("/api/animas");
    _renderAnimaDropdown();
    if (_animas.length > 0 && !_selectedAnima) {
      _selectAnima(_animas[0].name);
    }
  } catch (err) {
    logger.error("Failed to load animas", err);
  }
}

function _renderAnimaDropdown() {
  const select = _$("chatPageAnimaSelect");
  if (!select) return;

  let html = `<option value="" disabled>${t("chat.anima_select")}</option>`;
  for (const p of _animas) {
    const selected = p.name === _selectedAnima ? " selected" : "";
    if (p.status === "bootstrapping" || p.bootstrapping) {
      html += `<option value="${escapeHtml(p.name)}"${selected} disabled>\u23F3 ${escapeHtml(p.name)} (${escapeHtml(p.status || "bootstrapping")})</option>`;
    } else if (p.status === "not_found" || p.status === "stopped") {
      html += `<option value="${escapeHtml(p.name)}"${selected}>\uD83D\uDCA4 ${escapeHtml(p.name)} (${escapeHtml(p.status || "stopped")})</option>`;
    } else {
      const statusLabel = p.status ? ` (${p.status})` : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${statusLabel}</option>`;
    }
  }
  select.innerHTML = html;
}

async function _selectAnima(name) {
  const prevAnima = _selectedAnima;
  const currentInput = _$("chatPageInput");
  if (prevAnima && currentInput) {
    _saveDraft(prevAnima, currentInput.value || "");
  }

  _selectedAnima = name;
  _pendingQueue = [];
  _hidePendingIndicator();
  _selectedThreadId = "default";
  _updateVoiceAnima(name);

  if (!_threads[name]) {
    _threads[name] = [{ id: "default", label: "メイン", unread: false }];
  }

  const select = _$("chatPageAnimaSelect");
  if (select) select.value = name;

  const input = _$("chatPageInput");
  const sendBtn = _$("chatPageSendBtn");
  if (input) { input.disabled = false; input.placeholder = t("chat.message_to", { name }); }
  if (sendBtn) sendBtn.disabled = false;
  if (input) {
    input.value = _loadDraft(name);
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }
  _updateSendButton();

  // Load conversation history (activity_log API) + anima detail in parallel
  const tid = _selectedThreadId;
  const needConv = !_historyState[name]?.[tid] || _historyState[name][tid].sessions.length === 0;
  const convPromise = needConv
    ? _fetchConversationHistory(name, _HISTORY_PAGE_SIZE, null, tid).catch(() => null)
    : Promise.resolve(null);
  const detailPromise = api(`/api/animas/${encodeURIComponent(name)}`).catch(() => null);

  const [conv, detail] = await Promise.all([convPromise, detailPromise]);

  // Apply conversation history
  if (!_historyState[name]) _historyState[name] = {};
  if (conv && conv.sessions && conv.sessions.length > 0) {
    _historyState[name][tid] = {
      sessions: conv.sessions,
      hasMore: conv.has_more || false,
      nextBefore: conv.next_before || null,
      loading: false,
    };
  } else if (needConv) {
    _historyState[name][tid] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
  }

  _renderThreadTabs();
  _renderChat();

  // Apply anima detail
  if (detail) {
    _animaDetail = detail;
    _renderAnimaState();
  } else {
    _animaDetail = null;
    const stateEl = _$("chatAnimaState");
    if (stateEl) stateEl.textContent = t("animas.detail_load_failed");
  }

  // Load secondary data in parallel
  const secondaryPromises = [_updateAvatar(), _loadMemoryTab(), _loadActivity()];
  if (_activeRightTab === "history") secondaryPromises.push(_loadSessionList());
  await Promise.all(secondaryPromises);
}

// ── Thread Tabs ────────────────────────────

function _renderThreadTabs() {
  const container = _$("chatThreadTabs");
  if (!container || !_selectedAnima) return;

  const list = _threads[_selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
  let html = "";
  for (const t of list) {
    const activeClass = t.id === _selectedThreadId ? " active" : "";
    const closeBtn = t.id !== "default"
      ? ` <button type="button" class="thread-tab-close" data-thread="${escapeHtml(t.id)}" title="スレッドを閉じる" aria-label="閉じる">&times;</button>`
      : "";
    html += `<span class="thread-tab-wrap"><button type="button" class="thread-tab${activeClass}" data-thread="${escapeHtml(t.id)}">${escapeHtml(t.label)}</button>${closeBtn}</span>`;
  }
  html += `<button type="button" class="thread-tab-new" id="chatNewThreadBtn" title="新しいスレッド">＋</button>`;

  container.innerHTML = html;

  container.querySelectorAll(".thread-tab").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const tid = e.target.dataset.thread;
      if (tid) _selectThread(tid);
    });
  });
  container.querySelectorAll(".thread-tab-close").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const tid = e.target.dataset.thread;
      if (tid) _closeThread(tid);
    });
  });
  const newBtn = _$("chatNewThreadBtn");
  if (newBtn) newBtn.addEventListener("click", () => _createNewThread());
}

async function _selectThread(threadId) {
  if (threadId === _selectedThreadId) return;
  _selectedThreadId = threadId;
  _renderThreadTabs();

  const name = _selectedAnima;
  if (!name) return;

  const hs = _historyState[name]?.[threadId];
  const needLoad = !hs || hs.sessions.length === 0;
  if (needLoad) {
    try {
      const conv = await _fetchConversationHistory(name, _HISTORY_PAGE_SIZE, null, threadId);
      if (!_historyState[name]) _historyState[name] = {};
      if (conv && conv.sessions && conv.sessions.length > 0) {
        _historyState[name][threadId] = {
          sessions: conv.sessions,
          hasMore: conv.has_more || false,
          nextBefore: conv.next_before || null,
          loading: false,
        };
      } else {
        _historyState[name][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
      }
    } catch {
      if (!_historyState[name]) _historyState[name] = {};
      _historyState[name][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
    }
  }
  _renderChat();
}

function _createNewThread() {
  if (!_selectedAnima) return;
  const threadId = crypto.randomUUID().slice(0, 8);
  const list = _threads[_selectedAnima] || [{ id: "default", label: "メイン", unread: false }];
  list.push({ id: threadId, label: "新しいスレッド", unread: false });
  _threads[_selectedAnima] = list;

  if (!_chatHistories[_selectedAnima]) _chatHistories[_selectedAnima] = {};
  _chatHistories[_selectedAnima][threadId] = [];

  if (!_historyState[_selectedAnima]) _historyState[_selectedAnima] = {};
  _historyState[_selectedAnima][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };

  _renderThreadTabs();
  _selectThread(threadId);
}

function _closeThread(threadId) {
  if (threadId === "default") return;
  if (!_selectedAnima) return;

  const list = _threads[_selectedAnima];
  if (!list) return;
  const idx = list.findIndex((t) => t.id === threadId);
  if (idx < 0) return;

  list.splice(idx, 1);
  delete _chatHistories[_selectedAnima]?.[threadId];
  delete _historyState[_selectedAnima]?.[threadId];

  if (_selectedThreadId === threadId) {
    _selectedThreadId = "default";
  }
  _renderThreadTabs();
  _renderChat();
}

// ── Avatar ─────────────────────────────────

async function _updateAvatar() {
  const container = _$("chatPageAvatar");
  if (!container || !_selectedAnima) {
    if (container) container.innerHTML = "";
    _bustupUrl = null;
    return;
  }

  _bustupUrl = null;
  const name = _selectedAnima;
  const candidates = ["avatar_bustup.png", "avatar_chibi.png"];
  for (const filename of candidates) {
    const url = `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        if (filename === "avatar_bustup.png") _bustupUrl = url;
        container.innerHTML = `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="anima-avatar-img">`;
        container.style.cursor = _bustupUrl ? "pointer" : "";
        return;
      }
    } catch { /* try next */ }
  }
  container.style.cursor = "";
  container.innerHTML = `<div class="anima-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</div>`;
}

// ── Bustup Overlay ──────────────────────────

function _showBustupOverlay() {
  if (!_bustupUrl || !_selectedAnima) return;
  _removeBustupOverlay();

  const overlay = document.createElement("div");
  overlay.className = "bustup-overlay";
  overlay.id = "chatBustupOverlay";
  overlay.innerHTML = `<img class="bustup-overlay-img" src="${escapeHtml(_bustupUrl)}" alt="${escapeHtml(_selectedAnima)}">`;
  overlay.addEventListener("click", _dismissBustupOverlay);
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add("visible"));
}

function _dismissBustupOverlay() {
  const overlay = document.getElementById("chatBustupOverlay");
  if (!overlay) return;
  overlay.classList.remove("visible");
  overlay.classList.add("hiding");
  overlay.addEventListener("transitionend", () => overlay.remove(), { once: true });
}

function _removeBustupOverlay() {
  document.getElementById("chatBustupOverlay")?.remove();
}

function _onBustupEscape(e) {
  if (e.key === "Escape") _dismissBustupOverlay();
}

// ── Chat Rendering ─────────────────────────

function _renderChat(scrollToBottom = true) {
  const messagesEl = _$("chatPageMessages");
  if (!messagesEl) return;

  const name = _selectedAnima;
  const tid = _selectedThreadId;
  const history = _chatHistories[name]?.[tid] || [];
  const hs = _historyState[name]?.[tid] || { sessions: [], hasMore: false, nextBefore: null, loading: false };

  if (hs.sessions.length === 0 && history.length === 0) {
    if (hs.loading) {
      messagesEl.innerHTML = `<div class="chat-empty"><span class="tool-spinner"></span> ${t("common.loading")}</div>`;
    } else {
      messagesEl.innerHTML = `<div class="chat-empty">${t("chat.messages_empty")}</div>`;
    }
    return;
  }

  let topHtml = "";
  // Sentinel for infinite scroll
  if (hs.hasMore) {
    if (hs.loading) {
      topHtml += `<div class="history-loading-more"><span class="tool-spinner"></span> ${t("chat.past_loading")}</div>`;
    }
    topHtml += '<div class="chat-load-sentinel"></div>';
  }

  const prevScrollHeight = messagesEl.scrollHeight;

  // Render history sessions
  let sessionsHtml = "";
  for (let si = 0; si < hs.sessions.length; si++) {
    const session = hs.sessions[si];
    sessionsHtml += _renderSessionDivider(session, si === 0);
    if (session.messages) {
      for (const msg of session.messages) {
        sessionsHtml += _renderHistoryMessage(msg);
      }
    }
  }

  // Render live chat messages
  let liveHtml = "";
  if (history.length > 0) {
    if (hs.sessions.length > 0) {
      liveHtml += `<div class="session-divider"><span class="session-divider-label">${t("chat.current_session")}</span></div>`;
    }
    liveHtml += history.map(m => {
      const ts = m.timestamp ? smartTimestamp(m.timestamp) : "";
      const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

      if (m.role === "thinking") {
        return `<div class="chat-bubble thinking"><span class="thinking-animation">${t("chat.thinking")}</span></div>`;
      }
      if (m.role === "assistant") {
        const streamClass = m.streaming ? " streaming" : "";
        let thinkingHtml = "";
        if (m.thinkingText) {
          const thSummary = `Thinking (${m.thinkingText.length} chars)`;
          const thRendered = renderSafeMarkdown(m.thinkingText);
          thinkingHtml = `<details class="thinking-block"><summary class="thinking-summary"><span class="thinking-icon">💭</span> ${escapeHtml(thSummary)}</summary><div class="thinking-content">${thRendered}</div></details>`;
        }
        let content = "";
        if (m.text) {
          content = renderMarkdown(m.text);
        } else if (m.streaming) {
          content = '<span class="cursor-blink"></span>';
        }
        const toolHtml = m.activeTool
          ? `<div class="tool-indicator"><span class="tool-spinner"></span>${t("chat.tool_running", { tool: m.activeTool })}</div>`
          : "";
        return `<div class="chat-bubble assistant${streamClass}">${thinkingHtml}${content}${toolHtml}${tsHtml}</div>`;
      }
      const imagesHtml = renderChatImages(m.images);
      const textHtml = m.text ? `<div class="chat-text">${escapeHtml(m.text)}</div>` : "";
      return `<div class="chat-bubble user">${imagesHtml}${textHtml}${tsHtml}</div>`;
    }).join("");
  }

  messagesEl.innerHTML = topHtml + sessionsHtml + liveHtml;

  // Bind tool call handlers for history messages
  _bindToolCallHandlers(messagesEl);

  if (scrollToBottom) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else {
    const newScrollHeight = messagesEl.scrollHeight;
    messagesEl.scrollTop += (newScrollHeight - prevScrollHeight);
  }

  _observeChatSentinel();
}

// ── SSE Streaming Chat ─────────────────────

function _renderStreamingBubble(msg) {
  const messagesEl = _$("chatPageMessages");
  if (!messagesEl) return;
  const bubble = messagesEl.querySelector(".chat-bubble.assistant.streaming");
  if (!bubble) return;

  let html = "";

  if (msg.thinkingText) {
    const open = msg.thinking ? " open" : "";
    const summary = msg.thinking ? "Thinking..." : `Thinking (${msg.thinkingText.length} chars)`;
    const thHtml = renderSafeMarkdown(msg.thinkingText);
    html += `<details class="thinking-block"${open}><summary class="thinking-summary"><span class="thinking-icon">💭</span> ${escapeHtml(summary)}</summary><div class="thinking-content">${thHtml}</div></details>`;
  }

  if (msg.heartbeatRelay) {
    html += `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${t("chat.heartbeat_relay")}</div>`;
    if (msg.heartbeatText) {
      html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    }
  } else if (msg.afterHeartbeatRelay && !msg.text) {
    html = `<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>${t("chat.heartbeat_relay_done")}</div>`;
  } else if (msg.text) {
    html = renderMarkdown(msg.text);
  } else {
    html = '<span class="cursor-blink"></span>';
  }

  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${t("chat.tool_running", { tool: msg.activeTool })}</div>`;
  }

  bubble.innerHTML = html;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Infinite Scroll ─────────────────────────────

function _setupChatObserver() {
  if (_chatObserver) _chatObserver.disconnect();
  const messagesEl = _$("chatPageMessages");
  if (!messagesEl) return;

  _chatObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) _loadOlderMessages();
      }
    },
    { root: messagesEl, rootMargin: "200px 0px 0px 0px" },
  );
}

function _observeChatSentinel() {
  if (!_chatObserver) return;
  const messagesEl = _$("chatPageMessages");
  if (!messagesEl) return;
  const sentinel = messagesEl.querySelector(".chat-load-sentinel");
  if (sentinel) _chatObserver.observe(sentinel);
}

async function _loadOlderMessages() {
  const name = _selectedAnima;
  const tid = _selectedThreadId;
  if (!name) return;
  const hs = _historyState[name]?.[tid];
  if (!hs || !hs.hasMore || hs.loading) return;
  if (_isChatStreaming) return;

  hs.loading = true;
  _renderChat(false);

  try {
    const data = await _fetchConversationHistory(name, _HISTORY_PAGE_SIZE, hs.nextBefore, tid);

    if (data && data.sessions && data.sessions.length > 0) {
      hs.sessions = [...data.sessions, ...hs.sessions];
      hs.hasMore = data.has_more || false;
      hs.nextBefore = data.next_before || null;
    } else {
      hs.hasMore = false;
    }
  } catch (err) {
    logger.error("Failed to load older messages", { error: err.message });
  }
  hs.loading = false;

  _renderChat(false);
}

// ── Conversation History API ──────────────────

async function _fetchConversationHistory(animaName, limit = _HISTORY_PAGE_SIZE, before = null, threadId = "default") {
  let url = `/api/animas/${encodeURIComponent(animaName)}/conversation/history?limit=${limit}`;
  if (before) {
    url += `&before=${encodeURIComponent(before)}`;
  }
  url += `&thread_id=${encodeURIComponent(threadId)}`;
  return await api(url);
}

// ── History Message Rendering ─────────────────

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

  const input = tc.input || "";
  if (input) {
    const inputStr = typeof input === "string" ? input : JSON.stringify(input, null, 2);
    html += `<div class="tool-call-label">\u5165\u529B</div>`;
    html += `<div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">\u7D50\u679C</div>`;
    if (resultStr.length > _TOOL_RESULT_TRUNCATE) {
      const truncated = resultStr.slice(0, _TOOL_RESULT_TRUNCATE);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">\u3082\u3063\u3068\u898B\u308B</button>`;
    } else {
      html += `<div class="tool-call-content">${escapeHtml(resultStr)}</div>`;
    }
  }

  return html;
}

function _bindToolCallHandlers(container) {
  if (!container) return;

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
  if (isFirst) return "";

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

function _submitChat() {
  const input = _$("chatPageInput");
  if (!input) return;
  const msg = input.value.trim();
  const hasImages = _imageInputManager && _imageInputManager.getImageCount() > 0;

  // ── Not streaming ──
  if (!_isChatStreaming) {
    if (msg || hasImages) {
      _pendingQueue.push({
        text: msg,
        images: _imageInputManager?.getPendingImages() || [],
        displayImages: _imageInputManager?.getDisplayImages() || [],
      });
      _imageInputManager?.clearImages();
    }
    if (_pendingQueue.length === 0) return;
    const next = _pendingQueue.shift();
    _showPendingIndicator();
    if (_pendingQueue.length === 0) _hidePendingIndicator();
    _sendChat(next.text, { images: next.images, displayImages: next.displayImages });
    return;
  }

  // ── Streaming + has input → add to queue ──
  if (msg || hasImages) {
    _pendingQueue.push({
      text: msg,
      images: _imageInputManager?.getPendingImages() || [],
      displayImages: _imageInputManager?.getDisplayImages() || [],
    });
    input.value = "";
    input.style.height = "auto";
    _imageInputManager?.clearImages();
    _showPendingIndicator();
    _updateSendButton();
    return;
  }

  // ── Streaming + empty input + has queue → interrupt & drain queue ──
  if (_pendingQueue.length > 0) {
    _interruptAndSendPending();
    return;
  }

  // ── Streaming + empty input + no queue → just stop ──
  _stopStreaming();
}

async function _sendChat(message, overrideImages = null) {
  const name = _selectedAnima;
  const images = overrideImages?.images || _imageInputManager?.getPendingImages() || [];
  const displayImages = overrideImages?.displayImages || _imageInputManager?.getDisplayImages() || [];
  if (!name || (!message.trim() && images.length === 0)) return;

  // Guard: block sending to bootstrapping animas
  const currentAnima = _animas.find((p) => p.name === name);
  if (currentAnima?.status === "bootstrapping" || currentAnima?.bootstrapping) {
    const msgs = _$("chatPageMessages");
    if (msgs) {
      const systemMsg = document.createElement("div");
      systemMsg.className = "chat-bubble assistant";
      systemMsg.textContent = t("chat.bootstrapping");
      msgs.appendChild(systemMsg);
      msgs.scrollTop = msgs.scrollHeight;
    }
    return;
  }

  const tid = _selectedThreadId;
  if (!_chatHistories[name]) _chatHistories[name] = {};
  if (!_chatHistories[name][tid]) _chatHistories[name][tid] = [];
  const history = _chatHistories[name][tid];

  // Update thread label on first message if it's "新しいスレッド"
  const threadList = _threads[name] || [];
  const threadEntry = threadList.find((t) => t.id === tid);
  if (threadEntry && threadEntry.label === "新しいスレッド" && message.trim()) {
    threadEntry.label = message.trim().slice(0, 20) + (message.trim().length > 20 ? "..." : "");
    _renderThreadTabs();
  }

  const sendTs = new Date().toISOString();
  history.push({ role: "user", text: message, images: displayImages, timestamp: sendTs });
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: sendTs, thinkingText: "", thinking: false };
  history.push(streamingMsg);
  _renderChat();

  const input = _$("chatPageInput");
  const sendBtn = _$("chatPageSendBtn");
  _isChatStreaming = true;
  _chatAbortController = new AbortController();
  _updateSendButton();
  if (input) input.placeholder = t("chat.message_to", { name });

  if (!overrideImages) {
    _imageInputManager?.clearImages();
  }

  _addLocalActivity("chat", name, `${t("chat.user_prefix")} ${message}`);

  try {
    let sendSucceeded = false;
    const currentUser = localStorage.getItem("animaworks_user") || "human";
    const bodyObj = { message, from_person: currentUser, thread_id: tid };
    if (images.length > 0) {
      bodyObj.images = images;
    }
    const body = JSON.stringify(bodyObj);

    logger.debug(`_sendChat: starting stream for ${name} msg_len=${message.length}`);
    await streamChat(name, body, _chatAbortController.signal, {
      onTextDelta: (text) => {
        streamingMsg.afterHeartbeatRelay = false;
        streamingMsg.text += text;
        logger.debug(`onTextDelta: delta_len=${text.length} total_len=${streamingMsg.text.length}`);
        _renderStreamingBubble(streamingMsg);
      },
      onToolStart: (toolName) => {
        logger.debug(`onToolStart: ${toolName}`);
        streamingMsg.activeTool = toolName;
        _renderStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {
        logger.debug("onToolEnd");
        streamingMsg.activeTool = null;
        _renderStreamingBubble(streamingMsg);
      },
      onChainStart: () => {
        logger.debug("onChainStart");
      },
      onHeartbeatRelayStart: ({ message }) => {
        logger.debug(`onHeartbeatRelayStart: ${message}`);
        streamingMsg.heartbeatRelay = true;
        streamingMsg.heartbeatText = "";
        streamingMsg.text = "";
        _renderStreamingBubble(streamingMsg);
        _addLocalActivity("system", name, `${t("chat.heartbeat_relay")}: ${message}`);
      },
      onHeartbeatRelay: ({ text }) => {
        streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
        logger.debug(`onHeartbeatRelay: delta_len=${text.length} total_len=${(streamingMsg.heartbeatText || "").length}`);
        _renderStreamingBubble(streamingMsg);
      },
      onHeartbeatRelayDone: () => {
        logger.debug(`onHeartbeatRelayDone: transitioning to afterHeartbeatRelay, text_len=${streamingMsg.text.length}`);
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = true;
        _renderStreamingBubble(streamingMsg);
      },
      onThinkingStart: () => {
        streamingMsg.thinkingText = "";
        streamingMsg.thinking = true;
        _renderStreamingBubble(streamingMsg);
      },
      onThinkingDelta: (text) => {
        streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text;
        _renderStreamingBubble(streamingMsg);
      },
      onThinkingEnd: () => {
        streamingMsg.thinking = false;
        _renderStreamingBubble(streamingMsg);
      },
      onError: ({ message: errorMsg }) => {
        logger.debug(`onError: ${errorMsg}`);
        streamingMsg.text += `\n${t("chat.error_prefix")} ${errorMsg}`;
        streamingMsg.streaming = false;
        _renderChat();
      },
      onDone: ({ summary }) => {
        const summaryLen = (summary || "").length;
        const textLen = streamingMsg.text.length;
        logger.debug(`onDone: summary_len=${summaryLen} text_len=${textLen} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
        const text = summary || streamingMsg.text;
        streamingMsg.text = text || t("chat.empty_response");
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = false;
        logger.debug(`onDone: final text_len=${streamingMsg.text.length}`);
        _renderChat();
        _addLocalActivity("chat", name, `${t("chat.response_prefix")} ${streamingMsg.text.slice(0, 100)}`);
      },
    });

    // Ensure streaming is finalized if stream ended without done event
    if (streamingMsg.streaming) {
      logger.debug(`_sendChat: finalize fallback — text_len=${streamingMsg.text.length} afterRelay=${streamingMsg.afterHeartbeatRelay}`);
      streamingMsg.streaming = false;
      if (!streamingMsg.text) {
        streamingMsg.text = streamingMsg.afterHeartbeatRelay
          ? t("chat.receive_failed")
          : t("chat.empty_response");
      }
      streamingMsg.afterHeartbeatRelay = false;
      _renderChat();
    }
    logger.debug(`_sendChat: stream completed for ${name}`);
    sendSucceeded = true;

    const input = _$("chatPageInput");
    if (sendSucceeded && input && input.value.trim() === message.trim()) {
      input.value = "";
      input.style.height = "auto";
      _clearDraft(name);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      if (!streamingMsg.text) streamingMsg.text = t("chat.interrupted");
      _renderChat();
    } else {
      logger.error("Chat stream error", { anima: name, error: err.message, name: err.name });
      streamingMsg.text += `\n${t("chat.error_prefix")} ${err.message}`;
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      _renderChat();
    }
  } finally {
    _isChatStreaming = false;
    _chatAbortController = null;
    if (input) {
      input.placeholder = t("chat.message_to", { name });
      _saveDraft(name, input.value || "");
      input.focus();
    }
    _updateSendButton();

    // Auto-send next queued message
    if (_pendingQueue.length > 0) {
      const next = _pendingQueue.shift();
      _showPendingIndicator();
      if (_pendingQueue.length === 0) _hidePendingIndicator();
      setTimeout(() => {
        _sendChat(next.text, { images: next.images, displayImages: next.displayImages });
      }, 150);
    }
  }
}

// ── Queue Helpers ─────────────────

function _addToQueue() {
  const input = _$("chatPageInput");
  if (!input) return;
  const msg = input.value.trim();
  const hasImages = _imageInputManager && _imageInputManager.getImageCount() > 0;
  if (!msg && !hasImages) return;
  _pendingQueue.push({
    text: msg,
    images: _imageInputManager?.getPendingImages() || [],
    displayImages: _imageInputManager?.getDisplayImages() || [],
  });
  input.value = "";
  input.style.height = "auto";
  _imageInputManager?.clearImages();
  _showPendingIndicator();
  _updateSendButton();
}

function _showPendingIndicator() {
  const bar = _$("chatPagePending");
  const list = _$("chatPagePendingList");
  const label = _$("chatPagePendingLabel");
  if (!bar || !list) return;
  if (_pendingQueue.length === 0) { bar.style.display = "none"; return; }
  if (label) label.textContent = `${t("chat.queue_label")} (${_pendingQueue.length})`;
  list.innerHTML = _pendingQueue.map((p, i) => {
    const txt = escapeHtml(p.text.length > 60 ? p.text.slice(0, 60) + "…" : p.text);
    const img = p.images?.length ? ` <span style="opacity:0.6">(+${p.images.length} images)</span>` : "";
    return `<div class="pending-queue-item" data-idx="${i}">` +
      `<span class="pending-queue-item-num">${i + 1}.</span>` +
      `<span class="pending-queue-item-text">${txt || "(images only)"}${img}</span>` +
      `<button class="pending-queue-item-del" data-idx="${i}" type="button">✕</button>` +
      `</div>`;
  }).join("");
  bar.style.display = "";

  list.onclick = (e) => {
    const delBtn = e.target.closest(".pending-queue-item-del");
    if (delBtn) {
      e.stopPropagation();
      const idx = parseInt(delBtn.dataset.idx, 10);
      _pendingQueue.splice(idx, 1);
      _showPendingIndicator();
      _updateSendButton();
      return;
    }
    const item = e.target.closest(".pending-queue-item");
    if (item) {
      const idx = parseInt(item.dataset.idx, 10);
      const removed = _pendingQueue.splice(idx, 1)[0];
      if (removed) {
        const input = _$("chatPageInput");
        if (input) { input.value = removed.text; input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; input.focus(); }
      }
      _showPendingIndicator();
      _updateSendButton();
    }
  };
}

function _hidePendingIndicator() {
  const bar = _$("chatPagePending");
  if (bar) bar.style.display = "none";
}

function _updateSendButton() {
  const sendBtn = _$("chatPageSendBtn");
  const queueBtn = _$("chatPageQueueBtn");
  const inputVal = _$("chatPageInput")?.value?.trim() || "";
  const hasInput = inputVal.length > 0;

  if (queueBtn) {
    queueBtn.disabled = !hasInput || !_selectedAnima;
  }

  if (!sendBtn) return;
  sendBtn.classList.remove("stop", "interrupt");
  if (!_isChatStreaming) {
    sendBtn.textContent = (_pendingQueue.length > 0 || hasInput) ? "↑" : "↑";
    sendBtn.disabled = !_selectedAnima || (!hasInput && _pendingQueue.length === 0);
  } else if (hasInput) {
    sendBtn.textContent = "↑";
    sendBtn.disabled = false;
  } else if (_pendingQueue.length > 0) {
    sendBtn.textContent = "■↑";
    sendBtn.classList.add("interrupt");
    sendBtn.disabled = false;
  } else {
    sendBtn.textContent = "■";
    sendBtn.classList.add("stop");
    sendBtn.disabled = false;
  }
}

function _stopStreaming() {
  if (!_selectedAnima) return;
  if (_chatAbortController) {
    _chatAbortController.abort();
  }
  fetch(`/api/animas/${encodeURIComponent(_selectedAnima)}/interrupt`, {
    method: "POST",
  }).catch(() => {});
}

function _interruptAndSendPending() {
  _stopStreaming();
}

// ── Mobile Tab Switching ─────────────────────

function _switchMobileTab(panel) {
  const chatTab = _$("chatMobileTabChat");
  const infoTab = _$("chatMobileTabInfo");
  const mainPanel = _container?.querySelector(".chat-page-main");
  const sidePanel = _container?.querySelector(".chat-page-sidebar");
  if (!mainPanel || !sidePanel) return;

  if (panel === "chat") {
    chatTab?.classList.add("active");
    infoTab?.classList.remove("active");
    mainPanel.classList.remove("mobile-hidden");
    sidePanel.classList.add("mobile-hidden");
  } else {
    chatTab?.classList.remove("active");
    infoTab?.classList.add("active");
    mainPanel.classList.add("mobile-hidden");
    sidePanel.classList.remove("mobile-hidden");
  }
}

// ── Right Tab Switching ────────────────────

function _switchRightTab(tab) {
  _activeRightTab = tab;
  const tabs = { state: "chatPaneState", activity: "chatPaneActivity", history: "chatPaneHistory" };

  for (const btn of (_container?.querySelectorAll(".right-tab") || [])) {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  }
  for (const [key, id] of Object.entries(tabs)) {
    const el = _$(id);
    if (el) el.style.display = key === tab ? "" : "none";
  }

  if (tab === "history" && _selectedAnima) {
    const detail = _$("chatHistoryDetail");
    const list = _$("chatHistorySessionList");
    if (detail) detail.style.display = "none";
    if (list) list.style.display = "";
    _loadSessionList();
  }
  if (tab === "activity") _loadActivity();
}

// ── Anima State ───────────────────────────

function _renderAnimaState() {
  const el = _$("chatAnimaState");
  if (!el) return;

  const d = _animaDetail;
  if (!d || !d.state) {
    el.textContent = t("animas.no_state");
    return;
  }
  el.textContent = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
}

// ── Activity Feed ──────────────────────────

function _addLocalActivity(type, animaName, summary) {
  const feed = _$("chatActivityFeed");
  if (!feed) return;

  // Remove empty state
  const empty = feed.querySelector(".activity-empty");
  if (empty) empty.remove();

  const icon = getIcon(type);
  const ts = new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const entry = document.createElement("div");
  entry.className = "activity-entry";
  entry.innerHTML = `
    <span class="activity-icon">${icon}</span>
    <span class="activity-time">${ts}</span>
    <div class="activity-body">
      <span class="activity-anima">${escapeHtml(animaName)}</span>
      <span class="activity-summary"> ${escapeHtml(summary)}</span>
    </div>`;
  feed.appendChild(entry);
  feed.scrollTop = feed.scrollHeight;

  while (feed.children.length > 200) {
    feed.removeChild(feed.firstChild);
  }
}

async function _loadActivity() {
  if (!_selectedAnima) return;

  try {
    const data = await api(`/api/activity/recent?hours=6&anima=${encodeURIComponent(_selectedAnima)}`);
    const events = data.events || [];
    const feed = _$("chatActivityFeed");
    if (!feed) return;

    if (events.length === 0) {
      feed.innerHTML = `<div class="activity-empty">${t("activity.empty")}</div>`;
      return;
    }

    feed.innerHTML = events.slice(0, 50).map(evt => {
      const icon = getIcon(evt.type);
      const ts = timeStr(evt.ts);
      const summary = getDisplaySummary(evt);
      return `
        <div class="activity-entry">
          <span class="activity-icon">${icon}</span>
          <span class="activity-time">${escapeHtml(ts)}</span>
          <div class="activity-body">
            <span class="activity-anima">${escapeHtml(evt.anima || "")}</span>
            <span class="activity-summary"> ${escapeHtml(summary)}</span>
          </div>
        </div>`;
    }).join("");
  } catch {
    // Silent fail — keep existing content
  }
}

// ── History Panel ──────────────────────────

async function _loadSessionList() {
  const list = _$("chatHistorySessionList");
  if (!list || !_selectedAnima) {
    if (list) list.innerHTML = `<div class="loading-placeholder">${t("chat.anima_select_first")}</div>`;
    return;
  }

  list.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/sessions`);
    let html = "";

    // Active Conversation
    if (data.active_conversation) {
      const ac = data.active_conversation;
      const lastTime = ac.last_timestamp ? timeStr(ac.last_timestamp) : "--:--";
      html += `
        <div class="session-section-header">${t("chat.history_current")}</div>
        <div class="session-item session-active" data-type="active">
          <div class="session-item-title">${t("ws.conversation_active")}</div>
          <div class="session-item-meta">
            ${t("chat.session_turns", { count: ac.total_turn_count })} ${ac.has_summary ? t("chat.session_summary") : ""} | ${t("chat.last_label")}: ${lastTime}
          </div>
        </div>`;
    }

    // Archived Sessions
    if (data.archived_sessions && data.archived_sessions.length > 0) {
      html += `<div class="session-section-header">${t("chat.history_archive")}</div>`;
      for (const s of data.archived_sessions) {
        const ts = s.timestamp ? timeStr(s.timestamp) : s.id;
        html += `
          <div class="session-item" data-type="archive" data-id="${escapeHtml(s.id)}">
            <div class="session-item-title">${escapeHtml(s.trigger || t("chat.session"))} (${t("chat.session_turns", { count: s.turn_count })})</div>
            <div class="session-item-meta">${ts} | ctx: ${(s.context_usage_ratio * 100).toFixed(0)}%</div>
            ${s.original_prompt_preview ? `<div class="session-item-preview">${escapeHtml(s.original_prompt_preview)}</div>` : ""}
          </div>`;
      }
    }

    // Transcripts
    if (data.transcripts && data.transcripts.length > 0) {
      html += `<div class="session-section-header">${t("chat.history_transcript")}</div>`;
      for (const t of data.transcripts) {
        html += `
          <div class="session-item" data-type="transcript" data-date="${escapeHtml(t.date)}">
            <div class="session-item-title">${escapeHtml(t.date)}</div>
            <div class="session-item-meta">${t.message_count} messages</div>
          </div>`;
      }
    }

    // Episodes
    if (data.episodes && data.episodes.length > 0) {
      html += `<div class="session-section-header">${t("chat.episode_log")}</div>`;
      for (const e of data.episodes) {
        html += `
          <div class="session-item" data-type="episode" data-date="${escapeHtml(e.date)}">
            <div class="session-item-title">${escapeHtml(e.date)}</div>
            <div class="session-item-preview">${escapeHtml(e.preview)}</div>
          </div>`;
      }
    }

    if (!html) html = `<div class="loading-placeholder">${t("chat.history_empty")}</div>`;
    list.innerHTML = html;

    // Bind click handlers
    list.querySelectorAll(".session-item").forEach(item => {
      item.addEventListener("click", () => {
        const type = item.dataset.type;
        if (type === "active") _loadActiveConversation();
        else if (type === "archive") _loadArchivedSession(item.dataset.id);
        else if (type === "transcript") _loadTranscript(item.dataset.date);
        else if (type === "episode") _loadEpisode(item.dataset.date);
      });
    });
  } catch (err) {
    list.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

function _showHistoryDetail(title) {
  const list = _$("chatHistorySessionList");
  const detail = _$("chatHistoryDetail");
  const titleEl = _$("chatHistoryDetailTitle");
  if (list) list.style.display = "none";
  if (detail) detail.style.display = "";
  if (titleEl) titleEl.textContent = title;
}

async function _loadActiveConversation() {
  if (!_selectedAnima) return;
  _showHistoryDetail(t("chat.history_detail_title"));
  const conv = _$("chatHistoryConversation");
  if (conv) conv.innerHTML = '<div class="loading-placeholder">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';

  try {
    const data = await _fetchConversationHistory(_selectedAnima, 50, null, _selectedThreadId);
    _renderConversationHistoryDetail(data);
  } catch {
    if (conv) conv.innerHTML = '<div class="loading-placeholder">\u8AAD\u307F\u8FBC\u307F\u5931\u6557</div>';
  }
}

function _renderConversationHistoryDetail(data) {
  const conv = _$("chatHistoryConversation");
  if (!conv) return;

  if (!data || !data.sessions || data.sessions.length === 0) {
    conv.innerHTML = '<div class="loading-placeholder">\u4F1A\u8A71\u30C7\u30FC\u30BF\u304C\u3042\u308A\u307E\u305B\u3093</div>';
    return;
  }

  let html = "";
  for (let si = 0; si < data.sessions.length; si++) {
    const session = data.sessions[si];
    html += _renderSessionDivider(session, si === 0);
    if (session.messages) {
      for (const msg of session.messages) {
        html += _renderHistoryMessage(msg);
      }
    }
  }

  if (!html) html = '<div class="loading-placeholder">\u4F1A\u8A71\u30C7\u30FC\u30BF\u304C\u3042\u308A\u307E\u305B\u3093</div>';
  conv.innerHTML = html;

  // Bind tool call handlers
  _bindToolCallHandlers(conv);

  conv.scrollTop = conv.scrollHeight;
}

async function _loadArchivedSession(sessionId) {
  if (!_selectedAnima) return;
  _showHistoryDetail(t("chat.session_detail", { id: sessionId }));
  const conv = _$("chatHistoryConversation");
  if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/sessions/${encodeURIComponent(sessionId)}`);
    if (data.markdown) {
      if (conv) conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.markdown)}</div>`;
    } else if (data.data) {
      const d = data.data;
      let html = `<div class="history-session-meta">
        <div><strong>${t("chat.state_label")}</strong> ${escapeHtml(d.trigger || t("chat.state_unknown"))}</div>
        <div><strong>${t("chat.turn_count")}</strong> ${d.turn_count || 0}</div>
        <div><strong>${t("chat.context_usage")}</strong> ${((d.context_usage_ratio || 0) * 100).toFixed(0)}%</div>
      </div>`;
      if (d.original_prompt) {
        html += `<div class="history-section"><div class="history-section-label">${t("chat.request_label")}</div><pre class="history-pre">${escapeHtml(d.original_prompt)}</pre></div>`;
      }
      if (d.accumulated_response) {
        html += `<div class="history-section"><div class="history-section-label">${t("chat.response_label")}</div><div>${renderMarkdown(d.accumulated_response)}</div></div>`;
      }
      if (conv) conv.innerHTML = html;
    } else {
      if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("chat.no_data")}</div>`;
    }
  } catch {
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
  }
}

async function _loadTranscript(date) {
  if (!_selectedAnima) return;
  _showHistoryDetail(t("chat.transcript_detail", { date }));
  const conv = _$("chatHistoryConversation");
  if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/transcripts/${encodeURIComponent(date)}`);
    _renderTranscriptDetail(data);
  } catch {
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
  }
}

function _renderTranscriptDetail(data) {
  const conv = _$("chatHistoryConversation");
  if (!conv) return;

  let html = "";
  if (data.turns && data.turns.length > 0) {
    for (const turn of data.turns) {
      const ts = turn.timestamp ? timeStr(turn.timestamp) : "";
      const bubbleClass = turn.role === "assistant" ? "assistant" : "user";
      const roleLabel = turn.role === "human" ? t("chat.role_human") : turn.role;
      const content = turn.role === "assistant" ? renderMarkdown(turn.content || "") : escapeHtml(turn.content || "");
      html += `
        <div class="history-turn">
          <div class="history-turn-meta">${ts} - ${escapeHtml(roleLabel)}</div>
          <div class="chat-bubble ${bubbleClass}">${content}</div>
        </div>`;
    }
  }

  if (!html) html = `<div class="loading-placeholder">${t("chat.history_no_data")}</div>`;
  conv.innerHTML = html;
  conv.scrollTop = conv.scrollHeight;
}

async function _loadEpisode(date) {
  if (!_selectedAnima) return;
  _showHistoryDetail(t("chat.episode_detail", { date }));
  const conv = _$("chatHistoryConversation");
  if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  try {
    const data = await api(`/api/animas/${encodeURIComponent(_selectedAnima)}/episodes/${encodeURIComponent(date)}`);
    if (conv) conv.innerHTML = `<div class="history-markdown">${renderMarkdown(data.content || t("chat.no_content"))}</div>`;
  } catch {
    if (conv) conv.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}</div>`;
  }
}

// ── Memory Browser ─────────────────────────

async function _loadMemoryTab() {
  const fileList = _$("chatMemoryFileList");
  if (!fileList) return;

  if (!_selectedAnima) {
    fileList.innerHTML = `<div class="loading-placeholder">${t("chat.anima_select_first")}</div>`;
    return;
  }

  fileList.innerHTML = `<div class="loading-placeholder">${t("common.loading")}</div>`;

  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${_activeMemoryTab}`;

  try {
    const data = await api(endpoint);
    const files = data.files || [];
    if (files.length === 0) {
      fileList.innerHTML = `<div class="loading-placeholder">${t("memory.no_files")}</div>`;
      return;
    }

    fileList.innerHTML = files.map(f =>
      `<div class="memory-file-item" data-file="${escapeHtml(f)}" data-tab="${_activeMemoryTab}">${escapeHtml(f)}</div>`
    ).join("");

    fileList.querySelectorAll(".memory-file-item").forEach(item => {
      item.addEventListener("click", () => {
        _loadMemoryContent(item.dataset.tab, item.dataset.file);
      });
    });
  } catch (err) {
    fileList.innerHTML = `<div class="loading-placeholder">${t("common.load_failed")}: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Voice Chat Callbacks ────────────────────

function _buildVoiceChatCallbacks(animaName) {
  return {
    addUserBubble(text) {
      const tid = _selectedThreadId;
      if (!_chatHistories[animaName]) _chatHistories[animaName] = {};
      if (!_chatHistories[animaName][tid]) _chatHistories[animaName][tid] = [];
      _chatHistories[animaName][tid].push({ role: 'user', text, timestamp: new Date().toISOString() });
      _renderChat();
    },
    addStreamingBubble() {
      const tid = _selectedThreadId;
      if (!_chatHistories[animaName]) _chatHistories[animaName] = {};
      if (!_chatHistories[animaName][tid]) _chatHistories[animaName][tid] = [];
      const msg = { role: 'assistant', text: '', streaming: true, activeTool: null, timestamp: new Date().toISOString() };
      _chatHistories[animaName][tid].push(msg);
      _renderChat();
      return msg;
    },
    updateStreamingBubble(msg) {
      _renderStreamingBubble(msg);
    },
    finalizeStreamingBubble(_msg) {
      _renderChat();
    },
  };
}

function _updateVoiceAnima(animaName) {
  updateVoiceUIAnima(animaName);
  const chatInputForm = _$("chatPageForm") || document.querySelector(".chat-input-form");
  if (chatInputForm && animaName) {
    initVoiceUI(chatInputForm, animaName, _buildVoiceChatCallbacks(animaName));
  }
}

// ── Image Input Initialization ──────────────

function _initImageInput() {
  const chatMain = _container?.querySelector(".chat-page-main");
  const previewEl = _$("chatPagePreviewBar");
  const chatInput = _$("chatPageInput");

  if (!chatMain || !previewEl || !chatInput) return;

  _imageInputManager = createImageInput({
    container: chatMain,
    inputArea: chatInput,
    previewContainer: previewEl,
  });

  // Initialize lightbox for image clicks
  initLightbox();

  // Initialize voice input
  const chatInputFormEl = _$("chatPageForm") || document.querySelector(".chat-input-form");
  if (chatInputFormEl && _selectedAnima) {
    initVoiceUI(chatInputFormEl, _selectedAnima, _buildVoiceChatCallbacks(_selectedAnima));
  }
}

async function _loadMemoryContent(tab, file) {
  if (!_selectedAnima) return;

  const fileList = _$("chatMemoryFileList");
  const contentArea = _$("chatMemoryContentArea");
  const titleEl = _$("chatMemoryContentTitle");
  const bodyEl = _$("chatMemoryContentBody");

  if (fileList) fileList.style.display = "none";
  if (contentArea) contentArea.style.display = "";
  if (titleEl) titleEl.textContent = file;
    if (bodyEl) bodyEl.textContent = t("common.loading");

  const endpoint = `/api/animas/${encodeURIComponent(_selectedAnima)}/${tab}/${encodeURIComponent(file)}`;

  try {
    const data = await api(endpoint);
    if (bodyEl) bodyEl.textContent = data.content || t("chat.no_content");
  } catch (err) {
    if (bodyEl) bodyEl.textContent = `${t("chat.error_prefix")} ${err.message}`;
  }
}
