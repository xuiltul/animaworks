// ── Board Page (Channels & DMs) ──────────────
import { api } from "../modules/api.js";
import { escapeHtml, renderMarkdown, smartTimestamp } from "../modules/state.js";
import { createLogger } from "../shared/logger.js";
import { t } from "/shared/i18n.js";
import { bustupCandidates, resolveCachedAvatar } from "../modules/avatar-resolver.js";
import { preloadAvatars } from "../modules/image-cache.js";

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
let _loadingMore = false;
let _hasMore = false;
let _currentOffset = 0;
let _dmFilterAnima = "";     // "" = show all, else filter by participant name
let _avatarCache = {};       // name -> url | null (null = no avatar)
let _dmLoaded = false;       // DM list lazy-load flag
let _lastMessageTs = "";     // newest message timestamp for incremental polling
let _discordChannels = null; // Discord channel list (lazy-loaded)

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
  _loadingMore = false;
  _hasMore = false;
  _discordChannels = null;
  _currentOffset = 0;
  _dmLoaded = false;
  _lastMessageTs = "";

  container.innerHTML = `
    <div class="board-layout">
      <!-- Mobile toggle for sidebar -->
      <button class="board-mobile-toggle" id="boardMobileToggle">${t("board.channel_toggle_hide")}</button>

      <!-- Left Sidebar -->
      <div class="board-sidebar" id="boardSidebar">
        <div class="board-sidebar-section">Channels</div>
        <div id="boardChannelList">
          <div class="loading-placeholder">${t("board.loading")}</div>
        </div>
        <div class="board-sidebar-divider"></div>
        <div class="board-sidebar-section board-dm-section-header">
          <span>DMs</span>
          <select id="boardDmFilter" class="board-dm-filter" title="${t("board.dm_filter_title")}">
            <option value="">${t("board.dm_filter_all")}</option>
          </select>
        </div>
        <div id="boardDmList">
          <div class="loading-placeholder">${t("common.loading")}</div>
        </div>
      </div>

      <!-- Right Panel -->
      <div class="board-main" id="boardMain">
        <div class="board-placeholder" id="boardPlaceholder">
          <div class="board-placeholder-icon">&#x1F4CB;</div>
          <div class="board-placeholder-text">${t("board.channel_select")}</div>
        </div>

        <!-- Channel/DM view (hidden initially) -->
        <div id="boardView" style="display:none;">
          <div class="board-header" id="boardHeader">
            <div class="board-header-title" id="boardHeaderTitle"></div>
            <div class="board-header-meta" id="boardHeaderMeta"></div>
          </div>
          <div class="board-messages" id="boardMessages">
            <div class="board-messages-empty">${t("board.messages_empty")}</div>
          </div>
          <form class="board-input-bar" id="boardInputBar" style="display:none;">
            <textarea
              id="boardInput"
              class="board-input"
              placeholder="${t("board.message_placeholder")}"
              autocomplete="off"
              rows="1"
            ></textarea>
            <button type="submit" class="board-send-btn" id="boardSendBtn">${t("board.send")}</button>
          </form>
        </div>
      </div>
    </div>
  `;

  _bindEvents();
  _loadSidebarData();

  // Polling fallback: incremental refresh every 30s
  _refreshInterval = setInterval(() => {
    if (_selectedType && _selectedName) {
      _pollNewMessages();
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
  _loadingMore = false;
  _hasMore = false;
  _currentOffset = 0;
  _dmFilterAnima = "";
  _avatarCache = {};
  _dmLoaded = false;
  _lastMessageTs = "";
}

// ── Event Binding ──────────────────────────

function _bindEvents() {
  // Form submit
  _addListener("boardInputBar", "submit", (e) => {
    e.preventDefault();
    _submitMessage();
  });

  // Textarea: Enter sends, Ctrl/Cmd+Enter always sends
  _addListener("boardInput", "keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      _submitMessage();
    } else if (e.key === "Enter" && !e.shiftKey) {
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

  // DM filter change
  _addListener("boardDmFilter", "change", () => {
    const select = _$("boardDmFilter");
    _dmFilterAnima = select ? select.value : "";
    _renderDmList();
  });

  // Mobile sidebar toggle
  _addListener("boardMobileToggle", "click", () => {
    const sidebar = _$("boardSidebar");
    if (!sidebar) return;
    _sidebarCollapsed = !_sidebarCollapsed;
    sidebar.classList.toggle("mobile-hidden", _sidebarCollapsed);
    const btn = _$("boardMobileToggle");
    if (btn) {
      btn.textContent = _sidebarCollapsed ? t("board.channel_toggle_show") : t("board.channel_toggle_hide");
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

// ── Avatar Cache ────────────────────────────

async function _resolveAvatar(name) {
  if (name in _avatarCache) return _avatarCache[name];
  const url = await resolveCachedAvatar(name, bustupCandidates(), "S");
  _avatarCache[name] = url;
  return url;
}

async function _preloadAvatars(names) {
  const uncached = names.filter(n => !(n in _avatarCache));
  if (uncached.length === 0) return;
  await Promise.all(uncached.map(n => _resolveAvatar(n)));
}

function _avatarHtml(name, isHuman) {
  const initial = name.charAt(0).toUpperCase();
  const url = _avatarCache[name];
  if (url && !isHuman) {
    return `<div class="board-msg-avatar"><img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="board-msg-avatar-img"></div>`;
  }
  const cls = isHuman ? "board-msg-avatar human" : "board-msg-avatar";
  return `<div class="${cls}">${escapeHtml(initial)}</div>`;
}

function _miniAvatarHtml(name) {
  const url = _avatarCache[name];
  if (url) {
    return `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" class="board-dm-mini-avatar">`;
  }
  return `<span class="board-dm-mini-avatar-placeholder">${escapeHtml(name.charAt(0).toUpperCase())}</span>`;
}

// ── Sidebar Data Loading ───────────────────

async function _loadSidebarData() {
  // Load channels first (lightweight), render immediately
  const channels = await api("/api/channels").catch(() => []);
  _channels = channels || [];
  _renderChannelList();

  // Defer DM list loading to avoid blocking initial render
  _loadDmList();
}

async function _loadDmList() {
  const dms = await api("/api/dm").catch(() => []);
  _dmPairs = dms || [];
  _dmLoaded = true;

  const dmNames = new Set();
  for (const dm of _dmPairs) {
    for (const p of dm.participants || []) dmNames.add(p);
  }
  _preloadAvatars([...dmNames]);

  _renderDmList();
}

function _renderChannelList() {
  const el = _$("boardChannelList");
  if (!el) return;

  if (_channels.length === 0) {
    el.innerHTML = `<div class="loading-placeholder" style="padding:8px 16px; font-size:0.82rem;">${t("board.channel_none")}</div>`;
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

  // Populate filter dropdown with unique participant names
  _populateDmFilter();

  // Apply filter + sort alphabetically by pair key
  const filtered = (_dmFilterAnima
    ? _dmPairs.filter(dm => (dm.participants || []).includes(_dmFilterAnima))
    : [..._dmPairs]
  ).sort((a, b) => (a.pair || "").localeCompare(b.pair || ""));

  if (filtered.length === 0) {
    const msg = _dmPairs.length === 0 ? t("board.dm_none") : t("board.dm_none_filtered");
    el.innerHTML = `<div class="loading-placeholder" style="padding:8px 16px; font-size:0.82rem;">${msg}</div>`;
    return;
  }

  el.innerHTML = filtered.map(dm => {
    const pair = dm.pair || "";
    const parts = dm.participants || [];
    const activeClass = _selectedType === "dm" && _selectedName === pair ? " active" : "";
    const count = dm.message_count ?? 0;
    const label = parts.length === 2
      ? `${_miniAvatarHtml(parts[0])}<span>${escapeHtml(parts[0])}</span><span class="board-dm-arrow">↔</span>${_miniAvatarHtml(parts[1])}<span>${escapeHtml(parts[1])}</span>`
      : `<span>${escapeHtml(parts.join(" ↔ ") || pair)}</span>`;
    return `
      <div class="board-sidebar-item board-dm-item${activeClass}" data-type="dm" data-name="${escapeHtml(pair)}" tabindex="0" role="button">
        <span class="board-dm-participants">${label}</span>
        ${count > 0 ? `<span class="board-sidebar-item-count">${count}</span>` : ""}
      </div>`;
  }).join("");

  _bindSidebarClicks(el);
}

function _populateDmFilter() {
  const select = _$("boardDmFilter");
  if (!select) return;

  const names = new Set();
  for (const dm of _dmPairs) {
    for (const p of dm.participants || []) names.add(p);
  }
  const sorted = [...names].sort((a, b) => a.localeCompare(b, "ja"));

  const prev = select.value;
  const optionsHtml = [`<option value="">${t("board.dm_filter_all")}</option>`]
    .concat(sorted.map(n => `<option value="${escapeHtml(n)}"${n === prev ? " selected" : ""}>${escapeHtml(n)}</option>`))
    .join("");

  if (select.innerHTML !== optionsHtml) {
    select.innerHTML = optionsHtml;
    select.value = prev || "";
  }
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
    if (btn) btn.textContent = t("board.channel_toggle_show");
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
    if (metaEl) {
      metaEl.textContent = "";
      _renderChannelMembers(name, metaEl);
    }
  } else {
    const dmInfo = _dmPairs.find(d => d.pair === name);
    const participants = dmInfo ? (dmInfo.participants || []).join(" ↔ ") : name;
    if (titleEl) titleEl.textContent = participants;
    if (metaEl) metaEl.textContent = t("board.direct_message");
  }

  // Show/hide input bar (hidden for DMs)
  const inputBar = _$("boardInputBar");
  if (inputBar) inputBar.style.display = type === "channel" ? "flex" : "none";

  // Update input placeholder
  const input = _$("boardInput");
  if (input && type === "channel") {
    input.placeholder = t("board.message_post", { channel: name });
  }

  // Bind scroll handler for infinite scroll
  const messagesEl = _$("boardMessages");
  if (messagesEl && !messagesEl.__scrollBound) {
    messagesEl.addEventListener("scroll", _onMessagesScroll);
    _boundListeners.push({ el: messagesEl, event: "scroll", handler: _onMessagesScroll });
    messagesEl.__scrollBound = true;
  }

  // Load messages
  _loadMessages(type, name, false);
}

// ── Channel Members UI ────────────────────

async function _renderChannelMembers(channelName, metaEl) {
  // Lazy-load Discord channel list
  if (_discordChannels === null) {
    try {
      const dc = await api("/api/discord/channels");
      _discordChannels = dc.channels || [];
    } catch {
      _discordChannels = [];
    }
  }
  if (_discordChannels.length === 0) return;

  // Find Discord channel by board mapping
  const ch = _discordChannels.find(c => c.board === channelName || c.name === channelName);
  if (!ch) return;

  const members = ch.members || [];
  const allAnimas = [...new Set(_discordChannels.flatMap(c => c.members || []))].sort();

  function _render() {
    const tags = members.map((m, i) => {
      const isLead = i === 0;
      const bg = isLead ? "var(--color-warning-light,#fff3cd)" : "var(--color-primary-light,#e8f0fe)";
      const fg = isLead ? "var(--color-warning-dark,#856404)" : "var(--color-primary,#0066cc)";
      const star = isLead ? "★ " : "";
      const leadBtn = !isLead ? `<button class="board-member-lead" data-name="${escapeHtml(m)}" title="${t("board.members_set_lead")}" style="background:none;border:none;cursor:pointer;color:var(--text-secondary,#aaa);font-size:0.7rem;padding:0;line-height:1;">☆</button>` : "";
      return `<span style="display:inline-flex;align-items:center;gap:0.2rem;background:${bg};color:${fg};border-radius:10px;padding:0.1rem 0.5rem;font-size:0.75rem;">` +
        `${star}${escapeHtml(m)}${leadBtn}<button class="board-member-rm" data-name="${escapeHtml(m)}" style="background:none;border:none;cursor:pointer;color:var(--text-secondary,#666);font-size:0.75rem;padding:0;line-height:1;">✕</button></span>`;
    }).join(" ");

    // Animas not in this channel
    const available = allAnimas.filter(a => !members.includes(a));
    const addDropdown = available.length > 0
      ? `<select id="boardMemberAdd" style="font-size:0.75rem;padding:0.1rem 0.3rem;border:1px solid var(--border,#ddd);border-radius:4px;background:var(--bg-secondary,#f9f9f9);color:var(--text-primary,#333);"><option value="">+ ${t("board.members_add")}</option>${available.map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join("")}</select>`
      : "";

    metaEl.innerHTML = `<span style="font-size:0.75rem;color:var(--text-secondary,#888);margin-right:0.3rem;">${t("board.members")}:</span>${tags || `<span style="font-size:0.75rem;color:var(--text-secondary,#888);">${t("board.members_none")}</span>`} ${addDropdown}`;

    // Bind remove buttons
    metaEl.querySelectorAll(".board-member-rm").forEach(btn => {
      btn.addEventListener("click", async () => {
        const nm = btn.dataset.name;
        const idx = members.indexOf(nm);
        if (idx !== -1) members.splice(idx, 1);
        ch.members = [...members];
        await _saveMembership(ch.id, members);
        _render();
      });
    });

    // Bind lead buttons (☆ -> promote to first position)
    metaEl.querySelectorAll(".board-member-lead").forEach(btn => {
      btn.addEventListener("click", async () => {
        const nm = btn.dataset.name;
        const idx = members.indexOf(nm);
        if (idx > 0) {
          members.splice(idx, 1);
          members.unshift(nm);
          ch.members = [...members];
          await _saveMembership(ch.id, members);
          _render();
        }
      });
    });

    // Bind add dropdown
    const addSel = document.getElementById("boardMemberAdd");
    if (addSel) {
      addSel.addEventListener("change", async () => {
        const val = addSel.value;
        if (!val || members.includes(val)) { addSel.value = ""; return; }
        members.push(val);
        ch.members = [...members];
        await _saveMembership(ch.id, members);
        _render();
      });
    }
  }

  _render();
}

async function _saveMembership(channelId, members) {
  try {
    await fetch(`/api/discord/channel-members/${encodeURIComponent(channelId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ members }),
    });
  } catch (e) {
    logger.error("Failed to save channel membership", e);
  }
}

// ── Message Loading ────────────────────────

async function _loadMessages(type, name, isPolling) {
  // Skip if selection changed during load
  if (type !== _selectedType || name !== _selectedName) return;

  const messagesEl = _$("boardMessages");
  if (!messagesEl) return;

  if (!isPolling) {
    _currentOffset = 0;
    _hasMore = false;
    _lastMessageTs = "";
    messagesEl.innerHTML = `<div class="board-messages-empty">${t("board.loading")}</div>`;
  }

  try {
    let data;
    const fetchOffset = isPolling ? 0 : _currentOffset;
    if (type === "channel") {
      data = await api(`/api/channels/${encodeURIComponent(name)}?limit=50&offset=${fetchOffset}`);
    } else {
      data = await api(`/api/dm/${encodeURIComponent(name)}?limit=50`);
    }

    // Verify selection hasn't changed
    if (type !== _selectedType || name !== _selectedName) return;

    _messages = data.messages || [];
    _total = data.total || _messages.length;
    _hasMore = data.has_more || false;
    _updateLastTs();

    // Preload avatars for senders, then render
    const senders = [...new Set(_messages.map(m => m.from || "unknown"))];
    await _preloadAvatars(senders);
    if (type !== _selectedType || name !== _selectedName) return;
    _renderMessages();
    if (!isPolling) _scrollToBottom();
  } catch (err) {
    if (type !== _selectedType || name !== _selectedName) return;
    if (!isPolling) {
      messagesEl.innerHTML = `<div class="board-messages-empty">${t("board.load_failed")}: ${escapeHtml(err.message)}</div>`;
    }
    logger.error("Failed to load messages", { type, name, error: err.message });
  }
}

async function _pollNewMessages() {
  if (!_selectedType || !_selectedName) return;

  // DMs don't support incremental polling yet
  if (_selectedType !== "channel" || !_lastMessageTs) {
    _loadMessages(_selectedType, _selectedName, true);
    return;
  }

  try {
    const data = await api(
      `/api/channels/${encodeURIComponent(_selectedName)}?since=${encodeURIComponent(_lastMessageTs)}`
    );
    if (_selectedType !== "channel" || _selectedName !== data.channel) return;

    const newMsgs = data.messages || [];
    if (newMsgs.length === 0) return;

    // Update total from server response
    _total = data.total || _total;

    const senders = [...new Set(newMsgs.map(m => m.from || "unknown"))];
    await _preloadAvatars(senders);
    if (_selectedType !== "channel" || _selectedName !== data.channel) return;

    const wasAtBottom = _isScrolledToBottom(_$("boardMessages"));
    _messages.push(...newMsgs);
    _updateLastTs();
    _renderMessages();
    if (wasAtBottom) _scrollToBottom();
  } catch (err) {
    logger.error("Failed to poll new messages", { error: err.message });
  }
}

function _updateLastTs() {
  if (_messages.length > 0) {
    const last = _messages[_messages.length - 1];
    if (last.ts && last.ts > _lastMessageTs) {
      _lastMessageTs = last.ts;
    }
  }
}

async function _loadOlderMessages() {
  if (_loadingMore || !_hasMore || _selectedType !== "channel") return;
  _loadingMore = true;

  const messagesEl = _$("boardMessages");
  if (!messagesEl) { _loadingMore = false; return; }

  const prevScrollHeight = messagesEl.scrollHeight;
  _currentOffset = _messages.length;

  try {
    const data = await api(
      `/api/channels/${encodeURIComponent(_selectedName)}?limit=50&offset=${_currentOffset}`
    );

    if (_selectedType !== "channel" || _selectedName !== data.channel) {
      _loadingMore = false;
      return;
    }

    const older = data.messages || [];
    _messages = [...older, ..._messages];
    _hasMore = data.has_more || false;
    _renderMessages();

    const newScrollHeight = messagesEl.scrollHeight;
    messagesEl.scrollTop = newScrollHeight - prevScrollHeight;
  } catch (err) {
    logger.error("Failed to load older messages", { error: err.message });
  } finally {
    _loadingMore = false;
  }
}

function _onMessagesScroll(e) {
  if (e.target.scrollTop < 100 && _hasMore && !_loadingMore) {
    _loadOlderMessages();
  }
}

// ── Message Rendering ──────────────────────

function _renderMessages() {
  const messagesEl = _$("boardMessages");
  if (!messagesEl) return;

  if (_messages.length === 0) {
    messagesEl.innerHTML = `<div class="board-messages-empty">${t("board.messages_empty")}</div>`;
    return;
  }

  const wasAtBottom = _isScrolledToBottom(messagesEl);

  const loaderHtml = _hasMore
    ? `<div class="board-messages-loader" style="text-align:center;padding:12px;color:#888;font-size:0.82rem;">${t("board.loading")}</div>`
    : (_messages.length >= _total && _total > 50
      ? `<div class="board-messages-end" style="text-align:center;padding:8px;color:#666;font-size:0.8rem;">${t("board.messages_end")}</div>`
      : '');

  const msgItems = _messages.map(msg => {
    const from = msg.from || "unknown";
    const isHuman = msg.source === "human";
    const ts = msg.ts ? smartTimestamp(msg.ts) : "";
    const badge = isHuman ? '<span class="board-msg-badge">human</span>' : "";
    const text = msg.text || "";

    return `
      <div class="board-msg">
        ${_avatarHtml(from, isHuman)}
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

  messagesEl.innerHTML = loaderHtml + msgItems;

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
      errorDiv.textContent = `${t("board.send_failed")}: ${err.message}`;
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
    _total += 1;
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
