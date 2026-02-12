/* ============================================
   Digital Person — Dashboard App
   ============================================ */

(function () {
  "use strict";

  // ── State ──────────────────────────────────
  const state = {
    persons: [],            // PersonStatus[]
    selectedPerson: null,   // string (name)
    personDetail: null,     // full detail object
    chatHistories: {},      // { [name]: [{role, text}] }
    activeMemoryTab: "episodes",
    wsConnected: false,
  };

  // ── DOM refs ───────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $id = (id) => document.getElementById(id);

  const dom = {
    systemStatus: $id("systemStatus"),
    systemStatusText: $id("systemStatusText"),
    personList: $id("personList"),
    chatMessages: $id("chatMessages"),
    chatForm: $id("chatForm"),
    chatInput: $id("chatInput"),
    chatSendBtn: $id("chatSendBtn"),
    activityFeed: $id("activityFeed"),
    personStateSection: $id("personStateSection"),
    personStateContent: $id("personStateContent"),
    memoryFileList: $id("memoryFileList"),
    memoryContentArea: $id("memoryContentArea"),
    memoryContentTitle: $id("memoryContentTitle"),
    memoryContentBody: $id("memoryContentBody"),
    memoryBackBtn: $id("memoryBackBtn"),
  };

  // ── Helpers ────────────────────────────────
  function timeStr(isoOrTs) {
    if (!isoOrTs) return "--:--";
    const d = new Date(isoOrTs);
    if (isNaN(d.getTime())) return "--:--";
    return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  }

  function nowTimeStr() {
    return new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function statusClass(status) {
    if (!status) return "status-offline";
    const s = status.toLowerCase();
    if (s === "idle" || s === "running") return "status-idle";
    if (s === "thinking" || s === "processing" || s === "busy") return "status-thinking";
    if (s === "error") return "status-error";
    return "status-offline";
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
    return res.json();
  }

  // ── Person List ────────────────────────────
  async function loadPersons() {
    try {
      state.persons = await api("/api/persons");
      renderPersonList();
      if (state.persons.length > 0 && !state.selectedPerson) {
        selectPerson(state.persons[0].name);
      }
    } catch (err) {
      console.error("Failed to load persons:", err);
      dom.personList.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
    }
  }

  function renderPersonList() {
    if (state.persons.length === 0) {
      dom.personList.innerHTML = '<div class="loading-placeholder">パーソンが見つかりません</div>';
      return;
    }
    dom.personList.innerHTML = state.persons.map((p) => {
      const selected = p.name === state.selectedPerson ? " selected" : "";
      const sc = statusClass(p.status);
      const task = p.current_task ? escapeHtml(p.current_task) : "待機中";
      const badge = (p.pending_messages && p.pending_messages > 0)
        ? `<span class="person-card-badge" data-count="${p.pending_messages}">${p.pending_messages}</span>`
        : "";
      return `
        <div class="person-card${selected}" data-name="${escapeHtml(p.name)}">
          <span class="status-dot ${sc}"></span>
          <div class="person-card-info">
            <div class="person-card-name">${escapeHtml(p.name)}</div>
            <div class="person-card-task">${task}</div>
          </div>
          ${badge}
        </div>`;
    }).join("");

    // Click handlers
    dom.personList.querySelectorAll(".person-card").forEach((card) => {
      card.addEventListener("click", () => {
        selectPerson(card.dataset.name);
      });
    });
  }

  // ── Person Selection ───────────────────────
  async function selectPerson(name) {
    state.selectedPerson = name;

    // Highlight in list
    dom.personList.querySelectorAll(".person-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.name === name);
    });

    // Enable chat
    dom.chatInput.disabled = false;
    dom.chatSendBtn.disabled = false;
    dom.chatInput.placeholder = `${name} にメッセージ...`;

    // Render chat history
    renderChat();

    // Load detail
    try {
      state.personDetail = await api(`/api/persons/${encodeURIComponent(name)}`);
      renderPersonState();
      loadMemoryTab(state.activeMemoryTab);
    } catch (err) {
      console.error("Failed to load person detail:", err);
      state.personDetail = null;
      dom.personStateSection.style.display = "none";
      dom.memoryFileList.innerHTML = '<div class="loading-placeholder">詳細の読み込み失敗</div>';
    }
  }

  // ── Person State ───────────────────────────
  function renderPersonState() {
    const d = state.personDetail;
    if (!d || !d.state) {
      dom.personStateSection.style.display = "none";
      return;
    }
    dom.personStateSection.style.display = "";
    const stateText = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
    dom.personStateContent.textContent = stateText;
  }

  // ── Chat ───────────────────────────────────
  function renderMarkdown(text) {
    try {
      return marked.parse(text, { breaks: true });
    } catch {
      return escapeHtml(text);
    }
  }

  function renderChat() {
    const name = state.selectedPerson;
    const history = state.chatHistories[name] || [];
    if (history.length === 0) {
      dom.chatMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
      return;
    }
    dom.chatMessages.innerHTML = history.map((m) => {
      if (m.role === "thinking") {
        return `<div class="chat-bubble thinking"><span class="thinking-animation">考え中</span></div>`;
      }
      if (m.role === "assistant") {
        return `<div class="chat-bubble assistant">${renderMarkdown(m.text)}</div>`;
      }
      return `<div class="chat-bubble user">${escapeHtml(m.text)}</div>`;
    }).join("");
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  }

  async function sendChat(message) {
    const name = state.selectedPerson;
    if (!name || !message.trim()) return;

    // Init history
    if (!state.chatHistories[name]) state.chatHistories[name] = [];
    const history = state.chatHistories[name];

    // Add user message
    history.push({ role: "user", text: message });
    // Add thinking placeholder
    history.push({ role: "thinking", text: "" });
    renderChat();

    // Disable input while processing
    dom.chatInput.disabled = true;
    dom.chatSendBtn.disabled = true;

    // Add activity
    addActivity("chat", name, `ユーザー: ${message}`);

    try {
      const data = await api(`/api/persons/${encodeURIComponent(name)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      // Remove thinking, add response (skip if WebSocket already delivered it)
      const thinkIdx = history.findIndex((m) => m.role === "thinking");
      if (thinkIdx !== -1) history.splice(thinkIdx, 1);
      const responseText = data.response || "(空の応答)";
      const last = history[history.length - 1];
      if (!last || last.role !== "assistant" || last.text !== responseText) {
        history.push({ role: "assistant", text: responseText });
        addActivity("chat", name, `応答: ${(data.response || "").slice(0, 100)}`);
      }
      renderChat();
    } catch (err) {
      // Remove thinking, add error
      const thinkIdx = history.findIndex((m) => m.role === "thinking");
      if (thinkIdx !== -1) history.splice(thinkIdx, 1);
      history.push({ role: "assistant", text: `[エラー] ${err.message}` });
      renderChat();
    } finally {
      dom.chatInput.disabled = false;
      dom.chatSendBtn.disabled = false;
      dom.chatInput.focus();
    }
  }

  // ── Activity Feed ──────────────────────────
  const TYPE_ICONS = {
    heartbeat: "\uD83D\uDC93",   // heart
    cron: "\u23F0",               // alarm clock
    chat: "\uD83D\uDCAC",        // speech bubble
    system: "\u2699\uFE0F",      // gear
  };

  let activityEmpty = true;

  function addActivity(type, personName, summary) {
    if (activityEmpty) {
      dom.activityFeed.innerHTML = "";
      activityEmpty = false;
    }

    const icon = TYPE_ICONS[type] || TYPE_ICONS.system;
    const entry = document.createElement("div");
    entry.className = "activity-entry";
    entry.innerHTML = `
      <span class="activity-icon">${icon}</span>
      <span class="activity-time">${nowTimeStr()}</span>
      <div class="activity-body">
        <span class="activity-person">${escapeHtml(personName)}</span>
        <span class="activity-summary"> ${escapeHtml(summary)}</span>
      </div>`;
    dom.activityFeed.appendChild(entry);
    dom.activityFeed.scrollTop = dom.activityFeed.scrollHeight;

    // Cap at 200 entries
    while (dom.activityFeed.children.length > 200) {
      dom.activityFeed.removeChild(dom.activityFeed.firstChild);
    }
  }

  // ── Memory Browser ─────────────────────────
  function activateMemoryTab(tab) {
    state.activeMemoryTab = tab;
    document.querySelectorAll(".memory-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    });
    // Hide content detail, show list
    dom.memoryContentArea.style.display = "none";
    dom.memoryFileList.style.display = "";
    loadMemoryTab(tab);
  }

  async function loadMemoryTab(tab) {
    const name = state.selectedPerson;
    if (!name) {
      dom.memoryFileList.innerHTML = '<div class="loading-placeholder">パーソンを選択してください</div>';
      return;
    }

    dom.memoryFileList.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

    let endpoint;
    if (tab === "episodes") endpoint = `/api/persons/${encodeURIComponent(name)}/episodes`;
    else if (tab === "knowledge") endpoint = `/api/persons/${encodeURIComponent(name)}/knowledge`;
    else endpoint = `/api/persons/${encodeURIComponent(name)}/procedures`;

    try {
      const data = await api(endpoint);
      const files = data.files || [];
      if (files.length === 0) {
        dom.memoryFileList.innerHTML = '<div class="loading-placeholder">ファイルがありません</div>';
        return;
      }
      dom.memoryFileList.innerHTML = files.map((f) =>
        `<div class="memory-file-item" data-file="${escapeHtml(f)}" data-tab="${tab}">${escapeHtml(f)}</div>`
      ).join("");

      dom.memoryFileList.querySelectorAll(".memory-file-item").forEach((item) => {
        item.addEventListener("click", () => {
          loadMemoryContent(item.dataset.tab, item.dataset.file);
        });
      });
    } catch (err) {
      console.error("Failed to load memory files:", err);
      dom.memoryFileList.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
    }
  }

  async function loadMemoryContent(tab, file) {
    const name = state.selectedPerson;
    if (!name) return;

    let endpoint;
    if (tab === "episodes") endpoint = `/api/persons/${encodeURIComponent(name)}/episodes/${encodeURIComponent(file)}`;
    else if (tab === "knowledge") endpoint = `/api/persons/${encodeURIComponent(name)}/knowledge/${encodeURIComponent(file)}`;
    else endpoint = `/api/persons/${encodeURIComponent(name)}/procedures/${encodeURIComponent(file)}`;

    dom.memoryFileList.style.display = "none";
    dom.memoryContentArea.style.display = "";
    dom.memoryContentTitle.textContent = file;
    dom.memoryContentBody.textContent = "読み込み中...";

    try {
      const data = await api(endpoint);
      dom.memoryContentBody.textContent = data.content || "(内容なし)";
    } catch (err) {
      dom.memoryContentBody.textContent = `[エラー] ${err.message}`;
    }
  }

  // ── WebSocket ──────────────────────────────
  let ws = null;
  let wsReconnectTimer = null;
  const WS_RECONNECT_DELAY = 3000;

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws`;

    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.error("WebSocket creation failed:", err);
      scheduleReconnect();
      return;
    }

    ws.addEventListener("open", () => {
      state.wsConnected = true;
      updateSystemStatus();
      console.log("WebSocket connected");
    });

    ws.addEventListener("close", () => {
      state.wsConnected = false;
      updateSystemStatus();
      console.log("WebSocket closed, reconnecting...");
      scheduleReconnect();
    });

    ws.addEventListener("error", (err) => {
      console.error("WebSocket error:", err);
    });

    ws.addEventListener("message", (evt) => {
      handleWsMessage(evt.data);
    });
  }

  function scheduleReconnect() {
    if (wsReconnectTimer) return;
    wsReconnectTimer = setTimeout(() => {
      wsReconnectTimer = null;
      connectWebSocket();
    }, WS_RECONNECT_DELAY);
  }

  function handleWsMessage(raw) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      console.warn("Non-JSON WS message:", raw);
      return;
    }

    const eventType = msg.type || msg.event;
    const data = msg.data || msg;

    switch (eventType) {
      case "person.status": {
        const personName = data.name || data.person;
        const statusVal = data.status;
        if (personName) {
          // Update person in list
          const existing = state.persons.find((p) => p.name === personName);
          if (existing) {
            existing.status = statusVal;
            if (data.current_task !== undefined) existing.current_task = data.current_task;
          }
          renderPersonList();
          addActivity("system", personName, `ステータス: ${statusVal || "不明"}`);
        }
        break;
      }

      case "person.heartbeat": {
        const personName = data.name || data.person;
        if (personName) {
          addActivity("heartbeat", personName, data.summary || "ハートビート実行");
          // Refresh person detail if selected
          if (personName === state.selectedPerson) {
            refreshSelectedPerson();
          }
        }
        break;
      }

      case "person.cron": {
        const personName = data.name || data.person;
        if (personName) {
          addActivity("cron", personName, data.summary || data.job || "スケジュール実行");
        }
        break;
      }

      case "chat.response": {
        const personName = data.person || data.name;
        const response = data.response || data.message;
        if (personName && response) {
          // Update chat history if this person is selected
          if (state.chatHistories[personName]) {
            const history = state.chatHistories[personName];
            // Remove any lingering thinking bubble
            const thinkIdx = history.findIndex((m) => m.role === "thinking");
            if (thinkIdx !== -1) history.splice(thinkIdx, 1);
            // Only add if not already the last message (avoid duplicates from REST response)
            const last = history[history.length - 1];
            if (!last || last.text !== response) {
              history.push({ role: "assistant", text: response });
            }
            if (personName === state.selectedPerson) renderChat();
          }
          addActivity("chat", personName, `応答: ${response.slice(0, 100)}`);
        }
        break;
      }

      default:
        // Unknown event types: show in activity as system event
        if (data.name || data.person) {
          addActivity("system", data.name || data.person, JSON.stringify(data).slice(0, 120));
        }
        break;
    }
  }

  // ── System Status ──────────────────────────
  async function loadSystemStatus() {
    try {
      const data = await api("/api/system/status");
      const personCount = data.persons || 0;
      const schedulerRunning = data.scheduler_running;
      const dot = dom.systemStatus.querySelector(".status-dot");
      if (schedulerRunning) {
        dot.className = "status-dot status-idle";
        dom.systemStatusText.textContent = `稼働中 (${personCount}名)`;
      } else {
        dot.className = "status-dot status-error";
        dom.systemStatusText.textContent = "スケジューラ停止";
      }
    } catch {
      updateSystemStatus();
    }
  }

  function updateSystemStatus() {
    const dot = dom.systemStatus.querySelector(".status-dot");
    if (state.wsConnected) {
      dot.className = "status-dot status-idle";
      dom.systemStatusText.textContent = `接続済 (${state.persons.length}名)`;
    } else {
      dot.className = "status-dot status-offline";
      dom.systemStatusText.textContent = "再接続中...";
    }
  }

  // Periodically refresh person list to keep statuses current
  async function refreshSelectedPerson() {
    if (!state.selectedPerson) return;
    try {
      state.personDetail = await api(`/api/persons/${encodeURIComponent(state.selectedPerson)}`);
      renderPersonState();
    } catch {
      // Silently ignore refresh errors
    }
  }

  // ── Event Bindings ─────────────────────────
  function bindEvents() {
    // Chat form submit
    dom.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const msg = dom.chatInput.value.trim();
      if (!msg) return;
      dom.chatInput.value = "";
      sendChat(msg);
    });

    // Memory tabs
    document.querySelectorAll(".memory-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        activateMemoryTab(btn.dataset.tab);
      });
    });

    // Memory back button
    dom.memoryBackBtn.addEventListener("click", () => {
      dom.memoryContentArea.style.display = "none";
      dom.memoryFileList.style.display = "";
    });
  }

  // ── Init ───────────────────────────────────
  async function init() {
    bindEvents();
    await loadPersons();
    loadSystemStatus();
    connectWebSocket();

    // Periodic refresh: person list every 30s, system status every 60s
    setInterval(async () => {
      try {
        state.persons = await api("/api/persons");
        renderPersonList();
      } catch { /* ignore */ }
    }, 30000);

    setInterval(loadSystemStatus, 60000);
  }

  // Start when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
