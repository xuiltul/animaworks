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
    activeRightTab: "state",
    wsConnected: false,
  };

  // ── DOM refs ───────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $id = (id) => document.getElementById(id);

  const dom = {
    systemStatus: $id("systemStatus"),
    systemStatusText: $id("systemStatusText"),
    personDropdown: $id("personDropdown"),
    chatMessages: $id("chatMessages"),
    chatForm: $id("chatForm"),
    chatInput: $id("chatInput"),
    chatSendBtn: $id("chatSendBtn"),
    activityFeed: $id("activityFeed"),
    personStateContent: $id("personStateContent"),
    rightTabContent: $id("rightTabContent"),
    tabState: $id("tabState"),
    tabActivity: $id("tabActivity"),
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

  // ── Person Dropdown ────────────────────────
  async function loadPersons() {
    try {
      state.persons = await api("/api/persons");
      renderPersonDropdown();
      if (state.persons.length > 0 && !state.selectedPerson) {
        selectPerson(state.persons[0].name);
      }
    } catch (err) {
      console.error("Failed to load persons:", err);
    }
  }

  function renderPersonDropdown() {
    const dropdown = dom.personDropdown;
    const currentValue = dropdown.value;

    // Build options
    let html = '<option value="" disabled>パーソンを選択...</option>';
    for (const p of state.persons) {
      const statusLabel = p.status ? ` (${p.status})` : "";
      const selected = p.name === state.selectedPerson ? " selected" : "";
      html += `<option value="${escapeHtml(p.name)}"${selected}>${escapeHtml(p.name)}${statusLabel}</option>`;
    }
    dropdown.innerHTML = html;
  }

  // ── Person Selection ───────────────────────
  async function selectPerson(name) {
    state.selectedPerson = name;

    // Update dropdown
    dom.personDropdown.value = name;

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
      dom.personStateContent.textContent = "詳細の読み込み失敗";
      dom.memoryFileList.innerHTML = '<div class="loading-placeholder">詳細の読み込み失敗</div>';
    }
  }

  // ── Person State ───────────────────────────
  function renderPersonState() {
    const d = state.personDetail;
    if (!d || !d.state) {
      dom.personStateContent.textContent = "状態情報なし";
      return;
    }
    const stateText = typeof d.state === "string" ? d.state : JSON.stringify(d.state, null, 2);
    dom.personStateContent.textContent = stateText;
  }

  // ── Right Panel Tab Switching ──────────────
  function activateRightTab(tab) {
    state.activeRightTab = tab;
    document.querySelectorAll(".right-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tab);
    });
    document.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.dataset.tab === tab);
    });
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
        const streamClass = m.streaming ? " streaming" : "";
        let content = "";
        if (m.text) {
          content = renderMarkdown(m.text);
        } else if (m.streaming) {
          content = '<span class="cursor-blink"></span>';
        }
        const toolHtml = m.activeTool
          ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(m.activeTool)} を実行中...</div>`
          : "";
        return `<div class="chat-bubble assistant${streamClass}">${content}${toolHtml}</div>`;
      }
      return `<div class="chat-bubble user">${escapeHtml(m.text)}</div>`;
    }).join("");
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  }

  // ── SSE Streaming ─────────────────────────

  function parseSSEEvents(buffer) {
    const parsed = [];
    // Split by double newline (SSE event boundary)
    const parts = buffer.split("\n\n");
    // Last part may be incomplete
    const remaining = parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;
      let eventName = "message";
      let dataLines = [];
      for (const line of part.split("\n")) {
        if (line.startsWith("event: ")) {
          eventName = line.slice(7);
        } else if (line.startsWith("data: ")) {
          dataLines.push(line.slice(6));
        }
      }
      if (dataLines.length > 0) {
        try {
          parsed.push({ event: eventName, data: JSON.parse(dataLines.join("\n")) });
        } catch (e) {
          console.warn("SSE parse error:", e, dataLines);
        }
      }
    }
    return { parsed, remaining };
  }

  function renderStreamingBubble(msg) {
    const bubble = dom.chatMessages.querySelector(".chat-bubble.assistant.streaming");
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

    if (msg.activeTool) {
      html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`;
    }

    bubble.innerHTML = html;
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  }

  async function sendChat(message) {
    const name = state.selectedPerson;
    if (!name || !message.trim()) return;

    if (!state.chatHistories[name]) state.chatHistories[name] = [];
    const history = state.chatHistories[name];

    // Add user message + empty streaming bubble
    history.push({ role: "user", text: message });
    const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null };
    history.push(streamingMsg);
    renderChat();

    dom.chatInput.disabled = true;
    dom.chatSendBtn.disabled = true;
    addActivity("chat", name, `ユーザー: ${message}`);

    try {
      const response = await fetch(
        `/api/persons/${encodeURIComponent(name)}/chat/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
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
              streamingMsg.activeTool = null;
              renderStreamingBubble(streamingMsg);
              break;

            case "chain_start":
              // Session continuation — stream continues seamlessly
              break;

            case "error":
              streamingMsg.text += `\n[エラー] ${evt.data.message}`;
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
          renderPersonDropdown();
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
          if (state.chatHistories[personName]) {
            const history = state.chatHistories[personName];
            // Skip if we're currently streaming for this person (SSE handles display)
            const isStreaming = history.some((m) => m.streaming);
            if (isStreaming) break;
            // Remove any lingering thinking bubble
            const thinkIdx = history.findIndex((m) => m.role === "thinking");
            if (thinkIdx !== -1) history.splice(thinkIdx, 1);
            // Only add if not already the last message (avoid duplicates)
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

  // ── Textarea Auto-Resize ────────────────────
  function autoResizeTextarea() {
    const el = dom.chatInput;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }

  function submitChat() {
    const msg = dom.chatInput.value.trim();
    if (!msg) return;
    dom.chatInput.value = "";
    dom.chatInput.style.height = "auto";
    sendChat(msg);
  }

  // ── Event Bindings ─────────────────────────
  function bindEvents() {
    // Person dropdown change
    dom.personDropdown.addEventListener("change", (e) => {
      const name = e.target.value;
      if (name) selectPerson(name);
    });

    // Chat form submit (button click or programmatic)
    dom.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      submitChat();
    });

    // Textarea: Ctrl+Enter to send, Enter for newline, auto-resize
    dom.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        submitChat();
      }
    });

    dom.chatInput.addEventListener("input", autoResizeTextarea);

    // Right panel tabs (State / Activity)
    document.querySelectorAll(".right-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        activateRightTab(btn.dataset.tab);
      });
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
        renderPersonDropdown();
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
