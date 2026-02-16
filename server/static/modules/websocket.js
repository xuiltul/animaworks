/* ── WebSocket ─────────────────────────────── */

import { state, dom } from "./state.js";
import { addActivity } from "./activity.js";
import { renderPersonDropdown, updatePersonAvatar, refreshSelectedPerson } from "./persons.js";
import { updateSystemStatus } from "./status.js";
import { renderChat } from "./chat.js";

let ws = null;
let wsReconnectTimer = null;
const WS_INITIAL_DELAY = 1000;
const WS_MAX_DELAY = 30000;
const WS_BACKOFF_MULTIPLIER = 2;
let wsReconnectAttempt = 0;

export function connectWebSocket() {
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
    wsReconnectAttempt = 0;
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
  const delay = Math.min(
    WS_INITIAL_DELAY * Math.pow(WS_BACKOFF_MULTIPLIER, wsReconnectAttempt),
    WS_MAX_DELAY
  );
  const jitter = Math.random() * 1000;
  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    wsReconnectAttempt++;
    connectWebSocket();
  }, delay + jitter);
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
          const isStreaming = history.some((m) => m.streaming);
          if (isStreaming) break;
          const thinkIdx = history.findIndex((m) => m.role === "thinking");
          if (thinkIdx !== -1) history.splice(thinkIdx, 1);
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

    case "person.bootstrap": {
      const personName = data.name;
      const bsStatus = data.status;
      if (personName) {
        if (bsStatus === "started") {
          const existing = state.persons.find((p) => p.name === personName);
          if (existing) {
            existing.status = "bootstrapping";
            existing.bootstrapping = true;
          }
          renderPersonDropdown();
          addActivity("system", personName, "ブートストラップ開始");
        } else if (bsStatus === "completed") {
          const existing = state.persons.find((p) => p.name === personName);
          if (existing) {
            existing.status = "idle";
            existing.bootstrapping = false;
          }
          renderPersonDropdown();
          addActivity("system", personName, "ブートストラップ完了");
          // Pop-in animation for activated person
          const el = document.querySelector(`[data-person="${personName}"]`);
          if (el) {
            el.classList.add("person-item--just-activated");
            el.addEventListener("animationend", () => {
              el.classList.remove("person-item--just-activated");
            }, { once: true });
          }
          if (personName === state.selectedPerson) {
            refreshSelectedPerson();
          }
        } else if (bsStatus === "failed") {
          const existing = state.persons.find((p) => p.name === personName);
          if (existing) {
            existing.status = "error";
            existing.bootstrapping = false;
          }
          renderPersonDropdown();
          addActivity("system", personName, "ブートストラップ失敗");
        }
      }
      break;
    }

    case "person.assets_updated": {
      const personName = data.name;
      if (personName) {
        addActivity("system", personName, `アセット更新: ${(data.assets || []).join(", ")}`);
        if (personName === state.selectedPerson) {
          updatePersonAvatar();
        }
      }
      break;
    }

    case "person.remake_preview_ready":
    case "person.remake_progress":
    case "person.remake_complete": {
      // Delegate to assets page handler if registered
      if (typeof window.__assetsWsHandler === "function") {
        window.__assetsWsHandler(eventType, data);
      }
      if (eventType === "person.remake_complete" && data.name) {
        const steps = (data.steps_completed || []).join(", ");
        addActivity("system", data.name, `アセットリメイク完了: ${steps}`);
      }
      break;
    }

    case "person.notification": {
      const personName = data.person || data.name;
      const subject = data.subject || "";
      const body = data.body || "";
      const priority = data.priority || "normal";
      if (personName) {
        // Add to chat history as assistant notification message
        if (!state.chatHistories[personName]) {
          state.chatHistories[personName] = [];
        }
        const notifText = subject ? `**${subject}**\n${body}` : body;
        state.chatHistories[personName].push({
          role: "assistant",
          text: notifText,
          notification: true,
          priority: priority,
        });
        if (personName === state.selectedPerson) {
          renderChat();
        }
        // Show toast notification
        showNotificationToast(personName, subject, body, priority);
        // Add to activity log
        addActivity("notification", personName, subject || body.slice(0, 100));
      }
      break;
    }

    case "person.interaction": {
      const fromPerson = data.from_person || "";
      const toPerson = data.to_person || "";
      if (fromPerson || toPerson) {
        const label = `${fromPerson} → ${toPerson}`;
        addActivity("message", label, data.summary || "メッセージ送信");
      }
      break;
    }

    case "ping":
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "pong" }));
      }
      return;

    default:
      if (data.name || data.person) {
        addActivity("system", data.name || data.person, JSON.stringify(data).slice(0, 120));
      }
      break;
  }
}

// ── Toast Notifications ─────────────────────

function showNotificationToast(personName, subject, body, priority) {
  // Create toast container if it doesn't exist
  let container = document.getElementById("notificationToasts");
  if (!container) {
    container = document.createElement("div");
    container.id = "notificationToasts";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `notification-toast priority-${priority}`;

  const header = document.createElement("div");
  header.className = "notification-toast-header";
  header.textContent = personName;

  const subjectEl = document.createElement("div");
  subjectEl.className = "notification-toast-subject";
  subjectEl.textContent = subject;

  const bodyEl = document.createElement("div");
  bodyEl.className = "notification-toast-body";
  bodyEl.textContent = body.slice(0, 200);

  toast.appendChild(header);
  if (subject) toast.appendChild(subjectEl);
  toast.appendChild(bodyEl);
  container.appendChild(toast);

  // Auto-dismiss after 5 seconds (8 seconds for urgent)
  const dismissDelay = priority === "urgent" ? 8000 : 5000;
  setTimeout(() => {
    toast.classList.add("notification-toast-exit");
    toast.addEventListener("animationend", () => toast.remove());
  }, dismissDelay);

  // Click to dismiss
  toast.addEventListener("click", () => {
    toast.classList.add("notification-toast-exit");
    toast.addEventListener("animationend", () => toast.remove());
  });
}

// ── Visibility Change Reconnect ─────────────────────

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      wsReconnectAttempt = 0;
      scheduleReconnect();
    }
  }
});
