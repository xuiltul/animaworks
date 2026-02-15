/* ── WebSocket ─────────────────────────────── */

import { state, dom } from "./state.js";
import { addActivity } from "./activity.js";
import { renderPersonDropdown, updatePersonAvatar, refreshSelectedPerson } from "./persons.js";
import { updateSystemStatus } from "./status.js";

let ws = null;
let wsReconnectTimer = null;
const WS_RECONNECT_DELAY = 3000;

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

    default:
      if (data.name || data.person) {
        addActivity("system", data.name || data.person, JSON.stringify(data).slice(0, 120));
      }
      break;
  }
}
