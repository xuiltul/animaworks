/* ── WebSocket ─────────────────────────────── */

import { t } from "/shared/i18n.js";
import { state, dom } from "./state.js";
import { addActivity } from "./activity.js";
import { renderAnimaDropdown, updateAnimaAvatar, refreshSelectedAnima } from "./animas.js";
import { invalidateAvatarCache } from "./avatar-resolver.js";
import { updateSystemStatus } from "./status.js";
import { ChatSessionManager } from "../shared/chat/session-manager.js";
import { createLogger } from "../shared/logger.js";

const logger = createLogger("websocket");

const _eventHandlers = new Map();

let ws = null;
let wsReconnectTimer = null;
const WS_INITIAL_DELAY = 1000;
const WS_MAX_DELAY = 30000;
const WS_BACKOFF_MULTIPLIER = 2;
let wsReconnectAttempt = 0;
let wsConnectedAt = null;

export function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws`;

  logger.info("Connecting", { url, attempt: wsReconnectAttempt });

  try {
    ws = new WebSocket(url);
  } catch (err) {
    logger.error("WebSocket creation failed", { url, error: err.message });
    scheduleReconnect();
    return;
  }

  ws.addEventListener("open", () => {
    wsConnectedAt = Date.now();
    logger.info("Connected", { url, attempt: wsReconnectAttempt });
    wsReconnectAttempt = 0;
    state.wsConnected = true;
    updateSystemStatus();
  });

  ws.addEventListener("close", (evt) => {
    const duration = wsConnectedAt ? Date.now() - wsConnectedAt : 0;
    wsConnectedAt = null;
    state.wsConnected = false;
    updateSystemStatus();
    logger.warn("Connection closed", {
      code: evt.code,
      reason: evt.reason || "",
      wasClean: evt.wasClean,
      duration_ms: duration,
    });
    scheduleReconnect();
  });

  ws.addEventListener("error", (err) => {
    logger.error("Connection error", { readyState: ws?.readyState });
  });

  ws.addEventListener("message", (evt) => {
    handleWsMessage(evt.data);
  });
}

/**
 * Register handler for a WebSocket event type. Returns unsubscribe function.
 */
export function onEvent(type, fn) {
  if (!_eventHandlers.has(type)) _eventHandlers.set(type, new Set());
  _eventHandlers.get(type).add(fn);
  return () => _eventHandlers.get(type)?.delete(fn);
}

function scheduleReconnect() {
  if (wsReconnectTimer) return;
  const delay = Math.min(
    WS_INITIAL_DELAY * Math.pow(WS_BACKOFF_MULTIPLIER, wsReconnectAttempt),
    WS_MAX_DELAY
  );
  const jitter = Math.random() * 1000;
  logger.info("Scheduling reconnect", { attempt: wsReconnectAttempt + 1, delay_ms: Math.round(delay + jitter) });
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
    case "anima.status": {
      const animaName = data.name || data.anima;
      const statusVal = data.status;
      if (animaName) {
        const existing = state.animas.find((p) => p.name === animaName);
        if (existing) {
          existing.status = statusVal;
          if (data.current_task !== undefined) existing.current_task = data.current_task;
        }
        renderAnimaDropdown();
        addActivity("system", animaName, `${t("websocket.status")} ${statusVal || t("common.unknown")}`);
      }
      break;
    }

    case "anima.heartbeat": {
      const animaName = data.name || data.anima;
      if (animaName) {
        addActivity("heartbeat", animaName, data.summary || t("websocket.heartbeat"));
        if (animaName === state.selectedAnima) {
          refreshSelectedAnima();
        }
      }
      break;
    }

    case "anima.cron": {
      const animaName = data.name || data.anima;
      if (animaName) {
        addActivity("cron", animaName, data.summary || data.task || t("websocket.cron"));
      }
      break;
    }

    case "chat.response": {
      const animaName = data.anima || data.name;
      const response = data.response || data.message;
      if (animaName && response) {
        const mgr = ChatSessionManager.getInstance();
        const threadId = data.thread_id || "default";
        if (!mgr.isStreamingFor(animaName, threadId)) {
          mgr.addMessage(animaName, threadId, {
            role: "assistant", text: response, timestamp: new Date().toISOString(),
          });
        }
        addActivity("chat", animaName, `${t("chat.response_prefix")} ${response.slice(0, 100)}`);
      }
      break;
    }

    case "anima.bootstrap": {
      const animaName = data.name;
      const bsStatus = data.status;
      if (animaName) {
        if (bsStatus === "started") {
          const existing = state.animas.find((p) => p.name === animaName);
          if (existing) {
            existing.status = "bootstrapping";
            existing.bootstrapping = true;
          }
          renderAnimaDropdown();
          addActivity("system", animaName, t("websocket.bootstrap_start"));
        } else if (bsStatus === "completed") {
          const existing = state.animas.find((p) => p.name === animaName);
          if (existing) {
            existing.status = "idle";
            existing.bootstrapping = false;
          }
          renderAnimaDropdown();
          addActivity("system", animaName, t("websocket.bootstrap_done"));
          // Pop-in animation for activated anima
          const el = document.querySelector(`[data-anima="${animaName}"]`);
          if (el) {
            el.classList.add("anima-item--just-activated");
            el.addEventListener("animationend", () => {
              el.classList.remove("anima-item--just-activated");
            }, { once: true });
          }
          if (animaName === state.selectedAnima) {
            refreshSelectedAnima();
          }
        } else if (bsStatus === "failed") {
          const existing = state.animas.find((p) => p.name === animaName);
          if (existing) {
            existing.status = "error";
            existing.bootstrapping = false;
          }
          renderAnimaDropdown();
          addActivity("system", animaName, t("websocket.bootstrap_failed"));
        } else if (bsStatus === "max_retries_exceeded") {
          const existing = state.animas.find((p) => p.name === animaName);
          if (existing) {
            existing.status = "error";
            existing.bootstrapping = false;
          }
          renderAnimaDropdown();
          addActivity("system", animaName, t("websocket.bootstrap_max_retries"));
        }
      }
      break;
    }

    case "anima.assets_updated": {
      const animaName = data.name;
      if (animaName) {
        invalidateAvatarCache(animaName).catch(() => {});
        addActivity("system", animaName, `${t("websocket.assets_updated")} ${(data.assets || []).join(", ")}`);
        if (animaName === state.selectedAnima) {
          updateAnimaAvatar();
        }
      }
      break;
    }

    case "anima.remake_preview_ready":
    case "anima.remake_progress":
    case "anima.remake_complete": {
      // Delegate to assets page handler if registered
      if (typeof window.__assetsWsHandler === "function") {
        window.__assetsWsHandler(eventType, data);
      }
      if (eventType === "anima.remake_complete" && data.name) {
        const steps = (data.steps_completed || []).join(", ");
        addActivity("system", data.name, `${t("websocket.remake_complete")} ${steps}`);
      }
      break;
    }

    case "anima.proactive_message": {
      const pmAnimaName = data.anima || data.name;
      const pmSubject = data.subject || "";
      const pmBody = data.body || "";
      const pmPriority = data.priority || "normal";
      if (pmAnimaName) {
        const mgr = ChatSessionManager.getInstance();
        const pmText = pmSubject ? `**${pmSubject}**\n${pmBody}` : pmBody;
        const pmThreadId = data.thread_id || "default";
        mgr.addMessage(pmAnimaName, pmThreadId, {
          role: "assistant", text: pmText, proactive: true,
          priority: pmPriority, timestamp: new Date().toISOString(),
        });
      }
      break;
    }

    case "anima.notification": {
      const animaName = data.anima || data.name;
      const subject = data.subject || "";
      const body = data.body || "";
      const priority = data.priority || "normal";
      if (animaName) {
        // Show toast notification
        showNotificationToast(animaName, subject, body, priority);
        // Add to activity log
        addActivity("notification", animaName, subject || body.slice(0, 100));
      }
      break;
    }

    case "anima.tool_activity": {
      const animaName = data.name;
      const evtType = data.event || data.type || "";
      const toolName = data.tool_name || data.tool || "tool";
      if (animaName && evtType === "tool_start") {
        addActivity("tool", animaName, t("chat.tool_running", { tool: toolName }));
      } else if (animaName && evtType === "tool_detail") {
        addActivity("tool", animaName, `${toolName}: ${data.detail || ""}`);
      } else if (animaName && (evtType === "tool_end" || evtType === "tool_use")) {
        const suffix = data.is_error ? ` (${t("common.error")})` : "";
        addActivity("tool", animaName, `${t("chat.tool_done", { tool: toolName })}${suffix}`);
      } else if (animaName && evtType) {
        addActivity("tool", animaName, `${data.summary || evtType}`);
      }
      document.dispatchEvent(
        new CustomEvent("anima-tool-activity", { detail: { ...data, event: evtType, tool_name: toolName } })
      );
      break;
    }

    case "anima.interaction": {
      const fromAnima = data.from_person || "";
      const toAnima = data.to_person || "";
      if (fromAnima || toAnima) {
        const label = `${fromAnima} → ${toAnima}`;
        addActivity("message", label, data.summary || t("websocket.message_sent"));
      }
      break;
    }

    case "board.post": {
      // Delegate to board page handler if registered
      if (typeof window.__boardWsHandler === "function") {
        window.__boardWsHandler(data);
      }
      const boardFrom = data.from || "unknown";
      const boardChannel = data.channel || "";
      if (boardChannel) {
        addActivity("message", boardFrom, `#${boardChannel}: ${(data.text || "").slice(0, 80)}`);
      }
      break;
    }

    case "ping":
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "pong" }));
      }
      return;

    default:
      if (data.name || data.anima) {
        const summary = data.summary || data.message || eventType || t("websocket.event");
        addActivity("system", data.name || data.anima, summary);
      }
      break;
  }

  // Dispatch to registered handlers
  const handlers = _eventHandlers.get(eventType);
  if (handlers) for (const fn of handlers) { try { fn(data); } catch (e) { logger.error("Event handler error", { type: eventType, error: e.message }); } }
}

// ── Toast Notifications ─────────────────────

function showNotificationToast(animaName, subject, body, priority) {
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
  header.textContent = animaName;

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
      logger.info("Page visible, triggering reconnect");
      wsReconnectAttempt = 0;
      scheduleReconnect();
    }
  }
});
