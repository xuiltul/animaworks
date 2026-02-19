// ── WebSocket Connection Manager ──────────────────────
// Auto-reconnect + event dispatch.

import { setState } from "./state.js";
import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-workspace");

const WS_INITIAL_DELAY = 1000;
const WS_MAX_DELAY = 30000;
const WS_BACKOFF_MULTIPLIER = 2;

let ws = null;
let reconnectTimer = null;
let reconnectAttempt = 0;
let connectedAt = null;
const eventHandlers = new Map(); // type -> Set<callback>

function getWsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

export function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const url = getWsUrl();
  logger.info("Connecting", { url, attempt: reconnectAttempt });

  try {
    ws = new WebSocket(url);
  } catch (err) {
    logger.error("WebSocket creation failed", { url, error: err.message });
    scheduleReconnect();
    return;
  }

  ws.addEventListener("open", () => {
    connectedAt = Date.now();
    logger.info("Connected", { url, attempt: reconnectAttempt });
    reconnectAttempt = 0;
    setState({ wsConnected: true });
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  });

  ws.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      // Respond to server ping with pong
      if (msg.type === "ping") {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "pong" }));
        }
        logger.debug("Heartbeat pong sent");
        return;
      }
      const handlers = eventHandlers.get(msg.type);
      if (handlers) {
        handlers.forEach((fn) => fn(msg.data, msg.type));
      }
      // Also fire wildcard listeners
      const wildcards = eventHandlers.get("*");
      if (wildcards) {
        wildcards.forEach((fn) => fn(msg.data, msg.type));
      }
    } catch {
      logger.warn("Non-JSON message received");
    }
  });

  ws.addEventListener("close", (evt) => {
    const duration = connectedAt ? Date.now() - connectedAt : 0;
    connectedAt = null;
    logger.warn("Connection closed", {
      code: evt.code,
      reason: evt.reason || "",
      wasClean: evt.wasClean,
      duration_ms: duration,
    });
    setState({ wsConnected: false });
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    logger.error("Connection error", { readyState: ws?.readyState });
    setState({ wsConnected: false });
  });
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  const delay = Math.min(
    WS_INITIAL_DELAY * Math.pow(WS_BACKOFF_MULTIPLIER, reconnectAttempt),
    WS_MAX_DELAY
  );
  const jitter = Math.random() * 1000;
  logger.info("Scheduling reconnect", { attempt: reconnectAttempt + 1, delay_ms: Math.round(delay + jitter) });
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    reconnectAttempt++;
    connect();
  }, delay + jitter);
}

/**
 * Register handler for a WebSocket event type.
 * Use "*" to listen to all events.
 * Returns unsubscribe function.
 */
export function onEvent(type, fn) {
  if (!eventHandlers.has(type)) {
    eventHandlers.set(type, new Set());
  }
  eventHandlers.get(type).add(fn);
  return () => eventHandlers.get(type).delete(fn);
}

export function disconnect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

// ── Visibility Change Reconnect ─────────────────────

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      logger.info("Page visible, triggering reconnect");
      reconnectAttempt = 0;
      scheduleReconnect();
    }
  }
});
