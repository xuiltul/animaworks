// ── API Client ──────────────────────
// Thin fetch wrapper for all REST endpoints.

import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-api");
const BASE = "";

async function request(path, opts = {}) {
  try {
    const res = await fetch(`${BASE}${path}`, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      logger.error("API request failed", { url: path, status: res.status, detail: text.slice(0, 200) });
      throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
  } catch (err) {
    if (err.message && !err.message.startsWith("API ")) {
      logger.error("Network error", { url: path, error: err.message });
    }
    throw err;
  }
}

function post(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ── Anima ──────────────────────

export function fetchAnimas() {
  return request("/api/animas");
}

export function fetchAnimaDetail(name) {
  return request(`/api/animas/${encodeURIComponent(name)}`);
}

// ── Chat ──────────────────────

export function greetAnima(name) {
  return post(`/api/animas/${encodeURIComponent(name)}/greet`, {});
}

export function sendChat(name, message, userName) {
  return post(`/api/animas/${encodeURIComponent(name)}/chat`, {
    message,
    from_person: userName || "human",
  });
}

/**
 * Start SSE chat stream. Returns the raw Response for manual reading.
 */
export function sendChatStream(name, message, userName) {
  return fetch(`/api/animas/${encodeURIComponent(name)}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, from_person: userName || "human" }),
  });
}

// ── Memory ──────────────────────

export function fetchEpisodes(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/episodes`);
}

export function fetchEpisode(name, date) {
  return request(`/api/animas/${encodeURIComponent(name)}/episodes/${encodeURIComponent(date)}`);
}

export function fetchKnowledge(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/knowledge`);
}

export function fetchKnowledgeTopic(name, topic) {
  return request(`/api/animas/${encodeURIComponent(name)}/knowledge/${encodeURIComponent(topic)}`);
}

export function fetchProcedures(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/procedures`);
}

export function fetchProcedure(name, proc) {
  return request(`/api/animas/${encodeURIComponent(name)}/procedures/${encodeURIComponent(proc)}`);
}

// ── Session ──────────────────────

export function fetchSessions(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/sessions`);
}

export function fetchSession(name, sessionId) {
  return request(`/api/animas/${encodeURIComponent(name)}/sessions/${encodeURIComponent(sessionId)}`);
}

export function fetchConversationFull(name, limit = 50, offset = 0) {
  return request(`/api/animas/${encodeURIComponent(name)}/conversation/full?limit=${limit}&offset=${offset}`);
}

export function fetchTranscript(name, date) {
  return request(`/api/animas/${encodeURIComponent(name)}/transcripts/${encodeURIComponent(date)}`);
}

// ── System ──────────────────────

export function fetchSystemStatus() {
  return request("/api/system/status");
}

export function fetchSharedUsers() {
  return request("/api/shared/users");
}

export function reloadSystem() {
  return post("/api/system/reload", {});
}

export function triggerHeartbeat(name) {
  return post(`/api/animas/${encodeURIComponent(name)}/trigger`, {});
}

// ── Assets ──────────────────────

export function assetUrl(name, filename) {
  return `/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
}

export function fetchAssets(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/assets`);
}

export function fetchAssetMetadata(name) {
  return request(`/api/animas/${encodeURIComponent(name)}/assets/metadata`);
}

/**
 * Check if an asset exists by issuing a HEAD request.
 * Returns the URL if it exists, or null otherwise.
 */
export async function probeAsset(name, filename) {
  const url = assetUrl(name, filename);
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok ? url : null;
  } catch {
    return null;
  }
}

// ── Messages ──────────────────────

export async function fetchMessage(messageId) {
  return request(`/api/messages/${encodeURIComponent(messageId)}`);
}
