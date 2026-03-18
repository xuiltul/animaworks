/* ── API Helper ────────────────────────────── */

import { createLogger } from "../shared/logger.js";

const logger = createLogger("api");

/**
 * Stream SSE response and invoke callbacks for each event type.
 *
 * @param {string} path - API path
 * @param {Object} opts - fetch options (method, headers, body, signal)
 * @param {function(Object): void} [opts.onProgress] - Called for "progress" events with { phase }
 * @param {function(Object): void} [opts.onResult] - Called for "result" events with payload
 * @param {function(Object): void} [opts.onError] - Called for "error" events with { code, message }
 * @returns {Promise<void>}
 */
export async function apiStream(path, opts = {}) {
  const { onProgress, onResult, onError, ...fetchOpts } = opts;
  fetchOpts.credentials = "same-origin";

  const res = await fetch(path, fetchOpts);

  if (res.status === 401) {
    logger.warn("Unauthorized, redirecting to login", { url: path });
    window.location.hash = "";
    window.location.reload();
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    logger.error("API request failed", { url: path, status: res.status, statusText: res.statusText });
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let event = "";
    let data = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        data = line.slice(6);
      } else if (line === "" && event && data) {
        try {
          const payload = JSON.parse(data);
          if (event === "progress" && onProgress) onProgress(payload);
          else if (event === "result" && onResult) onResult(payload);
          else if (event === "error" && onError) onError(payload);
        } catch {
          // ignore parse errors
        }
        event = "";
        data = "";
      }
    }
  }

  if (buffer.trim()) {
    const lines = buffer.split("\n");
    let event = "";
    let data = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data = line.slice(6);
      else if (line === "" && event && data) {
        try {
          const payload = JSON.parse(data);
          if (event === "progress" && onProgress) onProgress(payload);
          else if (event === "result" && onResult) onResult(payload);
          else if (event === "error" && onError) onError(payload);
        } catch {
          // ignore
        }
      }
    }
  }
}

export async function api(path, opts = {}) {
  try {
    // Always include credentials for cookie-based auth
    opts.credentials = "same-origin";
    const res = await fetch(path, opts);

    if (res.status === 401) {
      // Redirect to login screen on auth failure
      logger.warn("Unauthorized, redirecting to login", { url: path });
      window.location.hash = "";
      window.location.reload();
      throw new Error("Unauthorized");
    }

    if (!res.ok) {
      logger.error("API request failed", { url: path, status: res.status, statusText: res.statusText });
      throw new Error(`API ${res.status}: ${res.statusText}`);
    }
    return res.json();
  } catch (err) {
    if (err.message && !err.message.startsWith("API ") && err.message !== "Unauthorized") {
      logger.error("Network error", { url: path, error: err.message });
    }
    throw err;
  }
}
