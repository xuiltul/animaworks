/* ── API Helper ────────────────────────────── */

import { createLogger } from "../shared/logger.js";

const logger = createLogger("api");

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
