// ── API Client ──────────────────────
// Thin API client for all workspace REST endpoints.

import { api } from "../../modules/api.js";
import { basePath } from "/shared/base-path.js";

const BASE = basePath;

function request(path, opts = {}) {
  return api(path, opts);
}

// ── Anima ──────────────────────

export function fetchAnimas() {
  return request("/api/animas");
}

// ── System ──────────────────────

export function fetchSystemStatus() {
  return request("/api/system/status");
}

export function fetchSharedUsers() {
  return request("/api/shared/users");
}

// ── Assets ──────────────────────

export function assetUrl(name, filename) {
  return `${BASE}/api/animas/${encodeURIComponent(name)}/assets/${encodeURIComponent(filename)}`;
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
    const res = await api(url, {
      method: "HEAD",
      cache: "default",
      rawResponse: true,
      redirectOnUnauthorized: false,
      throwOnHttpError: false,
      logErrors: false,
    });
    return res.ok ? url : null;
  } catch {
    return null;
  }
}

// ── Messages ──────────────────────

export async function fetchMessage(messageId) {
  return request(`/api/messages/${encodeURIComponent(messageId)}`);
}
