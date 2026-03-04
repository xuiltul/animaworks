/**
 * Avatar asset resolver — returns the right filenames based on display mode.
 *
 * Each mode only returns its own asset variants (no cross-mode fallback).
 * If the asset doesn't exist, callers should show the initial placeholder.
 */

import { getCachedImage, invalidateCache } from "./image-cache.js";

/** @type {Map<string, string|null>} animaName -> resolved URL (or null) */
const _headProbeCache = new Map();

export function isRealisticMode() {
  return document.body.classList.contains("mode-realistic");
}

/**
 * Return bustup candidates for probing (HEAD request).
 * Each mode returns only its own variants — no cross-mode fallback.
 */
export function bustupCandidates() {
  if (isRealisticMode()) {
    return ["avatar_bustup_realistic.png"];
  }
  return ["avatar_bustup.png", "avatar_chibi.png"];
}

/**
 * Return bustup expression filename candidates.
 * @param {string} expression - e.g. "neutral", "smile", "troubled"
 */
export function bustupExpressionCandidates(expression) {
  if (isRealisticMode()) {
    const realistic =
      expression === "neutral"
        ? "avatar_bustup_realistic.png"
        : `avatar_bustup_${expression}_realistic.png`;
    return [realistic];
  }
  const anime =
    expression === "neutral"
      ? "avatar_bustup.png"
      : `avatar_bustup_${expression}.png`;
  return [anime];
}

/**
 * Build the asset URL for a given anima name and filename.
 */
export function assetUrl(animaName, filename) {
  return `/api/animas/${encodeURIComponent(animaName)}/assets/${encodeURIComponent(filename)}`;
}

/**
 * Probe candidates via HEAD and return the first available URL (or null).
 * Results are cached per animaName to avoid repeated HEAD requests.
 * @param {string} animaName
 * @param {string[]} candidates - list of filenames to try
 * @returns {Promise<string|null>}
 */
export async function resolveAvatar(animaName, candidates) {
  if (_headProbeCache.has(animaName)) return _headProbeCache.get(animaName);

  for (const filename of candidates) {
    const url = assetUrl(animaName, filename);
    try {
      const resp = await fetch(url, { method: "HEAD" });
      if (resp.ok) {
        _headProbeCache.set(animaName, url);
        return url;
      }
    } catch {
      /* network error — try next */
    }
  }
  _headProbeCache.set(animaName, null);
  return null;
}

/**
 * Resolve avatar URL and return a cached/resized version via image-cache.
 * @param {string} animaName
 * @param {string[]} candidates
 * @param {"S"|"M"|"L"} size
 * @returns {Promise<string|null>}
 */
export async function resolveCachedAvatar(animaName, candidates, size = "S") {
  const url = await resolveAvatar(animaName, candidates);
  if (!url) return null;
  try {
    return await getCachedImage(url, size);
  } catch {
    return url;
  }
}

/**
 * Invalidate HEAD probe cache and image cache for a given anima.
 * Call when assets are regenerated.
 * @param {string} animaName
 */
export async function invalidateAvatarCache(animaName) {
  const oldUrl = _headProbeCache.get(animaName);
  _headProbeCache.delete(animaName);
  if (oldUrl) {
    await invalidateCache(oldUrl);
  }
}
