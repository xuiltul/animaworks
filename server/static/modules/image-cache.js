/**
 * Image resize & cache module.
 * Resizes large avatar PNGs to small WebP thumbnails via Canvas API,
 * caches them in Cache API for persistence and Map for session-level reuse.
 */

const CACHE_NAME = "anima-avatars-v1";
const SIZES = { S: 96, M: 192, L: 400 };
const WEBP_QUALITY = 0.85;
const REVALIDATE_INTERVAL = 60 * 60 * 1000;

/** @type {Map<string, string>} cacheKey -> blobURL */
const _memCache = new Map();

/** @type {Map<string, number>} cacheKey -> timestamp of last ETag check */
const _checkedAt = new Map();

/** @type {boolean|null} */
let _webpSupported = null;

function _cacheKey(url, size) {
  return `${url}?_thumb=${size}`;
}

function _hasCacheApi() {
  try {
    return typeof caches !== "undefined" && caches.open;
  } catch {
    return false;
  }
}

function _supportsWebP() {
  if (_webpSupported !== null) return _webpSupported;
  try {
    const c = document.createElement("canvas");
    c.width = 1;
    c.height = 1;
    _webpSupported = c.toDataURL("image/webp").startsWith("data:image/webp");
  } catch {
    _webpSupported = false;
  }
  return _webpSupported;
}

/**
 * Load a blob into an ImageBitmap (or Image element as fallback).
 * @param {Blob} blob
 * @returns {Promise<ImageBitmap|HTMLImageElement>}
 */
function _loadImage(blob) {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(blob);
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image load failed")); };
    img.src = url;
  });
}

/**
 * Resize image blob: center-crop to square, scale to targetSize.
 * @param {Blob} blob
 * @param {number} targetSize
 * @returns {Promise<Blob>}
 */
async function _resizeImage(blob, targetSize) {
  const img = await _loadImage(blob);
  const w = img.width;
  const h = img.height;
  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;

  let canvas, ctx;
  if (typeof OffscreenCanvas !== "undefined") {
    canvas = new OffscreenCanvas(targetSize, targetSize);
    ctx = canvas.getContext("2d");
  } else {
    canvas = document.createElement("canvas");
    canvas.width = targetSize;
    canvas.height = targetSize;
    ctx = canvas.getContext("2d");
  }

  ctx.drawImage(img, sx, sy, side, side, 0, 0, targetSize, targetSize);

  if (img.close) img.close();

  const mime = _supportsWebP() ? "image/webp" : "image/png";
  const quality = _supportsWebP() ? WEBP_QUALITY : undefined;

  if (canvas.convertToBlob) {
    return canvas.convertToBlob({ type: mime, quality });
  }
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), mime, quality);
  });
}

/**
 * Store a blob + metadata into Cache API.
 * @param {Cache} cache
 * @param {string} key
 * @param {Blob} blob
 * @param {string} etag
 */
async function _storeInCache(cache, key, blob, etag) {
  const resp = new Response(blob, {
    headers: { "Content-Type": blob.type, "X-Etag": etag || "" },
  });
  await cache.put(new Request(key), resp);
  const meta = JSON.stringify({ etag: etag || "", checkedAt: Date.now() });
  await cache.put(new Request(key + "__meta"), new Response(meta));
}

/**
 * Read metadata entry from Cache API.
 * @param {Cache} cache
 * @param {string} key
 * @returns {Promise<{etag: string, checkedAt: number}|null>}
 */
async function _readMeta(cache, key) {
  try {
    const resp = await cache.match(new Request(key + "__meta"));
    if (!resp) return null;
    return JSON.parse(await resp.text());
  } catch {
    return null;
  }
}

/**
 * Schedule a background ETag revalidation for a cache entry.
 * If the remote ETag changed, evict the stale entry so next access re-fetches.
 * @param {string} url  Original asset URL
 * @param {string} key  Cache key
 * @param {string} oldEtag
 */
function _scheduleRevalidation(url, key, oldEtag) {
  (async () => {
    try {
      const resp = await fetch(url, { method: "HEAD" });
      const newEtag = resp.headers.get("etag") || "";
      if (newEtag && newEtag !== oldEtag) {
        await invalidateCache(url);
      } else {
        _checkedAt.set(key, Date.now());
        if (_hasCacheApi()) {
          try {
            const cache = await caches.open(CACHE_NAME);
            const meta = JSON.stringify({ etag: newEtag || oldEtag, checkedAt: Date.now() });
            await cache.put(new Request(key + "__meta"), new Response(meta));
          } catch { /* best effort */ }
        }
      }
    } catch { /* network error — keep cached version */ }
  })();
}

/**
 * Get a cached, resized image. Returns a blob URL or the original URL on failure.
 * @param {string} url  Original image URL
 * @param {"S"|"M"|"L"} size
 * @returns {Promise<string>}
 */
export async function getCachedImage(url, size = "S") {
  if (!url) return url;
  const targetSize = SIZES[size] || SIZES.S;
  const key = _cacheKey(url, size);

  try {
    const memHit = _memCache.get(key);
    if (memHit) return memHit;

    if (_hasCacheApi()) {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(new Request(key));
      if (cached) {
        const blob = await cached.blob();
        const blobUrl = URL.createObjectURL(blob);
        _memCache.set(key, blobUrl);

        const meta = await _readMeta(cache, key);
        const lastChecked = _checkedAt.get(key) || (meta && meta.checkedAt) || 0;
        if (Date.now() - lastChecked > REVALIDATE_INTERVAL) {
          _scheduleRevalidation(url, key, (meta && meta.etag) || "");
        }
        return blobUrl;
      }
    }

    const resp = await fetch(url);
    if (!resp.ok) return url;

    const etag = resp.headers.get("etag") || "";
    const originalBlob = await resp.blob();
    const resizedBlob = await _resizeImage(originalBlob, targetSize);
    const blobUrl = URL.createObjectURL(resizedBlob);
    _memCache.set(key, blobUrl);
    _checkedAt.set(key, Date.now());

    if (_hasCacheApi()) {
      try {
        const cache = await caches.open(CACHE_NAME);
        await _storeInCache(cache, key, resizedBlob, etag);
      } catch { /* Cache API write failed — memory cache still works */ }
    }

    return blobUrl;
  } catch {
    return url;
  }
}

/**
 * Invalidate all size variants for a URL from both Cache API and memory cache.
 * @param {string} url  Original image URL
 */
export async function invalidateCache(url) {
  if (!url) return;
  for (const size of Object.keys(SIZES)) {
    const key = _cacheKey(url, size);
    const blobUrl = _memCache.get(key);
    if (blobUrl) {
      try { URL.revokeObjectURL(blobUrl); } catch { /* ignore */ }
      _memCache.delete(key);
    }
    _checkedAt.delete(key);
  }

  if (_hasCacheApi()) {
    try {
      const cache = await caches.open(CACHE_NAME);
      for (const size of Object.keys(SIZES)) {
        const key = _cacheKey(url, size);
        await cache.delete(new Request(key));
        await cache.delete(new Request(key + "__meta"));
      }
    } catch { /* best effort */ }
  }
}

/**
 * Batch preload avatars.
 * @param {Array<{url: string, size?: string}>} entries
 * @returns {Promise<void>}
 */
export async function preloadAvatars(entries) {
  if (!entries || entries.length === 0) return;
  await Promise.all(
    entries.map((e) => getCachedImage(e.url, e.size || "S").catch(() => null))
  );
}
