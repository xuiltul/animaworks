// ── Model Cache Module ──────────────────────
// IndexedDB-based persistent cache for GLB model files.
// Survives browser restarts, unlike HTTP cache which can be evicted.

const DB_NAME = "animaworks-model-cache";
const DB_VERSION = 1;
const STORE_NAME = "models";

/**
 * IndexedDB cache for 3D model ArrayBuffers.
 * Provides persistent storage that survives browser restart and is not
 * subject to HTTP cache eviction under storage pressure.
 */
export class ModelCache {
  constructor() {
    /** @type {IDBDatabase | null} */
    this._db = null;
    /** @type {Promise<IDBDatabase> | null} */
    this._opening = null;
  }

  /**
   * Open (or reuse) the IndexedDB connection.
   * @returns {Promise<IDBDatabase>}
   */
  async open() {
    if (this._db) return this._db;
    if (this._opening) return this._opening;

    this._opening = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "url" });
          store.createIndex("lastAccessed", "lastAccessed");
        }
      };

      request.onsuccess = () => {
        this._db = request.result;
        this._opening = null;
        resolve(this._db);
      };

      request.onerror = () => {
        this._opening = null;
        console.warn("model-cache: IndexedDB open failed", request.error);
        reject(request.error);
      };
    });

    return this._opening;
  }

  /**
   * Retrieve a cached ArrayBuffer by URL.
   * Updates lastAccessed timestamp on hit.
   * @param {string} url
   * @returns {Promise<ArrayBuffer | null>}
   */
  async get(url) {
    try {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(url);

        req.onsuccess = () => {
          const record = req.result;
          if (record) {
            // Update last-accessed for LRU tracking
            record.lastAccessed = Date.now();
            store.put(record);
            resolve(record.data);
          } else {
            resolve(null);
          }
        };
        req.onerror = () => reject(req.error);
      });
    } catch {
      return null;
    }
  }

  /**
   * Store an ArrayBuffer in the cache.
   * @param {string} url
   * @param {ArrayBuffer} arrayBuffer
   */
  async put(url, arrayBuffer) {
    try {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        store.put({
          url,
          data: arrayBuffer,
          size: arrayBuffer.byteLength,
          lastAccessed: Date.now(),
          cachedAt: Date.now(),
        });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch (err) {
      console.warn("model-cache: put failed for", url, err);
    }
  }

  /**
   * Delete a single cached entry.
   * @param {string} url
   */
  async delete(url) {
    try {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).delete(url);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch (err) {
      console.warn("model-cache: delete failed for", url, err);
    }
  }

  /**
   * Clear all cached models.
   */
  async clear() {
    try {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch (err) {
      console.warn("model-cache: clear failed", err);
    }
  }

  /**
   * Load a GLTF/GLB model through the cache.
   *
   * On cache hit: parses directly from the stored ArrayBuffer (no network).
   * On cache miss: fetches from network, stores in cache, then parses.
   *
   * @param {string} url        - URL of the GLB file
   * @param {import("three/addons/loaders/GLTFLoader.js").GLTFLoader} loader
   * @returns {Promise<import("three").GLTF>}
   */
  async loadGLTF(url, loader) {
    // Try cache first
    const cached = await this.get(url);
    if (cached) {
      return new Promise((resolve, reject) => {
        loader.parse(cached, "", resolve, reject);
      });
    }

    // Fetch from network
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`model-cache: fetch failed for ${url}: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();

    // Store in cache (fire-and-forget)
    this.put(url, buffer).catch(() => {});

    // Parse
    return new Promise((resolve, reject) => {
      loader.parse(buffer, "", resolve, reject);
    });
  }
}

/** Shared singleton cache instance. */
export const modelCache = new ModelCache();
