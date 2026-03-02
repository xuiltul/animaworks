// ── Chat Session Manager ──────────────────────
// Unified session manager for all chat consumers (Chat Page, Workspace, Legacy).
// Singleton EventTarget; manages anima:thread → ChatSession map.

import { createHistoryState, applyHistoryData, mergePolledHistory } from "./history-loader.js";

// ── ChatSession ──────────────────────

class ChatSession {
  constructor(animaName, threadId) {
    this.animaName = animaName;
    this.threadId = threadId;
    this.messages = [];
    this.historyState = createHistoryState();
    this._streamSeq = 0;
    this._streamingMsg = null;
    this._abortController = null;
    this._pendingQueue = [];
  }

  get isStreaming() {
    return Boolean(this._streamingMsg?.streaming);
  }

  nextStreamId() {
    return `${this.animaName}:${this.threadId}:${Date.now()}:${++this._streamSeq}`;
  }
}

// ── ChatSessionManager ──────────────────────

let _instance = null;

/**
 * @typedef {object} ManagerConfig
 * @property {function} streamChat - (animaName, body, signal, callbacks) => Promise
 * @property {function} fetchActiveStream - (animaName, threadId?) => Promise<object|null>
 * @property {function} fetchStreamProgress - (animaName, responseId) => Promise<object|null>
 * @property {function} getUser - () => string
 * @property {function} fetchHistory - (animaName, limit, before, threadId) => Promise<object>
 */

export class ChatSessionManager extends EventTarget {
  /** @type {Map<string, ChatSession>} */
  #sessions = new Map();
  /** @type {ManagerConfig|null} */
  #config = null;

  static getInstance() {
    if (!_instance) _instance = new ChatSessionManager();
    return _instance;
  }

  static resetInstance() {
    _instance = null;
  }

  /**
   * One-time configuration with shared dependencies.
   * @param {ManagerConfig} config
   */
  configure(config) {
    this.#config = config;
  }

  // ── Session Management ──────────────────────

  #key(anima, thread) {
    return `${anima}:${thread}`;
  }

  getSession(anima, thread = "default") {
    const key = this.#key(anima, thread);
    if (!this.#sessions.has(key)) {
      this.#sessions.set(key, new ChatSession(anima, thread));
    }
    return this.#sessions.get(key);
  }

  hasSession(anima, thread = "default") {
    return this.#sessions.has(this.#key(anima, thread));
  }

  destroySession(anima, thread = "default") {
    const key = this.#key(anima, thread);
    const session = this.#sessions.get(key);
    if (!session) return;
    if (session._abortController) session._abortController.abort();
    session._streamingMsg = null;
    session._pendingQueue.length = 0;
    this.#sessions.delete(key);
  }

  destroyAllForAnima(anima) {
    for (const key of [...this.#sessions.keys()]) {
      if (key.startsWith(`${anima}:`)) {
        this.destroySession(anima, key.split(":").slice(1).join(":"));
      }
    }
  }

  // ── Message Management ──────────────────────

  getMessages(anima, thread = "default") {
    return this.getSession(anima, thread).messages;
  }

  setMessages(anima, thread, messages) {
    this.getSession(anima, thread).messages = messages;
  }

  addMessage(anima, thread, msg) {
    const session = this.getSession(anima, thread);
    session.messages.push(msg);
    this.#dispatch("messages-changed", { anima, thread });
  }

  keepOnlyStreaming(anima, thread) {
    const session = this.getSession(anima, thread);
    session.messages = session.messages.filter(m => m.streaming);
  }

  // ── History Management ──────────────────────

  getHistoryState(anima, thread = "default") {
    return this.getSession(anima, thread).historyState;
  }

  setHistoryState(anima, thread, hs) {
    this.getSession(anima, thread).historyState = hs;
  }

  async loadHistory(anima, thread = "default", limit = 50) {
    const session = this.getSession(anima, thread);
    session.historyState = { ...createHistoryState(), loading: true };
    this.#dispatch("history-loading", { anima, thread });

    try {
      const data = await this.#config.fetchHistory(anima, limit, null, thread);
      session.historyState = createHistoryState();
      applyHistoryData(session.historyState, data);
    } catch {
      session.historyState = createHistoryState();
    }
    this.#dispatch("history-loaded", { anima, thread });
    return session.historyState;
  }

  async loadMoreHistory(anima, thread = "default", limit = 50) {
    const session = this.getSession(anima, thread);
    const hs = session.historyState;
    if (!hs || !hs.hasMore || hs.loading) return;

    hs.loading = true;
    this.#dispatch("history-loading", { anima, thread });

    try {
      const data = await this.#config.fetchHistory(anima, limit, hs.nextBefore, thread);
      applyHistoryData(hs, data, { prepend: true });
    } catch {
      hs.hasMore = false;
    }
    hs.loading = false;
    this.#dispatch("history-loaded", { anima, thread });
  }

  mergePolledHistory(anima, thread, pollData) {
    const session = this.getSession(anima, thread);
    const prev = session.historyState;
    const { changed, merged } = mergePolledHistory(prev, pollData);
    if (changed) {
      session.historyState = merged;
    }
    return { changed };
  }

  // ── Pending Queue ──────────────────────

  getPendingQueue(anima, thread = "default") {
    return this.getSession(anima, thread)._pendingQueue;
  }

  enqueue(anima, thread, item) {
    this.getSession(anima, thread)._pendingQueue.push(item);
    this.#dispatch("queue-changed", { anima, thread });
  }

  dequeue(anima, thread) {
    const q = this.getSession(anima, thread)._pendingQueue;
    const item = q.shift();
    this.#dispatch("queue-changed", { anima, thread });
    return item;
  }

  removeFromQueue(anima, thread, index) {
    const q = this.getSession(anima, thread)._pendingQueue;
    if (index >= 0 && index < q.length) {
      const removed = q.splice(index, 1)[0];
      this.#dispatch("queue-changed", { anima, thread });
      return removed;
    }
    return undefined;
  }

  clearQueue(anima, thread) {
    this.getSession(anima, thread)._pendingQueue.length = 0;
    this.#dispatch("queue-changed", { anima, thread });
  }

  // ── Streaming ──────────────────────

  isStreamingForAnima(anima) {
    for (const [key, session] of this.#sessions) {
      if (key.startsWith(`${anima}:`) && session.isStreaming) return true;
    }
    return false;
  }

  isStreamingFor(anima, thread = "default") {
    const key = this.#key(anima, thread);
    const session = this.#sessions.get(key);
    return session ? session.isStreaming : false;
  }

  getStreamingContext(anima) {
    for (const [key, session] of this.#sessions) {
      if (key.startsWith(`${anima}:`) && session.isStreaming) {
        return { thread: session.threadId, streamingMsg: session._streamingMsg };
      }
    }
    return null;
  }

  getAbortController(anima, thread = "default") {
    return this.getSession(anima, thread)._abortController;
  }

  /**
   * Send a chat message via SSE streaming.
   * Manages session state; UI updates are injected via callbacks.
   *
   * @param {string} anima
   * @param {string} thread
   * @param {string} text
   * @param {object} options
   * @param {Array}  [options.images] - Base64 images for API
   * @param {Array}  [options.displayImages] - Display images for chat bubble
   * @param {object} [options.callbacks] - SSE event callbacks (onTextDelta, onDone, etc.)
   * @param {function} [options.onFinally]
   * @returns {Promise<{ streamingMsg, success, queued, error }>}
   */
  async sendChat(anima, thread, text, options = {}) {
    const { images = [], displayImages = [], callbacks = {}, onFinally } = options;

    const session = this.getSession(anima, thread);
    if (session.isStreaming) {
      this.enqueue(anima, thread, { text, images, displayImages });
      return { streamingMsg: null, success: false, queued: true };
    }
    const streamId = session.nextStreamId();
    const sendTs = new Date().toISOString();

    session.messages.push({ role: "user", text, images: displayImages, timestamp: sendTs });

    const streamingMsg = {
      role: "assistant", text: "", streaming: true, activeTool: null,
      timestamp: sendTs, thinkingText: "", thinking: false, streamId,
    };
    session.messages.push(streamingMsg);
    session._streamingMsg = streamingMsg;
    session._abortController = new AbortController();

    // Deliver streamingMsg synchronously before async streaming starts
    // so callbacks can reference it without TDZ errors.
    callbacks.onStreamCreated?.(streamingMsg);

    this.#dispatch("messages-changed", { anima, thread, streamingMsg });
    this.#dispatch("stream-state-changed", { anima, thread, isStreaming: true });

    try {
      const user = this.#config.getUser();
      const bodyObj = { message: text || "", from_person: user, thread_id: thread };
      if (images.length > 0) bodyObj.images = images;

      await this.#config.streamChat(
        anima, JSON.stringify(bodyObj), session._abortController.signal, callbacks,
      );
      return { streamingMsg, success: true };
    } catch (err) {
      if (err.name === "AbortError") {
        callbacks.onAbort?.();
      } else {
        callbacks.onError?.({ message: err.message });
      }
      return { streamingMsg, success: false, error: err };
    } finally {
      session._streamingMsg = null;
      session._abortController = null;
      this.#dispatch("stream-state-changed", { anima, thread, isStreaming: false });
      onFinally?.();
    }
  }

  /**
   * Resume an active stream (page reload recovery).
   *
   * @param {string} anima
   * @param {string} thread
   * @param {object} options
   * @param {object} [options.callbacks] - SSE event callbacks
   * @param {function} [options.onProgress] - Called with recovered progress before resume
   * @param {function} [options.onFinally]
   * @returns {Promise<{ streamingMsg, success }>}
   */
  async resumeStream(anima, thread, options = {}) {
    const { callbacks = {}, onProgress, onFinally } = options;
    const session = this.getSession(anima, thread);
    if (session.isStreaming) return { streamingMsg: null, success: false };

    try {
      const active = await this.#config.fetchActiveStream(anima, thread);
      if (!active || active.status !== "streaming") return { streamingMsg: null, success: false };

      const progress = await this.#config.fetchStreamProgress(anima, active.response_id);
      if (!progress) return { streamingMsg: null, success: false };

      onProgress?.({ progress, responseId: active.response_id });

      const streamId = session.nextStreamId();
      const streamingMsg = {
        role: "assistant", text: progress.full_text || "", streaming: true,
        activeTool: progress.active_tool || null,
        timestamp: new Date().toISOString(),
        thinkingText: "", thinking: false, streamId,
      };
      session.messages.push(streamingMsg);
      session._streamingMsg = streamingMsg;
      session._abortController = new AbortController();

      // Deliver streamingMsg synchronously before async streaming starts
      callbacks.onStreamCreated?.(streamingMsg);

      this.#dispatch("messages-changed", { anima, thread, streamingMsg });
      this.#dispatch("stream-state-changed", { anima, thread, isStreaming: true });

      const user = this.#config.getUser();
      const resumeBody = JSON.stringify({
        message: "", from_person: user,
        resume: active.response_id,
        last_event_id: progress.last_event_id || "",
      });

      await this.#config.streamChat(
        anima, resumeBody, session._abortController.signal, callbacks,
      );
      return { streamingMsg, success: true };
    } catch (err) {
      if (err.name !== "AbortError") {
        callbacks.onError?.({ message: err.message });
      }
      return { streamingMsg: null, success: false, error: err };
    } finally {
      session._streamingMsg = null;
      session._abortController = null;
      this.#dispatch("stream-state-changed", { anima, thread, isStreaming: false });
      onFinally?.();
    }
  }

  /**
   * Stop streaming for a specific anima (client abort + server interrupt).
   */
  stopStreaming(anima) {
    for (const [key, session] of this.#sessions) {
      if (key.startsWith(`${anima}:`) && session._abortController) {
        session._abortController.abort();
      }
    }
    fetch(`/api/animas/${encodeURIComponent(anima)}/interrupt`, { method: "POST" }).catch(() => {});
  }

  /**
   * Abort streaming (client-side only, no server interrupt).
   */
  abortStreaming(anima) {
    for (const [key, session] of this.#sessions) {
      if (key.startsWith(`${anima}:`) && session._abortController) {
        session._abortController.abort();
      }
    }
  }

  // ── Event Dispatch ──────────────────────

  #dispatch(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }
}

export { mergePolledHistory, createHistoryState, applyHistoryData };
