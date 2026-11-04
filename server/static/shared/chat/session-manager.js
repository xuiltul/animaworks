// ── Chat Session Manager ──────────────────────
// Unified session manager for all chat consumers (Chat Page, Workspace, Legacy).
// Singleton EventTarget; manages anima:thread → ChatSession map.

import {
  createHistoryState, applyHistoryData, mergePolledHistory, hasHistoryContent, hasFeedConversation,
} from "./history-loader.js";
import { api } from "../../modules/api.js";

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
    this._prefetchedHistory = null;
    /** Epoch ms when the current stream started; null when idle. */
    this._streamingSince = null;
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
 * @property {function} [streamMeetingChat] - (roomId, body, signal, callbacks) => Promise
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

  async loadHistory(anima, thread = "default", limit = 50, fetchHistory = null) {
    const session = this.getSession(anima, thread);
    const previous = session.historyState;
    session._prefetchedHistory = null;
    session.historyState = { ...createHistoryState(), loading: true };
    this.#dispatch("history-loading", { anima, thread });

    try {
      const data = await (fetchHistory || this.#config.fetchHistory)(anima, limit, null, thread);
      session.historyState = createHistoryState();
      applyHistoryData(session.historyState, data);
      // Only a successful fetch can tell "no history" from "not fetched yet".
      session.historyState.loaded = true;
    } catch {
      // A failed fetch says nothing about the history: keep what was already
      // on screen, and otherwise report the failure instead of looking empty.
      session.historyState = hasHistoryContent(previous)
        ? { ...previous, loading: false }
        : { ...createHistoryState(), loadFailed: true };
    }
    this.#dispatch("history-loaded", { anima, thread });
    return session.historyState;
  }

  async loadMoreHistory(anima, thread = "default", limit = 50, fetchHistory = null) {
    const session = this.getSession(anima, thread);
    const hs = session.historyState;
    if (!hs || !hs.hasMore || hs.loading) return;

    hs.loading = true;
    hs.olderFailed = false;
    this.#dispatch("history-loading", { anima, thread });

    try {
      const fetcher = fetchHistory || this.#config.fetchHistory;
      const prefetched = session._prefetchedHistory;
      const usePrefetched = prefetched?.before === hs.nextBefore && prefetched.fetchHistory === fetcher;
      if (prefetched && !usePrefetched) session._prefetchedHistory = null;
      const data = usePrefetched
        ? await prefetched.promise
        : await fetcher(anima, limit, hs.nextBefore, thread);
      if (usePrefetched) session._prefetchedHistory = null;
      applyHistoryData(hs, data, { prepend: true });
    } catch {
      session._prefetchedHistory = null;
      // hasMore stays true: the older pages still exist and can be retried.
      hs.olderFailed = true;
    }
    hs.loading = false;
    this.#dispatch("history-loaded", { anima, thread });
  }

  prefetchMoreHistory(anima, thread = "default", limit = 50, fetchHistory = null) {
    const session = this.getSession(anima, thread);
    const hs = session.historyState;
    if (!hs?.hasMore || hs.loading || !hs.nextBefore) return null;
    const fetcher = fetchHistory || this.#config.fetchHistory;
    if (session._prefetchedHistory?.before === hs.nextBefore
        && session._prefetchedHistory.fetchHistory === fetcher) return session._prefetchedHistory.promise;

    const before = hs.nextBefore;
    const promise = fetcher(anima, limit, before, thread);
    session._prefetchedHistory = { before, fetchHistory: fetcher, promise };
    promise.catch(() => {
      if (session._prefetchedHistory?.before === before) session._prefetchedHistory = null;
    });
    return promise;
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

  /**
   * Epoch ms when the current stream for anima:thread started, or null when
   * not streaming. Used by pollers to bypass the streaming guard when a stream
   * has been "in progress" for suspiciously long (stuck SSE fail-safe).
   */
  getStreamingSince(anima, thread = "default") {
    const session = this.#sessions.get(this.#key(anima, thread));
    if (!session?.isStreaming) return null;
    return session._streamingSince ?? null;
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
    const { images = [], displayImages = [], files = [], displayFiles = [], callbacks = {}, onFinally, model } = options;

    const session = this.getSession(anima, thread);
    if (session.isStreaming) {
      this.enqueue(anima, thread, { text, images, displayImages, files, displayFiles });
      return { streamingMsg: null, success: false, queued: true };
    }
    const streamId = session.nextStreamId();
    const sendTs = new Date().toISOString();
    const user = this.#config.getUser();

    const userMessage = { role: "user", text, images: displayImages, files: displayFiles, timestamp: sendTs };
    session.messages.push(userMessage);

    const streamingMsg = {
      role: "assistant", text: "", streaming: true, activeTool: null,
      timestamp: sendTs, thinkingText: "", thinking: false, streamId,
      to_person: user,
      source_key: "chat",
    };
    session.messages.push(streamingMsg);
    session._streamingMsg = streamingMsg;
    session._streamingSince = Date.now();
    session._abortController = new AbortController();

    // Deliver streamingMsg synchronously before async streaming starts
    // so callbacks can reference it without TDZ errors.
    callbacks.onStreamCreated?.(streamingMsg);

    this.#dispatch("messages-changed", { anima, thread, streamingMsg });
    this.#dispatch("stream-state-changed", { anima, thread, isStreaming: true });

    try {
      const bodyObj = { message: text || "", from_person: user, thread_id: thread };
      if (images.length > 0) bodyObj.images = images;
      if (files.length > 0) bodyObj.files = files;
      if (model) bodyObj.model = model;

      const streamCallbacks = {
        ...callbacks,
        onStreamStart: (data) => {
          if (Array.isArray(data?.files) && data.files.length > 0) {
            userMessage.files = data.files;
            callbacks.onFilesReady?.(data.files);
          }
          callbacks.onStreamStart?.(data);
        },
      };

      await this.#config.streamChat(
        anima, JSON.stringify(bodyObj), session._abortController.signal, streamCallbacks,
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
      session._streamingSince = null;
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
        toolHistory: progress.tool_history || [],
        timestamp: new Date().toISOString(),
        thinkingText: "", thinking: false, streamId,
        to_person: this.#config.getUser(),
        source_key: "chat",
      };
      session.messages.push(streamingMsg);
      session._streamingMsg = streamingMsg;
      session._streamingSince = Date.now();
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
      session._streamingSince = null;
      session._abortController = null;
      this.#dispatch("stream-state-changed", { anima, thread, isStreaming: false });
      onFinally?.();
    }
  }

  /**
   * Send a meeting chat message via SSE streaming.
   * Uses session key "meeting:{roomId}" for meeting-specific state.
   *
   * @param {string} roomId - Meeting room ID
   * @param {string} text - Message text
   * @param {object} options
   * @param {object} [options.callbacks] - SSE event callbacks (onTextDelta, onSpeakerStart, onSpeakerEnd, etc.)
   * @param {function} [options.onFinally]
   * @returns {Promise<{ success, queued, error }>}
   */
  async sendMeetingChat(roomId, text, options = {}) {
    const { callbacks = {}, onFinally } = options;

    const sessionKey = `meeting:${roomId}`;
    const session = this.getSession(sessionKey, "default");

    if (!this.#config.streamMeetingChat) {
      return { success: false, error: new Error("streamMeetingChat not configured") };
    }
    if (session.isStreaming) {
      return { streamingMsg: null, success: false, queued: true };
    }

    const sendTs = new Date().toISOString();
    session.messages.push({ role: "user", text, timestamp: sendTs });

    session._abortController = new AbortController();
    session._streamingMsg = { streaming: true };
    session._streamingSince = Date.now();

    this.#dispatch("stream-state-changed", { anima: sessionKey, thread: "default", isStreaming: true });

    try {
      const user = this.#config.getUser();
      const bodyObj = { message: text || "", from_person: user };

      await this.#config.streamMeetingChat(
        roomId, JSON.stringify(bodyObj), session._abortController.signal, callbacks,
      );
      return { success: true };
    } catch (err) {
      if (err.name === "AbortError") {
        callbacks.onAbort?.();
      } else {
        callbacks.onError?.({ message: err.message });
      }
      return { success: false, error: err };
    } finally {
      session._streamingMsg = null;
      session._streamingSince = null;
      session._abortController = null;
      this.#dispatch("stream-state-changed", { anima: sessionKey, thread: "default", isStreaming: false });
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
    api(`/api/animas/${encodeURIComponent(anima)}/interrupt`, { method: "POST" }).catch(() => {});
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
