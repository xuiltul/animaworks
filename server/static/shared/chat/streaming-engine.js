// ── Shared Streaming Engine ──────────────────────
// DOM-independent streaming/send/queue management for chat.
// Consumers inject UI callbacks; the engine manages lifecycle.

/**
 * @typedef {object} StreamingEngineConfig
 * @property {function} streamChat - (animaName, body, signal, callbacks) => Promise
 * @property {function} fetchActiveStream - (animaName) => Promise<object|null>
 * @property {function} fetchStreamProgress - (animaName, responseId) => Promise<object|null>
 * @property {function} getUser - () => string
 * @property {function} [onStateChange] - (state: { isStreaming, pendingCount }) => void
 */

/**
 * Create a streaming engine instance.
 * @param {StreamingEngineConfig} config
 */
export function createStreamingEngine(config) {
  const { streamChat, fetchActiveStream, fetchStreamProgress, getUser, onStateChange } = config;

  let _abortController = null;
  let _streamingContext = null;  // { anima, thread }
  const _pendingQueue = [];     // Array<{ text, images, displayImages }>

  function _notifyState() {
    onStateChange?.({
      isStreaming: !!_streamingContext,
      pendingCount: _pendingQueue.length,
      streamingContext: _streamingContext ? { ..._streamingContext } : null,
    });
  }

  /**
   * Get current streaming state.
   */
  function getState() {
    return {
      isStreaming: !!_streamingContext,
      streamingContext: _streamingContext ? { ..._streamingContext } : null,
      pendingCount: _pendingQueue.length,
      pendingQueue: [..._pendingQueue],
    };
  }

  /**
   * Check if currently streaming for a specific anima+thread.
   */
  function isStreamingFor(animaName, threadId) {
    return _streamingContext?.anima === animaName && _streamingContext?.thread === threadId;
  }

  /**
   * Add item to the pending queue.
   * @param {{ text: string, images?: Array, displayImages?: Array }} item
   */
  function enqueue(item) {
    _pendingQueue.push(item);
    _notifyState();
  }

  /**
   * Remove and return the next item from the queue.
   * @returns {{ text: string, images?: Array, displayImages?: Array }|undefined}
   */
  function dequeue() {
    const item = _pendingQueue.shift();
    _notifyState();
    return item;
  }

  /**
   * Remove an item from the queue by index.
   */
  function removeFromQueue(index) {
    if (index >= 0 && index < _pendingQueue.length) {
      const removed = _pendingQueue.splice(index, 1)[0];
      _notifyState();
      return removed;
    }
    return undefined;
  }

  /**
   * Clear all pending queue items.
   */
  function clearQueue() {
    _pendingQueue.length = 0;
    _notifyState();
  }

  /**
   * Send a chat message via SSE streaming.
   * @param {string} animaName
   * @param {string} threadId
   * @param {string} text
   * @param {object} [options]
   * @param {Array}  [options.images]
   * @param {function} [options.onBeforeStream] - Called before streaming starts, receives { streamingMsg, abortController }
   * @param {object}  options.callbacks - SSE event callbacks forwarded to streamChat
   * @param {function} [options.onFinally] - Called in finally block
   * @returns {Promise<{ success: boolean }>}
   */
  async function send(animaName, threadId, text, options = {}) {
    const { images, callbacks = {}, onBeforeStream, onFinally } = options;

    _streamingContext = { anima: animaName, thread: threadId };
    _abortController = new AbortController();
    _notifyState();

    onBeforeStream?.({ abortController: _abortController });

    try {
      const user = getUser();
      const bodyObj = { message: text || "", from_person: user, thread_id: threadId };
      if (images && images.length > 0) bodyObj.images = images;
      const body = JSON.stringify(bodyObj);

      await streamChat(animaName, body, _abortController.signal, callbacks);
      return { success: true };
    } catch (err) {
      if (err.name === "AbortError") {
        callbacks.onAbort?.();
      } else {
        callbacks.onError?.({ message: err.message });
      }
      return { success: false, error: err };
    } finally {
      _streamingContext = null;
      _abortController = null;
      _notifyState();
      onFinally?.();

      if (_pendingQueue.length > 0) {
        callbacks.onQueueDrain?.();
      }
    }
  }

  /**
   * Resume an active stream (page reload recovery).
   * @param {string} animaName
   * @param {string} threadId
   * @param {object} options
   * @param {object} options.callbacks - SSE event callbacks
   * @param {function} [options.onProgress] - Called with progress data before resuming
   * @param {function} [options.onFinally]
   * @returns {Promise<boolean>} Whether resume was successful
   */
  async function resume(animaName, threadId, options = {}) {
    if (_streamingContext || _abortController) return false;

    const { callbacks = {}, onProgress, onFinally } = options;

    try {
      const active = await fetchActiveStream(animaName);
      if (!active || active.status !== "streaming") return false;

      const progress = await fetchStreamProgress(animaName, active.response_id);
      if (!progress) return false;

      onProgress?.({ progress, responseId: active.response_id });

      _streamingContext = { anima: animaName, thread: threadId };
      _abortController = new AbortController();
      _notifyState();

      const user = getUser();
      const resumeBody = JSON.stringify({
        message: "",
        from_person: user,
        resume: active.response_id,
        last_event_id: progress.last_event_id || "",
      });

      await streamChat(animaName, resumeBody, _abortController.signal, callbacks);
      return true;
    } catch (err) {
      if (err.name !== "AbortError") {
        callbacks.onError?.({ message: err.message });
      }
      return false;
    } finally {
      _streamingContext = null;
      _abortController = null;
      _notifyState();
      onFinally?.();

      if (_pendingQueue.length > 0) {
        callbacks.onQueueDrain?.();
      }
    }
  }

  /**
   * Stop the current stream.
   * @param {string} animaName - For sending interrupt API call
   */
  function stop(animaName) {
    if (_abortController) _abortController.abort();
    if (animaName) {
      fetch(`/api/animas/${encodeURIComponent(animaName)}/interrupt`, { method: "POST" }).catch(() => {});
    }
  }

  /**
   * Abort the current stream (client-side only, no server interrupt).
   */
  function abort() {
    if (_abortController) _abortController.abort();
  }

  return {
    getState,
    isStreamingFor,
    enqueue,
    dequeue,
    removeFromQueue,
    clearQueue,
    send,
    resume,
    stop,
    abort,
  };
}
