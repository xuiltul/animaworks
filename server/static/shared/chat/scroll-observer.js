// ── Shared Scroll Observer ──────────────────────
// IntersectionObserver-based infinite scroll for chat history.

/**
 * Create a scroll observer for upward infinite scroll.
 * @param {object} config
 * @param {HTMLElement} config.container - The scrollable messages container
 * @param {string}   [config.sentinelSelector=".chat-load-sentinel"] - CSS selector for the sentinel element
 * @param {function}  config.onLoadMore - Called when sentinel becomes visible
 * @returns {{ observe: function, disconnect: function, refresh: function }}
 */
export function createScrollObserver(config) {
  const { container, onLoadMore, sentinelSelector = ".chat-load-sentinel" } = config;

  let observer = null;

  function _createObserver() {
    if (observer) observer.disconnect();
    if (!container) return;

    observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) onLoadMore();
        }
      },
      { root: container, rootMargin: "200px 0px 0px 0px" },
    );
  }

  function observe() {
    if (!observer) _createObserver();
    _observeSentinel();
  }

  function _observeSentinel() {
    if (!observer || !container) return;
    const sentinel = container.querySelector(sentinelSelector);
    if (sentinel) observer.observe(sentinel);
  }

  function disconnect() {
    if (observer) {
      observer.disconnect();
      observer = null;
    }
  }

  function refresh() {
    _observeSentinel();
  }

  return { observe, disconnect, refresh };
}
