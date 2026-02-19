// ── Touch Gesture Utility ──────────────────────────────────
// Swipe gesture detection for mobile interactions.
// Importable by both dashboard and workspace modules.

/**
 * Detects horizontal swipe gestures on a DOM element.
 *
 * @example
 *   const sw = new SwipeHandler(document.body);
 *   sw.onSwipeLeft((info) => console.log("swiped left", info.startX));
 *   sw.onSwipeRight((info) => console.log("swiped right", info.startX));
 */
export class SwipeHandler {
  /**
   * @param {HTMLElement} element - Element to attach touch listeners to
   * @param {Object} [options]
   * @param {number} [options.threshold=50] - Minimum horizontal distance (px)
   * @param {number} [options.maxVerticalDrift=30] - Maximum vertical drift (px)
   */
  constructor(element, options = {}) {
    this._el = element;
    // 50px: comfortable swipe distance that avoids accidental triggers from taps/scrolls
    this._threshold = options.threshold ?? 50;
    // 30px: allows slight vertical wobble during horizontal swipe without cancelling
    this._maxDrift = options.maxVerticalDrift ?? 30;
    this._leftCallbacks = [];
    this._rightCallbacks = [];
    this._startX = 0;
    this._startY = 0;
    this._tracking = false;

    this._onTouchStart = this._onTouchStart.bind(this);
    this._onTouchMove = this._onTouchMove.bind(this);
    this._onTouchEnd = this._onTouchEnd.bind(this);

    element.addEventListener("touchstart", this._onTouchStart, { passive: true });
    element.addEventListener("touchmove", this._onTouchMove, { passive: true });
    element.addEventListener("touchend", this._onTouchEnd, { passive: true });
  }

  /**
   * Register a callback for left swipe.
   * @param {function({ startX: number, startY: number, dx: number, dy: number }): void} callback
   * @returns {SwipeHandler}
   */
  onSwipeLeft(callback) {
    this._leftCallbacks.push(callback);
    return this;
  }

  /**
   * Register a callback for right swipe.
   * @param {function({ startX: number, startY: number, dx: number, dy: number }): void} callback
   * @returns {SwipeHandler}
   */
  onSwipeRight(callback) {
    this._rightCallbacks.push(callback);
    return this;
  }

  /** @private */
  _onTouchStart(e) {
    if (e.touches.length !== 1) return;
    const touch = e.touches[0];
    this._startX = touch.clientX;
    this._startY = touch.clientY;
    this._tracking = true;
  }

  /** @private */
  _onTouchMove(e) {
    if (!this._tracking) return;
    const touch = e.touches[0];
    const dy = Math.abs(touch.clientY - this._startY);
    if (dy > this._maxDrift) {
      this._tracking = false;
    }
  }

  /** @private */
  _onTouchEnd(e) {
    if (!this._tracking) return;
    this._tracking = false;

    const touch = e.changedTouches[0];
    const dx = touch.clientX - this._startX;
    const dy = touch.clientY - this._startY;

    if (Math.abs(dy) > this._maxDrift) return;
    if (Math.abs(dx) < this._threshold) return;

    const info = {
      startX: this._startX,
      startY: this._startY,
      dx,
      dy,
    };

    if (dx > 0) {
      for (const cb of this._rightCallbacks) cb(info);
    } else {
      for (const cb of this._leftCallbacks) cb(info);
    }
  }

  /** Remove all listeners and callbacks. */
  destroy() {
    this._el.removeEventListener("touchstart", this._onTouchStart);
    this._el.removeEventListener("touchmove", this._onTouchMove);
    this._el.removeEventListener("touchend", this._onTouchEnd);
    this._leftCallbacks = [];
    this._rightCallbacks = [];
  }
}
